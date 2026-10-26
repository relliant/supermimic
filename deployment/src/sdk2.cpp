#include "supermimic/runtime.hpp"
#include "supermimic/safety.hpp"

#include <Eigen/Geometry>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include <atomic>
#include <csignal>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <poll.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace supermimic;
using LowState = unitree_hg::msg::dds_::LowState_;
using LowCommand = unitree_hg::msg::dds_::LowCmd_;

const std::array<const char*, kJointCount> hardware_joint_names{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
};

volatile std::sig_atomic_t interrupted = 0;
void handleSignal(int) { interrupted = 1; }

struct Options {
    std::string mode = "monitor";
    std::string interface;
    std::string bundle;
    std::string safety_profile;
    int domain = -1;
    int threads = 1;
    double seconds = 0;
    bool enable_motors = false;
    bool simulation_autostart = false;
};

Options parseOptions(int count, char** arguments) {
    Options options;
    for (int index = 1; index < count; ++index) {
        const std::string argument = arguments[index];
        auto value = [&]() -> std::string {
            if (++index == count) throw std::runtime_error("Missing value for " + argument);
            return arguments[index];
        };
        if (argument == "--mode") options.mode = value();
        else if (argument == "--interface") options.interface = value();
        else if (argument == "--bundle") options.bundle = value();
        else if (argument == "--safety-profile") options.safety_profile = value();
        else if (argument == "--domain") options.domain = std::stoi(value());
        else if (argument == "--threads") options.threads = std::stoi(value());
        else if (argument == "--seconds") options.seconds = std::stod(value());
        else if (argument == "--enable-motors") options.enable_motors = true;
        else if (argument == "--simulation-autostart") options.simulation_autostart = true;
        else if (argument == "--help") {
            std::cout << "supermimic_g1 --bundle DIR --interface IFACE --domain ID\n"
                         "  [--mode monitor|simulation|hardware] [--safety-profile JSON]\n"
                         "  [--threads N] [--seconds N] [--enable-motors]\n"
                         "  [--simulation-autostart]\n"
                         "Default monitor mode never publishes commands or requests control.\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + argument);
    }
    if (options.bundle.empty() || options.interface.empty() || options.domain < 0 || options.domain > 230)
        throw std::runtime_error("Explicit bundle, network interface and DDS domain [0,230] are required");
    if (options.mode != "monitor" && options.mode != "simulation" && options.mode != "hardware")
        throw std::runtime_error("Invalid mode");
    if (options.mode == "simulation" && (options.interface != "lo" || options.domain == 0))
        throw std::runtime_error("Simulation requires loopback interface lo and nonzero DDS domain");
    if (options.mode == "hardware" && (!options.enable_motors || options.interface == "lo"))
        throw std::runtime_error("Hardware requires --enable-motors and an explicit non-loopback interface");
    if (options.mode == "monitor" && (options.enable_motors || options.simulation_autostart))
        throw std::runtime_error("Monitor mode cannot request actuation or control ownership");
    if (options.mode != "simulation" && options.simulation_autostart)
        throw std::runtime_error("Automatic start is allowed only in isolated simulation");
    if (options.threads < 1 || !std::isfinite(options.seconds) || options.seconds < 0)
        throw std::runtime_error("Invalid timing or inference threads");
    if (std::getenv("CYCLONEDDS_URI"))
        throw std::runtime_error("Unset CYCLONEDDS_URI; explicit interface binding must not be overridden");
    return options;
}

// This is the SDK's word-wise non-reflected CRC, not Ethernet/zlib CRC32.
template <typename Message>
std::uint32_t messageCrc(const Message& message) {
    static_assert(sizeof(Message) % sizeof(std::uint32_t) == 0);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&message);
    std::uint32_t checksum = 0xffffffffU;
    for (std::size_t index = 0; index < sizeof(Message) / 4 - 1; ++index) {
        std::uint32_t word;
        std::memcpy(&word, bytes + index * 4, sizeof(word));
        for (std::uint32_t bit = 0x80000000U; bit != 0; bit >>= 1) {
            checksum = (checksum & 0x80000000U) ? (checksum << 1) ^ 0x04c11db7U : checksum << 1;
            if (word & bit) checksum ^= 0x04c11db7U;
        }
    }
    return checksum;
}

struct SensorSnapshot {
    RobotState state;
    JointArray temperature{}, measured_torque{};
    bool motor_fault = false;
    std::uint8_t machine_mode = 0;
    MonotonicClock::time_point received{};
};

struct CommandSnapshot {
    JointArray position{};
    std::array<float, 4> expected_quaternion{1, 0, 0, 0};
    MonotonicClock::time_point computed{};
};

class Controller {
public:
    Controller(Options options, const Bundle& bundle)
        : options_(std::move(options)), bundle_(bundle), policy_(bundle, options_.threads),
          history_(bundle.history_length) {
        for (std::size_t joint = 0; joint < kJointCount; ++joint)
            if (bundle_.manifest.at("joint_names").at(joint) != hardware_joint_names[joint])
                throw std::runtime_error("Bundle joint names/order do not match G1 hardware motor indices");
        if (!options_.safety_profile.empty())
            profile_ = SafetyProfile::load(options_.safety_profile, bundle_, options_.mode == "hardware");
        else if (options_.mode != "monitor")
            throw std::runtime_error("Actuation requires an explicit safety profile");
        std::signal(SIGINT, handleSignal);
        std::signal(SIGTERM, handleSignal);
        unitree::robot::ChannelFactory::Instance()->Init(options_.domain, options_.interface);
        state_channel_ = unitree::robot::ChannelFactory::Instance()->CreateRecvChannel<LowState>(
            "rt/lowstate", std::bind(&Controller::receiveState, this, std::placeholders::_1), 1);
    }

    ~Controller() {
        running_ = false;
        if (writer_.joinable()) writer_.join();
        if (state_channel_) {
            // Explicitly close while this thread still owns the reader. Simply dropping the
            // channel can leave a DDS callback holding the last entity reference, causing
            // CycloneDDS 0.10.2's prevent_callbacks() to destroy/wait on its own callback.
            try {
                auto reader = state_channel_->GetReader()->GetNative();
                reader.close();
            } catch (const std::exception& error) {
                std::cerr << "DDS reader shutdown error: " << error.what() << std::endl;
            }
            state_channel_.reset();
        }
    }

    int run() {
        std::cout << "MODE " << options_.mode << " interface=" << options_.interface
                  << " domain=" << options_.domain << std::endl;
        if (options_.mode == "monitor") return monitor();

        SensorSnapshot initial = waitForState();
        validateOrThrow(initial);
        if (options_.mode == "hardware") {
            if (!confirm("ARM", "Review protection equipment and emergency stop; type ARM to request control")) return 0;
            switcher_ = std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
            switcher_->SetTimeout(3.0F);
            switcher_->Init();
            std::string form, current_mode;
            if (switcher_->CheckMode(form, current_mode) != 0)
                throw std::runtime_error("Cannot establish motion-control ownership");
            // Releasing the manufacturer's controller is an external, supervised procedure.
            // Never leave it relinquished because a later preflight/publisher setup failed.
            if (!current_mode.empty())
                throw std::runtime_error("Stock controller active: follow the manufacturer's supervised ownership procedure first");
        } else if (!options_.simulation_autostart && !confirm("START", "Type START for isolated simulation")) return 0;

        // Validate start velocities before creating the first command publisher.
        initial = snapshot();
        validateOrThrow(initial);
        validateStart(initial);
        publisher_ = std::make_unique<unitree::robot::ChannelPublisher<LowCommand>>("rt/lowcmd");
        publisher_->InitChannel();
        previous_position_ = initial.state.joint_position;
        expected_quaternion_ = initial.state.root_quaternion_wxyz;
        publishTarget(initial.state.joint_position, initial, 0.02);
        actuation_started_ = true;
        writer_ = std::thread(&Controller::writeLoop, this);

        bool success = false;
        try {
            if (options_.mode == "hardware") prepare(initial);
            if (!interrupted && !fault_ && options_.mode == "hardware" &&
                !confirm("PLAY", "Preparation complete; type PLAY to start the validated reference"))
                throw std::runtime_error("Playback was not confirmed");
            const SensorSnapshot aligned_state = snapshot();
            validateOrThrow(aligned_state);
            validateStart(aligned_state);
            for (std::size_t joint = 0; joint < kJointCount; ++joint)
                if (std::abs(aligned_state.state.joint_position[joint] - bundle_.referenceJointPosition(0)[joint]) >
                    profile_->maximum_start_position_error)
                    throw std::runtime_error("Start pose does not match the selected reference");
            ReferenceAlignment alignment(bundle_.referenceQuaternion(0), aligned_state.state.root_quaternion_wxyz);
            history_.reset();
            previous_action_.fill(0);
            std::cout << "TRACKING" << std::endl;
            const auto start = MonotonicClock::now();
            auto deadline = start;
            for (std::size_t frame = 0; frame < bundle_.frame_count && !interrupted && !fault_; ++frame) {
                if (options_.seconds > 0 && std::chrono::duration<double>(MonotonicClock::now() - start).count() >= options_.seconds) break;
                if (pollStop()) break;
                const auto inference_start = MonotonicClock::now();
                if (inference_start - deadline > std::chrono::milliseconds(20))
                    throw std::runtime_error("Policy scheduling deadline missed");
                const SensorSnapshot sensors = snapshot();
                validateOrThrow(sensors);
                const auto reference_rotation = alignment.align(bundle_.referenceQuaternion(frame));
                expected_quaternion_ = reference_rotation;
                checkTilt(sensors.state.root_quaternion_wxyz, reference_rotation);
                const auto observation = buildObservation(sensors.state, previous_action_, reference_rotation,
                                                           bundle_.default_joint_position);
                const auto action = policy_.run(history_.append(observation), bundle_.goal(frame));
                if (std::chrono::duration<double, std::milli>(MonotonicClock::now() - inference_start).count() >
                    profile_->inference_timeout_ms) throw std::runtime_error("Inference deadline missed");
                JointArray target{};
                for (std::size_t joint = 0; joint < kJointCount; ++joint)
                    target[joint] = bundle_.default_joint_position[joint] + bundle_.action_scale[joint] * action[joint];
                publishTarget(target, sensors, 0.02);
                deadline += std::chrono::milliseconds(20);
                std::this_thread::sleep_until(deadline);
            }
            success = !fault_ && !interrupted;
        } catch (const std::exception& error) { latchFault(error.what()); }
        running_ = false;
        if (writer_.joinable()) writer_.join();
        success = success && !fault_ && !interrupted;
        std::cout << "STOPPED " << (success ? "completed reference or requested duration" : "fault or operator stop") << std::endl;
        return success && !fault_ && !interrupted ? 0 : 2;
    }

private:
    void receiveState(const void* data) {
        const auto& message = *static_cast<const LowState*>(data);
        if (options_.mode != "simulation" && message.crc() != messageCrc(message)) return;
        if (message.mode_pr() != 0) { if (actuation_started_) latchFault("LowState is not PR mode"); return; }
        SensorSnapshot incoming;
        incoming.received = MonotonicClock::now();
        incoming.machine_mode = message.mode_machine();
        incoming.state.root_quaternion_wxyz = message.imu_state().quaternion();
        incoming.state.angular_velocity_body = message.imu_state().gyroscope();
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
            const auto& motor = message.motor_state()[joint];
            incoming.state.joint_position[joint] = motor.q();
            incoming.state.joint_velocity[joint] = motor.dq();
            incoming.temperature[joint] = std::max(motor.temperature()[0], motor.temperature()[1]);
            incoming.measured_torque[joint] = motor.tau_est();
            incoming.motor_fault = incoming.motor_fault || motor.motorstate() != 0;
        }
        if (profile_) {
            const auto& calibration = profile_->sensor_to_pelvis_wxyz;
            const Eigen::Quaternionf pelvis_from_sensor(calibration[0], calibration[1], calibration[2], calibration[3]);
            const auto& measured = incoming.state.root_quaternion_wxyz;
            const Eigen::Quaternionf world_from_sensor(measured[0], measured[1], measured[2], measured[3]);
            const Eigen::Quaternionf world_from_pelvis = world_from_sensor * pelvis_from_sensor.conjugate();
            incoming.state.root_quaternion_wxyz = {world_from_pelvis.w(), world_from_pelvis.x(), world_from_pelvis.y(), world_from_pelvis.z()};
            const auto& gyro = incoming.state.angular_velocity_body;
            const Eigen::Vector3f converted = pelvis_from_sensor * Eigen::Vector3f(gyro[0], gyro[1], gyro[2]);
            incoming.state.angular_velocity_body = {converted.x(), converted.y(), converted.z()};
        }
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        if (have_state_) {
            const auto tick_delta = static_cast<std::int32_t>(message.tick() - last_tick_);
            if (tick_delta <= 0) return;
            if (actuation_started_ && incoming.machine_mode != sensors_.machine_mode) {
                latchFault("Machine mode changed during actuation");
                return;
            }
        }
        sensors_ = incoming;
        last_tick_ = message.tick();
        have_state_ = true;
    }

    SensorSnapshot snapshot() {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        return sensors_;
    }

    SensorSnapshot waitForState() {
        const auto deadline = MonotonicClock::now() + std::chrono::seconds(10);
        while (!interrupted && MonotonicClock::now() < deadline) {
            const auto sensors = snapshot();
            if (!snapshotExpired(MonotonicClock::now(), sensors.received, 100)) return sensors;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("No fresh state received within ten seconds");
    }

    void validateOrThrow(const SensorSnapshot& sensors) {
        if (snapshotExpired(MonotonicClock::now(), sensors.received, profile_->state_timeout_ms))
            throw std::runtime_error("State watchdog expired");
        const auto error = validateRobotState(sensors.state, *profile_, sensors.temperature,
                                              sensors.measured_torque, sensors.motor_fault);
        if (!error.empty()) throw std::runtime_error(error);
    }

    void validateStart(const SensorSnapshot& sensors) {
        const auto reference_velocity = bundle_.referenceJointVelocity(0);
        const auto root_velocity = bundle_.referenceRootVelocity(0);
        for (float velocity : root_velocity)
            if (options_.mode == "hardware" && std::abs(velocity) > profile_->maximum_reference_start_speed)
                throw std::runtime_error("Reference root starts in motion; use a validated near-rest start segment");
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
            if (options_.mode == "hardware" && std::abs(sensors.state.joint_velocity[joint]) > profile_->maximum_start_joint_speed)
                throw std::runtime_error("Robot is not near rest");
            if (options_.mode == "hardware" && std::abs(reference_velocity[joint]) > profile_->maximum_reference_start_speed)
                throw std::runtime_error("Reference starts in motion; choose a near-rest reference segment");
        }
    }

    void checkTilt(const std::array<float, 4>& robot, const std::array<float, 4>& reference) {
        const Eigen::Quaternionf robot_rotation(robot[0], robot[1], robot[2], robot[3]);
        const Eigen::Quaternionf reference_rotation(reference[0], reference[1], reference[2], reference[3]);
        const float cosine = (robot_rotation.toRotationMatrix().transpose() * Eigen::Vector3f::UnitZ()).dot(
            reference_rotation.toRotationMatrix().transpose() * Eigen::Vector3f::UnitZ());
        if (std::acos(std::clamp(cosine, -1.0F, 1.0F)) > profile_->maximum_tilt_error)
            throw std::runtime_error("Reference-relative tilt limit");
    }

    void publishTarget(const JointArray& requested, const SensorSnapshot& sensors, double elapsed) {
        const auto computed = MonotonicClock::now();
        if (last_target_time_ != MonotonicClock::time_point{})
            elapsed = std::min(elapsed, std::chrono::duration<double>(computed - last_target_time_).count());
        const LimitedTarget bounded = limitTarget(requested, previous_position_, sensors.state, *profile_, bundle_, elapsed);
        previous_position_ = bounded.position;
        previous_action_ = bounded.applied_action;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            command_ = {bounded.position, expected_quaternion_, computed};
        }
        last_target_time_ = computed;
        if (bounded.limited) ++limited_steps_;
    }

    void prepare(const SensorSnapshot& initial) {
        std::cout << "PREPARING (press x then Enter to stop)" << std::endl;
        const auto reference = bundle_.referenceJointPosition(0);
        const auto start = MonotonicClock::now();
        auto deadline = start;
        while (!interrupted && !fault_) {
            if (pollStop()) throw std::runtime_error("Operator stopped preparation");
            const double fraction = std::min(1.0, std::chrono::duration<double>(MonotonicClock::now() - start).count() /
                                                    profile_->preparation_seconds);
            const auto sensors = snapshot();
            validateOrThrow(sensors);
            JointArray target{};
            for (std::size_t joint = 0; joint < kJointCount; ++joint)
                target[joint] = initial.state.joint_position[joint] + static_cast<float>(fraction) *
                    (reference[joint] - initial.state.joint_position[joint]);
            publishTarget(target, sensors, 0.02);
            if (fraction >= 1) break;
            deadline += std::chrono::milliseconds(20);
            if (MonotonicClock::now() - deadline > std::chrono::milliseconds(20))
                throw std::runtime_error("Preparation scheduling deadline missed");
            std::this_thread::sleep_until(deadline);
        }
    }

    bool confirm(const std::string& expected, const std::string& prompt) {
        if (!isatty(STDIN_FILENO)) throw std::runtime_error("Interactive terminal required for operator arming");
        std::cout << prompt << " (x stops): " << std::flush;
        const auto requested = MonotonicClock::now();
        std::string response;
        while (!interrupted && !fault_) {
            pollfd input{STDIN_FILENO, POLLIN, 0};
            if (poll(&input, 1, 20) > 0) {
                if (!std::getline(std::cin, response)) return false;
                if (response == "x" || response == "STOP") return false;
                return response == expected;
            }
                if (actuation_started_) {
                const auto sensors = snapshot();
                validateOrThrow(sensors);
                publishTarget(previous_position_, sensors, 0.02);
            }
            if (MonotonicClock::now() - requested > std::chrono::seconds(30)) return false;
        }
        return false;
    }

    bool pollStop() {
        pollfd input{STDIN_FILENO, POLLIN, 0};
        if (isatty(STDIN_FILENO) && poll(&input, 1, 0) > 0) {
            std::string response;
            if (std::getline(std::cin, response) && (response == "x" || response == "STOP")) return true;
        }
        return false;
    }

    int monitor() {
        const auto start = MonotonicClock::now();
        while (!interrupted) {
            if (options_.seconds > 0 && std::chrono::duration<double>(MonotonicClock::now() - start).count() >= options_.seconds) break;
            const auto sensors = snapshot();
            const double age = sensors.received == MonotonicClock::time_point{} ? -1 :
                std::chrono::duration<double, std::milli>(MonotonicClock::now() - sensors.received).count();
            std::cout << "MONITOR state_age_ms=" << age << " q0=" << sensors.state.joint_position[0]
                      << " machine_mode=" << static_cast<int>(sensors.machine_mode)
                      << " motor_fault=" << sensors.motor_fault << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return 0;
    }

    void latchFault(const std::string& reason) {
        if (!fault_.exchange(true)) std::cerr << "FAULT " << reason << std::endl;
    }

    void send(const CommandSnapshot& command, const SensorSnapshot& sensors, bool damping) {
        LowCommand outgoing{};
        outgoing.mode_pr() = 0;
        outgoing.mode_machine() = sensors.machine_mode;
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
            auto& motor = outgoing.motor_cmd()[joint];
            motor.mode() = 1;
            motor.q() = damping ? 0 : command.position[joint];
            motor.dq() = 0;
            motor.kp() = damping ? 0 : profile_->kp[joint];
            // Saturate damping gain using the current measured speed as a secondary guard.
            motor.kd() = damping ? std::min(profile_->damping_kd[joint],
                profile_->torque_limit[joint] / std::max(0.01F, std::abs(sensors.state.joint_velocity[joint]))) : profile_->kd[joint];
            motor.tau() = 0;
        }
        // Slots 29..34 stay mode=0; no hand or undocumented motor commands are emitted.
        outgoing.crc() = messageCrc(outgoing);
        if (!publisher_->Write(outgoing)) throw std::runtime_error("DDS command publication failed");
    }

    void writeLoop() {
        auto deadline = MonotonicClock::now();
        try {
            while (running_ && !interrupted && !fault_) {
                const auto now = MonotonicClock::now();
                if (now - deadline > std::chrono::milliseconds(20)) throw std::runtime_error("Writer scheduling deadline missed");
                const auto sensors = snapshot();
                validateOrThrow(sensors);
                CommandSnapshot command;
                { std::lock_guard<std::mutex> lock(command_mutex_); command = command_; }
                if (snapshotExpired(MonotonicClock::now(), command.computed, profile_->command_timeout_ms))
                    throw std::runtime_error("Independent command watchdog expired");
                checkTilt(sensors.state.root_quaternion_wxyz, command.expected_quaternion);
                // Re-evaluate torque at writer rate, not just when inference produced the target.
                for (std::size_t joint = 0; joint < kJointCount; ++joint) {
                    const float predicted = profile_->kp[joint] * (command.position[joint] - sensors.state.joint_position[joint]) -
                                            profile_->kd[joint] * sensors.state.joint_velocity[joint];
                    if (!std::isfinite(predicted) || std::abs(predicted) > profile_->torque_limit[joint])
                        throw std::runtime_error("Writer torque envelope exceeded");
                }
                send(command, sensors, false);
                deadline += std::chrono::milliseconds(2);
                std::this_thread::sleep_until(deadline);
            }
        } catch (const std::exception& error) { latchFault(error.what()); }
        const auto stop_deadline = MonotonicClock::now() + std::chrono::duration<double>(profile_->shutdown_seconds);
        while (MonotonicClock::now() < stop_deadline) {
            const auto sensors = snapshot();
            // Do not synthesize a damping command from stale/NaN feedback.
            if (snapshotExpired(MonotonicClock::now(), sensors.received, profile_->state_timeout_ms)) break;
            bool finite_velocity = true;
            for (float velocity : sensors.state.joint_velocity) finite_velocity &= std::isfinite(velocity);
            if (!finite_velocity) break;
            try { send({}, sensors, true); } catch (...) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        running_ = false;
        std::cout << "WRITER_EXIT limited_steps=" << limited_steps_.load() << std::endl;
    }

    Options options_;
    const Bundle& bundle_;
    PolicyRunner policy_;
    ObservationHistory history_;
    std::optional<SafetyProfile> profile_;
    unitree::robot::ChannelPtr<LowState> state_channel_;
    std::unique_ptr<unitree::robot::ChannelPublisher<LowCommand>> publisher_;
    std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> switcher_;
    std::mutex sensor_mutex_, command_mutex_;
    SensorSnapshot sensors_;
    CommandSnapshot command_;
    bool have_state_ = false;
    std::uint32_t last_tick_ = 0;
    std::atomic<bool> fault_{false}, running_{true}, actuation_started_{false};
    std::atomic<std::size_t> limited_steps_{0};
    JointArray previous_position_{}, previous_action_{};
    std::array<float, 4> expected_quaternion_{1, 0, 0, 0};
    MonotonicClock::time_point last_target_time_{};
    std::thread writer_;
};
}  // namespace

int main(int argument_count, char** arguments) {
    try {
        const Options options = parseOptions(argument_count, arguments);
        const Bundle bundle(options.bundle);
        Controller controller(options, bundle);
        return controller.run();
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
}
