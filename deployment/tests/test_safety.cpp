#include "supermimic/safety.hpp"

#include <filesystem>
#include <iostream>
#include <limits>

namespace {
using namespace supermimic;

void require(bool condition, const char* explanation) {
    if (!condition) throw std::runtime_error(explanation);
}

template <typename Operation>
void requireRejected(Operation operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Unsafe operation unexpectedly accepted");
}

void testStateAndTiming() {
    const auto now = MonotonicClock::now();
    require(snapshotExpired(now, {}, 100), "Missing state must expire");
    require(snapshotExpired(now, now - std::chrono::milliseconds(101), 100), "Stale snapshot accepted");
    require(snapshotExpired(now, now + std::chrono::milliseconds(1), 100), "Future timestamp accepted");
    require(!snapshotExpired(now, now - std::chrono::milliseconds(50), 100), "Fresh snapshot rejected");

    SafetyProfile profile;
    profile.position_lower.fill(-2);
    profile.position_upper.fill(2);
    profile.velocity_limit.fill(3);
    profile.torque_limit.fill(5);
    RobotState state;
    JointArray temperature{}, torque{};
    require(validateRobotState(state, profile, temperature, torque, false).empty(), "Valid state rejected");
    for (float velocity : {-3.1F, 3.1F}) {
        state.joint_velocity[17] = velocity;
        require(!validateRobotState(state, profile, temperature, torque, false).empty(), "Asymmetric velocity guard");
    }
    state.joint_velocity.fill(0);
    state.joint_position[3] = std::numeric_limits<float>::quiet_NaN();
    require(!validateRobotState(state, profile, temperature, torque, false).empty(), "NaN position accepted");
    state.joint_position.fill(0);
    state.root_quaternion_wxyz = {0, 0, 0, 0};
    require(!validateRobotState(state, profile, temperature, torque, false).empty(), "Invalid quaternion accepted");
    state.root_quaternion_wxyz = {1, 0, 0, 0};
    temperature[9] = 100;
    require(!validateRobotState(state, profile, temperature, torque, false).empty(), "Temperature guard missing");
    temperature.fill(0);
    torque[5] = -6;
    require(!validateRobotState(state, profile, temperature, torque, false).empty(), "Negative torque limit missing");
    torque.fill(0);
    require(!validateRobotState(state, profile, temperature, torque, true).empty(), "Motor fault accepted");
}

void testTargetAndHardwareGate(const std::filesystem::path& directory) {
    Bundle bundle(directory);
    SafetyProfile profile;
    profile.position_lower = bundle.joint_lower;
    profile.position_upper = bundle.joint_upper;
    profile.velocity_limit = bundle.velocity_limit;
    profile.torque_limit = bundle.effort_limit;
    profile.kp = bundle.kp;
    profile.kd = bundle.kd;
    profile.target_rate_limit.fill(1);
    RobotState state;
    state.joint_position = bundle.default_joint_position;
    JointArray requested = state.joint_position;
    requested[0] += 1;
    auto result = limitTarget(requested, state.joint_position, state, profile, bundle, 0.01);
    require(result.limited, "Slew rate did not limit target");
    require(std::abs(result.position[0] - state.joint_position[0] - 0.01F) < 1e-6F, "Wrong actual-dt rate limit");
    require(std::abs(result.applied_action[0] * bundle.action_scale[0] - 0.01F) < 1e-6F,
            "Feedback is not the applied dimensionless action");
    requested[1] = std::numeric_limits<float>::infinity();
    requireRejected([&] { limitTarget(requested, state.joint_position, state, profile, bundle, 0.02); });
    requested = state.joint_position;
    requireRejected([&] { limitTarget(requested, state.joint_position, state, profile, bundle, 0); });
    state.joint_velocity[0] = profile.torque_limit[0] / profile.kd[0] * 2;
    requireRejected([&] { limitTarget(requested, requested, state, profile, bundle, 0.02); });

    const auto profile_path = std::filesystem::temp_directory_path() /
        ("supermimic-safety-" + std::to_string(MonotonicClock::now().time_since_epoch().count()) + ".json");
    {
        std::ofstream output(profile_path);
        output << nlohmann::json{{"schema_version", 1}, {"hardware_validated", false},
            {"pelvis_imu_verified", false}, {"safe_exit_verified", false}, {"imu_frame", "pelvis"},
            {"sensor_to_pelvis_wxyz", {1, 0, 0, 0}}};
    }
    try {
        SafetyProfile::load(profile_path.string(), bundle, false);
        requireRejected([&] { SafetyProfile::load(profile_path.string(), bundle, true); });
    } catch (...) {
        std::filesystem::remove(profile_path);
        throw;
    }
    std::filesystem::remove(profile_path);
}
}  // namespace

int main(int argument_count, char** arguments) {
    try {
        testStateAndTiming();
        if (argument_count == 2) testTargetAndHardwareGate(arguments[1]);
        std::cout << "safety tests passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
