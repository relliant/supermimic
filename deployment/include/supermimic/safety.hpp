#pragma once

#include "supermimic/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace supermimic {

using MonotonicClock = std::chrono::steady_clock;
using JointArray = std::array<float, kJointCount>;

struct SafetyProfile {
    bool hardware_validated = false;
    bool pelvis_imu_verified = false;
    bool safe_exit_verified = false;
    JointArray position_lower{}, position_upper{}, velocity_limit{}, torque_limit{};
    JointArray target_rate_limit{}, kp{}, kd{}, damping_kd{};
    std::array<float, 4> sensor_to_pelvis_wxyz{1, 0, 0, 0};
    double state_timeout_ms = 100;
    double command_timeout_ms = 100;
    double inference_timeout_ms = 20;
    double maximum_temperature = 70;
    double maximum_tilt_error = 0.6;
    double preparation_seconds = 5;
    double shutdown_seconds = 0.2;
    double maximum_start_joint_speed = 0.2;
    double maximum_reference_start_speed = 0.2;
    double maximum_start_position_error = 0.1;

    static SafetyProfile load(const std::string& filename, const Bundle& bundle, bool hardware) {
        std::ifstream input(filename);
        if (!input) throw std::runtime_error("Cannot open safety profile: " + filename);
        nlohmann::json configuration;
        input >> configuration;
        if (configuration.at("schema_version").get<int>() != 1)
            throw std::runtime_error("Unsupported safety profile version");
        if (hardware && configuration.value("manifest_sha256", std::string{}) != bundle.manifest_sha256)
            throw std::runtime_error("Hardware profile does not match this exact model/reference manifest");
        SafetyProfile profile;
        profile.hardware_validated = configuration.at("hardware_validated").get<bool>();
        profile.pelvis_imu_verified = configuration.at("pelvis_imu_verified").get<bool>();
        profile.safe_exit_verified = configuration.at("safe_exit_verified").get<bool>();
        if (configuration.at("imu_frame").get<std::string>() != "pelvis")
            throw std::runtime_error("Only a verified, rigid pelvis IMU is supported; torso IMU needs articulated compensation");
        const auto& calibration = configuration.at("sensor_to_pelvis_wxyz");
        if (!calibration.is_array() || calibration.size() != 4)
            throw std::runtime_error("IMU calibration requires exactly four wxyz values");
        profile.sensor_to_pelvis_wxyz = calibration.get<std::array<float, 4>>();
        double quaternion_norm = 0;
        for (float value : profile.sensor_to_pelvis_wxyz) quaternion_norm += value * value;
        if (!std::isfinite(quaternion_norm) || std::abs(quaternion_norm - 1) > 1e-4)
            throw std::runtime_error("IMU calibration quaternion must be normalized");

        auto read_joint_array = [&](const char* key, const JointArray& baseline) {
            JointArray result = baseline;
            if (configuration.contains(key)) {
                const auto& values = configuration.at(key);
                if (!values.is_array() || values.size() != kJointCount)
                    throw std::runtime_error(std::string("Expected exactly 29 joint values for ") + key);
                result = values.get<JointArray>();
            }
            else if (hardware) throw std::runtime_error(std::string("Hardware profile requires explicit ") + key);
            for (float value : result)
                if (!std::isfinite(value)) throw std::runtime_error(std::string("Nonfinite safety parameter: ") + key);
            return result;
        };
        profile.position_lower = read_joint_array("position_lower", bundle.joint_lower);
        profile.position_upper = read_joint_array("position_upper", bundle.joint_upper);
        profile.velocity_limit = read_joint_array("velocity_limit", bundle.velocity_limit);
        profile.torque_limit = read_joint_array("torque_limit", bundle.effort_limit);
        profile.kp = read_joint_array("kp", bundle.kp);
        profile.kd = read_joint_array("kd", bundle.kd);
        JointArray default_rate{}, default_damping{};
        default_rate.fill(3);
        default_damping.fill(2);
        profile.target_rate_limit = read_joint_array("target_rate_limit", default_rate);
        profile.damping_kd = read_joint_array("damping_kd", default_damping);
        auto read_positive = [&](const char* key, double fallback) {
            if (hardware && !configuration.contains(key))
                throw std::runtime_error(std::string("Hardware profile requires explicit ") + key);
            const double value = configuration.value(key, fallback);
            if (!std::isfinite(value) || value <= 0)
                throw std::runtime_error(std::string("Expected positive safety parameter: ") + key);
            return value;
        };
        profile.state_timeout_ms = read_positive("state_timeout_ms", profile.state_timeout_ms);
        profile.command_timeout_ms = read_positive("command_timeout_ms", profile.command_timeout_ms);
        profile.inference_timeout_ms = read_positive("inference_timeout_ms", profile.inference_timeout_ms);
        profile.maximum_temperature = read_positive("maximum_temperature", profile.maximum_temperature);
        profile.maximum_tilt_error = read_positive("maximum_tilt_error", profile.maximum_tilt_error);
        profile.preparation_seconds = read_positive("preparation_seconds", profile.preparation_seconds);
        profile.shutdown_seconds = read_positive("shutdown_seconds", profile.shutdown_seconds);
        profile.maximum_start_joint_speed = read_positive("maximum_start_joint_speed", profile.maximum_start_joint_speed);
        profile.maximum_reference_start_speed = read_positive("maximum_reference_start_speed", profile.maximum_reference_start_speed);
        profile.maximum_start_position_error = read_positive("maximum_start_position_error", profile.maximum_start_position_error);
        if (profile.inference_timeout_ms > 20 || profile.state_timeout_ms > 100 || profile.command_timeout_ms > 100)
            throw std::runtime_error("Profile exceeds the runtime timing budget (20/100/100 ms)");
        if (profile.shutdown_seconds > 2)
            throw std::runtime_error("Shutdown damping is bounded to at most two seconds");
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
            if (profile.position_lower[joint] < bundle.joint_lower[joint] ||
                profile.position_upper[joint] > bundle.joint_upper[joint] ||
                profile.position_lower[joint] >= profile.position_upper[joint] ||
                profile.velocity_limit[joint] <= 0 || profile.velocity_limit[joint] > bundle.velocity_limit[joint] ||
                profile.torque_limit[joint] <= 0 || profile.torque_limit[joint] > bundle.effort_limit[joint] ||
                profile.target_rate_limit[joint] <= 0 || profile.kp[joint] < 0 || profile.kd[joint] < 0 ||
                profile.damping_kd[joint] <= 0)
                throw std::runtime_error("Invalid or out-of-model joint safety limits at index " + std::to_string(joint));
        }
        if (hardware && (!profile.hardware_validated || !profile.pelvis_imu_verified || !profile.safe_exit_verified))
            throw std::runtime_error("Hardware requires a reviewed profile, verified pelvis IMU and verified safe exit");
        return profile;
    }
};

inline std::string validateRobotState(const RobotState& state, const SafetyProfile& profile,
                                      const JointArray& temperature, const JointArray& measured_torque,
                                      bool motor_fault) {
    if (motor_fault) return "motor fault";
    double quaternion_norm = 0;
    for (float value : state.root_quaternion_wxyz) quaternion_norm += value * value;
    if (!std::isfinite(quaternion_norm) || std::abs(quaternion_norm - 1) > 0.05) return "invalid IMU quaternion";
    for (float value : state.angular_velocity_body)
        if (!std::isfinite(value)) return "nonfinite gyro";
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        const float position = state.joint_position[joint];
        const float velocity = state.joint_velocity[joint];
        if (!std::isfinite(position) || !std::isfinite(velocity) || !std::isfinite(temperature[joint]) ||
            !std::isfinite(measured_torque[joint])) return "nonfinite motor state";
        if (position < profile.position_lower[joint] || position > profile.position_upper[joint]) return "joint position limit";
        if (std::abs(velocity) > profile.velocity_limit[joint]) return "joint velocity limit";
        if (std::abs(measured_torque[joint]) > profile.torque_limit[joint]) return "measured torque limit";
        if (temperature[joint] >= profile.maximum_temperature) return "motor overtemperature";
    }
    return {};
}

inline bool snapshotExpired(MonotonicClock::time_point now, MonotonicClock::time_point timestamp, double budget_ms) {
    return timestamp == MonotonicClock::time_point{} || timestamp > now ||
        std::chrono::duration<double, std::milli>(now - timestamp).count() > budget_ms;
}

struct LimitedTarget {
    JointArray position{};
    JointArray applied_action{};
    bool limited = false;
};

inline LimitedTarget limitTarget(const JointArray& requested_position, const JointArray& previous_position,
                                  const RobotState& state, const SafetyProfile& profile,
                                  const Bundle& bundle, double elapsed_seconds) {
    if (!std::isfinite(elapsed_seconds) || elapsed_seconds <= 0 || elapsed_seconds > 0.1)
        throw std::runtime_error("Invalid target update interval");
    LimitedTarget result;
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (!std::isfinite(requested_position[joint]) || !std::isfinite(previous_position[joint]))
            throw std::runtime_error("Nonfinite target");
        const float maximum_delta = profile.target_rate_limit[joint] * static_cast<float>(elapsed_seconds);
        const float lower = std::max(profile.position_lower[joint], previous_position[joint] - maximum_delta);
        const float upper = std::min(profile.position_upper[joint], previous_position[joint] + maximum_delta);
        if (lower > upper) throw std::runtime_error("No feasible bounded target");
        const float target = std::clamp(requested_position[joint], lower, upper);
        const float expected_torque = profile.kp[joint] * (target - state.joint_position[joint]) -
                                      profile.kd[joint] * state.joint_velocity[joint];
        if (!std::isfinite(expected_torque) || std::abs(expected_torque) > profile.torque_limit[joint])
            throw std::runtime_error("Predicted PD torque exceeds safety envelope");
        result.position[joint] = target;
        result.applied_action[joint] = (target - bundle.default_joint_position[joint]) / bundle.action_scale[joint];
        result.limited = result.limited || target != requested_position[joint];
    }
    return result;
}

}  // namespace supermimic
