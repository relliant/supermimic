#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace supermimic {

inline constexpr std::size_t kJointCount = 29;
inline constexpr std::size_t kObservationDimension = 99;

struct RobotState {
    std::array<float, kJointCount> joint_position{};
    std::array<float, kJointCount> joint_velocity{};
    std::array<float, 4> root_quaternion_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
    std::array<float, 3> angular_velocity_body{};
};

class Bundle {
public:
    explicit Bundle(std::filesystem::path directory);

    nlohmann::json manifest;
    std::string manifest_sha256;
    std::size_t frame_count{};
    std::size_t history_length{};
    std::size_t goal_frames{};
    std::array<float, kJointCount> default_joint_position{};
    std::array<float, kJointCount> action_scale{};
    std::array<float, kJointCount> kp{};
    std::array<float, kJointCount> kd{};
    std::array<float, kJointCount> effort_limit{};
    std::array<float, kJointCount> velocity_limit{};
    std::array<float, kJointCount> joint_lower{};
    std::array<float, kJointCount> joint_upper{};

    std::array<float, 4> referenceQuaternion(std::size_t frame) const;
    std::array<float, kJointCount> referenceJointPosition(std::size_t frame) const;
    std::array<float, kJointCount> referenceJointVelocity(std::size_t frame) const;
    // World-frame linear velocity followed by world-frame angular velocity.
    std::array<float, 6> referenceRootVelocity(std::size_t frame) const;
    std::vector<float> goal(std::size_t frame) const;

private:
    friend class PolicyRunner;
    std::filesystem::path directory_;
    std::vector<float> reference_;
    std::vector<float> goals_;
};

std::array<float, kObservationDimension> buildObservation(
    const RobotState& state,
    const std::array<float, kJointCount>& previous_action,
    const std::array<float, 4>& reference_quaternion_wxyz,
    const std::array<float, kJointCount>& default_joint_position);

// Apply the initial reference-to-robot yaw offset to subsequent reference frames.
class ReferenceAlignment {
public:
    ReferenceAlignment(const std::array<float, 4>& initial_reference_quaternion_wxyz,
                       const std::array<float, 4>& initial_robot_quaternion_wxyz);
    std::array<float, 4> align(const std::array<float, 4>& reference_quaternion_wxyz) const;

private:
    std::array<float, 4> yaw_offset_wxyz_;
};

std::array<float, 4> alignReferenceYaw(
    const std::array<float, 4>& reference_quaternion_wxyz,
    const std::array<float, 4>& initial_robot_quaternion_wxyz);

class ObservationHistory {
public:
    explicit ObservationHistory(std::size_t history_length);
    void reset();
    const std::vector<float>& append(
        const std::array<float, kObservationDimension>& observation);

private:
    std::vector<float> history_;
    bool initialized_{false};
};

class PolicyRunner {
public:
    explicit PolicyRunner(const Bundle& bundle, int threads = 1);
    ~PolicyRunner();
    PolicyRunner(PolicyRunner&&) noexcept;
    PolicyRunner& operator=(PolicyRunner&&) noexcept;
    PolicyRunner(const PolicyRunner&) = delete;
    PolicyRunner& operator=(const PolicyRunner&) = delete;

    // A runner owns reusable tensor buffers and must not be called concurrently.
    std::array<float, kJointCount> run(const std::vector<float>& history,
                                      const std::vector<float>& goal);

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace supermimic
