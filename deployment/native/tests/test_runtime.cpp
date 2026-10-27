#include "supermimic/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openssl/evp.h>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(float actual, float expected, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 2.0e-6F, message);
}

template <typename Function>
void requireThrows(Function function, const std::string& expected_message) {
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected_message) != std::string::npos,
                "Wrong exception: " + std::string(error.what()) + "; expected " + expected_message);
        return;
    }
    throw std::runtime_error("Expected exception containing: " + expected_message);
}

std::string digestFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    require(EVP_Digest(bytes.data(), bytes.size(), digest.data(), &digest_size, EVP_sha256(), nullptr) == 1,
            "Test SHA-256 failed");
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        result << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return result.str();
}

void writeFloats(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    for (float value : values) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int byte = 0; byte < 4; ++byte) {
            stream.put(static_cast<char>((bits >> (8U * byte)) & 0xffU));
        }
    }
    require(static_cast<bool>(stream), "Test binary write failed");
}

class TestBundle {
public:
    TestBundle() {
        const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (std::size_t attempt = 0; attempt < 100; ++attempt) {
            directory = std::filesystem::temp_directory_path() /
                ("supermimic-runtime-test-" + std::to_string(seed) + '-' + std::to_string(attempt));
            if (std::filesystem::create_directory(directory)) {
                break;
            }
            require(attempt < 99, "Cannot create unique temporary test directory");
        }
        manifest = {{"schema_version", 1}, {"control_dt", 0.02}, {"history_length", 3},
                    {"goal_frames", 2}, {"observation_dim", 99}, {"goal_dim", 146},
                    {"action_dim", 29}, {"frame_count", 2}, {"policy_file", "policy.onnx"},
                    {"reference_file", "reference.bin"}, {"goals_file", "goals.bin"}};
        manifest["joint_names"] = nlohmann::json::array();
        for (std::size_t joint = 0; joint < 29; ++joint) {
            manifest["joint_names"].push_back("joint_" + std::to_string(joint));
        }
        for (const char* field : {"default_joint_position", "action_scale", "kp", "kd", "effort_limit",
                                  "velocity_limit", "joint_lower", "joint_upper"}) {
            const float value = std::string(field) == "default_joint_position" ? 0.0F :
                                std::string(field) == "joint_lower" ? -3.0F : 3.0F;
            manifest[field] = std::vector<float>(29, value);
        }
        reference.resize(2 * 71);
        reference[3] = 1;
        reference[71 + 3] = 1;
        for (std::size_t joint = 0; joint < 29; ++joint) {
            reference[7 + joint] = 0.01F * static_cast<float>(joint);
            reference[71 + 7 + joint] = -0.02F * static_cast<float>(joint);
            reference[36 + joint] = 50.0F + static_cast<float>(joint);
            reference[71 + 36 + joint] = -20.0F - static_cast<float>(joint);
        }
        for (std::size_t component = 0; component < 6; ++component) {
            reference[65 + component] = 10.0F + static_cast<float>(component);
            reference[71 + 65 + component] = -10.0F - static_cast<float>(component);
        }
        goals.resize(2 * 146);
        for (std::size_t index = 0; index < goals.size(); ++index) {
            goals[index] = 0.125F * static_cast<float>(index);
        }
        std::ofstream(directory / "policy.onnx", std::ios::binary) << "test-policy-payload";
        std::ofstream(directory / "parity.json") << "{\"samples\": []}";
        writeFloats(directory / "reference.bin", reference);
        writeFloats(directory / "goals.bin", goals);
        refreshHashes();
    }

    ~TestBundle() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
    TestBundle(const TestBundle&) = delete;
    TestBundle& operator=(const TestBundle&) = delete;

    void writeManifest() const {
        std::ofstream stream(directory / "manifest.json");
        stream << manifest.dump(2);
        require(static_cast<bool>(stream), "Test manifest write failed");
    }

    void refreshHashes() {
        for (const char* filename : {"policy.onnx", "reference.bin", "goals.bin", "parity.json"}) {
            manifest["sha256"][filename] = digestFile(directory / filename);
        }
        writeManifest();
    }

    std::filesystem::path directory;
    nlohmann::json manifest;
    std::vector<float> reference;
    std::vector<float> goals;
};

void testObservation() {
    supermimic::RobotState state;
    std::array<float, 29> previous_action{};
    std::array<float, 29> default_position{};
    for (std::size_t joint = 0; joint < 29; ++joint) {
        state.joint_position[joint] = 0.25F * static_cast<float>(joint);
        default_position[joint] = -0.125F * static_cast<float>(joint);
        state.joint_velocity[joint] = 4.0F * static_cast<float>(joint);
        previous_action[joint] = -2.0F * static_cast<float>(joint);
    }
    state.angular_velocity_body = {2, -4, 6};
    const float half_sqrt = std::sqrt(0.5F);
    const std::array<float, 4> yaw_quaternion{half_sqrt, 0, 0, half_sqrt};
    const auto observation = supermimic::buildObservation(state, previous_action, yaw_quaternion, default_position);
    for (std::size_t joint = 0; joint < 29; ++joint) {
        requireNear(observation[joint], 0.375F * static_cast<float>(joint), "Raw joint delta mismatch");
        requireNear(observation[29 + joint], state.joint_velocity[joint], "Joint velocities must not use OBS_SCALE");
        requireNear(observation[64 + joint], previous_action[joint], "Previous action offset mismatch");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        requireNear(observation[58 + axis], state.angular_velocity_body[axis], "Angular velocity must not use OBS_SCALE");
        requireNear(observation[61 + axis], axis == 2 ? -1.0F : 0.0F, "Projected gravity mismatch");
    }
    const std::array<float, 6> expected_yaw{0, -1, 1, 0, 0, 0};
    for (std::size_t index = 0; index < 6; ++index) {
        requireNear(observation[93 + index], expected_yaw[index], "6D rotation must flatten two columns row-major");
    }
    state.root_quaternion_wxyz = {half_sqrt, half_sqrt, 0, 0};
    const auto rolled = supermimic::buildObservation(state, previous_action, yaw_quaternion, default_position);
    requireNear(rolled[61], 0, "Roll gravity x");
    requireNear(rolled[62], -1, "Roll gravity y");
    requireNear(rolled[63], 0, "Roll gravity z");
    const std::array<float, 6> expected_relative{0, -1, 0, 0, -1, 0};
    for (std::size_t index = 0; index < 6; ++index) {
        requireNear(rolled[93 + index], expected_relative[index], "Relative rotation must be R transpose times Rref");
    }
    state.root_quaternion_wxyz.fill(0);
    requireThrows([&] { supermimic::buildObservation(state, previous_action, yaw_quaternion, default_position); }, "norm");
    state.root_quaternion_wxyz = {1, 0, 0, 0};
    state.joint_velocity[1] = std::numeric_limits<float>::infinity();
    requireThrows([&] { supermimic::buildObservation(state, previous_action, yaw_quaternion, default_position); }, "non-finite");
}

void testYawAlignment() {
    const float half_sqrt = std::sqrt(0.5F);
    const std::array<float, 4> identity{1, 0, 0, 0};
    const std::array<float, 4> robot_yaw{half_sqrt, 0, 0, half_sqrt};
    const supermimic::ReferenceAlignment alignment(identity, robot_yaw);
    const auto aligned = alignment.align(identity);
    requireNear(aligned[0], half_sqrt, "Initial yaw alignment w");
    requireNear(aligned[3], half_sqrt, "Initial yaw alignment z");
    const auto advanced = alignment.align(robot_yaw);
    requireNear(std::abs(advanced[3]), 1, "Reference yaw evolution must be preserved");
    requireNear(advanced[0], 0, "Reference yaw evolution w");
    const auto rolled = alignment.align({half_sqrt, half_sqrt, 0, 0});
    supermimic::RobotState state;
    state.root_quaternion_wxyz = rolled;
    const auto features = supermimic::buildObservation(state, {}, rolled, {});
    requireNear(features[62], -1, "Yaw alignment must preserve reference tilt");
    const auto utility = supermimic::alignReferenceYaw(identity, robot_yaw);
    requireNear(utility[3], half_sqrt, "Yaw utility mismatch");
}

void testHistory() {
    requireThrows([] { supermimic::ObservationHistory history(0); }, "positive");
    supermimic::ObservationHistory history(3);
    std::array<float, 99> first{};
    std::array<float, 99> second{};
    std::array<float, 99> third{};
    std::array<float, 99> fourth{};
    first.fill(1);
    second.fill(2);
    third.fill(3);
    fourth.fill(4);
    const auto& initialized = history.append(first);
    require(initialized.size() == 297 && std::all_of(initialized.begin(), initialized.end(),
            [](float value) { return value == 1; }), "First observation must repeat across history");
    const float* buffer = initialized.data();
    history.append(second);
    history.append(third);
    const auto& shifted = history.append(fourth);
    require(shifted.data() == buffer, "History must reuse its storage");
    for (std::size_t index = 0; index < shifted.size(); ++index) {
        requireNear(shifted[index], static_cast<float>(2 + index / 99), "History must be oldest-first, newest-last");
    }
    history.reset();
    const auto& reset = history.append(second);
    require(std::all_of(reset.begin(), reset.end(), [](float value) { return value == 2; }), "Reset must reinitialize all rows");
    supermimic::ObservationHistory one_row(1);
    one_row.append(first);
    require(one_row.append(second).front() == 2, "Single-row history append");
    second[0] = std::numeric_limits<float>::quiet_NaN();
    requireThrows([&] { history.append(second); }, "non-finite");
}

void testBundleLayout() {
    TestBundle fixture;
    const supermimic::Bundle bundle(fixture.directory);
    require(bundle.frame_count == 2 && bundle.history_length == 3 && bundle.goal_frames == 2,
            "Manifest dimensions were not loaded");
    requireNear(bundle.referenceQuaternion(1)[0], 1, "Quaternion binary offset");
    const auto positions = bundle.referenceJointPosition(1);
    requireNear(positions[28], -0.56F, "Joint binary offset or row width");
    for (std::size_t frame = 0; frame < 2; ++frame) {
        const auto joint_velocity = bundle.referenceJointVelocity(frame);
        for (std::size_t joint = 0; joint < 29; ++joint) {
            requireNear(joint_velocity[joint], fixture.reference[frame * 71 + 36 + joint],
                        "Joint velocity binary offset or row stride");
        }
        const auto root_velocity = bundle.referenceRootVelocity(frame);
        for (std::size_t component = 0; component < 6; ++component) {
            requireNear(root_velocity[component], fixture.reference[frame * 71 + 65 + component],
                        "World root linear/angular velocity binary offset or row stride");
        }
    }
    const auto goal = bundle.goal(1);
    require(goal.size() == 146, "Goal dimension");
    requireNear(goal.front(), fixture.goals[146], "Goal frame stride");
    requireNear(goal.back(), fixture.goals.back(), "Goal final value");
    requireThrows([&] { bundle.goal(2); }, "out of range");
    requireThrows([&] { bundle.referenceQuaternion(2); }, "out of range");
    requireThrows([&] { bundle.referenceJointPosition(2); }, "out of range");
    requireThrows([&] { bundle.referenceJointVelocity(2); }, "out of range");
    requireThrows([&] { bundle.referenceRootVelocity(2); }, "out of range");
    fixture.manifest["sha256"].erase("parity.json");
    fixture.writeManifest();
    const supermimic::Bundle without_optional_parity(fixture.directory);
    require(without_optional_parity.frame_count == 2, "Parity must remain optional for core bundles");
}

void testRejectedPackages() {
    {
        TestBundle fixture;
        std::ofstream(fixture.directory / "policy.onnx", std::ios::app) << "corruption";
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "SHA-256 mismatch");
    }
    {
        TestBundle fixture;
        fixture.goals.pop_back();
        writeFloats(fixture.directory / "goals.bin", fixture.goals);
        fixture.refreshHashes();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "Incorrect binary size");
    }
    {
        TestBundle fixture;
        fixture.reference.back() = std::numeric_limits<float>::quiet_NaN();
        writeFloats(fixture.directory / "reference.bin", fixture.reference);
        fixture.refreshHashes();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "non-finite");
    }
    {
        TestBundle fixture;
        fixture.reference[3] = 0;
        writeFloats(fixture.directory / "reference.bin", fixture.reference);
        fixture.refreshHashes();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "unit norm");
    }
    {
        TestBundle fixture;
        fixture.manifest["policy_file"] = "../policy.onnx";
        fixture.writeManifest();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "fixed basenames");
    }
    {
        TestBundle fixture;
        std::filesystem::rename(fixture.directory / "reference.bin", fixture.directory / "outside.bin");
        std::filesystem::create_symlink("outside.bin", fixture.directory / "reference.bin");
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "symlinked");
    }
    {
        TestBundle fixture;
        fixture.manifest["goal_dim"] = 73;
        fixture.writeManifest();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "dimensions");
    }
    {
        TestBundle fixture;
        fixture.manifest["history_length"] = -1;
        fixture.writeManifest();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "positive integer");
    }
    {
        TestBundle fixture;
        fixture.manifest["sha256"]["../outside.bin"] = std::string(64, '0');
        fixture.writeManifest();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "sha256 must contain");
    }
    {
        TestBundle fixture;
        fixture.manifest["joint_upper"][0] = -1;
        fixture.writeManifest();
        requireThrows([&] { supermimic::Bundle bundle(fixture.directory); }, "joint limits");
    }
}

}  // namespace

int main() {
    try {
        testObservation();
        testYawAlignment();
        testHistory();
        testBundleLayout();
        testRejectedPackages();
        std::cout << "Runtime core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime test failure: " << error.what() << '\n';
        return 1;
    }
}
