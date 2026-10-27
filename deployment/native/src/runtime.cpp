#include "supermimic/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>
#include <openssl/evp.h>

#ifdef SUPERMIMIC_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace supermimic {
namespace {

constexpr std::size_t kReferenceWidth = 71;

std::size_t checkedProduct(std::size_t first, std::size_t second) {
    if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second) {
        throw std::runtime_error("Bundle dimensions overflow addressable memory");
    }
    return first * second;
}

std::size_t positiveSize(const nlohmann::json& value, const std::string& name) {
    if (!value.is_number_integer() || (value.is_number_unsigned()
            ? value.get<std::uint64_t>() == 0 : value.get<std::int64_t>() <= 0)) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (result > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(name + " exceeds addressable memory");
    }
    return static_cast<std::size_t>(result);
}

template <typename Container>
void requireFinite(const Container& values, const std::string& name) {
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error(name + " contains non-finite values");
    }
}

Eigen::Quaternionf quaternion(const std::array<float, 4>& values) {
    requireFinite(values, "Quaternion");
    Eigen::Quaternionf result(values[0], values[1], values[2], values[3]);
    const float norm = result.norm();
    if (!std::isfinite(norm) || norm < 1.0e-6F) {
        throw std::runtime_error("Quaternion has invalid norm");
    }
    result.coeffs() /= norm;
    return result;
}

std::array<float, 4> quaternionArray(const Eigen::Quaternionf& value) {
    return {value.w(), value.x(), value.y(), value.z()};
}

float yaw(const Eigen::Quaternionf& value) {
    const Eigen::Matrix3f rotation = value.toRotationMatrix();
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

std::filesystem::path packageFile(const std::filesystem::path& directory,
                                  const std::string& filename) {
    const auto path = directory / filename;
    // Fixed basenames and no symlinks keep package contents inside the bundle.
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)) ||
        !std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Missing, non-regular, or symlinked package file: " + filename);
    }
    return path;
}

std::string sha256(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot read package file: " + path.string());
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("Cannot initialize SHA-256");
    }
    std::array<char, 65536> buffer{};
    while (stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || stream.gcount()) {
        if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(stream.gcount())) != 1) {
            throw std::runtime_error("SHA-256 update failed");
        }
    }
    if (!stream.eof()) {
        throw std::runtime_error("Error reading package file: " + path.string());
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
        throw std::runtime_error("SHA-256 finalization failed");
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        result << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return result.str();
}

std::vector<float> readFloats(const std::filesystem::path& path, std::size_t count) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "Runtime requires IEEE-754 float32");
    const std::size_t byte_count = checkedProduct(count, sizeof(float));
    if (std::filesystem::file_size(path) != byte_count ||
        byte_count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Incorrect binary size: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    std::vector<float> result(count);
    std::array<unsigned char, 4> bytes{};
    for (float& value : result) {
        stream.read(reinterpret_cast<char*>(bytes.data()), 4);
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
            (static_cast<std::uint32_t>(bytes[3]) << 24U);
        std::memcpy(&value, &bits, sizeof(value));
    }
    if (!stream) {
        throw std::runtime_error("Cannot read binary package file: " + path.string());
    }
    requireFinite(result, path.filename().string());
    return result;
}

std::array<float, kJointCount> jointArray(const nlohmann::json& manifest, const char* name) {
    const auto& values = manifest.at(name);
    if (!values.is_array() || values.size() != kJointCount ||
        !std::all_of(values.begin(), values.end(), [](const auto& value) { return value.is_number(); })) {
        throw std::runtime_error(std::string(name) + " must contain 29 numeric values");
    }
    auto result = values.get<std::array<float, kJointCount>>();
    requireFinite(result, name);
    return result;
}

}  // namespace

Bundle::Bundle(std::filesystem::path directory) : directory_(std::filesystem::canonical(directory)) {
    manifest_sha256 = sha256(packageFile(directory_, "manifest.json"));
    std::ifstream manifest_stream(packageFile(directory_, "manifest.json"));
    manifest = nlohmann::json::parse(manifest_stream);
    if (!manifest.is_object() || positiveSize(manifest.at("schema_version"), "schema_version") != 1 ||
        !manifest.at("control_dt").is_number() ||
        !std::isfinite(manifest.at("control_dt").get<double>()) ||
        std::abs(manifest.at("control_dt").get<double>() - 0.02) > 1.0e-9) {
        throw std::runtime_error("Bundle must use schema_version=1 and control_dt=0.02");
    }
    frame_count = positiveSize(manifest.at("frame_count"), "frame_count");
    history_length = positiveSize(manifest.at("history_length"), "history_length");
    goal_frames = positiveSize(manifest.at("goal_frames"), "goal_frames");
    const auto goal_dimension = checkedProduct(goal_frames, 73);
    checkedProduct(history_length, kObservationDimension);
    if (positiveSize(manifest.at("observation_dim"), "observation_dim") != kObservationDimension ||
        positiveSize(manifest.at("action_dim"), "action_dim") != kJointCount ||
        positiveSize(manifest.at("goal_dim"), "goal_dim") != goal_dimension) {
        throw std::runtime_error("Invalid observation, action, or goal dimensions");
    }
    const auto& joint_names = manifest.at("joint_names");
    std::set<std::string> unique_names;
    if (!joint_names.is_array() || joint_names.size() != kJointCount) {
        throw std::runtime_error("joint_names must contain 29 unique names");
    }
    for (const auto& name : joint_names) {
        if (!name.is_string() || name.get<std::string>().empty() ||
            !unique_names.insert(name.get<std::string>()).second) {
            throw std::runtime_error("joint_names must contain 29 unique nonempty strings");
        }
    }
    default_joint_position = jointArray(manifest, "default_joint_position");
    action_scale = jointArray(manifest, "action_scale");
    kp = jointArray(manifest, "kp");
    kd = jointArray(manifest, "kd");
    effort_limit = jointArray(manifest, "effort_limit");
    velocity_limit = jointArray(manifest, "velocity_limit");
    joint_lower = jointArray(manifest, "joint_lower");
    joint_upper = jointArray(manifest, "joint_upper");
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (action_scale[joint] <= 0 || kp[joint] < 0 || kd[joint] < 0 ||
            effort_limit[joint] <= 0 || velocity_limit[joint] <= 0 ||
            joint_lower[joint] >= joint_upper[joint] ||
            default_joint_position[joint] < joint_lower[joint] ||
            default_joint_position[joint] > joint_upper[joint]) {
            throw std::runtime_error("Invalid gains, scales, or joint limits at index " + std::to_string(joint));
        }
    }
    const auto& hashes = manifest.at("sha256");
    if (!hashes.is_object() || (hashes.size() != 3 && hashes.size() != 4) ||
        (hashes.size() == 4 && !hashes.contains("parity.json"))) {
        throw std::runtime_error("sha256 must contain three payload files and optionally parity.json");
    }
    std::vector<std::pair<const char*, const char*>> payloads{
        {"policy_file", "policy.onnx"}, {"reference_file", "reference.bin"}, {"goals_file", "goals.bin"}};
    if (hashes.contains("parity.json")) {
        payloads.emplace_back(nullptr, "parity.json");
    }
    for (const auto& entry : payloads) {
        if (entry.first != nullptr && manifest.at(entry.first) != entry.second) {
            throw std::runtime_error("Package filenames must use the schema's fixed basenames");
        }
        const auto path = packageFile(directory_, entry.second);
        const auto& hash_value = hashes.at(entry.second);
        if (!hash_value.is_string()) {
            throw std::runtime_error("Package SHA-256 must be a hexadecimal string");
        }
        auto expected = hash_value.get<std::string>();
        if (expected.size() != 64 || !std::all_of(expected.begin(), expected.end(), [](char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f') || (character >= 'A' && character <= 'F');
            })) {
            throw std::runtime_error("Invalid package SHA-256: " + std::string(entry.second));
        }
        std::transform(expected.begin(), expected.end(), expected.begin(), [](char character) {
            return character >= 'A' && character <= 'F' ? static_cast<char>(character - 'A' + 'a') : character;
        });
        if (sha256(path) != expected) {
            throw std::runtime_error("Package SHA-256 mismatch: " + std::string(entry.second));
        }
        if (std::filesystem::file_size(path) == 0) {
            throw std::runtime_error("Empty package payload: " + std::string(entry.second));
        }
    }
    reference_ = readFloats(directory_ / "reference.bin", checkedProduct(frame_count, kReferenceWidth));
    goals_ = readFloats(directory_ / "goals.bin", checkedProduct(frame_count, goal_dimension));
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto values = referenceQuaternion(frame);
        double norm_squared = 0;
        for (float value : values) {
            norm_squared += static_cast<double>(value) * value;
        }
        if (std::abs(norm_squared - 1.0) > 0.002) {
            throw std::runtime_error("Reference quaternion must have unit norm at frame " + std::to_string(frame));
        }
    }
}

std::array<float, 4> Bundle::referenceQuaternion(std::size_t frame) const {
    if (frame >= frame_count) {
        throw std::out_of_range("Reference frame out of range");
    }
    std::array<float, 4> result{};
    std::copy_n(reference_.begin() + frame * kReferenceWidth + 3, result.size(), result.begin());
    return result;
}

std::array<float, kJointCount> Bundle::referenceJointPosition(std::size_t frame) const {
    if (frame >= frame_count) {
        throw std::out_of_range("Reference frame out of range");
    }
    std::array<float, kJointCount> result{};
    std::copy_n(reference_.begin() + frame * kReferenceWidth + 7, result.size(), result.begin());
    return result;
}

std::array<float, kJointCount> Bundle::referenceJointVelocity(std::size_t frame) const {
    if (frame >= frame_count) {
        throw std::out_of_range("Reference frame out of range");
    }
    std::array<float, kJointCount> result{};
    std::copy_n(reference_.begin() + frame * kReferenceWidth + 36, result.size(), result.begin());
    return result;
}

std::array<float, 6> Bundle::referenceRootVelocity(std::size_t frame) const {
    if (frame >= frame_count) {
        throw std::out_of_range("Reference frame out of range");
    }
    std::array<float, 6> result{};
    std::copy_n(reference_.begin() + frame * kReferenceWidth + 65, result.size(), result.begin());
    return result;
}

std::vector<float> Bundle::goal(std::size_t frame) const {
    if (frame >= frame_count) {
        throw std::out_of_range("Goal frame out of range");
    }
    const auto dimension = goal_frames * 73;
    const auto start = goals_.begin() + frame * dimension;
    return {start, start + dimension};
}

std::array<float, kObservationDimension> buildObservation(
    const RobotState& state, const std::array<float, kJointCount>& previous_action,
    const std::array<float, 4>& reference_quaternion_wxyz,
    const std::array<float, kJointCount>& default_joint_position) {
    requireFinite(state.joint_position, "Joint position");
    requireFinite(state.joint_velocity, "Joint velocity");
    requireFinite(state.angular_velocity_body, "Body angular velocity");
    requireFinite(previous_action, "Previous action");
    requireFinite(default_joint_position, "Default joint position");
    const Eigen::Matrix3f rotation = quaternion(state.root_quaternion_wxyz).toRotationMatrix();
    const Eigen::Matrix3f relative_rotation = rotation.transpose() *
        quaternion(reference_quaternion_wxyz).toRotationMatrix();
    std::array<float, kObservationDimension> observation{};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        observation[joint] = state.joint_position[joint] - default_joint_position[joint];
        observation[29 + joint] = state.joint_velocity[joint];
        observation[64 + joint] = previous_action[joint];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        observation[58 + axis] = state.angular_velocity_body[axis];
        observation[61 + axis] = -rotation(2, static_cast<Eigen::Index>(axis));
        for (std::size_t column = 0; column < 2; ++column) {
            observation[93 + axis * 2 + column] = relative_rotation(
                static_cast<Eigen::Index>(axis), static_cast<Eigen::Index>(column));
        }
    }
    requireFinite(observation, "Observation");
    return observation;
}

ReferenceAlignment::ReferenceAlignment(const std::array<float, 4>& initial_reference_quaternion_wxyz,
                                       const std::array<float, 4>& initial_robot_quaternion_wxyz) {
    const float offset = yaw(quaternion(initial_robot_quaternion_wxyz)) -
                         yaw(quaternion(initial_reference_quaternion_wxyz));
    yaw_offset_wxyz_ = quaternionArray(Eigen::Quaternionf(Eigen::AngleAxisf(offset, Eigen::Vector3f::UnitZ())));
}

std::array<float, 4> ReferenceAlignment::align(const std::array<float, 4>& reference_quaternion_wxyz) const {
    return quaternionArray((quaternion(yaw_offset_wxyz_) * quaternion(reference_quaternion_wxyz)).normalized());
}

std::array<float, 4> alignReferenceYaw(const std::array<float, 4>& reference_quaternion_wxyz,
                                      const std::array<float, 4>& initial_robot_quaternion_wxyz) {
    return ReferenceAlignment(reference_quaternion_wxyz, initial_robot_quaternion_wxyz).align(reference_quaternion_wxyz);
}

ObservationHistory::ObservationHistory(std::size_t history_length) {
    if (history_length == 0) {
        throw std::invalid_argument("History length must be positive");
    }
    history_.resize(checkedProduct(history_length, kObservationDimension));
}

void ObservationHistory::reset() {
    initialized_ = false;
    std::fill(history_.begin(), history_.end(), 0.0F);
}

const std::vector<float>& ObservationHistory::append(
    const std::array<float, kObservationDimension>& observation) {
    requireFinite(observation, "Observation");
    if (!initialized_) {
        for (std::size_t offset = 0; offset < history_.size(); offset += kObservationDimension) {
            std::copy(observation.begin(), observation.end(), history_.begin() + offset);
        }
        initialized_ = true;
    } else {
        // memmove explicitly permits overlapping source and destination ranges.
        std::memmove(history_.data(), history_.data() + kObservationDimension,
                     (history_.size() - kObservationDimension) * sizeof(float));
        std::copy(observation.begin(), observation.end(), history_.end() - kObservationDimension);
    }
    return history_;
}

#ifdef SUPERMIMIC_WITH_ORT
class PolicyRunner::Implementation {
public:
    Implementation(const std::filesystem::path& policy_path, std::size_t history_dimension,
                   std::size_t goal_dimension, int threads)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "supermimic"),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          history_buffer_(history_dimension), goal_buffer_(goal_dimension) {
        if (threads < 1) {
            throw std::invalid_argument("Inference threads must be positive");
        }
        // Configure options before constructing the session; use CPU only.
        options_.SetIntraOpNumThreads(threads);
        options_.SetInterOpNumThreads(1);
        options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(environment_, policy_path.c_str(), options_);
        if (session_.GetInputCount() != 2 || session_.GetOutputCount() != 1) {
            throw std::runtime_error("Policy must have two float32 inputs and one float32 output");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        std::array<bool, 2> assigned{false, false};
        for (std::size_t index = 0; index < 2; ++index) {
            input_names_[index] = session_.GetInputNameAllocated(index, allocator).get();
            auto shape = tensorShape(session_.GetInputTypeInfo(index));
            const std::size_t dimension = elementCount(shape);
            if (input_names_[index] != "obs_hist" && input_names_[index] != "goal")
                throw std::runtime_error("Unexpected policy input name");
            const bool is_history = input_names_[index] == "obs_hist";
            const std::vector<std::int64_t> expected_shape = is_history
                ? std::vector<std::int64_t>{1, static_cast<std::int64_t>(history_dimension / 99), 99}
                : std::vector<std::int64_t>{1, static_cast<std::int64_t>(goal_dimension)};
            if (shape != expected_shape)
                throw std::runtime_error("Policy input rank/shape does not match deployment contract");
            const std::size_t slot = is_history ? 0 : 1;
            if (assigned[slot] || dimension != (is_history ? history_dimension : goal_dimension)) {
                throw std::runtime_error("Policy input dimensions do not match bundle");
            }
            assigned[slot] = true;
            auto& buffer = is_history ? history_buffer_ : goal_buffer_;
            input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_, buffer.data(), buffer.size(), shape.data(), shape.size()));
        }
        if (!assigned[0] || !assigned[1]) {
            throw std::runtime_error("Policy is missing history or goal input");
        }
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
        auto output_shape = tensorShape(session_.GetOutputTypeInfo(0));
        if (output_name_ != "action" || output_shape != std::vector<std::int64_t>{1, 29}) {
            throw std::runtime_error("Policy output must be action[1,29]");
        }
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_, output_buffer_.data(),
            output_buffer_.size(), output_shape.data(), output_shape.size());
    }

    std::array<float, kJointCount> run(const std::vector<float>& history, const std::vector<float>& goal) {
        if (history.size() != history_buffer_.size() || goal.size() != goal_buffer_.size()) {
            throw std::invalid_argument("Inference input dimension mismatch");
        }
        requireFinite(history, "Policy history");
        requireFinite(goal, "Policy goal");
        std::copy(history.begin(), history.end(), history_buffer_.begin());
        std::copy(goal.begin(), goal.end(), goal_buffer_.begin());
        const std::array<const char*, 2> names{input_names_[0].c_str(), input_names_[1].c_str()};
        const char* output_name = output_name_.c_str();
        session_.Run(Ort::RunOptions{nullptr}, names.data(), input_tensors_.data(), input_tensors_.size(),
                     &output_name, &output_tensor_, 1);
        requireFinite(output_buffer_, "Policy action");
        return output_buffer_;
    }

private:
    static std::vector<std::int64_t> tensorShape(const Ort::TypeInfo& information) {
        if (information.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error("Policy inputs and outputs must be tensors");
        }
        auto tensor = information.GetTensorTypeAndShapeInfo();
        if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Policy tensors must use float32");
        }
        auto shape = tensor.GetShape();
        if (shape.empty()) {
            throw std::runtime_error("Policy tensors cannot be scalars");
        }
        if (shape.front() < 0) {
            shape.front() = 1;
        }
        return shape;
    }

    static std::size_t elementCount(const std::vector<std::int64_t>& shape) {
        std::size_t count = 1;
        for (auto dimension : shape) {
            if (dimension <= 0) {
                throw std::runtime_error("Only the policy batch dimension may be dynamic");
            }
            count = checkedProduct(count, static_cast<std::size_t>(dimension));
        }
        return count;
    }

    Ort::Env environment_;
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_;
    std::vector<float> history_buffer_;
    std::vector<float> goal_buffer_;
    std::array<float, kJointCount> output_buffer_{};
    std::array<std::string, 2> input_names_;
    std::string output_name_;
    std::vector<Ort::Value> input_tensors_;
    Ort::Value output_tensor_{nullptr};
};

PolicyRunner::PolicyRunner(const Bundle& bundle, int threads)
    : implementation_(std::make_unique<Implementation>(bundle.directory_ / "policy.onnx",
          checkedProduct(bundle.history_length, kObservationDimension), checkedProduct(bundle.goal_frames, 73), threads)) {}

std::array<float, kJointCount> PolicyRunner::run(const std::vector<float>& history,
                                               const std::vector<float>& goal) {
    if (!implementation_) {
        throw std::logic_error("Cannot use a moved-from policy runner");
    }
    return implementation_->run(history, goal);
}
#else
class PolicyRunner::Implementation {};

PolicyRunner::PolicyRunner(const Bundle&, int) {
    throw std::runtime_error("Policy inference unavailable: rebuild with SUPERMIMIC_WITH_ORT=ON");
}

std::array<float, kJointCount> PolicyRunner::run(const std::vector<float>&, const std::vector<float>&) {
    throw std::runtime_error("Policy inference unavailable: rebuild with SUPERMIMIC_WITH_ORT=ON");
}
#endif

PolicyRunner::~PolicyRunner() = default;
PolicyRunner::PolicyRunner(PolicyRunner&&) noexcept = default;
PolicyRunner& PolicyRunner::operator=(PolicyRunner&&) noexcept = default;

}  // namespace supermimic
