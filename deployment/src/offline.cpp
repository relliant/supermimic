#include "supermimic/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path bundle;
    std::filesystem::path parity;
    std::size_t benchmark_count{0};
};

Arguments parseArguments(int argument_count, char** argument_values) {
    Arguments arguments;
    for (int index = 1; index < argument_count; ++index) {
        const std::string option = argument_values[index];
        if (option == "--help") {
            std::cout << "Usage: supermimic_offline --bundle PATH [--parity PATH] [--benchmark N]\n";
            std::exit(0);
        }
        if (index + 1 >= argument_count) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const std::string value = argument_values[++index];
        if (option == "--bundle") {
            arguments.bundle = value;
        } else if (option == "--parity") {
            arguments.parity = value;
        } else if (option == "--benchmark") {
            if (value.empty() || !std::all_of(value.begin(), value.end(), [](char character) {
                    return character >= '0' && character <= '9';
                })) {
                throw std::invalid_argument("Benchmark count must be a positive integer");
            }
            const auto count = std::stoull(value);
            if (count == 0 || count > std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("Benchmark count must be a positive addressable integer");
            }
            arguments.benchmark_count = static_cast<std::size_t>(count);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }
    if (arguments.bundle.empty()) {
        throw std::invalid_argument("--bundle PATH is required");
    }
    if (arguments.parity.empty() && arguments.benchmark_count == 0) {
        arguments.benchmark_count = 1000;
    }
    return arguments;
}

std::vector<float> numericVector(const nlohmann::json& sample, const char* key, std::size_t dimension) {
    const auto& field = sample.at(key);
    if (!field.is_array() || field.size() != dimension) {
        throw std::runtime_error(std::string("Parity ") + key + " dimension mismatch");
    }
    std::vector<float> result;
    result.reserve(dimension);
    for (const auto& value : field) {
        if (!value.is_number() || !std::isfinite(value.get<float>())) {
            throw std::runtime_error(std::string("Parity ") + key + " must contain finite numbers");
        }
        result.push_back(value.get<float>());
    }
    return result;
}

template <std::size_t Dimension>
std::array<float, Dimension> numericArray(const nlohmann::json& sample, const char* key) {
    const auto values = numericVector(sample, key, Dimension);
    std::array<float, Dimension> result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
}

struct ErrorSummary {
    double maximum_absolute{0};
    double maximum_relative{0};
    std::size_t mismatches{0};
    std::size_t values{0};

    template <typename Container>
    void compare(const Container& actual, const std::vector<float>& expected,
                 double absolute_tolerance, double relative_tolerance) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error("Comparison dimension mismatch");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double absolute_error = std::abs(static_cast<double>(actual[index]) - expected[index]);
            const double magnitude = std::abs(static_cast<double>(expected[index]));
            maximum_absolute = std::max(maximum_absolute, absolute_error);
            maximum_relative = std::max(maximum_relative, absolute_error / std::max(magnitude, 1.0e-8));
            if (!std::isfinite(actual[index]) || absolute_error > absolute_tolerance + relative_tolerance * magnitude) {
                ++mismatches;
            }
            ++values;
        }
    }

    void print(const char* name) const {
        std::cout << "parity " << name << " max_absolute_error=" << maximum_absolute
                  << " max_relative_error=" << maximum_relative << " mismatches=" << mismatches
                  << '/' << values << '\n';
    }
};

bool checkParity(const std::filesystem::path& path, const supermimic::Bundle& bundle,
                 supermimic::PolicyRunner& runner) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot open parity file: " + path.string());
    }
    nlohmann::json parity;
    stream >> parity;
    const auto& samples = parity.is_array() ? parity : parity.at("samples");
    if (!samples.is_array() || samples.empty()) {
        throw std::runtime_error("Parity file must contain a nonempty samples array");
    }
    supermimic::ObservationHistory history(bundle.history_length);
    ErrorSummary observation_error;
    ErrorSummary history_error;
    ErrorSummary goal_error;
    ErrorSummary action_error;
    std::size_t sample_index = 0;
    for (const auto& sample : samples) {
        const auto& frame_value = sample.at("frame");
        if (!frame_value.is_number_integer() ||
            (!frame_value.is_number_unsigned() && frame_value.get<std::int64_t>() < 0) ||
            frame_value.get<std::uint64_t>() >= bundle.frame_count) {
            throw std::runtime_error("Parity frame outside bundle");
        }
        const auto frame = frame_value.get<std::size_t>();
        supermimic::RobotState state;
        state.joint_position = numericArray<29>(sample, "joint_position");
        state.joint_velocity = numericArray<29>(sample, "joint_velocity");
        state.root_quaternion_wxyz = numericArray<4>(sample, "root_quaternion_wxyz");
        state.angular_velocity_body = numericArray<3>(sample, "angular_velocity_body");
        const auto previous_action = numericArray<29>(sample, "previous_action");
        if (sample.value("reset_history", false)) {
            history.reset();
        }
        const auto observation = supermimic::buildObservation(state, previous_action,
            bundle.referenceQuaternion(frame), bundle.default_joint_position);
        const auto& accumulated = history.append(observation);
        const auto goal = bundle.goal(frame);
        const auto action = runner.run(accumulated, goal);
        observation_error.compare(observation, numericVector(sample, "observation", 99), 2.0e-5, 2.0e-5);
        history_error.compare(accumulated, numericVector(sample, "history", bundle.history_length * 99), 2.0e-5, 2.0e-5);
        goal_error.compare(goal, numericVector(sample, "goal", bundle.goal_frames * 73), 2.0e-5, 2.0e-5);
        const auto expected_action = numericVector(sample, "action", 29);
        const auto previous_mismatches = action_error.mismatches;
        action_error.compare(action, expected_action, 1.0e-5, 1.0e-4);
        if (action_error.mismatches != previous_mismatches) {
            std::cerr << "Action parity failed at sample " << sample_index << " frame " << frame << '\n';
        }
        ++sample_index;
    }
    observation_error.print("observation");
    history_error.print("history");
    goal_error.print("goal");
    action_error.print("action");
    const bool passed = observation_error.mismatches == 0 && history_error.mismatches == 0 &&
                        goal_error.mismatches == 0 && action_error.mismatches == 0;
    std::cout << "parity samples=" << samples.size() << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

void benchmark(const supermimic::Bundle& bundle, supermimic::PolicyRunner& runner, std::size_t count) {
    using Clock = std::chrono::steady_clock;
    supermimic::ObservationHistory history(bundle.history_length);
    std::array<float, 29> previous_action{};
    auto pipeline = [&](std::size_t index) {
        const auto frame = index % bundle.frame_count;
        supermimic::RobotState state;
        state.joint_position = bundle.referenceJointPosition(frame);
        state.root_quaternion_wxyz = bundle.referenceQuaternion(frame);
        const auto observation = supermimic::buildObservation(state, previous_action,
            bundle.referenceQuaternion(frame), bundle.default_joint_position);
        const auto& accumulated = history.append(observation);
        const auto goal = bundle.goal(frame);
        previous_action = runner.run(accumulated, goal);
    };
    constexpr std::size_t warmup_count = 20;
    for (std::size_t index = 0; index < warmup_count; ++index) {
        pipeline(index);
    }
    history.reset();
    previous_action.fill(0.0F);
    std::vector<double> elapsed_milliseconds;
    elapsed_milliseconds.reserve(count);
    std::size_t overruns = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto start = Clock::now();
        pipeline(index);
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        elapsed_milliseconds.push_back(elapsed);
        if (elapsed > 20.0) {
            ++overruns;
        }
    }
    std::sort(elapsed_milliseconds.begin(), elapsed_milliseconds.end());
    auto percentile = [&](double fraction) {
        const auto index = static_cast<std::size_t>(std::ceil(fraction * count)) - 1;
        return elapsed_milliseconds.at(index);
    };
    std::cout << "pipeline_ms samples=" << count << " warmup=" << warmup_count
              << " p50=" << percentile(0.50) << " p95=" << percentile(0.95)
              << " p99=" << percentile(0.99) << " max=" << elapsed_milliseconds.back()
              << " overruns_20ms=" << overruns << '\n';
}

}  // namespace

int main(int argument_count, char** argument_values) {
    try {
        const auto arguments = parseArguments(argument_count, argument_values);
        const supermimic::Bundle bundle(arguments.bundle);
        supermimic::PolicyRunner runner(bundle);
        std::cout << std::setprecision(8);
        bool parity_passed = true;
        if (!arguments.parity.empty()) {
            parity_passed = checkParity(arguments.parity, bundle, runner);
        }
        if (arguments.benchmark_count > 0) {
            benchmark(bundle, runner, arguments.benchmark_count);
        }
        return parity_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "supermimic_offline: " << error.what() << '\n';
        return 1;
    }
}
