cmake_minimum_required(VERSION 3.18)
project(supermimic_deployment LANGUAGES CXX)

# Keep existing build commands and executable locations stable.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
include(CTest)
add_subdirectory(native)
