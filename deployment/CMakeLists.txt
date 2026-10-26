cmake_minimum_required(VERSION 3.18)
project(supermimic_deployment LANGUAGES CXX)

option(SUPERMIMIC_WITH_ORT "Build CPU ONNX Runtime policy inference" ON)
option(SUPERMIMIC_WITH_SDK2 "Build the Unitree SDK2 robot executable" OFF)
set(ORT_ROOT "" CACHE PATH "ONNX Runtime installation prefix")
set(SDK2_ROOT "" CACHE PATH "Unitree SDK2 installation or source prefix")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.6 REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
# Extracted Debian development packages keep generated OpenSSL headers in a
# multiarch include directory that is not on the compiler's default search path.
find_path(SUPERMIMIC_OPENSSL_CONFIG_INCLUDE_DIR openssl/opensslconf.h
          HINTS "${OPENSSL_INCLUDE_DIR}" PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}")
if(SUPERMIMIC_OPENSSL_CONFIG_INCLUDE_DIR)
    set_property(TARGET OpenSSL::Crypto APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 "${SUPERMIMIC_OPENSSL_CONFIG_INCLUDE_DIR}")
endif()

add_library(supermimic_runtime src/runtime.cpp)
target_compile_features(supermimic_runtime PUBLIC cxx_std_17)
target_include_directories(supermimic_runtime PUBLIC include)
target_link_libraries(supermimic_runtime PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json
                      PRIVATE OpenSSL::Crypto)
target_compile_options(supermimic_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(SUPERMIMIC_WITH_ORT)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS "${ORT_ROOT}" ENV ORT_ROOT
              PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
                 HINTS "${ORT_ROOT}" ENV ORT_ROOT PATH_SUFFIXES lib lib64)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime CPU not found. Set ORT_ROOT, or use -DSUPERMIMIC_WITH_ORT=OFF for core-only tests.")
    endif()
    target_compile_definitions(supermimic_runtime PRIVATE SUPERMIMIC_WITH_ORT=1)
    target_include_directories(supermimic_runtime SYSTEM PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(supermimic_runtime PRIVATE "${ONNXRUNTIME_LIBRARY}")
endif()

add_executable(supermimic_offline src/offline.cpp)
target_link_libraries(supermimic_offline PRIVATE supermimic_runtime)

if(SUPERMIMIC_WITH_SDK2 AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/sdk2.cpp")
    message(WARNING "Skipping supermimic_g1: src/sdk2.cpp is not present yet. Reconfigure after adding it.")
endif()

if(SUPERMIMIC_WITH_SDK2 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/sdk2.cpp")
    if(NOT SUPERMIMIC_WITH_ORT)
        message(FATAL_ERROR "SUPERMIMIC_WITH_SDK2 requires SUPERMIMIC_WITH_ORT")
    endif()
    find_package(Threads REQUIRED)
    # The SDK's package exports unitree_sdk2; source installations can also be used.
    find_package(unitree_sdk2 QUIET CONFIG HINTS "${SDK2_ROOT}" "${SDK2_ROOT}/lib/cmake/unitree_sdk2")
    add_executable(supermimic_g1 src/sdk2.cpp)
    target_link_libraries(supermimic_g1 PRIVATE supermimic_runtime Threads::Threads)
    if(TARGET unitree_sdk2::unitree_sdk2)
        target_link_libraries(supermimic_g1 PRIVATE unitree_sdk2::unitree_sdk2)
    elseif(TARGET unitree_sdk2)
        target_link_libraries(supermimic_g1 PRIVATE unitree_sdk2)
    else()
        find_path(UNITREE_SDK2_INCLUDE_DIR unitree/robot/channel/channel_factory.hpp
                  HINTS "${SDK2_ROOT}" PATH_SUFFIXES include)
        find_library(UNITREE_SDK2_LIBRARY NAMES unitree_sdk2
                     HINTS "${SDK2_ROOT}" PATH_SUFFIXES lib "lib/${CMAKE_SYSTEM_PROCESSOR}")
        find_path(CYCLONEDDS_INCLUDE_DIR dds/dds.h
                  HINTS "${SDK2_ROOT}/thirdparty" PATH_SUFFIXES include include/ddsc)
        find_path(CYCLONEDDS_CXX_INCLUDE_DIR dds/dds.hpp
                  HINTS "${SDK2_ROOT}/thirdparty" PATH_SUFFIXES include/ddscxx include)
        find_library(CYCLONEDDS_LIBRARY NAMES ddsc
                     HINTS "${SDK2_ROOT}/thirdparty" PATH_SUFFIXES lib "lib/${CMAKE_SYSTEM_PROCESSOR}")
        find_library(CYCLONEDDS_CXX_LIBRARY NAMES ddscxx
                     HINTS "${SDK2_ROOT}/thirdparty" PATH_SUFFIXES lib "lib/${CMAKE_SYSTEM_PROCESSOR}")
        if(NOT UNITREE_SDK2_INCLUDE_DIR OR NOT UNITREE_SDK2_LIBRARY OR
           NOT CYCLONEDDS_INCLUDE_DIR OR NOT CYCLONEDDS_CXX_INCLUDE_DIR OR
           NOT CYCLONEDDS_LIBRARY OR NOT CYCLONEDDS_CXX_LIBRARY)
            message(FATAL_ERROR "SDK2/CycloneDDS not found. Set SDK2_ROOT to a complete SDK2 installation.")
        endif()
        target_include_directories(supermimic_g1 SYSTEM PRIVATE "${UNITREE_SDK2_INCLUDE_DIR}"
                                   "${CYCLONEDDS_INCLUDE_DIR}" "${CYCLONEDDS_CXX_INCLUDE_DIR}")
        target_link_libraries(supermimic_g1 PRIVATE "${UNITREE_SDK2_LIBRARY}"
                              "${CYCLONEDDS_CXX_LIBRARY}" "${CYCLONEDDS_LIBRARY}")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_runtime tests/test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE supermimic_runtime OpenSSL::Crypto)
    add_test(NAME supermimic_runtime_core COMMAND test_runtime)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_safety.cpp")
        add_executable(test_safety tests/test_safety.cpp)
        target_link_libraries(test_safety PRIVATE supermimic_runtime)
        add_test(NAME supermimic_safety COMMAND test_safety)
    endif()
endif()
