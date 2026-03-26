cmake_minimum_required(VERSION 3.14)

# ============================================================================
# Platform-Adaptive Compiler Configuration
# ============================================================================
# This project supports dual-platform development:
#   - Linux: Uses GCC (default system compiler)
#   - macOS: Uses Clang (default system compiler)
#
# IMPORTANT: SystemC/SystemC-AMS libraries must be compiled with the SAME
# compiler as this project. Set SYSTEMC_HOME to the matching library path.
#
# Manual override:
#   cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
#   cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
# ============================================================================

if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        # Linux: prefer GCC
        find_program(DEFAULT_C_COMPILER gcc)
        find_program(DEFAULT_CXX_COMPILER g++)
        set(PLATFORM_COMPILER_NAME "GCC")
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        # macOS: prefer Clang (system default, stable SDK paths)
        find_program(DEFAULT_C_COMPILER clang)
        find_program(DEFAULT_CXX_COMPILER clang++)
        set(PLATFORM_COMPILER_NAME "Clang")
    endif()
    
    if(DEFAULT_C_COMPILER AND DEFAULT_CXX_COMPILER)
        set(CMAKE_C_COMPILER ${DEFAULT_C_COMPILER} CACHE FILEPATH "C compiler")
        set(CMAKE_CXX_COMPILER ${DEFAULT_CXX_COMPILER} CACHE FILEPATH "C++ compiler")
        message(STATUS "Platform: ${CMAKE_HOST_SYSTEM_NAME} -> Using ${PLATFORM_COMPILER_NAME}")
    endif()
endif()

project(serdes)

# ============================================================================
# macOS SDK Configuration (fix Clang header search paths)
# ============================================================================
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Find macOS SDK path
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MACOS_SDK_PATH)
        set(CMAKE_OSX_SYSROOT ${MACOS_SDK_PATH})
        # Add C++ standard library headers explicitly
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH} -I${MACOS_SDK_PATH}/usr/include/c++/v1")
        message(STATUS "macOS SDK: ${MACOS_SDK_PATH}")
    endif()
endif()

# Set C++ standard globally - must match SystemC library
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ============================================================================
# Find SystemC and SystemC-AMS
# ============================================================================
# Usage:
#   1. Set environment variables:
#      export SYSTEMC_HOME=/path/to/systemc-2.3.4
#      export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4
#
#   2. Or use CMake options:
#      cmake -DSYSTEMC_HOME=/path/to/systemc -DSYSTEMC_AMS_HOME=/path/to/systemc-ams ..
#
#   3. Or install to standard location:
#      - /usr/local/systemc-2.3.4
#      - /opt/systemc-2.3.4
#      - macOS Homebrew: /opt/homebrew/opt/systemc
# ============================================================================

find_package(SystemC REQUIRED)

if(NOT SystemC_FOUND OR NOT SystemCAMS_FOUND)
    message(FATAL_ERROR "
SystemC or SystemC-AMS not found!

Please install SystemC/SystemC-AMS and set one of:

1. Environment variables (recommended):
   export SYSTEMC_HOME=/path/to/systemc-2.3.4
   export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4

2. CMake options:
   cmake -DSYSTEMC_HOME=/path/to/systemc -DSYSTEMC_AMS_HOME=/path/to/systemc-ams ..

3. Install to standard location:
   - /usr/local/systemc-2.3.4
   - /opt/systemc-2.3.4
   - macOS Homebrew: /opt/homebrew/opt/systemc

See README.md for installation instructions.
")
endif()

# ============================================================================
# Include directories
# ============================================================================
include_directories(
    ${SystemC_INCLUDE_DIRS}
    ${SystemCAMS_INCLUDE_DIRS}
    include
    include/ams
    include/common
    include/de
    tb/top
    tb/periphery
)

# ============================================================================
# Source files
# ============================================================================
file(GLOB_RECURSE AMS_SOURCES src/ams/*.cpp)
file(GLOB_RECURSE DE_SOURCES src/de/*.cpp)

# ============================================================================
# Create library
# ============================================================================
add_library(serdes_lib STATIC ${AMS_SOURCES} ${DE_SOURCES})

# Link SystemC libraries
target_link_libraries(serdes_lib
    SystemC::systemc
    SystemC::systemc-ams
)

# ============================================================================
# Add subdirectories
# ============================================================================
add_subdirectory(tb)

# Optionally build tests if BUILD_TESTING is enabled
option(BUILD_TESTING "Build unit tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS serdes_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ============================================================================
# Print summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================================================")
message(STATUS "SerDes Project Configuration Summary")
message(STATUS "========================================================================")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "SystemC_HOME:      ${SYSTEMC_HOME}")
message(STATUS "SystemC-AMS_HOME:  ${SYSTEMC_AMS_HOME}")
message(STATUS "Build Tests:       ${BUILD_TESTING}")
message(STATUS "========================================================================")
message(STATUS "")
