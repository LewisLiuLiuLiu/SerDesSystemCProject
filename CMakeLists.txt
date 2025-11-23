cmake_minimum_required(VERSION 3.15)
project(SerDes VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# Find SystemC Library
# ============================================================================
if(DEFINED SYSTEMC_HOME)
    message(STATUS "Using SYSTEMC_HOME: ${SYSTEMC_HOME}")
elseif(DEFINED ENV{SYSTEMC_HOME})
    set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
    message(STATUS "Using SYSTEMC_HOME from environment: ${SYSTEMC_HOME}")
else()
    # Try common installation paths
    set(SYSTEMC_SEARCH_PATHS
        /usr/local/systemc-2.3.4
        /usr/local/systemc
        /opt/systemc-2.3.4
        /opt/systemc
    )
    foreach(path ${SYSTEMC_SEARCH_PATHS})
        if(EXISTS ${path})
            set(SYSTEMC_HOME ${path})
            message(STATUS "Found SystemC at: ${SYSTEMC_HOME}")
            break()
        endif()
    endforeach()
endif()

if(NOT SYSTEMC_HOME)
    message(FATAL_ERROR "SystemC not found. Please set SYSTEMC_HOME")
endif()

set(SYSTEMC_INCLUDE_DIRS ${SYSTEMC_HOME}/include)
set(SYSTEMC_LIBRARY_DIRS ${SYSTEMC_HOME}/lib-macosx64 ${SYSTEMC_HOME}/lib)

find_library(SYSTEMC_LIBRARY
    NAMES systemc
    PATHS ${SYSTEMC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
)

if(NOT SYSTEMC_LIBRARY)
    message(FATAL_ERROR "SystemC library not found in ${SYSTEMC_LIBRARY_DIRS}")
endif()

message(STATUS "SystemC include: ${SYSTEMC_INCLUDE_DIRS}")
message(STATUS "SystemC library: ${SYSTEMC_LIBRARY}")

# ============================================================================
# Find SystemC-AMS Library
# ============================================================================
if(DEFINED SYSTEMC_AMS_HOME)
    message(STATUS "Using SYSTEMC_AMS_HOME: ${SYSTEMC_AMS_HOME}")
elseif(DEFINED ENV{SYSTEMC_AMS_HOME})
    set(SYSTEMC_AMS_HOME $ENV{SYSTEMC_AMS_HOME})
    message(STATUS "Using SYSTEMC_AMS_HOME from environment: ${SYSTEMC_AMS_HOME}")
else()
    # Try common installation paths
    set(SYSTEMC_AMS_SEARCH_PATHS
        /usr/local/systemc-ams-2.3.4
        /usr/local/systemc-ams
        /opt/systemc-ams-2.3.4
        /opt/systemc-ams
    )
    foreach(path ${SYSTEMC_AMS_SEARCH_PATHS})
        if(EXISTS ${path})
            set(SYSTEMC_AMS_HOME ${path})
            message(STATUS "Found SystemC-AMS at: ${SYSTEMC_AMS_HOME}")
            break()
        endif()
    endforeach()
endif()

if(NOT SYSTEMC_AMS_HOME)
    message(FATAL_ERROR "SystemC-AMS not found. Please set SYSTEMC_AMS_HOME")
endif()

set(SYSTEMC_AMS_INCLUDE_DIRS ${SYSTEMC_AMS_HOME}/include)
set(SYSTEMC_AMS_LIBRARY_DIRS ${SYSTEMC_AMS_HOME}/lib-macosx64 ${SYSTEMC_AMS_HOME}/lib)

find_library(SYSTEMC_AMS_LIBRARY
    NAMES systemc-ams
    PATHS ${SYSTEMC_AMS_LIBRARY_DIRS}
    NO_DEFAULT_PATH
)

if(NOT SYSTEMC_AMS_LIBRARY)
    message(FATAL_ERROR "SystemC-AMS library not found in ${SYSTEMC_AMS_LIBRARY_DIRS}")
endif()

message(STATUS "SystemC-AMS include: ${SYSTEMC_AMS_INCLUDE_DIRS}")
message(STATUS "SystemC-AMS library: ${SYSTEMC_AMS_LIBRARY}")

# ============================================================================
# Third-party Libraries
# ============================================================================

# JSON library (header-only)
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# YAML library
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found, will fetch from source")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# GoogleTest (for testing)
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ============================================================================
# Include directories
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${SYSTEMC_INCLUDE_DIRS}
    ${SYSTEMC_AMS_INCLUDE_DIRS}
)

# ============================================================================
# Source files
# ============================================================================
file(GLOB_RECURSE AMS_SOURCES "src/ams/*.cpp")
file(GLOB_RECURSE DE_SOURCES "src/de/*.cpp")
file(GLOB_RECURSE SYSTEM_SOURCES "src/system/*.cpp")

set(ALL_SOURCES
    ${AMS_SOURCES}
    ${DE_SOURCES}
    ${SYSTEM_SOURCES}
)

# ============================================================================
# SerDes static library
# ============================================================================
add_library(serdes_lib STATIC ${ALL_SOURCES})
target_link_libraries(serdes_lib
    ${SYSTEMC_AMS_LIBRARY}
    ${SYSTEMC_LIBRARY}
    nlohmann_json::nlohmann_json
    yaml-cpp
)

# ============================================================================
# Testbenches
# ============================================================================
add_subdirectory(tb)

# ============================================================================
# Tests
# ============================================================================
if(BUILD_TESTING)
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
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  SystemC: ${SYSTEMC_LIBRARY}")
message(STATUS "  SystemC-AMS: ${SYSTEMC_AMS_LIBRARY}")
message(STATUS "  Build testing: ${BUILD_TESTING}")
message(STATUS "")
