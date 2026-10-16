#!/bin/bash
# Fix SystemC-AMS to link against SystemC 2.3.4

set -e

SYSTEMC_ROOT=/mnt/d/systemCProjects/systemCsrc/systemc-2.3.4-install
SYSTEMCAMS_SRC=/mnt/d/systemCProjects/systemCsrc/systemc-ams-2.3.4-src
SYSTEMCAMS_INSTALL=/mnt/d/systemCProjects/systemCsrc/systemc-ams-install

echo "=== Step 1: Clean up ==="
rm -rf $SYSTEMCAMS_INSTALL
rm -rf $SYSTEMCAMS_SRC/build
mkdir -p $SYSTEMCAMS_INSTALL

echo "=== Step 2: Patch CMakeLists.txt to use correct SystemC ==="
cd $SYSTEMCAMS_SRC

# Backup original
cp CMakeLists.txt CMakeLists.txt.bak

# Create patched version that properly links to SystemC
cat > CMakeLists.txt << 'EOF'
###############################################################################
#
#    Copyright 2015-2023
#    COSEDA Technologies GmbH
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###############################################################################

###############################################################################
#  CMakeLists.txe - simple cmake flow
#  Original Author: Paul Ehrlich COSEDA Technologies GmbH
###############################################################################

cmake_minimum_required(VERSION 3.1)

project(systemc_ams CXX C)

###############################################################################
# SystemC Configuration - CRITICAL FIX
###############################################################################

# Find SystemC properly
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SYSTEMC QUIET systemc)
endif()

# If not found via pkg-config, use environment variable or default path
if(NOT SYSTEMC_FOUND)
    if(DEFINED ENV{SYSTEMC_HOME})
        set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
    else()
        set(SYSTEMC_HOME "/mnt/d/systemCProjects/systemCsrc/systemc-2.3.4-install")
    endif()
    
    set(SYSTEMC_INCLUDE_DIRS ${SYSTEMC_HOME}/include)
    set(SYSTEMC_LIBRARY_DIRS ${SYSTEMC_HOME}/lib)
    set(SYSTEMC_LIBRARIES systemc)
    
    message(STATUS "Using SystemC from: ${SYSTEMC_HOME}")
endif()

# Include SystemC headers
include_directories(${SYSTEMC_INCLUDE_DIRS})

# Link directories for SystemC
link_directories(${SYSTEMC_LIBRARY_DIRS})

# Set RPATH to ensure runtime uses correct library
set(CMAKE_INSTALL_RPATH "${SYSTEMC_LIBRARY_DIRS}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

###############################################################################
# Build options
###############################################################################

option (ENABLE_PARALLEL_TRACING "Enable parallel tracing and thus add a pthread dependency." OFF)
option (DISABLE_REFERENCE_NODE_CLUSTERING "Disables clustering for refrence nodes - reference nodes ignored for clustering." OFF)
option (DISABLE_PERFORMANCE_STATISTICS "Disables performance data collection and removes dependency from high precision counter and chrono" OFF)

mark_as_advanced(
        ENABLE_PARALLEL_TRACING
        DISABLE_REFERENCE_NODE_CLUSTERING
        DISABLE_PERFORMANCE_STATISTICS)

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT ENABLE_PARALLEL_TRACING)
    add_compile_definitions(DISABLE_PARALLEL_TRACING)
endif(NOT ENABLE_PARALLEL_TRACING)  
if(DISABLE_REFERENCE_NODE_CLUSTERING)
    add_compile_definitions(DISABLE_REFERENCE_NODE_CLUSTERING)
endif(DISABLE_REFERENCE_NODE_CLUSTERING)    
if(DISABLE_PERFORMANCE_STATISTICS)
    add_compile_definitions(DISABLE_PERFORMANCE_STATISTICS)
endif(DISABLE_PERFORMANCE_STATISTICS)   



###############################################################################
# Configure status
###############################################################################

if (ENABLE_PARALLEL_TRACING)
  message ("ENABLE_PARALLEL_TRACING = ${ENABLE_PARALLEL_TRACING}")
else (ENABLE_PARALLEL_TRACING)
  message (STATUS "ENABLE_PARALLEL_TRACING = ${ENABLE_PARALLEL_TRACING}")
endif (ENABLE_PARALLEL_TRACING)

if (DISABLE_REFERENCE_NODE_CLUSTERING)
  message ("DISABLE_REFERENCE_NODE_CLUSTERING = ${DISABLE_REFERENCE_NODE_CLUSTERING}")
else (DISABLE_REFERENCE_NODE_CLUSTERING)
  message (STATUS "DISABLE_REFERENCE_NODE_CLUSTERING = ${DISABLE_REFERENCE_NODE_CLUSTERING}")
endif (DISABLE_REFERENCE_NODE_CLUSTERING)

if (DISABLE_PERFORMANCE_STATISTICS)
  message ("DISABLE_PERFORMANCE_STATISTICS = ${DISABLE_PERFORMANCE_STATISTICS}")
else (DISABLE_PERFORMANCE_STATISTICS)
  message (STATUS "DISABLE_PERFORMANCE_STATISTICS = ${DISABLE_PERFORMANCE_STATISTICS}")
endif (DISABLE_PERFORMANCE_STATISTICS)

macro(install_headers)
	string(REPLACE ${CMAKE_SOURCE_DIR}/src "" REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
	install(FILES ${ARGV} DESTINATION include/${REL_DIR})
endmacro(install_headers)

add_subdirectory(src)

install(
	FILES 
		AUTHORS
		ChangeLog
		COPYING
		INSTALL
		LICENSE
		NEWS
		NOTICE
		README
		RELEASENOTES
	DESTINATION .)
EOF

echo "=== Step 3: Configure with CMake ==="
export SYSTEMC_HOME=$SYSTEMC_ROOT
export LD_LIBRARY_PATH=$SYSTEMC_ROOT/lib:$LD_LIBRARY_PATH

mkdir -p build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$SYSTEMCAMS_INSTALL \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_VERBOSE_MAKEFILE=ON

echo "=== Step 4: Build ==="
make -j4

echo "=== Step 5: Install ==="
make install

echo "=== Step 6: Verify linking ==="
echo "Checking library dependencies:"
readelf -d $SYSTEMCAMS_INSTALL/liblinux64/libsystemc-ams.so | grep NEEDED | grep -E "systemc|sca"

echo ""
echo "Running ldd:"
ldd $SYSTEMCAMS_INSTALL/liblinux64/libsystemc-ams.so | grep systemc

echo ""
echo "=== Done! ==="
