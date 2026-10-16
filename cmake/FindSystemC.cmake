# FindSystemC.cmake
# Find SystemC and SystemC-AMS libraries
#
# This module will find SystemC and SystemC-AMS using the following methods:
# 1. Environment variables: SYSTEMC_HOME, SYSTEMC_AMS_HOME
# 2. CMake options: -DSYSTEMC_HOME=path -DSYSTEMC_AMS_HOME=path
# 3. Common installation paths
#
# Variables set by this module:
#   SystemC_FOUND         - True if SystemC found
#   SystemC_INCLUDE_DIRS  - SystemC include directories
#   SystemC_LIBRARIES     - SystemC libraries to link
#   SystemCAMS_FOUND      - True if SystemC-AMS found
#   SystemCAMS_INCLUDE_DIRS - SystemC-AMS include directories
#   SystemCAMS_LIBRARIES  - SystemC-AMS libraries to link
#
# Imported targets:
#   SystemC::systemc      - Imported target for SystemC
#   SystemC::systemc-ams  - Imported target for SystemC-AMS

include(FindPackageHandleStandardArgs)

# ============================================================================
# Helper function to find SystemC in common locations
# ============================================================================
function(_find_systemc_base VAR_NAME)
    # Check environment variable first
    if(DEFINED ENV{${VAR_NAME}} AND EXISTS "$ENV{${VAR_NAME}}")
        set(${VAR_NAME} "$ENV{${VAR_NAME}}" PARENT_SCOPE)
        return()
    endif()
    
    # Check CMake variable
    if(DEFINED ${VAR_NAME} AND EXISTS "${${VAR_NAME}}")
        return()
    endif()
    
    # Common installation paths to search
    set(SEARCH_PATHS
        # macOS Homebrew
        /opt/homebrew/opt/systemc
        /usr/local/opt/systemc
        # Linux standard paths
        /usr/local/systemc-2.3.4
        /usr/local/systemc
        /opt/systemc-2.3.4
        /opt/systemc
        /usr/local
        # Windows (typical locations)
        "C:/systemc-2.3.4"
        "C:/Program Files/SystemC"
        # User home directory
        "$ENV{HOME}/systemc-2.3.4"
        "$ENV{HOME}/systemc"
        "$ENV{HOME}/.local/systemc"
    )
    
    foreach(PATH ${SEARCH_PATHS})
        if(EXISTS "${PATH}/include/systemc.h")
            set(${VAR_NAME} "${PATH}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# ============================================================================
# Find SystemC
# ============================================================================
_find_systemc_base(SYSTEMC_HOME)

if(SYSTEMC_HOME)
    message(STATUS "Looking for SystemC in: ${SYSTEMC_HOME}")
    
    # Find include directory
    find_path(SystemC_INCLUDE_DIR
        NAMES systemc.h
        PATHS ${SYSTEMC_HOME}/include
        NO_DEFAULT_PATH
    )
    
    # Find library (support multiple naming conventions)
    find_library(SystemC_LIBRARY
        NAMES systemc
        PATHS ${SYSTEMC_HOME}/lib
                ${SYSTEMC_HOME}/lib-macosx64
                ${SYSTEMC_HOME}/lib-linux64
                ${SYSTEMC_HOME}/lib64
        NO_DEFAULT_PATH
    )
endif()

# ============================================================================
# Find SystemC-AMS
# ============================================================================
# Use SYSTEMC_AMS_HOME if set, otherwise fall back to SYSTEMC_HOME
if(DEFINED ENV{SYSTEMC_AMS_HOME} AND EXISTS "$ENV{SYSTEMC_AMS_HOME}")
    set(SYSTEMC_AMS_HOME "$ENV{SYSTEMC_AMS_HOME}")
elseif(DEFINED SYSTEMC_AMS_HOME AND EXISTS "${SYSTEMC_AMS_HOME}")
    # Already set via CMake option
else()
    set(SYSTEMC_AMS_HOME ${SYSTEMC_HOME})
endif()

if(SYSTEMC_AMS_HOME)
    message(STATUS "Looking for SystemC-AMS in: ${SYSTEMC_AMS_HOME}")
    
    # Find include directory (SystemC-AMS uses systemc-ams.h or scams/)
    find_path(SystemCAMS_INCLUDE_DIR
        NAMES systemc-ams.h
        PATHS ${SYSTEMC_AMS_HOME}/include
        NO_DEFAULT_PATH
    )
    
    # If not found, try looking for scams subdirectory
    if(NOT SystemCAMS_INCLUDE_DIR)
        find_path(SystemCAMS_INCLUDE_DIR
            NAMES scams/systemc-ams.h
            PATHS ${SYSTEMC_AMS_HOME}/include
            NO_DEFAULT_PATH
        )
    endif()
    
    # Find library
    find_library(SystemCAMS_LIBRARY
        NAMES systemc-ams
        PATHS ${SYSTEMC_AMS_HOME}/lib
                ${SYSTEMC_AMS_HOME}/lib-macosx64
                ${SYSTEMC_AMS_HOME}/lib-linux64
                ${SYSTEMC_AMS_HOME}/lib64
        NO_DEFAULT_PATH
    )
endif()

# ============================================================================
# Handle standard arguments
# ============================================================================
find_package_handle_standard_args(SystemC
    REQUIRED_VARS 
        SystemC_LIBRARY 
        SystemC_INCLUDE_DIR
    FAIL_MESSAGE "SystemC not found. Please set SYSTEMC_HOME environment variable or use -DSYSTEMC_HOME=path"
)

find_package_handle_standard_args(SystemCAMS
    REQUIRED_VARS 
        SystemCAMS_LIBRARY 
        SystemCAMS_INCLUDE_DIR
    FAIL_MESSAGE "SystemC-AMS not found. Please set SYSTEMC_AMS_HOME environment variable or use -DSYSTEMC_AMS_HOME=path"
)

# ============================================================================
# Create imported targets
# ============================================================================
if(SystemC_FOUND AND NOT TARGET SystemC::systemc)
    add_library(SystemC::systemc UNKNOWN IMPORTED)
    set_target_properties(SystemC::systemc PROPERTIES
        IMPORTED_LOCATION "${SystemC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SystemC_INCLUDE_DIR}"
    )
endif()

if(SystemCAMS_FOUND AND NOT TARGET SystemC::systemc-ams)
    add_library(SystemC::systemc-ams UNKNOWN IMPORTED)
    set_target_properties(SystemC::systemc-ams PROPERTIES
        IMPORTED_LOCATION "${SystemCAMS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SystemCAMS_INCLUDE_DIR}"
    )
endif()

# ============================================================================
# Set legacy variables for backward compatibility
# ============================================================================
if(SystemC_FOUND)
    set(SystemC_INCLUDE_DIRS ${SystemC_INCLUDE_DIR})
    set(SystemC_LIBRARIES ${SystemC_LIBRARY})
    set(SYSTEMC_HOME ${SYSTEMC_HOME} CACHE PATH "SystemC installation path" FORCE)
endif()

if(SystemCAMS_FOUND)
    set(SystemCAMS_INCLUDE_DIRS ${SystemCAMS_INCLUDE_DIR})
    set(SystemCAMS_LIBRARIES ${SystemCAMS_LIBRARY})
    set(SYSTEMC_AMS_HOME ${SYSTEMC_AMS_HOME} CACHE PATH "SystemC-AMS installation path" FORCE)
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
if(SystemC_FOUND AND SystemCAMS_FOUND)
    message(STATUS "")
    message(STATUS "SystemC Configuration:")
    message(STATUS "  SystemC_HOME:       ${SYSTEMC_HOME}")
    message(STATUS "  SystemC_INCLUDE:    ${SystemC_INCLUDE_DIR}")
    message(STATUS "  SystemC_LIBRARY:    ${SystemC_LIBRARY}")
    message(STATUS "  SystemC-AMS_HOME:   ${SYSTEMC_AMS_HOME}")
    message(STATUS "  SystemC-AMS_INCLUDE: ${SystemCAMS_INCLUDE_DIR}")
    message(STATUS "  SystemC-AMS_LIBRARY: ${SystemCAMS_LIBRARY}")
    message(STATUS "")
endif()
