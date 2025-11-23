# ============================================================================
# SerDes SystemC-AMS Project Makefile
# ============================================================================

# Compiler settings
CXX := clang++
CXXFLAGS := -std=c++14 -Wall -Wextra -O2

# SystemC paths
SYSTEMC_HOME ?= /usr/local/systemc-2.3.4
SYSTEMC_AMS_HOME ?= /usr/local/systemc-ams-2.3.4

# Check if SystemC is installed
ifeq ($(wildcard $(SYSTEMC_HOME)),)
    $(error SystemC not found at $(SYSTEMC_HOME). Please set SYSTEMC_HOME)
endif

ifeq ($(wildcard $(SYSTEMC_AMS_HOME)),)
    $(error SystemC-AMS not found at $(SYSTEMC_AMS_HOME). Please set SYSTEMC_AMS_HOME)
endif

# Include directories
INCLUDES := -Iinclude \
            -I$(SYSTEMC_HOME)/include \
            -I$(SYSTEMC_AMS_HOME)/include

# Library directories and libraries
LDFLAGS := -L$(SYSTEMC_HOME)/lib-macosx64 \
           -L$(SYSTEMC_HOME)/lib \
           -L$(SYSTEMC_AMS_HOME)/lib-macosx64 \
           -L$(SYSTEMC_AMS_HOME)/lib

LIBS := -lsystemc-ams -lsystemc -lm

# Source files
AMS_SOURCES := $(wildcard src/ams/*.cpp)
DE_SOURCES := $(wildcard src/de/*.cpp)
SYSTEM_SOURCES := $(wildcard src/system/*.cpp)
ALL_SOURCES := $(AMS_SOURCES) $(DE_SOURCES) $(SYSTEM_SOURCES)

# Object files
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
OBJECTS := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SOURCES))

# Library
LIB_DIR := $(BUILD_DIR)/lib
STATIC_LIB := $(LIB_DIR)/libserdes.a

# Testbench sources and executables
TB_SOURCES := $(wildcard tb/*.cpp)
TB_EXECUTABLES := $(patsubst tb/%.cpp,$(BUILD_DIR)/bin/%,$(TB_SOURCES))

# Test sources and executables
TEST_SOURCES := $(wildcard tests/unit/*.cpp tests/integration/*.cpp)
TEST_EXECUTABLE := $(BUILD_DIR)/bin/run_tests

# ============================================================================
# Targets
# ============================================================================

.PHONY: all lib tb tests clean help run

# Default target
all: lib tb

# Help target
help:
	@echo "SerDes SystemC-AMS Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build library and testbenches (default)"
	@echo "  lib      - Build SerDes static library"
	@echo "  tb       - Build testbenches"
	@echo "  tests    - Build and run unit tests"
	@echo "  clean    - Remove all build artifacts"
	@echo "  run      - Run simple testbench"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  SYSTEMC_HOME     - Path to SystemC installation (current: $(SYSTEMC_HOME))"
	@echo "  SYSTEMC_AMS_HOME - Path to SystemC-AMS installation (current: $(SYSTEMC_AMS_HOME))"
	@echo ""

# Build static library
lib: $(STATIC_LIB)

$(STATIC_LIB): $(OBJECTS) | $(LIB_DIR)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Compile source files
$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build testbenches
tb: $(TB_EXECUTABLES)

$(BUILD_DIR)/bin/%: tb/%.cpp $(STATIC_LIB) | $(BUILD_DIR)/bin
	@echo "Building testbench: $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lserdes $(LDFLAGS) $(LIBS)

# Build tests
tests: $(TEST_EXECUTABLE)
	@echo "Running tests..."
	$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): tests/test_main.cpp $(TEST_SOURCES) $(STATIC_LIB) | $(BUILD_DIR)/bin
	@echo "Building tests: $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(TEST_SOURCES) -o $@ -L$(LIB_DIR) -lserdes $(LDFLAGS) $(LIBS)

# Run simple testbench
run: $(BUILD_DIR)/bin/simple_link_tb
	@echo "Running simple link testbench..."
	$<

# Create directories
$(BUILD_DIR) $(OBJ_DIR) $(LIB_DIR) $(BUILD_DIR)/bin:
	@mkdir -p $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# ============================================================================
# Debug information
# ============================================================================

.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  CXX          : $(CXX)"
	@echo "  CXXFLAGS     : $(CXXFLAGS)"
	@echo "  SYSTEMC_HOME : $(SYSTEMC_HOME)"
	@echo "  SYSTEMC_AMS_HOME : $(SYSTEMC_AMS_HOME)"
	@echo ""
	@echo "Sources:"
	@echo "  AMS sources  : $(words $(AMS_SOURCES)) files"
	@echo "  DE sources   : $(words $(DE_SOURCES)) files"
	@echo "  System sources : $(words $(SYSTEM_SOURCES)) files"
	@echo "  Total sources: $(words $(ALL_SOURCES)) files"
	@echo ""
	@echo "Testbenches:"
	@echo "  TB sources   : $(words $(TB_SOURCES)) files"
	@echo ""
