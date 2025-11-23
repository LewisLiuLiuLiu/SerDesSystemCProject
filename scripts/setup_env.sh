#!/bin/bash
# ============================================================================
# Environment Setup Script for SerDes SystemC-AMS Project
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a path exists
check_path() {
    if [ -d "$1" ]; then
        print_info "Found: $1"
        return 0
    else
        print_warn "Not found: $1"
        return 1
    fi
}

# ============================================================================
# Main Setup
# ============================================================================

print_info "SerDes SystemC-AMS Environment Setup"
echo ""

# Check for SystemC
print_info "Checking for SystemC installation..."
if [ -z "$SYSTEMC_HOME" ]; then
    # Try to find SystemC in common locations
    for path in /usr/local/systemc-2.3.4 /usr/local/systemc /opt/systemc-2.3.4 /opt/systemc; do
        if check_path "$path"; then
            export SYSTEMC_HOME="$path"
            print_info "Setting SYSTEMC_HOME=$SYSTEMC_HOME"
            break
        fi
    done
    
    if [ -z "$SYSTEMC_HOME" ]; then
        print_error "SystemC not found. Please set SYSTEMC_HOME manually."
        exit 1
    fi
else
    print_info "SYSTEMC_HOME is already set: $SYSTEMC_HOME"
    check_path "$SYSTEMC_HOME"
fi

# Check for SystemC-AMS
print_info "Checking for SystemC-AMS installation..."
if [ -z "$SYSTEMC_AMS_HOME" ]; then
    # Try to find SystemC-AMS in common locations
    for path in /usr/local/systemc-ams-2.3.4 /usr/local/systemc-ams /opt/systemc-ams-2.3.4 /opt/systemc-ams; do
        if check_path "$path"; then
            export SYSTEMC_AMS_HOME="$path"
            print_info "Setting SYSTEMC_AMS_HOME=$SYSTEMC_AMS_HOME"
            break
        fi
    done
    
    if [ -z "$SYSTEMC_AMS_HOME" ]; then
        print_error "SystemC-AMS not found. Please set SYSTEMC_AMS_HOME manually."
        exit 1
    fi
else
    print_info "SYSTEMC_AMS_HOME is already set: $SYSTEMC_AMS_HOME"
    check_path "$SYSTEMC_AMS_HOME"
fi

# Check for required tools
print_info "Checking for required tools..."

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    print_info "Found: $CMAKE_VERSION"
else
    print_warn "CMake not found"
fi

# Check make
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -n1)
    print_info "Found: $MAKE_VERSION"
else
    print_warn "Make not found"
fi

# Check C++ compiler
if command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    print_info "Found: $CLANG_VERSION"
elif command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    print_info "Found: $GCC_VERSION"
else
    print_error "No C++ compiler found (clang++ or g++)"
fi

# Check Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    print_info "Found: $GIT_VERSION"
else
    print_warn "Git not found"
fi

echo ""
print_info "Environment setup complete!"
print_info "You can now build the project using:"
print_info "  cmake -B build -S . && cmake --build build"
print_info "Or:"
print_info "  make"
echo ""

# Print environment variables for sourcing
echo "# To use these settings, run:"
echo "# source $0"
echo ""
echo "export SYSTEMC_HOME=$SYSTEMC_HOME"
echo "export SYSTEMC_AMS_HOME=$SYSTEMC_AMS_HOME"
