#!/bin/bash

# run_wavegen_tests.sh
# Script to run all Wave Generation module unit tests independently
#
# Each test is run in a separate process to avoid SystemC simulator state conflicts
# Special handling for paired tests (seed comparison and reproducibility)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Standard test executables (compatible with bash 3.2)
TEST_NAMES=(
    "test_wave_gen_basic_functionality"
    "test_wave_gen_code_balance"
    "test_wave_gen_debug_lfsr"
    "test_wave_gen_debug_time"
    "test_wave_gen_invalid_pulse_width"
    "test_wave_gen_invalid_sample_rate"
    "test_wave_gen_jitter_config"
    "test_wave_gen_long_stability"
    "test_wave_gen_mean_value"
    "test_wave_gen_nrz_level"
    "test_wave_gen_prbs15"
    "test_wave_gen_prbs23"
    "test_wave_gen_prbs31"
    "test_wave_gen_prbs7"
    "test_wave_gen_prbs9"
    "test_wave_gen_prbs_mode"
    "test_wave_gen_pulse_basic"
    "test_wave_gen_pulse_timing"
)

TEST_DESCS=(
    "Basic Functionality Tests"
    "Code Balance Tests"
    "Debug LFSR Tests"
    "Debug Time Tests"
    "Invalid Pulse Width Tests"
    "Invalid Sample Rate Tests"
    "Jitter Configuration Tests"
    "Long Stability Tests"
    "Mean Value Tests"
    "NRZ Level Tests"
    "PRBS15 Tests"
    "PRBS23 Tests"
    "PRBS31 Tests"
    "PRBS7 Tests"
    "PRBS9 Tests"
    "PRBS Mode Tests"
    "Pulse Basic Tests"
    "Pulse Timing Tests"
)

# Find test executables directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build/tests"

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo -e "${RED}Error: Build directory not found: $BUILD_DIR${NC}"
    echo -e "${YELLOW}Please build the tests first:${NC}"
    echo "  cd ${SCRIPT_DIR}/../build/tests && make"
    exit 1
fi

# Change to build directory
cd "$BUILD_DIR" || exit 1

# Statistics
total_tests=0
passed_tests=0
failed_tests=0
declare -a failed_test_names

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Wave Generation Module Unit Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run a single test executable
run_single_test() {
    local test_exec="$1"
    local desc="$2"
    
    # Check if executable exists
    if [[ ! -x "./$test_exec" ]]; then
        echo -e "${YELLOW}Warning: $test_exec not found, skipping...${NC}"
        return 1
    fi
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Suite: ${desc}${NC}"
    echo -e "${BLUE}Executable: ${test_exec}${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Get list of individual test cases
    test_list=$(./"$test_exec" --gtest_list_tests 2>&1 | grep -v "^[[:space:]]*$")
    
    # Parse test suite and test case names
    local current_suite=""
    local suite_passed=0
    local suite_failed=0
    
    while IFS= read -r line; do
        # Check if this is a test suite line (ends with .)
        if [[ "$line" =~ ^([A-Za-z0-9_]+)\.$ ]]; then
            current_suite="${BASH_REMATCH[1]}"
        # Check if this is a test case line (starts with spaces)
        elif [[ "$line" =~ ^[[:space:]]+([A-Za-z0-9_]+) ]]; then
            test_case="${BASH_REMATCH[1]}"
            test_filter="${current_suite}.${test_case}"
            
            echo -e "${BLUE}Running: ${test_filter}${NC}"
            
            # Run individual test in separate process
            test_output=$(./"$test_exec" --gtest_filter="${test_filter}" 2>&1)
            test_exit=$?
            
            # Count this test
            total_tests=$((total_tests + 1))
            
            # Check result
            if [[ $test_exit -eq 0 ]]; then
                passed_tests=$((passed_tests + 1))
                suite_passed=$((suite_passed + 1))
                echo -e "${GREEN}  ✓ ${test_filter} PASSED${NC}"
            else
                failed_tests=$((failed_tests + 1))
                suite_failed=$((suite_failed + 1))
                failed_test_names+=("${test_filter}")
                echo -e "${RED}  ✗ ${test_filter} FAILED${NC}"
                # Show error details
                echo "$test_output" | grep -A 5 "\[  FAILED  \]\|Failure\|Error"
            fi
        fi
    done <<< "$test_list"
    
    echo ""
    echo -e "${BLUE}Suite Summary: ${GREEN}${suite_passed} passed${NC}, ${RED}${suite_failed} failed${NC}"
    echo ""
}

# Run standard tests
for i in "${!TEST_NAMES[@]}"; do
    run_single_test "${TEST_NAMES[$i]}" "${TEST_DESCS[$i]}"
done

# ============================================================================
# Special Paired Tests: Different Seeds (must produce DIFFERENT sequences)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paired Test: Different Seeds Comparison${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ -x "./test_wave_gen_seed_run1" ]] && [[ -x "./test_wave_gen_seed_run2" ]]; then
    total_tests=$((total_tests + 1))
    
    echo -e "${BLUE}Running: test_wave_gen_seed_run1${NC}"
    seed_output1=$(./test_wave_gen_seed_run1 2>&1)
    seed_exit1=$?
    
    echo -e "${BLUE}Running: test_wave_gen_seed_run2${NC}"
    seed_output2=$(./test_wave_gen_seed_run2 2>&1)
    seed_exit2=$?
    
    if [[ $seed_exit1 -ne 0 ]] || [[ $seed_exit2 -ne 0 ]]; then
        failed_tests=$((failed_tests + 1))
        failed_test_names+=("DifferentSeeds.Comparison")
        echo -e "${RED}  ✗ DifferentSeeds.Comparison FAILED - Test execution error${NC}"
        if [[ $seed_exit1 -ne 0 ]]; then
            echo "$seed_output1" | grep -A 3 "FAILED\|Error"
        fi
        if [[ $seed_exit2 -ne 0 ]]; then
            echo "$seed_output2" | grep -A 3 "FAILED\|Error"
        fi
    else
        # Extract SEED_RESULT lines and compare LFSR states
        result1=$(echo "$seed_output1" | grep "SEED_RESULT:" | head -1)
        result2=$(echo "$seed_output2" | grep "SEED_RESULT:" | head -1)
        
        # Extract LFSR state (field 3) - this should differ for different seeds
        lfsr1=$(echo "$result1" | cut -d: -f3)
        lfsr2=$(echo "$result2" | cut -d: -f3)
        
        # Extract hash values (field 5) for additional comparison
        hash1=$(echo "$result1" | cut -d: -f5)
        hash2=$(echo "$result2" | cut -d: -f5)
        
        if [[ -z "$lfsr1" ]] || [[ -z "$lfsr2" ]]; then
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("DifferentSeeds.Comparison")
            echo -e "${RED}  ✗ DifferentSeeds.Comparison FAILED - Could not extract results${NC}"
        elif [[ "$lfsr1" == "$lfsr2" ]]; then
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("DifferentSeeds.Comparison")
            echo -e "${RED}  ✗ DifferentSeeds.Comparison FAILED - Different seeds produced SAME LFSR state!${NC}"
            echo "    Seed 12345 LFSR: $lfsr1"
            echo "    Seed 54321 LFSR: $lfsr2"
        else
            passed_tests=$((passed_tests + 1))
            echo -e "${GREEN}  ✓ DifferentSeeds.Comparison PASSED - Different seeds produce different LFSR states${NC}"
            echo "    Seed 12345 LFSR: $lfsr1"
            echo "    Seed 54321 LFSR: $lfsr2"
        fi
    fi
else
    echo -e "${YELLOW}Warning: seed_run1/run2 tests not found, skipping...${NC}"
fi

echo ""

# ============================================================================
# Special Paired Tests: Reproducibility (must produce IDENTICAL sequences)
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paired Test: Reproducibility Comparison${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ -x "./test_wave_gen_repro_run1" ]] && [[ -x "./test_wave_gen_repro_run2" ]]; then
    total_tests=$((total_tests + 1))
    
    echo -e "${BLUE}Running: test_wave_gen_repro_run1${NC}"
    repro_output1=$(./test_wave_gen_repro_run1 2>&1)
    repro_exit1=$?
    
    echo -e "${BLUE}Running: test_wave_gen_repro_run2${NC}"
    repro_output2=$(./test_wave_gen_repro_run2 2>&1)
    repro_exit2=$?
    
    if [[ $repro_exit1 -ne 0 ]] || [[ $repro_exit2 -ne 0 ]]; then
        failed_tests=$((failed_tests + 1))
        failed_test_names+=("Reproducibility.Comparison")
        echo -e "${RED}  ✗ Reproducibility.Comparison FAILED - Test execution error${NC}"
        if [[ $repro_exit1 -ne 0 ]]; then
            echo "$repro_output1" | grep -A 3 "FAILED\|Error"
        fi
        if [[ $repro_exit2 -ne 0 ]]; then
            echo "$repro_output2" | grep -A 3 "FAILED\|Error"
        fi
    else
        # Extract all SAMPLE lines and compare
        samples1=$(echo "$repro_output1" | grep "^SAMPLE:" | sort)
        samples2=$(echo "$repro_output2" | grep "^SAMPLE:" | sort)
        
        if [[ -z "$samples1" ]] || [[ -z "$samples2" ]]; then
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("Reproducibility.Comparison")
            echo -e "${RED}  ✗ Reproducibility.Comparison FAILED - Could not extract samples${NC}"
        elif [[ "$samples1" != "$samples2" ]]; then
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("Reproducibility.Comparison")
            echo -e "${RED}  ✗ Reproducibility.Comparison FAILED - Same seed produced DIFFERENT sequences!${NC}"
            # Show first difference
            diff_result=$(diff <(echo "$samples1") <(echo "$samples2") | head -10)
            echo "    First differences:"
            echo "$diff_result"
        else
            passed_tests=$((passed_tests + 1))
            sample_count=$(echo "$samples1" | wc -l | tr -d ' ')
            echo -e "${GREEN}  ✓ Reproducibility.Comparison PASSED - Same seed produces identical sequences${NC}"
            echo "    Verified $sample_count samples match exactly"
        fi
    fi
else
    echo -e "${YELLOW}Warning: repro_run1/run2 tests not found, skipping...${NC}"
fi

echo ""

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total tests run:    ${BLUE}${total_tests}${NC}"
echo -e "Tests passed:       ${GREEN}${passed_tests}${NC}"
echo -e "Tests failed:       ${RED}${failed_tests}${NC}"

if [[ $failed_tests -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed test suites:${NC}"
    for failed_test in "${failed_test_names[@]}"; do
        echo -e "  ${RED}✗ ${failed_test}${NC}"
    done
    echo ""
    exit 1
else
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All tests PASSED! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    exit 0
fi
