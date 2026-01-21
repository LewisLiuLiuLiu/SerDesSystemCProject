/**
 * @file test_clock_generation_basic.cpp
 * @brief Unit tests for ClockGenerationTdf module
 * 
 * Tests cover:
 * - Basic functionality (IDEAL mode)
 * - Different frequency configurations
 * - Phase continuity verification
 * - Time step adaptation
 * - Parameter validation
 * - Clock type switching
 * 
 * @version 0.1
 * @date 2026-01-21
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include <vector>
#include "ams/clock_generation.h"
#include "common/parameters.h"

using namespace serdes;

// ============================================================================
// Phase Monitor Module (for testing)
// ============================================================================

/**
 * @brief Phase monitor module to capture clock phase output
 * Records phase values for post-simulation analysis
 */
class PhaseMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> phase_in;
    
    std::vector<double> m_phase_samples;
    std::vector<double> m_time_samples;
    int m_sample_limit;
    
    PhaseMonitor(sc_core::sc_module_name nm, int sample_limit = 1000)
        : sca_tdf::sca_module(nm)
        , phase_in("phase_in")
        , m_sample_limit(sample_limit)
    {}
    
    void set_attributes() {
        phase_in.set_rate(1);
    }
    
    void processing() {
        if (static_cast<int>(m_phase_samples.size()) < m_sample_limit) {
            m_phase_samples.push_back(phase_in.read());
            m_time_samples.push_back(sc_core::sc_time_stamp().to_seconds());
        }
    }
    
    // Analysis methods
    double get_mean_phase() const {
        if (m_phase_samples.empty()) return 0.0;
        double sum = 0.0;
        for (double p : m_phase_samples) sum += p;
        return sum / m_phase_samples.size();
    }
    
    double get_max_phase() const {
        if (m_phase_samples.empty()) return 0.0;
        double max_val = m_phase_samples[0];
        for (double p : m_phase_samples) {
            if (p > max_val) max_val = p;
        }
        return max_val;
    }
    
    double get_min_phase() const {
        if (m_phase_samples.empty()) return 0.0;
        double min_val = m_phase_samples[0];
        for (double p : m_phase_samples) {
            if (p < min_val) min_val = p;
        }
        return min_val;
    }
    
    std::vector<double> get_phase_increments() const {
        std::vector<double> increments;
        for (size_t i = 1; i < m_phase_samples.size(); ++i) {
            double delta = m_phase_samples[i] - m_phase_samples[i-1];
            // Handle phase wrap-around
            if (delta < -M_PI) delta += 2.0 * M_PI;
            increments.push_back(delta);
        }
        return increments;
    }
    
    int count_phase_wraps() const {
        int wraps = 0;
        for (size_t i = 1; i < m_phase_samples.size(); ++i) {
            if (m_phase_samples[i] < m_phase_samples[i-1] - M_PI) {
                wraps++;
            }
        }
        return wraps;
    }
};

// ============================================================================
// Clock Generation Testbench
// ============================================================================

SC_MODULE(ClockGenTestbench) {
    ClockGenerationTdf* clk_gen;
    PhaseMonitor* monitor;
    
    sca_tdf::sca_signal<double> sig_phase;
    
    ClockParams params;
    int sample_limit;
    
    ClockGenTestbench(sc_core::sc_module_name nm, const ClockParams& p, int samples = 1000)
        : sc_core::sc_module(nm)
        , params(p)
        , sample_limit(samples)
    {
        clk_gen = new ClockGenerationTdf("clk_gen", params);
        monitor = new PhaseMonitor("monitor", sample_limit);
        
        clk_gen->clk_phase(sig_phase);
        monitor->phase_in(sig_phase);
    }
    
    ~ClockGenTestbench() {
        delete clk_gen;
        delete monitor;
    }
    
    // Access monitor data
    const std::vector<double>& get_phase_samples() const {
        return monitor->m_phase_samples;
    }
    
    const std::vector<double>& get_time_samples() const {
        return monitor->m_time_samples;
    }
    
    double get_mean_phase() const { return monitor->get_mean_phase(); }
    double get_max_phase() const { return monitor->get_max_phase(); }
    double get_min_phase() const { return monitor->get_min_phase(); }
    std::vector<double> get_phase_increments() const { return monitor->get_phase_increments(); }
    int count_phase_wraps() const { return monitor->count_phase_wraps(); }
};

// ============================================================================
// Test Cases
// ============================================================================

// Test 1: Basic IDEAL clock functionality
TEST(ClockGenerationBasicTest, IdealClockBasic) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;  // 10 GHz
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 500);
    
    // Run for 50 clock periods (500 samples at 100 samples/period)
    sc_core::sc_start(50.0 / params.frequency, sc_core::SC_SEC);
    
    // Verify samples were collected
    EXPECT_GT(tb->get_phase_samples().size(), 0u);
    
    // Verify phase range is within [0, 2*pi)
    EXPECT_GE(tb->get_min_phase(), 0.0);
    EXPECT_LT(tb->get_max_phase(), 2.0 * M_PI + 0.01);  // Small tolerance
    
    delete tb;
}

// Test 2: Phase range verification
TEST(ClockGenerationBasicTest, PhaseRangeVerification) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 40e9;  // 40 GHz
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 1000);
    
    // Run for 10 clock periods
    sc_core::sc_start(10.0 / params.frequency, sc_core::SC_SEC);
    
    // All phase values should be in [0, 2*pi) range
    for (double phase : tb->get_phase_samples()) {
        EXPECT_GE(phase, 0.0) << "Phase should be non-negative";
        EXPECT_LT(phase, 2.0 * M_PI + 1e-10) << "Phase should be less than 2*pi";
    }
    
    delete tb;
}

// Test 3: Phase continuity (increments should be constant)
TEST(ClockGenerationBasicTest, PhaseContinuity) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 20e9;  // 20 GHz
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 500);
    
    // Run for 5 clock periods
    sc_core::sc_start(5.0 / params.frequency, sc_core::SC_SEC);
    
    std::vector<double> increments = tb->get_phase_increments();
    
    // Skip if not enough samples
    if (increments.size() < 2) {
        delete tb;
        GTEST_SKIP() << "Not enough samples collected";
    }
    
    // Expected increment: 2*pi / 100 (100 samples per period)
    double expected_increment = 2.0 * M_PI / 100.0;
    
    // All increments should be approximately equal (for ideal clock)
    for (double inc : increments) {
        EXPECT_NEAR(inc, expected_increment, 1e-10) 
            << "Phase increment should be constant for ideal clock";
    }
    
    delete tb;
}

// Test 4: Different frequency configurations
TEST(ClockGenerationBasicTest, FrequencyConfigurations) {
    std::vector<double> test_frequencies = {1e9, 10e9, 40e9, 80e9};
    
    for (double freq : test_frequencies) {
        ClockParams params;
        params.type = ClockType::IDEAL;
        params.frequency = freq;
        
        ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 200);
        
        // Run for 2 clock periods
        sc_core::sc_start(2.0 / freq, sc_core::SC_SEC);
        
        // Verify phase wraps occurred (at least 1 for 2 periods)
        int wraps = tb->count_phase_wraps();
        EXPECT_GE(wraps, 1) << "Should have at least 1 phase wrap for frequency " << freq;
        
        delete tb;
    }
}

// Test 5: Time step adaptation
TEST(ClockGenerationBasicTest, TimeStepAdaptation) {
    std::vector<double> test_frequencies = {10e9, 40e9, 80e9};
    std::vector<double> expected_timesteps = {1e-12, 0.25e-12, 0.125e-12};
    
    for (size_t i = 0; i < test_frequencies.size(); ++i) {
        ClockParams params;
        params.type = ClockType::IDEAL;
        params.frequency = test_frequencies[i];
        
        ClockGenerationTdf* clk_gen = new ClockGenerationTdf("clk_gen", params);
        
        // Verify expected timestep calculation
        double expected_ts = 1.0 / (test_frequencies[i] * 100.0);
        EXPECT_NEAR(clk_gen->get_expected_timestep(), expected_timesteps[i], 1e-15)
            << "Time step should adapt to frequency " << test_frequencies[i];
        
        delete clk_gen;
    }
}

// Test 6: Parameter validation - invalid frequency
TEST(ClockGenerationBasicTest, InvalidFrequencyValidation) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    
    // Test zero frequency
    params.frequency = 0.0;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Zero frequency should throw exception";
    
    // Test negative frequency
    params.frequency = -10e9;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Negative frequency should throw exception";
}

// Test 7: Parameter validation - extreme frequency
TEST(ClockGenerationBasicTest, ExtremeFrequencyValidation) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    
    // Test too high frequency (> 1 THz)
    params.frequency = 2e12;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Frequency above 1 THz should throw exception";
    
    // Test too low frequency (< 1 Hz)
    params.frequency = 0.5;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Frequency below 1 Hz should throw exception";
}

// Test 8: PLL parameter validation
TEST(ClockGenerationBasicTest, PllParameterValidation) {
    ClockParams params;
    params.type = ClockType::PLL;
    params.frequency = 40e9;
    
    // Test invalid charge pump current
    params.pll.cp_current = -1e-5;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Negative CP current should throw exception";
    
    // Reset and test invalid loop filter resistance
    params.pll.cp_current = 5e-5;
    params.pll.lf_R = 0.0;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Zero LF resistance should throw exception";
    
    // Reset and test invalid divider
    params.pll.lf_R = 10000;
    params.pll.divider = -1;
    EXPECT_THROW(
        ClockGenerationTdf* clk = new ClockGenerationTdf("clk", params),
        std::invalid_argument
    ) << "Negative divider should throw exception";
}

// Test 9: Clock type switching (PLL mode falls back to IDEAL)
TEST(ClockGenerationBasicTest, ClockTypeSwitching) {
    std::vector<ClockType> types = {ClockType::IDEAL, ClockType::PLL, ClockType::ADPLL};
    
    for (ClockType type : types) {
        ClockParams params;
        params.type = type;
        params.frequency = 10e9;
        
        ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 200);
        
        // Run for 2 clock periods
        sc_core::sc_start(2.0 / params.frequency, sc_core::SC_SEC);
        
        // All modes should produce valid phase output
        EXPECT_GT(tb->get_phase_samples().size(), 0u) 
            << "Clock type " << static_cast<int>(type) << " should produce output";
        
        // Phase should be in valid range
        EXPECT_GE(tb->get_min_phase(), 0.0);
        EXPECT_LT(tb->get_max_phase(), 2.0 * M_PI + 0.01);
        
        delete tb;
    }
}

// Test 10: Cycle counting verification
TEST(ClockGenerationBasicTest, CycleCountVerification) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;  // 10 GHz
    
    int expected_cycles = 10;
    int samples_per_cycle = 100;
    int total_samples = expected_cycles * samples_per_cycle + 50;  // Extra samples
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, total_samples);
    
    // Run for expected number of cycles
    sc_core::sc_start(static_cast<double>(expected_cycles) / params.frequency, sc_core::SC_SEC);
    
    // Count phase wraps
    int actual_wraps = tb->count_phase_wraps();
    
    // Should have approximately expected_cycles wraps
    EXPECT_NEAR(actual_wraps, expected_cycles, 1) 
        << "Cycle count should match expected value";
    
    delete tb;
}

// Test 11: Debug interface verification
TEST(ClockGenerationBasicTest, DebugInterface) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 40e9;
    
    ClockGenerationTdf* clk_gen = new ClockGenerationTdf("clk_gen", params);
    
    // Verify debug interface returns correct values
    EXPECT_EQ(clk_gen->get_frequency(), 40e9) << "Frequency should match";
    EXPECT_EQ(clk_gen->get_type(), ClockType::IDEAL) << "Type should match";
    EXPECT_NEAR(clk_gen->get_expected_timestep(), 0.25e-12, 1e-15) << "Timestep should match";
    
    delete clk_gen;
}

// Test 12: Initial phase should be zero
TEST(ClockGenerationBasicTest, InitialPhaseZero) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, 100);
    
    // Run for just a short time
    sc_core::sc_start(1e-12, sc_core::SC_SEC);
    
    // First phase sample should be zero (or very close to it)
    if (!tb->get_phase_samples().empty()) {
        EXPECT_NEAR(tb->get_phase_samples()[0], 0.0, 1e-10) 
            << "Initial phase should be zero";
    }
    
    delete tb;
}

// Test 13: Long simulation numerical stability
TEST(ClockGenerationBasicTest, LongSimulationStability) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;  // 10 GHz
    
    // Run for 1000 cycles to check numerical stability
    int num_cycles = 1000;
    int samples_per_cycle = 100;
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, num_cycles * samples_per_cycle + 100);
    
    sc_core::sc_start(static_cast<double>(num_cycles) / params.frequency, sc_core::SC_SEC);
    
    // All phase values should still be in valid range after long simulation
    for (double phase : tb->get_phase_samples()) {
        EXPECT_GE(phase, 0.0) << "Phase should remain non-negative";
        EXPECT_LT(phase, 2.0 * M_PI + 1e-9) << "Phase should remain less than 2*pi";
    }
    
    // Phase increments should still be consistent
    std::vector<double> increments = tb->get_phase_increments();
    double expected_increment = 2.0 * M_PI / 100.0;
    
    // Check last 100 increments
    size_t start_idx = increments.size() > 100 ? increments.size() - 100 : 0;
    for (size_t i = start_idx; i < increments.size(); ++i) {
        EXPECT_NEAR(increments[i], expected_increment, 1e-9) 
            << "Phase increment should remain stable after long simulation";
    }
    
    delete tb;
}

// Test 14: Mean phase distribution
TEST(ClockGenerationBasicTest, MeanPhaseDistribution) {
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;
    
    // Run for many cycles to get good distribution
    int num_cycles = 100;
    
    ClockGenTestbench* tb = new ClockGenTestbench("tb", params, num_cycles * 100 + 50);
    
    sc_core::sc_start(static_cast<double>(num_cycles) / params.frequency, sc_core::SC_SEC);
    
    // Mean phase should be approximately pi (uniform distribution over [0, 2*pi))
    double mean_phase = tb->get_mean_phase();
    EXPECT_NEAR(mean_phase, M_PI, 0.1) 
        << "Mean phase should be approximately pi for uniform distribution";
    
    delete tb;
}
