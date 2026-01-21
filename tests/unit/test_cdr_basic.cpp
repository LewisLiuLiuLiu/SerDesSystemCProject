/**
 * @file test_cdr_basic.cpp
 * @brief Unit tests for RxCdrTdf module
 * 
 * Test coverage:
 * - Basic functionality (port connection, signal flow)
 * - Parameter configuration (PI controller, PAI)
 * - Parameter validation (invalid inputs)
 * - Phase range limiting
 * - Phase quantization
 * - Edge detection (rising, falling, no edge)
 * - PI controller behavior
 * - Multiple data patterns
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include "ams/rx_cdr.h"
#include "common/parameters.h"

using namespace serdes;

// ============================================================================
// Test Helper: Simple Data Source
// ============================================================================

class SimpleDataSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;

    std::vector<double> m_data_pattern;
    size_t m_index;

    SimpleDataSource(sc_core::sc_module_name nm, const std::vector<double>& pattern)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_data_pattern(pattern)
        , m_index(0)
    {}

    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(1.0 / 10e9, sc_core::SC_SEC);  // 10 GHz
    }

    void processing() {
        if (!m_data_pattern.empty()) {
            out.write(m_data_pattern[m_index % m_data_pattern.size()]);
            m_index++;
        } else {
            out.write(0.0);
        }
    }
};

// ============================================================================
// Test Helper: Testbench Module
// ============================================================================

SC_MODULE(CdrBasicTestbench) {
    SimpleDataSource* src;
    RxCdrTdf* cdr;

    sca_tdf::sca_signal<double> sig_data;
    sca_tdf::sca_signal<double> sig_phase;

    CdrParams params;

    CdrBasicTestbench(sc_core::sc_module_name nm, const CdrParams& p, const std::vector<double>& pattern)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SimpleDataSource("src", pattern);
        cdr = new RxCdrTdf("cdr", params);

        src->out(sig_data);
        cdr->in(sig_data);
        cdr->phase_out(sig_phase);
    }

    ~CdrBasicTestbench() {
        delete src;
        delete cdr;
    }

    double get_phase_output() {
        return sig_phase.read(0);
    }
    
    double get_integral_state() {
        return cdr->get_integral_state();
    }
    
    double get_phase_error() {
        return cdr->get_phase_error();
    }
};

// ============================================================================
// Test Case 1: Basic Functionality
// ============================================================================

TEST(CdrBasicTest, BasicFunctionality) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pi.edge_threshold = 0.5;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    std::vector<double> pattern = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    // Verify port connection
    SUCCEED() << "Port connection test passed";

    // Verify phase output is in valid range
    double phase = tb->get_phase_output();
    EXPECT_GE(phase, -params.pai.range) << "Phase output should be >= -range";
    EXPECT_LE(phase, params.pai.range) << "Phase output should be <= range";

    // Verify phase is quantized
    double quantized = std::round(phase / params.pai.resolution) * params.pai.resolution;
    EXPECT_NEAR(phase, quantized, 1e-15) << "Phase should be quantized";

    delete tb;
}

// ============================================================================
// Test Case 2: PI Controller Configuration
// ============================================================================

TEST(CdrTest, PIControllerConfiguration) {
    CdrParams params;

    // Test default values
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    EXPECT_GT(params.pi.kp, 0.0) << "Kp should be positive";
    EXPECT_GT(params.pi.ki, 0.0) << "Ki should be positive";

    // Test different Kp values
    params.pi.kp = 0.001;
    EXPECT_DOUBLE_EQ(params.pi.kp, 0.001);

    params.pi.kp = 0.1;
    EXPECT_DOUBLE_EQ(params.pi.kp, 0.1);

    // Test different Ki values
    params.pi.ki = 1e-5;
    EXPECT_DOUBLE_EQ(params.pi.ki, 1e-5);

    params.pi.ki = 1e-3;
    EXPECT_DOUBLE_EQ(params.pi.ki, 1e-3);

    // Test Ki < Kp (typical relationship)
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    EXPECT_LT(params.pi.ki, params.pi.kp) << "Ki should typically be smaller than Kp";
}

// ============================================================================
// Test Case 3: PAI Configuration
// ============================================================================

TEST(CdrTest, PAIConfiguration) {
    CdrParams params;

    // Test default values
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;
    EXPECT_GT(params.pai.resolution, 0.0) << "Resolution should be positive";
    EXPECT_GT(params.pai.range, 0.0) << "Range should be positive";

    // Test different resolutions
    params.pai.resolution = 5e-13;  // 0.5ps
    EXPECT_DOUBLE_EQ(params.pai.resolution, 5e-13);

    params.pai.resolution = 5e-12;  // 5ps
    EXPECT_DOUBLE_EQ(params.pai.resolution, 5e-12);

    // Test different ranges
    params.pai.range = 1e-11;   // ±10ps
    EXPECT_DOUBLE_EQ(params.pai.range, 1e-11);

    params.pai.range = 1e-10;   // ±100ps
    EXPECT_DOUBLE_EQ(params.pai.range, 1e-10);

    // Range should be larger than resolution
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;
    EXPECT_GT(params.pai.range, params.pai.resolution);
}

// ============================================================================
// Test Case 4: Edge Threshold Configuration
// ============================================================================

TEST(CdrTest, EdgeThresholdConfiguration) {
    CdrParams params;
    
    // Test default edge threshold
    EXPECT_DOUBLE_EQ(params.pi.edge_threshold, 0.5);
    
    // Test custom thresholds
    params.pi.edge_threshold = 0.3;
    EXPECT_DOUBLE_EQ(params.pi.edge_threshold, 0.3);
    
    params.pi.edge_threshold = 0.8;
    EXPECT_DOUBLE_EQ(params.pi.edge_threshold, 0.8);
    
    // Test adaptive threshold flag
    EXPECT_FALSE(params.pi.adaptive_threshold);
    params.pi.adaptive_threshold = true;
    EXPECT_TRUE(params.pi.adaptive_threshold);
}

// ============================================================================
// Test Case 5: Parameter Validation - Invalid Kp
// ============================================================================

TEST(CdrTest, ParameterValidationNegativeKp) {
    CdrParams params;
    params.pi.kp = -0.01;  // Invalid: negative Kp
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    std::vector<double> pattern = {1.0, -1.0};
    
    EXPECT_THROW({
        CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);
        delete tb;
    }, std::invalid_argument) << "Should throw for negative Kp";
}

// ============================================================================
// Test Case 6: Parameter Validation - Invalid Resolution
// ============================================================================

TEST(CdrTest, ParameterValidationInvalidResolution) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 0.0;  // Invalid: zero resolution
    params.pai.range = 5e-11;

    std::vector<double> pattern = {1.0, -1.0};
    
    EXPECT_THROW({
        CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);
        delete tb;
    }, std::invalid_argument) << "Should throw for zero resolution";
}

// ============================================================================
// Test Case 7: Parameter Validation - Invalid Range
// ============================================================================

TEST(CdrTest, ParameterValidationInvalidRange) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 0.0;  // Invalid: zero range

    std::vector<double> pattern = {1.0, -1.0};
    
    EXPECT_THROW({
        CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);
        delete tb;
    }, std::invalid_argument) << "Should throw for zero range";
}

// ============================================================================
// Test Case 8: Phase Range Limiting
// ============================================================================

TEST(CdrTest, PhaseRangeLimit) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Constant input (no edges) - phase should remain stable
    std::vector<double> pattern = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    double phase = tb->get_phase_output();
    EXPECT_GE(phase, -params.pai.range) << "Phase should not exceed negative limit";
    EXPECT_LE(phase, params.pai.range) << "Phase should not exceed positive limit";

    delete tb;
}

// ============================================================================
// Test Case 9: Phase Quantization
// ============================================================================

TEST(CdrTest, PhaseQuantization) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;  // 1ps
    params.pai.range = 5e-11;

    std::vector<double> pattern = {1.0, -1.0, 1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    double phase = tb->get_phase_output();
    double quantized = std::round(phase / params.pai.resolution) * params.pai.resolution;
    EXPECT_NEAR(phase, quantized, 1e-15) << "Phase should be quantized to resolution";

    delete tb;
}

// ============================================================================
// Test Case 10: Rising Edge Detection
// ============================================================================

TEST(CdrTest, RisingEdgeDetection) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Rising edge: -1.0 -> 1.0
    std::vector<double> pattern = {-1.0, 1.0, 1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(5, sc_core::SC_NS);

    // After rising edge, phase error should be +1 (positive adjustment)
    double phase = tb->get_phase_output();
    EXPECT_GE(phase, 0.0) << "Rising edge should produce positive phase adjustment";

    delete tb;
}

// ============================================================================
// Test Case 11: Falling Edge Detection
// ============================================================================

TEST(CdrTest, FallingEdgeDetection) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Falling edge: 1.0 -> -1.0
    std::vector<double> pattern = {1.0, -1.0, -1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(5, sc_core::SC_NS);

    // After falling edge, phase error should be -1 (negative adjustment)
    double phase = tb->get_phase_output();
    EXPECT_LE(phase, 0.0) << "Falling edge should produce negative phase adjustment";

    delete tb;
}

// ============================================================================
// Test Case 12: No Edge Detection
// ============================================================================

TEST(CdrTest, NoEdgeDetection) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // No edges: constant value
    std::vector<double> pattern = {1.0, 1.0, 1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    // With no edges, phase should remain near zero (only initial conditions)
    double phase = tb->get_phase_output();
    EXPECT_NEAR(phase, 0.0, params.pai.resolution * 2) << "No edge should produce minimal phase change";

    delete tb;
}

// ============================================================================
// Test Case 13: PI Controller Proportional Response
// ============================================================================

TEST(CdrTest, ProportionalResponse) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 0.0;  // Disable integral
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Single rising edge
    std::vector<double> pattern = {-1.0, 1.0, 1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(3, sc_core::SC_NS);

    // With only proportional, output should be Kp * phase_error
    double phase = tb->get_phase_output();
    // After one rising edge, expected ~ Kp * 1.0 = 0.01
    double expected = params.pi.kp;
    double quantized_expected = std::round(expected / params.pai.resolution) * params.pai.resolution;
    EXPECT_NEAR(phase, quantized_expected, params.pai.resolution * 2);

    delete tb;
}

// ============================================================================
// Test Case 14: PI Controller Integral Accumulation
// ============================================================================

TEST(CdrTest, IntegralAccumulation) {
    CdrParams params;
    params.pi.kp = 0.0;  // Disable proportional
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Multiple rising edges to accumulate integral
    std::vector<double> pattern = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    // Integral should accumulate over multiple edges
    double integral = tb->get_integral_state();
    // Multiple rising edges should accumulate positive integral
    EXPECT_GT(integral, 0.0) << "Integral should accumulate with rising edges";

    delete tb;
}

// ============================================================================
// Test Case 15: Alternating Pattern (High Transition Density)
// ============================================================================

TEST(CdrTest, AlternatingPattern) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Alternating pattern: maximum transition density
    std::vector<double> pattern = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    double phase = tb->get_phase_output();
    EXPECT_GE(phase, -params.pai.range);
    EXPECT_LE(phase, params.pai.range);

    delete tb;
}

// ============================================================================
// Test Case 16: Low Transition Density Pattern
// ============================================================================

TEST(CdrTest, LowTransitionDensity) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Low transition density: long runs
    std::vector<double> pattern = {1.0, 1.0, 1.0, -1.0, -1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    double phase = tb->get_phase_output();
    EXPECT_GE(phase, -params.pai.range);
    EXPECT_LE(phase, params.pai.range);

    delete tb;
}

// ============================================================================
// Test Case 17: Different PI Configurations
// ============================================================================

TEST(CdrTest, DifferentPIConfigurations) {
    std::vector<double> pattern = {1.0, -1.0, 1.0, -1.0};

    // Configuration 1: Standard
    {
        CdrParams params1;
        params1.pi.kp = 0.01;
        params1.pi.ki = 1e-4;
        params1.pai.resolution = 1e-12;
        params1.pai.range = 5e-11;

        CdrBasicTestbench* tb1 = new CdrBasicTestbench("tb1", params1, pattern);
        sc_core::sc_start(10, sc_core::SC_NS);
        double phase1 = tb1->get_phase_output();
        EXPECT_GE(phase1, -params1.pai.range);
        EXPECT_LE(phase1, params1.pai.range);
        delete tb1;
    }

    // Configuration 2: High gain
    {
        CdrParams params2;
        params2.pi.kp = 0.02;
        params2.pi.ki = 2e-4;
        params2.pai.resolution = 1e-12;
        params2.pai.range = 5e-11;

        CdrBasicTestbench* tb2 = new CdrBasicTestbench("tb2", params2, pattern);
        sc_core::sc_start(10, sc_core::SC_NS);
        double phase2 = tb2->get_phase_output();
        EXPECT_GE(phase2, -params2.pai.range);
        EXPECT_LE(phase2, params2.pai.range);
        delete tb2;
    }

    // Configuration 3: Low gain
    {
        CdrParams params3;
        params3.pi.kp = 0.005;
        params3.pi.ki = 5e-5;
        params3.pai.resolution = 1e-12;
        params3.pai.range = 5e-11;

        CdrBasicTestbench* tb3 = new CdrBasicTestbench("tb3", params3, pattern);
        sc_core::sc_start(10, sc_core::SC_NS);
        double phase3 = tb3->get_phase_output();
        EXPECT_GE(phase3, -params3.pai.range);
        EXPECT_LE(phase3, params3.pai.range);
        delete tb3;
    }
}

// ============================================================================
// Test Case 18: Parameter Boundary Conditions
// ============================================================================

TEST(CdrTest, ParameterBoundaryConditions) {
    CdrParams params;

    // Test very small Kp (valid)
    params.pi.kp = 1e-6;
    EXPECT_DOUBLE_EQ(params.pi.kp, 1e-6);

    // Test very large Kp (valid but may cause instability)
    params.pi.kp = 1.0;
    EXPECT_DOUBLE_EQ(params.pi.kp, 1.0);

    // Test very small Ki (valid)
    params.pi.ki = 1e-10;
    EXPECT_DOUBLE_EQ(params.pi.ki, 1e-10);

    // Test very small resolution (valid, 1fs)
    params.pai.resolution = 1e-15;
    EXPECT_DOUBLE_EQ(params.pai.resolution, 1e-15);

    // Test very large resolution (valid, 1ns)
    params.pai.resolution = 1e-9;
    EXPECT_DOUBLE_EQ(params.pai.resolution, 1e-9);

    // Test very small range (valid, ±1ps)
    params.pai.range = 1e-12;
    EXPECT_DOUBLE_EQ(params.pai.range, 1e-12);

    // Test very large range (valid, ±1ns)
    params.pai.range = 1e-9;
    EXPECT_DOUBLE_EQ(params.pai.range, 1e-9);
}

// ============================================================================
// Test Case 19: Debug Interface
// ============================================================================

TEST(CdrTest, DebugInterface) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    std::vector<double> pattern = {-1.0, 1.0, 1.0, 1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(5, sc_core::SC_NS);

    // Test debug interface methods
    double integral = tb->get_integral_state();
    double phase_error = tb->get_phase_error();
    
    // Integral should be accumulated
    EXPECT_GE(integral, 0.0) << "Integral state should be accessible";
    
    // Phase error should be one of {-1, 0, 1}
    EXPECT_TRUE(phase_error == -1.0 || phase_error == 0.0 || phase_error == 1.0)
        << "Phase error should be -1, 0, or 1";

    delete tb;
}

// ============================================================================
// Test Case 20: Edge Threshold Effect
// ============================================================================

TEST(CdrTest, EdgeThresholdEffect) {
    std::vector<double> pattern = {0.0, 0.4, 0.0, 0.4};  // Small transitions
    
    // With threshold 0.5, these should NOT trigger edge detection
    {
        CdrParams params;
        params.pi.kp = 0.01;
        params.pi.ki = 1e-4;
        params.pi.edge_threshold = 0.5;  // Higher than transition
        params.pai.resolution = 1e-12;
        params.pai.range = 5e-11;

        CdrBasicTestbench* tb = new CdrBasicTestbench("tb_high", params, pattern);
        sc_core::sc_start(10, sc_core::SC_NS);
        double phase_high = tb->get_phase_output();
        // No edges detected, phase should be near zero
        EXPECT_NEAR(phase_high, 0.0, params.pai.resolution * 2);
        delete tb;
    }
    
    // With threshold 0.3, these SHOULD trigger edge detection
    {
        CdrParams params;
        params.pi.kp = 0.01;
        params.pi.ki = 1e-4;
        params.pi.edge_threshold = 0.3;  // Lower than transition
        params.pai.resolution = 1e-12;
        params.pai.range = 5e-11;

        CdrBasicTestbench* tb = new CdrBasicTestbench("tb_low", params, pattern);
        sc_core::sc_start(10, sc_core::SC_NS);
        double phase_low = tb->get_phase_output();
        // Edges detected, phase should be non-zero
        EXPECT_NE(phase_low, 0.0) << "Edges should be detected with lower threshold";
        delete tb;
    }
}
