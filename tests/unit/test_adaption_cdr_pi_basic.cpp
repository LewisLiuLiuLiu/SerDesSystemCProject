/**
 * @file test_adaption_cdr_pi_basic.cpp
 * @brief Unit test for AdaptionDe - CDR PI Basic Function
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionCdrTest, CdrPiBasicFunction) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = true;
    params.cdr_pi.kp = 0.01;
    params.cdr_pi.ki = 1e-4;
    params.cdr_pi.phase_range = 5e-11;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set initial phase error
    tb->src->set_phase_error(1e-11);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Verify phase command is within range
    double phase_cmd = tb->get_phase_cmd();
    EXPECT_GE(phase_cmd, -params.cdr_pi.phase_range) << "Phase cmd should be >= -range";
    EXPECT_LE(phase_cmd, params.cdr_pi.phase_range) << "Phase cmd should be <= range";
    
    // Verify quantization
    double resolution = params.cdr_pi.phase_resolution;
    double quantized = std::round(phase_cmd / resolution) * resolution;
    EXPECT_NEAR(phase_cmd, quantized, 1e-15) << "Phase cmd should be quantized";
    
    // Clean up
    delete tb;
}
