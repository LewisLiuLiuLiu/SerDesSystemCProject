/**
 * @file test_adaption_agc_convergence.cpp
 * @brief Unit test for AdaptionDe - AGC Convergence
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionAgcTest, AgcConvergence) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.agc.initial_gain = 1.0;
    params.agc.target_amplitude = 0.4;
    params.agc.kp = 0.5;        // Increased for faster convergence
    params.agc.ki = 500.0;      // Increased for faster convergence
    params.slow_update_period = 1e-7;  // 100 ns for faster updates
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    params.safety.freeze_on_error = false;  // Disable freeze for this test
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set amplitude significantly below target to trigger gain increase
    tb->src->set_amplitude(0.15);  // Much lower than target 0.4
    
    // Run simulation long enough for multiple slow path updates
    // With 100ns period, 10us = 100 updates
    sc_core::sc_start(10, sc_core::SC_US);
    
    // Verify gain increased from initial value
    double final_gain = tb->get_vga_gain();
    EXPECT_GT(final_gain, params.agc.initial_gain) << "Gain should increase when amplitude < target";
    EXPECT_LT(final_gain, params.agc.gain_max) << "Gain should be within limits";
    
    // Clean up
    delete tb;
}
