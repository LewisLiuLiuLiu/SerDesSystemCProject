/**
 * @file test_adaption_agc_basic.cpp
 * @brief Unit test for AdaptionDe - AGC Basic Function
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionAgcTest, AgcBasicFunction) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.agc.initial_gain = 2.0;
    params.agc.target_amplitude = 0.4;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set amplitude below target (should increase gain)
    tb->src->set_amplitude(0.3);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Verify gain is within valid range
    double gain = tb->get_vga_gain();
    EXPECT_GE(gain, params.agc.gain_min) << "Gain should be >= gain_min";
    EXPECT_LE(gain, params.agc.gain_max) << "Gain should be <= gain_max";
    
    // Clean up
    delete tb;
}
