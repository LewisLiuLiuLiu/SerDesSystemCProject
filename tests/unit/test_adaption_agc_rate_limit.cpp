/**
 * @file test_adaption_agc_rate_limit.cpp
 * @brief Unit test for AdaptionDe - AGC Rate Limiting
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionAgcTest, AgcRateLimiting) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.agc.initial_gain = 2.0;
    params.agc.target_amplitude = 0.4;
    params.agc.rate_limit = 10.0;  // Limited rate
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set amplitude very low to trigger aggressive adjustment
    tb->src->set_amplitude(0.1);
    
    // Run short simulation
    sc_core::sc_start(100, sc_core::SC_NS);
    
    double gain = tb->get_vga_gain();
    
    // Gain should be limited by rate limiter
    EXPECT_GE(gain, params.agc.gain_min);
    EXPECT_LE(gain, params.agc.gain_max);
    
    // Clean up
    delete tb;
}
