/**
 * @file test_adaption_threshold_hysteresis.cpp
 * @brief Unit test for AdaptionDe - Hysteresis output
 * 
 * NOTE (2026-10): hysteresis adaptation was removed from AdaptionDe;
 * the hysteresis output is a fixed positive default (0.02 V).
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionThresholdTest, HysteresisFixedDefault) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Hysteresis is a fixed positive default (0.02 V), no longer adapted
    double hysteresis = tb->get_hysteresis();
    EXPECT_DOUBLE_EQ(hysteresis, 0.02) << "Sampler hysteresis should be the fixed default";
    EXPECT_GT(hysteresis, 0.0);
    
    // Clean up
    delete tb;
}
