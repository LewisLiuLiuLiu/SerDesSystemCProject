/**
 * @file test_adaption_threshold_basic.cpp
 * @brief Unit test for AdaptionDe - Sampler threshold behavior
 * 
 * NOTE (2026-10): sampler threshold adaptation was removed from AdaptionDe;
 * the threshold output is a fixed default (0.0). This test locks that contract.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionThresholdTest, ThresholdFixedDefault) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Threshold must stay at its fixed default (no drift)
    double threshold = tb->get_threshold();
    EXPECT_DOUBLE_EQ(threshold, 0.0) << "Sampler threshold should be the fixed default";
    
    // Clean up
    delete tb;
}
