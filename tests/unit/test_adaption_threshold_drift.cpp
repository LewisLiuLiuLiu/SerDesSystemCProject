/**
 * @file test_adaption_threshold_drift.cpp
 * @brief Unit test for AdaptionDe - threshold stability under adverse statistics
 * 
 * NOTE (2026-10): threshold adaptation was removed from AdaptionDe. This test
 * verifies the fixed threshold does not drift even under worst-case statistics,
 * and that the Vref command stays within its clamp range.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionThresholdTest, ThresholdStabilityUnderAdverseStats) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = true;
    params.vref_adapt.enabled = true;
    params.cdr_pi.enabled = false;
    params.fast_update_period = 1e-8;
    params.slow_update_period = 1e-7;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Adverse statistics: low confidence, strong asymmetry, TRACK state
    tb->stats->set_confidence(0.05);
    tb->stats->set_delta(0.9);
    tb->stats->set_dfe_state(2);
    
    // Run simulation
    sc_core::sc_start(5, sc_core::SC_US);
    
    // Threshold must remain the fixed default (no drift)
    double threshold = tb->get_threshold();
    EXPECT_DOUBLE_EQ(threshold, 0.0);
    
    // Vref command must stay within its clamp range
    double vref = tb->get_vref_cmd();
    EXPECT_GE(vref, 0.01);
    EXPECT_LE(vref, 0.5);
    
    // Clean up
    delete tb;
}
