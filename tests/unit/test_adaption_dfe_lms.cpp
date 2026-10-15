/**
 * @file test_adaption_dfe_lms.cpp
 * @brief Unit test for AdaptionDe - Vref acquisition response
 * 
 * NOTE (2026-10): DFE tap update algorithms (lms/sign-lms) moved to
 * DfeAdaptTdf. This test covers the remaining AdaptionDe path: in ACQUIRE
 * state the Vref command grows when joint confidence exceeds the target.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionDfeTest, VrefAcquireTowardConfidenceTarget) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = true;
    params.vref_adapt.enabled = true;
    params.cdr_pi.enabled = false;
    params.safety.freeze_on_error = false;  // 隔离 vref 路径
    params.slow_update_period = 1e-7;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // ACQUIRE state with confidence above the target (0.7)
    tb->stats->set_dfe_state(1);
    tb->stats->set_confidence(0.95);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Vref should increase above the initial value
    double vref = tb->get_vref_cmd();
    EXPECT_GT(vref, params.vref_adapt.vref_initial)
        << "Vref should increase when confidence exceeds the target";
    EXPECT_LE(vref, 0.5);
    
    // Clean up
    delete tb;
}
