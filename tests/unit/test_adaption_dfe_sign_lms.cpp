/**
 * @file test_adaption_dfe_sign_lms.cpp
 * @brief Unit test for AdaptionDe - Vref tracking correction
 * 
 * NOTE (2026-10): DFE tap update algorithms moved to DfeAdaptTdf. This test
 * covers the remaining AdaptionDe path: in TRACK state a large asymmetry
 * metric (Delta) triggers a downward Vref correction.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionDfeTest, VrefTrackAsymmetryCorrection) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = true;
    params.vref_adapt.enabled = true;
    params.cdr_pi.enabled = false;
    params.safety.freeze_on_error = false;  // 隔离 vref 路径
    params.slow_update_period = 1e-7;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // TRACK state with asymmetry above the alert level (0.15)
    tb->stats->set_dfe_state(2);
    tb->stats->set_delta(0.9);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Vref should be corrected downward
    double vref = tb->get_vref_cmd();
    EXPECT_LT(vref, params.vref_adapt.vref_initial)
        << "Vref should be corrected down when asymmetry exceeds the alert level";
    EXPECT_GE(vref, 0.01);
    
    // Clean up
    delete tb;
}
