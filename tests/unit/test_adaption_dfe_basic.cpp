/**
 * @file test_adaption_dfe_basic.cpp
 * @brief Unit test for AdaptionDe - Vref command generation (DFE-related path)
 * 
 * NOTE (2026-10): DFE tap updates moved to DfeAdaptTdf. This test covers the
 * remaining DFE-related behavior in AdaptionDe: Vref command generation from
 * DfeAdaptTdf statistics (STARTUP state tracks the initial Vref).
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionDfeTest, VrefCommandBasic) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = true;
    params.vref_adapt.enabled = true;
    params.cdr_pi.enabled = false;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // STARTUP state: use initial Vref
    tb->stats->set_dfe_state(0);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Verify Vref command is generated and within the clamp range
    double vref = tb->get_vref_cmd();
    EXPECT_GT(vref, 0.0) << "Vref command should be generated";
    EXPECT_GE(vref, 0.01);
    EXPECT_LE(vref, 0.5);
    EXPECT_NEAR(vref, params.vref_adapt.vref_initial, 1e-9)
        << "STARTUP state should track the initial Vref";
    
    // Clean up
    delete tb;
}
