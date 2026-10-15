/**
 * @file test_adaption_cdr_pi_antiwindup.cpp
 * @brief Unit test for AdaptionDe - CDR PI Anti-Windup
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionCdrTest, CdrPiAntiWindup) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = true;
    params.cdr_pi.kp = 0.1;
    params.cdr_pi.ki = 1e-3;
    params.cdr_pi.phase_range = 2e-11;
    params.cdr_pi.anti_windup = true;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set large phase error to trigger saturation
    tb->src->set_phase_error(5e-11);
    
    // Run simulation
    sc_core::sc_start(1, sc_core::SC_US);
    
    // Phase command should be clamped
    double phase_cmd = tb->get_phase_cmd();
    EXPECT_GE(phase_cmd, -params.cdr_pi.phase_range);
    EXPECT_LE(phase_cmd, params.cdr_pi.phase_range);
    
    // Clean up
    delete tb;
}
