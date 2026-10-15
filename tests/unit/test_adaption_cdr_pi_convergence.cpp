/**
 * @file test_adaption_cdr_pi_convergence.cpp
 * @brief Unit test for AdaptionDe - CDR PI Convergence
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionCdrTest, CdrPiConvergence) {
    AdaptionParams params;
    params.agc.enabled = false;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = true;
    params.cdr_pi.kp = 0.01;
    params.cdr_pi.ki = 1e-4;
    params.cdr_pi.phase_range = 5e-11;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Set constant phase error
    tb->src->set_phase_error(1e-12);
    
    // Run longer simulation
    sc_core::sc_start(5, sc_core::SC_US);
    
    // Phase command should adjust to compensate
    double phase_cmd = tb->get_phase_cmd();
    EXPECT_GE(phase_cmd, -params.cdr_pi.phase_range);
    EXPECT_LE(phase_cmd, params.cdr_pi.phase_range);
    
    // Clean up
    delete tb;
}
