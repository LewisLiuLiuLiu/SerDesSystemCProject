/**
 * @file test_adaption_freeze_mechanism.cpp
 * @brief Unit test for AdaptionDe - Freeze mechanism (disabled gate)
 * 
 * NOTE (2026-10): the freeze trigger is now amplitude/phase anomaly based
 * (error-count bursts are no longer wired into AdaptionDe). This test verifies
 * that freeze_on_error=false keeps the module unfrozen even under an anomaly.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionFreezeTest, FreezeDisabledGate) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = true;
    params.safety.freeze_on_error = false;  // Freeze disabled
    params.fast_update_period = 1e-8;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Normal operating conditions
    tb->src->set_amplitude(0.3);
    tb->src->set_phase_error(1e-11);  // 10 ps, valid
    
    // Wait for system to stabilize
    sc_core::sc_start(200, sc_core::SC_NS);
    
    // Verify not frozen in normal conditions
    EXPECT_FALSE(tb->get_freeze_flag()) << "Should not be frozen with normal conditions";
    
    // Inject an amplitude anomaly (outside [0.01, 2.0])
    tb->src->set_amplitude(2.5);
    sc_core::sc_start(200, sc_core::SC_NS);
    
    // With freeze_on_error=false, must still not be frozen
    EXPECT_FALSE(tb->get_freeze_flag()) << "Should not freeze when freeze_on_error=false";
    
    // Clean up
    delete tb;
}
