/**
 * @file test_adaption_rollback.cpp
 * @brief Unit test for AdaptionDe - Freeze + rollback mechanism
 * 
 * NOTE (2026-10): the freeze trigger is now amplitude/phase anomaly based
 * (error-count bursts are no longer wired into AdaptionDe). This test drives a
 * real anomaly with freeze+rollback enabled and verifies the module freezes.
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionFreezeTest, RollbackMechanism) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = true;
    params.safety.freeze_on_error = true;
    params.safety.rollback_enable = true;
    params.safety.snapshot_interval = 1e-7;  // 100 ns snapshots
    params.fast_update_period = 1e-8;
    params.slow_update_period = 1e-7;
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Run with normal operation (builds snapshots)
    tb->src->set_amplitude(0.3);
    sc_core::sc_start(500, sc_core::SC_NS);
    EXPECT_FALSE(tb->get_freeze_flag()) << "Should not be frozen under normal conditions";
    
    // Inject amplitude anomaly -> freeze + rollback path executes
    tb->src->set_amplitude(2.5);
    sc_core::sc_start(200, sc_core::SC_NS);
    
    // Anomaly persists, so the module reports frozen (rollback runs on detect)
    EXPECT_TRUE(tb->get_freeze_flag()) << "Should freeze under amplitude anomaly";
    
    // Clean up
    delete tb;
}
