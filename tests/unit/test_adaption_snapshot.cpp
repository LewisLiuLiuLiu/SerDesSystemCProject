/**
 * @file test_adaption_snapshot.cpp
 * @brief Unit test for AdaptionDe - Snapshot Management
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionFreezeTest, SnapshotManagement) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.dfe.enabled = true;
    params.vref_adapt.enabled = true;
    params.cdr_pi.enabled = true;
    params.fast_update_period = 1e-8;   // 10 ns for fast updates
    params.slow_update_period = 1e-7;   // 100 ns for slow updates
    params.safety.snapshot_interval = 1e-7;  // 100 ns snapshots
    params.safety.freeze_on_error = false;  // Disable freeze for this test
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Run long enough for multiple snapshots and updates
    sc_core::sc_start(2, sc_core::SC_US);
    
    // System should be working normally with updates
    int update_count = tb->get_update_count();
    EXPECT_GT(update_count, 0) << "Updates should occur, got " << update_count;
    EXPECT_GE(update_count, 50) << "Should have many updates in 2us with 10ns fast period";
    
    // Clean up
    delete tb;
}
