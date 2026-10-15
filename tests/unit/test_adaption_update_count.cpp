/**
 * @file test_adaption_update_count.cpp
 * @brief Unit test for AdaptionDe - Update Count Verification
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionBasicTest, UpdateCountVerification) {
    AdaptionParams params;
    params.fast_update_period = 1e-8;  // 10 ns for fast updates
    params.cdr_pi.enabled = true;       // Enable fast path updates
    params.safety.freeze_on_error = false;  // Disable freeze for this test
    
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Wait for reset and initial stabilization
    sc_core::sc_start(50, sc_core::SC_NS);
    int initial_count = tb->get_update_count();
    
    // Run for enough time to see multiple updates (20 update periods)
    sc_core::sc_start(200, sc_core::SC_NS);
    int final_count = tb->get_update_count();
    
    // Verify updates occurred
    EXPECT_GT(final_count, initial_count) << "Update count should increase";
    EXPECT_GE(final_count, 10) << "Should have at least 10 updates in 200ns with 10ns period";
    
    // Clean up
    delete tb;
}
