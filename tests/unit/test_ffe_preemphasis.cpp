/**
 * @file test_ffe_preemphasis.cpp
 * @brief Unit test for TxFfeTdf module - Pre-emphasis Mode Configuration
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeModesTest, PreemphasisModeConfiguration) {
    TxFfeParams params;
    
    // Balanced pre-emphasis configuration
    params.taps = {0.1, 0.6, -0.15, -0.05};
    
    EXPECT_GT(params.taps[0], 0.0) << "Pre-cursor should be positive";
    EXPECT_GT(params.taps[1], params.taps[0]) << "Main cursor should be largest";
    EXPECT_LT(params.taps[2], 0.0) << "Post-cursor 1 should be negative";
    EXPECT_LT(params.taps[3], 0.0) << "Post-cursor 2 should be negative";
    
    // Verify main tap is the largest
    double max_abs = 0.0;
    int max_idx = -1;
    for (size_t i = 0; i < params.taps.size(); ++i) {
        if (std::abs(params.taps[i]) > max_abs) {
            max_abs = std::abs(params.taps[i]);
            max_idx = i;
        }
    }
    EXPECT_EQ(max_idx, 1) << "Main tap should be at index 1";
}
