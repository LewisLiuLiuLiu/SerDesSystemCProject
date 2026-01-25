/**
 * @file test_ffe_tap_coefficients.cpp
 * @brief Unit test for TxFfeTdf module - Tap Coefficients Configuration
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeBasicTest, TapCoefficientsConfiguration) {
    TxFfeParams params;
    
    // Test default configuration
    EXPECT_EQ(params.taps.size(), 3);
    EXPECT_DOUBLE_EQ(params.taps[0], 0.2);
    EXPECT_DOUBLE_EQ(params.taps[1], 0.6);
    EXPECT_DOUBLE_EQ(params.taps[2], 0.2);
    
    // Test custom configuration - 5 taps
    params.taps = {0.05, 0.15, 0.6, -0.15, -0.05};
    EXPECT_EQ(params.taps.size(), 5);
    
    // Verify main tap is the largest
    double max_tap = 0.0;
    for (double t : params.taps) {
        if (std::abs(t) > std::abs(max_tap)) max_tap = t;
    }
    EXPECT_DOUBLE_EQ(max_tap, 0.6) << "Main tap should be the largest";
}
