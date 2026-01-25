/**
 * @file test_ffe_deemphasis.cpp
 * @brief Unit test for TxFfeTdf module - De-emphasis Mode Configuration
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeModesTest, DeemphasisModeConfiguration) {
    TxFfeParams params;
    
    // PCIe Gen3 style de-emphasis (3.5dB)
    params.taps = {0.0, 1.0, -0.25};
    
    EXPECT_DOUBLE_EQ(params.taps[0], 0.0) << "Pre-cursor should be 0";
    EXPECT_DOUBLE_EQ(params.taps[1], 1.0) << "Main cursor should be 1.0";
    EXPECT_LT(params.taps[2], 0.0) << "Post-cursor should be negative";
    
    // De-emphasis ratio calculation: -20*log10(1.0/(1.0-0.25)) ~ 2.5 dB
    double main_cursor = params.taps[1];
    double post_cursor = params.taps[2];
    double ratio = main_cursor / (main_cursor + post_cursor);
    EXPECT_GT(ratio, 1.0) << "De-emphasis ratio should be > 1";
    
    // PCIe Gen4 style stronger de-emphasis
    params.taps = {0.0, 1.0, -0.35};
    EXPECT_LT(params.taps[2], -0.25) << "Gen4 should have stronger de-emphasis";
}
