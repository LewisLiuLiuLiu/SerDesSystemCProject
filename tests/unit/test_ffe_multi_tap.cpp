/**
 * @file test_ffe_multi_tap.cpp
 * @brief Unit test for TxFfeTdf module - Multi Tap Configuration
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeTapsTest, MultiTapConfiguration) {
    // 3-tap configuration
    TxFfeParams params3;
    params3.taps = {0.2, 0.6, 0.2};
    EXPECT_EQ(params3.taps.size(), 3);
    
    double sum3 = 0.0;
    for (double t : params3.taps) sum3 += t;
    EXPECT_NEAR(sum3, 1.0, 0.001) << "3-tap sum should be 1.0";
    
    // 5-tap configuration
    TxFfeParams params5;
    params5.taps = {0.05, 0.15, 0.6, 0.15, 0.05};
    EXPECT_EQ(params5.taps.size(), 5);
    
    double sum5 = 0.0;
    for (double t : params5.taps) sum5 += t;
    EXPECT_NEAR(sum5, 1.0, 0.001) << "5-tap sum should be 1.0";
    
    // 7-tap configuration
    TxFfeParams params7;
    params7.taps = {0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02};
    EXPECT_EQ(params7.taps.size(), 7);
    
    double sum7 = 0.0;
    for (double t : params7.taps) sum7 += t;
    EXPECT_NEAR(sum7, 1.0, 0.001) << "7-tap sum should be 1.0";
}
