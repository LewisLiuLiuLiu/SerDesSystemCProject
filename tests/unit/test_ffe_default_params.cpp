/**
 * @file test_ffe_default_params.cpp
 * @brief Unit test for TxFfeTdf module - Default Parameter Verification
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeBasicTest, DefaultParameterVerification) {
    TxFfeParams params;
    
    // Verify default values
    EXPECT_EQ(params.taps.size(), 3) << "Default should have 3 taps";
    EXPECT_DOUBLE_EQ(params.taps[0], 0.2);
    EXPECT_DOUBLE_EQ(params.taps[1], 0.6);
    EXPECT_DOUBLE_EQ(params.taps[2], 0.2);
    
    // Verify default tap sum
    double sum = params.taps[0] + params.taps[1] + params.taps[2];
    EXPECT_DOUBLE_EQ(sum, 1.0) << "Default tap sum should be 1.0";
    
    // Verify main tap (middle) is largest
    EXPECT_GT(params.taps[1], params.taps[0]);
    EXPECT_GT(params.taps[1], params.taps[2]);
}
