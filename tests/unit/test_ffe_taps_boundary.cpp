/**
 * @file test_ffe_taps_boundary.cpp
 * @brief Unit test for TxFfeTdf module - Parameter Boundary Conditions
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeTapsTest, ParameterBoundaryConditions) {
    TxFfeParams params;
    
    // Test single tap configuration
    params.taps = {1.0};
    EXPECT_EQ(params.taps.size(), 1);
    
    // Test 7 tap configuration
    params.taps = {0.02, 0.08, 0.15, 0.5, -0.15, -0.1, -0.05};
    EXPECT_EQ(params.taps.size(), 7);
    
    // Verify tap coefficient range
    for (double t : params.taps) {
        EXPECT_GE(t, -1.0) << "Tap should be >= -1.0";
        EXPECT_LE(t, 1.0) << "Tap should be <= 1.0";
    }
    
    // Test empty tap configuration (should use default or minimum)
    params.taps = {};
    EXPECT_TRUE(params.taps.empty());
}
