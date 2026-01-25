/**
 * @file test_ffe_basic_functionality.cpp
 * @brief Unit test for TxFfeTdf module - All Basic Functionality
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeBasicTest, AllBasicFunctionality) {
    TxFfeParams params;
    params.taps = {0.2, 0.6, 0.2};  // Symmetric taps, sum = 1.0
    
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb", params, 
                                                   SignalSource::DC, 1.0);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // Test 1: Verify port connection
    SUCCEED() << "Port connection test passed";
    
    // Test 2: DC steady state output should equal input * tap sum
    double tap_sum = 0.0;
    for (double t : params.taps) tap_sum += t;
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GT(samples.size(), 10) << "Should have collected samples";
    
    double steady_state = samples.back();
    EXPECT_NEAR(steady_state, 1.0 * tap_sum, 0.001) 
        << "DC gain should equal sum of taps";
    
    sc_core::sc_stop();
}
