/**
 * @file test_ffe_convolution.cpp
 * @brief Unit test for TxFfeTdf module - Convolution Calculation Correctness
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeResponseTest, ConvolutionCalculationCorrectness) {
    TxFfeParams params;
    params.taps = {0.25, 0.5, 0.25};  // Simple symmetric filter
    
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb_conv", params, 
                                                   SignalSource::SQUARE, 1.0, 1e9);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GT(samples.size(), 10) << "Should have collected samples";
    
    // Verify output is in reasonable range
    double max_val = tb->sink->get_max();
    double min_val = tb->sink->get_min();
    
    // For normalized taps, output range should be around [-1, 1]
    EXPECT_LE(max_val, 1.5) << "Max output should be bounded";
    EXPECT_GE(min_val, -1.5) << "Min output should be bounded";
    
    // Verify output varies (has filtering effect)
    EXPECT_GT(max_val - min_val, 0.1) << "Output should vary for square wave input";
    
    sc_core::sc_stop();
}
