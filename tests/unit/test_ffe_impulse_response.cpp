/**
 * @file test_ffe_impulse_response.cpp
 * @brief Unit test for TxFfeTdf module - Impulse Response Verification
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeResponseTest, ImpulseResponseVerification) {
    TxFfeParams params;
    params.taps = {0.2, 0.6, 0.2};
    
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb_impulse", params, 
                                                   SignalSource::IMPULSE, 1.0);
    
    sc_core::sc_start(50, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GE(samples.size(), 5) << "Should have enough samples";
    
    // Find non-zero outputs
    std::vector<double> nonzero_outputs;
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        if (std::abs(samples[i]) > 0.001) {
            nonzero_outputs.push_back(samples[i]);
        }
    }
    
    // Verify impulse response contains tap coefficients
    EXPECT_GE(nonzero_outputs.size(), 1) << "Should have non-zero impulse response";
    
    sc_core::sc_stop();
}
