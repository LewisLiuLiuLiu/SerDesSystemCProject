/**
 * @file test_tx_top_basic.cpp
 * @brief Unit test for TX Top module - Basic functionality and port connection
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(TxTopBasicTest, PortConnectionAndSignalFlow) {
    // Default TX parameters
    TxParams params;
    params.ffe.taps = {0.0, 1.0, 0.0};  // Pass-through FFE
    params.mux_lane = 0;
    params.driver.dc_gain = 1.0;
    params.driver.vswing = 0.8;
    params.driver.vcm_out = 0.6;
    params.driver.sat_mode = "none";
    params.driver.poles.clear();  // Disable bandwidth filtering
    
    // Step input to test signal flow
    double input_amplitude = 0.5;
    
    TxTopTestbench tb(params, TxSignalSource::STEP, input_amplitude, 10e9, 1.0);
    
    sc_core::sc_start(200, sc_core::SC_NS);
    
    // Verify output exists and is non-zero
    const auto& diff_samples = tb.get_output_diff();
    ASSERT_GT(diff_samples.size(), 0);
    
    // Check that output has reasonable values
    double dc_output = tb.monitor->get_dc_diff();
    EXPECT_NE(dc_output, 0.0);
    
    // Find transition in output (signal flow verification)
    bool found_transition = false;
    for (size_t i = 1; i < diff_samples.size(); ++i) {
        if (std::abs(diff_samples[i] - diff_samples[i-1]) > 0.05) {
            found_transition = true;
            break;
        }
    }
    EXPECT_TRUE(found_transition);
    
    sc_core::sc_stop();
}
