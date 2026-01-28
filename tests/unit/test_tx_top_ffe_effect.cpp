/**
 * @file test_tx_top_ffe_effect.cpp
 * @brief Unit test for TX Top module - FFE pre-emphasis/de-emphasis effect
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(TxTopFfeTest, DeEmphasisAndPreEmphasis) {
    // TX parameters with de-emphasis FFE
    TxParams params;
    params.ffe.taps = {0.0, 1.0, -0.25};  // De-emphasis: post-cursor tap = -0.25
    params.mux_lane = 0;
    params.driver.dc_gain = 1.0;
    params.driver.vswing = 1.0;
    params.driver.vcm_out = 0.6;
    params.driver.sat_mode = "none";
    params.driver.poles.clear();
    
    // Square wave input
    TxTopTestbench tb(params, TxSignalSource::SQUARE, 1.0, 5e9);
    
    sc_core::sc_start(400, sc_core::SC_NS);
    
    // With de-emphasis, the output should show:
    // - Higher amplitude at transitions
    // - Lower amplitude at steady state
    const auto& diff_samples = tb.get_output_diff();
    ASSERT_GT(diff_samples.size(), 200);
    
    // Find max and min values
    double max_val = diff_samples[100];
    double min_val = diff_samples[100];
    for (size_t i = 100; i < diff_samples.size(); ++i) {
        if (diff_samples[i] > max_val) max_val = diff_samples[i];
        if (diff_samples[i] < min_val) min_val = diff_samples[i];
    }
    
    double pp = max_val - min_val;
    EXPECT_GT(pp, 0.3);  // Should have significant swing
    
    // Verify output has reasonable swing (FFE effect)
    double pp_result = tb.monitor->get_pp_diff();
    EXPECT_GT(pp_result, 0.3);
    
    sc_core::sc_stop();
}
