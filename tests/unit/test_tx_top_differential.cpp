/**
 * @file test_tx_top_differential.cpp
 * @brief Unit test for TX Top module - Differential signal integrity
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(TxTopDifferentialTest, SymmetryAndInversion) {
    // TX parameters
    TxParams params;
    params.ffe.taps = {0.0, 1.0, 0.0};
    params.mux_lane = 0;
    params.driver.dc_gain = 0.8;
    params.driver.vswing = 0.8;
    params.driver.vcm_out = 0.6;
    params.driver.sat_mode = "none";
    params.driver.poles.clear();
    params.driver.imbalance.gain_mismatch = 0.0;  // No mismatch
    params.driver.imbalance.skew = 0.0;
    
    // Square wave input
    TxTopTestbench tb(params, TxSignalSource::SQUARE, 0.5, 5e9);
    
    sc_core::sc_start(400, sc_core::SC_NS);
    
    // Verify differential symmetry
    EXPECT_TRUE(tb.monitor->is_symmetric(0.15));
    
    // Verify that out_p and out_n are inverted (relative to common mode)
    const auto& samples_p = tb.get_output_p();
    const auto& samples_n = tb.get_output_n();
    const auto& samples_cm = tb.monitor->samples_cm;
    
    ASSERT_GT(samples_p.size(), 100);
    
    int inverted_count = 0;
    for (size_t i = 100; i < samples_p.size(); ++i) {
        double p_dev = samples_p[i] - samples_cm[i];
        double n_dev = samples_n[i] - samples_cm[i];
        
        // out_p deviation should be opposite to out_n deviation
        if (p_dev * n_dev < 0) {
            inverted_count++;
        }
    }
    
    // Most samples should show inverted relationship
    double inverted_ratio = static_cast<double>(inverted_count) / (samples_p.size() - 100);
    EXPECT_GT(inverted_ratio, 0.9);
    
    // Calculate differential peak-to-peak (DC is ~0 for symmetric square wave)
    double diff_pp = tb.monitor->get_pp_diff();
    EXPECT_GT(diff_pp, 0.01);
    
    sc_core::sc_stop();
}
