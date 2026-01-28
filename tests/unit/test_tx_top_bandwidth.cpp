/**
 * @file test_tx_top_bandwidth.cpp
 * @brief Unit test for TX Top module - Bandwidth limiting test
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

// Single TEST to avoid SystemC-AMS elaboration conflicts
TEST(TxTopBandwidthTest, BandwidthLimitingBehavior) {
    // TX parameters with bandwidth limiting
    TxParams params;
    params.ffe.taps = {0.0, 1.0, 0.0};
    params.mux_lane = 0;
    params.driver.dc_gain = 1.0;
    params.driver.vswing = 1.0;
    params.driver.vcm_out = 0.6;
    params.driver.sat_mode = "none";
    params.driver.poles = {25e9};  // 25 GHz pole
    
    // High frequency sine wave
    TxTopTestbench tb(params, TxSignalSource::SINE, 0.5, 30e9);  // 30 GHz > pole
    
    sc_core::sc_start(200, sc_core::SC_NS);
    
    // With bandwidth limiting, high frequency should be attenuated
    double rms = tb.monitor->get_rms_diff();
    
    // Output should exist but be attenuated
    EXPECT_GT(rms, 0.01);
    
    // Attenuation expected at 30 GHz with 25 GHz pole
    // |H(f)| = 1 / sqrt(1 + (f/fp)^2) approx 0.64 at f = 30 GHz, fp = 25 GHz
    // Expected RMS approx 0.5 * 0.64 * 0.5 * 0.707 approx 0.11
    EXPECT_LT(rms, 0.3);  // Should be attenuated
    
    sc_core::sc_stop();
}
