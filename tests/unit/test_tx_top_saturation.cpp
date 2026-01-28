/**
 * @file test_tx_top_saturation.cpp
 * @brief Unit test for TX Top module - Saturation characteristics
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

// Single TEST to avoid SystemC-AMS elaboration conflicts
TEST(TxTopSaturationTest, SaturationBehavior) {
    // TX parameters with soft saturation
    TxParams params;
    params.ffe.taps = {0.0, 1.0, 0.0};
    params.mux_lane = 0;
    params.driver.dc_gain = 2.0;  // High gain to drive into saturation
    params.driver.vswing = 0.8;
    params.driver.vcm_out = 0.6;
    params.driver.sat_mode = "soft";
    params.driver.vlin = 0.5;
    params.driver.poles.clear();
    
    // Large input to drive into saturation
    TxTopTestbench tb(params, TxSignalSource::DC, 1.0);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // With soft saturation, output should be limited
    double pp = tb.monitor->get_pp_diff();
    
    // Output should not exceed vswing significantly
    EXPECT_LT(pp, 1.2);  // Allow some margin
    
    // Also verify DC output is limited (for soft saturation)
    double dc_diff = std::abs(tb.monitor->get_dc_diff());
    EXPECT_LE(dc_diff, params.driver.vswing + 0.2);
    
    sc_core::sc_stop();
}
