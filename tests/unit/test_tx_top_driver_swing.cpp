/**
 * @file test_tx_top_driver_swing.cpp
 * @brief Unit test for TX Top module - Driver output swing verification
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(TxTopDriverTest, OutputSwingAndCommonMode) {
    // TX parameters
    TxParams params;
    params.ffe.taps = {0.0, 1.0, 0.0};  // Pass-through FFE
    params.mux_lane = 0;
    params.driver.dc_gain = 1.0;
    params.driver.vswing = 0.8;  // 800mV swing
    params.driver.vcm_out = 0.5;  // 500mV common mode
    params.driver.sat_mode = "soft";
    params.driver.vlin = 0.5;
    params.driver.poles.clear();
    
    // Square wave input
    TxTopTestbench tb(params, TxSignalSource::SQUARE, 1.0, 5e9);
    
    sc_core::sc_start(400, sc_core::SC_NS);
    
    // Verify output swing
    double pp = tb.monitor->get_pp_diff();
    
    // With vswing=0.8V and voltage division, expect pp < 0.8V
    EXPECT_LT(pp, 0.9);  // Should not exceed vswing
    EXPECT_GT(pp, 0.1);  // Should have significant output
    
    // Verify common mode voltage
    double actual_cm = tb.monitor->get_dc_cm();
    
    // Common mode is affected by impedance matching voltage division
    // vcm_channel = vcm_out * Z0/(Zout+Z0) = 0.5 * 50/(50+50) = 0.25V
    double voltage_div_factor = 50.0 / (params.driver.output_impedance + 50.0);
    double expected_cm = params.driver.vcm_out * voltage_div_factor;
    EXPECT_NEAR(actual_cm, expected_cm, 0.15);
    
    sc_core::sc_stop();
}
