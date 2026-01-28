/**
 * @file test_tx_top_integration.cpp
 * @brief Unit test for TX Top module - Full chain integration test
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

// Single TEST to avoid SystemC-AMS elaboration conflicts
TEST(TxTopIntegrationTest, FullChainPcie) {
    // PCIe Gen3-like configuration
    TxParams params;
    params.ffe.taps = {0.0, 1.0, -0.25};  // 3.5dB de-emphasis
    params.mux_lane = 0;
    params.driver.dc_gain = 0.8;
    params.driver.vswing = 0.8;  // 800mV swing
    params.driver.vcm_out = 0.6;
    params.driver.output_impedance = 50.0;
    params.driver.sat_mode = "soft";
    params.driver.vlin = 0.5;
    params.driver.poles = {40e9};
    
    // PRBS-like input
    TxTopTestbench tb(params, TxSignalSource::PRBS, 1.0, 8e9);  // 8 Gbps
    
    sc_core::sc_start(2000, sc_core::SC_NS);
    
    // Verify complete signal chain
    const auto& samples = tb.get_output_diff();
    ASSERT_GT(samples.size(), 500);
    
    // Check output characteristics
    double pp = tb.monitor->get_pp_diff();
    double cm = tb.monitor->get_dc_cm();
    
    // Output swing should be reasonable
    EXPECT_GT(pp, 0.1);
    EXPECT_LT(pp, 1.0);
    
    // Common mode is affected by impedance matching voltage division
    // vcm_channel = vcm_out * Z0/(Zout+Z0) = 0.6 * 50/(50+50) = 0.3V
    double voltage_div_factor = 50.0 / (params.driver.output_impedance + 50.0);
    double expected_cm = params.driver.vcm_out * voltage_div_factor;
    EXPECT_NEAR(cm, expected_cm, 0.15);
    
    sc_core::sc_stop();
}
