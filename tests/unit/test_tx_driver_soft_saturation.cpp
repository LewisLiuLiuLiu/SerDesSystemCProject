/**
 * @file test_tx_driver_soft_saturation.cpp
 * @brief Unit test for TX Driver module - Soft Saturation (tanh)
 */

#include "tx_driver_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(TxDriverSaturationTest, SoftSaturationTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;  // Channel-side Vpp → soft limit Vsat·vdf = 0.4V at channel side
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "soft";
    params.vlin = 0.4;  // Deprecated/ignored — verifies unit-gain behavior is vlin-independent
    params.poles.clear();
    
    // Large input that should saturate
    double input_diff = 2.0;  // Much larger than vswing
    
    TxDriverTestbench tb(params, TxDriverDifferentialSource::DC, input_diff);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // With unit-gain soft saturation, the channel-side differential
    // approaches but never exceeds vswing/2 = 0.4V peak.
    double actual_diff = std::abs(tb.monitor->get_dc_diff());
    double max_output = params.vswing / 2.0;  // 0.4V channel-side limit
    
    // Should be close to max but slightly less (tanh never reaches 1)
    EXPECT_LT(actual_diff, max_output);
    EXPECT_GT(actual_diff, max_output * 0.95);
    
    sc_core::sc_stop();
}
