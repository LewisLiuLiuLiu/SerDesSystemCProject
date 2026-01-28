/**
 * @file test_tx_top_wavegen_output.cpp
 * @brief Unit test for TX Top module - WaveGen integration test
 */

#include "tx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

// Single TEST to avoid SystemC-AMS elaboration conflicts
TEST(TxTopWaveGenTest, PrbsOutputBehavior) {
    // WaveGen parameters
    WaveGenParams wave_params;
    wave_params.type = PRBSType::PRBS7;
    wave_params.single_pulse = 0.0;  // PRBS mode
    
    // TX parameters
    TxParams tx_params;
    tx_params.ffe.taps = {0.0, 1.0, -0.2};  // De-emphasis
    tx_params.mux_lane = 0;
    tx_params.driver.dc_gain = 1.0;
    tx_params.driver.vswing = 0.8;
    tx_params.driver.vcm_out = 0.6;
    tx_params.driver.sat_mode = "soft";
    tx_params.driver.vlin = 0.5;
    tx_params.driver.poles = {50e9};
    
    TxTopTestbenchWithWaveGen tb(wave_params, tx_params, 100e9, 12345);
    
    sc_core::sc_start(500, sc_core::SC_NS);
    
    // Verify output has PRBS characteristics
    const auto& diff_samples = tb.monitor->samples_diff;
    ASSERT_GT(diff_samples.size(), 100);
    
    // Check for transitions (PRBS should have many transitions)
    int transitions = 0;
    for (size_t i = 101; i < diff_samples.size(); ++i) {
        if (std::abs(diff_samples[i] - diff_samples[i-1]) > 0.05) {
            transitions++;
        }
    }
    
    // PRBS should have significant number of transitions
    double transition_rate = static_cast<double>(transitions) / (diff_samples.size() - 100);
    EXPECT_GT(transition_rate, 0.01);
    
    // Check output swing
    double pp = tb.monitor->get_pp_diff();
    EXPECT_GT(pp, 0.1);
    EXPECT_LT(pp, 1.2);  // Should not exceed vswing significantly
    
    sc_core::sc_stop();
}
