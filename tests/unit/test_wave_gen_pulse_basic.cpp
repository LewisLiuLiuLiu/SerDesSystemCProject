/**
 * @file test_wave_gen_pulse_basic.cpp
 * @brief Unit test for WaveGenerationTdf module - Single Pulse Mode Basic
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenPulseTest, SinglePulseModeBasic) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    params.single_pulse = 100e-12;  // 100 ps pulse
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_pulse", params, 80e9, 12345, 200);
    
    // Verify pulse mode is detected
    EXPECT_TRUE(tb->is_pulse_mode()) << "Should be in pulse mode";
    
    sc_core::sc_start(5, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 0);
    
    // All samples should be +/-1.0V
    for (const auto& s : samples) {
        EXPECT_TRUE(std::abs(s - 1.0) < 1e-9 || std::abs(s + 1.0) < 1e-9);
    }
    
    sc_core::sc_stop();
}
