/**
 * @file test_wave_gen_pulse_timing.cpp
 * @brief Unit test for WaveGenerationTdf module - Single Pulse Mode Timing
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenPulseTest, SinglePulseModeTiming) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    params.single_pulse = 100e-12;  // 100 ps pulse
    
    double sample_rate = 80e9;
    double timestep = 1.0 / sample_rate;  // 12.5 ps
    int pulse_samples = static_cast<int>(params.single_pulse / timestep);  // 8 samples
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_pulse_timing", params, sample_rate, 12345, 100);
    
    sc_core::sc_start(2, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GE(samples.size(), static_cast<size_t>(pulse_samples + 5));
    
    // First 'pulse_samples' should be +1.0 (high)
    // Note: Due to floating-point timing, we check strictly within pulse duration
    for (int i = 0; i < pulse_samples && i < static_cast<int>(samples.size()); ++i) {
        EXPECT_NEAR(samples[i], 1.0, 1e-9) 
            << "Sample " << i << " during pulse should be +1.0V";
    }
    
    // After pulse, should be -1.0 (low)
    // Note: Skip boundary sample (pulse_samples) due to floating-point comparison
    // in the implementation (m_time < single_pulse). Start checking from pulse_samples + 1
    // to avoid false failures from floating-point accumulation errors.
    for (size_t i = pulse_samples + 1; i < samples.size(); ++i) {
        EXPECT_NEAR(samples[i], -1.0, 1e-9)
            << "Sample " << i << " after pulse should be -1.0V";
    }
    
    sc_core::sc_stop();
}
