/**
 * @file test_wave_gen_jitter_config.cpp
 * @brief Unit test for WaveGenerationTdf module - Jitter Parameter Configuration
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenSeedTest, JitterParameterConfiguration) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    params.jitter.RJ_sigma = 5e-12;  // 5 ps RJ
    params.jitter.SJ_freq.push_back(5e6);   // 5 MHz
    params.jitter.SJ_pp.push_back(20e-12);  // 20 ps pp
    
    // Should not throw with jitter parameters
    WaveGenTestbench* tb = new WaveGenTestbench("tb_jitter", params, 80e9, 12345, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 0);
    
    // Output should still be +/-1.0V
    for (const auto& s : samples) {
        EXPECT_TRUE(std::abs(s - 1.0) < 1e-9 || std::abs(s + 1.0) < 1e-9);
    }
    
    sc_core::sc_stop();
}
