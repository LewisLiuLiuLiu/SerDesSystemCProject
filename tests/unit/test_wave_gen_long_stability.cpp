/**
 * @file test_wave_gen_long_stability.cpp
 * @brief Unit test for WaveGenerationTdf module - Long Simulation Stability
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenStatisticsTest, LongSimulationStability) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_long", params, 80e9, 12345, 50000);
    
    // Run for 1 microsecond
    sc_core::sc_start(1, sc_core::SC_US);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 40000) << "Should have many samples after 1us";
    
    // Verify no NaN or Inf values
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_FALSE(std::isnan(samples[i])) << "Sample " << i << " is NaN";
        EXPECT_FALSE(std::isinf(samples[i])) << "Sample " << i << " is Inf";
    }
    
    sc_core::sc_stop();
}
