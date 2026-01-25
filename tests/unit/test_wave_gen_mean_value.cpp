/**
 * @file test_wave_gen_mean_value.cpp
 * @brief Unit test for WaveGenerationTdf module - Mean Value Verification
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenStatisticsTest, MeanValueVerification) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_mean", params, 80e9, 12345, 10000);
    
    sc_core::sc_start(200, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 5000);
    
    // Calculate mean
    double sum = 0.0;
    for (const auto& s : samples) {
        sum += s;
    }
    double mean = sum / samples.size();
    
    // Mean of balanced PRBS should be close to 0
    EXPECT_LT(std::abs(mean), 0.1) << "Mean should be close to 0 for PRBS";
    
    sc_core::sc_stop();
}
