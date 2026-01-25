/**
 * @file test_wave_gen_prbs9.cpp
 * @brief Unit test for WaveGenerationTdf module - PRBS Type 9
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenPrbsTest, PRBSType9) {
    WaveGenParams params;
    params.type = PRBSType::PRBS9;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_prbs9", params, 80e9, 12345, 200);
    
    sc_core::sc_start(50, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 0);
    
    for (const auto& s : samples) {
        EXPECT_TRUE(std::abs(s - 1.0) < 1e-9 || std::abs(s + 1.0) < 1e-9);
    }
    
    sc_core::sc_stop();
}
