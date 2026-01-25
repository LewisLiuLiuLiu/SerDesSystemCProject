/**
 * @file test_wave_gen_prbs_mode.cpp
 * @brief Unit test for WaveGenerationTdf module - PRBS Mode Verification
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenBasicTest, PRBSModeVerification) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    params.single_pulse = 0.0;  // Explicitly disable pulse mode
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_prbs_mode", params, 80e9, 12345, 100);
    
    // Verify PRBS mode is active
    EXPECT_FALSE(tb->is_pulse_mode()) << "Should be in PRBS mode";
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 0);
    
    // PRBS should have both +1.0 and -1.0 values (not all same)
    int positive_count = 0;
    int negative_count = 0;
    for (const auto& s : samples) {
        if (s > 0) positive_count++;
        else negative_count++;
    }
    EXPECT_GT(positive_count, 0) << "PRBS should have positive values";
    EXPECT_GT(negative_count, 0) << "PRBS should have negative values";
    
    sc_core::sc_stop();
}
