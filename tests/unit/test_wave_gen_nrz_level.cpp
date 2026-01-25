/**
 * @file test_wave_gen_nrz_level.cpp
 * @brief Unit test for WaveGenerationTdf module - NRZ Level Verification
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenBasicTest, NRZLevelVerification) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb", params, 80e9, 12345, 500);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_samples();
    EXPECT_GT(samples.size(), 0);
    
    // Verify all samples are either +1.0 or -1.0
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_TRUE(std::abs(samples[i] - 1.0) < 1e-9 || 
                    std::abs(samples[i] + 1.0) < 1e-9)
            << "Sample " << i << " = " << samples[i] << " is not +/-1.0V";
    }
    
    sc_core::sc_stop();
}
