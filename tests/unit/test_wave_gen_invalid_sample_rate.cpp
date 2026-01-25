/**
 * @file test_wave_gen_invalid_sample_rate.cpp
 * @brief Unit test for WaveGenerationTdf module - Invalid Sample Rate
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenValidationTest, InvalidSampleRate) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    // Zero sample rate should throw
    EXPECT_THROW({
        WaveGenerationTdf* wave_gen = new WaveGenerationTdf("wave_gen", params, 0.0, 12345);
        delete wave_gen;
    }, std::invalid_argument) << "Should throw for zero sample rate";
    
    // Negative sample rate should throw
    EXPECT_THROW({
        WaveGenerationTdf* wave_gen = new WaveGenerationTdf("wave_gen", params, -80e9, 12345);
        delete wave_gen;
    }, std::invalid_argument) << "Should throw for negative sample rate";
}
