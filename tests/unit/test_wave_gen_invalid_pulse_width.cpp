/**
 * @file test_wave_gen_invalid_pulse_width.cpp
 * @brief Unit test for WaveGenerationTdf module - Invalid Pulse Width
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenValidationTest, InvalidPulseWidth) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    params.single_pulse = -100e-12;  // Negative pulse width
    
    EXPECT_THROW({
        WaveGenerationTdf* wave_gen = new WaveGenerationTdf("wave_gen", params, 80e9, 12345);
        delete wave_gen;
    }, std::invalid_argument) << "Should throw for negative pulse width";
}
