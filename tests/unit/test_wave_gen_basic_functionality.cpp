/**
 * @file test_wave_gen_basic_functionality.cpp
 * @brief Unit test for WaveGenerationTdf module - Basic Functionality
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenBasicTest, BasicFunctionality) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb", params, 80e9, 12345, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    // Verify samples were collected
    EXPECT_GT(tb->get_samples().size(), 0) << "Should have collected samples";
    
    sc_core::sc_stop();
}
