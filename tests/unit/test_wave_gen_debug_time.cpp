/**
 * @file test_wave_gen_debug_time.cpp
 * @brief Unit test for WaveGenerationTdf module - Debug Interface Current Time
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenDebugTest, DebugInterfaceCurrentTime) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_time", params, 80e9, 12345, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    double current_time = tb->get_current_time();
    EXPECT_GT(current_time, 0.0) << "Current time should be positive after simulation";
    
    sc_core::sc_stop();
}
