/**
 * @file test_wave_gen_debug_lfsr.cpp
 * @brief Unit test for WaveGenerationTdf module - Debug Interface LFSR State
 */

#include "wave_generation_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(WaveGenDebugTest, DebugInterfaceLFSRState) {
    WaveGenParams params;
    params.type = PRBSType::PRBS31;
    
    WaveGenTestbench* tb = new WaveGenTestbench("tb_debug", params, 80e9, 12345, 100);
    
    sc_core::sc_start(10, sc_core::SC_NS);
    
    unsigned int lfsr_state = tb->get_lfsr_state();
    EXPECT_NE(lfsr_state, 0u) << "LFSR state should not be zero";
    
    sc_core::sc_stop();
}
