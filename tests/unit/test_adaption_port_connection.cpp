/**
 * @file test_adaption_port_connection.cpp
 * @brief Unit test for AdaptionDe - Port Connection Verification
 */

#include "adaption_test_common.h"
#include <cmath>

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionBasicTest, PortConnection) {
    AdaptionParams params;
    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);
    
    // Run short simulation
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // Outputs must be published with valid values after the first update
    EXPECT_GE(tb->get_vga_gain(), tb->params.agc.gain_min);
    EXPECT_LE(tb->get_vga_gain(), tb->params.agc.gain_max);
    EXPECT_DOUBLE_EQ(tb->get_threshold(), 0.0);
    EXPECT_FALSE(tb->get_freeze_flag());
    EXPECT_TRUE(std::isfinite(tb->get_phase_cmd()));
    EXPECT_LE(std::abs(tb->get_phase_cmd()), tb->params.cdr_pi.phase_range + 1e-18);
    
    // Clean up: stop and delete
    delete tb;
}
