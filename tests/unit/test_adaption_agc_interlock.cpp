/**
 * @file test_adaption_agc_interlock.cpp
 * @brief Unit test for AdaptionDe - AGC saturation interlock (v1)
 *
 * When the VGA output approaches its clamp level, gain increases can no
 * longer raise the amplitude. The interlock must:
 * - block gain INCREASE while gain_saturated is asserted
 * - resume normal PI control once the flag is released
 *
 * Single TEST per binary (SystemC elaboration constraint).
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionAgcTest, SaturationInterlockBlocksIncrease) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.agc.initial_gain = 2.0;
    params.agc.target_amplitude = 0.4;
    params.slow_update_period = 1e-7;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    params.safety.freeze_on_error = false;  // 隔离冻结机制

    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);

    // 幅度 0.2 < 目标 0.4（PI 想上调），同时拉高饱和互锁
    tb->src->set_amplitude(0.2);
    tb->set_gain_saturated(true);

    sc_core::sc_start(1, sc_core::SC_US);

    // 互锁生效：增益必须精确保持在初值（正误差被冻结）
    double gain_locked = tb->get_vga_gain();
    EXPECT_NEAR(gain_locked, params.agc.initial_gain, 1e-6)
        << "Saturation interlock must block gain increase";

    // 解除互锁（信号退出饱和）→ PI 恢复，增益重新上调
    tb->set_gain_saturated(false);
    sc_core::sc_start(2, sc_core::SC_US);

    double gain_released = tb->get_vga_gain();
    EXPECT_GT(gain_released, params.agc.initial_gain)
        << "Gain must resume increasing after interlock release";

    delete tb;
}
