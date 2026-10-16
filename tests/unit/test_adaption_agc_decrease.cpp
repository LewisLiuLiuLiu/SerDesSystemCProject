/**
 * @file test_adaption_agc_decrease.cpp
 * @brief Unit test for AdaptionDe - AGC decrease direction (position-form PI)
 *
 * Complements adaption_agc_convergence (increase direction): with
 * amplitude ABOVE target, the position-form PI must command a gain
 * decrease from the initial value, staying within [gain_min, gain_max].
 *
 * Single TEST per binary (SystemC elaboration constraint).
 */

#include "adaption_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(AdaptionAgcTest, GainDecreasesWhenAboveTarget) {
    AdaptionParams params;
    params.agc.enabled = true;
    params.agc.initial_gain = 1.0;
    params.agc.target_amplitude = 0.4;
    params.agc.kp = 0.5;
    params.slow_update_period = 1e-7;
    params.dfe.enabled = false;
    params.vref_adapt.enabled = false;
    params.cdr_pi.enabled = false;
    params.safety.freeze_on_error = false;

    AdaptionBasicTestbench* tb = new AdaptionBasicTestbench("tb", params);

    // 幅度 0.6 > 目标 0.4 → 增益应下调
    tb->src->set_amplitude(0.6);

    sc_core::sc_start(1, sc_core::SC_US);

    double gain = tb->get_vga_gain();
    EXPECT_LT(gain, params.agc.initial_gain)
        << "Gain should decrease when amplitude > target";
    EXPECT_GE(gain, params.agc.gain_min) << "Gain should be >= gain_min";
    EXPECT_LE(gain, params.agc.gain_max) << "Gain should be <= gain_max";

    delete tb;
}
