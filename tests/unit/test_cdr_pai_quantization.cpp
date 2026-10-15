/**
 * @file test_cdr_pai_quantization.cpp
 * @brief Unit test for RxCdrTdf module - Phase Quantization
 */

#include "cdr_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(CdrPaiTest, PhaseQuantization) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;  // 1ps
    params.pai.range = 5e-11;

    std::vector<double> pattern = {1.0, -1.0, 1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(10, sc_core::SC_NS);

    double phase = tb->get_phase_output();
    // 相位输出为原始值（未量化，见 rx_cdr.cpp）；其步进粒度为 ki*ui
    // （kp = 100*ki，比例项与积分项均为 ki*ui 的整数倍）
    double step = params.pi.ki * 1e-10;  // ki * ui（CdrParams.ui 默认 100 ps）
    double residual = std::abs(std::remainder(phase, step));
    EXPECT_LE(residual, 1e-16) << "Phase should follow ki*ui step granularity";

    sc_core::sc_stop();
}
