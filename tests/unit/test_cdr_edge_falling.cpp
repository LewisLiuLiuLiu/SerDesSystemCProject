/**
 * @file test_cdr_edge_falling.cpp
 * @brief Unit test for RxCdrTdf module - Falling Edge Detection
 */

#include "cdr_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(CdrEdgeTest, FallingEdgeDetection) {
    CdrParams params;
    params.pi.kp = 0.01;
    params.pi.ki = 1e-4;
    params.pai.resolution = 1e-12;
    params.pai.range = 5e-11;

    // Falling edge: 1.0 -> -1.0
    std::vector<double> pattern = {1.0, -1.0, -1.0, -1.0};
    CdrBasicTestbench* tb = new CdrBasicTestbench("tb", params, pattern);

    sc_core::sc_start(5, sc_core::SC_NS);

    // 注：BBPD 相位误差的符号由亚 UI 采样关系决定，而非边沿极性本身；
    // 此处验证下降沿被检测到（相位发生调整）且幅度在范围内。
    double phase = tb->get_phase_output();
    EXPECT_TRUE(std::isfinite(phase));
    EXPECT_LE(std::abs(phase), params.pai.range);
    EXPECT_GT(std::abs(phase), 0.0) << "Falling edge should trigger a phase adjustment";

    sc_core::sc_stop();
}
