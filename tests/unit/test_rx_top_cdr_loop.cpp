/**
 * @file test_rx_top_cdr_loop.cpp
 * @brief CDR closed-loop tests for RxTopModule
 * 
 * Tests:
 * - Phase tracking behavior
 * - Lock acquisition
 * - CDR state evolution
 * 
 * NOTE (2026-10): SystemC 每进程仅允许一次精化；所有仿真实例预先构建后
 * 单次 sc_start，再分场景断言。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"
#include <cmath>

using namespace serdes;
using namespace serdes::test;

TEST(RxTopCdrLoopTest, SimulationScenarios) {
    AdaptionParams adaption_params = get_default_adaption_params();
    
    // --- 预先构建全部场景（实例前缀唯一化） ---
    RxParams params_track = get_default_rx_params();
    RxTopTestbench tb_track(params_track, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "k1");
    
    RxParams params_integral = get_default_rx_params();
    RxTopTestbench tb_integral(params_integral, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "k2");
    
    RxParams params_aggr = get_aggressive_cdr_params();
    RxTopTestbench tb_aggr(params_aggr, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "k3");
    
    RxParams params_cons = get_default_rx_params();
    params_cons.cdr.pi.kp = 0.001;   // 10x smaller
    params_cons.cdr.pi.ki = 1e-5;    // 10x smaller
    RxTopTestbench tb_cons(params_cons, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "k4");
    
    RxParams params_square = get_default_rx_params();
    RxTopTestbench tb_square(params_square, adaption_params, RxDifferentialSource::SQUARE, 0.4, 5e9, 1.0, "k5");
    
    RxParams params_limit = get_default_rx_params();
    params_limit.cdr.pai.range = 1e-11;  // Very small range
    RxTopTestbench tb_limit(params_limit, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "k6");
    
    // 积分状态初值（仿真前读取）
    double initial_integral = tb_integral.get_cdr_integral_state();
    
    // --- 单次统一推进（1000 ns） ---
    sc_core::sc_start(1000, sc_core::SC_NS);
    
    // 场景 1：CDR 相位有限且在合理范围内
    {
        double cdr_phase = tb_track.get_cdr_phase();
        EXPECT_TRUE(std::isfinite(cdr_phase)) << "CDR phase should be finite";
        EXPECT_LE(std::abs(cdr_phase), params_track.cdr.pai.range * 2)
            << "CDR phase should be within reasonable range";
    }
    
    // 场景 2：积分状态演化（有限性）
    {
        double final_integral = tb_integral.get_cdr_integral_state();
        EXPECT_TRUE(std::isfinite(initial_integral)) << "Initial integral should be finite";
        EXPECT_TRUE(std::isfinite(final_integral)) << "Final integral should be finite";
    }
    
    // 场景 3：激进 CDR 增益 —— 数据有效
    ASSERT_GT(tb_aggr.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_aggr.monitor->is_valid_data())
        << "Aggressive CDR should produce valid data";
    
    // 场景 4：保守 CDR 增益 —— 相位有限
    EXPECT_TRUE(std::isfinite(tb_cons.get_cdr_phase())) << "CDR phase should be finite";
    
    // 场景 5：方波输入 —— 有效数据与翻转
    ASSERT_GT(tb_square.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_square.monitor->is_valid_data())
        << "Square wave should produce valid output";
    EXPECT_GT(tb_square.monitor->count_transitions(), 0)
        << "Square wave should produce transitions";
    
    // 场景 6：相位范围限制
    {
        double cdr_phase = tb_limit.get_cdr_phase();
        EXPECT_LE(std::abs(cdr_phase), params_limit.cdr.pai.range * 1.1)
            << "CDR phase should be limited by PAI range";
    }
    
    sc_core::sc_stop();
}
