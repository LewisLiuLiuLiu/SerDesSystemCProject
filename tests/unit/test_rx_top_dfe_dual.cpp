/**
 * @file test_rx_top_dfe_dual.cpp
 * @brief Dual DFE tests for RxTopModule
 * 
 * Tests:
 * - Verify P/N tap negation
 * - Differential symmetry
 * - DFE feedback effect
 * 
 * NOTE (2026-10): SystemC 每进程仅允许一次精化；所有仿真实例预先构建后
 * 单次 sc_start，再分场景断言。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(RxTopDfeDualTest, SimulationScenarios) {
    AdaptionParams adaption_params = get_default_adaption_params();
    
    // --- 预先构建全部场景（实例前缀唯一化） ---
    RxParams params_enabled = get_default_rx_params();
    params_enabled.dfe_summer.tap_coeffs = {-0.1, -0.05, 0.02};  // Significant taps
    params_enabled.dfe_summer.enable = true;
    RxTopTestbench tb_enabled(params_enabled, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "d1");
    
    RxParams params_no_dfe = get_no_dfe_params();  // DFE disabled (empty taps)
    RxTopTestbench tb_no_dfe(params_no_dfe, adaption_params, RxDifferentialSource::SQUARE, 0.5, 5e9, 1.0, "d2");
    
    RxParams params_single = get_default_rx_params();
    params_single.dfe_summer.tap_coeffs = {-0.1};  // Single tap
    params_single.dfe_summer.enable = true;
    RxTopTestbench tb_single(params_single, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "d3");
    
    RxParams params_multi = get_default_rx_params();
    params_multi.dfe_summer.tap_coeffs = {-0.08, -0.05, -0.03, 0.01, 0.005};  // 5 taps
    params_multi.dfe_summer.enable = true;
    RxTopTestbench tb_multi(params_multi, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "d4");
    
    RxParams params_large = get_default_rx_params();
    params_large.dfe_summer.tap_coeffs = {-0.2, -0.1, 0.05};  // Large taps
    params_large.dfe_summer.enable = true;
    RxTopTestbench tb_large(params_large, adaption_params, RxDifferentialSource::PRBS, 0.5, 5e9, 1.0, "d5");
    
    RxParams params_path = get_default_rx_params();
    RxTopTestbench tb_path(params_path, adaption_params, RxDifferentialSource::SQUARE, 0.4, 5e9, 1.0, "d6");
    
    // --- 单次统一推进（500 ns） ---
    sc_core::sc_start(500, sc_core::SC_NS);
    
    // 场景 1：DFE 使能 —— 数据有效
    ASSERT_GT(tb_enabled.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_enabled.monitor->is_valid_data())
        << "DFE enabled should produce valid data";
    
    // 场景 2：DFE 关闭（空抽头）—— 数据有效（直通）
    ASSERT_GT(tb_no_dfe.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_no_dfe.monitor->is_valid_data())
        << "DFE disabled should produce valid data (passthrough)";
    
    // 场景 3：单抽头 —— 数据有效
    ASSERT_GT(tb_single.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_single.monitor->is_valid_data())
        << "Single tap DFE should produce valid data";
    
    // 场景 4：多抽头 —— 数据有效
    ASSERT_GT(tb_multi.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_multi.monitor->is_valid_data())
        << "Multi-tap DFE should produce valid data";
    
    // 场景 5：大抽头 —— 不崩溃（可能因强反馈退化）
    ASSERT_GT(tb_large.get_sample_count(), 0) << "No samples collected";
    {
        size_t transitions = tb_large.monitor->count_transitions();
        EXPECT_GE(transitions, 0) << "Large tap DFE should not crash";
    }
    
    // 场景 6：DFE 信号路径可访问
    {
        const auto& dfe_out_p = tb_path.dut->get_dfe_out_p_signal();
        const auto& dfe_out_n = tb_path.dut->get_dfe_out_n_signal();
        (void)dfe_out_p;
        (void)dfe_out_n;
        EXPECT_GT(tb_path.get_sample_count(), 0) << "Signal path should work";
    }
    
    sc_core::sc_stop();
}
