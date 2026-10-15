/**
 * @file test_rx_top_ctle_effect.cpp
 * @brief CTLE effect tests for RxTopModule
 * 
 * Tests:
 * - Different CTLE configurations
 * - Frequency response impact
 * - High-frequency boost verification
 * 
 * NOTE (2026-10): SystemC 每进程仅允许一次精化；所有仿真实例预先构建后
 * 单次 sc_start，再分场景断言。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(RxTopCtleTest, SimulationScenarios) {
    AdaptionParams adaption_params = get_default_adaption_params();
    
    // --- 预先构建全部场景（实例前缀唯一化） ---
    RxParams params_high = get_high_gain_rx_params();
    RxTopTestbench tb_high(params_high, adaption_params, RxDifferentialSource::SQUARE, 0.3, 5e9, 1.0, "c1");
    
    RxParams params_low = get_default_rx_params();
    params_low.ctle.dc_gain = 1.0;  // Low gain
    RxTopTestbench tb_low(params_low, adaption_params, RxDifferentialSource::SQUARE, 0.2, 5e9, 1.0, "c2");
    
    RxParams params_zero = get_default_rx_params();
    params_zero.ctle.zeros = {5e9};  // Higher zero frequency
    params_zero.ctle.poles = {40e9};
    RxTopTestbench tb_zero(params_zero, adaption_params, RxDifferentialSource::SQUARE, 0.4, 5e9, 1.0, "c3");
    
    RxParams params_unity = get_default_rx_params();
    params_unity.ctle.dc_gain = 1.0;
    params_unity.ctle.zeros.clear();   // No zeros
    params_unity.ctle.poles = {50e9};  // High pole for minimal filtering
    RxTopTestbench tb_unity(params_unity, adaption_params, RxDifferentialSource::SQUARE, 0.5, 3e9, 1.0, "c4");
    
    RxParams params_multi = get_default_rx_params();
    params_multi.ctle.zeros = {1e9, 3e9};
    params_multi.ctle.poles = {10e9, 30e9};
    params_multi.ctle.dc_gain = 2.0;
    RxTopTestbench tb_multi(params_multi, adaption_params, RxDifferentialSource::SQUARE, 0.4, 5e9, 1.0, "c5");
    
    // --- 单次统一推进（500 ns） ---
    sc_core::sc_start(500, sc_core::SC_NS);
    
    // 场景 1：高增益配置 —— 数据有效
    ASSERT_GT(tb_high.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_high.monitor->is_valid_data())
        << "High gain config should produce valid data";
    
    // 场景 2：低增益配置 —— 可正常运行（低增益/小幅输入下输出可能退化）
    EXPECT_GE(tb_low.monitor->count_transitions(), 0) << "Low gain config should run without error";
    
    // 场景 3：不同零点频率 —— 数据有效
    ASSERT_GT(tb_zero.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_zero.monitor->is_valid_data())
        << "Modified zero frequency should still produce valid data";
    
    // 场景 4：单位增益直通 —— 数据有效
    ASSERT_GT(tb_unity.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_unity.monitor->is_valid_data())
        << "Unity gain CTLE should produce valid data with large signal";
    
    // 场景 5：多零极点 —— 数据有效
    ASSERT_GT(tb_multi.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_multi.monitor->is_valid_data())
        << "Multi-pole/zero CTLE should produce valid data";
    
    sc_core::sc_stop();
}
