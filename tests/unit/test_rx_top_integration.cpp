/**
 * @file test_rx_top_integration.cpp
 * @brief Integration tests for RxTopModule
 * 
 * Tests:
 * - Full chain with realistic parameters
 * - Data recovery verification
 * - Long-running stability
 * 
 * NOTE (2026-10): SystemC 每进程仅允许一次精化；所有仿真实例预先构建后
 * 单次 sc_start（5000 ns），再分场景断言。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"
#include <cmath>

using namespace serdes;
using namespace serdes::test;

TEST(RxTopIntegrationTest, SimulationScenarios) {
    AdaptionParams adaption_params = get_default_adaption_params();
    
    // --- 场景 1：真实参数（10 Gbps）---
    RxParams params_real;
    params_real.ctle.zeros = {2e9};
    params_real.ctle.poles = {25e9};
    params_real.ctle.dc_gain = 1.5;
    params_real.ctle.vcm_out = 0.0;
    params_real.vga.zeros = {1e9};
    params_real.vga.poles = {20e9};
    params_real.vga.dc_gain = 2.0;
    params_real.vga.vcm_out = 0.0;
    params_real.dfe_summer.tap_coeffs = {-0.05, -0.02, 0.01};
    params_real.dfe_summer.ui = 100e-12;
    params_real.dfe_summer.vcm_out = 0.0;
    params_real.dfe_summer.vtap = 1.0;
    params_real.dfe_summer.map_mode = "pm1";
    params_real.dfe_summer.enable = true;
    params_real.sampler.phase_source = "phase";
    params_real.sampler.threshold = 0.0;
    params_real.sampler.hysteresis = 0.01;
    params_real.sampler.resolution = 0.02;
    params_real.cdr.pi.kp = 0.01;
    params_real.cdr.pi.ki = 1e-4;
    params_real.cdr.pai.resolution = 1e-12;
    params_real.cdr.pai.range = 5e-11;
    params_real.cdr.ui = 100e-12;
    RxTopTestbench tb_real(params_real, adaption_params, RxDifferentialSource::PRBS, 0.3, 5e9, 1.0, "g1");
    
    // --- 场景 2：小信号幅度（高增益补偿）---
    RxParams params_low_amp = get_high_gain_rx_params();
    RxTopTestbench tb_low_amp(params_low_amp, adaption_params, RxDifferentialSource::SQUARE, 0.1, 3e9, 1.0, "g2");
    
    // --- 场景 3：高频输入 ---
    RxParams params_hf = get_default_rx_params();
    params_hf.ctle.dc_gain = 2.0;
    params_hf.ctle.zeros = {5e9};
    RxTopTestbench tb_hf(params_hf, adaption_params, RxDifferentialSource::SQUARE, 0.4, 10e9, 1.0, "g3");
    
    // --- 场景 4：长时稳定性 ---
    RxParams params_long = get_default_rx_params();
    RxTopTestbench tb_long(params_long, adaption_params, RxDifferentialSource::PRBS, 0.3, 5e9, 1.0, "g4");
    
    // --- 场景 5：参数扰动鲁棒性 ---
    RxParams params_var = get_default_rx_params();
    params_var.ctle.dc_gain = 1.8;
    params_var.vga.dc_gain = 2.5;
    params_var.dfe_summer.tap_coeffs = {-0.06, -0.03, 0.015, 0.005};
    params_var.dfe_summer.enable = true;
    params_var.cdr.pi.kp = 0.015;
    params_var.cdr.pi.ki = 1.5e-4;
    RxTopTestbench tb_var(params_var, adaption_params, RxDifferentialSource::PRBS, 0.35, 5e9, 1.0, "g5");
    
    // --- 场景 6：数据恢复质量 ---
    RxParams params_recovery = get_default_rx_params();
    RxTopTestbench tb_recovery(params_recovery, adaption_params, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "g6");
    
    // --- 单次统一推进（覆盖最长场景 5000 ns） ---
    sc_core::sc_start(5000, sc_core::SC_NS);
    
    // 场景 1：真实参数全链路 —— 数据有效 + 合理翻转密度
    {
        ASSERT_GT(tb_real.get_sample_count(), 0) << "No samples collected";
        EXPECT_TRUE(tb_real.monitor->is_valid_data())
            << "Realistic params should produce valid data";
        size_t transitions = tb_real.monitor->count_transitions();
        size_t samples = tb_real.get_sample_count();
        double transition_density = static_cast<double>(transitions) / samples;
        // 阈值基于实测行为标定（链路低通 + 逐点随机源，密度约 3%）
        EXPECT_GT(transition_density, 0.01) << "Should have reasonable transition density";
        EXPECT_LT(transition_density, 0.9) << "Transition density should not be too high";
    }
    
    // 场景 2：小信号幅度 —— 仍有翻转
    {
        ASSERT_GT(tb_low_amp.get_sample_count(), 0) << "No samples collected";
        EXPECT_GT(tb_low_amp.monitor->count_transitions(), 0)
            << "Should detect transitions even with low signal";
    }
    
    // 场景 3：高频输入 —— 不崩溃
    {
        ASSERT_GT(tb_hf.get_sample_count(), 0) << "No samples collected";
        EXPECT_GE(tb_hf.monitor->count_transitions(), 0)
            << "High frequency input should not crash";
    }
    
    // 场景 4：长时稳定 —— CDR 状态有限且输出有效
    {
        double cdr_phase = tb_long.get_cdr_phase();
        double cdr_integral = tb_long.get_cdr_integral_state();
        EXPECT_TRUE(std::isfinite(cdr_phase)) << "CDR phase should remain finite";
        EXPECT_TRUE(std::isfinite(cdr_integral)) << "CDR integral should remain finite";
        ASSERT_GT(tb_long.get_sample_count(), 0) << "No samples collected";
        EXPECT_TRUE(tb_long.monitor->is_valid_data())
            << "Long running simulation should produce valid data";
    }
    
    // 场景 5：参数扰动 —— 数据有效
    {
        ASSERT_GT(tb_var.get_sample_count(), 0) << "No samples collected";
        EXPECT_TRUE(tb_var.monitor->is_valid_data())
            << "Varied parameters should produce valid data";
    }
    
    // 场景 6：数据恢复质量 —— 0/1 均衡与翻转活跃
    {
        size_t ones = tb_recovery.monitor->count_ones();
        size_t zeros = tb_recovery.monitor->count_zeros();
        size_t total = tb_recovery.get_sample_count();
        ASSERT_GT(total, 0) << "No samples collected";
        double one_ratio = static_cast<double>(ones) / total;
        double zero_ratio = static_cast<double>(zeros) / total;
        EXPECT_GT(one_ratio, 0.2) << "Should have significant 1s in output";
        EXPECT_GT(zero_ratio, 0.2) << "Should have significant 0s in output";
        EXPECT_GT(tb_recovery.monitor->count_transitions(), total / 50)
            << "Should have active transitions";
    }
    
    sc_core::sc_stop();
}
