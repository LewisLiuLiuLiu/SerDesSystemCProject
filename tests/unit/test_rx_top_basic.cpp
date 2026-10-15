/**
 * @file test_rx_top_basic.cpp
 * @brief Basic functionality tests for RxTopModule
 * 
 * Tests:
 * - Port connection verification
 * - Signal propagation through chain
 * - Output data validity
 * 
 * NOTE (2026-10): SystemC 每进程仅允许一次精化；所有仿真实例预先构建后
 * 单次 sc_start，再分场景断言。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(RxTopBasicTest, SimulationScenarios) {
    RxParams params = get_default_rx_params();
    AdaptionParams adaption_params = get_default_adaption_params();
    
    // --- 预先构建全部场景（实例前缀唯一化） ---
    RxTopTestbench tb_square(params, adaption_params, RxDifferentialSource::SQUARE, 0.5, 5e9, 1.0, "t1");
    RxTopTestbench tb_dc_pos(params, adaption_params, RxDifferentialSource::DC, 0.5, 10e9, 1.0, "t2");
    RxTopTestbench tb_dc_neg(params, adaption_params, RxDifferentialSource::DC, -0.5, 10e9, 1.0, "t3");
    RxTopTestbench tb_prbs(params, adaption_params, RxDifferentialSource::PRBS, 0.5, 10e9, 1.0, "t4");
    
    // --- 单次统一推进（覆盖最长场景 1000 ns） ---
    sc_core::sc_start(1000, sc_core::SC_NS);
    
    // 场景 1：SQUARE 输入 —— 数据有效性与翻转
    ASSERT_GT(tb_square.get_sample_count(), 0) << "No samples collected";
    EXPECT_TRUE(tb_square.monitor->is_valid_data())
        << "Output should contain both 0s and 1s for square wave input";
    EXPECT_GT(tb_square.monitor->count_transitions(), 0)
        << "Output should have transitions for square wave input";
    
    // 场景 2：正 DC —— 主要为 1
    {
        size_t ones = tb_dc_pos.monitor->count_ones();
        size_t total = tb_dc_pos.get_sample_count();
        ASSERT_GT(total, 0) << "No samples collected";
        EXPECT_GT(ones, total / 2) << "Positive DC input should produce mostly 1s after settling";
    }
    
    // 场景 3：负 DC —— 主要为 0
    {
        size_t zeros = tb_dc_neg.monitor->count_zeros();
        size_t total = tb_dc_neg.get_sample_count();
        ASSERT_GT(total, 0) << "No samples collected";
        EXPECT_GT(zeros, total / 2) << "Negative DC input should produce mostly 0s after settling";
    }
    
    // 场景 4：调试接口可访问 —— CDR 状态有限、内部信号可取
    EXPECT_TRUE(std::isfinite(tb_square.get_cdr_phase())) << "CDR phase should be finite";
    EXPECT_TRUE(std::isfinite(tb_square.get_cdr_integral_state())) << "CDR integral should be finite";
    {
        const auto& ctle_out_p = tb_dc_pos.dut->get_ctle_out_p_signal();
        const auto& vga_out_p = tb_dc_pos.dut->get_vga_out_p_signal();
        const auto& dfe_out_p = tb_dc_pos.dut->get_dfe_out_p_signal();
        (void)ctle_out_p;
        (void)vga_out_p;
        (void)dfe_out_p;
    }
    
    // 场景 5：PRBS —— 均衡性与翻转
    {
        ASSERT_GT(tb_prbs.get_sample_count(), 0) << "No samples collected";
        size_t ones = tb_prbs.monitor->count_ones();
        size_t total = tb_prbs.get_sample_count();
        double one_ratio = static_cast<double>(ones) / total;
        EXPECT_GT(one_ratio, 0.3) << "PRBS should have significant number of 1s";
        EXPECT_LT(one_ratio, 0.7) << "PRBS should have significant number of 0s";
        // 阈值基于实测行为标定（源按逐采样点随机 + 链路低通，翻转密度约 3-7%）
        EXPECT_GT(tb_prbs.monitor->count_transitions(), total / 50)
            << "PRBS should have many transitions";
    }
    
    sc_core::sc_stop();
}
