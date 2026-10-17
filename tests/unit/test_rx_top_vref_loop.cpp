/**
 * @file test_rx_top_vref_loop.cpp
 * @brief RxTopModule Vref 闭环集成测试（批次3）
 *
 * 验证完整的 Vref 闭环链路：
 *   ±Vref 比较器（data 频闪采样）→ DfeAdaptTdf 统计 → AdaptionDe.update_vref_command
 *   → vref_cmd → 比较器动态门限（vref_threshold_de）
 *
 * 场景设计（确定性方向）：
 *   - vref_initial = 0.05V，信号在数据中心幅度约 ±0.5V → P_pos/P_neg ≈ 1
 *     → C ≈ 1.0 > target_confidence 0.7
 *   - DFE 状态机在首个统计周期后进入 ACQUIRE（需 C>0.4 且双极性计数>100）
 *   - ACQUIRE 分支 vref *= (1 + alpha*(C-target)) → vref 单调上移
 *   - 断言：初值为 vref_initial；运行一个 ACQUIRE 窗口后 vref 增大且保持有界
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "rx_top_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(RxTopVrefLoopTest, VrefAdaptsUpwardWhenConfidenceHigh) {
    AdaptionParams ap = get_default_adaption_params();

    // DFE 统计 + Vref 自适应（默认关闭，此处显式开启）
    ap.dfe.enabled = true;
    ap.dfe.num_taps = 3;
    ap.dfe.initial_taps = {0.0, 0.0, 0.0};
    ap.dfe.stats_period = 8192;      // 统计周期 819.2ns（UI=100ps）
    ap.vref_adapt.enabled = true;
    ap.vref_adapt.vref_pos = 0.15;
    ap.vref_adapt.vref_neg = -0.15;
    ap.vref_adapt.vref_initial = 0.05;
    ap.vref_adapt.target_confidence = 0.7;
    ap.vref_adapt.adapt_alpha = 0.05;
    ap.slow_update_period = 5e-8;    // 50ns 慢路径更新

    RxParams params = get_default_rx_params();
    RxTopTestbench tb(params, ap, RxDifferentialSource::PRBS, 0.4, 5e9, 1.0, "v1");

    // 初值：vref_cmd 应为 vref_initial（慢路径 t=0 发布）
    sc_core::sc_start(80, sc_core::SC_NS);
    double vref0 = tb.dut->get_vref_cmd();
    EXPECT_NEAR(vref0, 0.05, 1e-9)
        << "initial vref command must equal vref_initial";

    // 运行到 1.58us：覆盖首个统计周期（819ns）之后的 ACQUIRE 窗口
    sc_core::sc_start(1500, sc_core::SC_NS);
    double vref1 = tb.dut->get_vref_cmd();

    EXPECT_GT(vref1, vref0 + 0.002)
        << "closed loop: vref must increase while C (approx 1.0) > target (0.7)";
    EXPECT_LT(vref1, 0.5) << "vref must stay within the configured clamp range";

    sc_core::sc_stop();
}
