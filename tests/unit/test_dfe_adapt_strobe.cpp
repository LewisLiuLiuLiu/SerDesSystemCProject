/**
 * @file test_dfe_adapt_strobe.cpp
 * @brief DfeAdaptTdf 真实数据频闪时基 + 单控制器 Vref 镜像回归（P0-3）
 *
 * 旧实现按硬编码 100ps UI 自建内部触发（且为函数级 static，多实例互相
 * 污染），忽略 CDR 的真实采样频闪。新实现完全由 sampling_trigger
 * （CDR data_trigger，每 UI 1 个）驱动。
 *
 * 本测试令频闪周期 = 70ps（故意 ≠ 100ps），运行 700ns：
 *   - 期望 tap 更新/计数 = 700ns / 70ps = 10000 次（±2）
 *   - 旧实现会得到 ~7000 次（按 100ps 计），可判别
 *   - vref_current 应镜像 AdaptionDe 下发的 vref_cmd（单控制器）
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/dfe_adapt_tdf.h"
#include <cmath>

using namespace serdes;

namespace {

constexpr double kTs = 10e-12;
constexpr int kStrobePeriodSteps = 7;  // 70 ps

class ConstSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    ConstSource(sc_core::sc_module_name nm, double v)
        : sca_tdf::sca_module(nm), out("out"), m_v(v) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out.write(m_v); }
private:
    double m_v;
};

class PeriodicStrobe : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> out;
    PeriodicStrobe(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out"), m_step(0) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        out.write((m_step % kStrobePeriodSteps) == 3);
        ++m_step;
    }
private:
    long long m_step;
};

} // namespace

TEST(DfeAdaptStrobeTest, UsesRealStrobeAndMirrorsVrefCmd) {
    AdaptionParams::DfeAdaptParams dfe;
    dfe.enabled = true;
    dfe.num_taps = 3;
    dfe.mu_startup = 0.001;
    dfe.tap_min = -2.0;
    dfe.tap_max = 2.0;
    dfe.stats_period = 100;
    dfe.initial_taps = {0.0, 0.0, 0.0};

    AdaptionParams::VrefAdaptParams vref;
    vref.enabled = true;
    vref.vref_pos = 0.15;
    vref.vref_neg = -0.15;
    vref.vref_initial = 0.15;
    vref.target_confidence = 0.7;

    ConstSource data("data", 1.0);       // d = +1
    ConstSource src_s_pos("src_s_pos", 1.0);     // signal > +Vref
    ConstSource src_s_neg("src_s_neg", 1.0);
    PeriodicStrobe strobe("strobe");

    DfeAdaptTdf dut("dut", dfe, vref);

    sca_tdf::sca_signal<double> s_data, s_pos, s_neg;
    sca_tdf::sca_signal<bool> s_strobe;

    data.out(s_data);
    src_s_pos.out(s_pos);
    src_s_neg.out(s_neg);
    strobe.out(s_strobe);
    dut.data_in(s_data);
    dut.vref_pos_in(s_pos);
    dut.vref_neg_in(s_neg);
    dut.sampling_trigger(s_strobe);

    // DE 控制/输出绑定
    sc_core::sc_signal<int> s_mode, s_state, s_update_count_sig;
    sc_core::sc_signal<bool> s_reset;
    sc_core::sc_signal<double> s_vref_cmd, s_vref_current;
    sc_core::sc_signal<double> s_tap[8];
    sc_core::sc_signal<int> s_NA, s_NB, s_NC, s_ND;
    sc_core::sc_signal<double> s_Ppos, s_Pneg, s_C, s_Delta, s_mu;

    dut.mode_de(s_mode);
    dut.reset_de(s_reset);
    dut.vref_cmd_de(s_vref_cmd);
    dut.tap1_de(s_tap[0]);
    dut.tap2_de(s_tap[1]);
    dut.tap3_de(s_tap[2]);
    dut.tap4_de(s_tap[3]);
    dut.tap5_de(s_tap[4]);
    dut.tap6_de(s_tap[5]);
    dut.tap7_de(s_tap[6]);
    dut.tap8_de(s_tap[7]);
    dut.stat_N_A(s_NA);
    dut.stat_N_B(s_NB);
    dut.stat_N_C(s_NC);
    dut.stat_N_D(s_ND);
    dut.stat_P_pos(s_Ppos);
    dut.stat_P_neg(s_Pneg);
    dut.stat_C(s_C);
    dut.stat_Delta(s_Delta);
    dut.stat_state(s_state);
    dut.stat_mu(s_mu);
    dut.stat_update_count(s_update_count_sig);
    dut.vref_current(s_vref_current);

    // 初值：数据模式、非复位、vref 命令 0.12
    s_mode.write(2);
    s_reset.write(false);
    s_vref_cmd.write(0.12);

    const double sim_time = 700e-9;  // 700 ns → 10000 strobes @ 70ps
    sc_core::sc_start(sim_time, sc_core::SC_SEC);

    // 1) 时基：更新次数 = 700ns / 70ps（旧硬编码 100ps 实现约为 7000）
    const int expected_updates = static_cast<int>(sim_time / (kTs * kStrobePeriodSteps));
    EXPECT_NEAR(dut.get_update_count(), expected_updates, 2)
        << "DfeAdaptTdf must follow the real CDR strobe (70ps), not a hardcoded UI";

    // 2) 单控制器镜像：vref_current 应等于 AdaptionDe 下发的命令
    EXPECT_NEAR(s_vref_current.read(), 0.12, 1e-12)
        << "vref_current must mirror the AdaptionDe command (single controller)";

    // 3) 统计输出已按周期发布（100 次频闪/周期 → 100 个周期）
    EXPECT_GT(s_NB.read(), 0) << "statistics must be published periodically";

    sc_core::sc_stop();
}
