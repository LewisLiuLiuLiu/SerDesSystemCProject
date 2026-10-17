/**
 * @file test_dfe_summer_strobe.cpp
 * @brief RxDfeSummerTdf 历史频闪与多抽头对齐回归（P0-3）
 *
 * 旧实现每个 TDF 时间步都移位历史缓冲区：ts=ui/10 时同一判决被重复
 * 插入 10 次，所有抽头退化为对最近判决的单一反馈（多抽头 DFE 失效）。
 * 新实现仅在 CDR data_trigger 频闪步移位一次。
 *
 * 本测试用确定性 bit 序列 + 恒定数据频闪，逐 UI 对账输出：
 *   v_eq[k] = v_main - Σ tap[i] * map(b[k-1-i])
 * 其中 history 在频闪后 = [b[k-1], b[k-2], b[k-3]]（含 1 步输入延迟）。
 * 若历史逐步移位（旧行为），结果会对不上本公式。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_dfe_summer.h"
#include <cmath>
#include <vector>

using namespace serdes;

namespace {

constexpr double kTs = 10e-12;   // 10 ps
constexpr int kStepsPerUi = 10;  // UI = 100 ps

// 恒定差分源
class ConstDiffSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p, out_n;
    ConstDiffSource(sc_core::sc_module_name nm, double vp, double vn)
        : sca_tdf::sca_module(nm), out_p("out_p"), out_n("out_n"), m_vp(vp), m_vn(vn) {}
    void set_attributes() override {
        out_p.set_rate(1); out_n.set_rate(1);
        out_p.set_timestep(kTs, sc_core::SC_SEC);
        out_n.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out_p.write(m_vp); out_n.write(m_vn); }
private:
    double m_vp, m_vn;
};

// 数据判决源：每个 UI（10 步）切换一次 bit 值
class BitSequenceSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    BitSequenceSource(sc_core::sc_module_name nm, const std::vector<double>& bits)
        : sca_tdf::sca_module(nm), out("out"), m_bits(bits) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        long long ui = static_cast<long long>(get_time().to_seconds() / (kTs * kStepsPerUi) + 1e-9);
        out.write(m_bits[ui % m_bits.size()]);
    }
private:
    std::vector<double> m_bits;
};

// 频闪源：每 UI 一次，位于每 UI 第 2 步（模拟 CDR data_trigger 时序）
class UiStrobeSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> out;
    UiStrobeSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out"), m_step(0) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        out.write((m_step % kStepsPerUi) == 2);
        ++m_step;
    }
private:
    long long m_step;
};

// 差分输出记录器
class DiffRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p, in_n;
    std::vector<double> diff;
    DiffRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in_p("in_p"), in_n("in_n") {}
    void set_attributes() override {
        in_p.set_rate(1); in_n.set_rate(1);
        in_p.set_timestep(kTs, sc_core::SC_SEC);
        in_n.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { diff.push_back(in_p.read() - in_n.read()); }
};

double map_pm1(double bit) { return (bit > 0.5) ? 1.0 : -1.0; }

} // namespace

TEST(DfeSummerStrobeTest, HistoryShiftsOncePerUi) {
    RxDfeSummerParams params;
    params.enable = true;
    params.tap_coeffs = {0.4, 0.3, 0.2};
    params.vtap = 1.0;
    params.vcm_out = 0.0;
    params.map_mode = "pm1";
    params.sat_enable = false;
    params.ui = kTs * kStepsPerUi;

    // bit 序列（0/1），周期 10 UI
    const std::vector<double> bits = {1, 0, 1, 1, 0, 0, 1, 0, 1, 0};

    ConstDiffSource src("src", 0.5, -0.5);            // v_main = 1.0
    BitSequenceSource data("data", bits);
    UiStrobeSource strobe("strobe");
    RxDfeSummerTdf dut("dut", params);
    DiffRecorder rec("rec");

    sca_tdf::sca_signal<double> s_in_p, s_in_n, s_out_p, s_out_n;
    sca_tdf::sca_signal<double> s_data;
    sca_tdf::sca_signal<bool> s_strobe;
    sc_core::sc_signal<double> s_tap_dummy[5];  // 抽头 DE 端口占位（本测试用静态抽头）

    src.out_p(s_in_p);
    src.out_n(s_in_n);
    data.out(s_data);
    strobe.out(s_strobe);

    dut.in_p(s_in_p);
    dut.in_n(s_in_n);
    dut.data_in(s_data);
    dut.data_strobe(s_strobe);
    dut.tap1_de(s_tap_dummy[0]);
    dut.tap2_de(s_tap_dummy[1]);
    dut.tap3_de(s_tap_dummy[2]);
    dut.tap4_de(s_tap_dummy[3]);
    dut.tap5_de(s_tap_dummy[4]);
    dut.out_p(s_out_p);
    dut.out_n(s_out_n);
    rec.in_p(s_out_p);
    rec.in_n(s_out_n);

    const int n_ui = 30;
    sc_core::sc_start(n_ui * kStepsPerUi * kTs, sc_core::SC_SEC);

    // 对账：history 在步 10k+3 移位后 = [b_k, b_{k-1}, b_{k-2}]
    // （data_in 延迟 1 步、频闪延迟 1 步）。取每 UI 第 5 步的输出。
    for (int k = 2; k < n_ui - 1; ++k) {
        int step = k * kStepsPerUi + 5;
        ASSERT_LT(step, static_cast<int>(rec.diff.size()));

        double b0 = map_pm1(bits[k % bits.size()]);
        double b1 = map_pm1(bits[(k - 1) % bits.size()]);
        double b2 = map_pm1(bits[(k - 2) % bits.size()]);
        double expected = 1.0 - (0.4 * b0 + 0.3 * b1 + 0.2 * b2);

        EXPECT_NEAR(rec.diff[step], expected, 1e-9)
            << "UI " << k << ": multi-tap history misaligned (per-step shift bug?)";
    }

    sc_core::sc_stop();
}
