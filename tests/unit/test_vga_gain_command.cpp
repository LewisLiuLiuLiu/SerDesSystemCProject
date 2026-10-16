/**
 * @file test_vga_gain_command.cpp
 * @brief Unit test for RxVgaTdf - AGC gain command port (gain_cmd_de)
 *
 * Verifies the runtime gain overlay semantics:
 * - cmd <= 0 keeps the static dc_gain (default behavior, legacy compatible)
 * - cmd > 0 overrides via scaling the fixed LTF by cmd / dc_gain
 *
 * Single TEST per binary (SystemC elaboration constraint).
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_vga.h"
#include <cmath>
#include <vector>

using namespace serdes;

namespace {

// 恒定差分源
class DcDiffSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;

    DcDiffSource(sc_core::sc_module_name nm, double amp, double vcm = 0.6)
        : sca_tdf::sca_module(nm), out_p("out_p"), out_n("out_n"),
          m_amp(amp), m_vcm(vcm) {}

    void set_attributes() override {
        out_p.set_rate(1);
        out_n.set_rate(1);
        out_p.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
        out_n.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    void processing() override {
        out_p.write(m_vcm + 0.5 * m_amp);
        out_n.write(m_vcm - 0.5 * m_amp);
    }
private:
    double m_amp;
    double m_vcm;
};

// 恒定 VDD
class ConstVdd : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    ConstVdd(sc_core::sc_module_name nm, double v = 1.0)
        : sca_tdf::sca_module(nm), out("out"), m_v(v) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    void processing() override { out.write(m_v); }
private:
    double m_v;
};

// 增益命令驱动（DE 域，分三相写入）
class GainCmdDriver : public sc_core::sc_module {
public:
    sc_core::sc_out<double> cmd;
    SC_HAS_PROCESS(GainCmdDriver);
    GainCmdDriver(sc_core::sc_module_name nm)
        : sc_core::sc_module(nm), cmd("cmd") {
        SC_THREAD(run);
    }
    void run() {
        cmd.write(0.0);   // 相位1: 静态 dc_gain
        wait(sc_core::sc_time(50, sc_core::SC_NS));
        cmd.write(4.0);   // 相位2: scale = 4/2 = 2 → 0.2V
        wait(sc_core::sc_time(50, sc_core::SC_NS));
        cmd.write(1.0);   // 相位3: scale = 1/2 = 0.5 → 0.05V
        wait(sc_core::sc_time(50, sc_core::SC_NS));
    }
};

// 记录差分输出
class DiffRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    std::vector<double> t;
    std::vector<double> diff;

    DiffRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in_p("in_p"), in_n("in_n") {}

    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
        in_p.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
        in_n.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    void processing() override {
        t.push_back(get_time().to_seconds());
        diff.push_back(in_p.read() - in_n.read());
    }
    double mean_in(double t0, double t1) const {
        double s = 0.0;
        int n = 0;
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] >= t0 && t[i] < t1) { s += diff[i]; ++n; }
        }
        return (n > 0) ? s / n : 0.0;
    }
};

} // namespace

TEST(VgaGainCommandTest, StaticAndCommandedGain) {
    RxVgaParams params;
    params.zeros.clear();   // 关闭 LTF → 纯 dc_gain 路径，便于精确对账
    params.poles.clear();
    params.dc_gain = 2.0;
    params.vcm_out = 0.6;
    params.sat_min = -0.5;  // Vsat = 0.5，0.05V 输入不饱和
    params.sat_max = 0.5;

    const double amp = 0.05;  // 差分输入幅度

    DcDiffSource src("src", amp);
    ConstVdd vdd("vdd");
    RxVgaTdf vga("vga", params);
    GainCmdDriver cmd_drv("cmd_drv");
    DiffRecorder rec("rec");

    sca_tdf::sca_signal<double> s_in_p, s_in_n, s_vdd, s_out_p, s_out_n;
    sc_core::sc_signal<double> s_gain_cmd;

    src.out_p(s_in_p);
    src.out_n(s_in_n);
    vdd.out(s_vdd);
    vga.in_p(s_in_p);
    vga.in_n(s_in_n);
    vga.vdd(s_vdd);
    vga.out_p(s_out_p);
    vga.out_n(s_out_n);
    vga.gain_cmd_de(s_gain_cmd);
    cmd_drv.cmd(s_gain_cmd);
    rec.in_p(s_out_p);
    rec.in_n(s_out_n);

    sc_core::sc_start(150, sc_core::SC_NS);

    // VGA 对差分信号有 tanh 软饱和（Vsat = 0.5*(sat_max-sat_min) = 0.5V），
    // 期望值需过同一物理公式：vout_diff = Vsat * tanh(vin_eff / Vsat)
    const double vsat = 0.5 * (params.sat_max - params.sat_min);
    auto expected = [&](double vin) { return vsat * std::tanh(vin / vsat); };

    // 相位1 (20-50ns): cmd=0 → 静态增益 2.0 → vin = 0.10V
    EXPECT_NEAR(rec.mean_in(20e-9, 50e-9), expected(2.0 * amp), 0.003)
        << "cmd<=0 must keep static dc_gain";

    // 相位2 (70-100ns): cmd=4.0 → vin = 0.20V
    EXPECT_NEAR(rec.mean_in(70e-9, 100e-9), expected(4.0 * amp), 0.003)
        << "cmd=4.0 must scale output to 4x input";

    // 相位3 (120-150ns): cmd=1.0 → vin = 0.05V
    EXPECT_NEAR(rec.mean_in(120e-9, 150e-9), expected(1.0 * amp), 0.003)
        << "cmd=1.0 must scale output to 1x input";

    sc_core::sc_stop();
}
