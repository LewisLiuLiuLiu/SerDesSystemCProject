/**
 * @file test_sampler_hysteresis_behavior.cpp
 * @brief RxSamplerTdf 迟滞参数（已废弃）行为回归
 *
 * 2026-10 重审结论：真实数据 slicer 没有刻意的 Schmitt 迟滞；比较器
 * 残余记忆为 mV 级动态失调，归入 offset/噪声建模。`hysteresis` 参数
 * 已从判决路径移除（配置兼容保留、被忽略）。
 *
 * 本测试用两个 bench（hysteresis=0 与 0.1，其他参数相同）验证：
 * 对同一输入，两者判决完全一致——迟滞参数不再影响判决。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_sampler.h"
#include <cmath>

using namespace serdes;

namespace {

constexpr double kTs = 10e-12;

class DcDiffSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p, out_n;
    DcDiffSource(sc_core::sc_module_name nm, double vdiff)
        : sca_tdf::sca_module(nm), out_p("out_p"), out_n("out_n"), m_v(vdiff) {}
    void set_attributes() override {
        out_p.set_rate(1); out_n.set_rate(1);
        out_p.set_timestep(kTs, sc_core::SC_SEC);
        out_n.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out_p.write(0.5 * m_v); out_n.write(-0.5 * m_v); }
private:
    double m_v;
};

class ConstZero : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    ConstZero(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out") {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out.write(0.0); }
};

class AlwaysTrigger : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> out;
    AlwaysTrigger(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out") {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out.write(true); }
};

struct Bench {
    DcDiffSource* src;
    ConstZero* clk;
    AlwaysTrigger* trig;
    RxSamplerTdf* dut;

    sca_tdf::sca_signal<double> s_in_p, s_in_n, s_clk, s_out;
    sca_tdf::sca_signal<bool> s_trig;
    sc_core::sc_signal<bool> s_out_de;
    sc_core::sc_signal<double> s_vref_dummy;

    Bench(const char* pfx, double hyst)
    {
        RxSamplerParams p;
        p.threshold = 0.0;
        p.hysteresis = hyst;      // 被忽略
        p.resolution = 0.02;
        p.phase_source = "phase";

        src = new DcDiffSource((std::string(pfx) + "_src").c_str(), 0.03);
        clk = new ConstZero((std::string(pfx) + "_clk").c_str());
        trig = new AlwaysTrigger((std::string(pfx) + "_trig").c_str());
        dut = new RxSamplerTdf((std::string(pfx) + "_dut").c_str(), p);

        src->out_p(s_in_p);
        src->out_n(s_in_n);
        clk->out(s_clk);
        trig->out(s_trig);
        dut->in_p(s_in_p);
        dut->in_n(s_in_n);
        dut->clk_sample(s_clk);
        dut->sampling_trigger(s_trig);
        dut->vref_threshold_de(s_vref_dummy);
        dut->data_out(s_out);
        dut->data_out_de(s_out_de);
    }
    double output() const { return s_out.read(0); }
};

} // namespace

TEST(SamplerHysteresisTest, DeprecatedHysteresisIgnored) {
    // v=0.03V、门限 0、resolution=0.02：|0.03| > 0.02 → 确定性判 1
    Bench a("a", 0.0);
    Bench b("b", 0.1);   // 若迟滞仍生效，旧模型需要 v > 0.05 才判 1

    sc_core::sc_start(5, sc_core::SC_NS);

    EXPECT_NEAR(a.output(), 1.0, 0.1);
    EXPECT_NEAR(b.output(), 1.0, 0.1)
        << "hysteresis=0.1 must not change the decision (deprecated parameter)";
    EXPECT_EQ(a.output(), b.output())
        << "hysteresis must be fully ignored by the decision path";

    sc_core::sc_stop();
}
