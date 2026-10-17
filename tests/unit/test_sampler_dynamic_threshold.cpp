/**
 * @file test_sampler_dynamic_threshold.cpp
 * @brief RxSamplerTdf 动态门限（Vref 闭环）回归
 *
 * 2026-10 批次3：
 * - 新增 vref_threshold_de：AdaptionDe.vref_cmd → ±Vref 比较器动态门限
 * - 模糊判决区从 |v_diff| < resolution 修正为 |v_diff - thr| < resolution
 *   （围绕有效门限，而非围绕 0；对 ±Vref 比较器此前是错的）
 *
 * 三个 bench 同进程（单次精化、单次 sc_start）：
 *   A 正常极性：静态 0.15 时 v=0.12 → 0；动态 cmd=0.10 后 → 1
 *   B 反极性（-Vref）：静态时 v=0.05 → 0；动态 cmd=0.10（门限 -0.10）→ 1
 *   C 模糊区位置：v=0.14、静态门限 0.15、resolution=0.02 → |v-thr|=0.01
 *     落入围绕"门限"的模糊区（旧实现围绕 0 → 会恒判 0）→ 判决随机混合
 *
 * 注：RX 采样器的参数校验要求 hysteresis < resolution，使迟滞分支实际
 *     不可达（模糊区完全覆盖迟滞区）；本测试不对迟滞行为作断言。
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

// 逐采样统计 1 的个数（验证模糊区统计行为）
class OnesCounter : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    long long ones = 0;
    long long total = 0;
    OnesCounter(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in("in") {}
    void set_attributes() override {
        in.set_rate(1);
        in.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        if (in.read() > 0.5) ++ones;
        ++total;
    }
};

// 动态 Vref 命令驱动（DE 域，50ns 后从 0 切到 cmd）
class VrefCmdDriver : public sc_core::sc_module {
public:
    sc_core::sc_out<double> cmd;
    SC_HAS_PROCESS(VrefCmdDriver);
    VrefCmdDriver(sc_core::sc_module_name nm, double cmd_val)
        : sc_core::sc_module(nm), cmd("cmd"), m_cmd(cmd_val) {
        SC_THREAD(run);
    }
    void run() {
        cmd.write(0.0);
        wait(sc_core::sc_time(50, sc_core::SC_NS));
        cmd.write(m_cmd);
        wait(sc_core::sc_time(50, sc_core::SC_NS));
    }
private:
    double m_cmd;
};

struct SamplerBench {
    DcDiffSource* src;
    ConstZero* clk;
    AlwaysTrigger* trig;
    RxSamplerTdf* dut;
    VrefCmdDriver* cmd_drv;
    OnesCounter* counter = nullptr;

    sca_tdf::sca_signal<double> s_in_p, s_in_n, s_clk, s_out;
    sca_tdf::sca_signal<bool> s_trig;
    sc_core::sc_signal<bool> s_out_de;
    sc_core::sc_signal<double> s_cmd;

    SamplerBench(const char* pfx, double vdiff, bool invert, double cmd,
                 bool with_counter = false)
    {
        RxSamplerParams p;
        p.threshold = 0.15;
        p.hysteresis = 0.01;
        p.resolution = 0.02;
        p.phase_source = "phase";
        p.threshold_invert = invert;

        src = new DcDiffSource((std::string(pfx) + "_src").c_str(), vdiff);
        clk = new ConstZero((std::string(pfx) + "_clk").c_str());
        trig = new AlwaysTrigger((std::string(pfx) + "_trig").c_str());
        dut = new RxSamplerTdf((std::string(pfx) + "_dut").c_str(), p);
        cmd_drv = new VrefCmdDriver((std::string(pfx) + "_cmd").c_str(), cmd);

        src->out_p(s_in_p);
        src->out_n(s_in_n);
        clk->out(s_clk);
        trig->out(s_trig);

        dut->in_p(s_in_p);
        dut->in_n(s_in_n);
        dut->clk_sample(s_clk);
        dut->sampling_trigger(s_trig);
        dut->vref_threshold_de(s_cmd);
        dut->data_out(s_out);
        dut->data_out_de(s_out_de);
        cmd_drv->cmd(s_cmd);

        if (with_counter) {
            counter = new OnesCounter((std::string(pfx) + "_cnt").c_str());
            counter->in(s_out);
        }
    }
    bool output() const { return s_out.read(0) > 0.5; }
};

} // namespace

TEST(SamplerDynamicThresholdTest, DynamicVrefPolarityAndFuzzyRegion) {
    // A: 正常极性，v=0.12；静态 0.15 → 0，动态 0.10 → 1
    SamplerBench a("a", 0.12, false, 0.09);
    // B: 反极性（-Vref 比较器），v=0.05；动态 0.10 → 门限 -0.10 → 1
    SamplerBench b("b", 0.05, true, 0.09);
    // C: 模糊区位置（cmd 恒 0，静态门限 0.15）
    SamplerBench c("c", 0.14, false, 0.0, /*with_counter=*/true);

    sc_core::sc_start(40, sc_core::SC_NS);

    // 相位1（cmd=0 → 静态门限 0.15）
    EXPECT_FALSE(a.output()) << "static threshold 0.15: v=0.12 must decide 0";
    EXPECT_FALSE(b.output()) << "static threshold 0.15 (invert applies to cmd only): v=0.05 must decide 0";

    sc_core::sc_start(60, sc_core::SC_NS);  // 到 100ns，cmd 已切换

    // 相位2（cmd=0.10）
    EXPECT_TRUE(a.output()) << "dynamic threshold 0.09: v=0.12 must decide 1";
    EXPECT_TRUE(b.output()) << "inverted dynamic threshold -0.09: v=0.05 must decide 1";

    // C：模糊区围绕门限（|0.14-0.15|=0.01 < res=0.02）→ 随机混合；
    // 旧实现模糊区围绕 0（|0.14| >= 0.02）→ 恒为 0。
    long long n = c.counter->total;
    long long ones = c.counter->ones;
    EXPECT_GT(n, 1000);
    EXPECT_GT(ones, n / 5) << "fuzzy region must be around the threshold (not 0)";
    EXPECT_LT(ones, n - n / 5) << "fuzzy region must be around the threshold (not 0)";

    sc_core::sc_stop();
}
