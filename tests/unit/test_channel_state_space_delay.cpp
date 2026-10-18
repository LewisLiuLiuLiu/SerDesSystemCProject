/**
 * @file test_channel_state_space_delay.cpp
 * @brief ChannelSParamTdf 传输延迟应用回归（2026-10）
 *
 * 背景：full_model.delay_s 曾被解析但从未使用，state-space 信道输出没有
 * 延迟（真实夹具 ~4ns）。修复后用每输出环形缓冲实现
 * round(delay/dt) 个采样的延迟。本测试用 1 状态、1 输入、1 输出、
 * 对角端口的模型（A=-1e9, B=1e9, C=1, D=0 → DC 增益 1，τ=1ns）+ 1ns 延迟：
 *   y(t) = 0                      (t < 1ns)
 *   y(t) = 1 - exp(-1e9*(t-1ns))  (t >= 1ns, 连续 sca_ss 求解)
 * 校验阶跃响应在延迟前为 0、延迟后按一阶规律上升。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/channel_sparam.h"
#include "common/parameters.h"
#include <fstream>
#include <cmath>
#include <cstdio>

using namespace serdes;

namespace {

constexpr double kTs = 10e-12;   // 10 ps

// 恒定 1.0 阶跃源（t>=0）
class StepSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    StepSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out") {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { out.write(1.0); }
};

// 记录每个时间步的输出
class OutRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    std::vector<double> vals;
    OutRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in("in") {}
    void set_attributes() override {
        in.set_rate(1);
        in.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override { vals.push_back(in.read()); }
};

std::string write_config() {
    const std::string path = "test_channel_ss_delay_config.json";
    std::ofstream f(path);
    f << R"({
  "method": "state_space",
  "full_model": {
    "n_diff_ports": 1,
    "n_outputs": 1,
    "n_states": 1,
    "port_pairs": [[0, 0]],
    "delay_s": [1e-9],
    "state_space": {
      "A": [[-1e9]],
      "B": [[1.0e9]],
      "C": [[1.0]],
      "D": [[0.0]],
      "E": [[0.0]]
    }
  },
  "port_config": {"active_inputs": [0], "active_outputs": [0]}
})";
    f.close();
    return path;
}

} // namespace

TEST(ChannelStateSpaceDelayTest, AppliesTransportDelay) {
    const std::string cfg = write_config();

    ChannelParams cp;
    ChannelExtendedParams ext;
    ext.method = ChannelMethod::STATE_SPACE;
    ext.config_file = cfg;

    StepSource src("src");
    ChannelSParamTdf dut("dut", cp, ext);
    OutRecorder rec("rec");

    sca_tdf::sca_signal<double> s_in, s_out;

    src.out(s_in);
    dut.in[0](s_in);
    dut.out[0](s_out);
    rec.in(s_out);

    sc_core::sc_start(4, sc_core::SC_NS);

    // 采样点索引（每步 10ps；延迟 1ns = 100 步）
    auto at_ns = [&](double t_ns) -> double {
        size_t i = static_cast<size_t>(t_ns * 1e-9 / kTs + 0.5);
        EXPECT_LT(i, rec.vals.size());
        return (i < rec.vals.size()) ? rec.vals[i] : 0.0;
    };

    // 延迟窗口内：输出保持 0（环形缓冲尚未填满）
    EXPECT_NEAR(at_ns(0.2), 0.0, 1e-12) << "no output before the delay elapses";
    EXPECT_NEAR(at_ns(0.9), 0.0, 1e-9);

    // 延迟之后：一阶上升 y = 1 - exp(-(t-1ns)/1ns)
    EXPECT_NEAR(at_ns(1.2), 1.0 - std::exp(-0.2), 0.05);
    EXPECT_NEAR(at_ns(1.5), 1.0 - std::exp(-0.5), 0.05);
    EXPECT_NEAR(at_ns(3.0), 1.0 - std::exp(-2.0), 0.05);

    std::remove(cfg.c_str());
    sc_core::sc_stop();
}
