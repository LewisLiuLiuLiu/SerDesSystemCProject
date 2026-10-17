/**
 * @file test_data_strobe_latch.cpp
 * @brief DataStrobeLatchTdf 单元测试——对外 data_out 干净 1/UI 判决流
 *
 * 场景：输入流模拟主采样器的"edge+data 混合保持流"：
 *   每 UI（10 步）中，第 0-4 步为边沿伪比特（0.9），第 5-9 步为数据比特。
 * 频闪位于数据相位（对应 CDR data_trigger 时刻）。
 * 期望：锁存器只在频闪后更新，输出在整 UI 内保持该 UI 的数据比特，
 *       且永不出现 0.9 的边沿伪比特。
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/data_strobe_latch.h"
#include <cmath>
#include <vector>

using namespace serdes;

namespace {

constexpr double kTs = 10e-12;
constexpr int kStepsPerUi = 10;

// 混合保持流源：每 UI [0.9 x5][bit x5]
class MixedStreamSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    MixedStreamSource(sc_core::sc_module_name nm, const std::vector<double>& bits)
        : sca_tdf::sca_module(nm), out("out"), m_bits(bits), m_step(0) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        int phase = static_cast<int>(m_step % kStepsPerUi);
        long long ui = m_step / kStepsPerUi;
        out.write((phase < 5) ? 0.9 : m_bits[ui % m_bits.size()]);
        ++m_step;
    }
private:
    std::vector<double> m_bits;
    long long m_step;
};

// 数据频闪源（每 UI 第 6 步，模拟"采样器已写入数据判决"的频闪时刻）
class DataStrobeSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> out;
    DataStrobeSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out"), m_step(0) {}
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(kTs, sc_core::SC_SEC);
    }
    void processing() override {
        out.write((m_step % kStepsPerUi) == 6);
        ++m_step;
    }
private:
    long long m_step;
};

// 逐时间步记录锁存器输出
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

} // namespace

TEST(DataStrobeLatchTest, CleanOneSamplePerUi) {
    const std::vector<double> bits = {1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    const int n_ui = 30;

    MixedStreamSource src("src", bits);
    DataStrobeSource strobe("strobe");
    DataStrobeLatchTdf dut("dut");
    OutRecorder rec("rec");

    sca_tdf::sca_signal<double> s_in, s_out;
    sca_tdf::sca_signal<bool> s_strobe;

    src.out(s_in);
    strobe.out(s_strobe);
    dut.in(s_in);
    dut.data_strobe(s_strobe);
    dut.out(s_out);
    rec.in(s_out);

    sc_core::sc_start(n_ui * kStepsPerUi * kTs, sc_core::SC_SEC);

    ASSERT_GE(rec.vals.size(), static_cast<size_t>(n_ui * kStepsPerUi));

    // 1) 输出永不出现边沿伪比特 0.9
    for (size_t i = 0; i < rec.vals.size(); ++i) {
        ASSERT_TRUE(rec.vals[i] == 0.0 || rec.vals[i] == 1.0)
            << "unexpected output " << rec.vals[i] << " at step " << i;
    }

    // 2) 频闪后（10k+8）与下一 UI 边沿相位（10(k+1)+2）都应等于该 UI 数据比特
    //    （频闪源@10k+6 → 模块内 strobe 延迟1 → 10k+7 锁存，
    //      in 延迟1 → 读到 10k+6 的数据比特）
    for (int k = 0; k < n_ui - 1; ++k) {
        double expected = bits[k % bits.size()];
        EXPECT_DOUBLE_EQ(rec.vals[10 * k + 8], expected) << "post-strobe, UI " << k;
        EXPECT_DOUBLE_EQ(rec.vals[10 * (k + 1) + 2], expected)
            << "edge phase of UI " << k + 1 << " must keep the latched data bit";
    }

    sc_core::sc_stop();
}
