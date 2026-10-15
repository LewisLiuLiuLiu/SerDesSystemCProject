/**
 * @file test_dfe_tap_feedback.cpp
 * @brief Unit tests for RxDfeTdf - Tap feedback functionality tests
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include "ams/rx_dfe.h"
#include "common/parameters.h"

using namespace serdes;

// ============================================================================
// Helper Modules
// ============================================================================

/**
 * @brief Sequence source that outputs a predefined sequence of values
 */
class SequenceSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    std::vector<double> m_sequence;
    size_t m_index;
    double m_default_value;
    
    SequenceSource(sc_core::sc_module_name nm, const std::vector<double>& seq, 
                   double default_val = 0.0)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_sequence(seq)
        , m_index(0)
        , m_default_value(default_val)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);  // 10ps timestep
    }
    
    void processing() override {
        if (m_index < m_sequence.size()) {
            out.write(m_sequence[m_index]);
            m_index++;
        } else {
            out.write(m_default_value);
        }
    }
};

/**
 * @brief Signal sink for capturing output samples
 */
class SignalSink : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    std::vector<double> samples;
    
    SignalSink(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , in("in")
    {}
    
    void set_attributes() override {
        in.set_rate(1);
        in.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        samples.push_back(in.read());
    }
    
    void clear() { samples.clear(); }
};

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief DFE testbench with sequence input for tap feedback testing
 */
SC_MODULE(DfeTapTestbench) {
    SequenceSource* src;
    RxDfeTdf* dfe;
    SignalSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    RxDfeParams params;
    
    DfeTapTestbench(sc_core::sc_module_name nm, const RxDfeParams& p, 
                    const std::vector<double>& input_seq, double default_val = 0.0)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SequenceSource("src", input_seq, default_val);
        dfe = new RxDfeTdf("dfe", params);
        sink = new SignalSink("sink");
        
        // Connect signals
        src->out(sig_in);
        dfe->in(sig_in);
        dfe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~DfeTapTestbench() {
        // SystemC modules managed by simulator
    }
    
    const std::vector<double>& get_outputs() const {
        return sink->samples;
    }
};

// ============================================================================
// Test Cases - Tap Coefficient Effects
// ============================================================================

/**
 * @brief 全部抽头反馈场景（SystemC 每进程仅允许一次精化：统一构建 → 单次 sc_start → 分场景断言）
 */
TEST(RxDfeTapFeedbackTest, SimulationScenarios) {
    const std::vector<double> pulse5 = {1.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> seq8 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> seq7 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> input_p = {0.4, 0.3, -0.2, 0.1};
    const std::vector<double> seq3 = {1.0, 0.0, 0.0};
    const std::vector<double> seq_alt = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    const std::vector<double> seq_zero = {0.5, -0.3, 0.8, -0.1, 0.6};
    
    // --- 构建所有场景（模块名唯一化） ---
    RxDfeParams p_pos;    p_pos.taps = {0.2};              // 正抽头
    DfeTapTestbench* tb_pos = new DfeTapTestbench("tb_pos", p_pos, pulse5);
    
    RxDfeParams p_neg;    p_neg.taps = {-0.2};             // 负抽头
    DfeTapTestbench* tb_neg = new DfeTapTestbench("tb_neg", p_neg, pulse5);
    
    RxDfeParams p_small;  p_small.taps = {-0.1};           // 小抽头（慢衰减）
    DfeTapTestbench* tb_small = new DfeTapTestbench("tb_small", p_small, seq8);
    
    RxDfeParams p_multi;  p_multi.taps = {-0.3, -0.2, -0.1};  // 多抽头
    DfeTapTestbench* tb_multi = new DfeTapTestbench("tb_multi", p_multi, seq7);
    
    RxDfeParams p_p;      p_p.taps = {-0.15, -0.08, -0.04};   // P 路
    DfeTapTestbench* tb_p = new DfeTapTestbench("tb_p", p_p, input_p);
    
    RxDfeParams p_tiny;   p_tiny.taps = {-0.001};          // 边界小抽头
    DfeTapTestbench* tb_tiny = new DfeTapTestbench("tb_tiny", p_tiny, seq3);
    
    RxDfeParams p_alt;    p_alt.taps = {-0.1};             // 交替输入
    DfeTapTestbench* tb_alt = new DfeTapTestbench("tb_alt", p_alt, seq_alt);
    
    RxDfeParams p_zero;   p_zero.taps = {0.0};             // 零抽头直通
    DfeTapTestbench* tb_zero = new DfeTapTestbench("tb_zero", p_zero, seq_zero);
    
    // --- 单次统一推进（覆盖最长场景 100 ps） ---
    sc_core::sc_start(100, sc_core::SC_PS);
    
    // --- 正抽头：y[1] 为负反馈 ---
    {
        const auto& outputs = tb_pos->get_outputs();
        ASSERT_GE(outputs.size(), 4u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], -0.2, 1e-10);
        EXPECT_NEAR(outputs[2], 0.04, 1e-10);
    }
    
    // --- 负抽头：y[1] 为正反馈 ---
    {
        const auto& outputs = tb_neg->get_outputs();
        ASSERT_GE(outputs.size(), 4u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.2, 1e-10);
        EXPECT_NEAR(outputs[2], 0.04, 1e-10);
    }
    
    // --- 抽头幅度影响衰减速度（小抽头） ---
    {
        const auto& outputs = tb_small->get_outputs();
        ASSERT_GE(outputs.size(), 3u);
        EXPECT_NEAR(outputs[1], 0.1, 1e-10);
        EXPECT_NEAR(outputs[2], 0.01, 1e-10);
    }
    
    // --- 多抽头交互 ---
    {
        const auto& outputs = tb_multi->get_outputs();
        ASSERT_GE(outputs.size(), 5u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.3, 1e-10);
        EXPECT_NEAR(outputs[2], 0.29, 1e-10);
        EXPECT_NEAR(outputs[3], 0.247, 1e-10);
    }
    
    // --- 差分 P/N 抽头一致性 + 首样本直通 ---
    {
        RxDfeParams params_n;
        params_n.taps = {0.15, 0.08, 0.04};
        for (size_t i = 0; i < p_p.taps.size(); ++i) {
            EXPECT_NEAR(params_n.taps[i], -p_p.taps[i], 1e-15)
                << "N path tap should be negated at index " << i;
        }
        const auto& outputs = tb_p->get_outputs();
        ASSERT_GE(outputs.size(), 1u);
        EXPECT_NEAR(outputs[0], 0.4, 1e-10);
    }
    
    // --- 边界小抽头 ---
    {
        const auto& outputs = tb_tiny->get_outputs();
        ASSERT_GE(outputs.size(), 2u);
        EXPECT_NEAR(outputs[1], 0.001, 1e-10);
    }
    
    // --- 交替输入 ---
    {
        const auto& outputs = tb_alt->get_outputs();
        ASSERT_GE(outputs.size(), 5u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], -0.9, 1e-10);
        EXPECT_NEAR(outputs[2], 0.91, 1e-10);
        EXPECT_NEAR(outputs[3], -0.909, 1e-10);
    }
    
    // --- 零抽头直通 ---
    {
        const auto& outputs = tb_zero->get_outputs();
        ASSERT_GE(outputs.size(), 5u);
        for (size_t i = 0; i < seq_zero.size(); ++i) {
            EXPECT_NEAR(outputs[i], seq_zero[i], 1e-10)
                << "Zero tap should act as passthrough at index " << i;
        }
    }
    
    sc_core::sc_stop();
}

// sc_main 由 tests/test_main.cpp 统一提供（GTest 入口）
