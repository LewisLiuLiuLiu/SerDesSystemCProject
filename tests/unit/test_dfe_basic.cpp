/**
 * @file test_dfe_basic.cpp
 * @brief Unit tests for RxDfeTdf - Basic functionality tests
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
 * @brief Single-ended signal source for DFE testing
 */
class SingleEndedSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    double m_value;
    
    SingleEndedSource(sc_core::sc_module_name nm, double value = 0.0)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_value(value)
    {}
    
    void set_value(double v) { m_value = v; }
    
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);  // 10ps timestep
    }
    
    void processing() override {
        out.write(m_value);
    }
};

/**
 * @brief Single-ended signal sink for capturing DFE output
 */
class SingleEndedSink : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    std::vector<double> samples;
    
    SingleEndedSink(sc_core::sc_module_name nm)
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
    
    double get_last() const {
        return samples.empty() ? 0.0 : samples.back();
    }
    
    void clear() { samples.clear(); }
};

/**
 * @brief Sequence source that outputs a predefined sequence of values
 */
class SequenceSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    std::vector<double> m_sequence;
    size_t m_index;
    
    SequenceSource(sc_core::sc_module_name nm, const std::vector<double>& seq)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_sequence(seq)
        , m_index(0)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        if (m_index < m_sequence.size()) {
            out.write(m_sequence[m_index]);
            m_index++;
        } else {
            out.write(0.0);
        }
    }
};

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Basic DFE testbench with constant input
 */
SC_MODULE(DfeBasicTestbench) {
    SingleEndedSource* src;
    RxDfeTdf* dfe;
    SingleEndedSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    RxDfeParams params;
    
    DfeBasicTestbench(sc_core::sc_module_name nm, const RxDfeParams& p, double input_val = 0.0)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SingleEndedSource("src", input_val);
        dfe = new RxDfeTdf("dfe", params);
        sink = new SingleEndedSink("sink");
        
        // Connect signals
        src->out(sig_in);
        dfe->in(sig_in);
        dfe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~DfeBasicTestbench() {
        // SystemC modules managed by simulator
    }
    
    double get_output() const {
        return sink->get_last();
    }
    
    const std::vector<double>& get_samples() const {
        return sink->samples;
    }
};

/**
 * @brief DFE testbench with sequence input
 */
SC_MODULE(DfeSequenceTestbench) {
    SequenceSource* src;
    RxDfeTdf* dfe;
    SingleEndedSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    RxDfeParams params;
    
    DfeSequenceTestbench(sc_core::sc_module_name nm, const RxDfeParams& p, 
                         const std::vector<double>& input_seq)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SequenceSource("src", input_seq);
        dfe = new RxDfeTdf("dfe", params);
        sink = new SingleEndedSink("sink");
        
        // Connect signals
        src->out(sig_in);
        dfe->in(sig_in);
        dfe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~DfeSequenceTestbench() {
        // SystemC modules managed by simulator
    }
    
    const std::vector<double>& get_outputs() const {
        return sink->samples;
    }
};

// ============================================================================
// Test Cases
// ============================================================================

/**
 * @brief Test default parameter construction
 */
TEST(RxDfeBasicTest, DefaultConstruction) {
    RxDfeParams params;
    
    // Verify default values from parameters.h
    EXPECT_EQ(params.taps.size(), 3u);
    EXPECT_NEAR(params.taps[0], -0.05, 1e-10);
    EXPECT_NEAR(params.taps[1], -0.02, 1e-10);
    EXPECT_NEAR(params.taps[2], 0.01, 1e-10);
    EXPECT_EQ(params.update, "sign-lms");
    EXPECT_NEAR(params.mu, 1e-4, 1e-15);
}

/**
 * @brief Test port connection (instantiation without simulation)
 */
TEST(RxDfeBasicTest, PortConnection) {
    RxDfeParams params;
    DfeBasicTestbench* tb = new DfeBasicTestbench("tb", params, 0.0);
    
    // If we reach here without exception, port connection succeeded
    SUCCEED() << "DFE port connection test passed";
    
    // Note: SystemC modules managed by simulator, no manual delete
}

/**
 * @brief 全部仿真场景（SystemC 每进程仅允许一次精化：统一构建 → 单次 sc_start → 分场景断言）
 */
TEST(RxDfeBasicTest, SimulationScenarios) {
    // --- 构建所有场景（模块名唯一化） ---
    RxDfeParams p_zero;
    p_zero.taps = {-0.1, -0.05, 0.02};  // 3 taps
    DfeBasicTestbench* tb_zero = new DfeBasicTestbench("tb_zero", p_zero, 0.0);
    
    RxDfeParams p_const;
    p_const.taps = {-0.1, -0.05};  // 2 taps, sum = -0.15
    DfeBasicTestbench* tb_const = new DfeBasicTestbench("tb_const", p_const, 0.5);
    
    RxDfeParams p_single;
    p_single.taps = {-0.2};  // Single tap
    std::vector<double> seq_single = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    DfeSequenceTestbench* tb_single = new DfeSequenceTestbench("tb_single", p_single, seq_single);
    
    RxDfeParams p_multi;
    p_multi.taps = {-0.1, -0.05};  // Two taps
    std::vector<double> seq_multi = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    DfeSequenceTestbench* tb_multi = new DfeSequenceTestbench("tb_multi", p_multi, seq_multi);
    
    RxDfeParams p_empty;
    p_empty.taps = {};  // No taps - passthrough
    std::vector<double> seq_passthrough = {0.5, -0.3, 0.8, -0.1};
    DfeSequenceTestbench* tb_empty = new DfeSequenceTestbench("tb_empty", p_empty, seq_passthrough);
    
    RxDfeParams p_neg;
    p_neg.taps = {0.1, 0.05, -0.02};  // N path: negated taps
    std::vector<double> seq_neg = {0.5, 0.3, -0.2, 0.4};
    DfeSequenceTestbench* tb_neg = new DfeSequenceTestbench("tb_neg", p_neg, seq_neg);
    
    // --- 单次统一推进（覆盖最长场景 500 ps） ---
    sc_core::sc_start(500, sc_core::SC_PS);
    
    // --- 场景 1：零输入零输出（零历史） ---
    {
        const auto& samples = tb_zero->get_samples();
        ASSERT_GT(samples.size(), 0u);
        for (const auto& s : samples) {
            EXPECT_NEAR(s, 0.0, 1e-12) << "Zero input should produce zero output";
        }
    }
    
    // --- 场景 2：恒定输入稳态 y_ss = x / (1 + sum(taps)) ---
    {
        const auto& samples = tb_const->get_samples();
        ASSERT_GT(samples.size(), 20u);
        double input_val = 0.5;
        double tap_sum = -0.1 + -0.05;
        double expected_ss = input_val / (1.0 + tap_sum);
        EXPECT_NEAR(samples.back(), expected_ss, 0.01)
            << "Steady state should be x/(1+sum(taps))";
    }
    
    // --- 场景 3：单抽头递推 ---
    {
        const auto& outputs = tb_single->get_outputs();
        ASSERT_GE(outputs.size(), 3u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.2, 1e-10);
        EXPECT_NEAR(outputs[2], 0.04, 1e-10);
    }
    
    // --- 场景 4：多抽头递推 ---
    {
        const auto& outputs = tb_multi->get_outputs();
        ASSERT_GE(outputs.size(), 4u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.1, 1e-10);
        EXPECT_NEAR(outputs[2], 0.06, 1e-10);
        EXPECT_NEAR(outputs[3], 0.011, 1e-10);
    }
    
    // --- 场景 5：空抽头直通 ---
    {
        const auto& outputs = tb_empty->get_outputs();
        ASSERT_GE(outputs.size(), 4u);
        for (size_t i = 0; i < seq_passthrough.size(); ++i) {
            EXPECT_NEAR(outputs[i], seq_passthrough[i], 1e-10)
                << "Empty taps should act as passthrough at index " << i;
        }
    }
    
    // --- 场景 6：N 路抽头取反（参数一致性 + 首样本直通） ---
    {
        RxDfeParams params_p;
        params_p.taps = {-0.1, -0.05, 0.02};
        for (size_t i = 0; i < params_p.taps.size(); ++i) {
            EXPECT_NEAR(p_neg.taps[i], -params_p.taps[i], 1e-15)
                << "N path taps should be negated at index " << i;
        }
        const auto& outputs = tb_neg->get_outputs();
        ASSERT_GE(outputs.size(), 1u);
        EXPECT_NEAR(outputs[0], 0.5, 1e-10)
            << "First output should equal input (no history)";
    }
    
    sc_core::sc_stop();
}

// sc_main 由 tests/test_main.cpp 统一提供（GTest 入口）
