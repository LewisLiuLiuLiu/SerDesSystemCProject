/**
 * @file test_dfe_history_update.cpp
 * @brief Unit tests for RxDfeTdf - History buffer update (FIFO shift) tests
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
 * @brief Sequence source that outputs a predefined sequence
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

/**
 * @brief Signal sink for capturing output
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
};

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief DFE testbench for history update testing
 */
SC_MODULE(DfeHistoryTestbench) {
    SequenceSource* src;
    RxDfeTdf* dfe;
    SignalSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    RxDfeParams params;
    
    DfeHistoryTestbench(sc_core::sc_module_name nm, const RxDfeParams& p, 
                        const std::vector<double>& input_seq)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SequenceSource("src", input_seq);
        dfe = new RxDfeTdf("dfe", params);
        sink = new SignalSink("sink");
        
        src->out(sig_in);
        dfe->in(sig_in);
        dfe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~DfeHistoryTestbench() {}
    
    const std::vector<double>& get_outputs() const {
        return sink->samples;
    }
};

// ============================================================================
// Test Cases - History Buffer Update
// ============================================================================

/**
 * @brief 全部历史缓冲区场景（SystemC 每进程仅允许一次精化：统一构建 → 单次 sc_start → 分场景断言）
 */
TEST(RxDfeHistoryTest, SimulationScenarios) {
    // --- 构建所有场景（模块名唯一化） ---
    RxDfeParams p_initial;
    p_initial.taps = {-0.1, -0.05, -0.02};  // 3 taps
    std::vector<double> seq_initial = {0.5};
    DfeHistoryTestbench* tb_initial = new DfeHistoryTestbench("tb_initial", p_initial, seq_initial);
    
    RxDfeParams p_single;
    p_single.taps = {-0.5};  // Single tap
    std::vector<double> seq_single = {1.0, 2.0, 3.0, 4.0, 5.0};
    DfeHistoryTestbench* tb_single = new DfeHistoryTestbench("tb_single", p_single, seq_single);
    
    RxDfeParams p_multi;
    p_multi.taps = {-0.1, -0.2, -0.3};  // 3 taps
    std::vector<double> seq_impulse7 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    DfeHistoryTestbench* tb_multi = new DfeHistoryTestbench("tb_multi", p_multi, seq_impulse7);
    
    RxDfeParams p_store;
    p_store.taps = {-1.0};  // Tap = -1 to easily verify history content
    std::vector<double> seq_ones = {1.0, 1.0, 1.0, 1.0};
    DfeHistoryTestbench* tb_store = new DfeHistoryTestbench("tb_store", p_store, seq_ones);
    
    RxDfeParams p_size;
    p_size.taps = {-0.1, -0.08, -0.06, -0.04, -0.02};  // 5 taps
    std::vector<double> seq_impulse8 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    DfeHistoryTestbench* tb_size = new DfeHistoryTestbench("tb_size", p_size, seq_impulse8);
    
    RxDfeParams p_decay;
    p_decay.taps = {-0.3, -0.2};  // 2 taps
    std::vector<double> seq_impulse10 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    DfeHistoryTestbench* tb_decay = new DfeHistoryTestbench("tb_decay", p_decay, seq_impulse10);
    
    RxDfeParams p_pattern;
    p_pattern.taps = {-0.15, -0.1};
    std::vector<double> seq_alt = {0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5};
    DfeHistoryTestbench* tb_pattern = new DfeHistoryTestbench("tb_pattern", p_pattern, seq_alt);
    
    // --- 单次统一推进（覆盖最长场景 120 ps） ---
    sc_core::sc_start(120, sc_core::SC_PS);
    
    // --- 场景 1：初始历史全零 ---
    {
        const auto& outputs = tb_initial->get_outputs();
        ASSERT_GE(outputs.size(), 1u);
        EXPECT_NEAR(outputs[0], 0.5, 1e-10)
            << "First output should equal input (initial history is zero)";
    }
    
    // --- 场景 2：单抽头 FIFO 平移 ---
    {
        const auto& outputs = tb_single->get_outputs();
        ASSERT_GE(outputs.size(), 5u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 2.5, 1e-10);
        EXPECT_NEAR(outputs[2], 4.25, 1e-10);
        EXPECT_NEAR(outputs[3], 6.125, 1e-10);
        EXPECT_NEAR(outputs[4], 8.0625, 1e-10);
    }
    
    // --- 场景 3：多抽头 FIFO 平移 ---
    {
        const auto& outputs = tb_multi->get_outputs();
        ASSERT_GE(outputs.size(), 6u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.1, 1e-10);
        EXPECT_NEAR(outputs[2], 0.21, 1e-10);
        EXPECT_NEAR(outputs[3], 0.341, 1e-10);
        EXPECT_NEAR(outputs[4], 0.1061, 1e-10);
    }
    
    // --- 场景 4：历史存输出而非输入 ---
    {
        const auto& outputs = tb_store->get_outputs();
        ASSERT_GE(outputs.size(), 4u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 2.0, 1e-10);
        EXPECT_NEAR(outputs[2], 3.0, 1e-10);
        EXPECT_NEAR(outputs[3], 4.0, 1e-10);
    }
    
    // --- 场景 5：历史深度与抽头数一致 ---
    {
        const auto& outputs = tb_size->get_outputs();
        ASSERT_GE(outputs.size(), 7u);
        EXPECT_NEAR(outputs[0], 1.0, 1e-10);
        EXPECT_NEAR(outputs[1], 0.1, 1e-10);
        EXPECT_NEAR(outputs[2], 0.09, 1e-10);
        EXPECT_NEAR(outputs[3], 0.077, 1e-10);
        EXPECT_NEAR(outputs[4], 0.0609, 1e-10);
        EXPECT_NEAR(outputs[5], 0.04165, 1e-10);
    }
    
    // --- 场景 6：冲激响应衰减 ---
    {
        const auto& outputs = tb_decay->get_outputs();
        ASSERT_GE(outputs.size(), 8u);
        double last_significant = outputs[0];
        for (size_t i = 1; i < outputs.size(); ++i) {
            EXPECT_LT(std::abs(outputs[i]), std::abs(last_significant) * 2.0)
                << "Output should be bounded at index " << i;
            if (std::abs(outputs[i]) > 1e-6) {
                last_significant = outputs[i];
            }
        }
        EXPECT_LT(std::abs(outputs.back()), 0.01)
            << "Impulse response should decay toward zero";
    }
    
    // --- 场景 7：连续交替码型 ---
    {
        const auto& outputs = tb_pattern->get_outputs();
        ASSERT_GE(outputs.size(), 8u);
        for (size_t i = 0; i < outputs.size(); ++i) {
            EXPECT_LT(std::abs(outputs[i]), 2.0)
                << "Output should be bounded at index " << i;
        }
        double diff1 = std::abs(outputs[6] - outputs[4]);
        double diff2 = std::abs(outputs[7] - outputs[5]);
        EXPECT_LT(diff1, 0.1) << "Pattern should stabilize";
        EXPECT_LT(diff2, 0.1) << "Pattern should stabilize";
    }
    
    sc_core::sc_stop();
}

// sc_main 由 tests/test_main.cpp 统一提供（GTest 入口）
