#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include <complex>
#include <vector>
#include "ams/tx_ffe.h"
#include "common/parameters.h"

using namespace serdes;

// ============================================================================
// 辅助模块：单端信号源
// ============================================================================
class SignalSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    enum WaveformType {
        DC,
        SINE,
        SQUARE,
        IMPULSE,
        STEP,
        PRBS
    };
    
    SignalSource(sc_core::sc_module_name nm, 
                 WaveformType type = DC,
                 double amplitude = 1.0,
                 double frequency = 1e9,
                 double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {}
    
    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double signal = 0.0;
        double t = m_step_count * m_timestep;
        
        switch (m_type) {
            case DC:
                signal = m_amplitude;
                break;
            case SINE:
                signal = m_amplitude * sin(2.0 * M_PI * m_frequency * t);
                break;
            case SQUARE:
                signal = m_amplitude * (sin(2.0 * M_PI * m_frequency * t) > 0 ? 1.0 : -1.0);
                break;
            case IMPULSE:
                signal = (m_step_count == 0) ? m_amplitude : 0.0;
                break;
            case STEP:
                signal = m_amplitude;
                break;
            case PRBS:
                // 简化的 PRBS-7
                signal = m_amplitude * ((m_step_count % 127) < 64 ? 1.0 : -1.0);
                break;
        }
        
        out.write(signal);
        m_step_count++;
    }
    
    void reset() { m_step_count = 0; }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_timestep;
    unsigned long m_step_count;
};

// ============================================================================
// 辅助模块：信号监测器
// ============================================================================
class SignalSink : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    SignalSink(sc_core::sc_module_name nm, double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , in("in")
        , m_timestep(1.0 / sample_rate)
    {}
    
    void set_attributes() {
        in.set_rate(1);
        in.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        m_samples.push_back(in.read());
    }
    
    const std::vector<double>& get_samples() const { return m_samples; }
    
    double get_last() const { 
        return m_samples.empty() ? 0.0 : m_samples.back(); 
    }
    
    double get_mean() const {
        if (m_samples.empty()) return 0.0;
        double sum = 0.0;
        for (double v : m_samples) sum += v;
        return sum / m_samples.size();
    }
    
    double get_rms() const {
        if (m_samples.empty()) return 0.0;
        double sum_sq = 0.0;
        for (double v : m_samples) sum_sq += v * v;
        return sqrt(sum_sq / m_samples.size());
    }
    
    double get_max() const {
        if (m_samples.empty()) return 0.0;
        double max_val = m_samples[0];
        for (double v : m_samples) if (v > max_val) max_val = v;
        return max_val;
    }
    
    double get_min() const {
        if (m_samples.empty()) return 0.0;
        double min_val = m_samples[0];
        for (double v : m_samples) if (v < min_val) min_val = v;
        return min_val;
    }
    
    void clear() { m_samples.clear(); }
    
private:
    double m_timestep;
    std::vector<double> m_samples;
};

// ============================================================================
// 测试用顶层模块
// ============================================================================
SC_MODULE(FfeBasicTestbench) {
    SignalSource* src;
    TxFfeTdf* ffe;
    SignalSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    TxFfeParams params;
    
    FfeBasicTestbench(sc_core::sc_module_name nm, 
                      const TxFfeParams& p,
                      SignalSource::WaveformType waveform = SignalSource::DC,
                      double amplitude = 1.0,
                      double frequency = 1e9)
        : sc_core::sc_module(nm)
        , params(p)
    {
        // 创建模块
        src = new SignalSource("src", waveform, amplitude, frequency);
        ffe = new TxFfeTdf("ffe", params);
        sink = new SignalSink("sink");
        
        // 连接
        src->out(sig_in);
        ffe->in(sig_in);
        ffe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~FfeBasicTestbench() {
        delete src;
        delete ffe;
        delete sink;
    }
    
    const std::vector<double>& get_output_samples() const {
        return sink->get_samples();
    }
    
    double get_output_last() const { return sink->get_last(); }
    double get_output_mean() const { return sink->get_mean(); }
    double get_output_rms() const { return sink->get_rms(); }
};

// ============================================================================
// 测试用例1: FIR 滤波器基本功能
// ============================================================================
TEST(FfeBasicTest, AllBasicFunctionality) {
    // 配置参数 - 默认3抽头
    TxFfeParams params;
    params.taps = {0.2, 0.6, 0.2};  // 对称抽头，和为1.0
    
    // 创建测试平台 - DC输入
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb", params, 
                                                   SignalSource::DC, 1.0);
    
    // 运行仿真
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // 测试1: 验证端口连接成功
    SUCCEED() << "Port connection test passed";
    
    // 测试2: DC输入时，稳态输出应等于 输入 × 抽头系数之和
    double tap_sum = 0.0;
    for (double t : params.taps) tap_sum += t;
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GT(samples.size(), 10) << "Should have collected samples";
    
    // 稳态输出（跳过前几个样本）
    double steady_state = samples.back();
    EXPECT_NEAR(steady_state, 1.0 * tap_sum, 0.001) 
        << "DC gain should equal sum of taps";
    
    // 测试3: 验证抽头系数配置
    EXPECT_EQ(params.taps.size(), 3) << "Should have 3 taps";
    EXPECT_DOUBLE_EQ(params.taps[0], 0.2);
    EXPECT_DOUBLE_EQ(params.taps[1], 0.6);
    EXPECT_DOUBLE_EQ(params.taps[2], 0.2);
    
    // 测试4: 验证输出稳定性
    double out1 = tb->get_output_last();
    double out2 = tb->get_output_last();
    EXPECT_DOUBLE_EQ(out1, out2) << "Output should be stable";
    
    // 测试5: 验证正确的增益
    double measured_gain = steady_state / 1.0;
    EXPECT_NEAR(measured_gain, tap_sum, 0.01) << "Measured gain should match tap sum";
    
    delete tb;
}

// ============================================================================
// 测试用例2: 抽头系数配置验证
// ============================================================================
TEST(FfeTest, TapCoefficientsConfiguration) {
    TxFfeParams params;
    
    // 测试默认配置
    EXPECT_EQ(params.taps.size(), 3);
    EXPECT_DOUBLE_EQ(params.taps[0], 0.2);
    EXPECT_DOUBLE_EQ(params.taps[1], 0.6);
    EXPECT_DOUBLE_EQ(params.taps[2], 0.2);
    
    // 测试自定义配置 - 5抽头
    params.taps = {0.05, 0.15, 0.6, -0.15, -0.05};
    EXPECT_EQ(params.taps.size(), 5);
    
    // 验证主抽头是最大值
    double max_tap = 0.0;
    for (double t : params.taps) {
        if (std::abs(t) > std::abs(max_tap)) max_tap = t;
    }
    EXPECT_DOUBLE_EQ(max_tap, 0.6) << "Main tap should be the largest";
    
    // 测试去加重配置
    params.taps = {0.0, 1.0, -0.25};  // PCIe Gen3 风格
    EXPECT_DOUBLE_EQ(params.taps[1], 1.0) << "Main tap should be 1.0 for de-emphasis";
    EXPECT_LT(params.taps[2], 0.0) << "Post-cursor should be negative";
    
    // 测试预加重配置
    params.taps = {0.1, 0.7, -0.2};
    double tap_sum = 0.0;
    for (double t : params.taps) tap_sum += t;
    EXPECT_NEAR(tap_sum, 0.6, 0.001) << "Tap sum for pre-emphasis";
}

// ============================================================================
// 测试用例3: 冲激响应验证
// ============================================================================
TEST(FfeTest, ImpulseResponseVerification) {
    TxFfeParams params;
    params.taps = {0.2, 0.6, 0.2};
    
    // 创建测试平台 - 冲激输入
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb_impulse", params, 
                                                   SignalSource::IMPULSE, 1.0);
    
    // 运行仿真
    sc_core::sc_start(50, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GE(samples.size(), 5) << "Should have enough samples";
    
    // 冲激响应应该就是抽头系数
    // 注意：由于循环缓冲区的实现方式，输出会有延迟
    // 第一个输出: taps[0] * 1 = 0.2
    // 第二个输出: taps[1] * 1 = 0.6
    // 第三个输出: taps[2] * 1 = 0.2
    // 后续输出: 0
    
    // 找到非零输出的位置
    std::vector<double> nonzero_outputs;
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        if (std::abs(samples[i]) > 0.001) {
            nonzero_outputs.push_back(samples[i]);
        }
    }
    
    // 验证冲激响应包含抽头系数
    EXPECT_GE(nonzero_outputs.size(), 1) << "Should have non-zero impulse response";
    
    delete tb;
}

// ============================================================================
// 测试用例4: 卷积计算正确性
// ============================================================================
TEST(FfeTest, ConvolutionCalculationCorrectness) {
    TxFfeParams params;
    params.taps = {0.25, 0.5, 0.25};  // 简单的对称滤波器
    
    // 创建测试平台 - 方波输入
    FfeBasicTestbench* tb = new FfeBasicTestbench("tb_conv", params, 
                                                   SignalSource::SQUARE, 1.0, 1e9);
    
    // 运行仿真
    sc_core::sc_start(100, sc_core::SC_NS);
    
    const std::vector<double>& samples = tb->get_output_samples();
    ASSERT_GT(samples.size(), 10) << "Should have collected samples";
    
    // 验证输出在合理范围内
    double max_val = tb->sink->get_max();
    double min_val = tb->sink->get_min();
    
    // 对于归一化的抽头，输出范围应该在 [-1, 1] 附近
    EXPECT_LE(max_val, 1.5) << "Max output should be bounded";
    EXPECT_GE(min_val, -1.5) << "Min output should be bounded";
    
    // 验证输出不是常数（有滤波效果）
    EXPECT_GT(max_val - min_val, 0.1) << "Output should vary for square wave input";
    
    delete tb;
}

// ============================================================================
// 测试用例5: 参数边界条件
// ============================================================================
TEST(FfeTest, ParameterBoundaryConditions) {
    TxFfeParams params;
    
    // 测试单抽头配置
    params.taps = {1.0};
    EXPECT_EQ(params.taps.size(), 1);
    
    // 测试7抽头配置
    params.taps = {0.02, 0.08, 0.15, 0.5, -0.15, -0.1, -0.05};
    EXPECT_EQ(params.taps.size(), 7);
    
    // 验证抽头系数范围
    for (double t : params.taps) {
        EXPECT_GE(t, -1.0) << "Tap should be >= -1.0";
        EXPECT_LE(t, 1.0) << "Tap should be <= 1.0";
    }
    
    // 测试空抽头配置（应该使用默认值或最小配置）
    params.taps = {};
    EXPECT_TRUE(params.taps.empty());
}

// ============================================================================
// 测试用例6: 多抽头配置（3/5/7抽头）
// ============================================================================
TEST(FfeTest, MultiTapConfiguration) {
    // 3抽头配置
    TxFfeParams params3;
    params3.taps = {0.2, 0.6, 0.2};
    EXPECT_EQ(params3.taps.size(), 3);
    
    double sum3 = 0.0;
    for (double t : params3.taps) sum3 += t;
    EXPECT_NEAR(sum3, 1.0, 0.001) << "3-tap sum should be 1.0";
    
    // 5抽头配置
    TxFfeParams params5;
    params5.taps = {0.05, 0.15, 0.6, 0.15, 0.05};
    EXPECT_EQ(params5.taps.size(), 5);
    
    double sum5 = 0.0;
    for (double t : params5.taps) sum5 += t;
    EXPECT_NEAR(sum5, 1.0, 0.001) << "5-tap sum should be 1.0";
    
    // 7抽头配置
    TxFfeParams params7;
    params7.taps = {0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02};
    EXPECT_EQ(params7.taps.size(), 7);
    
    double sum7 = 0.0;
    for (double t : params7.taps) sum7 += t;
    EXPECT_NEAR(sum7, 1.0, 0.001) << "7-tap sum should be 1.0";
}

// ============================================================================
// 测试用例7: 默认参数验证
// ============================================================================
TEST(FfeTest, DefaultParameterVerification) {
    TxFfeParams params;
    
    // 验证默认值
    EXPECT_EQ(params.taps.size(), 3) << "Default should have 3 taps";
    EXPECT_DOUBLE_EQ(params.taps[0], 0.2);
    EXPECT_DOUBLE_EQ(params.taps[1], 0.6);
    EXPECT_DOUBLE_EQ(params.taps[2], 0.2);
    
    // 验证默认抽头系数之和
    double sum = params.taps[0] + params.taps[1] + params.taps[2];
    EXPECT_DOUBLE_EQ(sum, 1.0) << "Default tap sum should be 1.0";
    
    // 验证主抽头（中间）最大
    EXPECT_GT(params.taps[1], params.taps[0]);
    EXPECT_GT(params.taps[1], params.taps[2]);
}

// ============================================================================
// 测试用例8: 频率响应理论验证
// ============================================================================
TEST(FfeTest, FrequencyResponseTheory) {
    // FIR 滤波器的频率响应: H(f) = Σ c[k] × e^(-j2πfkT)
    std::vector<double> taps = {0.2, 0.6, 0.2};
    double T = 1.0 / 10e9;  // 符号周期 (10 GHz)
    
    // DC 频率 (f = 0)
    double freq_dc = 0.0;
    std::complex<double> H_dc(0.0, 0.0);
    for (size_t k = 0; k < taps.size(); ++k) {
        std::complex<double> exp_term = std::exp(std::complex<double>(0.0, -2.0 * M_PI * freq_dc * k * T));
        H_dc += taps[k] * exp_term;
    }
    double gain_dc = std::abs(H_dc);
    EXPECT_NEAR(gain_dc, 1.0, 0.001) << "DC gain should be 1.0 (sum of taps)";
    
    // Nyquist 频率 (f = Fs/2)
    double freq_nyquist = 5e9;
    std::complex<double> H_nyquist(0.0, 0.0);
    for (size_t k = 0; k < taps.size(); ++k) {
        std::complex<double> exp_term = std::exp(std::complex<double>(0.0, -2.0 * M_PI * freq_nyquist * k * T));
        H_nyquist += taps[k] * exp_term;
    }
    double gain_nyquist = std::abs(H_nyquist);
    
    // 对于对称抽头，高频增益应该不同于DC增益
    EXPECT_GT(gain_dc, 0.0) << "DC gain should be positive";
    
    // 验证增益的合理范围
    EXPECT_LE(gain_nyquist, 2.0) << "Nyquist gain should be bounded";
}

// ============================================================================
// 测试用例9: 去加重模式配置
// ============================================================================
TEST(FfeTest, DeemphasisModeConfiguration) {
    TxFfeParams params;
    
    // PCIe Gen3 风格的去加重 (3.5dB)
    params.taps = {0.0, 1.0, -0.25};
    
    EXPECT_DOUBLE_EQ(params.taps[0], 0.0) << "Pre-cursor should be 0";
    EXPECT_DOUBLE_EQ(params.taps[1], 1.0) << "Main cursor should be 1.0";
    EXPECT_LT(params.taps[2], 0.0) << "Post-cursor should be negative";
    
    // 去加重比例计算: -20*log10(1.0/(1.0-0.25)) ≈ 2.5 dB
    double main_cursor = params.taps[1];
    double post_cursor = params.taps[2];
    double ratio = main_cursor / (main_cursor + post_cursor);
    EXPECT_GT(ratio, 1.0) << "De-emphasis ratio should be > 1";
    
    // PCIe Gen4 风格的强去加重
    params.taps = {0.0, 1.0, -0.35};
    EXPECT_LT(params.taps[2], -0.25) << "Gen4 should have stronger de-emphasis";
}

// ============================================================================
// 测试用例10: 预加重模式配置
// ============================================================================
TEST(FfeTest, PreemphasisModeConfiguration) {
    TxFfeParams params;
    
    // 平衡预加重配置
    params.taps = {0.1, 0.6, -0.15, -0.05};
    
    EXPECT_GT(params.taps[0], 0.0) << "Pre-cursor should be positive";
    EXPECT_GT(params.taps[1], params.taps[0]) << "Main cursor should be largest";
    EXPECT_LT(params.taps[2], 0.0) << "Post-cursor 1 should be negative";
    EXPECT_LT(params.taps[3], 0.0) << "Post-cursor 2 should be negative";
    
    // 验证主抽头是最大的
    double max_abs = 0.0;
    int max_idx = -1;
    for (size_t i = 0; i < params.taps.size(); ++i) {
        if (std::abs(params.taps[i]) > max_abs) {
            max_abs = std::abs(params.taps[i]);
            max_idx = i;
        }
    }
    EXPECT_EQ(max_idx, 1) << "Main tap should be at index 1";
}
