#ifndef TB_TX_FFE_HELPERS_H
#define TB_TX_FFE_HELPERS_H

#include <systemc-ams>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>
#include <complex>

namespace serdes {
namespace tb {

// ============================================================================
// 信号统计信息
// ============================================================================
struct SignalStats {
    double mean;
    double rms;
    double peak_to_peak;
    double min_value;
    double max_value;
};

// ============================================================================
// 单端信号源 - 支持多种波形
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
        , m_sample_rate(sample_rate)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
        , m_prbs_state(0x7F)  // PRBS-7 初始状态
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
                signal = generate_prbs7() ? m_amplitude : -m_amplitude;
                break;
        }
        
        out.write(signal);
        m_step_count++;
    }
    
    void reset() { 
        m_step_count = 0; 
        m_prbs_state = 0x7F;
    }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_sample_rate;
    double m_timestep;
    unsigned long m_step_count;
    unsigned int m_prbs_state;
    
    // PRBS-7 生成器 (x^7 + x^6 + 1)
    bool generate_prbs7() {
        // 计算符号周期内的样本数
        unsigned long samples_per_symbol = static_cast<unsigned long>(m_sample_rate / m_frequency);
        if (samples_per_symbol == 0) samples_per_symbol = 1;
        
        // 每个符号周期更新一次 PRBS
        if (m_step_count % samples_per_symbol == 0) {
            unsigned int feedback = ((m_prbs_state >> 6) ^ (m_prbs_state >> 5)) & 1;
            m_prbs_state = ((m_prbs_state << 1) | feedback) & 0x7F;
        }
        
        return (m_prbs_state & 1) != 0;
    }
};

// ============================================================================
// 信号监测模块 - 记录波形数据
// ============================================================================
class SignalMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    SignalMonitor(sc_core::sc_module_name nm, 
                  const std::string& filename = "",
                  double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , in("in")
        , m_filename(filename)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {
        if (!m_filename.empty()) {
            m_file.open(m_filename);
            m_file << "time,value\n";
        }
    }
    
    ~SignalMonitor() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    
    void set_attributes() {
        in.set_rate(1);
        in.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double value = in.read();
        m_samples.push_back(value);
        
        if (m_file.is_open()) {
            double t = m_step_count * m_timestep;
            m_file << t << "," << value << "\n";
        }
        
        m_step_count++;
    }
    
    SignalStats get_stats() const {
        return calculate_stats(m_samples);
    }
    
    const std::vector<double>& get_samples() const { 
        return m_samples; 
    }
    
    double get_last() const {
        return m_samples.empty() ? 0.0 : m_samples.back();
    }
    
    void clear() {
        m_samples.clear();
        m_step_count = 0;
    }
    
private:
    std::string m_filename;
    std::ofstream m_file;
    double m_timestep;
    unsigned long m_step_count;
    std::vector<double> m_samples;
    
    SignalStats calculate_stats(const std::vector<double>& samples) const {
        SignalStats stats = {0, 0, 0, 1e9, -1e9};
        
        if (samples.empty()) return stats;
        
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (double v : samples) {
            sum += v;
            sum_sq += v * v;
            if (v < stats.min_value) stats.min_value = v;
            if (v > stats.max_value) stats.max_value = v;
        }
        
        stats.mean = sum / samples.size();
        stats.rms = sqrt(sum_sq / samples.size());
        stats.peak_to_peak = stats.max_value - stats.min_value;
        
        return stats;
    }
};

// ============================================================================
// 输入输出监测模块 - 同时监测输入和输出
// ============================================================================
class DualSignalMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_in<double> out;
    
    DualSignalMonitor(sc_core::sc_module_name nm, 
                      const std::string& filename = "",
                      double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , in("in")
        , out("out")
        , m_filename(filename)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {
        if (!m_filename.empty()) {
            m_file.open(m_filename);
            m_file << "time,input,output\n";
        }
    }
    
    ~DualSignalMonitor() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    
    void set_attributes() {
        in.set_rate(1);
        out.set_rate(1);
        in.set_timestep(m_timestep, sc_core::SC_SEC);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double in_val = in.read();
        double out_val = out.read();
        
        m_input_samples.push_back(in_val);
        m_output_samples.push_back(out_val);
        
        if (m_file.is_open()) {
            double t = m_step_count * m_timestep;
            m_file << t << "," << in_val << "," << out_val << "\n";
        }
        
        m_step_count++;
    }
    
    SignalStats get_input_stats() const {
        return calculate_stats(m_input_samples);
    }
    
    SignalStats get_output_stats() const {
        return calculate_stats(m_output_samples);
    }
    
    const std::vector<double>& get_input_samples() const { 
        return m_input_samples; 
    }
    
    const std::vector<double>& get_output_samples() const { 
        return m_output_samples; 
    }
    
private:
    std::string m_filename;
    std::ofstream m_file;
    double m_timestep;
    unsigned long m_step_count;
    std::vector<double> m_input_samples;
    std::vector<double> m_output_samples;
    
    SignalStats calculate_stats(const std::vector<double>& samples) const {
        SignalStats stats = {0, 0, 0, 1e9, -1e9};
        
        if (samples.empty()) return stats;
        
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (double v : samples) {
            sum += v;
            sum_sq += v * v;
            if (v < stats.min_value) stats.min_value = v;
            if (v > stats.max_value) stats.max_value = v;
        }
        
        stats.mean = sum / samples.size();
        stats.rms = sqrt(sum_sq / samples.size());
        stats.peak_to_peak = stats.max_value - stats.min_value;
        
        return stats;
    }
};

// ============================================================================
// FIR 频率响应分析器
// ============================================================================
class FirFrequencyAnalyzer {
public:
    // 计算 FIR 滤波器在指定频率的频率响应
    static std::complex<double> frequency_response(
        const std::vector<double>& taps,
        double frequency,
        double symbol_rate)
    {
        double T = 1.0 / symbol_rate;  // 符号周期
        std::complex<double> H(0.0, 0.0);
        
        for (size_t k = 0; k < taps.size(); ++k) {
            std::complex<double> exp_term = 
                std::exp(std::complex<double>(0.0, -2.0 * M_PI * frequency * k * T));
            H += taps[k] * exp_term;
        }
        
        return H;
    }
    
    // 计算增益（线性）
    static double gain_linear(const std::vector<double>& taps,
                              double frequency,
                              double symbol_rate)
    {
        return std::abs(frequency_response(taps, frequency, symbol_rate));
    }
    
    // 计算增益（dB）
    static double gain_db(const std::vector<double>& taps,
                          double frequency,
                          double symbol_rate)
    {
        double gain = gain_linear(taps, frequency, symbol_rate);
        if (gain <= 0.0) return -100.0;
        return 20.0 * log10(gain);
    }
    
    // 计算相位（度）
    static double phase_deg(const std::vector<double>& taps,
                            double frequency,
                            double symbol_rate)
    {
        std::complex<double> H = frequency_response(taps, frequency, symbol_rate);
        return std::arg(H) * 180.0 / M_PI;
    }
    
    // 计算 DC 增益
    static double dc_gain(const std::vector<double>& taps) {
        double sum = 0.0;
        for (double t : taps) sum += t;
        return sum;
    }
    
    // 计算 Nyquist 频率增益
    static double nyquist_gain(const std::vector<double>& taps) {
        double sum = 0.0;
        for (size_t k = 0; k < taps.size(); ++k) {
            sum += taps[k] * (k % 2 == 0 ? 1.0 : -1.0);
        }
        return std::abs(sum);
    }
};

// ============================================================================
// 预加重/去加重分析器
// ============================================================================
class EmphasisAnalyzer {
public:
    // 计算去加重量（dB）
    static double deemphasis_db(const std::vector<double>& taps) {
        if (taps.size() < 2) return 0.0;
        
        // 假设主抽头在索引1
        double main = taps.size() > 1 ? taps[1] : taps[0];
        double post = taps.size() > 2 ? taps[2] : 0.0;
        
        if (std::abs(main + post) < 1e-9) return 0.0;
        
        double ratio = main / (main + post);
        if (ratio <= 0.0) return 0.0;
        
        return 20.0 * log10(ratio);
    }
    
    // 计算预加重量（dB）
    static double preemphasis_db(const std::vector<double>& taps) {
        if (taps.size() < 2) return 0.0;
        
        double pre = taps[0];
        double main = taps.size() > 1 ? taps[1] : 0.0;
        
        if (std::abs(main) < 1e-9) return 0.0;
        
        double ratio = (main + pre) / main;
        if (ratio <= 0.0) return 0.0;
        
        return 20.0 * log10(ratio);
    }
    
    // 判断是否为去加重模式
    static bool is_deemphasis_mode(const std::vector<double>& taps) {
        if (taps.size() < 3) return false;
        
        // 去加重模式特征：主抽头为1.0，前置抽头为0或小值，后置抽头为负
        return (taps[0] < 0.1 && 
                taps[1] > 0.8 && 
                taps.size() > 2 && taps[2] < 0.0);
    }
    
    // 判断是否为预加重模式
    static bool is_preemphasis_mode(const std::vector<double>& taps) {
        if (taps.size() < 2) return false;
        
        // 预加重模式特征：前置抽头为正，主抽头不是1.0
        return (taps[0] > 0.05 && taps[1] < 0.9);
    }
};

} // namespace tb
} // namespace serdes

#endif // TB_TX_FFE_HELPERS_H
