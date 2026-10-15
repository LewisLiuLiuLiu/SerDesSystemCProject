#ifndef TB_RX_SAMPLER_HELPERS_H
#define TB_RX_SAMPLER_HELPERS_H

#include <systemc-ams>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>
#include <complex>

namespace serdes {
namespace tb {

// 信号统计信息
struct SignalStats {
    double mean;
    double rms;
    double peak_to_peak;
    double min_value;
    double max_value;
};

// 差分信号源 - 支持多种波形
class DiffSignalSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    enum WaveformType {
        DC,
        SINE,
        SQUARE,
        PRBS
    };
    
    DiffSignalSource(sc_core::sc_module_name nm, 
                     WaveformType type = DC,
                     double amplitude = 0.1,
                     double frequency = 1e9,
                     double vcm = 0.6,
                     double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , out_p("out_p")
        , out_n("out_n")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_vcm(vcm)
        , m_sample_rate(sample_rate)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {}
    
    void set_attributes() {
        out_p.set_rate(1);
        out_n.set_rate(1);
        out_p.set_timestep(m_timestep, sc_core::SC_SEC);
        out_n.set_timestep(m_timestep, sc_core::SC_SEC);
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
            case PRBS:
                // 简化的PRBS-7
                signal = m_amplitude * ((m_step_count % 127) < 64 ? 1.0 : -1.0);
                break;
        }
        
        out_p.write(m_vcm + 0.5 * signal);
        out_n.write(m_vcm - 0.5 * signal);
        m_step_count++;
    }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_vcm;
    double m_sample_rate;
    double m_timestep;
    unsigned long m_step_count;
};

// 时钟源模块
class ClockSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> clk_out;
    
    double m_frequency;
    double m_amplitude;
    double m_vcm;
    double m_sample_rate;
    double m_timestep;
    int m_step_count;
    
    ClockSource(sc_core::sc_module_name nm, 
                double frequency = 10e9, 
                double amplitude = 1.0, 
                double vcm = 0.5,
                double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , clk_out("clk_out")
        , m_frequency(frequency)
        , m_amplitude(amplitude)
        , m_vcm(vcm)
        , m_sample_rate(sample_rate)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {}
    
    void set_attributes() {
        clk_out.set_rate(1);
        clk_out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double t = m_step_count * m_timestep;
        double clk = m_vcm + 0.5 * m_amplitude * sin(2.0 * M_PI * m_frequency * t);
        clk_out.write(clk);
        m_step_count++;
    }
};

// 相位偏移源
class PhaseOffsetSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> phase_out;
    
    double m_offset;
    double m_sample_rate;
    double m_timestep;
    
    PhaseOffsetSource(sc_core::sc_module_name nm, 
                      double offset = 0.0,
                      double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , phase_out("phase_out")
        , m_offset(offset)
        , m_sample_rate(sample_rate)
        , m_timestep(1.0 / sample_rate)
    {}
    
    void set_attributes() {
        phase_out.set_rate(1);
        phase_out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        phase_out.write(m_offset);
    }
    
    void set_offset(double offset) {
        m_offset = offset;
    }
};

// 采样触发源 - 周期性输出 bool 采样触发（替代 CDR 触发，用于被动式 Sampler）
class TriggerSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> trigger_out;
    
    TriggerSource(sc_core::sc_module_name nm,
                  double trigger_rate = 10e9,
                  double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , trigger_out("trigger_out")
        , m_timestep(1.0 / sample_rate)
        , m_period_samples(static_cast<int>(std::lround(sample_rate / trigger_rate)))
        , m_step_count(0)
    {
        if (m_period_samples < 1) {
            m_period_samples = 1;
        }
    }
    
    void set_attributes() {
        trigger_out.set_rate(1);
        trigger_out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        trigger_out.write((m_step_count % m_period_samples) == 0);
        m_step_count++;
    }
    
private:
    double m_timestep;
    int m_period_samples;
    unsigned long m_step_count;
};

// 采样器信号监测模块 - 记录波形数据
class SamplerSignalMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    sca_tdf::sca_in<double> data_out;
    sca_tdf::sca_de::sca_in<bool> data_out_de;  // DE 域桥接输入（来自 sampler 的 data_out_de）
    
    SamplerSignalMonitor(sc_core::sc_module_name nm, 
                        const std::string& filename = "",
                        double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , in_p("in_p")
        , in_n("in_n")
        , data_out("data_out")
        , data_out_de("data_out_de")
        , m_filename(filename)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {
        if (!m_filename.empty()) {
            m_file.open(m_filename);
            m_file << "time(s),input+(V),input-(V),differential(V),tdf_output,de_output\n";
        }
    }
    
    ~SamplerSignalMonitor() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    
    void set_attributes() {
        in_p.set_rate(1);
        in_n.set_rate(1);
        data_out.set_rate(1);
        in_p.set_timestep(m_timestep, sc_core::SC_SEC);
        in_n.set_timestep(m_timestep, sc_core::SC_SEC);
        data_out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double vp = in_p.read(0);
        double vn = in_n.read(0);
        double diff = vp - vn;
        double tdf_output = data_out.read(0);
        bool de_output = data_out_de.read();
        
        if (m_file.is_open()) {
            double t = m_step_count * m_timestep;
            m_file << t << "," << vp << "," << vn << "," << diff << "," << tdf_output << "," << (de_output ? "1" : "0") << "\n";
        }
        
        m_step_count++;
    }
    
private:
    std::string m_filename;
    std::ofstream m_file;
    double m_timestep;
    unsigned long m_step_count;
};

// BER计算器 - 用于统计误码率
class BerCalculator {
public:
    // 从数字样本计算BER
    static double calculate_ber(const std::vector<bool>& expected, 
                                const std::vector<bool>& actual) {
        if (expected.empty() || actual.empty()) return 0.0;
        
        size_t errors = 0;
        size_t n = std::min(expected.size(), actual.size());
        
        for (size_t i = 0; i < n; ++i) {
            if (expected[i] != actual[i]) {
                errors++;
            }
        }
        
        return static_cast<double>(errors) / n;
    }
};

} // namespace tb
} // namespace serdes

#endif // TB_RX_SAMPL