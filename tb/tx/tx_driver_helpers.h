/**
 * @file tx_driver_helpers.h
 * @brief Helper modules for TX Driver transient testbench
 * 
 * Provides:
 * - DiffSignalSource: Configurable differential signal generator
 * - VddSource: Power supply with optional ripple
 * - SignalMonitor: Signal recording and statistics
 * - SignalStats: Statistical analysis structure
 */

#ifndef TX_DRIVER_HELPERS_H
#define TX_DRIVER_HELPERS_H

#include <systemc-ams>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace serdes {
namespace tb {

// ============================================================================
// Signal Statistics Structure
// ============================================================================

struct SignalStats {
    double mean;
    double rms;
    double peak_to_peak;
    double min_value;
    double max_value;
    double std_dev;
    
    SignalStats()
        : mean(0.0), rms(0.0), peak_to_peak(0.0)
        , min_value(0.0), max_value(0.0), std_dev(0.0) {}
    
    static SignalStats compute(const std::vector<double>& samples, size_t skip_initial = 0) {
        SignalStats stats;
        if (samples.empty() || skip_initial >= samples.size()) return stats;
        
        size_t n = samples.size() - skip_initial;
        
        // Mean
        double sum = 0.0;
        for (size_t i = skip_initial; i < samples.size(); ++i) {
            sum += samples[i];
        }
        stats.mean = sum / n;
        
        // RMS, min, max
        double sum_sq = 0.0;
        stats.min_value = samples[skip_initial];
        stats.max_value = samples[skip_initial];
        for (size_t i = skip_initial; i < samples.size(); ++i) {
            sum_sq += samples[i] * samples[i];
            stats.min_value = std::min(stats.min_value, samples[i]);
            stats.max_value = std::max(stats.max_value, samples[i]);
        }
        stats.rms = std::sqrt(sum_sq / n);
        stats.peak_to_peak = stats.max_value - stats.min_value;
        
        // Standard deviation
        double sum_var = 0.0;
        for (size_t i = skip_initial; i < samples.size(); ++i) {
            double diff = samples[i] - stats.mean;
            sum_var += diff * diff;
        }
        stats.std_dev = std::sqrt(sum_var / n);
        
        return stats;
    }
};

// ============================================================================
// Differential Signal Source
// ============================================================================

class DiffSignalSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    enum WaveformType {
        DC,
        SINE,
        SQUARE,
        STEP,
        PRBS,
        PULSE
    };
    
    struct Config {
        WaveformType type;
        double amplitude;      // Differential amplitude (Vpp/2)
        double frequency;      // Hz (for SINE, SQUARE)
        double vcm;            // Input common-mode voltage
        double step_time;      // Step transition time (for STEP)
        double duty_cycle;     // For SQUARE wave (0-1)
        double rise_time;      // Rise time for PULSE
        double fall_time;      // Fall time for PULSE
        double pulse_width;    // Pulse width
        unsigned int prbs_seed;// PRBS seed
        int prbs_order;        // PRBS order (7, 15, 23, 31)
        
        Config()
            : type(DC)
            , amplitude(0.5)
            , frequency(1e9)
            , vcm(0.0)
            , step_time(1e-9)
            , duty_cycle(0.5)
            , rise_time(10e-12)
            , fall_time(10e-12)
            , pulse_width(100e-12)
            , prbs_seed(12345)
            , prbs_order(7) {}
    };
    
    DiffSignalSource(sc_core::sc_module_name nm, const Config& cfg)
        : sca_tdf::sca_module(nm)
        , out_p("out_p")
        , out_n("out_n")
        , m_cfg(cfg)
        , m_prbs_state(cfg.prbs_seed)
        , m_current_bit(0)
        , m_bit_count(0)
    {}
    
    void set_attributes() override {
        out_p.set_rate(1);
        out_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void initialize() override {
        m_prbs_state = m_cfg.prbs_seed;
        m_current_bit = 0;
        m_bit_count = 0;
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double v_diff = 0.0;
        
        switch (m_cfg.type) {
            case DC:
                v_diff = m_cfg.amplitude;
                break;
                
            case SINE:
                v_diff = m_cfg.amplitude * std::sin(2.0 * M_PI * m_cfg.frequency * t);
                break;
                
            case SQUARE: {
                double period = 1.0 / m_cfg.frequency;
                double phase = std::fmod(t, period) / period;
                v_diff = (phase < m_cfg.duty_cycle) ? m_cfg.amplitude : -m_cfg.amplitude;
                break;
            }
                
            case STEP:
                v_diff = (t >= m_cfg.step_time) ? m_cfg.amplitude : 0.0;
                break;
                
            case PRBS: {
                // Simple PRBS implementation
                double bit_period = 1.0 / m_cfg.frequency;
                int bit_index = static_cast<int>(t / bit_period);
                if (bit_index != m_bit_count) {
                    m_bit_count = bit_index;
                    m_current_bit = generate_prbs_bit();
                }
                v_diff = m_current_bit ? m_cfg.amplitude : -m_cfg.amplitude;
                break;
            }
                
            case PULSE: {
                double period = 1.0 / m_cfg.frequency;
                double phase = std::fmod(t, period);
                if (phase < m_cfg.rise_time) {
                    v_diff = m_cfg.amplitude * (phase / m_cfg.rise_time);
                } else if (phase < m_cfg.rise_time + m_cfg.pulse_width) {
                    v_diff = m_cfg.amplitude;
                } else if (phase < m_cfg.rise_time + m_cfg.pulse_width + m_cfg.fall_time) {
                    double fall_phase = phase - m_cfg.rise_time - m_cfg.pulse_width;
                    v_diff = m_cfg.amplitude * (1.0 - fall_phase / m_cfg.fall_time);
                } else {
                    v_diff = 0.0;
                }
                break;
            }
        }
        
        out_p.write(m_cfg.vcm + 0.5 * v_diff);
        out_n.write(m_cfg.vcm - 0.5 * v_diff);
    }
    
private:
    Config m_cfg;
    unsigned int m_prbs_state;
    int m_current_bit;
    int m_bit_count;
    
    int generate_prbs_bit() {
        // PRBS-7: x^7 + x^6 + 1
        // PRBS-15: x^15 + x^14 + 1
        // PRBS-23: x^23 + x^18 + 1
        // PRBS-31: x^31 + x^28 + 1
        unsigned int feedback = 0;
        switch (m_cfg.prbs_order) {
            case 7:
                feedback = ((m_prbs_state >> 6) ^ (m_prbs_state >> 5)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | feedback) & 0x7F;
                break;
            case 15:
                feedback = ((m_prbs_state >> 14) ^ (m_prbs_state >> 13)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | feedback) & 0x7FFF;
                break;
            case 23:
                feedback = ((m_prbs_state >> 22) ^ (m_prbs_state >> 17)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | feedback) & 0x7FFFFF;
                break;
            case 31:
            default:
                feedback = ((m_prbs_state >> 30) ^ (m_prbs_state >> 27)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | feedback) & 0x7FFFFFFF;
                break;
        }
        return feedback;
    }
};

// ============================================================================
// VDD Source
// ============================================================================

class VddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    enum Mode {
        CONSTANT,
        SINUSOIDAL,
        RANDOM,
        STEP
    };
    
    struct Config {
        Mode mode;
        double nominal;        // Nominal voltage
        double ripple_amp;     // Ripple amplitude
        double ripple_freq;    // Ripple frequency
        double noise_sigma;    // Random noise sigma
        double step_time;      // Step transition time
        double step_delta;     // Step voltage change
        unsigned int seed;     // Random seed
        
        Config()
            : mode(CONSTANT)
            , nominal(1.0)
            , ripple_amp(0.01)
            , ripple_freq(100e6)
            , noise_sigma(0.001)
            , step_time(10e-9)
            , step_delta(0.05)
            , seed(42) {}
    };
    
    VddSource(sc_core::sc_module_name nm, const Config& cfg)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_cfg(cfg)
        , m_rng(cfg.seed)
        , m_noise_dist(0.0, cfg.noise_sigma)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double v = m_cfg.nominal;
        
        switch (m_cfg.mode) {
            case CONSTANT:
                // Just nominal voltage
                break;
                
            case SINUSOIDAL:
                v += m_cfg.ripple_amp * std::sin(2.0 * M_PI * m_cfg.ripple_freq * t);
                break;
                
            case RANDOM:
                v += m_noise_dist(m_rng);
                break;
                
            case STEP:
                if (t >= m_cfg.step_time) {
                    v += m_cfg.step_delta;
                }
                break;
        }
        
        out.write(v);
    }
    
private:
    Config m_cfg;
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;
};

// ============================================================================
// Signal Monitor
// ============================================================================

class SignalMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    std::vector<double> samples_p;
    std::vector<double> samples_n;
    std::vector<double> samples_diff;
    std::vector<double> samples_cm;
    std::vector<double> time_stamps;
    
    SignalMonitor(sc_core::sc_module_name nm, 
                  const std::string& output_file = "",
                  bool enable_csv = false)
        : sca_tdf::sca_module(nm)
        , in_p("in_p")
        , in_n("in_n")
        , m_output_file(output_file)
        , m_enable_csv(enable_csv)
        , m_csv_file(nullptr)
    {}
    
    ~SignalMonitor() {
        if (m_csv_file) {
            m_csv_file->close();
            delete m_csv_file;
        }
    }
    
    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void initialize() override {
        if (m_enable_csv && !m_output_file.empty()) {
            m_csv_file = new std::ofstream(m_output_file);
            if (m_csv_file->is_open()) {
                *m_csv_file << "time_s,out_p,out_n,out_diff,out_cm\n";
            }
        }
    }
    
    void processing() override {
        double vp = in_p.read();
        double vn = in_n.read();
        double t = get_time().to_seconds();
        
        samples_p.push_back(vp);
        samples_n.push_back(vn);
        samples_diff.push_back(vp - vn);
        samples_cm.push_back(0.5 * (vp + vn));
        time_stamps.push_back(t);
        
        if (m_csv_file && m_csv_file->is_open()) {
            *m_csv_file << t << ","
                       << vp << ","
                       << vn << ","
                       << (vp - vn) << ","
                       << (0.5 * (vp + vn)) << "\n";
        }
    }
    
    void clear() {
        samples_p.clear();
        samples_n.clear();
        samples_diff.clear();
        samples_cm.clear();
        time_stamps.clear();
    }
    
    SignalStats get_diff_stats(size_t skip_percent = 10) const {
        size_t skip = samples_diff.size() * skip_percent / 100;
        return SignalStats::compute(samples_diff, skip);
    }
    
    SignalStats get_cm_stats(size_t skip_percent = 10) const {
        size_t skip = samples_cm.size() * skip_percent / 100;
        return SignalStats::compute(samples_cm, skip);
    }
    
    void print_summary() const {
        auto diff_stats = get_diff_stats();
        auto cm_stats = get_cm_stats();
        
        std::cout << "\n=== Signal Monitor Summary ===" << std::endl;
        std::cout << "Samples collected: " << samples_diff.size() << std::endl;
        std::cout << "\nDifferential Signal:" << std::endl;
        std::cout << "  Mean:     " << diff_stats.mean * 1000 << " mV" << std::endl;
        std::cout << "  RMS:      " << diff_stats.rms * 1000 << " mV" << std::endl;
        std::cout << "  Pk-Pk:    " << diff_stats.peak_to_peak * 1000 << " mV" << std::endl;
        std::cout << "  StdDev:   " << diff_stats.std_dev * 1000 << " mV" << std::endl;
        std::cout << "\nCommon-Mode Signal:" << std::endl;
        std::cout << "  Mean:     " << cm_stats.mean * 1000 << " mV" << std::endl;
        std::cout << "  Pk-Pk:    " << cm_stats.peak_to_peak * 1000 << " mV" << std::endl;
        std::cout << "==============================\n" << std::endl;
    }
    
private:
    std::string m_output_file;
    bool m_enable_csv;
    std::ofstream* m_csv_file;
};

// ============================================================================
// Input Signal Monitor (for monitoring driver input)
// ============================================================================

class InputMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    sca_tdf::sca_in<double> vdd;
    
    std::vector<double> samples_in_diff;
    std::vector<double> samples_vdd;
    
    InputMonitor(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , in_p("in_p")
        , in_n("in_n")
        , vdd("vdd")
    {}
    
    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
        vdd.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        samples_in_diff.push_back(in_p.read() - in_n.read());
        samples_vdd.push_back(vdd.read());
    }
};

} // namespace tb
} // namespace serdes

#endif // TX_DRIVER_HELPERS_H
