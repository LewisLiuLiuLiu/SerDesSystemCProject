#ifndef TB_RX_ADAPTION_HELPERS_H
#define TB_RX_ADAPTION_HELPERS_H

#include <systemc>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>
#include <iomanip>

namespace serdes {
namespace tb {

// ============================================================================
// Statistics Structures
// ============================================================================

// Adaption convergence statistics
struct AdaptionStats {
    double agc_convergence_time;    // Time for AGC to converge (s)
    double dfe_convergence_time;    // Time for DFE to converge (s)
    double cdr_lock_time;           // Time for CDR to lock (s)
    double threshold_settle_time;   // Time for threshold to settle (s)
    
    double agc_steady_error;        // AGC steady-state error (%)
    double dfe_steady_error;        // DFE tap change after convergence
    double cdr_steady_rms;          // CDR phase error RMS (UI)
    
    int freeze_events;              // Number of freeze events
    int rollback_events;            // Number of rollback events
    int total_updates;              // Total update count
    
    double final_gain;              // Final VGA gain
    double final_threshold;         // Final sampler threshold
    double final_phase_cmd;         // Final phase command
    
    AdaptionStats()
        : agc_convergence_time(0), dfe_convergence_time(0)
        , cdr_lock_time(0), threshold_settle_time(0)
        , agc_steady_error(0), dfe_steady_error(0), cdr_steady_rms(0)
        , freeze_events(0), rollback_events(0), total_updates(0)
        , final_gain(0), final_threshold(0), final_phase_cmd(0) {}
};

// ============================================================================
// Signal Source Module - Generates test stimuli for Adaption module
// ============================================================================
class AdaptionSignalSource : public sc_core::sc_module {
public:
    // Output ports (match Adaption input ports)
    sc_core::sc_out<double> phase_error;
    sc_core::sc_out<double> amplitude_rms;
    sc_core::sc_out<int> error_count;
    sc_core::sc_out<double> isi_metric;
    sc_core::sc_out<int> mode;
    sc_core::sc_out<bool> reset;
    sc_core::sc_out<double> scenario_switch;
    
    // Stimulus types
    enum StimulusType {
        CONSTANT,           // Constant values
        STEP,               // Step response
        RAMP,               // Ramp response
        SINUSOIDAL,         // Sinusoidal variation
        RANDOM,             // Random noise
        FAULT_INJECTION     // Fault injection pattern
    };
    
    SC_HAS_PROCESS(AdaptionSignalSource);
    
    AdaptionSignalSource(sc_core::sc_module_name nm,
                         double sample_period = 1e-9,
                         unsigned int seed = 12345)
        : sc_core::sc_module(nm)
        , phase_error("phase_error")
        , amplitude_rms("amplitude_rms")
        , error_count("error_count")
        , isi_metric("isi_metric")
        , mode("mode")
        , reset("reset")
        , scenario_switch("scenario_switch")
        , m_sample_period(sample_period)
        , m_rng(seed)
        , m_noise_dist(0.0, 1.0)
        , m_step_count(0)
        // Default stimulus parameters
        , m_amplitude_target(0.4)
        , m_amplitude_noise(0.01)
        , m_phase_error_init(0.5)
        , m_phase_error_noise(0.001)
        , m_error_rate(0)
        , m_stimulus_type(CONSTANT)
        , m_step_time(0)
        , m_fault_time(0)
        , m_fault_duration(0)
        , m_current_mode(2)  // Data mode
        , m_reset_active(false)
    {
        SC_THREAD(stimulus_process);
    }
    
    // Configuration methods
    void set_stimulus_type(StimulusType type) { m_stimulus_type = type; }
    void set_amplitude_target(double amp) { m_amplitude_target = amp; }
    void set_amplitude_noise(double noise) { m_amplitude_noise = noise; }
    void set_phase_error_init(double phase) { m_phase_error_init = phase; }
    void set_phase_error_noise(double noise) { m_phase_error_noise = noise; }
    void set_error_rate(int rate) { m_error_rate = rate; }
    void set_step_time(double t) { m_step_time = t; }
    void set_fault_time(double t) { m_fault_time = t; }
    void set_fault_duration(double d) { m_fault_duration = d; }
    void set_mode(int m) { m_current_mode = m; }
    
    // Amplitude step configuration for AGC test
    void configure_amplitude_step(double amp1, double amp2, double step_time) {
        m_amplitude_target = amp1;
        m_amplitude_step_value = amp2;
        m_step_time = step_time;
        m_stimulus_type = STEP;
    }
    
    // Fault injection configuration
    void configure_fault_injection(double fault_time, double duration, int fault_type) {
        m_fault_time = fault_time;
        m_fault_duration = duration;
        m_fault_type = fault_type;
        m_stimulus_type = FAULT_INJECTION;
    }
    
private:
    void stimulus_process() {
        // Initial reset
        reset.write(true);
        wait(sc_core::sc_time(10, sc_core::SC_NS));
        reset.write(false);
        m_reset_active = false;
        
        // Set initial mode
        mode.write(m_current_mode);
        scenario_switch.write(0.0);
        
        while (true) {
            wait(sc_core::sc_time(m_sample_period, sc_core::SC_SEC));
            
            double current_time = sc_core::sc_time_stamp().to_seconds();
            
            // Generate phase error
            double phase_err = generate_phase_error(current_time);
            phase_error.write(phase_err);
            
            // Generate amplitude
            double amp = generate_amplitude(current_time);
            amplitude_rms.write(amp);
            
            // Generate error count
            int err = generate_error_count(current_time);
            error_count.write(err);
            
            // Generate ISI metric
            double isi = 0.1 + m_noise_dist(m_rng) * 0.01;
            isi_metric.write(isi);
            
            // Check for mode changes
            mode.write(m_current_mode);
            
            m_step_count++;
        }
    }
    
    double generate_phase_error(double t) {
        double base_error = m_phase_error_init;
        
        switch (m_stimulus_type) {
            case CONSTANT:
                // Decay over time (simulating CDR locking)
                base_error = m_phase_error_init * std::exp(-t * 1e6);
                break;
            case STEP:
                if (t > m_step_time) {
                    base_error = m_phase_error_init * 0.1;  // After step, small error
                }
                break;
            case SINUSOIDAL:
                base_error = m_phase_error_init * std::sin(2 * M_PI * 1e6 * t);
                break;
            case FAULT_INJECTION:
                if (t >= m_fault_time && t < m_fault_time + m_fault_duration) {
                    if (m_fault_type == 2) {  // Phase loss fault
                        base_error = 0.6;  // Large phase error
                    }
                }
                break;
            default:
                break;
        }
        
        // Add noise
        base_error += m_phase_error_noise * m_noise_dist(m_rng);
        
        return base_error;
    }
    
    double generate_amplitude(double t) {
        double amp = m_amplitude_target;
        
        switch (m_stimulus_type) {
            case STEP:
                if (t > m_step_time) {
                    amp = m_amplitude_step_value;
                }
                break;
            case RAMP:
                amp = m_amplitude_target * (1.0 + 0.1 * t / 1e-6);
                break;
            case SINUSOIDAL:
                amp = m_amplitude_target * (1.0 + 0.1 * std::sin(2 * M_PI * 1e6 * t));
                break;
            case FAULT_INJECTION:
                if (t >= m_fault_time && t < m_fault_time + m_fault_duration) {
                    if (m_fault_type == 1) {  // Amplitude fault
                        amp = 0.8;  // Abnormal amplitude
                    }
                }
                break;
            default:
                break;
        }
        
        // Add noise
        amp += m_amplitude_noise * m_noise_dist(m_rng);
        
        return amp;
    }
    
    int generate_error_count(double t) {
        int err = m_error_rate;
        
        switch (m_stimulus_type) {
            case FAULT_INJECTION:
                if (t >= m_fault_time && t < m_fault_time + m_fault_duration) {
                    if (m_fault_type == 0) {  // Error burst fault
                        err = 150;  // Large error count
                    }
                }
                break;
            default:
                // Random small errors
                err += static_cast<int>(std::abs(m_noise_dist(m_rng)) * 5);
                break;
        }
        
        return err;
    }
    
    double m_sample_period;
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;
    unsigned long m_step_count;
    
    double m_amplitude_target;
    double m_amplitude_noise;
    double m_amplitude_step_value;
    double m_phase_error_init;
    double m_phase_error_noise;
    int m_error_rate;
    
    StimulusType m_stimulus_type;
    double m_step_time;
    double m_fault_time;
    double m_fault_duration;
    int m_fault_type;
    int m_current_mode;
    bool m_reset_active;
};

// ============================================================================
// Monitor Module - Records Adaption outputs to CSV
// ============================================================================
class AdaptionMonitor : public sc_core::sc_module {
public:
    // Input ports (match Adaption output ports)
    sc_core::sc_in<double> vga_gain;
    sc_core::sc_in<double> dfe_tap1;
    sc_core::sc_in<double> dfe_tap2;
    sc_core::sc_in<double> dfe_tap3;
    sc_core::sc_in<double> dfe_tap4;
    sc_core::sc_in<double> dfe_tap5;
    sc_core::sc_in<double> dfe_tap6;
    sc_core::sc_in<double> dfe_tap7;
    sc_core::sc_in<double> dfe_tap8;
    sc_core::sc_in<double> sampler_threshold;
    sc_core::sc_in<double> sampler_hysteresis;
    sc_core::sc_in<double> phase_cmd;
    sc_core::sc_in<int> update_count;
    sc_core::sc_in<bool> freeze_flag;
    
    // Feedback inputs for recording
    sc_core::sc_in<double> phase_error;
    sc_core::sc_in<double> amplitude_rms;
    sc_core::sc_in<int> error_count;
    
    SC_HAS_PROCESS(AdaptionMonitor);
    
    AdaptionMonitor(sc_core::sc_module_name nm,
                    const std::string& filename,
                    double sample_period = 1e-9,
                    int num_dfe_taps = 5)
        : sc_core::sc_module(nm)
        , vga_gain("vga_gain")
        , dfe_tap1("dfe_tap1"), dfe_tap2("dfe_tap2")
        , dfe_tap3("dfe_tap3"), dfe_tap4("dfe_tap4")
        , dfe_tap5("dfe_tap5"), dfe_tap6("dfe_tap6")
        , dfe_tap7("dfe_tap7"), dfe_tap8("dfe_tap8")
        , sampler_threshold("sampler_threshold")
        , sampler_hysteresis("sampler_hysteresis")
        , phase_cmd("phase_cmd")
        , update_count("update_count")
        , freeze_flag("freeze_flag")
        , phase_error("phase_error")
        , amplitude_rms("amplitude_rms")
        , error_count("error_count")
        , m_filename(filename)
        , m_sample_period(sample_period)
        , m_num_dfe_taps(num_dfe_taps)
        , m_prev_freeze_flag(false)
        , m_freeze_events(0)
    {
        SC_THREAD(monitor_process);
        
        // Open file and write header
        m_file.open(m_filename);
        if (m_file.is_open()) {
            write_header();
        }
    }
    
    ~AdaptionMonitor() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    
    AdaptionStats get_stats(double UI = 2.5e-11) {
        AdaptionStats stats;
        
        // Calculate convergence times and errors from recorded data
        stats.freeze_events = m_freeze_events;
        stats.total_updates = m_update_history.empty() ? 0 : m_update_history.back();
        stats.final_gain = m_gain_history.empty() ? 0 : m_gain_history.back();
        stats.final_threshold = m_threshold_history.empty() ? 0 : m_threshold_history.back();
        stats.final_phase_cmd = m_phase_cmd_history.empty() ? 0 : m_phase_cmd_history.back();
        
        // Calculate AGC convergence time (gain change < 1% for 10 samples)
        stats.agc_convergence_time = calculate_agc_convergence_time();
        
        // Calculate DFE convergence time (tap change < 0.001 for 10 samples)
        stats.dfe_convergence_time = calculate_dfe_convergence_time();
        
        // Calculate CDR lock time (phase error < 0.01 UI for 100 samples)
        stats.cdr_lock_time = calculate_cdr_lock_time(UI);
        
        // Calculate steady-state errors
        stats.agc_steady_error = calculate_agc_steady_error();
        stats.cdr_steady_rms = calculate_cdr_steady_rms(UI);
        
        return stats;
    }
    
private:
    void write_header() {
        m_file << "time(s),vga_gain";
        for (int i = 1; i <= m_num_dfe_taps; ++i) {
            m_file << ",dfe_tap" << i;
        }
        m_file << ",sampler_threshold,sampler_hysteresis,phase_cmd";
        m_file << ",update_count,freeze_flag";
        m_file << ",phase_error,amplitude_rms,error_count" << std::endl;
    }
    
    void monitor_process() {
        wait(sc_core::sc_time(100, sc_core::SC_NS));  // Initial delay
        
        while (true) {
            wait(sc_core::sc_time(m_sample_period, sc_core::SC_SEC));
            
            double t = sc_core::sc_time_stamp().to_seconds();
            
            // Record data
            double gain = vga_gain.read();
            double taps[8] = {
                dfe_tap1.read(), dfe_tap2.read(), dfe_tap3.read(), dfe_tap4.read(),
                dfe_tap5.read(), dfe_tap6.read(), dfe_tap7.read(), dfe_tap8.read()
            };
            double threshold = sampler_threshold.read();
            double hysteresis = sampler_hysteresis.read();
            double phase = phase_cmd.read();
            int updates = update_count.read();
            bool freeze = freeze_flag.read();
            double phase_err = phase_error.read();
            double amp = amplitude_rms.read();
            int err_cnt = error_count.read();
            
            // Track freeze events
            if (freeze && !m_prev_freeze_flag) {
                m_freeze_events++;
            }
            m_prev_freeze_flag = freeze;
            
            // Store history for statistics
            m_time_history.push_back(t);
            m_gain_history.push_back(gain);
            m_threshold_history.push_back(threshold);
            m_phase_cmd_history.push_back(phase);
            m_phase_error_history.push_back(phase_err);
            m_update_history.push_back(updates);
            
            std::vector<double> tap_vec(taps, taps + 8);
            m_dfe_history.push_back(tap_vec);
            
            // Write to CSV
            if (m_file.is_open()) {
                m_file << std::scientific << std::setprecision(6);
                m_file << t << "," << gain;
                for (int i = 0; i < m_num_dfe_taps; ++i) {
                    m_file << "," << taps[i];
                }
                m_file << "," << threshold << "," << hysteresis << "," << phase;
                m_file << "," << updates << "," << (freeze ? 1 : 0);
                m_file << "," << phase_err << "," << amp << "," << err_cnt;
                m_file << std::endl;
            }
        }
    }
    
    double calculate_agc_convergence_time() {
        if (m_gain_history.size() < 20) return 0;
        
        for (size_t i = 10; i < m_gain_history.size() - 10; ++i) {
            bool converged = true;
            for (size_t j = i; j < i + 10 && j < m_gain_history.size() - 1; ++j) {
                double change = std::abs(m_gain_history[j+1] - m_gain_history[j]) / m_gain_history[j];
                if (change > 0.01) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                return m_time_history[i];
            }
        }
        return 0;  // Not converged
    }
    
    double calculate_dfe_convergence_time() {
        if (m_dfe_history.size() < 20) return 0;
        
        for (size_t i = 10; i < m_dfe_history.size() - 10; ++i) {
            bool converged = true;
            for (size_t j = i; j < i + 10 && j < m_dfe_history.size() - 1; ++j) {
                for (int k = 0; k < m_num_dfe_taps; ++k) {
                    double change = std::abs(m_dfe_history[j+1][k] - m_dfe_history[j][k]);
                    if (change > 0.001) {
                        converged = false;
                        break;
                    }
                }
                if (!converged) break;
            }
            if (converged) {
                return m_time_history[i];
            }
        }
        return 0;  // Not converged
    }
    
    double calculate_cdr_lock_time(double UI) {
        if (m_phase_error_history.size() < 110) return 0;
        
        for (size_t i = 10; i < m_phase_error_history.size() - 100; ++i) {
            bool locked = true;
            for (size_t j = i; j < i + 100; ++j) {
                if (std::abs(m_phase_error_history[j]) > 0.01 * UI) {
                    locked = false;
                    break;
                }
            }
            if (locked) {
                return m_time_history[i];
            }
        }
        return 0;  // Not locked
    }
    
    double calculate_agc_steady_error() {
        if (m_gain_history.size() < 100) return 0;
        
        // Use last 20% of data for steady-state calculation
        size_t start = m_gain_history.size() * 8 / 10;
        double sum = 0;
        for (size_t i = start; i < m_gain_history.size(); ++i) {
            sum += m_gain_history[i];
        }
        double mean = sum / (m_gain_history.size() - start);
        
        double var_sum = 0;
        for (size_t i = start; i < m_gain_history.size(); ++i) {
            var_sum += (m_gain_history[i] - mean) * (m_gain_history[i] - mean);
        }
        
        return std::sqrt(var_sum / (m_gain_history.size() - start)) / mean * 100;
    }
    
    double calculate_cdr_steady_rms(double UI) {
        if (m_phase_error_history.size() < 100) return 0;
        
        // Use last 20% of data
        size_t start = m_phase_error_history.size() * 8 / 10;
        double sum_sq = 0;
        for (size_t i = start; i < m_phase_error_history.size(); ++i) {
            double normalized = m_phase_error_history[i] / UI;
            sum_sq += normalized * normalized;
        }
        
        return std::sqrt(sum_sq / (m_phase_error_history.size() - start));
    }
    
    std::string m_filename;
    std::ofstream m_file;
    double m_sample_period;
    int m_num_dfe_taps;
    bool m_prev_freeze_flag;
    int m_freeze_events;
    
    // History for statistics
    std::vector<double> m_time_history;
    std::vector<double> m_gain_history;
    std::vector<double> m_threshold_history;
    std::vector<double> m_phase_cmd_history;
    std::vector<double> m_phase_error_history;
    std::vector<int> m_update_history;
    std::vector<std::vector<double>> m_dfe_history;
};

// ============================================================================
// Convergence Detector - Static helper functions
// ============================================================================
class ConvergenceDetector {
public:
    // Check AGC convergence (gain change < threshold for N samples)
    static bool check_agc_converged(const std::vector<double>& gain_history,
                                    double threshold = 0.01,
                                    int samples = 10) {
        if (gain_history.size() < static_cast<size_t>(samples + 1)) return false;
        
        for (size_t i = gain_history.size() - samples; i < gain_history.size() - 1; ++i) {
            double change = std::abs(gain_history[i+1] - gain_history[i]) / gain_history[i];
            if (change > threshold) return false;
        }
        return true;
    }
    
    // Check DFE convergence (all tap changes < threshold for N samples)
    static bool check_dfe_converged(const std::vector<std::vector<double>>& tap_history,
                                    int num_taps,
                                    double threshold = 0.001,
                                    int samples = 10) {
        if (tap_history.size() < static_cast<size_t>(samples + 1)) return false;
        
        for (size_t i = tap_history.size() - samples; i < tap_history.size() - 1; ++i) {
            for (int k = 0; k < num_taps; ++k) {
                double change = std::abs(tap_history[i+1][k] - tap_history[i][k]);
                if (change > threshold) return false;
            }
        }
        return true;
    }
    
    // Check CDR lock (phase error < threshold UI for N samples)
    static bool check_cdr_locked(const std::vector<double>& phase_error_history,
                                 double UI,
                                 double threshold_ui = 0.01,
                                 int samples = 100) {
        if (phase_error_history.size() < static_cast<size_t>(samples)) return false;
        
        for (size_t i = phase_error_history.size() - samples; i < phase_error_history.size(); ++i) {
            if (std::abs(phase_error_history[i]) > threshold_ui * UI) return false;
        }
        return true;
    }
};

} // namespace tb
} // namespace serdes

#endif // TB_RX_ADAPTION_HELPERS_H
