/**
 * @file cdr_helpers.h
 * @brief CDR Testbench Helper Modules
 * 
 * Contains auxiliary modules for CDR testing:
 * - DataSource: Configurable data pattern generator with jitter injection
 * - SimpleSampler: Basic sampler for CDR closed-loop testing
 * - CdrMonitor: Phase waveform recorder and statistics
 * - LoopBandwidthAnalyzer: Theoretical loop parameter calculator
 * - BERCalculator: Bit error rate calculator
 * 
 * @version 0.2
 * @date 2026-01-20
 */

#ifndef TB_RX_CDR_HELPERS_H
#define TB_RX_CDR_HELPERS_H

#include <systemc-ams>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>
#include <complex>
#include <algorithm>
#include <numeric>

namespace serdes {
namespace tb {

// ============================================================================
// Phase Statistics Structure
// ============================================================================

struct PhaseStats {
    double mean;              // Mean phase value (s)
    double rms;               // RMS phase value (s)
    double peak_to_peak;      // Peak-to-peak phase variation (s)
    double min_value;         // Minimum phase value (s)
    double max_value;         // Maximum phase value (s)
    double lock_time;         // Time to lock (s)
    double steady_state_rms;  // Steady-state RMS jitter (s)
};

// ============================================================================
// Data Source Module
// ============================================================================

/**
 * @class DataSource
 * @brief Configurable data pattern generator with jitter and frequency offset
 * 
 * Supports multiple waveform types:
 * - PRBS-7, PRBS-15, PRBS-31 pseudo-random sequences
 * - Alternating pattern (010101...)
 * - Sine wave
 * - Square wave
 * 
 * Supports jitter injection:
 * - Random jitter (Gaussian)
 * - Periodic/sinusoidal jitter
 * 
 * Supports frequency offset (ppm)
 */
class DataSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;

    enum WaveformType {
        PRBS7,           // 7-bit LFSR (127 bits)
        PRBS15,          // 15-bit LFSR (32767 bits)
        PRBS31,          // 31-bit LFSR (2^31-1 bits)
        ALTERNATING,     // 010101...
        SINE,            // Sine wave
        SQUARE           // Square wave
    };

    DataSource(sc_core::sc_module_name nm,
               WaveformType type = PRBS15,
               double amplitude = 1.0,
               double frequency = 10e9,
               double sample_rate = 100e9,
               double jitter_sigma = 0.0,       // Random jitter sigma (s)
               double sj_freq = 0.0,            // Sinusoidal jitter frequency (Hz)
               double sj_amplitude = 0.0,       // Sinusoidal jitter amplitude (s)
               double freq_offset_ppm = 0.0)    // Frequency offset (ppm)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency * (1.0 + freq_offset_ppm / 1e6))
        , m_sample_rate(sample_rate)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
        , m_jitter_sigma(jitter_sigma)
        , m_sj_freq(sj_freq)
        , m_sj_amplitude(sj_amplitude)
        , m_rng(std::random_device{}())
        , m_noise_dist(0.0, 1.0)
        , m_prbs_bit_count(0)
    {
        // Initialize PRBS state based on type
        switch (m_type) {
            case PRBS7:
                m_prbs_state = 0x7F;           // 7-bit all 1s
                m_prbs_length = 7;
                break;
            case PRBS15:
                m_prbs_state = 0x7FFF;         // 15-bit all 1s
                m_prbs_length = 15;
                break;
            case PRBS31:
                m_prbs_state = 0x7FFFFFFF;     // 31-bit all 1s
                m_prbs_length = 31;
                break;
            default:
                m_prbs_state = 0;
                m_prbs_length = 0;
                break;
        }
        
        // Calculate samples per bit
        m_samples_per_bit = static_cast<unsigned int>(m_sample_rate / m_frequency);
        if (m_samples_per_bit < 1) m_samples_per_bit = 1;
        
        // Generate first bit
        m_current_bit = generate_prbs_bit();
    }

    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }

    void processing() {
        double signal = 0.0;
        double t = m_step_count * m_timestep;

        // Generate base signal
        switch (m_type) {
            case PRBS7:
            case PRBS15:
            case PRBS31:
                // Update bit at symbol boundaries
                if (m_prbs_bit_count >= m_samples_per_bit) {
                    m_current_bit = generate_prbs_bit();
                    m_prbs_bit_count = 0;
                }
                signal = m_current_bit ? 1.0 : -1.0;
                m_prbs_bit_count++;
                break;
                
            case ALTERNATING:
                signal = ((m_step_count / m_samples_per_bit) % 2 == 0) ? 1.0 : -1.0;
                break;
                
            case SINE:
                signal = sin(2.0 * M_PI * m_frequency * t);
                break;
                
            case SQUARE:
                signal = (sin(2.0 * M_PI * m_frequency * t) > 0) ? 1.0 : -1.0;
                break;
        }

        // Apply amplitude
        signal *= m_amplitude;

        // Apply jitter (simplified: amplitude modulation to simulate timing jitter)
        double jitter_effect = 0.0;
        if (m_jitter_sigma > 0.0) {
            // Random jitter as amplitude noise
            jitter_effect += m_jitter_sigma * m_noise_dist(m_rng) * 1e12;  // Scale for effect
        }
        if (m_sj_freq > 0.0 && m_sj_amplitude > 0.0) {
            // Sinusoidal jitter as amplitude modulation
            jitter_effect += m_sj_amplitude * sin(2.0 * M_PI * m_sj_freq * t) * 1e12;
        }
        
        // Add small jitter effect to signal (simplified model)
        signal += jitter_effect * 0.01;

        out.write(signal);
        m_step_count++;
    }

private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_sample_rate;
    double m_timestep;
    unsigned long m_step_count;
    double m_jitter_sigma;
    double m_sj_freq;
    double m_sj_amplitude;

    // PRBS generator state
    uint32_t m_prbs_state;
    int m_prbs_length;
    unsigned int m_samples_per_bit;
    unsigned int m_prbs_bit_count;
    bool m_current_bit;

    // Random number generator
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;

    /**
     * @brief Generate next PRBS bit
     * @return Next bit value (true/false)
     */
    bool generate_prbs_bit() {
        bool bit = false;
        
        switch (m_prbs_length) {
            case 7:
                // PRBS-7: x^7 + x^6 + 1
                bit = ((m_prbs_state >> 6) ^ (m_prbs_state >> 5)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | bit) & 0x7F;
                break;
                
            case 15:
                // PRBS-15: x^15 + x^14 + 1
                bit = ((m_prbs_state >> 14) ^ (m_prbs_state >> 13)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | bit) & 0x7FFF;
                break;
                
            case 31:
                // PRBS-31: x^31 + x^28 + 1
                bit = ((m_prbs_state >> 30) ^ (m_prbs_state >> 27)) & 1;
                m_prbs_state = ((m_prbs_state << 1) | bit) & 0x7FFFFFFF;
                break;
                
            default:
                bit = false;
                break;
        }
        
        return bit;
    }
};

// ============================================================================
// Simple Sampler Module
// ============================================================================

/**
 * @class SimpleSampler
 * @brief Simplified sampler for CDR closed-loop testing
 * 
 * Performs threshold-based decision on input signal.
 * Phase offset input is available for future enhancement.
 */
class SimpleSampler : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_in<double> phase_offset;
    sca_tdf::sca_out<double> out;

    double m_threshold;
    double m_timestep;

    SimpleSampler(sc_core::sc_module_name nm,
                  double sample_rate = 10e9,
                  double threshold = 0.0)
        : sca_tdf::sca_module(nm)
        , in("in")
        , phase_offset("phase_offset")
        , out("out")
        , m_threshold(threshold)
        , m_timestep(1.0 / sample_rate)
    {}

    void set_attributes() {
        in.set_rate(1);
        phase_offset.set_rate(1);
        out.set_rate(1);
        in.set_timestep(m_timestep, sc_core::SC_SEC);
        phase_offset.set_timestep(m_timestep, sc_core::SC_SEC);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }

    void processing() {
        double data = in.read();
        // Note: phase_offset read but simplified - full implementation would
        // adjust sampling instant based on phase offset
        double offset = phase_offset.read();
        (void)offset;  // Suppress unused warning
        
        // Simple threshold decision
        double sampled = (data > m_threshold) ? 1.0 : -1.0;
        out.write(sampled);
    }
};

// ============================================================================
// CDR Monitor Module
// ============================================================================

/**
 * @class CdrMonitor
 * @brief Records CDR phase waveform and computes statistics
 * 
 * Features:
 * - Phase waveform recording to CSV file
 * - Lock detection based on phase variance
 * - Statistical analysis (mean, RMS, peak-to-peak)
 */
class CdrMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> phase_in;
    sca_tdf::sca_in<double> data_in;

    CdrMonitor(sc_core::sc_module_name nm,
               const std::string& filename = "",
               double sample_rate = 10e9)
        : sca_tdf::sca_module(nm)
        , phase_in("phase_in")
        , data_in("data_in")
        , m_filename(filename)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
        , m_lock_threshold(5e-12)  // 5ps lock threshold
        , m_locked(false)
        , m_lock_time(0.0)
        , m_lock_window(100)       // 100 samples for lock detection
    {
        if (!m_filename.empty()) {
            m_file.open(m_filename);
            m_file << "time_s,phase_s,phase_ps,phase_ui\n";
        }
    }

    ~CdrMonitor() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    void set_attributes() {
        phase_in.set_rate(1);
        data_in.set_rate(1);
        phase_in.set_timestep(m_timestep, sc_core::SC_SEC);
        data_in.set_timestep(m_timestep, sc_core::SC_SEC);
    }

    void processing() {
        double phase = phase_in.read();
        double t = m_step_count * m_timestep;

        m_phase_samples.push_back(phase);

        // Lock detection: check if phase variance is below threshold
        if (!m_locked && m_phase_samples.size() > m_lock_window) {
            double variance = calculate_variance(
                m_phase_samples.end() - m_lock_window, 
                m_phase_samples.end());
            if (variance < m_lock_threshold * m_lock_threshold) {
                m_locked = true;
                m_lock_time = t;
            }
        }

        // Write to file
        if (m_file.is_open()) {
            double phase_ps = phase * 1e12;
            double phase_ui = phase / m_timestep;
            m_file << t << "," << phase << "," << phase_ps << "," << phase_ui << "\n";
        }

        m_step_count++;
    }

    /**
     * @brief Compute phase statistics
     * @param ui_period Unit interval period (s)
     * @return PhaseStats structure with computed statistics
     */
    PhaseStats get_phase_stats(double ui_period = 1e-10) const {
        PhaseStats stats = {0, 0, 0, 1e9, -1e9, 0, 0};

        if (m_phase_samples.empty()) return stats;

        // Compute basic statistics
        double sum = 0.0;
        double sum_sq = 0.0;

        for (double v : m_phase_samples) {
            sum += v;
            sum_sq += v * v;
            if (v < stats.min_value) stats.min_value = v;
            if (v > stats.max_value) stats.max_value = v;
        }

        size_t n = m_phase_samples.size();
        stats.mean = sum / n;
        stats.rms = std::sqrt(sum_sq / n);
        stats.peak_to_peak = stats.max_value - stats.min_value;
        stats.lock_time = m_lock_time;

        // Compute steady-state RMS (after lock)
        if (m_locked && n > m_lock_window) {
            size_t lock_idx = static_cast<size_t>(m_lock_time / m_timestep);
            if (lock_idx < n) {
                double steady_sum = 0.0;
                double steady_sum_sq = 0.0;
                size_t steady_count = 0;
                
                for (size_t i = lock_idx; i < n; ++i) {
                    steady_sum += m_phase_samples[i];
                    steady_sum_sq += m_phase_samples[i] * m_phase_samples[i];
                    steady_count++;
                }
                
                if (steady_count > 0) {
                    double steady_mean = steady_sum / steady_count;
                    double steady_var = steady_sum_sq / steady_count - steady_mean * steady_mean;
                    stats.steady_state_rms = std::sqrt(std::max(0.0, steady_var));
                }
            }
        }

        return stats;
    }

    /**
     * @brief Check if CDR is locked
     * @return true if locked, false otherwise
     */
    bool is_locked() const {
        return m_locked;
    }

    /**
     * @brief Get raw phase samples
     * @return Vector of phase samples
     */
    const std::vector<double>& get_phase_samples() const {
        return m_phase_samples;
    }

private:
    std::string m_filename;
    std::ofstream m_file;
    double m_timestep;
    unsigned long m_step_count;
    std::vector<double> m_phase_samples;
    double m_lock_threshold;
    bool m_locked;
    double m_lock_time;
    size_t m_lock_window;

    /**
     * @brief Calculate variance of a range
     */
    double calculate_variance(std::vector<double>::const_iterator begin,
                              std::vector<double>::const_iterator end) const {
        if (begin == end) return 0.0;

        double sum = 0.0;
        double sum_sq = 0.0;
        size_t n = 0;

        for (auto it = begin; it != end; ++it) {
            sum += *it;
            sum_sq += (*it) * (*it);
            n++;
        }

        double mean = sum / n;
        return sum_sq / n - mean * mean;
    }
};

// ============================================================================
// Loop Bandwidth Analyzer
// ============================================================================

/**
 * @class LoopBandwidthAnalyzer
 * @brief Calculates theoretical CDR loop parameters
 */
class LoopBandwidthAnalyzer {
public:
    /**
     * @brief Calculate theoretical loop bandwidth
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param sample_rate Sample rate (Hz)
     * @return Loop bandwidth (Hz)
     * 
     * @note Based on linearized model: BW ≈ sqrt(Ki * Fs) / (2*pi)
     */
    static double calculate_theoretical_bandwidth(double kp, double ki, double sample_rate) {
        if (ki <= 0.0 || sample_rate <= 0.0) return 0.0;
        double omega_n = std::sqrt(ki * sample_rate);
        return omega_n / (2.0 * M_PI);
    }

    /**
     * @brief Calculate damping factor
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param sample_rate Sample rate (Hz)
     * @return Damping factor (zeta)
     * 
     * @note Based on linearized model: zeta = Kp / (2 * omega_n)
     */
    static double calculate_damping_factor(double kp, double ki, double sample_rate) {
        if (ki <= 0.0 || sample_rate <= 0.0) return 0.0;
        double omega_n = std::sqrt(ki * sample_rate);
        return kp / (2.0 * omega_n);
    }

    /**
     * @brief Calculate phase margin
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param sample_rate Sample rate (Hz)
     * @return Phase margin (degrees)
     */
    static double calculate_phase_margin(double kp, double ki, double sample_rate) {
        double zeta = calculate_damping_factor(kp, ki, sample_rate);
        if (zeta < 0.1) return 0.0;
        
        // Approximate phase margin for 2nd order system
        // PM ≈ arctan(2*zeta / sqrt(sqrt(1+4*zeta^4) - 2*zeta^2))
        double zeta2 = zeta * zeta;
        double zeta4 = zeta2 * zeta2;
        double term = std::sqrt(1 + 4 * zeta4) - 2 * zeta2;
        
        if (term <= 0) return 90.0;  // Highly damped
        
        double pm_rad = std::atan(2 * zeta / std::sqrt(term));
        return pm_rad * 180.0 / M_PI;
    }

    /**
     * @brief Calculate natural frequency
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param sample_rate Sample rate (Hz)
     * @return Natural frequency (Hz)
     */
    static double calculate_natural_frequency(double kp, double ki, double sample_rate) {
        if (ki <= 0.0 || sample_rate <= 0.0) return 0.0;
        return std::sqrt(ki * sample_rate) / (2.0 * M_PI);
    }
};

// ============================================================================
// Jitter Tolerance Tester
// ============================================================================

/**
 * @class JitterToleranceTester
 * @brief Helper class for jitter tolerance testing
 */
class JitterToleranceTester {
public:
    /**
     * @brief Estimate jitter tolerance at given frequency
     * @param frequency Jitter frequency (Hz)
     * @param loop_bandwidth CDR loop bandwidth (Hz)
     * @param ui_period Unit interval (s)
     * @return Estimated jitter tolerance (s)
     * 
     * @note Simplified model based on JTOL curve characteristics
     */
    static double estimate_jitter_tolerance(double frequency, 
                                           double loop_bandwidth,
                                           double ui_period) {
        if (loop_bandwidth <= 0 || ui_period <= 0) return 0.0;
        
        // Low frequency: full UI tracking
        if (frequency < loop_bandwidth / 10) {
            return ui_period;  // ~1 UI
        }
        // High frequency: reduced tolerance
        else if (frequency > loop_bandwidth * 10) {
            return ui_period * 0.3;  // ~0.3 UI intrinsic tolerance
        }
        // Mid frequency: -20dB/decade rolloff
        else {
            double ratio = frequency / loop_bandwidth;
            return ui_period / std::sqrt(1 + ratio * ratio);
        }
    }
};

// ============================================================================
// BER Calculator
// ============================================================================

/**
 * @class BERCalculator
 * @brief Bit Error Rate calculation utilities
 */
class BERCalculator {
public:
    /**
     * @brief Calculate BER from received and reference data
     * @param received Received bit sequence
     * @param reference Reference bit sequence
     * @return Bit error rate
     */
    static double calculate_ber(const std::vector<double>& received,
                               const std::vector<double>& reference) {
        if (received.empty() || reference.empty()) return 0.0;

        size_t n = std::min(received.size(), reference.size());
        size_t errors = 0;

        for (size_t i = 0; i < n; ++i) {
            bool rx_bit = (received[i] > 0);
            bool ref_bit = (reference[i] > 0);
            if (rx_bit != ref_bit) {
                errors++;
            }
        }

        return static_cast<double>(errors) / n;
    }

    /**
     * @brief Calculate Q-factor from BER
     * @param ber Bit error rate
     * @return Q-factor
     */
    static double calculate_q_factor(double ber) {
        if (ber <= 0.0 || ber >= 0.5) return 0.0;
        return std::sqrt(2.0) * erfcinv(2.0 * ber);
    }

    /**
     * @brief Convert Q-factor to dB
     * @param q Q-factor
     * @return Q-factor in dB
     */
    static double q_to_db(double q) {
        if (q <= 0.0) return -100.0;
        return 20.0 * std::log10(q);
    }

private:
    /**
     * @brief Inverse complementary error function (approximation)
     */
    static double erfcinv(double y) {
        if (y <= 0.0 || y >= 2.0) return 0.0;
        
        // Approximation using Newton's method
        double x = std::sqrt(-std::log(y / 2.0));
        
        // Refinement
        for (int i = 0; i < 3; ++i) {
            double erfc_x = std::erfc(x);
            double deriv = -2.0 / std::sqrt(M_PI) * std::exp(-x * x);
            x = x - (erfc_x - y) / deriv;
        }
        
        return x;
    }
};

} // namespace tb
} // namespace serdes

#endif // TB_RX_CDR_HELPERS_H
