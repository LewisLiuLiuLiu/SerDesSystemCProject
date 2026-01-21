/**
 * @file wave_gen_helpers.h
 * @brief Helper modules for Wave Generation testbench
 * 
 * Provides:
 * - WaveMonitor: TDF module for waveform recording
 * - WaveformStats: Statistics data structure
 * - StatisticsAnalyzer: Utility class for statistical analysis
 */

#ifndef WAVE_GEN_HELPERS_H
#define WAVE_GEN_HELPERS_H

#include <systemc-ams>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace serdes {

// ============================================================================
// WaveformStats - Statistics data structure
// ============================================================================

struct WaveformStats {
    double mean;              // Mean value (V)
    double rms;               // RMS value (V)
    double peak_to_peak;      // Peak-to-peak value (V)
    double min_value;         // Minimum value (V)
    double max_value;         // Maximum value (V)
    double balance;           // Code balance (0-1, ratio of imbalance)
    int positive_count;       // Count of positive samples
    int negative_count;       // Count of negative samples
    int edge_count;           // Number of transitions
    size_t sample_count;      // Total number of samples
    
    WaveformStats()
        : mean(0.0)
        , rms(0.0)
        , peak_to_peak(0.0)
        , min_value(0.0)
        , max_value(0.0)
        , balance(0.0)
        , positive_count(0)
        , negative_count(0)
        , edge_count(0)
        , sample_count(0)
    {}
    
    void print(std::ostream& os = std::cout) const {
        os << "=== Waveform Statistics ===" << std::endl;
        os << "  Samples: " << sample_count << std::endl;
        os << "  Mean: " << mean << " V" << std::endl;
        os << "  RMS: " << rms << " V" << std::endl;
        os << "  Peak-to-peak: " << peak_to_peak << " V" << std::endl;
        os << "  Min: " << min_value << " V" << std::endl;
        os << "  Max: " << max_value << " V" << std::endl;
        os << "  Balance: " << (balance * 100.0) << " %" << std::endl;
        os << "  Positive count: " << positive_count << std::endl;
        os << "  Negative count: " << negative_count << std::endl;
        os << "  Edge count: " << edge_count << std::endl;
    }
};

// ============================================================================
// WaveMonitor - TDF module for waveform recording
// ============================================================================

class WaveMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    WaveMonitor(sc_core::sc_module_name nm, 
                double sample_rate = 80e9,
                size_t max_samples = 100000)
        : sca_tdf::sca_module(nm)
        , in("in")
        , m_sample_rate(sample_rate)
        , m_max_samples(max_samples)
        , m_time(0.0)
        , m_prev_value(0.0)
        , m_first_sample(true)
    {}
    
    void set_attributes() {
        in.set_rate(1);
    }
    
    void processing() {
        if (m_samples.size() < m_max_samples) {
            double value = in.read();
            m_samples.push_back(value);
            m_timestamps.push_back(m_time);
            
            // Count edges
            if (!m_first_sample) {
                if ((value > 0 && m_prev_value < 0) || 
                    (value < 0 && m_prev_value > 0)) {
                    m_edge_count++;
                }
            }
            m_prev_value = value;
            m_first_sample = false;
        }
        m_time += 1.0 / m_sample_rate;
    }
    
    // Accessors
    const std::vector<double>& get_samples() const { return m_samples; }
    const std::vector<double>& get_timestamps() const { return m_timestamps; }
    size_t get_sample_count() const { return m_samples.size(); }
    int get_edge_count() const { return m_edge_count; }
    
    // Calculate statistics
    WaveformStats calculate_stats() const {
        WaveformStats stats;
        
        if (m_samples.empty()) {
            return stats;
        }
        
        stats.sample_count = m_samples.size();
        
        // Min/Max
        stats.min_value = *std::min_element(m_samples.begin(), m_samples.end());
        stats.max_value = *std::max_element(m_samples.begin(), m_samples.end());
        stats.peak_to_peak = stats.max_value - stats.min_value;
        
        // Mean
        double sum = std::accumulate(m_samples.begin(), m_samples.end(), 0.0);
        stats.mean = sum / stats.sample_count;
        
        // RMS
        double sum_sq = 0.0;
        for (const auto& s : m_samples) {
            sum_sq += s * s;
        }
        stats.rms = std::sqrt(sum_sq / stats.sample_count);
        
        // Code balance
        for (const auto& s : m_samples) {
            if (s > 0) stats.positive_count++;
            else stats.negative_count++;
        }
        stats.balance = std::abs(stats.positive_count - stats.negative_count) / 
                        static_cast<double>(stats.sample_count);
        
        // Edge count
        stats.edge_count = m_edge_count;
        
        return stats;
    }
    
    // Save to CSV file
    bool save_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < m_samples.size(); ++i) {
            file << std::scientific << m_timestamps[i] << "," << m_samples[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved " << m_samples.size() << " samples to " << filename << std::endl;
        return true;
    }
    
    void clear() {
        m_samples.clear();
        m_timestamps.clear();
        m_time = 0.0;
        m_edge_count = 0;
        m_first_sample = true;
    }
    
private:
    double m_sample_rate;
    size_t m_max_samples;
    double m_time;
    double m_prev_value;
    bool m_first_sample;
    int m_edge_count = 0;
    std::vector<double> m_samples;
    std::vector<double> m_timestamps;
};

// ============================================================================
// StatisticsAnalyzer - Utility class for statistical analysis
// ============================================================================

class StatisticsAnalyzer {
public:
    // Detect rising edge time in samples
    static double find_rising_edge(const std::vector<double>& samples,
                                   const std::vector<double>& timestamps) {
        for (size_t i = 1; i < samples.size(); ++i) {
            if (samples[i] > 0 && samples[i-1] < 0) {
                return timestamps[i];
            }
        }
        return -1.0;  // Not found
    }
    
    // Detect falling edge time in samples
    static double find_falling_edge(const std::vector<double>& samples,
                                    const std::vector<double>& timestamps) {
        for (size_t i = 1; i < samples.size(); ++i) {
            if (samples[i] < 0 && samples[i-1] > 0) {
                return timestamps[i];
            }
        }
        return -1.0;  // Not found
    }
    
    // Measure pulse width (time between first rising and falling edge)
    static double measure_pulse_width(const std::vector<double>& samples,
                                      const std::vector<double>& timestamps) {
        double rise_time = -1.0;
        double fall_time = -1.0;
        
        for (size_t i = 1; i < samples.size(); ++i) {
            if (rise_time < 0 && samples[i] > 0 && samples[i-1] <= 0) {
                rise_time = timestamps[i];
            } else if (rise_time >= 0 && fall_time < 0 && samples[i] < 0 && samples[i-1] >= 0) {
                fall_time = timestamps[i];
                break;
            }
        }
        
        if (rise_time >= 0 && fall_time >= 0) {
            return fall_time - rise_time;
        }
        return -1.0;  // Could not measure
    }
    
    // Count transitions
    static int count_transitions(const std::vector<double>& samples) {
        int count = 0;
        for (size_t i = 1; i < samples.size(); ++i) {
            if ((samples[i] > 0 && samples[i-1] < 0) ||
                (samples[i] < 0 && samples[i-1] > 0)) {
                count++;
            }
        }
        return count;
    }
    
    // Verify all samples are within expected NRZ levels
    static bool verify_nrz_levels(const std::vector<double>& samples,
                                  double tolerance = 1e-6) {
        for (const auto& s : samples) {
            if (std::abs(s - 1.0) > tolerance && std::abs(s + 1.0) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

} // namespace serdes

#endif // WAVE_GEN_HELPERS_H
