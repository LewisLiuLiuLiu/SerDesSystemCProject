#ifndef SERDES_WAVE_GENERATION_H
#define SERDES_WAVE_GENERATION_H

#include <systemc-ams>
#include "common/parameters.h"
#include <random>

namespace serdes {

/**
 * @brief Wave Generation TDF Module
 * 
 * Generates test stimulus signals for SerDes system, supporting:
 * - PRBS7/9/15/23/31 pseudo-random sequences
 * - Single-bit pulse (SBR) mode for transient response testing
 * - Random Jitter (RJ) and Sinusoidal Jitter (SJ) injection
 * - NRZ modulation (+1.0V/-1.0V)
 */
class WaveGenerationTdf : public sca_tdf::sca_module {
public:
    // Output port - NRZ modulated signal
    sca_tdf::sca_out<double> out;
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params Wave generation parameters
     * @param sample_rate Sampling rate in Hz (default: 640 GHz)
     * @param ui Unit interval in seconds (default: 100ps for 10Gbps)
     * @param seed Random seed for RNG (default: 12345)
     */
    WaveGenerationTdf(sc_core::sc_module_name nm, 
                      const WaveGenParams& params,
                      double sample_rate = 640e9,
                      double ui = 100e-12,
                      unsigned int seed = 12345);
    
    // TDF callback methods
    void set_attributes() override;
    void initialize() override;
    void processing() override;
    
    // Debug interface
    unsigned int get_lfsr_state() const { return m_lfsr_state; }
    double get_current_time() const { return m_time; }
    bool is_pulse_mode() const { return m_params.single_pulse > 0.0; }
    double get_sample_rate() const { return m_sample_rate; }
    double get_ui() const { return m_ui; }
    int get_samples_per_ui() const { return m_samples_per_ui; }
    
private:
    /**
     * @brief Generate next PRBS bit using LFSR
     * @return true for bit 1, false for bit 0
     */
    bool generate_prbs_bit();

    // T-S3 A1: edge-level jitter (EDGE-01) -------------------------------------------------
    struct JitterEdge {
        long long sample_index;   // floor((nominal_time + e_total) / timestep)
        double alpha;             // fractional part in [0,1)
        double old_level;
        double new_level;
    };
    void build_jittered_edges();

    WaveGenParams m_params;
    unsigned int m_lfsr_state;
    double m_sample_rate;
    double m_ui;                    // Unit interval (seconds)
    int m_samples_per_ui;           // Oversampling ratio
    int m_sample_counter;           // Counter within current UI
    double m_current_bit_value;     // Current bit value held during UI
    // T-S4 A2: causal finite-edge shaping (PRBS path, jitter disabled)
    double m_edge_from;             // level before the current UI transition
    double m_edge_to;               // level after the current UI transition
    double m_edge_start;            // absolute time of the transition start
    double m_time;
    unsigned int m_seed;
    std::mt19937 m_rng;

    // T-S3 A1: edge-level jitter state
    std::vector<double> m_jit_levels;      // per-UI NRZ levels (+1/-1)
    std::vector<JitterEdge> m_jit_edges;   // transitions with shifted sample indices
    size_t m_jit_edge_ptr;                 // next edge to apply
    long long m_jit_sample_index;          // current 2-ps grid index
    double m_jit_last_level;               // level after the last edge
};

} // namespace serdes

#endif // SERDES_WAVE_GENERATION_H
