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
     * @param sample_rate Sampling rate in Hz (default: 80 GHz)
     * @param seed Random seed for RNG (default: 12345)
     */
    WaveGenerationTdf(sc_core::sc_module_name nm, 
                      const WaveGenParams& params,
                      double sample_rate = 80e9,
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
    
private:
    /**
     * @brief Generate next PRBS bit using LFSR
     * @return true for bit 1, false for bit 0
     */
    bool generate_prbs_bit();
    
    WaveGenParams m_params;
    unsigned int m_lfsr_state;
    double m_sample_rate;
    double m_time;
    unsigned int m_seed;
    std::mt19937 m_rng;
};

} // namespace serdes

#endif // SERDES_WAVE_GENERATION_H
