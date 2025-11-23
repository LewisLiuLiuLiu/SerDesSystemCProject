#ifndef SERDES_RX_CTLE_H
#define SERDES_RX_CTLE_H

#include <systemc-ams>
#include "common/parameters.h"
#include <random>

namespace serdes {

/**
 * @brief Continuous Time Linear Equalizer (CTLE) - Differential I/O
 * 
 * Features:
 * - Differential input/output (in_p, in_n, out_p, out_n)
 * - Multi-zero/multi-pole transfer function
 * - Configurable output common mode voltage (vcm_out)
 * - Input offset (vos) with enable switch
 * - Input noise injection with enable switch
 * - Output saturation limiting (soft tanh)
 * - PSRR path (power supply rejection)
 * - CMFB (common mode feedback) loop
 * - CMRR (common mode rejection) path
 */
class RxCtleTdf : public sca_tdf::sca_module {
public:
    // Differential inputs
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    // Power supply input (optional, for PSRR)
    sca_tdf::sca_in<double> vdd;
    
    // Differential outputs
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params CTLE parameters
     */
    RxCtleTdf(sc_core::sc_module_name nm, const RxCtleParams& params);
    
    /**
     * @brief Destructor - clean up filter objects
     */
    ~RxCtleTdf();
    
    /**
     * @brief Set TDF module attributes
     */
    void set_attributes() override;
    
    /**
     * @brief Initialize filters and internal states
     */
    void initialize() override;
    
    /**
     * @brief Main processing function
     */
    void processing() override;

private:
    RxCtleParams m_params;
    
    // Linear transfer function filters
    sca_tdf::sca_ltf_nd* m_H_ctle;      // Main CTLE filter
    sca_tdf::sca_ltf_nd* m_H_psrr;      // PSRR path filter
    sca_tdf::sca_ltf_nd* m_H_cmrr;      // CMRR path filter
    sca_tdf::sca_ltf_nd* m_H_cmfb;      // CMFB loop filter
    
    // Internal states
    double m_vcm_prev;                   // Previous common mode output
    double m_out_p_prev;                 // Previous out_p for CMFB measurement
    double m_out_n_prev;                 // Previous out_n for CMFB measurement
    
    // Random number generator for noise
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;
    
    /**
     * @brief Build transfer function from zeros and poles
     * @param zeros Zero frequencies in Hz
     * @param poles Pole frequencies in Hz
     * @param dc_gain DC gain
     * @return Pointer to ltf_nd filter (caller owns memory)
     */
    sca_tdf::sca_ltf_nd* build_transfer_function(
        const std::vector<double>& zeros,
        const std::vector<double>& poles,
        double dc_gain);
    
    /**
     * @brief Apply soft saturation using tanh
     * @param x Input value
     * @param Vsat Saturation voltage
     * @return Saturated output
     */
    double apply_saturation(double x, double Vsat);
};

} // namespace serdes

#endif // SERDES_RX_CTLE_H
