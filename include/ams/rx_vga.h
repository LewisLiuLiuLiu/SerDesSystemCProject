#ifndef SERDES_RX_VGA_H
#define SERDES_RX_VGA_H

#include <systemc-ams>
#include "common/parameters.h"
#include <random>

namespace serdes {

/**
 * @brief Variable Gain Amplifier (VGA) - Differential I/O
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
class RxVgaTdf : public sca_tdf::sca_module {
public:
    // Differential inputs
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    // Power supply input (optional, for PSRR)
    sca_tdf::sca_in<double> vdd;
    
    // Differential outputs
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    // AGC gain command (DE domain, from AdaptionDe)
    // Linear gain. When > 0 it overrides the static dc_gain by scaling the
    // fixed transfer function (which embeds dc_gain) by cmd/dc_gain at
    // runtime. When <= 0 (e.g. unbound/driven 0), the static dc_gain is used.
    sca_tdf::sca_de::sca_in<double> gain_cmd_de;
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params VGA parameters
     */
    RxVgaTdf(sc_core::sc_module_name nm, const RxVgaParams& params);
    
    /**
     * @brief Destructor - clean up resources
     */
    ~RxVgaTdf();
    
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
    RxVgaParams m_params;
    
    // Linear transfer function filters (stack objects)
    sca_tdf::sca_ltf_nd m_ltf_vga;      // Main VGA filter
    sca_tdf::sca_ltf_nd m_ltf_psrr;     // PSRR path filter
    sca_tdf::sca_ltf_nd m_ltf_cmrr;     // CMRR path filter
    sca_tdf::sca_ltf_nd m_ltf_cmfb;     // CMFB loop filter
    
    // Transfer function coefficients for main VGA
    sca_util::sca_vector<double> m_num_vga;   // Numerator coefficients
    sca_util::sca_vector<double> m_den_vga;   // Denominator coefficients
    
    // Transfer function coefficients for PSRR
    sca_util::sca_vector<double> m_num_psrr;
    sca_util::sca_vector<double> m_den_psrr;
    
    // Transfer function coefficients for CMRR
    sca_util::sca_vector<double> m_num_cmrr;
    sca_util::sca_vector<double> m_den_cmrr;
    
    // Transfer function coefficients for CMFB
    sca_util::sca_vector<double> m_num_cmfb;
    sca_util::sca_vector<double> m_den_cmfb;
    
    // Filter enable flags
    bool m_vga_filter_enabled;
    bool m_psrr_enabled;
    bool m_cmrr_enabled;
    bool m_cmfb_enabled;
    
    // Internal states
    double m_vcm_prev;                   // Previous common mode output
    double m_out_p_prev;                 // Previous out_p for CMFB measurement
    double m_out_n_prev;                 // Previous out_n for CMFB measurement
    
    // Random number generator for noise
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;
    
    /**
     * @brief Build transfer function coefficients from zeros and poles
     * @param zeros Zero frequencies in Hz
     * @param poles Pole frequencies in Hz
     * @param dc_gain DC gain
     * @param num Output numerator coefficients
     * @param den Output denominator coefficients
     * 
     * Transfer function: H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
     * where wz_i = 2*pi*zeros[i], wp_j = 2*pi*poles[j]
     */
    void build_transfer_function(
        const std::vector<double>& zeros,
        const std::vector<double>& poles,
        double dc_gain,
        sca_util::sca_vector<double>& num,
        sca_util::sca_vector<double>& den);
    
    /**
     * @brief Multiply two polynomials
     * @param p1 First polynomial coefficients [c0, c1, c2, ...] for c0 + c1*s + c2*s^2 + ...
     * @param p2 Second polynomial coefficients
     * @return Result polynomial coefficients
     */
    std::vector<double> poly_multiply(
        const std::vector<double>& p1,
        const std::vector<double>& p2);
    
    /**
     * @brief Apply soft saturation using tanh
     * @param x Input value
     * @param Vsat Saturation voltage
     * @return Saturated output
     */
    double apply_saturation(double x, double Vsat);
};

} // namespace serdes

#endif // SERDES_RX_VGA_H
