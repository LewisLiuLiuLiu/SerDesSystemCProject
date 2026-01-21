#ifndef SERDES_TX_DRIVER_H
#define SERDES_TX_DRIVER_H

#include <systemc-ams>
#include "common/parameters.h"
#include <vector>
#include <string>

namespace serdes {

/**
 * @brief TX Driver TDF Module
 * 
 * The TX Driver is the final stage of the SerDes transmit chain.
 * It converts differential input signals into properly driven analog
 * outputs with support for:
 * - Bandwidth limiting (configurable poles)
 * - Soft/hard saturation
 * - PSRR (Power Supply Rejection Ratio) modeling
 * - Differential imbalance (gain mismatch, skew)
 * - Slew rate limiting
 * - Impedance matching
 * 
 * 8-Stage Processing Flow:
 * 1. Input reading (differential)
 * 2. DC gain adjustment
 * 3. Bandwidth limiting (pole filtering)
 * 4. Nonlinear saturation (soft/hard)
 * 5. PSRR path (VDD ripple coupling)
 * 6. Differential imbalance
 * 7. Slew rate limiting
 * 8. Impedance matching and output
 */
class TxDriverTdf : public sca_tdf::sca_module {
public:
    // ========================================================================
    // TDF Ports
    // ========================================================================
    
    // Differential input ports
    sca_tdf::sca_in<double> in_p;      ///< Differential input positive
    sca_tdf::sca_in<double> in_n;      ///< Differential input negative
    
    // Power supply input (for PSRR modeling)
    sca_tdf::sca_in<double> vdd;       ///< Power supply voltage
    
    // Differential output ports
    sca_tdf::sca_out<double> out_p;    ///< Differential output positive
    sca_tdf::sca_out<double> out_n;    ///< Differential output negative
    
    // ========================================================================
    // Constructor / Destructor
    // ========================================================================
    
    /**
     * @brief Construct a new TxDriverTdf object
     * @param nm Module name
     * @param params Driver parameters
     */
    TxDriverTdf(sc_core::sc_module_name nm, const TxDriverParams& params);
    
    /**
     * @brief Destructor
     */
    ~TxDriverTdf();
    
    // ========================================================================
    // TDF Callbacks
    // ========================================================================
    
    /**
     * @brief Set TDF attributes (sampling rate, timestep)
     */
    void set_attributes() override;
    
    /**
     * @brief Initialize internal state and build transfer functions
     */
    void initialize() override;
    
    /**
     * @brief Main processing function - 8-stage signal processing
     */
    void processing() override;
    
private:
    // ========================================================================
    // Parameters
    // ========================================================================
    TxDriverParams m_params;
    
    // ========================================================================
    // Bandwidth Filter (Pole-based lowpass)
    // ========================================================================
    sca_tdf::sca_ltf_nd m_bw_filter;
    sca_util::sca_vector<double> m_num_bw;
    sca_util::sca_vector<double> m_den_bw;
    bool m_bw_filter_enabled;
    
    // ========================================================================
    // PSRR Filter
    // ========================================================================
    sca_tdf::sca_ltf_nd m_psrr_filter;
    sca_util::sca_vector<double> m_num_psrr;
    sca_util::sca_vector<double> m_den_psrr;
    bool m_psrr_enabled;
    
    // ========================================================================
    // State Variables
    // ========================================================================
    double m_prev_vout_p;              ///< Previous output positive (for slew rate)
    double m_prev_vout_n;              ///< Previous output negative (for slew rate)
    double m_prev_vin_diff;            ///< Previous input differential (for skew)
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    /**
     * @brief Build transfer function coefficients from zeros and poles
     * @param zeros Zero frequencies (Hz)
     * @param poles Pole frequencies (Hz)
     * @param dc_gain DC gain (linear)
     * @param num Output numerator coefficients
     * @param den Output denominator coefficients
     */
    void build_transfer_function(
        const std::vector<double>& zeros,
        const std::vector<double>& poles,
        double dc_gain,
        sca_util::sca_vector<double>& num,
        sca_util::sca_vector<double>& den);
    
    /**
     * @brief Polynomial multiplication (convolution)
     * @param p1 First polynomial coefficients
     * @param p2 Second polynomial coefficients
     * @return Product polynomial coefficients
     */
    std::vector<double> poly_multiply(
        const std::vector<double>& p1,
        const std::vector<double>& p2);
    
    /**
     * @brief Apply soft saturation using tanh function
     * @param x Input value
     * @param Vsat Saturation voltage (maximum output)
     * @param Vlin Linear range parameter
     * @return Saturated output value
     */
    double apply_soft_saturation(double x, double Vsat, double Vlin);
    
    /**
     * @brief Apply hard saturation (clipping)
     * @param x Input value
     * @param Vsat Saturation voltage (maximum output)
     * @return Clipped output value
     */
    double apply_hard_saturation(double x, double Vsat);
    
    /**
     * @brief Apply slew rate limiting
     * @param v_new New voltage value
     * @param v_prev Previous voltage value
     * @param dt Time step (seconds)
     * @param SR_max Maximum slew rate (V/s)
     * @return Slew-rate limited voltage
     */
    double apply_slew_rate_limit(double v_new, double v_prev, double dt, double SR_max);
};

} // namespace serdes

#endif // SERDES_TX_DRIVER_H
