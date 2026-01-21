#include "ams/tx_driver.h"
#include <cmath>
#include <algorithm>

namespace serdes {

// ============================================================================
// Constructor / Destructor
// ============================================================================

TxDriverTdf::TxDriverTdf(sc_core::sc_module_name nm, const TxDriverParams& params)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , vdd("vdd")
    , out_p("out_p")
    , out_n("out_n")
    , m_params(params)
    , m_bw_filter_enabled(false)
    , m_psrr_enabled(false)
    , m_prev_vout_p(params.vcm_out)
    , m_prev_vout_n(params.vcm_out)
    , m_prev_vin_diff(0.0)
{
}

TxDriverTdf::~TxDriverTdf() {
    // No dynamic memory to clean up - all filters are stack objects
}

// ============================================================================
// TDF Callbacks
// ============================================================================

void TxDriverTdf::set_attributes() {
    // Set sampling rates for all ports
    in_p.set_rate(1);
    in_n.set_rate(1);
    vdd.set_rate(1);
    out_p.set_rate(1);
    out_n.set_rate(1);
    
    // Set default timestep (100 GHz sampling for high-speed SerDes)
    set_timestep(1.0 / 100e9, sc_core::SC_SEC);
}

void TxDriverTdf::initialize() {
    // Initialize state variables
    m_prev_vout_p = m_params.vcm_out;
    m_prev_vout_n = m_params.vcm_out;
    m_prev_vin_diff = 0.0;
    
    // Build bandwidth limiting transfer function if poles are defined
    // H(s) = dc_gain / prod(1 + s/wp_j)
    if (!m_params.poles.empty()) {
        std::vector<double> no_zeros;  // TX Driver has no zeros, only poles
        build_transfer_function(no_zeros, m_params.poles,
                               m_params.dc_gain, m_num_bw, m_den_bw);
        m_bw_filter_enabled = true;
    } else {
        // No poles defined, use simple DC gain
        m_num_bw.resize(1);
        m_num_bw(0) = m_params.dc_gain;
        m_den_bw.resize(1);
        m_den_bw(0) = 1.0;
        m_bw_filter_enabled = false;
    }
    
    // Build PSRR transfer function if enabled
    if (m_params.psrr.enable) {
        std::vector<double> no_zeros;
        build_transfer_function(no_zeros, m_params.psrr.poles,
                               m_params.psrr.gain, m_num_psrr, m_den_psrr);
        m_psrr_enabled = true;
    }
}

void TxDriverTdf::processing() {
    // ========================================================================
    // Stage 1: Read differential input
    // ========================================================================
    double v_in_p = in_p.read();
    double v_in_n = in_n.read();
    double v_vdd = vdd.read();
    
    double vin_diff = v_in_p - v_in_n;
    
    // ========================================================================
    // Stage 2: Apply DC gain (handled in filter or directly)
    // ========================================================================
    double vout_diff;
    
    // ========================================================================
    // Stage 3: Bandwidth limiting (pole-based lowpass filtering)
    // ========================================================================
    if (m_bw_filter_enabled) {
        // Apply Laplace transfer function using sca_ltf_nd
        // H(s) = dc_gain / prod(1 + s/wp_j)
        vout_diff = m_bw_filter(m_num_bw, m_den_bw, vin_diff);
    } else {
        // Fallback to simple DC gain
        vout_diff = m_params.dc_gain * vin_diff;
    }
    
    // ========================================================================
    // Stage 4: Nonlinear saturation
    // ========================================================================
    double Vsat = m_params.vswing / 2.0;  // Half-swing for differential
    
    if (m_params.sat_mode == "soft") {
        vout_diff = apply_soft_saturation(vout_diff, Vsat, m_params.vlin);
    } else if (m_params.sat_mode == "hard") {
        vout_diff = apply_hard_saturation(vout_diff, Vsat);
    }
    // else "none" - no saturation applied
    
    // ========================================================================
    // Stage 5: PSRR path - power supply noise coupling
    // ========================================================================
    if (m_psrr_enabled) {
        double vdd_ripple = v_vdd - m_params.psrr.vdd_nom;
        double vpsrr = m_psrr_filter(m_num_psrr, m_den_psrr, vdd_ripple);
        vout_diff += vpsrr;
    }
    
    // ========================================================================
    // Stage 6: Differential imbalance (gain mismatch and skew)
    // ========================================================================
    // Gain mismatch: split the differential signal unequally
    double gain_p = 1.0 + (m_params.imbalance.gain_mismatch / 200.0);
    double gain_n = 1.0 - (m_params.imbalance.gain_mismatch / 200.0);
    
    // Generate single-ended outputs around common-mode
    double vout_p_raw = m_params.vcm_out + 0.5 * vout_diff * gain_p;
    double vout_n_raw = m_params.vcm_out - 0.5 * vout_diff * gain_n;
    
    // Note: Skew (phase offset) is simplified here - a full implementation
    // would use fractional delay filters. For now, we apply a simple
    // first-order approximation based on signal derivative.
    // TODO: Implement fractional delay for accurate skew modeling
    
    // ========================================================================
    // Stage 7: Slew rate limiting
    // ========================================================================
    double vout_p, vout_n;
    
    if (m_params.slew_rate.enable) {
        double dt = get_timestep().to_seconds();
        vout_p = apply_slew_rate_limit(vout_p_raw, m_prev_vout_p, dt, 
                                       m_params.slew_rate.max_slew_rate);
        vout_n = apply_slew_rate_limit(vout_n_raw, m_prev_vout_n, dt,
                                       m_params.slew_rate.max_slew_rate);
    } else {
        vout_p = vout_p_raw;
        vout_n = vout_n_raw;
    }
    
    // ========================================================================
    // Stage 8: Impedance matching (voltage division)
    // ========================================================================
    // When output impedance matches load impedance (Z0), voltage divides by 2
    // V_channel = V_driver * Z0 / (Zout + Z0)
    const double Z0 = 50.0;  // Typical transmission line impedance
    double voltage_division_factor = Z0 / (m_params.output_impedance + Z0);
    
    // Apply voltage division to get channel-side voltage
    // Note: The common-mode voltage is also affected by the division
    double vchannel_p = m_params.vcm_out * voltage_division_factor + 
                       (vout_p - m_params.vcm_out) * voltage_division_factor;
    double vchannel_n = m_params.vcm_out * voltage_division_factor + 
                       (vout_n - m_params.vcm_out) * voltage_division_factor;
    
    // Write outputs
    out_p.write(vchannel_p);
    out_n.write(vchannel_n);
    
    // Update state for next cycle
    m_prev_vout_p = vout_p;
    m_prev_vout_n = vout_n;
    m_prev_vin_diff = vin_diff;
}

// ============================================================================
// Helper Methods
// ============================================================================

void TxDriverTdf::build_transfer_function(
    const std::vector<double>& zeros,
    const std::vector<double>& poles,
    double dc_gain,
    sca_util::sca_vector<double>& num,
    sca_util::sca_vector<double>& den)
{
    // Build numerator polynomial from zeros
    // Each zero at frequency fz contributes factor (1 + s/(2*pi*fz))
    // = (1 + s/wz) which in coefficient form is [1, 1/wz]
    std::vector<double> num_poly = {dc_gain};  // Start with DC gain
    
    for (double fz : zeros) {
        if (fz > 0.0) {
            double wz = 2.0 * M_PI * fz;
            std::vector<double> factor = {1.0, 1.0 / wz};  // (1 + s/wz)
            num_poly = poly_multiply(num_poly, factor);
        }
    }
    
    // Build denominator polynomial from poles
    // Each pole at frequency fp contributes factor (1 + s/(2*pi*fp))
    std::vector<double> den_poly = {1.0};  // Start with 1
    
    for (double fp : poles) {
        if (fp > 0.0) {
            double wp = 2.0 * M_PI * fp;
            std::vector<double> factor = {1.0, 1.0 / wp};  // (1 + s/wp)
            den_poly = poly_multiply(den_poly, factor);
        }
    }
    
    // Copy to sca_vector format
    num.resize(num_poly.size());
    for (size_t i = 0; i < num_poly.size(); ++i) {
        num(i) = num_poly[i];
    }
    
    den.resize(den_poly.size());
    for (size_t i = 0; i < den_poly.size(); ++i) {
        den(i) = den_poly[i];
    }
}

std::vector<double> TxDriverTdf::poly_multiply(
    const std::vector<double>& p1,
    const std::vector<double>& p2)
{
    if (p1.empty() || p2.empty()) {
        return {1.0};
    }
    
    size_t n1 = p1.size();
    size_t n2 = p2.size();
    std::vector<double> result(n1 + n2 - 1, 0.0);
    
    // Convolution: result[k] = sum(p1[i] * p2[k-i]) for all valid i
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            result[i + j] += p1[i] * p2[j];
        }
    }
    
    return result;
}

double TxDriverTdf::apply_soft_saturation(double x, double Vsat, double Vlin) {
    // Soft saturation using tanh function
    // Output smoothly approaches ±Vsat as input increases
    // Vlin controls the linear region width
    if (Vsat <= 0.0 || Vlin <= 0.0) {
        return x;  // No saturation
    }
    return Vsat * std::tanh(x / Vlin);
}

double TxDriverTdf::apply_hard_saturation(double x, double Vsat) {
    // Hard saturation (clipping)
    // Output is clamped to ±Vsat
    if (Vsat <= 0.0) {
        return x;  // No saturation
    }
    return std::max(-Vsat, std::min(Vsat, x));
}

double TxDriverTdf::apply_slew_rate_limit(double v_new, double v_prev, 
                                          double dt, double SR_max) {
    // Slew rate limiting
    // If the rate of change exceeds SR_max, limit the output change
    if (dt <= 0.0 || SR_max <= 0.0) {
        return v_new;
    }
    
    double dV = v_new - v_prev;
    double actual_SR = std::abs(dV) / dt;
    
    if (actual_SR > SR_max) {
        // Limit the change to maximum allowed
        double max_dV = SR_max * dt;
        return v_prev + std::copysign(max_dV, dV);
    }
    
    return v_new;
}

} // namespace serdes
