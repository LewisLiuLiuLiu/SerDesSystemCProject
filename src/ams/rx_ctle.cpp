#include "ams/rx_ctle.h"
#include <cmath>
#include <sstream>

namespace serdes {

RxCtleTdf::RxCtleTdf(sc_core::sc_module_name nm, const RxCtleParams& params)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , vdd("vdd")
    , out_p("out_p")
    , out_n("out_n")
    , m_params(params)
    , m_H_ctle(nullptr)
    , m_H_psrr(nullptr)
    , m_H_cmrr(nullptr)
    , m_H_cmfb(nullptr)
    , m_vcm_prev(params.vcm_out)
    , m_out_p_prev(params.vcm_out)
    , m_out_n_prev(params.vcm_out)
    , m_rng(std::random_device{}())
    , m_noise_dist(0.0, params.vnoise_sigma)
{
}

RxCtleTdf::~RxCtleTdf() {
    // Clean up dynamically allocated filter objects
    if (m_H_ctle) delete m_H_ctle;
    if (m_H_psrr) delete m_H_psrr;
    if (m_H_cmrr) delete m_H_cmrr;
    if (m_H_cmfb) delete m_H_cmfb;
}

void RxCtleTdf::set_attributes() {
    in_p.set_rate(1);
    in_n.set_rate(1);
    vdd.set_rate(1);
    out_p.set_rate(1);
    out_n.set_rate(1);
    
    // Set default timestep if needed
    set_timestep(1.0 / 100e9);  // 100 GHz sampling for high-speed SerDes
}

void RxCtleTdf::initialize() {
    // Build main CTLE transfer function H_ctle(s)
    m_H_ctle = build_transfer_function(
        m_params.zeros,
        m_params.poles,
        m_params.dc_gain
    );
    
    // Build PSRR transfer function if enabled
    if (m_params.psrr.enable && m_params.psrr.gain != 0.0) {
        m_H_psrr = build_transfer_function(
            m_params.psrr.zeros,
            m_params.psrr.poles,
            m_params.psrr.gain
        );
    }
    
    // Build CMRR transfer function if enabled
    if (m_params.cmrr.enable && m_params.cmrr.gain != 0.0) {
        m_H_cmrr = build_transfer_function(
            m_params.cmrr.zeros,
            m_params.cmrr.poles,
            m_params.cmrr.gain
        );
    }
    
    // Build CMFB loop filter if enabled
    if (m_params.cmfb.enable) {
        // First-order approximation: H_cmfb(s) = loop_gain / (1 + s/(2*pi*bandwidth))
        std::vector<double> cmfb_zeros;
        std::vector<double> cmfb_poles = {m_params.cmfb.bandwidth};
        m_H_cmfb = build_transfer_function(
            cmfb_zeros,
            cmfb_poles,
            m_params.cmfb.loop_gain
        );
    }
    
    // Initialize states
    m_vcm_prev = m_params.vcm_out;
    m_out_p_prev = m_params.vcm_out;
    m_out_n_prev = m_params.vcm_out;
}

void RxCtleTdf::processing() {
    // Step 1: Read differential and common-mode inputs
    double v_in_p = in_p.read();
    double v_in_n = in_n.read();
    
    // Calculate differential and common-mode inputs
    double vin_diff = v_in_p - v_in_n;
    double vin_cm = 0.5 * (v_in_p + v_in_n);
    
    // Step 2: Add offset if enabled
    if (m_params.offset_enable) {
        vin_diff += m_params.vos;
    }
    
    // Step 3: Add noise if enabled
    if (m_params.noise_enable) {
        vin_diff += m_noise_dist(m_rng);
    }
    
    // Step 4: Main CTLE filtering - apply to differential signal
    double vout_diff_linear = 0.0;
    if (m_H_ctle) {
        vout_diff_linear = m_H_ctle->estimate_nd(get_time().to_seconds(), vin_diff);
    } else {
        vout_diff_linear = m_params.dc_gain * vin_diff;
    }
    
    // Step 5: Apply soft saturation (tanh)
    double Vsat = 0.5 * (m_params.sat_max - m_params.sat_min);
    double vout_diff_sat = apply_saturation(vout_diff_linear, Vsat);
    
    // Step 6: PSRR path (optional)
    double vout_psrr_diff = 0.0;
    if (m_params.psrr.enable && m_H_psrr) {
        double vdd_current = vdd.read();
        double vdd_ripple = vdd_current - m_params.psrr.vdd_nom;
        vout_psrr_diff = m_H_psrr->estimate_nd(get_time().to_seconds(), vdd_ripple);
    }
    
    // Step 7: CMRR path (optional)
    double vout_cmrr_diff = 0.0;
    if (m_params.cmrr.enable && m_H_cmrr) {
        vout_cmrr_diff = m_H_cmrr->estimate_nd(get_time().to_seconds(), vin_cm);
    }
    
    // Step 8: Sum all differential contributions
    double vout_total_diff = vout_diff_sat + vout_psrr_diff + vout_cmrr_diff;
    
    // Step 9: Common-mode and CMFB
    double vcm_eff = m_params.vcm_out;
    
    if (m_params.cmfb.enable && m_H_cmfb) {
        // Measure previous output common mode (avoid algebraic loop)
        double vcm_meas = 0.5 * (m_out_p_prev + m_out_n_prev);
        
        // Common-mode error
        double e_cm = m_params.vcm_out - vcm_meas;
        
        // Apply CMFB loop filter
        double delta_vcm = m_H_cmfb->estimate_nd(get_time().to_seconds(), e_cm);
        
        // Effective common mode
        vcm_eff = m_params.vcm_out + delta_vcm;
    }
    
    // Step 10: Generate differential outputs
    double v_out_p = vcm_eff + 0.5 * vout_total_diff;
    double v_out_n = vcm_eff - 0.5 * vout_total_diff;
    
    // Step 11: Write outputs
    out_p.write(v_out_p);
    out_n.write(v_out_n);
    
    // Step 12: Update previous states for CMFB
    m_out_p_prev = v_out_p;
    m_out_n_prev = v_out_n;
}

sca_tdf::sca_ltf_nd* RxCtleTdf::build_transfer_function(
    const std::vector<double>& zeros,
    const std::vector<double>& poles,
    double dc_gain)
{
    if (poles.empty()) {
        // No poles - simple gain
        return nullptr;
    }
    
    // Build numerator and denominator for ltf_nd
    // H(s) = dc_gain * prod((s/wz + 1)) / prod((s/wp + 1))
    
    sca_util::sca_vector<double> num;
    sca_util::sca_vector<double> den;
    
    // Start with dc_gain in numerator
    num(0) = dc_gain;
    
    // Multiply numerator by (s/wz + 1) for each zero
    for (size_t i = 0; i < zeros.size(); ++i) {
        double wz = 2.0 * M_PI * zeros[i];
        sca_util::sca_vector<double> zero_term(2);
        zero_term(0) = 1.0;      // constant term
        zero_term(1) = 1.0 / wz; // s term coefficient
        
        // Convolve with existing numerator
        sca_util::sca_vector<double> new_num(num.length() + 1);
        for (int j = 0; j < num.length(); ++j) {
            new_num(j) += num(j) * zero_term(0);
            new_num(j + 1) += num(j) * zero_term(1);
        }
        num = new_num;
    }
    
    // Start denominator at 1
    den(0) = 1.0;
    
    // Multiply denominator by (s/wp + 1) for each pole
    for (size_t i = 0; i < poles.size(); ++i) {
        double wp = 2.0 * M_PI * poles[i];
        sca_util::sca_vector<double> pole_term(2);
        pole_term(0) = 1.0;      // constant term
        pole_term(1) = 1.0 / wp; // s term coefficient
        
        // Convolve with existing denominator
        sca_util::sca_vector<double> new_den(den.length() + 1);
        for (int j = 0; j < den.length(); ++j) {
            new_den(j) += den(j) * pole_term(0);
            new_den(j + 1) += den(j) * pole_term(1);
        }
        den = new_den;
    }
    
    // Create and return the transfer function
    return new sca_tdf::sca_ltf_nd(num, den);
}

double RxCtleTdf::apply_saturation(double x, double Vsat) {
    if (Vsat <= 0.0) {
        return x;  // No saturation
    }
    return std::tanh(x / Vsat) * Vsat;
}

} // namespace serdes
