#include "ams/rx_vga.h"
#include <cmath>
#include <sstream>

namespace serdes {

RxVgaTdf::RxVgaTdf(sc_core::sc_module_name nm, const RxVgaParams& params)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , vdd("vdd")
    , out_p("out_p")
    , out_n("out_n")
    , gain_cmd_de("gain_cmd_de")
    , m_params(params)
    , m_vga_filter_enabled(false)
    , m_psrr_enabled(false)
    , m_cmrr_enabled(false)
    , m_cmfb_enabled(false)
    , m_vcm_prev(params.vcm_out)
    , m_out_p_prev(params.vcm_out)
    , m_out_n_prev(params.vcm_out)
    , m_rng(std::random_device{}())
    , m_noise_dist(0.0, params.vnoise_sigma)
{
}

RxVgaTdf::~RxVgaTdf() {
    // No dynamic memory to clean up - all filters are stack objects
}


void RxVgaTdf::set_attributes() {
    in_p.set_rate(1);
    in_n.set_rate(1);
    vdd.set_rate(1);
    out_p.set_rate(1);
    out_n.set_rate(1);
    // Inherit timestep from upstream modules
}

void RxVgaTdf::initialize() {
    // Initialize states
    m_vcm_prev = m_params.vcm_out;
    m_out_p_prev = m_params.vcm_out;
    m_out_n_prev = m_params.vcm_out;
    
    // Build main VGA transfer function if zeros or poles are defined
    if (!m_params.zeros.empty() || !m_params.poles.empty()) {
        build_transfer_function(m_params.zeros, m_params.poles, 
                               m_params.dc_gain, m_num_vga, m_den_vga);
        m_vga_filter_enabled = true;
    } else {
        // No zeros/poles defined, use simple DC gain
        m_num_vga.resize(1);
        m_num_vga(0) = m_params.dc_gain;
        m_den_vga.resize(1);
        m_den_vga(0) = 1.0;
        m_vga_filter_enabled = false;
    }
    
    // Build PSRR transfer function if enabled
    if (m_params.psrr.enable) {
        build_transfer_function(m_params.psrr.zeros, m_params.psrr.poles,
                               m_params.psrr.gain, m_num_psrr, m_den_psrr);
        m_psrr_enabled = true;
    }
    
    // Build CMRR transfer function if enabled
    if (m_params.cmrr.enable) {
        build_transfer_function(m_params.cmrr.zeros, m_params.cmrr.poles,
                               m_params.cmrr.gain, m_num_cmrr, m_den_cmrr);
        m_cmrr_enabled = true;
    }
    
    // Build CMFB transfer function if enabled
    if (m_params.cmfb.enable) {
        // CMFB is typically a low-pass filter with given bandwidth
        std::vector<double> cmfb_zeros;  // No zeros for simple integrator
        std::vector<double> cmfb_poles = {m_params.cmfb.bandwidth};
        build_transfer_function(cmfb_zeros, cmfb_poles,
                               m_params.cmfb.loop_gain, m_num_cmfb, m_den_cmfb);
        m_cmfb_enabled = true;
    }
}

void RxVgaTdf::processing() {
    // Step 1: Read differential and common-mode inputs
    double v_in_p = in_p.read();
    double v_in_n = in_n.read();
    double v_vdd = vdd.read();
    
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
    
    // Step 4: Main VGA filtering with zero-pole transfer function
    // H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
    double vout_diff_linear;
    if (m_vga_filter_enabled) {
        // Apply Laplace transfer function using sca_ltf_nd
        // The ltf_nd operator() applies the filter: output = H(s) * input
        vout_diff_linear = m_ltf_vga(m_num_vga, m_den_vga, vin_diff);
    } else {
        // Fallback to simple DC gain
        vout_diff_linear = m_params.dc_gain * vin_diff;
    }
    
    // Step 4b: AGC gain command overlay.
    // The LTF was built with the static dc_gain baked in; scale it by
    // cmd/dc_gain so a runtime gain command takes effect without
    // rebuilding filter coefficients. cmd <= 0 keeps the static gain.
    double gain_cmd = gain_cmd_de.read();
    if (gain_cmd > 0.0 && m_params.dc_gain > 0.0) {
        vout_diff_linear *= gain_cmd / m_params.dc_gain;
    }
    
    // Step 5: Apply soft saturation (tanh)
    double Vsat = 0.5 * (m_params.sat_max - m_params.sat_min);
    double vout_diff_sat = apply_saturation(vout_diff_linear, Vsat);
    
    // Step 6: PSRR path - power supply noise coupling to differential output
    // Models how VDD variations affect the differential output
    double vout_psrr = 0.0;
    if (m_psrr_enabled) {
        double vdd_deviation = v_vdd - m_params.psrr.vdd_nom;
        vout_psrr = m_ltf_psrr(m_num_psrr, m_den_psrr, vdd_deviation);
    }
    
    // Step 7: CMRR path - common-mode to differential conversion
    // Models imperfect common-mode rejection
    double vout_cmrr = 0.0;
    if (m_cmrr_enabled) {
        vout_cmrr = m_ltf_cmrr(m_num_cmrr, m_den_cmrr, vin_cm);
    }
    
    // Step 8: Combine all differential contributions
    double vout_total_diff = vout_diff_sat + vout_psrr + vout_cmrr;
    
    // Step 9: Common-mode feedback (CMFB) loop
    // CMFB regulates output common-mode voltage to target vcm_out
    double vcm_eff = m_params.vcm_out;
    if (m_cmfb_enabled) {
        // Measure current output common-mode
        double vcm_measured = 0.5 * (m_out_p_prev + m_out_n_prev);
        // Error signal: difference from target
        double vcm_error = m_params.vcm_out - vcm_measured;
        // CMFB correction through loop filter
        double vcm_correction = m_ltf_cmfb(m_num_cmfb, m_den_cmfb, vcm_error);
        vcm_eff = m_params.vcm_out + vcm_correction;
    }
    
    // Step 10: Generate differential outputs around common-mode
    double v_out_p = vcm_eff + 0.5 * vout_total_diff;
    double v_out_n = vcm_eff - 0.5 * vout_total_diff;
    
    // Step 11: Write outputs
    out_p.write(v_out_p);
    out_n.write(v_out_n);
    
    // Step 12: Update previous states for CMFB
    m_out_p_prev = v_out_p;
    m_out_n_prev = v_out_n;
}

// apply_saturation: 应用软饱和函数
// 功能说明：
// 1. 使用双曲正切函数(tanh)实现软饱和特性
// 2. 当输入信号 x 超过饱和电压 Vsat 时，输出会平滑地趋向于 ±Vsat
// 3. 这模拟了实际模拟电路中的非线性饱和效应
// 4. 相比硬限幅(hard clipping)，tanh 函数提供了更真实的模拟电路行为
double RxVgaTdf::apply_saturation(double x, double Vsat) {
    if (Vsat <= 0.0) {
        return x;  // No saturation
    }
    return std::tanh(x / Vsat) * Vsat;
}

// build_transfer_function: 从零极点构建传递函数系数
// 传递函数形式: H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
// 其中 wz_i = 2*pi*zeros[i], wp_j = 2*pi*poles[j]
//
// 对于 sca_ltf_nd，多项式系数格式为:
// num(s) = num[0] + num[1]*s + num[2]*s^2 + ...
// den(s) = den[0] + den[1]*s + den[2]*s^2 + ...
void RxVgaTdf::build_transfer_function(
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

// poly_multiply: 多项式乘法
// 输入: p1 = [a0, a1, a2, ...] 表示 a0 + a1*s + a2*s^2 + ...
//       p2 = [b0, b1, b2, ...] 表示 b0 + b1*s + b2*s^2 + ...
// 输出: result = p1 * p2 的系数
std::vector<double> RxVgaTdf::poly_multiply(
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

} // namespace serdes
