#include "ams/rx_sampler.h"
#include <cmath>
#include <stdexcept>

namespace serdes {

RxSamplerTdf::RxSamplerTdf(sc_core::sc_module_name nm, const RxSamplerParams& params)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , clk_sample("clk_sample")
    , sampling_trigger("sampling_trigger")
    , vref_threshold_de("vref_threshold_de")
    , data_out("data_out")
    , data_out_de("data_out_de")
    , m_params(params)
    , m_prev_bit(false)
    , m_last_sampled_bit(false)
    , m_rng(params.noise_seed)
    , m_noise_dist(0.0, params.noise_sigma)
    , m_decision_dist(0.0, 1.0)
{
    // Validate parameters during construction
    validate_parameters();
}

void RxSamplerTdf::set_attributes() {
    // Set input and output rates
    in_p.set_rate(1);
    in_n.set_rate(1);
    clk_sample.set_rate(1);
    sampling_trigger.set_rate(1);
    data_out.set_rate(1);
    // data_out_de is DE domain, no need to set rate
    // Inherit timestep from upstream modules
}

void RxSamplerTdf::initialize() {
    // Initialize previous bit state
    m_prev_bit = false;
    m_last_sampled_bit = false;
    
    // Reset random number generator with configured seed
    m_rng.seed(m_params.noise_seed);
    m_noise_dist = std::normal_distribution<double>(0.0, m_params.noise_sigma);
}

void RxSamplerTdf::processing() {
    // Pure passive sampler: sample on trigger, hold otherwise
    // CDR generates triggers - Sampler just samples and holds
    
    bool trigger = sampling_trigger.read();
    
    if (trigger) {
        // Read differential inputs
        double v_in_p = in_p.read();
        double v_in_n = in_n.read();
        
        // Calculate differential voltage
        double v_diff = v_in_p - v_in_n;
        
        // Effective threshold: dynamic Vref command (>0) overrides static threshold
        double thr = m_params.threshold;
        double vref_cmd = vref_threshold_de.read();
        if (vref_cmd > 0.0) {
            thr = m_params.threshold_invert ? -vref_cmd : vref_cmd;
        }
        
        // Apply offset if enabled
        if (m_params.offset_enable) {
            v_diff += m_params.offset_value;
        }
        
        // Inject noise if enabled
        if (m_params.noise_enable) {
            v_diff += m_noise_dist(m_rng);
        }
        
        // Make decision and save
        m_last_sampled_bit = make_decision(v_diff, thr);
    }
    // else: no trigger, keep previous value (sample-and-hold)
    
    // Update previous bit state (for hysteresis in decision)
    m_prev_bit = m_last_sampled_bit;
    
    // Write outputs (always the last sampled value)
    data_out.write(m_last_sampled_bit ? 1.0 : 0.0);
    data_out_de.write(m_last_sampled_bit);
}

void RxSamplerTdf::validate_parameters() {
    // 注：hysteresis 已从判决路径移除（2026-10，配置兼容保留），
    // 不再校验 hysteresis < resolution。
    
    // Validate phase source parameter
    if (m_params.phase_source != "clock" && m_params.phase_source != "phase") {
        throw std::invalid_argument(
            "Phase source must be either 'clock' or 'phase'. "
            "Current value: " + m_params.phase_source
        );
    }
}

bool RxSamplerTdf::make_decision(double v_diff, double thr) {
    // 2026-10 重审：迟滞不参与判决。
    // 真实数据 slicer 没有刻意的 Schmitt 死区（那是信号检测/静噪比较器的
    // 需求）；比较器的残余记忆是 mV 级动态失调，归入 offset/噪声建模。
    // hysteresis 参数保留仅为配置兼容（被忽略）。
    // 判决 = 门限比较 + 门限周围的亚稳态模糊区。
    if (std::abs(v_diff - thr) < m_params.resolution) {
        return (m_decision_dist(m_rng) < 0.5) ? false : true;
    }
    return v_diff > thr;
}

} // namespace serdes
