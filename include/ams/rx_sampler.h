#ifndef SERDES_RX_SAMPLER_H
#define SERDES_RX_SAMPLER_H
#include <systemc-ams>
#include "common/parameters.h"
#include <random>

namespace serdes {

/**
 * @brief Sampler module for SerDes receiver
 * 
 * Features:
 * - Differential input/output (in_p, in_n, data_out)
 * - Clock-driven or phase-driven sampling
 * - Hysteresis-based decision with Schmitt trigger effect
 * - Fuzzy decision mechanism for resolution region
 * - Offset and noise injection
 * - Parameter validation
 */
class RxSamplerTdf : public sca_tdf::sca_module {
public:
    // Differential inputs
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    // Clock or sampling trigger inputs (select based on phase_source parameter)
    sca_tdf::sca_in<double> clk_sample;      // Clock input for clock-driven mode
    sca_tdf::sca_in<bool> sampling_trigger;  // Sampling trigger from CDR (phase-driven mode)
    
    /**
     * @brief 动态判决门限（DE 域，来自 AdaptionDe.vref_cmd）
     *
     * 输入为正值标量（V）。>0 时覆盖静态 threshold：
     *   thr = threshold_invert ? -vref_cmd : vref_cmd
     * 未驱动（0）时保持静态 threshold（主采样器即 slicer@0）。
     * 2026-10：Vref 闭环接线（单控制器：AdaptionDe 生成 → 比较器门限）。
     */
    sca_tdf::sca_de::sca_in<double> vref_threshold_de;
    
    // Digital outputs
    sca_tdf::sca_out<double> data_out;      // TDF domain output (analog-compatible)
    sca_tdf::sca_de::sca_out<bool> data_out_de;  // TDF to DE domain bridge output
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params Sampler parameters
     * @note Initializes data_out_de port
     */
    RxSamplerTdf(sc_core::sc_module_name nm, const RxSamplerParams& params);
    
    /**
     * @brief Get last sampled bit value
     * @return Last sampled bit (true=1, false=0)
     */
    bool get_last_sampled_bit() const { return m_last_sampled_bit; }
    
    /**
     * @brief Set TDF module attributes
     */
    void set_attributes();
    
    /**
     * @brief Initialize sampler internal states
     */
    void initialize();
    
    /**
     * @brief Main processing function
     */
    void processing();
    
private:
    RxSamplerParams m_params;
    
    // Internal states
    bool m_prev_bit;
    bool m_last_sampled_bit;      ///< Last sampled bit value (held between triggers)
    
    // Random number generator for noise and fuzzy decision
    std::mt19937 m_rng;
    std::normal_distribution<double> m_noise_dist;
    std::uniform_real_distribution<double> m_decision_dist;
    
    /**
     * @brief Validate sampler parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate_parameters();
    
    /**
     * @brief Make decision based on input voltage
     * @param v_diff Differential input voltage
     * @param thr Effective decision threshold (static or dynamic Vref)
     * @return Decision result (true for high, false for low)
     */
    bool make_decision(double v_diff, double thr);
};
}
#endif
