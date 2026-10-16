/**
 * @file rx_cdr.h
 * @brief Clock and Data Recovery (CDR) module for SerDes receiver
 * 
 * This module implements a behavioral-level CDR using Bang-Bang phase detector
 * and PI (Proportional-Integral) loop filter architecture.
 * 
 * Features:
 * - Bang-Bang phase detection based on data edge transitions
 * - Digital PI loop filter with configurable Kp and Ki gains
 * - Phase interpolator with configurable resolution and range
 * - Phase quantization and range limiting
 * 
 * @note This is a simplified implementation suitable for system-level simulation.
 *       The Bang-Bang PD uses edge polarity detection rather than a full
 *       data/edge sampler XOR architecture.
 * 
 * Sampling trigger timebase (2026-10 rewrite):
 *   For UI index k = 0,1,2,... and quantized PI phase correction qq:
 *     edge trigger instant:  k*ui + qq
 *     data trigger instant:  k*ui + sample_point*ui + qq
 *   Instants are aligned to the simulation step grid via ceil(t/timestep),
 *   so exactly 2 triggers per UI are emitted for any timestep configuration
 *   (integer scheduling — replaces floating-point phase accumulation /
 *   crossing detection, which intermittently skipped crossings; F-22/P0-1).
 * 
 * @version 0.3
 * @date 2026-10-16
 */

#ifndef SERDES_RX_CDR_H
#define SERDES_RX_CDR_H

#include <systemc-ams>
#include "common/parameters.h"

namespace serdes {

/**
 * @class RxCdrTdf
 * @brief TDF module implementing Clock and Data Recovery
 * 
 * The CDR extracts clock information from data transitions and generates
 * optimal sampling phase for the sampler module.
 * 
 * Signal Processing Flow:
 * 1. Read input data
 * 2. Edge detection (compare with previous bit)
 * 3. Bang-Bang phase detection (early/late decision)
 * 4. PI controller update (proportional + integral)
 * 5. Phase range limiting
 * 6. Phase quantization
 * 7. Output phase adjustment
 */
class RxCdrTdf : public sca_tdf::sca_module {
public:
    // ========================================================================
    // Ports
    // ========================================================================
    
    /**
     * @brief Data input port
     * Receives analog signal from DFE or sampler output
     */
    sca_tdf::sca_in<double> in;
    
    /**
     * @brief Phase adjustment output port (for monitoring/debug)
     * Outputs phase offset in seconds (s)
     * - Positive value: delay sampling (clock late)
     * - Negative value: advance sampling (clock early)
     */
    sca_tdf::sca_out<double> phase_out;
    
    /**
     * @brief Sampling trigger output port
     * Outputs true for one timestep when sampling should occur
     * Connected to sampler's sampling_trigger input
     */
    sca_tdf::sca_out<bool> sampling_trigger;

    /**
     * @brief BBPD 相位误差输出（DE 域，单位：秒）
     *
     * 输出 m_last_phase_error * ui（±UI 或 0，三元 BBPD 结果），供
     * AdaptionDe 的相位监视/冻结判据使用。写变更触发（事件驱动），
     * 避免逐采样刷 DE 事件。注意：rx_top 中承担相位锁定的是本模块内部
     * PI；AdaptionDe.cdr_pi 输出（phase_cmd）为监视量，不参与环路。
     */
    sca_tdf::sca_de::sca_out<double> phase_error_de;

    // ========================================================================
    // Constructor
    // ========================================================================
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params CDR parameters (PI controller and phase interpolator)
     */
    RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params);

    // ========================================================================
    // SystemC-AMS TDF Methods
    // ========================================================================
    
    /**
     * @brief Set module attributes (port rates)
     */
    void set_attributes();
    
    /**
     * @brief Initialize module state
     */
    void initialize();
    
    /**
     * @brief Main processing function (called each time step)
     */
    void processing();

    // ========================================================================
    // Debug Interface
    // ========================================================================
    
    /**
     * @brief Get current integral state of PI controller
     * @return Integral accumulator value
     */
    double get_integral_state() const { return m_integral; }
    
    /**
     * @brief Get last phase error from Bang-Bang PD
     * @return Phase error (+1, -1, or 0)
     */
    double get_phase_error() const { return m_last_phase_error; }
    
    /**
     * @brief Get current phase output (before quantization)
     * @return Raw phase value in seconds
     */
    double get_raw_phase() const { return m_phase; }

private:
    // ========================================================================
    // Sampling State Machine
    // ========================================================================
    
    /**
     * @brief Sampling state for double-sampling BBPD
     * 
     * Within each UI, CDR generates two triggers:
     * 1. EDGE: At UI boundary (phase = 0 or UI)
     * 2. DATA: At UI center (phase = UI/2)
     * 
     * State machine tracks which sample is expected next
     */
    enum class SampleState {
        WAIT_EDGE,   ///< Waiting for edge sample (at UI boundary)
        WAIT_DATA    ///< Waiting for data sample (at UI/2), then compare
    };
    
    SampleState m_sample_state;    ///< Current sampling state
    
    // ========================================================================
    // Sample Storage for BBPD
    // ========================================================================
    
    bool m_edge_sample;            ///< Edge sample value (at UI boundary)
    bool m_data_sample;            ///< Data sample value (at UI/2)
    bool m_prev_data_sample;       ///< Previous UI's data sample for transition detection
    double m_prev_data_analog;     ///< Previous UI's data sample (analog) for edge_threshold gating
    
    // ========================================================================
    // Member Variables
    // ========================================================================
    
    CdrParams m_params;           ///< CDR configuration parameters
    double m_phase;               ///< Current phase accumulation (s)
    double m_integral;            ///< PI controller integral state
    double m_last_phase_error;    ///< Last phase error from BB-PD

    // ---- 整数时基调度（2026-10 重写） ----
    long long m_step_index;       ///< 处理调用计数（仿真步号，从 0 起）
    long long m_edge_ui;          ///< 下一个待发 edge 触发对应的 UI 序号 k
    long long m_data_ui;          ///< 下一个待发 data 触发对应的 UI 序号 k
    long long m_next_edge_step;   ///< 下一 edge 触发的绝对步号
    long long m_next_data_step;   ///< 下一 data 触发的绝对步号
    double m_scheduled_qq;        ///< 当前调度使用的量化相位（qq）
    double m_ts_cached;           ///< 缓存的 timestep（首个 processing 惰性获取）
    double m_last_written_pe;     ///< phase_error_de 写变更检测值

    // ========================================================================
    // Private Methods
    // ========================================================================
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate_params();

    /// 当前量化相位（调度用）
    double quantized_phase() const;

    /// 时刻 t 对齐到步格（ceil），不低于 floor_step
    long long steps_for(double t, long long floor_step) const;

    /// 以给定 qq 重排尚未发出的 edge/data 触发（floor = 当前步）
    void reschedule(double qq);
};

} // namespace serdes

#endif // SERDES_RX_CDR_H
