#ifndef SERDES_DFE_ADAPT_TDF_H
#define SERDES_DFE_ADAPT_TDF_H

#include <systemc-ams>
#include <vector>
#include "common/parameters.h"

namespace serdes {

/**
 * @brief DfeAdaptTdf - TDF Domain DFE Adaptation Engine
 *
 * Implements the core DFE adaptation algorithm per the dual-threshold proposal:
 * - Receives three comparator outputs: d_k (main, thr=0), s_k (+Vref), s'_k (-Vref)
 * - Maintains decision history buffer at UI rate
 * - Executes Sign-Sign LMS tap update at UI rate: c[n] -= mu * sign_e * d[k-n]
 * - Accumulates statistics (N_A, N_B, N_C, N_D) every M UI
 * - Adapts Vref (optional, controlled by enable switch)
 * - Manages state machine: STARTUP -> ACQUIRE -> TRACK
 *
 * Architecture role (Plan A):
 *   All per-UI signal processing lives in TDF domain.
 *   DE domain (AdaptionDe) reads accumulated statistics for higher-level control.
 */
class DfeAdaptTdf : public sca_tdf::sca_module {
public:
    // ========================================================================
    // TDF Input Ports
    // ========================================================================

    // Three comparator outputs (all sca_in<double> with 0/1 values)
    sca_tdf::sca_in<double> data_in;          // Main sampler d_k (threshold = 0)
    sca_tdf::sca_in<double> vref_pos_in;      // +Vref comparator s_k (threshold = +Vref)
    sca_tdf::sca_in<double> vref_neg_in;      // -Vref comparator s'_k (threshold = -Vref)

    // Sampling trigger (shared with main sampler, from CDR)
    // 2026-10: 由 CDR 的 data_trigger 驱动（每 UI 1 个，data 时刻），
    // 不再使用内部硬编码 100ps 触发。输入端口整体延迟 1 步，
    // 与比较器输出对齐（P0-3）。
    sca_tdf::sca_in<bool> sampling_trigger;

    // ========================================================================
    // DE Input Ports (from AdaptionDe control)
    // ========================================================================
    sca_tdf::sca_de::sca_in<int> mode_de;           // Operating mode: 0=init, 1=training, 2=data, 3=freeze
    sca_tdf::sca_de::sca_in<bool> reset_de;         // Global reset
    sca_tdf::sca_de::sca_in<double> vref_cmd_de;    // Vref command from AdaptionDe (when adapt enabled)

    // ========================================================================
    // DE Output Ports (to DFE Summer)
    // ========================================================================
    sca_tdf::sca_de::sca_out<double> tap1_de;       // DFE tap coefficients
    sca_tdf::sca_de::sca_out<double> tap2_de;
    sca_tdf::sca_de::sca_out<double> tap3_de;
    sca_tdf::sca_de::sca_out<double> tap4_de;
    sca_tdf::sca_de::sca_out<double> tap5_de;
    sca_tdf::sca_de::sca_out<double> tap6_de;
    sca_tdf::sca_de::sca_out<double> tap7_de;
    sca_tdf::sca_de::sca_out<double> tap8_de;

    // ========================================================================
    // DE Output Ports (statistics to AdaptionDe)
    // ========================================================================
    sca_tdf::sca_de::sca_out<int> stat_N_A;         // d=+1 and s=+Vref count
    sca_tdf::sca_de::sca_out<int> stat_N_B;         // d=+1 total count
    sca_tdf::sca_de::sca_out<int> stat_N_C;         // d=-1 and s'=-Vref count
    sca_tdf::sca_de::sca_out<int> stat_N_D;         // d=-1 total count
    sca_tdf::sca_de::sca_out<double> stat_P_pos;    // P(s=1 | d=+1) = N_A / N_B
    sca_tdf::sca_de::sca_out<double> stat_P_neg;    // P(s'=0 | d=-1) = 1 - N_C/N_D
    sca_tdf::sca_de::sca_out<double> stat_C;        // Joint confidence = (P_pos + P_neg) / 2
    sca_tdf::sca_de::sca_out<double> stat_Delta;    // Asymmetry = P_pos - P_neg
    sca_tdf::sca_de::sca_out<int> stat_state;       // Current state machine state
    sca_tdf::sca_de::sca_out<double> stat_mu;       // Current step size mu
    sca_tdf::sca_de::sca_out<int> stat_update_count;// Total tap update count
    sca_tdf::sca_de::sca_out<double> vref_current;  // Current Vref value (for feedback to AdaptionDe)

    // ========================================================================
    // Constructor
    // ========================================================================
    SC_HAS_PROCESS(DfeAdaptTdf);

    /**
     * @brief Constructor
     * @param nm Module name
     * @param dfe_params DFE adaptation parameters
     * @param vref_params Vref adaptation parameters
     */
    DfeAdaptTdf(sc_core::sc_module_name nm,
                const AdaptionParams::DfeAdaptParams& dfe_params,
                const AdaptionParams::VrefAdaptParams& vref_params);

    // ========================================================================
    // TDF Callbacks
    // ========================================================================
    void set_attributes();
    void initialize();
    void processing();

    // ========================================================================
    // Public Accessors
    // ========================================================================
    const double* get_taps() const { return m_taps; }
    int get_update_count() const { return m_update_count; }
    int get_state() const { return m_state; }
    double get_mu() const { return m_current_mu; }

private:
    // ========================================================================
    // Parameters
    // ========================================================================
    const AdaptionParams::DfeAdaptParams& m_dfe_params;
    const AdaptionParams::VrefAdaptParams& m_vref_params;

    // ========================================================================
    // DFE State
    // ========================================================================
    static const int MAX_TAPS = 8;
    double m_taps[MAX_TAPS];             // DFE tap coefficients
    double m_history[MAX_TAPS];          // Decision history: ±1 (UI rate)
    int m_num_taps;                      // Active number of taps

    // ========================================================================
    // Statistics State
    // ========================================================================
    int m_N_A;                           // d=+1 and s=1 count (within current period)
    int m_N_B;                           // d=+1 total count
    int m_N_C;                           // d=-1 and s'=1 count
    int m_N_D;                           // d=-1 total count
    int m_ui_counter;                    // UI counter within current stats period
    double m_P_pos;                      // Cached P(s=1|d=+1)
    double m_P_neg;                      // Cached P(s'=0|d=-1)
    double m_C;                          // Joint confidence
    double m_Delta;                      // Asymmetry

    // ========================================================================
    // State Machine
    // ========================================================================
    int m_state;                         // 0=STARTUP, 1=ACQUIRE, 2=TRACK
    double m_current_mu;                 // Current step size
    double m_current_vref;               // Current Vref value (absolute, used for ±Vref)
    int m_update_count;                  // Total tap updates performed

    // ========================================================================
    // Held values (sampled on trigger)
    // ========================================================================
    bool m_prev_data;                    // Previous data decision
    bool m_hold_data;                    // Held data from last trigger
    bool m_hold_s_pos;                   // Held +Vref comparator from last trigger
    bool m_hold_s_neg;                   // Held -Vref comparator from last trigger

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * @brief Generate sign(e) per proposal formula
     *
     * sign_e = +1 if d=+1 and s=0 (amplitude not enough)
     * sign_e = -1 if d=+1 and s=1 (amplitude sufficient/overshoot)
     * sign_e = +1 if d=-1 and s'=0 (amplitude not enough)
     * sign_e = -1 if d=-1 and s'=1 (amplitude sufficient/overshoot)
     */
    double compute_sign_e(double d, double s_pos, double s_neg) const;

    /**
     * @brief Shift history buffer and insert new decision
     */
    void shift_history(double new_decision);

    /**
     * @brief Execute Sign-Sign LMS tap update for one UI
     */
    void update_taps(double sign_e);

    /**
     * @brief Accumulate one UI into statistics counters
     */
    void accumulate_stats(double d, double s_pos, double s_neg);

    /**
     * @brief Compute statistics ratios and write to DE output ports
     */
    void compute_and_output_stats();

    /**
     * @brief Reset statistics counters for new period
     */
    void reset_stats();

    /**
     * @brief Adapt Vref based on statistics (if enabled)
     *
     * 2026-10：单控制器架构——本函数仅镜像 AdaptionDe 下发的 vref_cmd
     * 用于监测输出，不再自闭环（避免与 AdaptionDe 双积分）。
     */
    void adapt_vref();

    /**
     * @brief State machine transition logic
     */
    void update_state_machine();

    /**
     * @brief Write all tap coefficients to DE output ports
     */
    void write_tap_outputs();

    /**
     * @brief Clamp value to range
     */
    double clamp(double val, double min_val, double max_val) const;

    /**
     * @brief Sign function (returns 0 for input 0)
     */
    static double sign_fn(double val);
};

} // namespace serdes

#endif // SERDES_DFE_ADAPT_TDF_H
