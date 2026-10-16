#ifndef SERDES_AMS_ADAPTION_H
#define SERDES_AMS_ADAPTION_H

#include <systemc>
#include <vector>
#include "common/parameters.h"

namespace serdes {

/**
 * @brief AdaptionDe - DE Domain Adaptive Control Module
 *
 * This module implements the DE-domain adaptive control algorithms for SerDes link:
 * - AGC (Automatic Gain Control) with PI controller
 * - CDR PI controller
 * - Threshold adaptation (based on statistics from DfeAdaptTdf)
 * - Vref command generation (for DfeAdaptTdf, when Vref adaptation is enabled)
 *
 * Note: DFE tap update has been moved to DfeAdaptTdf (TDF domain, Plan A).
 *       This module no longer maintains DFE history or performs tap updates.
 */
class AdaptionDe : public sc_core::sc_module {
public:
    // ========================================================================
    // Input Ports (from RX/CDR/System)
    // ========================================================================
    sc_core::sc_in<double> phase_error;       // Phase error from CDR (s or normalized UI)
    sc_core::sc_in<double> amplitude_rms;     // Amplitude RMS from RX amplitude statistics
    sc_core::sc_in<bool> gain_saturated;      // AGC 饱和互锁：VGA 接近限幅时禁止上调增益
    sc_core::sc_in<int> mode;                 // Operating mode: 0=init, 1=training, 2=data, 3=freeze
    sc_core::sc_in<bool> reset;               // Global reset signal
    sc_core::sc_in<double> scenario_switch;   // Scenario switch event (optional)

    // Statistics inputs from DfeAdaptTdf (DE domain, updated every M UI)
    sc_core::sc_in<int> stat_N_A;             // d=+1 and s=+Vref count
    sc_core::sc_in<int> stat_N_B;             // d=+1 total count
    sc_core::sc_in<int> stat_N_C;             // d=-1 and s'=-Vref count
    sc_core::sc_in<int> stat_N_D;             // d=-1 total count
    sc_core::sc_in<double> stat_P_pos;        // P(s=1|d=+1)
    sc_core::sc_in<double> stat_P_neg;        // P(s'=0|d=-1)
    sc_core::sc_in<double> stat_C;            // Joint confidence
    sc_core::sc_in<double> stat_Delta;        // Asymmetry metric
    sc_core::sc_in<int> stat_dfe_state;       // DFE state machine state (0=STARTUP,1=ACQUIRE,2=TRACK)

    // ========================================================================
    // Output Ports (to RX/CDR)
    // ========================================================================
    sc_core::sc_out<double> vga_gain;         // VGA gain setting (linear)
    sc_core::sc_out<double> ctle_zero;        // CTLE zero frequency (Hz, optional)
    sc_core::sc_out<double> ctle_pole;        // CTLE pole frequency (Hz, optional)
    sc_core::sc_out<double> ctle_dc_gain;     // CTLE DC gain (linear, optional)

    // Vref command output to DfeAdaptTdf
    sc_core::sc_out<double> vref_cmd;         // Vref command (absolute value, for ±Vref comparators)

    sc_core::sc_out<double> sampler_threshold;    // Sampler threshold (V)
    sc_core::sc_out<double> sampler_hysteresis;   // Sampler hysteresis (V)
    sc_core::sc_out<double> phase_cmd;            // Phase interpolator command (s)
    sc_core::sc_out<int> update_count;            // Update counter for diagnostics
    sc_core::sc_out<bool> freeze_flag;            // Freeze/rollback status flag

    // ========================================================================
    // Constructor and Process Declaration
    // ========================================================================
    SC_HAS_PROCESS(AdaptionDe);

    /**
     * @brief Constructor
     * @param nm Module name
     * @param params Adaption parameters
     */
    AdaptionDe(sc_core::sc_module_name nm, const AdaptionParams& params);

    /**
     * @brief Destructor
     */
    ~AdaptionDe();

    // ========================================================================
    // Public Methods for External Access
    // ========================================================================

    /**
     * @brief Get current AGC gain
     */
    double get_current_gain() const { return m_current_gain; }

    /**
     * @brief Get update count
     */
    int get_update_count() const { return m_update_count; }

    /**
     * @brief Check if frozen
     */
    bool is_frozen() const { return m_freeze_flag; }

private:
    // ========================================================================
    // Parameters
    // ========================================================================
    AdaptionParams m_params;

    // ========================================================================
    // Internal State Variables
    // ========================================================================
    // AGC state
    double m_agc_integral;          // PI controller integral term
    double m_current_gain;          // Current VGA gain
    double m_prev_gain;             // Previous gain (for rate limiting)

    // CDR PI state
    double m_cdr_integral;          // PI controller integral term
    double m_current_phase_cmd;     // Current phase command

    // Threshold state
    double m_current_threshold;     // Current sampler threshold
    double m_current_hysteresis;    // Current hysteresis window

    // Vref command state
    double m_current_vref_cmd;      // Current Vref command value

    // Update tracking
    int m_update_count;             // Total update count
    int m_fast_update_count;        // Fast path update count
    int m_slow_update_count;        // Slow path update count
    bool m_freeze_flag;             // Freeze flag
    bool m_signal_present;          // 信号已建立（幅度曾超过阈值）——启动阶段不触发幅度冻结

    // Snapshot structure for rollback
    struct Snapshot {
        double vga_gain;
        double threshold;
        double hysteresis;
        double phase_cmd;
        double vref_cmd;
        sc_core::sc_time timestamp;
        bool valid;

        Snapshot() : vga_gain(0), threshold(0), hysteresis(0),
                     phase_cmd(0), vref_cmd(0), valid(false) {}
    };
    std::vector<Snapshot> m_snapshots;
    sc_core::sc_time m_last_snapshot_time;

    // Timing
    sc_core::sc_time m_fast_period;
    sc_core::sc_time m_slow_period;

    // ========================================================================
    // SC_METHOD/SC_THREAD Processes
    // ========================================================================

    /**
     * @brief Fast path update process (CDR PI + threshold adaptation)
     * Called at fast_update_period interval
     */
    void fast_path_process();

    /**
     * @brief Slow path update process (AGC + Vref command generation)
     * Called at slow_update_period interval
     */
    void slow_path_process();

    /**
     * @brief Reset handler process
     * Triggered by reset signal
     */
    void reset_process();

    // ========================================================================
    // Algorithm Implementation Methods
    // ========================================================================

    /**
     * @brief AGC PI controller update
     * @param amplitude Current amplitude RMS
     * @return Updated gain value
     */
    double agc_pi_update(double amplitude);

    /**
     * @brief CDR PI controller update
     * @param phase_err Phase error input
     * @return Updated phase command
     */
    double cdr_pi_update(double phase_err);

    /**
     * @brief Vref command generation based on statistics from DfeAdaptTdf
     */
    void update_vref_command();

    // ========================================================================
    // Safety and Helper Methods
    // ========================================================================

    /**
     * @brief Check for freeze conditions
     * @return true if should freeze
     */
    bool check_freeze_condition();

    /**
     * @brief 更新“信号已建立”锁存：幅度首次超过存活阈值后置位。
     *        启动阶段（信号尚未建立）不做幅度下限冻结与 AGC 更新。
     */
    void update_signal_present();

    /**
     * @brief Save current parameters to snapshot
     */
    void save_snapshot();

    /**
     * @brief Rollback to last valid snapshot
     * @return true if rollback successful
     */
    bool rollback_to_snapshot();

    /**
     * @brief Clamp value to range
     */
    double clamp(double val, double min_val, double max_val) const;

    /**
     * @brief Initialize all internal states
     */
    void initialize_state();

    /**
     * @brief Write slow path outputs (AGC, CTLE, Vref)
     */
    void write_all_outputs();
};

} // namespace serdes

#endif // SERDES_AMS_ADAPTION_H
