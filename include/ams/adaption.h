#ifndef SERDES_AMS_ADAPTION_H
#define SERDES_AMS_ADAPTION_H

#include <systemc>
#include <vector>
#include "common/parameters.h"

namespace serdes {

/**
 * @brief AdaptionDe - DE Domain Adaptive Control Module
 * 
 * This module implements the adaptive control algorithms for SerDes link:
 * - AGC (Automatic Gain Control) with PI controller
 * - DFE tap update (LMS/Sign-LMS/NLMS algorithms)
 * - Threshold adaptation
 * - CDR PI controller
 * 
 * Features:
 * - Multi-rate scheduling: fast path (CDR PI, threshold) and slow path (AGC, DFE)
 * - Safety mechanisms: freeze on error, rollback to snapshot
 * - Parameter clamping and rate limiting
 */
class AdaptionDe : public sc_core::sc_module {
public:
    // ========================================================================
    // Input Ports (from RX/CDR/System)
    // ========================================================================
    sc_core::sc_in<double> phase_error;       // Phase error from CDR (s or normalized UI)
    sc_core::sc_in<double> amplitude_rms;     // Amplitude RMS from RX amplitude statistics
    sc_core::sc_in<int> error_count;          // Error count from Sampler decision errors
    sc_core::sc_in<double> isi_metric;        // ISI metric (optional, for DFE strategy)
    sc_core::sc_in<int> mode;                 // Operating mode: 0=init, 1=training, 2=data, 3=freeze
    sc_core::sc_in<bool> reset;               // Global reset signal
    sc_core::sc_in<double> scenario_switch;   // Scenario switch event (optional)
    
    // ========================================================================
    // Output Ports (to RX/CDR)
    // ========================================================================
    sc_core::sc_out<double> vga_gain;         // VGA gain setting (linear)
    sc_core::sc_out<double> ctle_zero;        // CTLE zero frequency (Hz, optional)
    sc_core::sc_out<double> ctle_pole;        // CTLE pole frequency (Hz, optional)
    sc_core::sc_out<double> ctle_dc_gain;     // CTLE DC gain (linear, optional)
    
    // DFE taps (fixed 8 independent ports)
    sc_core::sc_out<double> dfe_tap1;
    sc_core::sc_out<double> dfe_tap2;
    sc_core::sc_out<double> dfe_tap3;
    sc_core::sc_out<double> dfe_tap4;
    sc_core::sc_out<double> dfe_tap5;
    sc_core::sc_out<double> dfe_tap6;
    sc_core::sc_out<double> dfe_tap7;
    sc_core::sc_out<double> dfe_tap8;
    
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
     * @brief Get current DFE taps
     */
    const double* get_dfe_taps() const { return m_dfe_taps; }
    
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
    
    // DFE state
    double m_dfe_taps[8];           // Fixed 8 DFE taps
    double m_dfe_history[8];        // Decision history for feedback
    
    // CDR PI state
    double m_cdr_integral;          // PI controller integral term
    double m_current_phase_cmd;     // Current phase command
    
    // Threshold state
    double m_current_threshold;     // Current sampler threshold
    double m_current_hysteresis;    // Current hysteresis window
    int m_prev_error_count;         // Previous error count for trend detection
    
    // Update tracking
    int m_update_count;             // Total update count
    int m_fast_update_count;        // Fast path update count
    int m_slow_update_count;        // Slow path update count
    bool m_freeze_flag;             // Freeze flag
    
    // Snapshot structure for rollback
    struct Snapshot {
        double vga_gain;
        double dfe_taps[8];
        double threshold;
        double hysteresis;
        double phase_cmd;
        sc_core::sc_time timestamp;
        bool valid;
        
        Snapshot() : vga_gain(0), threshold(0), hysteresis(0), 
                     phase_cmd(0), valid(false) {
            for (int i = 0; i < 8; ++i) dfe_taps[i] = 0;
        }
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
     * @brief Slow path update process (AGC + DFE tap update)
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
     * @brief DFE LMS algorithm update
     * @param error Decision error
     */
    void dfe_lms_update(double error);
    
    /**
     * @brief DFE Sign-LMS algorithm update
     * @param error Decision error
     */
    void dfe_sign_lms_update(double error);
    
    /**
     * @brief DFE NLMS algorithm update
     * @param error Decision error
     */
    void dfe_nlms_update(double error);
    
    /**
     * @brief Threshold adaptation update
     * @param error_count Current error count
     * @return Updated threshold value
     */
    double threshold_adapt(int error_count);
    
    /**
     * @brief CDR PI controller update
     * @param phase_err Phase error input
     * @return Updated phase command
     */
    double cdr_pi_update(double phase_err);
    
    // ========================================================================
    // Safety and Helper Methods
    // ========================================================================
    
    /**
     * @brief Check for freeze conditions
     * @return true if should freeze
     */
    bool check_freeze_condition();
    
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
     * @brief Sign function
     */
    double sign(double val) const;
    
    /**
     * @brief Write all DFE tap outputs
     */
    void write_dfe_outputs();
    
    /**
     * @brief Initialize all internal states
     */
    void initialize_state();
    
    /**
     * @brief Write all outputs with current state
     */
    void write_all_outputs();
};

} // namespace serdes

#endif // SERDES_AMS_ADAPTION_H
