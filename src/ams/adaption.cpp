/**
 * @file adaption.cpp
 * @brief AdaptionDe class implementation - DE domain adaptive control module
 * 
 * Implements four adaptive algorithms:
 * - AGC (Automatic Gain Control) with PI controller
 * - DFE tap update (LMS/Sign-LMS/NLMS)
 * - Threshold adaptation
 * - CDR PI controller
 */

#include "ams/adaption.h"
#include <cmath>
#include <algorithm>

namespace serdes {

// ============================================================================
// Constructor
// ============================================================================
AdaptionDe::AdaptionDe(sc_core::sc_module_name nm, const AdaptionParams& params)
    : sc_core::sc_module(nm)
    // Input ports
    , phase_error("phase_error")
    , amplitude_rms("amplitude_rms")
    , error_count("error_count")
    , isi_metric("isi_metric")
    , mode("mode")
    , reset("reset")
    , scenario_switch("scenario_switch")
    // Output ports
    , vga_gain("vga_gain")
    , ctle_zero("ctle_zero")
    , ctle_pole("ctle_pole")
    , ctle_dc_gain("ctle_dc_gain")
    , dfe_tap1("dfe_tap1")
    , dfe_tap2("dfe_tap2")
    , dfe_tap3("dfe_tap3")
    , dfe_tap4("dfe_tap4")
    , dfe_tap5("dfe_tap5")
    , dfe_tap6("dfe_tap6")
    , dfe_tap7("dfe_tap7")
    , dfe_tap8("dfe_tap8")
    , sampler_threshold("sampler_threshold")
    , sampler_hysteresis("sampler_hysteresis")
    , phase_cmd("phase_cmd")
    , update_count("update_count")
    , freeze_flag("freeze_flag")
    // Parameters
    , m_params(params)
{
    // Initialize timing
    m_fast_period = sc_core::sc_time(params.fast_update_period, sc_core::SC_SEC);
    m_slow_period = sc_core::sc_time(params.slow_update_period, sc_core::SC_SEC);
    
    // Initialize state
    initialize_state();
    
    // Register processes
    SC_THREAD(fast_path_process);
    sensitive << reset;
    
    SC_THREAD(slow_path_process);
    sensitive << reset;
    
    SC_METHOD(reset_process);
    sensitive << reset.pos();
    dont_initialize();
}

// ============================================================================
// Destructor
// ============================================================================
AdaptionDe::~AdaptionDe() {
    // Nothing to clean up
}

// ============================================================================
// State Initialization
// ============================================================================
void AdaptionDe::initialize_state() {
    // AGC state
    m_agc_integral = 0.0;
    m_current_gain = m_params.agc.initial_gain;
    m_prev_gain = m_current_gain;
    
    // DFE state - initialize from params or zeros
    for (int i = 0; i < 8; ++i) {
        if (i < static_cast<int>(m_params.dfe.initial_taps.size())) {
            m_dfe_taps[i] = m_params.dfe.initial_taps[i];
        } else {
            m_dfe_taps[i] = 0.0;
        }
        m_dfe_history[i] = 0.0;
    }
    
    // CDR PI state
    m_cdr_integral = 0.0;
    m_current_phase_cmd = m_params.cdr_pi.initial_phase;
    
    // Threshold state
    m_current_threshold = m_params.threshold.initial;
    m_current_hysteresis = m_params.threshold.hysteresis;
    m_prev_error_count = 0;
    
    // Update tracking
    m_update_count = 0;
    m_fast_update_count = 0;
    m_slow_update_count = 0;
    m_freeze_flag = false;
    
    // Snapshot management
    m_snapshots.clear();
    m_snapshots.reserve(100);  // Reserve space for snapshots
    m_last_snapshot_time = sc_core::SC_ZERO_TIME;
}

// ============================================================================
// Reset Handler
// ============================================================================
void AdaptionDe::reset_process() {
    if (reset.read()) {
        initialize_state();
        write_all_outputs();
    }
}

// ============================================================================
// Fast Path Process (CDR PI + Threshold)
// ============================================================================
void AdaptionDe::fast_path_process() {
    // Wait for reset to complete
    wait(sc_core::SC_ZERO_TIME);
    
    while (true) {
        // Wait for fast update period
        wait(m_fast_period);
        
        // Check mode and freeze flag
        int current_mode = mode.read();
        if (current_mode == 3 || m_freeze_flag) {
            // In freeze mode, skip update
            continue;
        }
        
        // Check for freeze conditions
        if (m_params.safety.freeze_on_error && check_freeze_condition()) {
            m_freeze_flag = true;
            freeze_flag.write(true);
            
            // Attempt rollback if enabled
            if (m_params.safety.rollback_enable) {
                rollback_to_snapshot();
            }
            continue;
        }
        
        // ====================================================================
        // CDR PI Update (fast path)
        // ====================================================================
        if (m_params.cdr_pi.enabled && (current_mode == 1 || current_mode == 2)) {
            double phase_err = phase_error.read();
            m_current_phase_cmd = cdr_pi_update(phase_err);
            phase_cmd.write(m_current_phase_cmd);
        }
        
        // ====================================================================
        // Threshold Adaptation (fast path)
        // ====================================================================
        if (m_params.threshold.enabled && (current_mode == 1 || current_mode == 2)) {
            int err_cnt = error_count.read();
            m_current_threshold = threshold_adapt(err_cnt);
            sampler_threshold.write(m_current_threshold);
            sampler_hysteresis.write(m_current_hysteresis);
        }
        
        // Update counters
        m_fast_update_count++;
        m_update_count++;
        update_count.write(m_update_count);
    }
}

// ============================================================================
// Slow Path Process (AGC + DFE)
// ============================================================================
void AdaptionDe::slow_path_process() {
    // Wait for reset to complete
    wait(sc_core::SC_ZERO_TIME);
    
    // Write initial outputs
    write_all_outputs();
    
    while (true) {
        // Wait for slow update period
        wait(m_slow_period);
        
        // Check mode and freeze flag
        int current_mode = mode.read();
        if (current_mode == 3 || m_freeze_flag) {
            // In freeze mode, skip update
            continue;
        }
        
        // ====================================================================
        // Snapshot Management
        // ====================================================================
        sc_core::sc_time current_time = sc_core::sc_time_stamp();
        sc_core::sc_time snapshot_interval(m_params.safety.snapshot_interval, sc_core::SC_SEC);
        
        if ((current_time - m_last_snapshot_time) >= snapshot_interval) {
            save_snapshot();
            m_last_snapshot_time = current_time;
        }
        
        // ====================================================================
        // AGC Update (slow path)
        // ====================================================================
        if (m_params.agc.enabled && (current_mode == 1 || current_mode == 2)) {
            double amp = amplitude_rms.read();
            m_current_gain = agc_pi_update(amp);
            vga_gain.write(m_current_gain);
        }
        
        // ====================================================================
        // DFE Tap Update (slow path)
        // ====================================================================
        if (m_params.dfe.enabled && (current_mode == 1 || current_mode == 2)) {
            // Use error_count as proxy for decision error magnitude
            // In real implementation, this would be actual decision error
            int err_cnt = error_count.read();
            double error_proxy = (err_cnt - m_prev_error_count) * 0.01;  // Scale to reasonable range
            
            // Select update algorithm
            if (m_params.dfe.algorithm == "lms") {
                dfe_lms_update(error_proxy);
            } else if (m_params.dfe.algorithm == "sign-lms") {
                dfe_sign_lms_update(error_proxy);
            } else if (m_params.dfe.algorithm == "nlms") {
                dfe_nlms_update(error_proxy);
            } else {
                // Default to sign-lms
                dfe_sign_lms_update(error_proxy);
            }
            
            write_dfe_outputs();
        }
        
        // Update counters
        m_slow_update_count++;
    }
}

// ============================================================================
// AGC PI Controller Update
// ============================================================================
double AdaptionDe::agc_pi_update(double amplitude) {
    // Calculate amplitude error
    double amp_error = m_params.agc.target_amplitude - amplitude;
    
    // PI controller
    double P = m_params.agc.kp * amp_error;
    m_agc_integral += m_params.agc.ki * amp_error * m_params.slow_update_period;
    
    // Calculate new gain
    double new_gain = m_current_gain + P + m_agc_integral;
    
    // Apply saturation clamp
    new_gain = clamp(new_gain, m_params.agc.gain_min, m_params.agc.gain_max);
    
    // Apply rate limiting
    double max_change = m_params.agc.rate_limit * m_params.slow_update_period;
    double gain_change = new_gain - m_prev_gain;
    if (std::abs(gain_change) > max_change) {
        new_gain = m_prev_gain + sign(gain_change) * max_change;
    }
    
    // Store previous gain for next iteration
    m_prev_gain = new_gain;
    
    return new_gain;
}

// ============================================================================
// DFE LMS Algorithm Update
// ============================================================================
void AdaptionDe::dfe_lms_update(double error) {
    int num_taps = std::min(m_params.dfe.num_taps, 8);
    
    for (int i = 0; i < num_taps; ++i) {
        // LMS update: tap[i] = tap[i] + mu * e(n) * x[n-i]
        double update = m_params.dfe.mu * error * m_dfe_history[i];
        
        // Apply leakage
        m_dfe_taps[i] = (1.0 - m_params.dfe.leakage) * m_dfe_taps[i] + update;
        
        // Apply saturation
        m_dfe_taps[i] = clamp(m_dfe_taps[i], m_params.dfe.tap_min, m_params.dfe.tap_max);
    }
    
    // Shift history (simplified - in real implementation, would use actual decisions)
    for (int i = 7; i > 0; --i) {
        m_dfe_history[i] = m_dfe_history[i-1];
    }
    m_dfe_history[0] = sign(error);  // Use sign of error as proxy for decision
}

// ============================================================================
// DFE Sign-LMS Algorithm Update
// ============================================================================
void AdaptionDe::dfe_sign_lms_update(double error) {
    int num_taps = std::min(m_params.dfe.num_taps, 8);
    
    // Check freeze threshold
    if (std::abs(error) > m_params.dfe.freeze_threshold) {
        return;  // Freeze update
    }
    
    for (int i = 0; i < num_taps; ++i) {
        // Sign-LMS update: tap[i] = tap[i] + mu * sign(e(n)) * sign(x[n-i])
        double update = m_params.dfe.mu * sign(error) * sign(m_dfe_history[i]);
        
        // Apply leakage
        m_dfe_taps[i] = (1.0 - m_params.dfe.leakage) * m_dfe_taps[i] + update;
        
        // Apply saturation
        m_dfe_taps[i] = clamp(m_dfe_taps[i], m_params.dfe.tap_min, m_params.dfe.tap_max);
    }
    
    // Shift history
    for (int i = 7; i > 0; --i) {
        m_dfe_history[i] = m_dfe_history[i-1];
    }
    m_dfe_history[0] = sign(error);
}

// ============================================================================
// DFE NLMS Algorithm Update
// ============================================================================
void AdaptionDe::dfe_nlms_update(double error) {
    int num_taps = std::min(m_params.dfe.num_taps, 8);
    
    // Calculate input power for normalization
    double input_power = 0.0;
    for (int i = 0; i < num_taps; ++i) {
        input_power += m_dfe_history[i] * m_dfe_history[i];
    }
    
    // Avoid division by zero
    double epsilon = 1e-10;
    double norm_factor = 1.0 / (input_power + epsilon);
    
    for (int i = 0; i < num_taps; ++i) {
        // NLMS update: tap[i] = tap[i] + mu * e(n) * x[n-i] / (||x||^2 + epsilon)
        double update = m_params.dfe.mu * error * m_dfe_history[i] * norm_factor;
        
        // Apply leakage
        m_dfe_taps[i] = (1.0 - m_params.dfe.leakage) * m_dfe_taps[i] + update;
        
        // Apply saturation
        m_dfe_taps[i] = clamp(m_dfe_taps[i], m_params.dfe.tap_min, m_params.dfe.tap_max);
    }
    
    // Shift history
    for (int i = 7; i > 0; --i) {
        m_dfe_history[i] = m_dfe_history[i-1];
    }
    m_dfe_history[0] = sign(error);
}

// ============================================================================
// Threshold Adaptation Update
// ============================================================================
double AdaptionDe::threshold_adapt(int err_cnt) {
    // Calculate error trend
    int error_delta = err_cnt - m_prev_error_count;
    m_prev_error_count = err_cnt;
    
    // Simple gradient descent approach
    // If errors increasing, adjust threshold in opposite direction
    double threshold_adjustment = 0.0;
    
    if (error_delta > 0) {
        // Errors increasing, try to find better threshold
        // Use sign of previous adjustment or random direction
        threshold_adjustment = -m_params.threshold.adapt_step;
    } else if (error_delta < 0) {
        // Errors decreasing, continue in same direction
        threshold_adjustment = m_params.threshold.adapt_step;
    }
    
    // Apply update with drift threshold check
    double new_threshold = m_current_threshold + threshold_adjustment;
    
    // Limit threshold drift from initial
    if (std::abs(new_threshold - m_params.threshold.initial) > m_params.threshold.drift_threshold) {
        new_threshold = m_current_threshold;  // Don't apply update if too far from initial
    }
    
    // Adjust hysteresis based on error rate (simplified)
    if (error_delta > 10) {
        // High error rate, increase hysteresis for stability
        m_current_hysteresis = std::min(m_current_hysteresis * 1.1, 0.1);
    } else if (error_delta < -10) {
        // Low error rate, can decrease hysteresis
        m_current_hysteresis = std::max(m_current_hysteresis * 0.99, 0.01);
    }
    
    return new_threshold;
}

// ============================================================================
// CDR PI Controller Update
// ============================================================================
double AdaptionDe::cdr_pi_update(double phase_err) {
    // PI controller
    double P = m_params.cdr_pi.kp * phase_err;
    
    // Integral term with anti-windup
    double new_integral = m_cdr_integral + m_params.cdr_pi.ki * phase_err * m_params.fast_update_period;
    
    // Calculate new phase command
    double new_phase_cmd = P + new_integral;
    
    // Apply range saturation
    double phase_range = m_params.cdr_pi.phase_range;
    if (std::abs(new_phase_cmd) > phase_range) {
        // Clamp output
        new_phase_cmd = clamp(new_phase_cmd, -phase_range, phase_range);
        
        // Anti-windup: don't update integral if saturated
        if (m_params.cdr_pi.anti_windup) {
            // Keep integral at current value (don't accumulate further)
        } else {
            m_cdr_integral = new_integral;
        }
    } else {
        m_cdr_integral = new_integral;
    }
    
    // Quantize to resolution
    double resolution = m_params.cdr_pi.phase_resolution;
    new_phase_cmd = std::round(new_phase_cmd / resolution) * resolution;
    
    return new_phase_cmd;
}

// ============================================================================
// Safety Methods
// ============================================================================
bool AdaptionDe::check_freeze_condition() {
    // Check for error burst
    int err_cnt = error_count.read();
    if (err_cnt > m_params.safety.error_burst_threshold) {
        return true;
    }
    
    // Check for amplitude anomaly
    double amp = amplitude_rms.read();
    if (amp < 0.01 || amp > 2.0) {  // Abnormal amplitude
        return true;
    }
    
    // Check for phase error out of range
    double phase_err = phase_error.read();
    if (std::abs(phase_err) > m_params.cdr_pi.phase_range * 2.0) {
        return true;
    }
    
    return false;
}

void AdaptionDe::save_snapshot() {
    Snapshot snap;
    snap.vga_gain = m_current_gain;
    for (int i = 0; i < 8; ++i) {
        snap.dfe_taps[i] = m_dfe_taps[i];
    }
    snap.threshold = m_current_threshold;
    snap.hysteresis = m_current_hysteresis;
    snap.phase_cmd = m_current_phase_cmd;
    snap.timestamp = sc_core::sc_time_stamp();
    snap.valid = true;
    
    // Keep limited number of snapshots (ring buffer behavior)
    if (m_snapshots.size() >= 100) {
        m_snapshots.erase(m_snapshots.begin());
    }
    m_snapshots.push_back(snap);
}

bool AdaptionDe::rollback_to_snapshot() {
    // Find last valid snapshot
    for (int i = static_cast<int>(m_snapshots.size()) - 1; i >= 0; --i) {
        if (m_snapshots[i].valid) {
            const Snapshot& snap = m_snapshots[i];
            
            // Restore state
            m_current_gain = snap.vga_gain;
            for (int j = 0; j < 8; ++j) {
                m_dfe_taps[j] = snap.dfe_taps[j];
            }
            m_current_threshold = snap.threshold;
            m_current_hysteresis = snap.hysteresis;
            m_current_phase_cmd = snap.phase_cmd;
            
            // Reset integrators
            m_agc_integral = 0.0;
            m_cdr_integral = 0.0;
            
            // Write restored outputs
            write_all_outputs();
            
            // Clear freeze flag after short delay
            m_freeze_flag = false;
            freeze_flag.write(false);
            
            return true;
        }
    }
    return false;
}

// ============================================================================
// Helper Methods
// ============================================================================
double AdaptionDe::clamp(double val, double min_val, double max_val) const {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

double AdaptionDe::sign(double val) const {
    if (val > 0) return 1.0;
    if (val < 0) return -1.0;
    return 0.0;
}

void AdaptionDe::write_dfe_outputs() {
    dfe_tap1.write(m_dfe_taps[0]);
    dfe_tap2.write(m_dfe_taps[1]);
    dfe_tap3.write(m_dfe_taps[2]);
    dfe_tap4.write(m_dfe_taps[3]);
    dfe_tap5.write(m_dfe_taps[4]);
    dfe_tap6.write(m_dfe_taps[5]);
    dfe_tap7.write(m_dfe_taps[6]);
    dfe_tap8.write(m_dfe_taps[7]);
}

void AdaptionDe::write_all_outputs() {
    // Write AGC output
    vga_gain.write(m_current_gain);
    
    // Write CTLE outputs (defaults)
    ctle_zero.write(2e9);      // Default 2 GHz
    ctle_pole.write(30e9);     // Default 30 GHz
    ctle_dc_gain.write(1.5);   // Default 1.5
    
    // Write DFE outputs
    write_dfe_outputs();
    
    // Write threshold outputs
    sampler_threshold.write(m_current_threshold);
    sampler_hysteresis.write(m_current_hysteresis);
    
    // Write CDR output
    phase_cmd.write(m_current_phase_cmd);
    
    // Write status outputs
    update_count.write(m_update_count);
    freeze_flag.write(m_freeze_flag);
}

} // namespace serdes
