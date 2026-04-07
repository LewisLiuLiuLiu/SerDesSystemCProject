/**
 * @file dfe_adapt_tdf.cpp
 * @brief DfeAdaptTdf class implementation - TDF domain DFE adaptation engine
 *
 * Implements the dual-threshold Sign-Sign LMS DFE adaptation algorithm:
 * - Per-UI history maintenance
 * - Per-UI Sign-Sign LMS tap update: c[n] -= mu * sign_e * d[k-n]
 * - Statistics accumulation every M UI
 * - Optional Vref adaptation
 * - State machine: STARTUP -> ACQUIRE -> TRACK with variable step size
 */

#include "ams/dfe_adapt_tdf.h"
#include <cmath>
#include <algorithm>

namespace serdes {

// Static const member definition (required by linker)
const int DfeAdaptTdf::MAX_TAPS;

// ============================================================================
// Constructor
// ============================================================================
DfeAdaptTdf::DfeAdaptTdf(sc_core::sc_module_name nm,
                           const AdaptionParams::DfeAdaptParams& dfe_params,
                           const AdaptionParams::VrefAdaptParams& vref_params)
    : sca_tdf::sca_module(nm)
    // TDF inputs
    , data_in("data_in")
    , vref_pos_in("vref_pos_in")
    , vref_neg_in("vref_neg_in")
    , sampling_trigger("sampling_trigger")
    // DE inputs
    , mode_de("mode_de")
    , reset_de("reset_de")
    , vref_cmd_de("vref_cmd_de")
    // DE outputs - taps
    , tap1_de("tap1_de")
    , tap2_de("tap2_de")
    , tap3_de("tap3_de")
    , tap4_de("tap4_de")
    , tap5_de("tap5_de")
    , tap6_de("tap6_de")
    , tap7_de("tap7_de")
    , tap8_de("tap8_de")
    // DE outputs - statistics
    , stat_N_A("stat_N_A")
    , stat_N_B("stat_N_B")
    , stat_N_C("stat_N_C")
    , stat_N_D("stat_N_D")
    , stat_P_pos("stat_P_pos")
    , stat_P_neg("stat_P_neg")
    , stat_C("stat_C")
    , stat_Delta("stat_Delta")
    , stat_state("stat_state")
    , stat_mu("stat_mu")
    , stat_update_count("stat_update_count")
    , vref_current("vref_current")
    // Parameters
    , m_dfe_params(dfe_params)
    , m_vref_params(vref_params)
    , m_num_taps(std::min(dfe_params.num_taps, MAX_TAPS))
    // State initialization
    , m_N_A(0)
    , m_N_B(0)
    , m_N_C(0)
    , m_N_D(0)
    , m_ui_counter(0)
    , m_P_pos(0.5)
    , m_P_neg(0.5)
    , m_C(0.5)
    , m_Delta(0.0)
    , m_state(0)  // STARTUP
    , m_current_mu(dfe_params.mu_startup)
    , m_current_vref(vref_params.enabled ? vref_params.vref_initial : vref_params.vref_pos)
    , m_update_count(0)
    , m_prev_data(false)
    , m_hold_data(false)
    , m_hold_s_pos(false)
    , m_hold_s_neg(false)
{
    // Initialize taps
    for (int i = 0; i < MAX_TAPS; ++i) {
        m_taps[i] = 0.0;
        m_history[i] = 0.0;
    }

    // Set initial taps from parameters if provided
    for (size_t i = 0; i < dfe_params.initial_taps.size() && i < static_cast<size_t>(MAX_TAPS); ++i) {
        m_taps[i] = dfe_params.initial_taps[i];
    }
}

// ============================================================================
// TDF Callbacks
// ============================================================================
void DfeAdaptTdf::set_attributes()
{
    // All inputs at rate 1
    data_in.set_rate(1);
    vref_pos_in.set_rate(1);
    vref_neg_in.set_rate(1);
    sampling_trigger.set_rate(1);
}

void DfeAdaptTdf::initialize()
{
    reset_stats();
    // Note: write_tap_outputs() and DE output writes cannot be called in initialize()
    // because sca_de::sca_out ports can only be written in processing().
    // They will be written on the first processing() call.
}

void DfeAdaptTdf::processing()
{
    // ========================================================================
    // Step 1: Read inputs
    // ========================================================================
    double d_raw = data_in.read();            // Main sampler output (0 or 1)
    double s_pos_raw = vref_pos_in.read();    // +Vref comparator output (0 or 1)
    double s_neg_raw = vref_neg_in.read();    // -Vref comparator output (0 or 1)
    // Note: trigger is now generated internally (see Step 1b)

    // Read DE control inputs
    int current_mode = static_cast<int>(mode_de.read());
    bool reset = static_cast<bool>(reset_de.read());
    
    // ========================================================================
    // Step 1b: Generate internal trigger based on time (not external signal)
    // This avoids cluster synchronization issues between CDR and DFE adaptation
    // ========================================================================
    static double last_ui_time = -1.0;
    double current_time = get_time().to_seconds();
    double ui_seconds = 100e-12;  // 100 ps for 10 Gbps (should be from config)
    double current_ui = std::floor(current_time / ui_seconds);
    
    bool trigger = false;
    if (current_ui > last_ui_time) {
        trigger = true;
        last_ui_time = current_ui;
    }

    // ========================================================================
    // Step 2: Handle reset
    // ========================================================================
    if (reset) {
        for (int i = 0; i < MAX_TAPS; ++i) {
            m_taps[i] = 0.0;
            m_history[i] = 0.0;
        }
        for (size_t i = 0; i < m_dfe_params.initial_taps.size() && i < static_cast<size_t>(MAX_TAPS); ++i) {
            m_taps[i] = m_dfe_params.initial_taps[i];
        }
        reset_stats();
        m_state = 0;  // STARTUP
        m_current_mu = m_dfe_params.mu_startup;
        m_current_vref = m_vref_params.enabled ? m_vref_params.vref_initial
                                               : m_vref_params.vref_pos;
        m_update_count = 0;
        write_tap_outputs();
        return;
    }

    // ========================================================================
    // Step 3: Skip if DFE not enabled or in freeze mode
    // ========================================================================
    if (!m_dfe_params.enabled || current_mode == 3) {
        return;
    }

    // ========================================================================
    // Step 4: Only process on sampling trigger (CDR-driven)
    // ========================================================================
    if (!trigger) {
        return;
    }

    // ========================================================================
    // Step 5: Hold comparator outputs on trigger edge
    // ========================================================================
    m_hold_data = (d_raw > 0.5);
    m_hold_s_pos = (s_pos_raw > 0.5);
    m_hold_s_neg = (s_neg_raw > 0.5);

    // Map data to ±1 for history and LMS
    double d_pm1 = m_hold_data ? 1.0 : -1.0;

    // ========================================================================
    // Step 6: Shift history and insert current decision
    // ========================================================================
    shift_history(d_pm1);

    // ========================================================================
    // Step 7: Generate sign(e) per proposal formula
    // ========================================================================
    double sign_e = compute_sign_e(d_raw, s_pos_raw, s_neg_raw);

    // ========================================================================
    // Step 8: Execute Sign-Sign LMS tap update (per UI)
    // ========================================================================
    update_taps(sign_e);

    // ========================================================================
    // Step 9: Accumulate statistics
    // ========================================================================
    accumulate_stats(d_raw, s_pos_raw, s_neg_raw);
    m_ui_counter++;

    // ========================================================================
    // Step 10: End-of-period processing (every M UI)
    // ========================================================================
    if (m_ui_counter >= m_dfe_params.stats_period) {
        compute_and_output_stats();

        // Vref adaptation (if enabled)
        if (m_vref_params.enabled) {
            adapt_vref();
        } else {
            // Use fixed Vref from config
            m_current_vref = m_vref_params.vref_pos;
        }

        // State machine update
        update_state_machine();

        // Write tap outputs (periodic, not per-UI, to reduce DE event rate)
        write_tap_outputs();

        // Reset counters for next period
        reset_stats();
        m_ui_counter = 0;
    }
}

// ============================================================================
// Compute sign(e) per proposal formula
// ============================================================================
double DfeAdaptTdf::compute_sign_e(double d, double s_pos, double s_neg) const
{
    // d is the main sampler output (0 or 1, where 1 means v > 0)
    // s_pos is the +Vref comparator output (0 or 1, where 1 means v > +Vref)
    // s_neg is the -Vref comparator output (0 or 1, where 1 means v > -Vref)

    if (d > 0.5) {
        // d = +1: check if signal exceeds +Vref
        // s_pos = 1: signal > +Vref → amplitude sufficient → sign_e = -1
        // s_pos = 0: signal < +Vref → amplitude not enough → sign_e = +1
        return (s_pos > 0.5) ? -1.0 : 1.0;
    } else {
        // d = -1: check if signal is below -Vref
        // s_neg = 1: signal > -Vref → not negative enough → sign_e = +1
        // s_neg = 0: signal < -Vref → sufficiently negative → sign_e = -1
        return (s_neg > 0.5) ? 1.0 : -1.0;
    }
}

// ============================================================================
// Shift history buffer
// ============================================================================
void DfeAdaptTdf::shift_history(double new_decision)
{
    for (int i = MAX_TAPS - 1; i > 0; --i) {
        m_history[i] = m_history[i - 1];
    }
    m_history[0] = new_decision;
}

// ============================================================================
// Sign-Sign LMS tap update (per UI)
// ============================================================================
void DfeAdaptTdf::update_taps(double sign_e)
{
    // Only update if sign_e is non-zero (i.e., we have meaningful error info)
    if (sign_e == 0.0) {
        return;
    }

    for (int i = 0; i < m_num_taps; ++i) {
        // Sign-Sign LMS: c[n] -= mu * sign(e) * d[k-n]
        // When sign_e = +1 (error positive, need to increase output), 
        // we decrease tap to reduce feedback (since v_eq = v_main - v_fb)
        double update = m_current_mu * sign_e * sign_fn(m_history[i]);

        // Apply leakage
        m_taps[i] = (1.0 - m_dfe_params.leakage) * m_taps[i] - update;

        // Apply saturation
        m_taps[i] = clamp(m_taps[i], m_dfe_params.tap_min, m_dfe_params.tap_max);
    }

    m_update_count++;
}

// ============================================================================
// Accumulate statistics
// ============================================================================
void DfeAdaptTdf::accumulate_stats(double d, double s_pos, double s_neg)
{
    if (d > 0.5) {
        // d = +1
        m_N_B++;                   // Count total d=+1
        if (s_pos > 0.5) {
            m_N_A++;               // Count d=+1 and s=1 (signal > +Vref)
        }
    } else {
        // d = -1
        m_N_D++;                   // Count total d=-1
        if (s_neg > 0.5) {
            m_N_C++;               // Count d=-1 and s'=1 (signal > -Vref)
        }
    }
}

// ============================================================================
// Compute statistics and write to DE outputs
// ============================================================================
void DfeAdaptTdf::compute_and_output_stats()
{
    // Compute probabilities
    if (m_N_B > 0) {
        m_P_pos = static_cast<double>(m_N_A) / static_cast<double>(m_N_B);
    }
    if (m_N_D > 0) {
        m_P_neg = 1.0 - static_cast<double>(m_N_C) / static_cast<double>(m_N_D);
    }

    m_C = (m_P_pos + m_P_neg) / 2.0;
    m_Delta = m_P_pos - m_P_neg;

    // Write to DE output ports
    stat_N_A.write(m_N_A);
    stat_N_B.write(m_N_B);
    stat_N_C.write(m_N_C);
    stat_N_D.write(m_N_D);
    stat_P_pos.write(m_P_pos);
    stat_P_neg.write(m_P_neg);
    stat_C.write(m_C);
    stat_Delta.write(m_Delta);
    stat_state.write(m_state);
    stat_mu.write(m_current_mu);
    stat_update_count.write(m_update_count);
    vref_current.write(m_current_vref);
}

// ============================================================================
// Reset statistics counters
// ============================================================================
void DfeAdaptTdf::reset_stats()
{
    m_N_A = 0;
    m_N_B = 0;
    m_N_C = 0;
    m_N_D = 0;
}

// ============================================================================
// Vref adaptation
// ============================================================================
void DfeAdaptTdf::adapt_vref()
{
    // Read Vref command from AdaptionDe (if connected), otherwise self-adapt
    double vref_cmd = vref_cmd_de.read();
    if (std::isfinite(vref_cmd) && vref_cmd > 0.0) {
        m_current_vref = vref_cmd;
        return;
    }

    // Self-adapt based on confidence metric
    // Target: C = target_confidence (e.g. 0.7)
    // If C < target, Vref is too high (too few samples exceed Vref), decrease Vref
    // If C > target, Vref is too low, increase Vref
    double error = m_C - m_vref_params.target_confidence;
    m_current_vref *= (1.0 + m_vref_params.adapt_alpha * error);

    // Clamp Vref to reasonable range
    m_current_vref = clamp(m_current_vref, 0.01, 0.5);
}

// ============================================================================
// State machine
// ============================================================================
void DfeAdaptTdf::update_state_machine()
{
    // Activity metric: fraction of UIs with non-trivial sign_e
    // Approximate using statistics: if N_A << N_B, most d=+1 have s=0 (need more signal)
    // Track this as a proxy for adaptation activity

    double activity = 0.5;  // Default mid-range

    switch (m_state) {
    case 0:  // STARTUP
        // Transition to ACQUIRE when confidence C is reasonable
        if (m_C > 0.4 && m_N_B > 100 && m_N_D > 100) {
            m_state = 1;
            m_current_mu = m_dfe_params.mu_acquire;
        }
        break;

    case 1:  // ACQUIRE
        // Transition to TRACK when confidence is high enough and stable
        if (m_C > m_vref_params.target_confidence * 0.9) {
            m_state = 2;
            m_current_mu = m_dfe_params.mu_track;
        }
        // If C drops too low, fall back to STARTUP
        if (m_C < 0.3) {
            m_state = 0;
            m_current_mu = m_dfe_params.mu_startup;
        }
        break;

    case 2:  // TRACK
        // Steady state: monitor for degradation
        // If C drops or Delta is too large, fall back to ACQUIRE
        if (m_C < m_vref_params.target_confidence * 0.7 ||
            std::abs(m_Delta) > m_vref_params.asymmetry_alert) {
            m_state = 1;
            m_current_mu = m_dfe_params.mu_acquire;
        }
        break;

    default:
        m_state = 0;
        m_current_mu = m_dfe_params.mu_startup;
        break;
    }
}

// ============================================================================
// Write tap coefficients to DE output ports
// ============================================================================
void DfeAdaptTdf::write_tap_outputs()
{
    tap1_de.write(m_taps[0]);
    tap2_de.write(m_taps[1]);
    tap3_de.write(m_taps[2]);
    tap4_de.write(m_taps[3]);
    tap5_de.write(m_taps[4]);
    tap6_de.write(m_taps[5]);
    tap7_de.write(m_taps[6]);
    tap8_de.write(m_taps[7]);
}

// ============================================================================
// Utility functions
// ============================================================================
double DfeAdaptTdf::clamp(double val, double min_val, double max_val) const
{
    return std::max(min_val, std::min(max_val, val));
}

double DfeAdaptTdf::sign_fn(double val)
{
    if (val > 0) return 1.0;
    if (val < 0) return -1.0;
    return 0.0;
}

} // namespace serdes
