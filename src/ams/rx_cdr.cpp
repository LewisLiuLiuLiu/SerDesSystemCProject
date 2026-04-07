/**
 * @file rx_cdr.cpp
 * @brief Implementation of Clock and Data Recovery (CDR) module
 * 
 * @version 0.2
 * @date 2026-01-20
 */

#include "ams/rx_cdr.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace serdes {

// ============================================================================
// Constructor
// ============================================================================

RxCdrTdf::RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , phase_out("phase_out")
    , sampling_trigger("sampling_trigger")
    , m_params(params)
    , m_sample_state(SampleState::WAIT_EDGE)
    , m_edge_sample(false)
    , m_data_sample(false)
    , m_prev_data_sample(false)
    , m_phase(0.0)
    , m_integral(0.0)
    , m_last_phase_error(0.0)
    , m_free_running_phase(0.0)
{
    // Validate parameters during construction
    validate_params();
}

// ============================================================================
// Parameter Validation
// ============================================================================

void RxCdrTdf::validate_params()
{
    // Validate PI controller parameters
    if (m_params.pi.kp < 0.0) {
        throw std::invalid_argument("CDR: Kp must be non-negative");
    }
    if (m_params.pi.ki < 0.0) {
        throw std::invalid_argument("CDR: Ki must be non-negative");
    }
    if (m_params.pi.edge_threshold <= 0.0) {
        throw std::invalid_argument("CDR: edge_threshold must be positive");
    }
    
    // Validate phase interpolator parameters
    if (m_params.pai.resolution <= 0.0) {
        throw std::invalid_argument("CDR: PAI resolution must be positive");
    }
    if (m_params.pai.range <= 0.0) {
        throw std::invalid_argument("CDR: PAI range must be positive");
    }
    if (m_params.pai.range < m_params.pai.resolution) {
        throw std::invalid_argument("CDR: PAI range must be >= resolution");
    }
}

// ============================================================================
// TDF Attribute Setup
// ============================================================================

void RxCdrTdf::set_attributes()
{
    // Set port rates (1:1 processing)
    in.set_rate(1);
    phase_out.set_rate(1);
    phase_out.set_delay(1);  // 添加延迟以打破 Sampler-CDR 反馈环路
    sampling_trigger.set_rate(1);
    sampling_trigger.set_delay(1);  // 添加延迟以打破环路
}

// ============================================================================
// Initialization
// ============================================================================

void RxCdrTdf::initialize()
{
    // Reset internal state
    m_phase = 0.0;
    m_integral = 0.0;
    m_last_phase_error = 0.0;
    m_free_running_phase = 0.0;
    
    // Initialize sampling state machine
    m_sample_state = SampleState::WAIT_EDGE;
    m_edge_sample = false;
    m_data_sample = false;
    m_prev_data_sample = false;
}

// ============================================================================
// Main Processing
// ============================================================================

void RxCdrTdf::processing()
{
    double timestep = get_timestep().to_seconds();
    
    // ========================================================================
    // Step 1: Read sampled value from Sampler (result of previous trigger)
    // ========================================================================
    // Due to TDF delay, we read the sample from the previous trigger here
    bool sampled_value = (in.read() > 0.5);
    
    // Store sample based on what we were waiting for
    if (m_sample_state == SampleState::WAIT_DATA) {
        // We were waiting for data sample (second trigger in UI)
        m_data_sample = sampled_value;
        
        // =====================================================================
        // Step 2: Bang-Bang Phase Detection (now we have both samples)
        // =====================================================================
        double phase_error = 0.0;
        
        // Check for data transition by comparing with previous UI's data
        if (m_data_sample != m_prev_data_sample) {
            // Data transition detected - can perform BBPD
            if (m_edge_sample == m_data_sample) {
                // Edge sample equals new data value → clock is EARLY
                // (we sampled edge after the transition)
                phase_error = -1.0;
            } else {
                // Edge sample equals old data value → clock is LATE
                // (we sampled edge before the transition)
                phase_error = +1.0;
            }
        }
        // No transition: phase_error remains 0 (no update)
        
        // Store for debug
        m_last_phase_error = phase_error;
        
        // =====================================================================
        // Step 3: PI controller update
        // =====================================================================
        // Update integral term
        m_integral += m_params.pi.ki * phase_error;
        
        // Calculate proportional term
        double prop_term = m_params.pi.kp * phase_error;
        
        // Total PI output (in UI units)
        double pi_output = prop_term + m_integral;
        
        // Scale to seconds
        m_phase = pi_output * m_params.ui;
        
        // =====================================================================
        // Step 4: Phase range limiting
        // =====================================================================
        double range = m_params.pai.range;
        m_phase = std::max(-range, std::min(range, m_phase));
        
        // Anti-windup
        if (m_phase >= range || m_phase <= -range) {
            m_integral = m_phase / m_params.ui - prop_term;
        }
        
        // Save current data sample for next comparison
        m_prev_data_sample = m_data_sample;
        
        // Reset state for next UI
        m_sample_state = SampleState::WAIT_EDGE;
    }
    else if (m_sample_state == SampleState::WAIT_EDGE) {
        // We were waiting for edge sample (first trigger in UI)
        m_edge_sample = sampled_value;
        // Next we wait for data sample
        m_sample_state = SampleState::WAIT_DATA;
    }
    
    // ========================================================================
    // Step 5: Generate sampling triggers (two per UI)
    // ========================================================================
    // Update free-running phase
    m_free_running_phase += timestep;
    
    // Calculate total phase with current CDR adjustment
    double quantized_phase = std::round(m_phase / m_params.pai.resolution) 
                             * m_params.pai.resolution;
    double total_phase = m_free_running_phase + quantized_phase;
    
    // Phase within UI
    double phase_in_ui = std::fmod(total_phase, m_params.ui);
    if (phase_in_ui < 0) phase_in_ui += m_params.ui;
    
    // Previous phase
    double prev_total = total_phase - timestep;
    double prev_in_ui = std::fmod(prev_total, m_params.ui);
    if (prev_in_ui < 0) prev_in_ui += m_params.ui;
    
    // Two sampling points within UI
    // Edge at UI boundary (0 or UI), Data at UI/2
    double edge_point = 0.0;  // or m_params.ui (same position)
    double data_point = m_params.ui / 2.0;
    
    bool trigger = false;
    
    // Check which trigger to generate based on state
    if (m_sample_state == SampleState::WAIT_EDGE) {
        // Waiting for edge trigger (at UI boundary)
        if (prev_in_ui < edge_point && phase_in_ui >= edge_point) {
            trigger = true;
        }
        // Handle wrap-around at UI boundary
        else if (prev_in_ui > phase_in_ui && phase_in_ui >= edge_point) {
            trigger = true;
        }
    }
    else { // WAIT_DATA
        // Waiting for data trigger (at UI/2)
        if (prev_in_ui < data_point && phase_in_ui >= data_point) {
            trigger = true;
        }
    }
    
    // ========================================================================
    // Step 6: Output
    // ========================================================================
    // Output the actual phase (not quantized) for better visualization
    phase_out.write(m_phase);
    sampling_trigger.write(trigger);
}

} // namespace serdes
