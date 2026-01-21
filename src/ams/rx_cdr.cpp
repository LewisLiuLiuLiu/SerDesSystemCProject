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
    , m_params(params)
    , m_phase(0.0)
    , m_prev_bit(0.0)
    , m_integral(0.0)
    , m_last_phase_error(0.0)
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
}

// ============================================================================
// Initialization
// ============================================================================

void RxCdrTdf::initialize()
{
    // Reset internal state
    m_phase = 0.0;
    m_prev_bit = 0.0;
    m_integral = 0.0;
    m_last_phase_error = 0.0;
}

// ============================================================================
// Main Processing
// ============================================================================

void RxCdrTdf::processing()
{
    // ========================================================================
    // Step 1: Read input data
    // ========================================================================
    double current_bit = in.read();
    
    // ========================================================================
    // Step 2-3: Edge detection + Bang-Bang phase detection
    // ========================================================================
    // Detect data transition by comparing with previous bit
    // If transition magnitude exceeds threshold, determine early/late
    
    double phase_error = 0.0;
    double threshold = m_params.pi.edge_threshold;
    double bit_diff = current_bit - m_prev_bit;
    
    if (std::abs(bit_diff) > threshold) {
        // Edge detected
        if (bit_diff > 0) {
            // Rising edge (0 -> 1): clock is late, need to advance
            phase_error = 1.0;
        } else {
            // Falling edge (1 -> 0): clock is early, need to delay
            phase_error = -1.0;
        }
    }
    // No edge: phase_error remains 0
    
    // Store for debug interface
    m_last_phase_error = phase_error;
    
    // ========================================================================
    // Step 4: PI controller update
    // ========================================================================
    // Standard PI controller: output = Kp * error + Ki * integral(error)
    // Discrete-time implementation:
    //   I[n] = I[n-1] + Ki * e[n]
    //   output[n] = Kp * e[n] + I[n]
    
    // Update integral term (accumulate phase error)
    m_integral += m_params.pi.ki * phase_error;
    
    // Calculate proportional term (instantaneous response)
    double prop_term = m_params.pi.kp * phase_error;
    
    // Total PI output
    m_phase = prop_term + m_integral;
    
    // ========================================================================
    // Step 5: Phase range limiting (clamp to ±range)
    // ========================================================================
    // Prevent phase accumulation from exceeding interpolator range
    double range = m_params.pai.range;
    m_phase = std::max(-range, std::min(range, m_phase));
    
    // Anti-windup: also limit integral state when hitting range limits
    if (m_phase >= range || m_phase <= -range) {
        // Clamp integral to prevent further windup
        m_integral = m_phase - prop_term;
    }
    
    // ========================================================================
    // Step 6: Phase quantization
    // ========================================================================
    // Quantize to phase interpolator resolution
    double resolution = m_params.pai.resolution;
    double quantized_phase = std::round(m_phase / resolution) * resolution;
    
    // ========================================================================
    // Step 7: Output and state update
    // ========================================================================
    // Update previous bit for next edge detection
    m_prev_bit = current_bit;
    
    // Write quantized phase adjustment to output port (unit: seconds)
    phase_out.write(quantized_phase);
}

} // namespace serdes
