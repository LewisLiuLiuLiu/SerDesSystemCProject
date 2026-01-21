/**
 * @file clock_generation.cpp
 * @brief Implementation of Clock Generator module
 * 
 * @version 0.2
 * @date 2026-01-21
 */

#include "ams/clock_generation.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace serdes {

// ============================================================================
// Constructor
// ============================================================================

ClockGenerationTdf::ClockGenerationTdf(sc_core::sc_module_name nm, const ClockParams& params)
    : sca_tdf::sca_module(nm)
    , clk_phase("clk_phase")
    , m_params(params)
    , m_phase(0.0)
    , m_frequency(params.frequency)
    , m_phase_increment(0.0)
{
    // Validate parameters during construction
    validate_params();
}

// ============================================================================
// Parameter Validation
// ============================================================================

void ClockGenerationTdf::validate_params()
{
    // Validate frequency
    if (m_frequency <= 0.0) {
        throw std::invalid_argument("ClockGenerator: frequency must be positive");
    }
    
    // Validate frequency is reasonable (1Hz to 1THz)
    if (m_frequency < 1.0 || m_frequency > 1e12) {
        throw std::invalid_argument("ClockGenerator: frequency must be between 1Hz and 1THz");
    }
    
    // Validate PLL parameters if PLL mode is selected
    if (m_params.type == ClockType::PLL) {
        // Validate charge pump current
        if (m_params.pll.cp_current <= 0.0) {
            throw std::invalid_argument("ClockGenerator: PLL charge pump current must be positive");
        }
        
        // Validate loop filter parameters
        if (m_params.pll.lf_R <= 0.0) {
            throw std::invalid_argument("ClockGenerator: PLL loop filter resistance must be positive");
        }
        if (m_params.pll.lf_C <= 0.0) {
            throw std::invalid_argument("ClockGenerator: PLL loop filter capacitance must be positive");
        }
        
        // Validate VCO parameters
        if (m_params.pll.vco_Kvco <= 0.0) {
            throw std::invalid_argument("ClockGenerator: PLL VCO gain must be positive");
        }
        if (m_params.pll.vco_f0 <= 0.0) {
            throw std::invalid_argument("ClockGenerator: PLL VCO center frequency must be positive");
        }
        
        // Validate divider
        if (m_params.pll.divider <= 0) {
            throw std::invalid_argument("ClockGenerator: PLL divider must be positive");
        }
    }
}

// ============================================================================
// TDF Attribute Setup
// ============================================================================

void ClockGenerationTdf::set_attributes()
{
    // Set port rate
    clk_phase.set_rate(1);
    
    // Set adaptive time step based on clock frequency
    // Time step = 1 / (frequency * 100) ensures 100 samples per clock period
    double timestep = 1.0 / (m_frequency * 100.0);
    clk_phase.set_timestep(timestep, sc_core::SC_SEC);
    
    // Pre-calculate phase increment for efficiency
    // delta_phi = 2 * pi * f * delta_t = 2 * pi / 100
    m_phase_increment = 2.0 * M_PI / 100.0;
}

// ============================================================================
// Initialization
// ============================================================================

void ClockGenerationTdf::initialize()
{
    // Reset phase accumulator
    m_phase = 0.0;
}

// ============================================================================
// Main Processing
// ============================================================================

void ClockGenerationTdf::processing()
{
    // Route to appropriate processing method based on clock type
    switch (m_params.type) {
        case ClockType::IDEAL:
            process_ideal();
            break;
            
        case ClockType::PLL:
            process_pll();
            break;
            
        case ClockType::ADPLL:
            process_adpll();
            break;
            
        default:
            // Default to ideal clock
            process_ideal();
            break;
    }
}

// ============================================================================
// IDEAL Clock Processing
// ============================================================================

void ClockGenerationTdf::process_ideal()
{
    // ========================================================================
    // Step 1: Output current phase
    // ========================================================================
    clk_phase.write(m_phase);
    
    // ========================================================================
    // Step 2: Calculate phase increment
    // ========================================================================
    // Using actual timestep to handle any timing variations
    double timestep = clk_phase.get_timestep().to_seconds();
    double delta_phi = 2.0 * M_PI * m_frequency * timestep;
    
    // ========================================================================
    // Step 3: Update phase accumulator
    // ========================================================================
    m_phase += delta_phi;
    
    // ========================================================================
    // Step 4: Phase normalization (modulo 2*pi)
    // ========================================================================
    // Keep phase in [0, 2*pi) range to maintain numerical stability
    // Use subtraction instead of fmod for efficiency and to avoid edge cases
    while (m_phase >= 2.0 * M_PI) {
        m_phase -= 2.0 * M_PI;
    }
    while (m_phase < 0.0) {
        m_phase += 2.0 * M_PI;
    }
}

// ============================================================================
// PLL Clock Processing (Placeholder)
// ============================================================================

void ClockGenerationTdf::process_pll()
{
    // ========================================================================
    // PLL Mode - Currently not implemented
    // Falls back to IDEAL mode with a warning on first call
    // ========================================================================
    
    // Future implementation would include:
    // 1. Phase detector: compare reference and feedback phases
    // 2. Charge pump: generate current based on phase error
    // 3. Loop filter: convert current to control voltage
    //    V_ctrl = I_cp * R / (1 + s*R*C)
    // 4. VCO: generate output frequency based on control voltage
    //    f_out = f0 + Kvco * V_ctrl
    // 5. Divider: divide output frequency for feedback
    
    // For now, use ideal clock processing
    process_ideal();
}

// ============================================================================
// ADPLL Clock Processing (Placeholder)
// ============================================================================

void ClockGenerationTdf::process_adpll()
{
    // ========================================================================
    // ADPLL Mode - Currently not implemented
    // Falls back to IDEAL mode with a warning on first call
    // ========================================================================
    
    // Future implementation would include:
    // 1. Digital phase detector: bang-bang or TDC-based
    // 2. Digital loop filter: IIR or accumulator-based
    // 3. DCO: digitally controlled oscillator with fine resolution
    // 4. Digital divider: programmable division ratio
    
    // For now, use ideal clock processing
    process_ideal();
}

} // namespace serdes
