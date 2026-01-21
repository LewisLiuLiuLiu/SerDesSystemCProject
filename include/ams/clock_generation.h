/**
 * @file clock_generation.h
 * @brief Clock Generator module for SerDes system
 * 
 * This module implements clock generation with multiple modes:
 * - IDEAL: Ideal clock with no jitter or noise
 * - PLL: Analog Phase-Locked Loop (parameter structure defined, not yet implemented)
 * - ADPLL: All-Digital PLL (parameter structure defined, not yet implemented)
 * 
 * Features:
 * - Phase accumulator architecture for precise phase generation
 * - Adaptive time step based on clock frequency
 * - Phase output in radians (0 to 2*pi range)
 * - Support for future PLL/ADPLL extension
 * 
 * @note Current implementation supports IDEAL mode only.
 *       PLL and ADPLL modes are planned for future versions.
 * 
 * @version 0.2
 * @date 2026-01-21
 */

#ifndef SERDES_CLOCK_GENERATION_H
#define SERDES_CLOCK_GENERATION_H

#include <systemc-ams>
#include "common/parameters.h"

namespace serdes {

/**
 * @class ClockGenerationTdf
 * @brief TDF module implementing Clock Generation
 * 
 * The clock generator provides phase output for downstream modules
 * (Sampler, CDR) to determine sampling instants.
 * 
 * Signal Processing Flow:
 * 1. Output current phase value
 * 2. Calculate phase increment (delta_phi = 2*pi * f * delta_t)
 * 3. Update phase accumulator
 * 4. Normalize phase to [0, 2*pi) range
 * 
 * Design Principles:
 * - Phase accumulator avoids floating-point cumulative errors
 * - Time step adapts to clock frequency (100 samples per period)
 * - Modulo 2*pi operation ensures numerical stability
 */
class ClockGenerationTdf : public sca_tdf::sca_module {
public:
    // ========================================================================
    // Ports
    // ========================================================================
    
    /**
     * @brief Clock phase output port
     * Outputs instantaneous phase in radians (0 to 2*pi range)
     * Downstream modules use this for sampling timing
     */
    sca_tdf::sca_out<double> clk_phase;

    // ========================================================================
    // Constructor
    // ========================================================================
    
    /**
     * @brief Constructor
     * @param nm Module name
     * @param params Clock generation parameters (type, frequency, PLL config)
     */
    ClockGenerationTdf(sc_core::sc_module_name nm, const ClockParams& params);

    // ========================================================================
    // SystemC-AMS TDF Methods
    // ========================================================================
    
    /**
     * @brief Set module attributes (port rates, time step)
     * 
     * Sets adaptive time step based on clock frequency:
     * timestep = 1 / (frequency * 100)
     * This ensures 100 samples per clock period.
     */
    void set_attributes();
    
    /**
     * @brief Initialize module state
     * 
     * Resets phase accumulator to initial value.
     */
    void initialize();
    
    /**
     * @brief Main processing function (called each time step)
     * 
     * For IDEAL mode:
     * - Outputs current phase
     * - Updates phase accumulator with delta_phi
     * - Normalizes phase to [0, 2*pi) range
     */
    void processing();

    // ========================================================================
    // Debug Interface
    // ========================================================================
    
    /**
     * @brief Get current phase value
     * @return Phase in radians [0, 2*pi)
     */
    double get_phase() const { return m_phase; }
    
    /**
     * @brief Get configured clock frequency
     * @return Frequency in Hz
     */
    double get_frequency() const { return m_frequency; }
    
    /**
     * @brief Get clock type
     * @return ClockType enum value
     */
    ClockType get_type() const { return m_params.type; }
    
    /**
     * @brief Get phase increment per time step
     * @return Phase increment in radians
     */
    double get_phase_increment() const { return m_phase_increment; }
    
    /**
     * @brief Get expected time step
     * @return Time step in seconds
     */
    double get_expected_timestep() const { return 1.0 / (m_frequency * 100.0); }

private:
    // ========================================================================
    // Member Variables
    // ========================================================================
    
    ClockParams m_params;           ///< Clock generation parameters
    double m_phase;                 ///< Current phase accumulator (radians)
    double m_frequency;             ///< Clock frequency (Hz)
    double m_phase_increment;       ///< Phase increment per time step (radians)

    // ========================================================================
    // Private Methods
    // ========================================================================
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate_params();
    
    /**
     * @brief Process IDEAL clock mode
     * Implements ideal clock with linear phase accumulation
     */
    void process_ideal();
    
    /**
     * @brief Process PLL clock mode (placeholder)
     * @note Not yet implemented, falls back to IDEAL mode
     */
    void process_pll();
    
    /**
     * @brief Process ADPLL clock mode (placeholder)
     * @note Not yet implemented, falls back to IDEAL mode
     */
    void process_adpll();
};

} // namespace serdes

#endif // SERDES_CLOCK_GENERATION_H
