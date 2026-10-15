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
 * @version 0.2
 * @date 2026-01-20
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
    double m_free_running_phase;  ///< Free-running phase accumulator for sampling trigger generation

    // ========================================================================
    // Private Methods
    // ========================================================================
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate_params();
};

} // namespace serdes

#endif // SERDES_RX_CDR_H
