#ifndef SERDES_TX_TOP_H
#define SERDES_TX_TOP_H

#include <systemc-ams>
#include "common/parameters.h"
#include "ams/tx_ffe.h"
#include "ams/tx_mux.h"
#include "ams/single_to_diff.h"
#include "ams/tx_driver.h"

namespace serdes {

/**
 * @brief TX Top-level Module
 * 
 * Integrates the complete TX signal chain:
 *   Input → FFE → Mux → SingleToDiff → Driver → Differential Output
 * 
 * Signal Flow:
 * - External single-ended input (from WaveGen or other source)
 * - FFE: Feed-Forward Equalizer for pre-emphasis/de-emphasis
 * - Mux: Lane selection (bypass in single-lane mode)
 * - SingleToDiff: Convert single-ended to differential
 * - Driver: Final output stage with bandwidth limiting, saturation, etc.
 * 
 * Note: WaveGen is NOT included in this module. It should be instantiated
 * externally and connected to the 'in' port.
 */
SC_MODULE(TxTopModule) {
public:
    // ========================================================================
    // External Ports (exposed to parent module)
    // ========================================================================
    
    // Input port - single-ended signal from external source (e.g., WaveGen)
    sca_tdf::sca_in<double> in;
    
    // Power supply input - for PSRR modeling
    sca_tdf::sca_in<double> vdd;
    
    // Differential output ports
    sca_tdf::sca_out<double> out_p;  ///< Positive terminal
    sca_tdf::sca_out<double> out_n;  ///< Negative terminal
    
    // ========================================================================
    // Constructor
    // ========================================================================
    
    /**
     * @brief Construct TX top module
     * @param nm Module name
     * @param tx_params TX parameters (FFE, Mux lane, Driver)
     */
    TxTopModule(sc_core::sc_module_name nm, const TxParams& tx_params);
    
    /**
     * @brief Destructor
     */
    ~TxTopModule();
    
    // ========================================================================
    // Debug Interface
    // ========================================================================
    
    /**
     * @brief Get FFE output signal (for debugging)
     */
    const sca_tdf::sca_signal<double>& get_ffe_out_signal() const { 
        return m_sig_ffe_out; 
    }
    
    /**
     * @brief Get Mux output signal (for debugging)
     */
    const sca_tdf::sca_signal<double>& get_mux_out_signal() const { 
        return m_sig_mux_out; 
    }
    
private:
    // ========================================================================
    // Sub-modules
    // ========================================================================
    TxFfeTdf* m_ffe;              ///< Feed-Forward Equalizer
    TxMuxTdf* m_mux;              ///< Lane Multiplexer
    SingleToDiffTdf* m_s2d;       ///< Single-ended to Differential converter
    TxDriverTdf* m_driver;        ///< Output Driver
    
    // ========================================================================
    // Internal Signals
    // ========================================================================
    sca_tdf::sca_signal<double> m_sig_ffe_out;   ///< FFE output
    sca_tdf::sca_signal<double> m_sig_mux_out;   ///< Mux output
    sca_tdf::sca_signal<double> m_sig_diff_p;    ///< Differential positive (S2D output)
    sca_tdf::sca_signal<double> m_sig_diff_n;    ///< Differential negative (S2D output)
    
    // ========================================================================
    // Parameters
    // ========================================================================
    TxParams m_params;
};

} // namespace serdes

#endif // SERDES_TX_TOP_H
