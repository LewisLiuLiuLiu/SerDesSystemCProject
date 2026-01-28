#ifndef SERDES_SINGLE_TO_DIFF_H
#define SERDES_SINGLE_TO_DIFF_H

#include <systemc-ams>

namespace serdes {

/**
 * @brief Single-ended to Differential Converter TDF Module
 * 
 * Converts a single-ended signal to differential pair:
 * - out_p = +input (positive terminal)
 * - out_n = -input (negative terminal, inverted)
 * 
 * This module is used to connect single-ended modules (FFE, Mux)
 * to differential input modules (Driver).
 */
class SingleToDiffTdf : public sca_tdf::sca_module {
public:
    // Input port - single-ended signal
    sca_tdf::sca_in<double> in;
    
    // Output ports - differential pair
    sca_tdf::sca_out<double> out_p;  ///< Positive terminal
    sca_tdf::sca_out<double> out_n;  ///< Negative terminal (inverted)
    
    /**
     * @brief Constructor
     * @param nm Module name
     */
    SingleToDiffTdf(sc_core::sc_module_name nm);
    
    // TDF callback methods
    void set_attributes() override;
    void processing() override;
};

} // namespace serdes

#endif // SERDES_SINGLE_TO_DIFF_H
