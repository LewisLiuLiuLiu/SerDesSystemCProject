#ifndef SERDES_BIQUAD_FILTER_H
#define SERDES_BIQUAD_FILTER_H

#include <systemc-ams>
#include <vector>
#include <complex>
#include <memory>

namespace serdes {

/**
 * Biquad filter section implementing a second-order IIR filter
 * 
 * Uses Direct Form II with trapezoidal integration for numerical stability.
 * 
 * For complex conjugate pole pair from pole-residue form:
 *   H(s) = r/(s-p) + conj(r)/(s-conj(p)) = (b1*s + b0) / (s^2 + a1*s + a2)
 * 
 * where:
 *   b1 = 2*Re(r)
 *   b0 = -2*Re(p*conj(r))
 *   a1 = -2*Re(p)
 *   a2 = |p|^2
 */
class BiquadSection {
public:
    BiquadSection();
    
    /**
     * Initialize with continuous-time coefficients
     * H(s) = (b0 + b1*s) / (s^2 + a1*s + a2)
     */
    void initialize(double b0, double b1, double a1, double a2, double timestep);
    
    double process(double input);
    void reset();
    bool is_initialized() const { return m_initialized; }

private:
    // Continuous-time coefficients (for reference)
    double m_b0, m_b1, m_a1, m_a2;
    double m_timestep;
    
    // Discrete-time state-space representation
    // x[n+1] = A*x[n] + B*u[n]
    // y[n]   = C*x[n] + D*u[n]
    double m_A11, m_A12, m_A21, m_A22;  // State matrix
    double m_B1, m_B2;                   // Input matrix
    double m_C1, m_C2;                   // Output matrix
    double m_D;                          // Direct feedthrough
    
    // State variables
    double m_x1, m_x2;
    
    bool m_initialized;
};

/**
 * Pole-residue filter data
 */
struct PoleResidueFilterData {
    std::vector<double> poles_real;
    std::vector<double> poles_imag;
    std::vector<double> residues_real;
    std::vector<double> residues_imag;
    double constant = 0.0;
    double proportional = 0.0;
    int order = 0;
    double dc_gain = 1.0;
    double mse = 0.0;
};

} // namespace serdes

#endif // SERDES_BIQUAD_FILTER_H
