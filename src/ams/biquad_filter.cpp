#include "ams/biquad_filter.h"
#include <cmath>
#include <iostream>

namespace serdes {

BiquadSection::BiquadSection()
    : m_b0(0.0), m_b1(0.0), m_a1(0.0), m_a2(0.0)
    , m_timestep(1e-12)
    , m_A11(0.0), m_A12(0.0), m_A21(0.0), m_A22(0.0)
    , m_B1(0.0), m_B2(0.0)
    , m_C1(0.0), m_C2(0.0)
    , m_D(0.0)
    , m_x1(0.0), m_x2(0.0)
    , m_initialized(false)
{
}

void BiquadSection::initialize(double b0, double b1, double a1, double a2, double timestep) {
    m_b0 = b0;
    m_b1 = b1;
    m_a1 = a1;
    m_a2 = a2;
    m_timestep = timestep;
    
    // Convert continuous-time transfer function to state-space
    // H(s) = (b0 + b1*s) / (s^2 + a1*s + a2)
    // 
    // Controllable canonical form:
    // A = [ 0      1    ]   B = [ 0 ]
    //     [ -a2    -a1  ]       [ 1 ]
    // C = [ b0     b1   ]   D = 0
    
    // Discretize using matrix exponential (simplified for second-order)
    // For small timestep, use: Ad = I + A*dt + (A*dt)^2/2
    
    double dt = timestep;
    
    // Continuous A matrix
    double Ac11 = 0.0, Ac12 = 1.0;
    double Ac21 = -a2, Ac22 = -a1;
    
    // Simple Euler discretization (for stability, use small enough dt)
    // Or use bilinear transform on state-space
    
    // Using bilinear transform approach for state-space:
    // Ad = (I - A*dt/2)^-1 * (I + A*dt/2)
    // Bd = (I - A*dt/2)^-1 * B * dt
    
    double I_minus_A_dt_2_11 = 1.0 - Ac11 * dt / 2.0;
    double I_minus_A_dt_2_12 = -Ac12 * dt / 2.0;
    double I_minus_A_dt_2_21 = -Ac21 * dt / 2.0;
    double I_minus_A_dt_2_22 = 1.0 - Ac22 * dt / 2.0;
    
    // Inverse of (I - A*dt/2)
    double det = I_minus_A_dt_2_11 * I_minus_A_dt_2_22 - I_minus_A_dt_2_12 * I_minus_A_dt_2_21;
    if (std::abs(det) < 1e-20) {
        std::cerr << "[ERROR] BiquadSection: Singular matrix in discretization" << std::endl;
        m_initialized = false;
        return;
    }
    
    double inv_11 = I_minus_A_dt_2_22 / det;
    double inv_12 = -I_minus_A_dt_2_12 / det;
    double inv_21 = -I_minus_A_dt_2_21 / det;
    double inv_22 = I_minus_A_dt_2_11 / det;
    
    double I_plus_A_dt_2_11 = 1.0 + Ac11 * dt / 2.0;
    double I_plus_A_dt_2_12 = Ac12 * dt / 2.0;
    double I_plus_A_dt_2_21 = Ac21 * dt / 2.0;
    double I_plus_A_dt_2_22 = 1.0 + Ac22 * dt / 2.0;
    
    // Ad = inv(I - A*dt/2) * (I + A*dt/2)
    m_A11 = inv_11 * I_plus_A_dt_2_11 + inv_12 * I_plus_A_dt_2_21;
    m_A12 = inv_11 * I_plus_A_dt_2_12 + inv_12 * I_plus_A_dt_2_22;
    m_A21 = inv_21 * I_plus_A_dt_2_11 + inv_22 * I_plus_A_dt_2_21;
    m_A22 = inv_21 * I_plus_A_dt_2_12 + inv_22 * I_plus_A_dt_2_22;
    
    // B = [0; 1], so Bd = inv(I - A*dt/2) * [0; 1] * dt
    m_B1 = inv_12 * dt;
    m_B2 = inv_22 * dt;
    
    // C = [b0, b1], D = 0
    m_C1 = b0;
    m_C2 = b1;
    m_D = 0.0;
    
    m_initialized = true;
    
    std::cout << "[DEBUG] BiquadSection initialized:"
              << " b0=" << b0 << " b1=" << b1 
              << " a1=" << a1 << " a2=" << a2 << std::endl;
}

double BiquadSection::process(double input) {
    if (!m_initialized) {
        return input;
    }
    
    // State update: x[n+1] = A*x[n] + B*u[n]
    double x1_new = m_A11 * m_x1 + m_A12 * m_x2 + m_B1 * input;
    double x2_new = m_A21 * m_x1 + m_A22 * m_x2 + m_B2 * input;
    
    // Output: y[n] = C*x[n] + D*u[n]
    double output = m_C1 * m_x1 + m_C2 * m_x2 + m_D * input;
    
    // Update state
    m_x1 = x1_new;
    m_x2 = x2_new;
    
    return output;
}

void BiquadSection::reset() {
    m_x1 = 0.0;
    m_x2 = 0.0;
}

} // namespace serdes
