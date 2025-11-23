#ifndef SERDES_COMMON_CONSTANTS_H
#define SERDES_COMMON_CONSTANTS_H

#include <cmath>

namespace serdes {

// ============================================================================
// Physical Constants
// ============================================================================

constexpr double PI = 3.141592653589793;
constexpr double TWO_PI = 2.0 * PI;
constexpr double E = 2.718281828459045;

// Speed of light (m/s)
constexpr double SPEED_OF_LIGHT = 299792458.0;

// Boltzmann constant (J/K)
constexpr double BOLTZMANN_CONST = 1.380649e-23;

// Absolute zero (K)
constexpr double ABSOLUTE_ZERO = -273.15;

// Standard temperature (K)
constexpr double STANDARD_TEMP = 300.0;

// ============================================================================
// Numerical Constants
// ============================================================================

// Small value for floating point comparisons
constexpr double EPSILON = 1e-15;

// Default tolerance for numerical operations
constexpr double DEFAULT_TOLERANCE = 1e-12;

// Maximum iterations for convergence
constexpr int MAX_ITERATIONS = 1000;

// ============================================================================
// SerDes Specific Constants
// ============================================================================

// Default sampling rate (Hz)
constexpr double DEFAULT_SAMPLING_RATE = 80e9;

// Default bit rate (bps)
constexpr double DEFAULT_BIT_RATE = 40e9;

// Default unit interval (s)
constexpr double DEFAULT_UI = 25e-12;

// Default simulation duration (s)
constexpr double DEFAULT_DURATION = 1e-6;

// Default random seed
constexpr unsigned int DEFAULT_SEED = 12345;

// Minimum/Maximum gain values
constexpr double MIN_GAIN = 0.1;
constexpr double MAX_GAIN = 100.0;

// Minimum/Maximum bandwidth (Hz)
constexpr double MIN_BANDWIDTH = 1e6;
constexpr double MAX_BANDWIDTH = 100e9;

} // namespace serdes

#endif // SERDES_COMMON_CONSTANTS_H
