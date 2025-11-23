#ifndef SERDES_COMMON_TYPES_H
#define SERDES_COMMON_TYPES_H

#include <string>
#include <vector>
#include <cstdint>

namespace serdes {

// ============================================================================
// Basic Types
// ============================================================================

using BitStream = std::vector<bool>;
using TapCoefficients = std::vector<double>;
using FrequencyList = std::vector<double>;

// ============================================================================
// Enumerations
// ============================================================================

// PRBS types
enum class PRBSType {
    PRBS7,
    PRBS9,
    PRBS15,
    PRBS23,
    PRBS31,
    CUSTOM
};

// Modulation types
enum class ModulationType {
    NRZ,      // Non-Return-to-Zero
    PAM2,     // Pulse Amplitude Modulation - 2 levels
    PAM4,     // Pulse Amplitude Modulation - 4 levels
    PAM8      // Pulse Amplitude Modulation - 8 levels
};

// Clock generation types
enum class ClockType {
    IDEAL,    // Ideal clock (no jitter/noise)
    PLL,      // Phase-Locked Loop
    ADPLL     // All-Digital PLL
};

// Phase detector types
enum class PhaseDetectorType {
    BANG_BANG,
    LINEAR,
    TRI_STATE,
    HOGGE
};

// DFE update algorithms
enum class DFEUpdateAlgorithm {
    LMS,           // Least Mean Squares
    SIGN_LMS,      // Sign-LMS
    NLMS,          // Normalized LMS
    RLS,           // Recursive Least Squares
    FIXED          // Fixed coefficients
};

// Build types
enum class BuildType {
    DEBUG,
    RELEASE
};

// Trace format
enum class TraceFormat {
    VCD,       // Value Change Dump
    TABULAR,   // Tabular format
    CSV        // Comma Separated Values
};

// ============================================================================
// Utility Functions
// ============================================================================

// Convert PRBS type to string
inline std::string PRBSTypeToString(PRBSType type) {
    switch (type) {
        case PRBSType::PRBS7:  return "PRBS7";
        case PRBSType::PRBS9:  return "PRBS9";
        case PRBSType::PRBS15: return "PRBS15";
        case PRBSType::PRBS23: return "PRBS23";
        case PRBSType::PRBS31: return "PRBS31";
        case PRBSType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

// Convert string to PRBS type
inline PRBSType StringToPRBSType(const std::string& str) {
    if (str == "PRBS7")  return PRBSType::PRBS7;
    if (str == "PRBS9")  return PRBSType::PRBS9;
    if (str == "PRBS15") return PRBSType::PRBS15;
    if (str == "PRBS23") return PRBSType::PRBS23;
    if (str == "PRBS31") return PRBSType::PRBS31;
    if (str == "CUSTOM") return PRBSType::CUSTOM;
    return PRBSType::PRBS31; // Default
}

// Convert modulation type to string
inline std::string ModulationTypeToString(ModulationType type) {
    switch (type) {
        case ModulationType::NRZ:  return "NRZ";
        case ModulationType::PAM2: return "PAM2";
        case ModulationType::PAM4: return "PAM4";
        case ModulationType::PAM8: return "PAM8";
        default: return "UNKNOWN";
    }
}

// Convert clock type to string
inline std::string ClockTypeToString(ClockType type) {
    switch (type) {
        case ClockType::IDEAL: return "IDEAL";
        case ClockType::PLL:   return "PLL";
        case ClockType::ADPLL: return "ADPLL";
        default: return "UNKNOWN";
    }
}

} // namespace serdes

#endif // SERDES_COMMON_TYPES_H
