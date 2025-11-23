#ifndef SERDES_COMMON_PARAMETERS_H
#define SERDES_COMMON_PARAMETERS_H

#include "types.h"
#include "constants.h"
#include <vector>
#include <string>

namespace serdes {

// ============================================================================
// Global Parameters
// ============================================================================
struct GlobalParams {
    double Fs;           // Sampling rate (Hz)
    double UI;           // Unit interval (s)
    double duration;     // Simulation duration (s)
    unsigned int seed;   // Random seed
    
    // Default constructor
    GlobalParams()
        : Fs(DEFAULT_SAMPLING_RATE)
        , UI(DEFAULT_UI)
        , duration(DEFAULT_DURATION)
        , seed(DEFAULT_SEED) {}
};

// ============================================================================
// Wave Generation Parameters
// ============================================================================
struct JitterParams {
    double RJ_sigma;                  // Random jitter standard deviation (s)
    std::vector<double> SJ_freq;      // Sinusoidal jitter frequencies (Hz)
    std::vector<double> SJ_pp;        // Sinusoidal jitter peak-to-peak values (s)
    
    JitterParams() : RJ_sigma(0.0) {}
};

struct ModulationParams {
    double AM;  // Amplitude modulation index
    double PM;  // Phase modulation index
    
    ModulationParams() : AM(0.0), PM(0.0) {}
};

struct WaveGenParams {
    PRBSType type;
    std::string poly;                 // Polynomial expression
    std::string init;                 // Initial state (hex)
    JitterParams jitter;
    ModulationParams modulation;
    
    WaveGenParams()
        : type(PRBSType::PRBS31)
        , poly("x^31 + x^28 + 1")
        , init("0x7FFFFFFF") {}
};

// ============================================================================
// TX Parameters
// ============================================================================
struct TxFfeParams {
    std::vector<double> taps;
    
    TxFfeParams() : taps({0.2, 0.6, 0.2}) {}
};

struct TxDriverParams {
    double swing;            // Output swing (V)
    double bw;               // Bandwidth (Hz)
    double sat;              // Saturation level (V)
    
    TxDriverParams()
        : swing(0.8)
        , bw(20e9)
        , sat(1.0) {}
};

struct TxParams {
    TxFfeParams ffe;
    int mux_lane;
    TxDriverParams driver;
    
    TxParams() : mux_lane(0) {}
};

// ============================================================================
// Channel Parameters
// ============================================================================
struct ChannelParams {
    std::string touchstone;     // S-parameter file path
    int ports;                  // Number of ports
    bool crosstalk;             // Crosstalk enable
    bool bidirectional;         // Bidirectional enable
    double attenuation_db;      // Simple model attenuation (dB)
    double bandwidth_hz;        // Simple model bandwidth (Hz)
    
    ChannelParams()
        : touchstone("")
        , ports(2)
        , crosstalk(false)
        , bidirectional(false)
        , attenuation_db(10.0)
        , bandwidth_hz(20e9) {}
};

// ============================================================================
// RX Parameters
// ============================================================================
struct RxCtleParams {
    // Main transfer function
    std::vector<double> zeros;       // Zero frequencies (Hz)
    std::vector<double> poles;       // Pole frequencies (Hz)
    double dc_gain;                  // DC gain (linear)
    
    // Common mode output voltage
    double vcm_out;                  // Differential output common mode voltage (V)
    
    // Offset
    bool offset_enable;              // Offset enable
    double vos;                      // Input offset voltage (V)
    
    // Noise
    bool noise_enable;               // Noise enable
    double vnoise_sigma;             // Noise standard deviation (V)
    
    // Saturation
    double sat_min;                  // Output minimum voltage (V)
    double sat_max;                  // Output maximum voltage (V)
    
    // PSRR (Power Supply Rejection Ratio)
    struct PsrrParams {
        bool enable;
        double gain;                 // PSRR path gain (linear)
        std::vector<double> zeros;   // Hz
        std::vector<double> poles;   // Hz
        double vdd_nom;              // Nominal supply voltage (V)
        
        PsrrParams()
            : enable(false)
            , gain(0.0)
            , vdd_nom(1.0) {}
    } psrr;
    
    // CMFB (Common Mode Feedback)
    struct CmfbParams {
        bool enable;
        double bandwidth;            // Loop bandwidth (Hz)
        double loop_gain;            // Loop gain (linear)
        
        CmfbParams()
            : enable(false)
            , bandwidth(1e6)
            , loop_gain(1.0) {}
    } cmfb;
    
    // CMRR (Common Mode Rejection Ratio)
    struct CmrrParams {
        bool enable;
        double gain;                 // CM->DIFF leakage gain (linear)
        std::vector<double> zeros;   // Hz
        std::vector<double> poles;   // Hz
        
        CmrrParams()
            : enable(false)
            , gain(0.0) {}
    } cmrr;
    
    RxCtleParams()
        : zeros({2e9})
        , poles({30e9})
        , dc_gain(1.5)
        , vcm_out(0.6)
        , offset_enable(false)
        , vos(0.0)
        , noise_enable(false)
        , vnoise_sigma(0.0)
        , sat_min(-0.5)
        , sat_max(0.5) {}
};

struct RxVgaParams {
    double gain;
    bool agc_enable;
    
    RxVgaParams()
        : gain(4.0)
        , agc_enable(false) {}
};

struct RxSamplerParams {
    double threshold;
    double hysteresis;
    
    RxSamplerParams()
        : threshold(0.0)
        , hysteresis(0.02) {}
};

struct RxDfeParams {
    std::vector<double> taps;
    std::string update;              // Update algorithm
    double mu;                       // Step size
    
    RxDfeParams()
        : taps({-0.05, -0.02, 0.01})
        , update("sign-lms")
        , mu(1e-4) {}
};

struct RxParams {
    RxCtleParams ctle;
    RxVgaParams vga;
    RxSamplerParams sampler;
    RxDfeParams dfe;
};

// ============================================================================
// CDR Parameters
// ============================================================================
struct CdrPiParams {
    double kp;   // Proportional gain
    double ki;   // Integral gain
    
    CdrPiParams() : kp(0.01), ki(1e-4) {}
};

struct CdrPaiParams {
    double resolution;  // Phase interpolator resolution (s)
    double range;       // Phase interpolator range (s)
    
    CdrPaiParams()
        : resolution(1e-12)
        , range(5e-11) {}
};

struct CdrParams {
    CdrPiParams pi;
    CdrPaiParams pai;
};

// ============================================================================
// Clock Generation Parameters
// ============================================================================
struct ClockPllParams {
    std::string pd_type;     // Phase detector type
    double cp_current;       // Charge pump current (A)
    double lf_R;             // Loop filter resistance (Ohm)
    double lf_C;             // Loop filter capacitance (F)
    double vco_Kvco;         // VCO gain (Hz/V)
    double vco_f0;           // VCO center frequency (Hz)
    int divider;             // Divider ratio
    
    ClockPllParams()
        : pd_type("tri-state")
        , cp_current(5e-5)
        , lf_R(10000)
        , lf_C(1e-10)
        , vco_Kvco(1e8)
        , vco_f0(1e10)
        , divider(4) {}
};

struct ClockParams {
    ClockType type;
    double frequency;           // Clock frequency (Hz)
    ClockPllParams pll;
    
    ClockParams()
        : type(ClockType::PLL)
        , frequency(40e9) {}
};

// ============================================================================
// Eye Diagram Parameters
// ============================================================================
struct EyeParams {
    int ui_bins;
    int amp_bins;
    double measure_length;
    
    EyeParams()
        : ui_bins(128)
        , amp_bins(128)
        , measure_length(1e-4) {}
};

// ============================================================================
// System Configuration (top-level)
// ============================================================================
struct SystemParams {
    GlobalParams global;
    WaveGenParams wave;
    TxParams tx;
    ChannelParams channel;
    RxParams rx;
    CdrParams cdr;
    ClockParams clock;
    EyeParams eye;
};

} // namespace serdes

#endif // SERDES_COMMON_PARAMETERS_H
