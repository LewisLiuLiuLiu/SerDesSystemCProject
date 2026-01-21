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
    double single_pulse;              // Single pulse width (s), >0 enables pulse mode
    JitterParams jitter;
    ModulationParams modulation;
    
    WaveGenParams()
        : type(PRBSType::PRBS31)
        , poly("x^31 + x^28 + 1")
        , init("0x7FFFFFFF")
        , single_pulse(0.0) {}        // Default 0.0 = PRBS mode
};

// ============================================================================
// TX Parameters
// ============================================================================
struct TxFfeParams {
    std::vector<double> taps;
    
    TxFfeParams() : taps({0.2, 0.6, 0.2}) {}
};

struct TxDriverParams {
    // Basic parameters
    double dc_gain;              // DC gain (linear)
    double vswing;               // Differential output swing Vpp (V)
    double vcm_out;              // Output common-mode voltage (V)
    double output_impedance;     // Output impedance (Ohm, differential)
    std::vector<double> poles;   // Pole frequencies (Hz) for bandwidth limiting
    std::string sat_mode;        // Saturation mode: "soft"/"hard"/"none"
    double vlin;                 // Soft saturation linear range parameter (V)
    
    // PSRR (Power Supply Rejection Ratio) sub-structure
    struct PsrrParams {
        bool enable;                     // Enable PSRR modeling
        double gain;                     // PSRR path gain (linear, e.g., 0.01 = -40dB)
        std::vector<double> poles;       // PSRR lowpass filter poles (Hz)
        double vdd_nom;                  // Nominal supply voltage (V)
        
        PsrrParams()
            : enable(false)
            , gain(0.01)
            , poles({1e9})
            , vdd_nom(1.0) {}
    } psrr;
    
    // Imbalance sub-structure
    struct ImbalanceParams {
        double gain_mismatch;            // Gain mismatch (%)
        double skew;                     // Phase skew (s)
        
        ImbalanceParams()
            : gain_mismatch(0.0)
            , skew(0.0) {}
    } imbalance;
    
    // Slew rate limiting sub-structure
    struct SlewRateParams {
        bool enable;                     // Enable slew rate limiting
        double max_slew_rate;            // Maximum slew rate (V/s)
        
        SlewRateParams()
            : enable(false)
            , max_slew_rate(1e12) {}
    } slew_rate;
    
    TxDriverParams()
        : dc_gain(1.0)
        , vswing(0.8)
        , vcm_out(0.6)
        , output_impedance(50.0)
        , poles({50e9})
        , sat_mode("soft")
        , vlin(1.0) {}
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
    
    RxVgaParams()
        : zeros({1e9})               // Default 1 GHz zero
        , poles({20e9})              // Default 20 GHz pole
        , dc_gain(2.0)               // Default 2.0 gain (VGA typically higher than CTLE)
        , vcm_out(0.6)
        , offset_enable(false)
        , vos(0.0)
        , noise_enable(false)
        , vnoise_sigma(0.0)
        , sat_min(-0.5)
        , sat_max(0.5) {}
};

struct RxSamplerParams {
    double threshold;
    double hysteresis;
    double sample_delay;
    std::string phase_source;
    double resolution;
    
    // Offset configuration
    bool offset_enable;
    double offset_value;
    
    // Noise configuration
    bool noise_enable;
    double noise_sigma;
    unsigned int noise_seed;
    
    RxSamplerParams()
        : threshold(0.0)
        , hysteresis(0.02)
        , sample_delay(0.0)
        , phase_source("clock")
        , resolution(0.02)
        , offset_enable(false)
        , offset_value(0.0)
        , noise_enable(false)
        , noise_sigma(0.0)
        , noise_seed(DEFAULT_SEED) {}  
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
    double kp;                    // Proportional gain
    double ki;                    // Integral gain
    double edge_threshold;        // Edge detection threshold (normalized)
    bool adaptive_threshold;      // Adaptive threshold enable
    
    CdrPiParams()
        : kp(0.01)
        , ki(1e-4)
        , edge_threshold(0.5)
        , adaptive_threshold(false) {}
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
    bool debug_enable;            // Debug output enable
    
    CdrParams() : debug_enable(false) {}
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
// Adaption Parameters (DE domain adaptive control)
// ============================================================================
struct AdaptionParams {
    // Basic parameters
    double Fs;                       // Sampling rate (Hz)
    double UI;                       // Unit interval (s)
    unsigned int seed;               // Random seed
    std::string update_mode;         // "event"|"periodic"|"multi-rate"
    double fast_update_period;       // Fast path update period (s)
    double slow_update_period;       // Slow path update period (s)
    
    // AGC (Automatic Gain Control) parameters
    struct AgcParams {
        bool enabled;                // Enable AGC algorithm
        double target_amplitude;     // Target amplitude (V)
        double kp;                   // PI proportional coefficient
        double ki;                   // PI integral coefficient
        double gain_min;             // Minimum gain (linear)
        double gain_max;             // Maximum gain (linear)
        double rate_limit;           // Gain change rate limit (linear/s)
        double initial_gain;         // Initial gain (linear)
        
        AgcParams()
            : enabled(true)
            , target_amplitude(0.4)
            , kp(0.1)
            , ki(100.0)
            , gain_min(0.5)
            , gain_max(8.0)
            , rate_limit(10.0)
            , initial_gain(2.0) {}
    } agc;
    
    // DFE (Decision Feedback Equalizer) tap update parameters
    struct DfeAdaptParams {
        bool enabled;                // Enable DFE online update
        int num_taps;                // Number of taps (1-8)
        std::string algorithm;       // Update algorithm: "lms"|"sign-lms"|"nlms"
        double mu;                   // Step size coefficient
        double leakage;              // Leakage coefficient (0-1)
        std::vector<double> initial_taps;  // Initial tap coefficients
        double tap_min;              // Minimum tap value
        double tap_max;              // Maximum tap value
        double freeze_threshold;     // Error threshold to freeze update
        
        DfeAdaptParams()
            : enabled(true)
            , num_taps(5)
            , algorithm("sign-lms")
            , mu(1e-4)
            , leakage(1e-6)
            , initial_taps({-0.05, -0.02, 0.01, 0.005, 0.002})
            , tap_min(-0.5)
            , tap_max(0.5)
            , freeze_threshold(0.5) {}
    } dfe;
    
    // Threshold adaptation parameters
    struct ThresholdAdaptParams {
        bool enabled;                // Enable threshold adaptation
        double initial;              // Initial threshold (V)
        double hysteresis;           // Hysteresis window (V)
        double adapt_step;           // Adjustment step size (V/update)
        double target_ber;           // Target BER for optimization
        double drift_threshold;      // Level drift threshold (V)
        
        ThresholdAdaptParams()
            : enabled(true)
            , initial(0.0)
            , hysteresis(0.02)
            , adapt_step(0.001)
            , target_ber(1e-12)
            , drift_threshold(0.05) {}
    } threshold;
    
    // CDR PI controller parameters
    struct CdrPiAdaptParams {
        bool enabled;                // Enable PI control
        double kp;                   // Proportional coefficient
        double ki;                   // Integral coefficient
        double phase_resolution;     // Phase command resolution (s)
        double phase_range;          // Phase command range (±s)
        bool anti_windup;            // Enable anti-windup
        double initial_phase;        // Initial phase command (s)
        
        CdrPiAdaptParams()
            : enabled(true)
            , kp(0.01)
            , ki(1e-4)
            , phase_resolution(1e-12)
            , phase_range(5e-11)
            , anti_windup(true)
            , initial_phase(0.0) {}
    } cdr_pi;
    
    // Safety and rollback parameters
    struct SafetyParams {
        bool freeze_on_error;        // Freeze all updates on error
        bool rollback_enable;        // Enable parameter rollback
        double snapshot_interval;    // Snapshot save interval (s)
        int error_burst_threshold;   // Error burst threshold to trigger freeze
        
        SafetyParams()
            : freeze_on_error(true)
            , rollback_enable(true)
            , snapshot_interval(1e-6)
            , error_burst_threshold(100) {}
    } safety;
    
    // Default constructor
    AdaptionParams()
        : Fs(80e9)
        , UI(2.5e-11)
        , seed(12345)
        , update_mode("multi-rate")
        , fast_update_period(2.5e-10)
        , slow_update_period(2.5e-7) {}
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
    AdaptionParams adaption;
};

} // namespace serdes

#endif // SERDES_COMMON_PARAMETERS_H
