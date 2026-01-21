#include "ams/wave_generation.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace serdes {

// PRBS polynomial configuration table
struct PRBSConfig {
    int length;              // LFSR length
    unsigned int mask;       // Mask for state bits
    int tap1;                // First tap position
    int tap2;                // Second tap position
    unsigned int default_init;  // Default initial state
};

// Standard PRBS configurations per ITU-T O.150
static const PRBSConfig PRBS_CONFIGS[] = {
    {7,  0x7F,       6,  5, 0x7F},       // PRBS7:  x^7 + x^6 + 1
    {9,  0x1FF,      8,  4, 0x1FF},      // PRBS9:  x^9 + x^5 + 1
    {15, 0x7FFF,     14, 13, 0x7FFF},    // PRBS15: x^15 + x^14 + 1
    {23, 0x7FFFFF,   22, 17, 0x7FFFFF},  // PRBS23: x^23 + x^18 + 1
    {31, 0x7FFFFFFF, 30, 27, 0x7FFFFFFF} // PRBS31: x^31 + x^28 + 1
};

WaveGenerationTdf::WaveGenerationTdf(sc_core::sc_module_name nm, 
                                     const WaveGenParams& params,
                                     double sample_rate,
                                     unsigned int seed)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_params(params)
    , m_lfsr_state(0)
    , m_sample_rate(sample_rate)
    , m_time(0.0)
    , m_seed(seed)
{
    // Parameter validation
    if (sample_rate <= 0.0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    if (params.single_pulse < 0.0) {
        throw std::invalid_argument("Single pulse width cannot be negative");
    }
    
    // Initialize random number generator
    m_rng.seed(m_seed);
}

void WaveGenerationTdf::set_attributes() {
    out.set_rate(1);
    out.set_timestep(1.0 / m_sample_rate, sc_core::SC_SEC);
}

void WaveGenerationTdf::initialize() {
    // Reset time
    m_time = 0.0;
    
    // Initialize LFSR state based on PRBS type
    int prbs_index = static_cast<int>(m_params.type);
    if (prbs_index >= 0 && prbs_index < 5) {
        m_lfsr_state = PRBS_CONFIGS[prbs_index].default_init;
    } else {
        // Default to PRBS31
        m_lfsr_state = 0x7FFFFFFF;
    }
    
    // Check for all-zero state (would cause LFSR to lock up)
    if (m_lfsr_state == 0) {
        std::cerr << "Warning: LFSR initial state is zero, using default value" << std::endl;
        m_lfsr_state = PRBS_CONFIGS[prbs_index >= 0 && prbs_index < 5 ? prbs_index : 4].default_init;
    }
    
    // Warning for pulse width quantization
    if (m_params.single_pulse > 0.0) {
        double timestep = 1.0 / m_sample_rate;
        double ratio = m_params.single_pulse / timestep;
        if (std::abs(ratio - std::round(ratio)) > 1e-9) {
            std::cerr << "Warning: single_pulse is not an integer multiple of timestep, "
                      << "actual pulse width may differ slightly" << std::endl;
        }
    }
    
    // Re-seed RNG for reproducibility
    m_rng.seed(m_seed);
}

bool WaveGenerationTdf::generate_prbs_bit() {
    int prbs_index = static_cast<int>(m_params.type);
    unsigned int feedback = 0;
    
    if (prbs_index >= 0 && prbs_index < 5) {
        const PRBSConfig& config = PRBS_CONFIGS[prbs_index];
        // Calculate feedback bit (XOR of two taps)
        feedback = ((m_lfsr_state >> config.tap1) ^ (m_lfsr_state >> config.tap2)) & 0x1;
        // Shift left and insert feedback
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & config.mask;
    } else {
        // Default to PRBS31
        feedback = ((m_lfsr_state >> 30) ^ (m_lfsr_state >> 27)) & 0x1;
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & 0x7FFFFFFF;
    }
    
    // Return LSB as output bit
    return (m_lfsr_state & 0x1) != 0;
}

void WaveGenerationTdf::processing() {
    double bit_value = 0.0;
    
    // Mode selection: Single-bit pulse vs PRBS
    if (m_params.single_pulse > 0.0) {
        // Single-bit pulse mode
        // Output +1.0V during pulse, -1.0V after
        if (m_time < m_params.single_pulse) {
            bit_value = 1.0;   // Pulse high level
        } else {
            bit_value = -1.0;  // Pulse low level (settled)
        }
    } else {
        // PRBS mode - generate next bit using LFSR
        bool bit = generate_prbs_bit();
        // NRZ modulation: bit 0 -> -1.0V, bit 1 -> +1.0V
        bit_value = bit ? 1.0 : -1.0;
    }
    
    // Jitter injection (simplified implementation)
    // Note: This is a demonstration implementation that calculates jitter
    // but doesn't truly modify sampling timestamps (TDF limitation)
    double jitter_offset = 0.0;
    
    // Random Jitter (RJ) - Gaussian distribution
    if (m_params.jitter.RJ_sigma > 0.0) {
        std::normal_distribution<double> dist(0.0, m_params.jitter.RJ_sigma);
        jitter_offset += dist(m_rng);
    }
    
    // Sinusoidal Jitter (SJ) - Multi-tone superposition
    for (size_t i = 0; i < m_params.jitter.SJ_freq.size() && 
                       i < m_params.jitter.SJ_pp.size(); ++i) {
        double sj_phase = 2.0 * M_PI * m_params.jitter.SJ_freq[i] * m_time;
        jitter_offset += m_params.jitter.SJ_pp[i] * std::sin(sj_phase);
    }
    
    // Note: jitter_offset is calculated for demonstration but not applied
    // to the output. True jitter modeling would require DE-TDF bridging
    // or dynamic timestep adjustment, which is not supported in pure TDF.
    (void)jitter_offset;  // Suppress unused variable warning
    
    // Write output
    out.write(bit_value);
    
    // Update time
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
