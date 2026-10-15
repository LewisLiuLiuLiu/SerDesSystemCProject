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
                                     double ui,
                                     unsigned int seed)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_params(params)
    , m_lfsr_state(0)
    , m_sample_rate(sample_rate)
    , m_ui(ui)
    , m_samples_per_ui(0)
    , m_sample_counter(0)
    , m_current_bit_value(0.0)
    , m_time(0.0)
    , m_seed(seed)
    , m_rng(seed)
{
    // Parameter validation
    if (sample_rate <= 0.0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    if (ui <= 0.0) {
        throw std::invalid_argument("UI must be positive");
    }
    if (params.single_pulse < 0.0) {
        throw std::invalid_argument("Single pulse width cannot be negative");
    }
    
    // Calculate oversampling ratio
    double exact_samples = ui * sample_rate;
    m_samples_per_ui = static_cast<int>(std::round(exact_samples));
    
    if (m_samples_per_ui < 1) {
        throw std::invalid_argument("Sample rate must be at least 1/UI");
    }
    
    std::cout << "  [WaveGen] Data rate: " << (1.0/ui)/1e9 << " Gbps" << std::endl;
    std::cout << "  [WaveGen] Sample rate: " << sample_rate/1e9 << " GHz" << std::endl;
    std::cout << "  [WaveGen] UI: " << ui*1e12 << " ps" << std::endl;
    std::cout << "  [WaveGen] Samples per UI: " << m_samples_per_ui << std::endl;
}

void WaveGenerationTdf::set_attributes() {
    out.set_rate(1);
    // Calculate timestep from UI and samples_per_ui
    double ui = m_ui;
    int samples_per_ui = m_samples_per_ui;
    double timestep = ui / samples_per_ui;
    
    std::cout << "  [WaveGen] set_attributes: ui=" << ui*1e12 << " ps, samples_per_ui=" 
              << samples_per_ui << ", timestep=" << timestep*1e12 << " ps" << std::endl;
    
    out.set_timestep(timestep, sc_core::SC_SEC);
}

void WaveGenerationTdf::initialize() {
    // Reset time and counter
    m_time = 0.0;
    m_sample_counter = 0;
    
    // Initialize LFSR state based on PRBS type
    int prbs_index = static_cast<int>(m_params.type);
    unsigned int default_state = 0;
    unsigned int mask = 0;
    
    if (prbs_index >= 0 && prbs_index < 5) {
        default_state = PRBS_CONFIGS[prbs_index].default_init;
        mask = PRBS_CONFIGS[prbs_index].mask;
    } else {
        // Default to PRBS31
        default_state = 0x7FFFFFFF;
        mask = 0x7FFFFFFF;
    }
    
    // Use seed to modify LFSR initial state
    m_lfsr_state = (default_state ^ (m_seed & mask)) & mask;
    
    // Check for all-zero state
    if (m_lfsr_state == 0) {
        m_lfsr_state = default_state;
    }
    
    // Generate first bit
    if (m_params.single_pulse > 0.0) {
        m_current_bit_value = 1.0;  // Pulse starts high
    } else {
        bool bit = generate_prbs_bit();
        m_current_bit_value = bit ? 1.0 : -1.0;
    }
    
    // Re-seed RNG for jitter
    m_rng.seed(m_seed);
    
    // Warning for pulse width quantization
    if (m_params.single_pulse > 0.0) {
        double ui_count = m_params.single_pulse / m_ui;
        if (std::abs(ui_count - std::round(ui_count)) > 1e-9) {
            std::cerr << "Warning: single_pulse is not an integer multiple of UI" << std::endl;
        }
    }
}

bool WaveGenerationTdf::generate_prbs_bit() {
    int prbs_index = static_cast<int>(m_params.type);
    unsigned int feedback = 0;
    
    if (prbs_index >= 0 && prbs_index < 5) {
        const PRBSConfig& config = PRBS_CONFIGS[prbs_index];
        feedback = ((m_lfsr_state >> config.tap1) ^ (m_lfsr_state >> config.tap2)) & 0x1;
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & config.mask;
    } else {
        feedback = ((m_lfsr_state >> 30) ^ (m_lfsr_state >> 27)) & 0x1;
        m_lfsr_state = ((m_lfsr_state << 1) | feedback) & 0x7FFFFFFF;
    }
    
    return (m_lfsr_state & 0x1) != 0;
}

void WaveGenerationTdf::processing() {
    // Mode selection: Single-bit pulse vs PRBS
    if (m_params.single_pulse > 0.0) {
        // Single-bit pulse mode: 每个采样点按时长判定；用采样点数的无量纲比较，
        // 避免累积浮点误差在边界处把脉冲拉长/缩短一个采样点
        double step_index = m_time * m_sample_rate;
        double pulse_steps = m_params.single_pulse * m_sample_rate;
        m_current_bit_value = (step_index < pulse_steps - 1e-9) ? 1.0 : -1.0;
    } else if (m_sample_counter == 0) {
        // PRBS mode - generate next bit at UI boundary using LFSR
        bool bit = generate_prbs_bit();
        m_current_bit_value = bit ? 1.0 : -1.0;
    }
    
    // Write output (held constant during UI for oversampling)
    out.write(m_current_bit_value);
    
    // Update counter and time
    m_sample_counter++;
    if (m_sample_counter >= m_samples_per_ui) {
        m_sample_counter = 0;
    }
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
