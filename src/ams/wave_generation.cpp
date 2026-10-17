#include "ams/wave_generation.h"
#include <algorithm>
#include <cmath>
#include <fstream>
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
    , m_edge_from(0.0)
    , m_edge_to(0.0)
    , m_edge_start(0.0)
    , m_time(0.0)
    , m_seed(seed)
    , m_rng(seed)
    , m_jit_edge_ptr(0)
    , m_jit_sample_index(0)
    , m_jit_last_level(0.0)
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
    if (m_params.edge_mode != "ideal") {
        if (m_params.edge_mode != "erf" && m_params.edge_mode != "linear") {
            std::cerr << "  [WaveGen] unknown edge_mode '" << m_params.edge_mode
                      << "' — falling back to ideal" << std::endl;
            m_params.edge_mode = "ideal";
        } else if (m_params.edge_tr <= 0.0) {
            std::cerr << "  [WaveGen] edge_tr must be > 0 — disabling edge shaping" << std::endl;
            m_params.edge_mode = "ideal";
        } else {
            std::cout << "  [WaveGen] Edge shaping: " << m_params.edge_mode
                      << ", tr=" << m_params.edge_tr * 1e12 << " ps (PULSE-01 tau=tr/2.326)"
                      << std::endl;
        }
    }
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

    // T-S4 A2: no pending transition at t=0
    m_edge_from = m_current_bit_value;
    m_edge_to = m_current_bit_value;
    m_edge_start = 0.0;

    // Re-seed RNG for jitter
    m_rng.seed(m_seed);

    // T-S3 A1: edge-level jitter (EDGE-01) — precompute bits, per-edge shifts and tau CSV
    if (m_params.jitter.enable) {
        build_jittered_edges();
    }
    
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

// ============================================================================
// T-S3 A1: edge-level jitter (EDGE-01)
//   x(t) = sum_m b_m g(t - (mT + e_m)),  e_m = e_m^RJ + e_m^DCD + e_m^SJ
//   RJ: iid Gaussian per edge; DCD: rising +delta / falling -delta (per-edge
//   direction, NOT per-symbol); SJ: A*sin(2*pi*f*t_nominal + phase).
//   The shifted edge is represented on the 2-ps grid by linear interpolation of
//   the step over one sample interval. Applied tau sequence is written to CSV.
// ============================================================================
void WaveGenerationTdf::build_jittered_edges() {
    const long long n_ui = m_params.ui_count;
    if (n_ui <= 0) {
        std::cerr << "[WaveGen] edge jitter enabled but ui_count<=0 — jitter disabled" << std::endl;
        return;
    }
    m_jit_levels.resize(static_cast<size_t>(n_ui));
    for (long long k = 0; k < n_ui; ++k) {
        bool bit = generate_prbs_bit();     // UI k bit (initialize() already consumed bit #1)
        m_jit_levels[k] = bit ? 1.0 : -1.0;
    }

    m_jit_edges.clear();
    m_jit_edge_ptr = 0;
    m_jit_sample_index = 0;

    const double ts = 1.0 / m_sample_rate;
    const double max_shift = 0.45 * m_ui;   // keep edges ordered / within one UI
    std::normal_distribution<double> rj_dist(0.0, m_params.jitter.RJ_sigma);
    const size_t n_sj = std::min(m_params.jitter.SJ_freq.size(), m_params.jitter.SJ_pp.size());

    std::ofstream csv;
    if (!m_params.jitter.edge_csv.empty()) {
        csv.open(m_params.jitter.edge_csv);
        if (csv.is_open()) {
            csv << "edge_index,nominal_time_ps,e_total_ps,e_rj_ps,e_dcd_ps,e_sj_ps\n";
        } else {
            std::cerr << "[WaveGen] cannot open tau CSV: " << m_params.jitter.edge_csv << std::endl;
        }
    }

    long long edge_index = 0;
    long long n_clamped = 0;
    for (long long k = 1; k < n_ui; ++k) {
        if (m_jit_levels[k] == m_jit_levels[k - 1]) continue;   // transitions only
        const bool rising = (m_jit_levels[k] > m_jit_levels[k - 1]);
        const double t_nom = static_cast<double>(k) * m_ui;

        const double e_rj = (m_params.jitter.RJ_sigma > 0.0) ? rj_dist(m_rng) : 0.0;
        const double e_dcd = rising ? m_params.jitter.DCD_delta : -m_params.jitter.DCD_delta;
        double e_sj = 0.0;
        for (size_t i = 0; i < n_sj; ++i) {
            const double amp = 0.5 * m_params.jitter.SJ_pp[i];   // SJ_pp -> peak amplitude
            e_sj += amp * std::sin(2.0 * M_PI * m_params.jitter.SJ_freq[i] * t_nom
                                   + m_params.jitter.SJ_phase);
        }
        const double e_raw = e_rj + e_dcd + e_sj;
        const double e_app = std::max(-max_shift, std::min(max_shift, e_raw));
        if (e_app != e_raw) ++n_clamped;

        const double p = (t_nom + e_app) / ts;
        long long q = static_cast<long long>(std::floor(p));
        double alpha = p - static_cast<double>(q);
        if (alpha < 0.0) alpha = 0.0;
        if (alpha >= 1.0) { alpha -= 1.0; ++q; }

        JitterEdge e;
        e.sample_index = q;
        e.alpha = alpha;
        e.old_level = m_jit_levels[k - 1];
        e.new_level = m_jit_levels[k];
        m_jit_edges.push_back(e);

        if (csv.is_open()) {
            csv << edge_index << "," << (t_nom * 1e12) << "," << (e_app * 1e12) << ","
                << (e_rj * 1e12) << "," << (e_dcd * 1e12) << "," << (e_sj * 1e12) << "\n";
        }
        ++edge_index;
    }
    if (csv.is_open()) csv.close();

    m_jit_last_level = m_jit_levels.back();
    std::cout << "  [WaveGen] Edge jitter (EDGE-01): " << n_ui << " UI, "
              << m_jit_edges.size() << " edges";
    if (n_clamped > 0) std::cout << ", clamped=" << n_clamped;
    if (!m_params.jitter.edge_csv.empty()) std::cout << ", tau CSV: " << m_params.jitter.edge_csv;
    std::cout << std::endl;
}

void WaveGenerationTdf::processing() {
    // T-S3 A1: edge-level jitter path (precomputed levels + shifted edges)
    if (m_params.jitter.enable && !m_jit_levels.empty()) {
        double value;
        const long long n = m_jit_sample_index;
        if (m_jit_edge_ptr < m_jit_edges.size()) {
            const JitterEdge& e = m_jit_edges[m_jit_edge_ptr];
            if (n < e.sample_index) {
                value = e.old_level;
            } else if (n == e.sample_index) {
                // linear interpolation of the shifted step over one sample interval
                value = e.new_level - e.alpha * (e.new_level - e.old_level);
            } else {
                value = e.new_level;
                ++m_jit_edge_ptr;
            }
        } else {
            value = m_jit_last_level;
        }
        out.write(value);
        ++m_jit_sample_index;
        m_time += 1.0 / m_sample_rate;
        return;
    }

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
        const double new_level = bit ? 1.0 : -1.0;
        m_edge_from = m_current_bit_value;
        m_edge_to = new_level;
        m_edge_start = m_time;
        m_current_bit_value = new_level;
    }

    // T-S4 A2: causal finite-edge shaping (PRBS path; jitter path returns above).
    // The transition starts at the UI boundary and is a causal step from the old
    // level to the new one. For "erf", tau = tr/2.326 reproduces the PULSE-01
    // edge derivative (s4p_channel.pulse_spectrum); the absolute edge delay is a
    // constant absorbed by the sampling-phase calibration.
    double value = m_current_bit_value;
    if (m_params.single_pulse <= 0.0 && m_params.edge_mode != "ideal"
        && m_edge_from != m_edge_to) {
        const double t = m_time - m_edge_start;
        double frac;
        if (m_params.edge_mode == "linear") {
            const double dur = m_params.edge_tr / 0.8;   // 10–90% = 0.8 full ramp
            frac = (dur > 0.0) ? (t / dur) : 1.0;
        } else {
            const double tau = m_params.edge_tr / 2.326;
            // Causal erf step: s(0)~0.01, s(tr)=0.5, s(2tr)~0.99.
            frac = 0.5 * (1.0 + std::erf((t - m_params.edge_tr) / tau));
        }
        frac = std::max(0.0, std::min(1.0, frac));
        value = m_edge_from + (m_edge_to - m_edge_from) * frac;
    }

    // Write output (held constant during UI for oversampling, or shaped edge)
    out.write(value);
    
    // Update counter and time
    m_sample_counter++;
    if (m_sample_counter >= m_samples_per_ui) {
        m_sample_counter = 0;
    }
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
