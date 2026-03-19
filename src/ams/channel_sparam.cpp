#include "ams/channel_sparam.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <complex>

// Use nlohmann json for parsing (included in third_party)
#include "../third_party/json.hpp"

using json = nlohmann::json;

namespace serdes {

// ============================================================================
// Constructors and Destructor
// ============================================================================

ChannelSParamTdf::ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_filter_state(0.0)
    , m_alpha(0.3)
    , m_delay_idx(0)
    , m_use_fft(false)
    , m_fft_size(0)
    , m_block_size(0)
    , m_block_idx(0)
    , m_config_loaded(false)
    , m_initialized(false)
{
    // Default to simple model for backward compatibility
    m_ext_params.method = ChannelMethod::SIMPLE;
}

ChannelSParamTdf::ChannelSParamTdf(sc_core::sc_module_name nm, 
                                   const ChannelParams& params,
                                   const ChannelExtendedParams& ext_params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_ext_params(ext_params)
    , m_filter_state(0.0)
    , m_alpha(0.3)
    , m_delay_idx(0)
    , m_use_fft(false)
    , m_fft_size(0)
    , m_block_size(0)
    , m_block_idx(0)
    , m_config_loaded(false)
    , m_initialized(false)
{
    std::cout << "[DEBUG] ChannelSParamTdf: Constructor called, method=" 
              << static_cast<int>(m_ext_params.method) 
              << ", config_file=" << ext_params.config_file << std::endl;
    
    // Load configuration if specified
    if (!ext_params.config_file.empty()) {
        load_config(ext_params.config_file);
        std::cout << "[DEBUG] ChannelSParamTdf: After load_config, method=" 
                  << static_cast<int>(m_ext_params.method) << std::endl;
    }
}

ChannelSParamTdf::~ChannelSParamTdf() {
    // Cleanup if needed
}

// ============================================================================
// SystemC-AMS Lifecycle Methods
// ============================================================================

void ChannelSParamTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
    
    // Set timestep for TDF module
    double timestep = 1.0 / m_ext_params.fs;
    set_timestep(timestep, sc_core::SC_SEC);
    
    // Set delay based on method
    if (m_ext_params.method == ChannelMethod::IMPULSE && m_use_fft) {
        // FFT convolution introduces block delay
        out.set_delay(m_block_size);
    }
}

void ChannelSParamTdf::initialize() {
    if (m_initialized) return;
    
    std::cout << "[DEBUG] ChannelSParamTdf: initialize() called, method=" 
              << static_cast<int>(m_ext_params.method) << std::endl;
    
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE:
            init_simple_model();
            break;
        case ChannelMethod::RATIONAL:
            init_rational_model();
            break;
        case ChannelMethod::IMPULSE:
            init_impulse_model();
            break;
        case ChannelMethod::POLE_RESIDUE:
            init_pole_residue_model();
            break;
        case ChannelMethod::STATE_SPACE:
            init_state_space_model();
            break;
    }
    
    m_initialized = true;
}

void ChannelSParamTdf::processing() {
    double x_in = in.read();
    double y_out = 0.0;
    
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE:
            y_out = process_simple(x_in);
            break;
        case ChannelMethod::RATIONAL:
            y_out = process_rational(x_in);
            break;
        case ChannelMethod::IMPULSE:
            if (m_use_fft) {
                y_out = process_impulse_fft(x_in);
            } else {
                y_out = process_impulse(x_in);
            }
            break;
        case ChannelMethod::POLE_RESIDUE:
            // Use custom state space implementation (numerical integration)
            y_out = process_pole_residue_ss(x_in);
            break;
        case ChannelMethod::STATE_SPACE:
            y_out = process_state_space(x_in);
            break;
    }
    
    out.write(y_out);
}

// ============================================================================
// Configuration Loading
// ============================================================================

bool ChannelSParamTdf::load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "ChannelSParamTdf: Cannot open config file: " << config_path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return parse_json_config(buffer.str());
}

bool ChannelSParamTdf::parse_json_config(const std::string& json_content) {
    try {
        json config = json::parse(json_content);
        
        // Get sampling frequency
        if (config.contains("fs")) {
            m_ext_params.fs = config["fs"].get<double>();
        }
        
        // Get method (case-insensitive comparison)
        std::string method_str = config.value("method", "simple");
        std::cout << "[DEBUG] ChannelSParamTdf: Loading method: " << method_str << std::endl;
        
        // Convert to lowercase for comparison
        std::string method_lower = method_str;
        std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(), ::tolower);
        std::cout << "[DEBUG] ChannelSParamTdf: method_lower: " << method_lower << std::endl;
        
        if (method_lower == "rational") {
            m_ext_params.method = ChannelMethod::RATIONAL;
        } else if (method_lower == "impulse") {
            m_ext_params.method = ChannelMethod::IMPULSE;
        } else if (method_lower == "pole_residue" || method_lower == "pole-residue") {
            m_ext_params.method = ChannelMethod::POLE_RESIDUE;
        } else if (method_lower == "state_space" || method_lower == "state-space") {
            m_ext_params.method = ChannelMethod::STATE_SPACE;
        } else {
            m_ext_params.method = ChannelMethod::SIMPLE;
        }
        
        // Parse filters (rational method)
        if (config.contains("filters")) {
            int filter_count = 0;
            for (auto it = config["filters"].begin(); it != config["filters"].end(); ++it) {
                filter_count++;
            }
            std::cout << "[DEBUG] ChannelSParamTdf: Found " << filter_count << " filters" << std::endl;
        }
        if (config.contains("filters") && !config["filters"].empty()) {
            // Use first filter (typically S21)
            for (auto it = config["filters"].begin(); it != config["filters"].end(); ++it) {
                const auto& filter = it.value();
                m_rational_data.num_coeffs.clear();
                m_rational_data.den_coeffs.clear();
                
                for (auto& v : filter["num"]) {
                    m_rational_data.num_coeffs.push_back(v.get<double>());
                }
                for (auto& v : filter["den"]) {
                    m_rational_data.den_coeffs.push_back(v.get<double>());
                }
                
                m_rational_data.order = filter.value("order", 0);
                m_rational_data.dc_gain = filter.value("dc_gain", 1.0);
                m_rational_data.mse = filter.value("mse", 0.0);
                
                std::cout << "[DEBUG] ChannelSParamTdf: Parsed rational filter '" << it.key() 
                          << "': num=" << m_rational_data.num_coeffs.size() 
                          << ", den=" << m_rational_data.den_coeffs.size() 
                          << ", order=" << m_rational_data.order << std::endl;
                
                break; // Only use first filter for now
            }
        }
        
        // Parse pole_residue filters (old format with filters object)
        if (config.contains("pole_residue_filters")) {
            int pr_count = 0;
            for (auto it = config["pole_residue_filters"].begin(); it != config["pole_residue_filters"].end(); ++it) {
                pr_count++;
            }
            std::cout << "[DEBUG] ChannelSParamTdf: Found " << pr_count << " pole-residue filters" << std::endl;
        }
        if (config.contains("pole_residue_filters") && !config["pole_residue_filters"].empty()) {
            for (auto it = config["pole_residue_filters"].begin(); it != config["pole_residue_filters"].end(); ++it) {
                const auto& pr = it.value();
                m_pole_residue_data.poles_real.clear(); m_pole_residue_data.poles_imag.clear();
                m_pole_residue_data.residues_real.clear(); m_pole_residue_data.residues_imag.clear();
                
                // Parse constant term
                m_pole_residue_data.constant = pr.value("constant", 0.0);
                
                // Parse poles and residues (complex pairs)
                if (pr.contains("poles") && pr.contains("residues")) {
                    const auto& poles = pr["poles"];
                    const auto& residues = pr["residues"];
                    
                    for (size_t i = 0; i < poles.size() && i < residues.size(); ++i) {
                        double pr_val = 0.0, pi_val = 0.0;
                        double rr_val = 0.0, ri_val = 0.0;
                        
                        if (poles[i].is_array() && poles[i].size() >= 2) {
                            pr_val = poles[i][0].get<double>();
                            pi_val = poles[i][1].get<double>();
                        }
                        if (residues[i].is_array() && residues[i].size() >= 2) {
                            rr_val = residues[i][0].get<double>();
                            ri_val = residues[i][1].get<double>();
                        }
                        
                        m_pole_residue_data.poles_real.push_back(pr_val); m_pole_residue_data.poles_imag.push_back(pi_val);
                        m_pole_residue_data.residues_real.push_back(rr_val); m_pole_residue_data.residues_imag.push_back(ri_val);
                    }
                }
                
                m_pole_residue_data.order = pr.value("order", static_cast<int>(m_pole_residue_data.poles_real.size()));
                m_pole_residue_data.dc_gain = pr.value("dc_gain", 1.0);
                m_pole_residue_data.mse = pr.value("mse", 0.0);
                
                std::cout << "[DEBUG] ChannelSParamTdf: Parsed pole-residue filter '" << it.key() 
                          << "': poles=" << m_pole_residue_data.poles_real.size() 
                          << ", constant=" << m_pole_residue_data.constant 
                          << ", order=" << m_pole_residue_data.order << std::endl;
                
                break; // Only use first filter for now
            }
        }
        
        // Parse new format pole_residue (from scikit-rf export)
        if (config.contains("pole_residue")) {
            const auto& pr = config["pole_residue"];
            m_pole_residue_data.poles_real.clear(); m_pole_residue_data.poles_imag.clear();
            m_pole_residue_data.residues_real.clear(); m_pole_residue_data.residues_imag.clear();
            
            // Parse constant and proportional terms
            m_pole_residue_data.constant = pr.value("constant", 0.0);
            m_pole_residue_data.proportional = pr.value("proportional", 0.0);
            
            // Parse poles and residues from separate real/imag arrays
            if (pr.contains("poles_real") && pr.contains("poles_imag") &&
                pr.contains("residues_real") && pr.contains("residues_imag")) {
                
                const auto& poles_real = pr["poles_real"];
                const auto& poles_imag = pr["poles_imag"];
                const auto& residues_real = pr["residues_real"];
                const auto& residues_imag = pr["residues_imag"];
                
                size_t n_poles = poles_real.size();
                if (poles_imag.size() < n_poles) n_poles = poles_imag.size();
                if (residues_real.size() < n_poles) n_poles = residues_real.size();
                if (residues_imag.size() < n_poles) n_poles = residues_imag.size();
                
                for (size_t i = 0; i < n_poles; ++i) {
                    double p_real = poles_real[i].get<double>();
                    double p_imag = poles_imag[i].get<double>();
                    double r_real = residues_real[i].get<double>();
                    double r_imag = residues_imag[i].get<double>();
                    
                    m_pole_residue_data.poles_real.push_back(p_real); m_pole_residue_data.poles_imag.push_back(p_imag);
                    m_pole_residue_data.residues_real.push_back(r_real); m_pole_residue_data.residues_imag.push_back(r_imag);
                }
            }
            
            m_pole_residue_data.order = pr.value("order", static_cast<int>(m_pole_residue_data.poles_real.size()));
            m_pole_residue_data.dc_gain = pr.value("dc_gain", 1.0);
            m_pole_residue_data.mse = pr.value("mse", 0.0);
            
            std::cout << "[DEBUG] ChannelSParamTdf: Parsed pole_residue (scikit-rf format): poles=" 
                      << m_pole_residue_data.poles_real.size() 
                      << ", constant=" << m_pole_residue_data.constant 
                      << ", proportional=" << m_pole_residue_data.proportional
                      << ", order=" << m_pole_residue_data.order << std::endl;
        }
        
        // Parse impulse responses
        if (config.contains("impulse_responses")) {
            int ir_count = 0;
            for (auto it = config["impulse_responses"].begin(); it != config["impulse_responses"].end(); ++it) {
                ir_count++;
            }
            std::cout << "[DEBUG] ChannelSParamTdf: Found " << ir_count << " impulse responses" << std::endl;
        }
        if (config.contains("impulse_responses") && !config["impulse_responses"].empty()) {
            for (auto it = config["impulse_responses"].begin(); it != config["impulse_responses"].end(); ++it) {
                const auto& ir = it.value();
                m_impulse_data.impulse.clear();
                m_impulse_data.time.clear();
                
                for (auto& v : ir["impulse"]) {
                    m_impulse_data.impulse.push_back(v.get<double>());
                }
                if (ir.contains("time")) {
                    for (auto& v : ir["time"]) {
                        m_impulse_data.time.push_back(v.get<double>());
                    }
                }
                
                m_impulse_data.length = ir.value("length", static_cast<int>(m_impulse_data.impulse.size()));
                m_impulse_data.dt = ir.value("dt", 1.0 / m_ext_params.fs);
                m_impulse_data.energy = ir.value("energy", 0.0);
                m_impulse_data.peak_time = ir.value("peak_time", 0.0);
                
                std::cout << "[DEBUG] ChannelSParamTdf: Parsed impulse response '" << it.key() 
                          << "': impulse=" << m_impulse_data.impulse.size() 
                          << ", time=" << m_impulse_data.time.size() 
                          << ", dt=" << m_impulse_data.dt << std::endl;
                
                break; // Only use first IR for now
            }
        }
        
        std::cout << "[DEBUG] ChannelSParamTdf: Configuration loaded successfully (fs=" 
                  << m_ext_params.fs << ", method=" << static_cast<int>(m_ext_params.method) 
                  << ")" << std::endl;
        m_config_loaded = true;
        return true;
        
    } catch (const json::exception& e) {
        std::cerr << "ChannelSParamTdf: JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Model Initialization
// ============================================================================

void ChannelSParamTdf::init_simple_model() {
    // Calculate filter coefficient from bandwidth
    double omega_c = 2.0 * M_PI * m_params.bandwidth_hz;
    double dt = 1.0 / m_ext_params.fs;
    
    // First-order IIR filter coefficient
    // Using bilinear transform approximation
    m_alpha = omega_c * dt / (1.0 + omega_c * dt);
    
    m_filter_state = 0.0;
}

void ChannelSParamTdf::init_rational_model() {
    if (m_rational_data.num_coeffs.empty() || m_rational_data.den_coeffs.empty()) {
        std::cerr << "ChannelSParamTdf: Rational filter coefficients not loaded" << std::endl;
        // Fall back to simple model
        m_ext_params.method = ChannelMethod::SIMPLE;
        init_simple_model();
        return;
    }
    
    // Initialize sca_vector for LTF
    // VF generates coefficients in DESCENDING order (np.polyval format): [bn, ..., b1, b0]
    // Test both interpretations to find the correct one
    int num_size = static_cast<int>(m_rational_data.num_coeffs.size());
    int den_size = static_cast<int>(m_rational_data.den_coeffs.size());
    
    m_num_vec.resize(num_size);
    m_den_vec.resize(den_size);
    
    // Use coefficients as-is (DESCENDING order like np.polyval)
    // H(s) = (bn*s^n + ... + b1*s + b0) / (an*s^n + ... + a1*s + a0)
    for (int i = 0; i < num_size; ++i) {
        m_num_vec(i) = m_rational_data.num_coeffs[i];
    }
    for (int i = 0; i < den_size; ++i) {
        m_den_vec(i) = m_rational_data.den_coeffs[i];
    }
    
    // DC gain = b0/a0 = last element in DESCENDING array
    if (num_size > 0 && den_size > 0) {
        m_rational_data.dc_gain = m_rational_data.num_coeffs[num_size-1] / m_rational_data.den_coeffs[den_size-1];
    }
    
    std::cout << "[DEBUG] ChannelSParamTdf: Rational filter initialized (DESCENDING order)" << std::endl;
    std::cout << "[DEBUG]   Order: " << m_rational_data.order << std::endl;
    std::cout << "[DEBUG]   DC gain: " << m_rational_data.dc_gain << std::endl;
    std::cout << "[DEBUG]   num[0]=" << m_num_vec(0) << " (highest power)" << std::endl;
    std::cout << "[DEBUG]   num[last]=" << m_num_vec(num_size-1) << " (constant term)" << std::endl;
}

void ChannelSParamTdf::init_impulse_model() {
    if (m_impulse_data.impulse.empty()) {
        std::cerr << "ChannelSParamTdf: Impulse response not loaded" << std::endl;
        // Fall back to simple model
        m_ext_params.method = ChannelMethod::SIMPLE;
        init_simple_model();
        return;
    }
    
    int L = static_cast<int>(m_impulse_data.impulse.size());
    
    // Decide whether to use FFT convolution
    m_use_fft = (L > m_ext_params.impulse.fft_threshold) && m_ext_params.impulse.use_fft;
    
    if (m_use_fft) {
        init_fft_convolution();
    } else {
        // Direct convolution: allocate delay line
        m_delay_line.resize(L, 0.0);
        m_delay_idx = 0;
    }
}

void ChannelSParamTdf::init_state_space_model() {
    try {
        // Access JSON data from config
        if (m_ext_params.config_file.empty()) {
            std::cerr << "ChannelSParamTdf: State-space method requires config file" << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
        // Load and parse the config file
        std::ifstream file(m_ext_params.config_file);
        if (!file.is_open()) {
            std::cerr << "ChannelSParamTdf: Cannot open config file: " << m_ext_params.config_file << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
        json config;
        file >> config;
        file.close();
        
        // Check for state_space section
        if (!config.contains("state_space")) {
            std::cerr << "ChannelSParamTdf: No state_space data in config" << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
        const auto& ss = config["state_space"];
        
        // Parse matrices A, B, C, D, E from JSON
        if (!ss.contains("A") || !ss.contains("B") || !ss.contains("C") || !ss.contains("D")) {
            std::cerr << "ChannelSParamTdf: State-space matrices A, B, C, D required" << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
        // Get dimensions
        const auto& A_json = ss["A"];
        int n_states = A_json.size();
        if (n_states == 0) {
            std::cerr << "ChannelSParamTdf: State-space A matrix is empty" << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
        int n_states_2 = A_json[0].size();
        const auto& C_json = ss["C"];
        int n_outputs = C_json.size();
        
        m_state_space.n_states = n_states;
        m_state_space.n_outputs = n_outputs;
        
        // Resize matrices
        m_state_space.A.resize(n_states, n_states);
        m_state_space.B.resize(n_states, 1);
        m_state_space.C.resize(n_outputs, n_states);
        m_state_space.D.resize(n_outputs, 1);
        m_state_space.E.resize(n_outputs, 1);
        
        // Fill A matrix (n x n)
        for (int i = 0; i < n_states; ++i) {
            for (int j = 0; j < n_states; ++j) {
                m_state_space.A(i+1, j+1) = A_json[i][j].get<double>();
            }
        }
        
        // Fill B matrix (n x 1)
        const auto& B_json = ss["B"];
        for (int i = 0; i < n_states; ++i) {
            m_state_space.B(i+1, 1) = B_json[i][0].get<double>();
        }
        
        // Fill C matrix (n_c x n)
        for (int i = 0; i < n_outputs; ++i) {
            for (int j = 0; j < n_states; ++j) {
                m_state_space.C(i+1, j+1) = C_json[i][j].get<double>();
            }
        }
        
        // Fill D matrix (n_c x 1)
        const auto& D_json = ss["D"];
        for (int i = 0; i < n_outputs; ++i) {
            m_state_space.D(i+1, 1) = D_json[i][0].get<double>();
        }
        
        // Fill E matrix (n_c x 1) - optional, default to 0
        if (ss.contains("E")) {
            const auto& E_json = ss["E"];
            for (int i = 0; i < n_outputs; ++i) {
                m_state_space.E(i+1, 1) = E_json[i][0].get<double>();
            }
        } else {
            for (int i = 0; i < n_outputs; ++i) {
                m_state_space.E(i+1, 1) = 0.0;
            }
        }
        
        // sca_ss filter will be used in processing() with matrices passed each time
        std::cout << "[DEBUG] ChannelSParamTdf: State-space model initialized" << std::endl;
        std::cout << "[DEBUG]   States: " << n_states << ", Outputs: " << n_outputs << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ChannelSParamTdf: Error initializing state-space model: " << e.what() << std::endl;
        m_ext_params.method = ChannelMethod::SIMPLE;
        init_simple_model();
    }
}

double ChannelSParamTdf::process_state_space(double x_in) {
    // Use sca_ss filter to compute output
    // sca_tdf::sca_ss::operator() with state vector:
    // (A, B, C, D, state_vector, input_vector, timestep)
    sca_util::sca_vector<double> in_vec(1);
    in_vec(1) = x_in;
    
    // State vector needs to be stored and maintained between calls
    // Initialize on first use if empty (using length() method)
    if (m_ss_state.length() == 0) {
        m_ss_state.resize(m_state_space.n_states);
        for (int i = 1; i <= m_state_space.n_states; ++i) {
            m_ss_state(i) = 0.0;
        }
    }
    
    // Use the overload without E matrix (E is typically for descriptor systems)
    // The sca_ss operator() returns a proxy that can be assigned to a vector
    sca_util::sca_vector<double> out_vec = m_ss_filter(
        m_state_space.A, m_state_space.B, m_state_space.C,
        m_state_space.D, m_ss_state, in_vec, get_timestep());
    
    return out_vec(1);
}

void ChannelSParamTdf::init_pole_residue_model() {
    size_t n_poles = m_pole_residue_data.poles_real.size();
    std::cout << "[DEBUG] ChannelSParamTdf: init_pole_residue_model called, poles=" 
              << n_poles << std::endl;
    
    if (n_poles == 0) {
        std::cerr << "ChannelSParamTdf: Pole-residue data not loaded" << std::endl;
        m_ext_params.method = ChannelMethod::SIMPLE;
        init_simple_model();
        return;
    }
    
    // Debug: print first few poles
    for (size_t i = 0; i < std::min(size_t(4), n_poles); ++i) {
        double pr = m_pole_residue_data.poles_real[i];
        double pi = m_pole_residue_data.poles_imag[i];
        double rr = m_pole_residue_data.residues_real[i];
        double ri = m_pole_residue_data.residues_imag[i];
        std::cout << "[DEBUG]   Pole[" << i << "]: " << pr << " + " << pi << "i, "
                  << "Residue: " << rr << " + " << ri << "i" << std::endl;
    }
    
    // Clear previous state space data
    m_pr_ss_A_flat.clear();
    m_pr_ss_B_flat.clear();
    m_pr_ss_C_flat.clear();
    m_pr_ss_D_flat.clear();
    m_pr_ss_states.clear();
    m_pr_ss_Ad.clear();
    m_pr_ss_Bd.clear();
    m_pr_ss_Cd.clear();
    m_pr_ss_Dd.clear();
    
    // Reset input history
    m_pr_input_prev = 0.0;
    
    // Calculate Nyquist frequency and timestep
    double nyquist_freq = m_ext_params.fs / 2.0;
    double max_pole_freq = 0.8 * nyquist_freq;
    double dt = 1.0 / m_ext_params.fs;
    
    int used_sections = 0;
    int skipped_sections = 0;
    
    for (size_t i = 0; i < n_poles; ++i) {
        double pr = m_pole_residue_data.poles_real[i];
        double pi = m_pole_residue_data.poles_imag[i];
        double rr = m_pole_residue_data.residues_real[i];
        double ri = m_pole_residue_data.residues_imag[i];
        
        std::complex<double> p1(pr, pi);
        std::complex<double> r1(rr, ri);
        
        // Check pole frequency (skip poles too close to Nyquist)
        double pole_freq = std::abs(p1) / (2.0 * M_PI);
        if (pole_freq > max_pole_freq) {
            skipped_sections++;
            continue;
        }
        
        // Continuous-time state space matrices
        std::vector<double> Ac, Bc, Cc, Dc;
        int n_states = 0;
        
        if (std::abs(pi) > 1e-12) {
            // Complex pole (scikit-rf only outputs upper half-plane poles)
            // For real-valued systems, we need the conjugate pair
            // H(s) = r/(s-p) + r*/(s-p*) = (b1*s + b0) / (s^2 + a1*s + a2)
            
            double b1 = 2.0 * rr;
            double b0 = -2.0 * (rr * pr + ri * pi);
            double a1 = -2.0 * pr;
            double a2 = pr * pr + pi * pi;
            
            // State space (controllable canonical form):
            Ac = {0.0, 1.0, -a2, -a1};  // 2x2
            Bc = {0.0, 1.0};             // 2x1
            Cc = {b0, b1};               // 1x2
            Dc = {0.0};                  // 1x1
            n_states = 2;
            
            used_sections++;
        } else {
            // Real pole: H(s) = r / (s - p)
            Ac = {pr};      // 1x1 (pr is negative for stable poles)
            Bc = {1.0};     // 1x1  (input directly to state)
            Cc = {rr};      // 1x1  (scale by residue)
            Dc = {0.0};     // 1x1
            n_states = 1;
            
            used_sections++;
        }
        
        // Discretize using bilinear transform (Tustin)
        // Convert continuous-time (A,B,C,D) to discrete-time (Ad,Bd,Cd,Dd)
        // Ad = (I - A*dt/2)^-1 * (I + A*dt/2)
        // Bd = (I - A*dt/2)^-1 * B * dt
        // Cd = C * (I - A*dt/2)^-1
        // Dd = D + C * (I - A*dt/2)^-1 * B * dt/2
        
        std::vector<double> Ad(n_states * n_states);
        std::vector<double> Bd(n_states);
        std::vector<double> Cd(n_states);
        double Dd = Dc[0];
        
        if (n_states == 1) {
            // Scalar case: simple formulas
            double a = Ac[0];
            double b = Bc[0];
            double c = Cc[0];
            
            // Ad = (1 + a*dt/2) / (1 - a*dt/2)
            double denom = 1.0 - a * dt / 2.0;
            Ad[0] = (1.0 + a * dt / 2.0) / denom;
            
            // Bd = b * dt / (1 - a*dt/2)
            Bd[0] = b * dt / denom;
            
            // Cd = c / (1 - a*dt/2)
            Cd[0] = c / denom;
            
            // Dd = c * b * dt / (2 * (1 - a*dt/2))
            Dd = Dc[0] + c * b * dt / (2.0 * denom);
        } else {
            // 2x2 case
            // (I - A*dt/2)
            double i00 = 1.0 - Ac[0] * dt / 2.0;
            double i01 = -Ac[1] * dt / 2.0;
            double i10 = -Ac[2] * dt / 2.0;
            double i11 = 1.0 - Ac[3] * dt / 2.0;
            
            // Inverse of (I - A*dt/2)
            double det_inv = 1.0 / (i00 * i11 - i01 * i10);
            double inv00 = i11 * det_inv;
            double inv01 = -i01 * det_inv;
            double inv10 = -i10 * det_inv;
            double inv11 = i00 * det_inv;
            
            // (I + A*dt/2)
            double p00 = 1.0 + Ac[0] * dt / 2.0;
            double p01 = Ac[1] * dt / 2.0;
            double p10 = Ac[2] * dt / 2.0;
            double p11 = 1.0 + Ac[3] * dt / 2.0;
            
            // Ad = inv(I - A*dt/2) * (I + A*dt/2)
            Ad[0] = inv00 * p00 + inv01 * p10;
            Ad[1] = inv00 * p01 + inv01 * p11;
            Ad[2] = inv10 * p00 + inv11 * p10;
            Ad[3] = inv10 * p01 + inv11 * p11;
            
            // Bd = inv(I - A*dt/2) * B * dt
            // B = [0; 1]
            Bd[0] = inv01 * dt;
            Bd[1] = inv11 * dt;
            
            // Cd = C * inv(I - A*dt/2)
            // C = [c0, c1]
            Cd[0] = Cc[0] * inv00 + Cc[1] * inv10;
            Cd[1] = Cc[0] * inv01 + Cc[1] * inv11;
            
            // Dd = D + C * inv(I - A*dt/2) * B * dt/2
            // = D + [Cd0, Cd1] * [0; 1] * dt/2 = D + Cd1 * dt/2
            Dd = Dc[0] + Cd[1] * dt / 2.0;
        }
        
        // Store discrete-time matrices
        m_pr_ss_Ad.push_back(Ad);
        m_pr_ss_Bd.push_back(Bd);
        m_pr_ss_Cd.push_back(Cd);
        m_pr_ss_Dd.push_back(Dd);
        m_pr_ss_states.push_back(std::vector<double>(n_states, 0.0));
    }
    
    std::cout << "[DEBUG] ChannelSParamTdf: Pole-residue filter initialized" << std::endl;
    std::cout << "[DEBUG]   State space sections: " << used_sections << std::endl;
    std::cout << "[DEBUG]   Skipped (freq > 0.8*Nyquist): " << skipped_sections << std::endl;
}

double ChannelSParamTdf::process_pole_residue_ss(double x_in) {
    // Apply constant term
    double y_out = m_pole_residue_data.constant * x_in;
    
    // Apply proportional term if non-zero
    // H_prop(s) = k*s corresponds to y(t) = k * dx/dt
    // In discrete time: y[n] = k * (x[n] - x[n-1]) / dt
    if (std::abs(m_pole_residue_data.proportional) > 1e-20) {
        double dt = 1.0 / m_ext_params.fs;
        y_out += m_pole_residue_data.proportional * (x_in - m_pr_input_prev) / dt;
    }
    m_pr_input_prev = x_in;
    
    // Apply cascaded discrete-time state space sections
    // Each section: x[n+1] = Ad*x[n] + Bd*u[n]
    //               y[n] = Cd*x[n] + Dd*u[n]
    double signal = x_in;
    
    for (size_t i = 0; i < m_pr_ss_states.size(); ++i) {
        const auto& Ad = m_pr_ss_Ad[i];
        const auto& Bd = m_pr_ss_Bd[i];
        const auto& Cd = m_pr_ss_Cd[i];
        double Dd = m_pr_ss_Dd[i];
        auto& state = m_pr_ss_states[i];
        
        int n_states = state.size();
        
        // Compute output: y = Cd*s + Dd*u
        double y = Dd * signal;
        for (int j = 0; j < n_states; ++j) {
            y += Cd[j] * state[j];
        }
        
        // Update state: s[n+1] = Ad*s[n] + Bd*u[n]
        std::vector<double> new_state(n_states);
        for (int j = 0; j < n_states; ++j) {
            new_state[j] = Bd[j] * signal;
            for (int k = 0; k < n_states; ++k) {
                new_state[j] += Ad[j * n_states + k] * state[k];
            }
        }
        
        // Store updated state
        for (int j = 0; j < n_states; ++j) {
            state[j] = new_state[j];
        }
        
        signal = y;
    }
    
    return y_out + signal;
}


void ChannelSParamTdf::init_fft_convolution() {
    int L = static_cast<int>(m_impulse_data.impulse.size());
    
    // Choose FFT size (next power of 2 >= 2*L)
    m_fft_size = 1;
    while (m_fft_size < 2 * L) {
        m_fft_size *= 2;
    }
    
    m_block_size = m_fft_size - L + 1;
    
    // Pre-compute FFT of impulse response (zero-padded)
    std::vector<double> h_padded(m_fft_size, 0.0);
    for (int i = 0; i < L; ++i) {
        h_padded[i] = m_impulse_data.impulse[i];
    }
    
    fft_real(h_padded, m_H_fft_real, m_H_fft_imag);
    
    // Allocate input block buffer
    m_input_block.resize(m_fft_size, 0.0);
    m_block_idx = 0;
    
    // Clear output queue
    while (!m_output_queue.empty()) {
        m_output_queue.pop_front();
    }
}

// ============================================================================
// Signal Processing Methods
// ============================================================================

double ChannelSParamTdf::process_simple(double x) {
    // Attenuation
    double attenuation_linear = std::pow(10.0, -m_params.attenuation_db / 20.0);
    
    // First-order low-pass filter
    m_filter_state = m_alpha * x + (1.0 - m_alpha) * m_filter_state;
    
    return attenuation_linear * m_filter_state;
}

double ChannelSParamTdf::process_rational(double x) {
    // Use sca_ltf_nd for rational function filtering
    // H(s) = num(s) / den(s)
    
    // The sca_ltf_nd operator computes the Laplace transfer function
    // Correct usage: pass numerator, denominator coefficients and input only
    double y = m_ltf_filter(m_num_vec, m_den_vec, x);
    return y;
}

double ChannelSParamTdf::process_impulse(double x) {
    // Direct convolution using circular buffer
    // y[n] = sum_{k=0}^{L-1} h[k] * x[n-k]
    
    int L = static_cast<int>(m_impulse_data.impulse.size());
    
    // Store new input in delay line
    m_delay_line[m_delay_idx] = x;
    
    // Compute convolution
    double y = 0.0;
    for (int k = 0; k < L; ++k) {
        int buf_pos = (m_delay_idx - k + L) % L;
        y += m_impulse_data.impulse[k] * m_delay_line[buf_pos];
    }
    
    // Update delay line index
    m_delay_idx = (m_delay_idx + 1) % L;
    
    return y;
}

double ChannelSParamTdf::get_dc_gain() const {
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE:
            return std::pow(10.0, -m_params.attenuation_db / 20.0);
        case ChannelMethod::RATIONAL:
            return m_rational_data.dc_gain;
        case ChannelMethod::IMPULSE:
            // DC gain is sum of impulse response
            {
                double sum = 0.0;
                for (double h : m_impulse_data.impulse) {
                    sum += h;
                }
                return sum * m_impulse_data.dt;
            }
        case ChannelMethod::POLE_RESIDUE:
            return m_pole_residue_data.dc_gain;
        default:
            return 1.0;
    }
}

} // namespace serdes


namespace serdes {

// ============================================================================
// FFT Implementation - Cooley-Tukey algorithm
// ============================================================================

// Helper: Bit reversal permutation
static void bit_reverse_copy(const std::vector<std::complex<double>>& in,
                              std::vector<std::complex<double>>& out) {
    int N = static_cast<int>(in.size());
    out.resize(N);
    int logN = static_cast<int>(std::log2(N));
    
    for (int i = 0; i < N; ++i) {
        int j = 0;
        for (int k = 0; k < logN; ++k) {
            j = (j << 1) | ((i >> k) & 1);
        }
        out[j] = in[i];
    }
}

// Cooley-Tukey iterative FFT
static void fft_cooley_tukey(std::vector<std::complex<double>>& data, bool inverse = false) {
    int N = static_cast<int>(data.size());
    if (N <= 1) return;
    
    // Check if N is power of 2
    if ((N & (N - 1)) != 0) {
        // Fall back to DFT for non-power-of-2
        std::vector<std::complex<double>> result(N, 0.0);
        for (int k = 0; k < N; ++k) {
            for (int n = 0; n < N; ++n) {
                double sign = inverse ? 1.0 : -1.0;
                double angle = sign * 2.0 * M_PI * k * n / N;
                result[k] += data[n] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
        }
        if (inverse) {
            for (int k = 0; k < N; ++k) {
                result[k] /= N;
            }
        }
        data = result;
        return;
    }
    
    // Bit reversal permutation
    std::vector<std::complex<double>> temp;
    bit_reverse_copy(data, temp);
    
    // Iterative FFT
    double sign = inverse ? 1.0 : -1.0;
    for (int len = 2; len <= N; len <<= 1) {
        double angle = sign * 2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        
        for (int i = 0; i < N; i += len) {
            std::complex<double> w(1.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = temp[i + j];
                std::complex<double> v = temp[i + j + len/2] * w;
                temp[i + j] = u + v;
                temp[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
    
    if (inverse) {
        for (int i = 0; i < N; ++i) {
            temp[i] /= N;
        }
    }
    
    data = temp;
}

double ChannelSParamTdf::process_impulse_fft(double x) {
    // Overlap-save FFT convolution
    
    // Add input to block
    m_input_block[m_block_idx++] = x;
    
    // Process when block is full
    if (m_block_idx == m_block_size) {
        // Shift old samples
        int L = static_cast<int>(m_impulse_data.impulse.size());
        for (int i = 0; i < L - 1; ++i) {
            m_input_block[m_fft_size - L + 1 + i] = m_input_block[i];
        }
        
        // FFT of input block
        std::vector<double> X_real, X_imag;
        fft_real(m_input_block, X_real, X_imag);
        
        // Frequency domain multiplication
        std::vector<double> Y_real(m_fft_size), Y_imag(m_fft_size);
        for (int i = 0; i < m_fft_size; ++i) {
            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            Y_real[i] = X_real[i] * m_H_fft_real[i] - X_imag[i] * m_H_fft_imag[i];
            Y_imag[i] = X_real[i] * m_H_fft_imag[i] + X_imag[i] * m_H_fft_real[i];
        }
        
        // IFFT
        std::vector<double> y_block;
        ifft_real(Y_real, Y_imag, y_block);
        
        // Add valid samples to output queue (discard first L-1 samples)
        int L_minus_1 = L - 1;
        for (int i = L_minus_1; i < m_fft_size; ++i) {
            m_output_queue.push_back(y_block[i]);
        }
        
        m_block_idx = 0;
    }
    
    // Return output from queue
    if (!m_output_queue.empty()) {
        double y = m_output_queue.front();
        m_output_queue.pop_front();
        return y;
    }
    
    return 0.0;
}

void ChannelSParamTdf::fft_real(const std::vector<double>& in,
                                 std::vector<double>& out_real,
                                 std::vector<double>& out_imag) {
    int N = static_cast<int>(in.size());
    out_real.resize(N);
    out_imag.resize(N);
    
    // Use Cooley-Tukey FFT (O(N log N))
    std::vector<std::complex<double>> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = std::complex<double>(in[i], 0.0);
    }
    
    fft_cooley_tukey(data, false);
    
    for (int i = 0; i < N; ++i) {
        out_real[i] = data[i].real();
        out_imag[i] = data[i].imag();
    }
}

void ChannelSParamTdf::ifft_real(const std::vector<double>& in_real,
                                  const std::vector<double>& in_imag,
                                  std::vector<double>& out) {
    int N = static_cast<int>(in_real.size());
    out.resize(N);
    
    // Use Cooley-Tukey IFFT (O(N log N))
    std::vector<std::complex<double>> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = std::complex<double>(in_real[i], in_imag[i]);
    }
    
    fft_cooley_tukey(data, true);
    
    for (int i = 0; i < N; ++i) {
        out[i] = data[i].real();
    }
}

} // namespace serdes
