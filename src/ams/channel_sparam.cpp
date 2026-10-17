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
    , m_filter_state_n(0.0)
    , m_alpha(0.3)
    , m_use_discrete(false)
    , m_discrete_dt(0.0)
    , m_config_loaded(false)
    , m_initialized(false)
{
    // Default to simple model for backward compatibility
    m_ext_params.method = ChannelMethod::SIMPLE;
    // Initialize with SISO ports (will be resized after config load)
    in.init(1);
    out.init(1);
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
    , m_filter_state_n(0.0)
    , m_alpha(0.3)
    , m_use_discrete(false)
    , m_discrete_dt(0.0)
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
    
    // Initialize ports based on config or params.ports
    int n_in = 1;
    int n_out = 1;
    
    if (!m_port_config.active_inputs.empty()) {
        // Use port config from JSON
        n_in = m_port_config.active_inputs.size();
        n_out = m_port_config.active_outputs.size();
    } else if (m_ext_params.method == ChannelMethod::SIMPLE && m_params.ports >= 4) {
        // SIMPLE mode with differential: 2 inputs, 2 outputs
        n_in = 2;
        n_out = 2;
    }
    
    in.init(n_in);
    out.init(n_out);
    
    std::cout << "[DEBUG] ChannelSParamTdf: Ports initialized: " 
              << n_in << " inputs, " << n_out << " outputs" << std::endl;
}

ChannelSParamTdf::~ChannelSParamTdf() {
    // Cleanup if needed
}

// ============================================================================
// SystemC-AMS Lifecycle Methods
// ============================================================================

void ChannelSParamTdf::set_attributes() {
    // Set rate for all input and output ports
    for (unsigned int i = 0; i < in.size(); ++i) {
        in[i].set_rate(1);
    }
    for (unsigned int i = 0; i < out.size(); ++i) {
        out[i].set_rate(1);
    }
    // Inherit timestep from upstream modules (e.g., WaveGen)
}

void ChannelSParamTdf::initialize() {
    if (m_initialized) return;
    
    std::cout << "[DEBUG] ChannelSParamTdf: initialize() called, method=" 
              << static_cast<int>(m_ext_params.method) << std::endl;
    
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE:
            init_simple_model();
            break;
        case ChannelMethod::STATE_SPACE:
            init_state_space_model();
            break;
    }

    // T-S4: the optional discrete realization is only valid at its export timestep.
    if (m_use_discrete) {
        const double dt = get_timestep().to_seconds();
        if (m_discrete_dt <= 0.0 || std::fabs(dt - m_discrete_dt) > 1e-18) {
            std::cerr << "[Channel] discrete_state_space dt=" << m_discrete_dt
                      << " s does not match TDF timestep " << dt
                      << " s — falling back to sca_ss" << std::endl;
            m_use_discrete = false;
        } else {
            m_disc_state.resize(m_active_discrete.n_states);
            for (int i = 1; i <= m_active_discrete.n_states; ++i) m_disc_state(i) = 0.0;
            m_disc_x.assign(m_active_discrete.n_states, 0.0);
            m_disc_xnew.assign(m_active_discrete.n_states, 0.0);
            std::cout << "[Channel] using exact ZOH discrete state space: "
                      << m_active_discrete.n_states << " states, dt=" << dt << " s" << std::endl;
        }
    }

    m_initialized = true;
}

void ChannelSParamTdf::processing() {
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE: {
            // Support both SISO and differential (2-port) modes
            double attenuation_linear = std::pow(10.0, -m_params.attenuation_db / 20.0);
            
            if (in.size() == 1) {
                // SISO mode
                double x_in = in[0].read();
                m_filter_state = m_alpha * x_in + (1.0 - m_alpha) * m_filter_state;
                out[0].write(attenuation_linear * m_filter_state);
            } else {
                // Differential mode: process P and N with independent states
                double x_p = in[0].read();
                double x_n = in[1].read();
                
                m_filter_state = m_alpha * x_p + (1.0 - m_alpha) * m_filter_state;
                m_filter_state_n = m_alpha * x_n + (1.0 - m_alpha) * m_filter_state_n;
                
                out[0].write(attenuation_linear * m_filter_state);
                out[1].write(attenuation_linear * m_filter_state_n);
            }
            break;
        }
        case ChannelMethod::STATE_SPACE: {
            // MIMO processing
            process_state_space_mimo();
            break;
        }
    }
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
        
        // Get method (case-insensitive comparison)
        std::string method_str = config.value("method", "simple");
        std::cout << "[DEBUG] ChannelSParamTdf: Loading method: " << method_str << std::endl;
        
        // Convert to lowercase for comparison
        std::string method_lower = method_str;
        std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(), ::tolower);
        std::cout << "[DEBUG] ChannelSParamTdf: method_lower: " << method_lower << std::endl;
        
        if (method_lower == "state_space" || method_lower == "state-space" ||
                   method_lower == "state_space_mimo") {
            m_ext_params.method = ChannelMethod::STATE_SPACE;
            
            // Parse port configuration NOW (needed for port initialization in constructor)
            if (config.contains("full_model")) {
                const auto& fm = config["full_model"];
                int n_inputs = fm.value("n_diff_ports", 1);
                int n_outputs = fm.value("n_outputs", 1);
                
                // Parse port_config or use defaults
                if (config.contains("port_config")) {
                    const auto& pc = config["port_config"];
                    for (const auto& inp : pc["active_inputs"]) {
                        m_port_config.active_inputs.push_back(inp.get<int>());
                    }
                    for (const auto& outp : pc["active_outputs"]) {
                        m_port_config.active_outputs.push_back(outp.get<int>());
                    }
                } else {
                    // Default: use all inputs and outputs
                    for (int i = 0; i < n_inputs; ++i) {
                        m_port_config.active_inputs.push_back(i);
                    }
                    for (int i = 0; i < n_outputs; ++i) {
                        m_port_config.active_outputs.push_back(i);
                    }
                }
                std::cout << "[DEBUG] ChannelSParamTdf: Port config parsed: "
                          << m_port_config.active_inputs.size() << " inputs, "
                          << m_port_config.active_outputs.size() << " outputs" << std::endl;
            }
        } else {
            // Default to SIMPLE for any other method (including legacy "rational", "impulse")
            m_ext_params.method = ChannelMethod::SIMPLE;
        }
        
        std::cout << "[DEBUG] ChannelSParamTdf: Configuration loaded successfully (method=" 
                  << static_cast<int>(m_ext_params.method) << ")" << std::endl;
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
    // Get timestep from SystemC-AMS (inherited from upstream modules)
    double dt = get_timestep().to_seconds();
    
    // First-order IIR filter coefficient
    // Using bilinear transform approximation
    m_alpha = omega_c * dt / (1.0 + omega_c * dt);
    
    // Initialize filter states (for SISO and differential modes)
    m_filter_state = 0.0;
    m_filter_state_n = 0.0;
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
        
        // Parse new format: full_model + port_config
        if (config.contains("full_model")) {
            // New MIMO format
            const auto& fm = config["full_model"];
            
            m_full_model.n_diff_ports = fm.value("n_diff_ports", 1);
            m_full_model.n_outputs = fm.value("n_outputs", 1);
            m_full_model.n_states = fm.value("n_states", 0);
            
            // Parse port_pairs
            if (fm.contains("port_pairs")) {
                for (const auto& pp : fm["port_pairs"]) {
                    m_full_model.port_pairs.push_back({pp[0].get<int>(), pp[1].get<int>()});
                }
            }
            
            // Parse delays
            if (fm.contains("delay_s")) {
                for (const auto& d : fm["delay_s"]) {
                    m_full_model.delays.push_back(d.get<double>());
                }
            }
            
            // Parse state_space matrices
            if (!fm.contains("state_space")) {
                std::cerr << "ChannelSParamTdf: No state_space in full_model" << std::endl;
                m_ext_params.method = ChannelMethod::SIMPLE;
                init_simple_model();
                return;
            }
            
            const auto& ss = fm["state_space"];
            int n_states = m_full_model.n_states;
            int n_inputs = m_full_model.n_diff_ports;
            int n_outputs = m_full_model.n_outputs;
            
            m_full_model.state_space.n_states = n_states;
            m_full_model.state_space.n_inputs = n_inputs;
            m_full_model.state_space.n_outputs = n_outputs;
            
            // Resize matrices
            m_full_model.state_space.A.resize(n_states, n_states);
            m_full_model.state_space.B.resize(n_states, n_inputs);
            m_full_model.state_space.C.resize(n_outputs, n_states);
            m_full_model.state_space.D.resize(n_outputs, n_inputs);
            m_full_model.state_space.E.resize(n_outputs, n_inputs);
            
            // Fill A matrix
            const auto& A_json = ss["A"];
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    m_full_model.state_space.A(i+1, j+1) = A_json[i][j].get<double>();
                }
            }
            
            // Fill B matrix (n_states x n_inputs)
            const auto& B_json = ss["B"];
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_inputs; ++j) {
                    m_full_model.state_space.B(i+1, j+1) = B_json[i][j].get<double>();
                }
            }
            
            // Fill C matrix (n_outputs x n_states)
            const auto& C_json = ss["C"];
            for (int i = 0; i < n_outputs; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    m_full_model.state_space.C(i+1, j+1) = C_json[i][j].get<double>();
                }
            }
            
            // Fill D matrix (n_outputs x n_inputs)
            const auto& D_json = ss["D"];
            for (int i = 0; i < n_outputs; ++i) {
                for (int j = 0; j < n_inputs; ++j) {
                    m_full_model.state_space.D(i+1, j+1) = D_json[i][j].get<double>();
                }
            }
            
            // Fill E matrix (n_outputs x n_inputs)
            if (ss.contains("E")) {
                const auto& E_json = ss["E"];
                for (int i = 0; i < n_outputs; ++i) {
                    for (int j = 0; j < n_inputs; ++j) {
                        m_full_model.state_space.E(i+1, j+1) = E_json[i][j].get<double>();
                    }
                }
            }
            
            // T-S4: optional exact zero-order-hold discrete realization.
            // It has the same MIMO dimensions as the continuous full model.
            if (fm.contains("discrete_state_space")) {
                const auto& dss = fm["discrete_state_space"];
                bool complete = dss.contains("A") && dss.contains("B")
                             && dss.contains("C") && dss.contains("D");
                if (!complete) {
                    std::cerr << "[Channel] discrete_state_space incomplete — ignored" << std::endl;
                } else {
                    auto& ds = m_full_model.discrete_state_space;
                    ds.n_states = n_states;
                    ds.n_inputs = n_inputs;
                    ds.n_outputs = n_outputs;
                    ds.A.resize(n_states, n_states);
                    ds.B.resize(n_states, n_inputs);
                    ds.C.resize(n_outputs, n_states);
                    ds.D.resize(n_outputs, n_inputs);
                    ds.E.resize(n_outputs, n_inputs);
                    auto fill = [](const json& src, sca_util::sca_matrix<double>& dst,
                                   int rows, int cols) {
                        for (int i = 0; i < rows; ++i)
                            for (int j = 0; j < cols; ++j)
                                dst(i + 1, j + 1) = src[i][j].get<double>();
                    };
                    fill(dss["A"], ds.A, n_states, n_states);
                    fill(dss["B"], ds.B, n_states, n_inputs);
                    fill(dss["C"], ds.C, n_outputs, n_states);
                    fill(dss["D"], ds.D, n_outputs, n_inputs);
                    if (dss.contains("E")) fill(dss["E"], ds.E, n_outputs, n_inputs);
                    m_full_model.discrete_dt = dss.value("dt", 0.0);
                    m_full_model.has_discrete = true;
                    std::cout << "[DEBUG]   Discrete ZOH state space parsed, dt="
                              << m_full_model.discrete_dt << " s" << std::endl;
                }
            }

            // Port config already parsed in parse_json_config(), just verify
            if (m_port_config.active_inputs.empty()) {
                // Fallback: use all inputs and outputs
                for (int i = 0; i < n_inputs; ++i) {
                    m_port_config.active_inputs.push_back(i);
                }
                for (int i = 0; i < n_outputs; ++i) {
                    m_port_config.active_outputs.push_back(i);
                }
            }
            
            // Extract active matrices
            extract_active_matrices();
            
            std::cout << "[DEBUG] ChannelSParamTdf: MIMO State-space model initialized" << std::endl;
            std::cout << "[DEBUG]   Full model: " << n_states << " states, " 
                      << n_inputs << " inputs, " << n_outputs << " outputs" << std::endl;
            std::cout << "[DEBUG]   Active: " << m_port_config.active_inputs.size() << " inputs, "
                      << m_port_config.active_outputs.size() << " outputs" << std::endl;
            
        } else if (config.contains("state_space")) {
            // Legacy format (backward compatibility)
            const auto& ss = config["state_space"];
            
            const auto& A_json = ss["A"];
            int n_states = A_json.size();
            const auto& C_json = ss["C"];
            int n_outputs = C_json.size();
            
            m_full_model.n_states = n_states;
            m_full_model.n_diff_ports = 1;
            m_full_model.n_outputs = n_outputs;
            
            m_full_model.state_space.n_states = n_states;
            m_full_model.state_space.n_inputs = 1;
            m_full_model.state_space.n_outputs = n_outputs;
            
            m_full_model.state_space.A.resize(n_states, n_states);
            m_full_model.state_space.B.resize(n_states, 1);
            m_full_model.state_space.C.resize(n_outputs, n_states);
            m_full_model.state_space.D.resize(n_outputs, 1);
            m_full_model.state_space.E.resize(n_outputs, 1);
            
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    m_full_model.state_space.A(i+1, j+1) = A_json[i][j].get<double>();
                }
            }
            
            const auto& B_json = ss["B"];
            for (int i = 0; i < n_states; ++i) {
                m_full_model.state_space.B(i+1, 1) = B_json[i][0].get<double>();
            }
            
            for (int i = 0; i < n_outputs; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    m_full_model.state_space.C(i+1, j+1) = C_json[i][j].get<double>();
                }
            }
            
            const auto& D_json = ss["D"];
            for (int i = 0; i < n_outputs; ++i) {
                m_full_model.state_space.D(i+1, 1) = D_json[i][0].get<double>();
            }
            
            if (ss.contains("E")) {
                const auto& E_json = ss["E"];
                for (int i = 0; i < n_outputs; ++i) {
                    m_full_model.state_space.E(i+1, 1) = E_json[i][0].get<double>();
                }
            }
            
            // Default SISO config
            m_port_config.active_inputs = {0};
            m_port_config.active_outputs = {0};
            
            extract_active_matrices();
            
            std::cout << "[DEBUG] ChannelSParamTdf: Legacy state-space model loaded" << std::endl;
        } else {
            std::cerr << "ChannelSParamTdf: No state_space or full_model in config" << std::endl;
            m_ext_params.method = ChannelMethod::SIMPLE;
            init_simple_model();
            return;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ChannelSParamTdf: Error initializing state-space model: " << e.what() << std::endl;
        m_ext_params.method = ChannelMethod::SIMPLE;
        init_simple_model();
    }
}

void ChannelSParamTdf::extract_active_matrices() {
    int n_states = m_full_model.n_states;
    int n_active_in = m_port_config.active_inputs.size();
    int n_active_out = m_port_config.active_outputs.size();
    
    m_active_ss.n_states = n_states;
    m_active_ss.n_inputs = n_active_in;
    m_active_ss.n_outputs = n_active_out;
    
    // Resize active matrices
    m_active_ss.A.resize(n_states, n_states);
    m_active_ss.B.resize(n_states, n_active_in);
    m_active_ss.C.resize(n_active_out, n_states);
    m_active_ss.D.resize(n_active_out, n_active_in);
    m_active_ss.E.resize(n_active_out, n_active_in);
    
    // A is shared (copy directly)
    m_active_ss.A = m_full_model.state_space.A;
    
    // Extract B columns for active inputs
    for (int j = 0; j < n_active_in; ++j) {
        int src_col = m_port_config.active_inputs[j] + 1;  // 1-based for sca_matrix
        for (int i = 1; i <= n_states; ++i) {
            m_active_ss.B(i, j + 1) = m_full_model.state_space.B(i, src_col);
        }
    }
    
    // Extract C rows for active outputs
    for (int i = 0; i < n_active_out; ++i) {
        int src_row = m_port_config.active_outputs[i] + 1;  // 1-based
        for (int j = 1; j <= n_states; ++j) {
            m_active_ss.C(i + 1, j) = m_full_model.state_space.C(src_row, j);
        }
    }
    
    // Extract D submatrix
    for (int i = 0; i < n_active_out; ++i) {
        int src_row = m_port_config.active_outputs[i] + 1;
        for (int j = 0; j < n_active_in; ++j) {
            int src_col = m_port_config.active_inputs[j] + 1;
            m_active_ss.D(i + 1, j + 1) = m_full_model.state_space.D(src_row, src_col);
        }
    }
    
    // Extract E submatrix
    for (int i = 0; i < n_active_out; ++i) {
        int src_row = m_port_config.active_outputs[i] + 1;
        for (int j = 0; j < n_active_in; ++j) {
            int src_col = m_port_config.active_inputs[j] + 1;
            m_active_ss.E(i + 1, j + 1) = m_full_model.state_space.E(src_row, src_col);
        }
    }
    
    // T-S4: extract the active submatrices of the optional discrete realization.
    if (m_full_model.has_discrete) {
        const auto& ds = m_full_model.discrete_state_space;
        m_active_discrete.n_states = n_states;
        m_active_discrete.n_inputs = n_active_in;
        m_active_discrete.n_outputs = n_active_out;
        m_active_discrete.A.resize(n_states, n_states);
        m_active_discrete.B.resize(n_states, n_active_in);
        m_active_discrete.C.resize(n_active_out, n_states);
        m_active_discrete.D.resize(n_active_out, n_active_in);
        m_active_discrete.E.resize(n_active_out, n_active_in);
        m_active_discrete.A = ds.A;
        for (int j = 0; j < n_active_in; ++j) {
            int src_col = m_port_config.active_inputs[j] + 1;
            for (int i = 1; i <= n_states; ++i)
                m_active_discrete.B(i, j + 1) = ds.B(i, src_col);
        }
        for (int i = 0; i < n_active_out; ++i) {
            int src_row = m_port_config.active_outputs[i] + 1;
            for (int j = 1; j <= n_states; ++j)
                m_active_discrete.C(i + 1, j) = ds.C(src_row, j);
            for (int j = 0; j < n_active_in; ++j) {
                int src_col = m_port_config.active_inputs[j] + 1;
                m_active_discrete.D(i + 1, j + 1) = ds.D(src_row, src_col);
                m_active_discrete.E(i + 1, j + 1) = ds.E(src_row, src_col);
            }
        }
        m_disc_state.resize(n_states);
        for (int i = 1; i <= n_states; ++i) m_disc_state(i) = 0.0;
        m_use_discrete = true;
        m_discrete_dt = m_full_model.discrete_dt;

        // Flat row-major copies for the hot path.
        auto flatten = [](const sca_util::sca_matrix<double>& M, int rows, int cols) {
            std::vector<double> v(static_cast<size_t>(rows) * cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    v[static_cast<size_t>(i) * cols + j] = M(i + 1, j + 1);
            return v;
        };
        m_disc_A = flatten(m_active_discrete.A, n_states, n_states);
        m_disc_B = flatten(m_active_discrete.B, n_states, n_active_in);
        m_disc_C = flatten(m_active_discrete.C, n_active_out, n_states);
        m_disc_D = flatten(m_active_discrete.D, n_active_out, n_active_in);
        m_disc_E = flatten(m_active_discrete.E, n_active_out, n_active_in);
        m_disc_x.assign(n_states, 0.0);
        m_disc_xnew.assign(n_states, 0.0);
    }

    // Initialize state vector
    m_ss_state.resize(n_states);
    for (int i = 1; i <= n_states; ++i) {
        m_ss_state(i) = 0.0;
    }
    
    std::cout << "[DEBUG] ChannelSParamTdf: Active matrices extracted" << std::endl;
    std::cout << "[DEBUG]   B: " << n_states << "x" << n_active_in << std::endl;
    std::cout << "[DEBUG]   C: " << n_active_out << "x" << n_states << std::endl;

    // T-S3 A2 fix: initialize derivative history for the E-term feedthrough
    m_u_prev.assign(static_cast<size_t>(n_active_in), 0.0);
    m_u_prev_valid = false;
}

void ChannelSParamTdf::process_state_space_mimo() {
    int n_in = m_port_config.active_inputs.size();
    int n_out = m_port_config.active_outputs.size();
    
    // Read inputs into vector
    sca_util::sca_vector<double> u(n_in);
    for (int i = 0; i < n_in; ++i) {
        u(i + 1) = in[i].read();
    }
    
    // State-space computation: exact ZOH recurrence (T-S4) or sca_ss.
    // y = C*x + D*u + E*du/dt
    sca_util::sca_vector<double> y(n_out);
    if (m_use_discrete) {
        const int ns = m_active_discrete.n_states;
        const int ni = n_in;
        for (int i = 0; i < n_out; ++i) {
            const double* Cr = &m_disc_C[static_cast<size_t>(i) * ns];
            const double* Dr = &m_disc_D[static_cast<size_t>(i) * ni];
            double acc = 0.0;
            for (int j = 0; j < ns; ++j) acc += Cr[j] * m_disc_x[j];
            for (int j = 0; j < ni; ++j) acc += Dr[j] * u(j + 1);
            y(i + 1) = acc;
        }
        for (int i = 0; i < ns; ++i) {
            const double* Ar = &m_disc_A[static_cast<size_t>(i) * ns];
            const double* Br = &m_disc_B[static_cast<size_t>(i) * ni];
            double acc = 0.0;
            for (int j = 0; j < ns; ++j) acc += Ar[j] * m_disc_x[j];
            for (int j = 0; j < ni; ++j) acc += Br[j] * u(j + 1);
            m_disc_xnew[i] = acc;
        }
        m_disc_x.swap(m_disc_xnew);
    } else {
        y = m_ss_filter(
            m_active_ss.A, m_active_ss.B, m_active_ss.C,
            m_active_ss.D, m_ss_state, u, get_timestep());
    }

    // T-S3 A2 fix: E-term feedthrough y += E * du/dt (previously parsed but ignored).
    // Use a first-order finite difference of the sampled input; skip the first call
    // to avoid a startup impulse.
    const double dt = get_timestep().to_seconds();
    if (dt > 0.0 && m_u_prev.size() == static_cast<size_t>(n_in)) {
        if (!m_u_prev_valid) {
            for (int j = 0; j < n_in; ++j) m_u_prev[j] = u(j + 1);
            m_u_prev_valid = true;
        } else {
            for (int i = 0; i < n_out; ++i) {
                double acc = 0.0;
                for (int j = 0; j < n_in; ++j) {
                    const double e = m_use_discrete
                        ? m_disc_E[static_cast<size_t>(i) * n_in + j]
                        : m_active_ss.E(i + 1, j + 1);
                    acc += e * (u(j + 1) - m_u_prev[j]) / dt;
                }
                y(i + 1) += acc;
            }
            for (int j = 0; j < n_in; ++j) m_u_prev[j] = u(j + 1);
        }
    }

    // Write outputs
    for (int i = 0; i < n_out; ++i) {
        out[i].write(y(i + 1));
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

double ChannelSParamTdf::get_dc_gain() const {
    switch (m_ext_params.method) {
        case ChannelMethod::SIMPLE:
            return std::pow(10.0, -m_params.attenuation_db / 20.0);
        case ChannelMethod::STATE_SPACE:
            // DC gain for state-space: D - C * inv(A) * B
            // Use LU decomposition for better numerical stability
            if (m_active_ss.n_states > 0 && m_active_ss.n_outputs > 0 
                && m_active_ss.n_inputs > 0) {
                try {
                    int n = m_active_ss.n_states;
                    int m = m_active_ss.n_inputs;   // Number of inputs
                    
                    // Copy A matrix for LU decomposition
                    std::vector<std::vector<double>> A_copy(n, std::vector<double>(n));
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            A_copy[i][j] = m_active_ss.A(i + 1, j + 1);
                        }
                    }
                    
                    // Copy B matrix (n x m)
                    std::vector<std::vector<double>> X(n, std::vector<double>(m));
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < m; ++j) {
                            X[i][j] = m_active_ss.B(i + 1, j + 1);
                        }
                    }
                    
                    // LU decomposition with partial pivoting
                    std::vector<int> pivot(n);
                    if (!lu_decompose(A_copy, pivot)) {
                        return m_active_ss.D(1, 1);  // Singular matrix
                    }
                    
                    // Solve A * X = B for X = A^-1 * B
                    if (!lu_solve(A_copy, pivot, X)) {
                        return m_active_ss.D(1, 1);  // Solve failed
                    }
                    
                    // Compute DC gain = D - C * X for first input/output
                    // For MIMO, return the (0,0) element as representative
                    double CX = 0.0;
                    for (int k = 0; k < n; ++k) {
                        CX += m_active_ss.C(1, k + 1) * X[k][0];
                    }
                    
                    return m_active_ss.D(1, 1) - CX;
                } catch (...) {
                    return m_active_ss.D(1, 1);  // Return D on error
                }
            }
            return 1.0;
        default:
            return 1.0;
    }
}

// ============================================================================
// LU Decomposition Helpers
// ============================================================================

bool ChannelSParamTdf::lu_decompose(std::vector<std::vector<double>>& A, 
                                     std::vector<int>& pivot) const {
    int n = static_cast<int>(A.size());
    if (n == 0) return false;
    
    pivot.resize(n);
    for (int i = 0; i < n; ++i) {
        pivot[i] = i;
    }
    
    const double TINY = 1e-20;
    
    for (int k = 0; k < n; ++k) {
        // Find pivot row
        double max_val = std::abs(A[k][k]);
        int max_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                max_row = i;
            }
        }
        
        // Check for singular matrix
        if (max_val < EPSILON) {
            A[k][k] = TINY;  // Prevent division by zero
        }
        
        // Swap rows if needed
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(pivot[k], pivot[max_row]);
        }
        
        // Compute multipliers and eliminate
        for (int i = k + 1; i < n; ++i) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
    
    return true;
}

bool ChannelSParamTdf::lu_solve(const std::vector<std::vector<double>>& LU, 
                                 const std::vector<int>& pivot,
                                 std::vector<std::vector<double>>& B) const {
    int n = static_cast<int>(LU.size());
    int m = static_cast<int>(B[0].size());
    
    if (n == 0 || m == 0) return false;
    
    // Forward substitution: solve L * Y = P * B
    for (int k = 0; k < n; ++k) {
        // Apply row permutation
        if (pivot[k] != k) {
            for (int j = 0; j < m; ++j) {
                std::swap(B[k][j], B[pivot[k]][j]);
            }
        }
        
        // Eliminate
        for (int i = k + 1; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                B[i][j] -= LU[i][k] * B[k][j];
            }
        }
    }
    
    // Back substitution: solve U * X = Y
    for (int k = n - 1; k >= 0; --k) {
        for (int j = 0; j < m; ++j) {
            B[k][j] /= LU[k][k];
        }
        
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                B[i][j] -= LU[i][k] * B[k][j];
            }
        }
    }
    
    return true;
}

} // namespace serdes
