/**
 * @file rx_link_tb.cpp
 * @brief Transient testbench for RX Link using RxTopModule
 * 
 * This testbench performs complete RX chain simulation:
 * WaveGen → Channel → S2D → RxTopModule → data_out
 * 
 * Key features:
 * - Uses RxTopModule for complete RX signal chain
 * - Multi-point signal recording for eye diagram analysis
 * - CSV export with timestamps
 * 
 * Test scenarios:
 * 0. BASIC_LINK     - Basic link establishment
 * 1. CTLE_SWEEP     - CTLE parameter sweep
 * 2. CDR_LOCK_TEST  - CDR locking behavior
 * 3. EYE_DIAGRAM    - Eye diagram data collection
 * 
 * Usage:
 *   ./rx_link_tb [scenario]
 * 
 * Examples:
 *   ./rx_link_tb basic
 *   ./rx_link_tb eye
 */

#include <systemc-ams>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <cstring>

// Project headers
#include "ams/wave_generation.h"
#include "ams/channel_sparam.h"
#include "ams/rx_top.h"
#include "common/parameters.h"

// Local helpers
#include "rx_link_helpers.h"

using namespace serdes;

// ============================================================================
// Test Scenarios
// ============================================================================

enum TestScenario {
    BASIC_LINK = 0,
    CTLE_SWEEP,
    CDR_LOCK_TEST,
    EYE_DIAGRAM
};

std::map<std::string, TestScenario> scenario_map = {
    {"basic", BASIC_LINK},
    {"ctle_sweep", CTLE_SWEEP},
    {"cdr_lock", CDR_LOCK_TEST},
    {"eye", EYE_DIAGRAM},
    {"0", BASIC_LINK},
    {"1", CTLE_SWEEP},
    {"2", CDR_LOCK_TEST},
    {"3", EYE_DIAGRAM}
};

std::string get_scenario_name(TestScenario scenario) {
    switch (scenario) {
        case BASIC_LINK: return "basic";
        case CTLE_SWEEP: return "ctle_sweep";
        case CDR_LOCK_TEST: return "cdr_lock";
        case EYE_DIAGRAM: return "eye";
        default: return "unknown";
    }
}

// ============================================================================
// RX Link Testbench (Refactored with RxTopModule)
// ============================================================================

SC_MODULE(RxLinkTestbench) {
    // ========================================================================
    // Sub-modules
    // ========================================================================
    
    // Signal sources
    WaveGenerationTdf* wavegen;
    ConstVddSource* vdd_src;
    
    // Channel
    ChannelSParamTdf* channel;
    
    // Single-to-Differential converter (Channel output is single-ended)
    SingleToDiffConverter* s2d;
    
    // RX Top Module (encapsulates CTLE, VGA, DFE, Sampler, CDR)
    RxTopModule* rx_top;
    
    // Signal recorder
    MultiPointSignalRecorder* recorder;
    
    // ========================================================================
    // Signals
    // ========================================================================
    
    // WaveGen → Channel
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_channel_out;
    
    // Channel → S2D → RxTop (differential)
    sca_tdf::sca_signal<double> sig_ch_out_p;
    sca_tdf::sca_signal<double> sig_ch_out_n;
    
    // VDD
    sca_tdf::sca_signal<double> sig_vdd;
    
    // RxTop output
    sca_tdf::sca_signal<double> sig_data_out;
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    WaveGenParams m_wave_params;
    ChannelParams m_channel_params;
    RxParams m_rx_params;
    AdaptionParams m_adaption_params;
    
    double m_sim_duration;
    double m_sample_rate;
    double m_ui;
    std::string m_output_prefix;
    
    // ========================================================================
    // Constructor
    // ========================================================================
    
    SC_CTOR(RxLinkTestbench)
        : sig_wavegen_out("sig_wavegen_out")
        , sig_channel_out("sig_channel_out")
        , sig_ch_out_p("sig_ch_out_p")
        , sig_ch_out_n("sig_ch_out_n")
        , sig_vdd("sig_vdd")
        , sig_data_out("sig_data_out")
        , m_sim_duration(2000e-9)
        , m_sample_rate(100e9)
        , m_ui(100e-12)
        , m_output_prefix("rx_link")
    {
    }
    
    // ========================================================================
    // Configuration Methods
    // ========================================================================
    
    void configure(TestScenario scenario) {
        // Default parameters
        configure_defaults();
        
        std::string scenario_name = get_scenario_name(scenario);
        m_output_prefix = "rx_link_" + scenario_name;
        
        switch (scenario) {
            case BASIC_LINK:
                configure_basic();
                break;
            case CTLE_SWEEP:
                configure_ctle_sweep();
                break;
            case CDR_LOCK_TEST:
                configure_cdr_lock();
                break;
            case EYE_DIAGRAM:
                configure_eye_diagram();
                break;
            default:
                configure_basic();
                break;
        }
    }
    
    void configure_defaults() {
        // Wave generation
        m_wave_params = WaveGenParams();
        m_wave_params.type = PRBSType::PRBS31;
        m_wave_params.single_pulse = 0.0;
        
        // Channel - simple model
        m_channel_params = ChannelParams();
        m_channel_params.attenuation_db = 6.0;
        m_channel_params.bandwidth_hz = 15e9;
        
        // RX parameters (using new unified RxParams with CDR)
        m_rx_params = RxParams();
        
        // CTLE
        m_rx_params.ctle.zeros = {2e9};
        m_rx_params.ctle.poles = {30e9};
        m_rx_params.ctle.dc_gain = 1.5;
        m_rx_params.ctle.vcm_out = 0.0;
        
        // VGA
        m_rx_params.vga.zeros = {1e9};
        m_rx_params.vga.poles = {20e9};
        m_rx_params.vga.dc_gain = 2.0;
        m_rx_params.vga.vcm_out = 0.0;
        
        // DFE Summer (差分架构)
        m_rx_params.dfe_summer.tap_coeffs = {-0.05, -0.02, 0.01};
        m_rx_params.dfe_summer.ui = 100e-12;
        m_rx_params.dfe_summer.vcm_out = 0.0;
        m_rx_params.dfe_summer.vtap = 1.0;
        m_rx_params.dfe_summer.map_mode = "pm1";
        m_rx_params.dfe_summer.enable = true;
        
        // Sampler (phase-driven mode is forced by RxTopModule)
        m_rx_params.sampler.phase_source = "phase";
        m_rx_params.sampler.threshold = 0.0;
        m_rx_params.sampler.hysteresis = 0.01;  // 需满足 hysteresis < resolution
        m_rx_params.sampler.resolution = 0.02;
        
        // CDR
        m_rx_params.cdr.pi.kp = 0.01;
        m_rx_params.cdr.pi.ki = 1e-4;
        m_rx_params.cdr.pi.edge_threshold = 0.5;
        m_rx_params.cdr.pai.resolution = 1e-12;
        m_rx_params.cdr.pai.range = 5e-11;
        m_rx_params.cdr.ui = 100e-12;
        
        // Timing
        m_sample_rate = 100e9;  // 100 GS/s
        m_ui = 100e-12;         // 100 ps (10 Gbps)
        m_sim_duration = 2000e-9;  // 2 us
        
        // Adaption parameters (disabled for testbench)
        m_adaption_params = AdaptionParams();
        m_adaption_params.Fs = 80e9;
        m_adaption_params.UI = 100e-12;
        m_adaption_params.seed = 12345;
        m_adaption_params.update_mode = "multi-rate";
        m_adaption_params.fast_update_period = 2.5e-10;
        m_adaption_params.slow_update_period = 2.5e-7;
        m_adaption_params.agc.enabled = false;
        m_adaption_params.agc.initial_gain = 2.0;
        m_adaption_params.dfe.enabled = false;
        m_adaption_params.dfe.num_taps = 3;
        m_adaption_params.dfe.algorithm = "sign-lms";
        m_adaption_params.dfe.initial_taps = {-0.05, -0.02, 0.01};
        m_adaption_params.vref_adapt.enabled = false;
        m_adaption_params.cdr_pi.enabled = false;
        m_adaption_params.safety.freeze_on_error = false;
        m_adaption_params.safety.rollback_enable = false;
    }
    
    void configure_basic() {
        std::cout << "Configuring BASIC_LINK test..." << std::endl;
        m_sim_duration = 2000e-9;  // 2 us
    }
    
    void configure_ctle_sweep() {
        std::cout << "Configuring CTLE_SWEEP test..." << std::endl;
        // Higher CTLE gain
        m_rx_params.ctle.dc_gain = 2.0;
        m_rx_params.ctle.zeros = {3e9};
        m_sim_duration = 2000e-9;
    }
    
    void configure_cdr_lock() {
        std::cout << "Configuring CDR_LOCK_TEST..." << std::endl;
        // Longer simulation for CDR to lock
        m_sim_duration = 5000e-9;  // 5 us
        // Stronger CDR gains
        m_rx_params.cdr.pi.kp = 0.02;
        m_rx_params.cdr.pi.ki = 2e-4;
    }
    
    void configure_eye_diagram() {
        std::cout << "Configuring EYE_DIAGRAM test..." << std::endl;
        // Collect enough data for eye diagram
        m_sim_duration = 10000e-9;  // 10 us for good statistics
    }
    
    // ========================================================================
    // Build Method
    // ========================================================================
    
    void build() {
        std::cout << "Building RX link testbench..." << std::endl;
        
        // Create WaveGen
        wavegen = new WaveGenerationTdf("wavegen", m_wave_params, m_sample_rate, m_ui, 12345);
        
        // Create Channel
        channel = new ChannelSParamTdf("channel", m_channel_params);
        
        // Create Single-to-Diff converter (Channel output → RxTop input)
        s2d = new SingleToDiffConverter("s2d", 0.0);
        
        // Create VDD source
        vdd_src = new ConstVddSource("vdd_src", 1.0);
        
        // Create RX Top Module (encapsulates CTLE, VGA, DFE Summer, Sampler, CDR)
        rx_top = new RxTopModule("rx_top", m_rx_params, m_adaption_params);
        
        // Create Recorder (simplified - connects to RxTop debug signals)
        recorder = new MultiPointSignalRecorder("recorder");
        
        // ====================================================================
        // Connect Signals
        // ====================================================================
        
        // WaveGen → Channel
        wavegen->out(sig_wavegen_out);
        channel->in[0](sig_wavegen_out);
        channel->out[0](sig_channel_out);
        
        // Channel → S2D → RxTop
        s2d->in(sig_channel_out);
        s2d->out_p(sig_ch_out_p);
        s2d->out_n(sig_ch_out_n);
        
        // VDD source
        vdd_src->out(sig_vdd);
        
        // RxTop connections
        rx_top->in_p(sig_ch_out_p);
        rx_top->in_n(sig_ch_out_n);
        rx_top->vdd(sig_vdd);
        rx_top->data_out(sig_data_out);
        
        // Recorder connections - using RxTop debug signals
        recorder->ch_out_p(sig_ch_out_p);
        recorder->ch_out_n(sig_ch_out_n);
        recorder->ctle_out_p(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_ctle_out_p_signal()));
        recorder->ctle_out_n(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_ctle_out_n_signal()));
        recorder->vga_out_p(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_vga_out_p_signal()));
        recorder->vga_out_n(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_vga_out_n_signal()));
        recorder->dfe_out_p(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_dfe_out_p_signal()));
        recorder->dfe_out_n(const_cast<sca_tdf::sca_signal<double>&>(rx_top->get_dfe_out_n_signal()));
        recorder->sampler_out(sig_data_out);
        // Note: CDR phase is accessed via debug interface, not signal connection
        
        std::cout << "RX link testbench built successfully." << std::endl;
    }
    
    // ========================================================================
    // Run Method
    // ========================================================================
    
    void run() {
        std::cout << "Running RX link simulation for " << m_sim_duration * 1e9 << " ns..." << std::endl;
        std::cout << "  Sample rate: " << m_sample_rate / 1e9 << " GS/s" << std::endl;
        std::cout << "  Data rate: " << 1.0 / m_ui / 1e9 << " Gbps" << std::endl;
        std::cout << "  UI: " << m_ui * 1e12 << " ps" << std::endl;
        
        sc_core::sc_start(m_sim_duration, sc_core::SC_SEC);
        
        std::cout << "Simulation completed." << std::endl;
    }
    
    // ========================================================================
    // Save Results
    // ========================================================================
    
    void save_results() {
        // Save waveform data
        std::string waveform_file = m_output_prefix + "_waveform.csv";
        recorder->save_waveform_csv(waveform_file);
        
        // Save eye diagram data
        std::string eye_file = m_output_prefix + "_eye.csv";
        recorder->save_eye_data_csv(eye_file, m_ui);
        
        // Save configuration
        std::string config_file = m_output_prefix + "_config.json";
        save_config(config_file);
    }
    
    void save_config(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open config file " << filename << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(6);
        
        file << "{\n";
        
        // Wave parameters
        file << "  \"wave\": {\n";
        file << "    \"type\": \"PRBS31\"\n";
        file << "  },\n";
        
        // Channel parameters
        file << "  \"channel\": {\n";
        file << "    \"attenuation_db\": " << m_channel_params.attenuation_db << ",\n";
        file << "    \"bandwidth_hz\": " << m_channel_params.bandwidth_hz << "\n";
        file << "  },\n";
        
        // RX parameters
        file << "  \"rx\": {\n";
        file << "    \"ctle\": {\n";
        file << "      \"dc_gain\": " << m_rx_params.ctle.dc_gain << "\n";
        file << "    },\n";
        file << "    \"vga\": {\n";
        file << "      \"dc_gain\": " << m_rx_params.vga.dc_gain << "\n";
        file << "    },\n";
        file << "    \"dfe_summer\": {\n";
        file << "      \"tap_coeffs\": [";
        for (size_t i = 0; i < m_rx_params.dfe_summer.tap_coeffs.size(); ++i) {
            file << m_rx_params.dfe_summer.tap_coeffs[i];
            if (i < m_rx_params.dfe_summer.tap_coeffs.size() - 1) file << ", ";
        }
        file << "],\n";
        file << "      \"enable\": " << (m_rx_params.dfe_summer.enable ? "true" : "false") << "\n";
        file << "    },\n";
        file << "    \"cdr\": {\n";
        file << "      \"kp\": " << m_rx_params.cdr.pi.kp << ",\n";
        file << "      \"ki\": " << m_rx_params.cdr.pi.ki << ",\n";
        file << "      \"ui\": " << m_rx_params.cdr.ui << "\n";
        file << "    }\n";
        file << "  },\n";
        
        // Simulation parameters
        file << "  \"simulation\": {\n";
        file << "    \"duration_s\": " << m_sim_duration << ",\n";
        file << "    \"sample_rate_hz\": " << m_sample_rate << ",\n";
        file << "    \"ui_s\": " << m_ui << "\n";
        file << "  }\n";
        
        file << "}\n";
        
        file.close();
        std::cout << "Saved configuration to " << filename << std::endl;
    }
    
    // ========================================================================
    // Print Summary
    // ========================================================================
    
    void print_summary() {
        recorder->print_summary();
        
        // Print CDR status (via RxTop debug interface)
        std::cout << "\nCDR Status:" << std::endl;
        std::cout << "  Final phase: " << rx_top->get_cdr_phase() * 1e12 << " ps" << std::endl;
        std::cout << "  Integral state: " << rx_top->get_cdr_integral_state() << std::endl;
    }
};

// ============================================================================
// Main Function
// ============================================================================

int sc_main(int argc, char* argv[]) {
    // Disable SystemC deprecation warnings
    sc_core::sc_report_handler::set_actions(
        "/IEEE_Std_1666/deprecated",
        sc_core::SC_DO_NOTHING
    );
    
    // Parse command line
    TestScenario scenario = BASIC_LINK;
    
    if (argc > 1) {
        std::string arg = argv[1];
        auto it = scenario_map.find(arg);
        if (it != scenario_map.end()) {
            scenario = it->second;
        } else {
            std::cout << "Unknown scenario: " << arg << std::endl;
            std::cout << "Available scenarios: basic, ctle_sweep, cdr_lock, eye" << std::endl;
            return 1;
        }
    }
    
    std::cout << "=== RX Link Transient Testbench ===" << std::endl;
    std::cout << "Scenario: " << get_scenario_name(scenario) << std::endl;
    
    // Create and run testbench
    RxLinkTestbench tb("tb");
    tb.configure(scenario);
    tb.build();
    tb.run();
    tb.save_results();
    tb.print_summary();
    
    sc_core::sc_stop();
    
    std::cout << "\nTestbench completed successfully." << std::endl;
    return 0;
}
