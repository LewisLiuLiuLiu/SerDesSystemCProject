/**
 * @file tx_driver_tran_tb.cpp
 * @brief Transient testbench for TX Driver module
 * 
 * Test scenarios:
 * 0. BASIC_FUNCTION    - Basic step response
 * 1. BANDWIDTH_TEST    - Sine sweep for bandwidth measurement
 * 2. SATURATION_TEST   - Saturation curve characterization
 * 3. PSRR_TEST         - Power supply rejection ratio test
 * 4. IMBALANCE_TEST    - Gain mismatch and skew test
 * 5. SLEW_RATE_TEST    - Slew rate limiting test
 * 6. PRBS_EYE_TEST     - PRBS eye diagram generation
 * 
 * Usage:
 *   ./tx_driver_tran_tb [scenario]
 * 
 * Examples:
 *   ./tx_driver_tran_tb basic
 *   ./tx_driver_tran_tb bandwidth
 *   ./tx_driver_tran_tb psrr
 */

#include <systemc-ams>
#include "ams/tx_driver.h"
#include "tx_driver_helpers.h"
#include <iostream>
#include <string>
#include <map>
#include <cstring>

using namespace serdes;
using namespace serdes::tb;

// ============================================================================
// Test Scenarios
// ============================================================================

enum TestScenario {
    BASIC_FUNCTION = 0,
    BANDWIDTH_TEST,
    SATURATION_TEST,
    PSRR_TEST,
    IMBALANCE_TEST,
    SLEW_RATE_TEST,
    PRBS_EYE_TEST
};

std::map<std::string, TestScenario> scenario_map = {
    {"basic", BASIC_FUNCTION},
    {"bandwidth", BANDWIDTH_TEST},
    {"saturation", SATURATION_TEST},
    {"psrr", PSRR_TEST},
    {"imbalance", IMBALANCE_TEST},
    {"slew", SLEW_RATE_TEST},
    {"prbs", PRBS_EYE_TEST}
};

std::string get_scenario_name(TestScenario scenario) {
    switch (scenario) {
        case BASIC_FUNCTION: return "basic";
        case BANDWIDTH_TEST: return "bandwidth";
        case SATURATION_TEST: return "saturation";
        case PSRR_TEST: return "psrr";
        case IMBALANCE_TEST: return "imbalance";
        case SLEW_RATE_TEST: return "slew";
        case PRBS_EYE_TEST: return "prbs";
        default: return "unknown";
    }
}

// ============================================================================
// Testbench Top Module
// ============================================================================

SC_MODULE(TxDriverTransientTestbench) {
    // Sub-modules
    DiffSignalSource* src;
    VddSource* vdd_src;
    TxDriverTdf* dut;
    SignalMonitor* monitor;
    
    // Signals
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_out_p;
    sca_tdf::sca_signal<double> sig_out_n;
    
    // Configuration
    TestScenario m_scenario;
    TxDriverParams m_params;
    DiffSignalSource::Config m_src_cfg;
    VddSource::Config m_vdd_cfg;
    double m_sim_duration;
    std::string m_output_file;
    
    SC_CTOR(TxDriverTransientTestbench)
        : sig_in_p("sig_in_p")
        , sig_in_n("sig_in_n")
        , sig_vdd("sig_vdd")
        , sig_out_p("sig_out_p")
        , sig_out_n("sig_out_n")
        , m_scenario(BASIC_FUNCTION)
        , m_sim_duration(100e-9)
    {
        // Will be configured in configure_scenario()
    }
    
    void configure_scenario(TestScenario scenario) {
        m_scenario = scenario;
        
        // Default parameters
        m_params = TxDriverParams();
        m_src_cfg = DiffSignalSource::Config();
        m_vdd_cfg = VddSource::Config();
        
        std::string scenario_name = get_scenario_name(scenario);
        m_output_file = "driver_tran_" + scenario_name + ".csv";
        
        switch (scenario) {
            case BASIC_FUNCTION:
                configure_basic();
                break;
            case BANDWIDTH_TEST:
                configure_bandwidth();
                break;
            case SATURATION_TEST:
                configure_saturation();
                break;
            case PSRR_TEST:
                configure_psrr();
                break;
            case IMBALANCE_TEST:
                configure_imbalance();
                break;
            case SLEW_RATE_TEST:
                configure_slew_rate();
                break;
            case PRBS_EYE_TEST:
                configure_prbs_eye();
                break;
        }
    }
    
    void configure_basic() {
        std::cout << "Configuring BASIC_FUNCTION test..." << std::endl;
        
        // Driver parameters
        m_params.dc_gain = 0.5;
        m_params.vswing = 0.8;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "soft";
        m_params.vlin = 0.5;
        m_params.poles = {50e9};
        
        // Source: step input
        m_src_cfg.type = DiffSignalSource::STEP;
        m_src_cfg.amplitude = 0.4;
        m_src_cfg.step_time = 10e-9;
        m_src_cfg.vcm = 0.0;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 100e-9;
    }
    
    void configure_bandwidth() {
        std::cout << "Configuring BANDWIDTH_TEST..." << std::endl;
        
        // Driver parameters with specific pole
        m_params.dc_gain = 1.0;
        m_params.vswing = 1.0;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "none";
        m_params.poles = {25e9};  // 25 GHz pole
        
        // Source: sine wave at pole frequency
        m_src_cfg.type = DiffSignalSource::SINE;
        m_src_cfg.amplitude = 0.2;
        m_src_cfg.frequency = 25e9;  // At pole frequency
        m_src_cfg.vcm = 0.0;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 10e-9;  // Several cycles
    }
    
    void configure_saturation() {
        std::cout << "Configuring SATURATION_TEST..." << std::endl;
        
        // Driver parameters for saturation test
        m_params.dc_gain = 1.0;
        m_params.vswing = 0.8;  // Vsat = 0.4V
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "soft";
        m_params.vlin = 0.3;
        m_params.poles.clear();  // No bandwidth limiting
        
        // Source: large sine wave to drive into saturation
        m_src_cfg.type = DiffSignalSource::SINE;
        m_src_cfg.amplitude = 1.0;  // Large amplitude
        m_src_cfg.frequency = 1e9;
        m_src_cfg.vcm = 0.0;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 10e-9;
    }
    
    void configure_psrr() {
        std::cout << "Configuring PSRR_TEST..." << std::endl;
        
        // Driver parameters with PSRR enabled
        m_params.dc_gain = 1.0;
        m_params.vswing = 0.8;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "none";
        m_params.poles.clear();
        m_params.psrr.enable = true;
        m_params.psrr.gain = 0.01;  // -40dB PSRR
        m_params.psrr.poles = {1e9};
        m_params.psrr.vdd_nom = 1.0;
        
        // Source: DC (no signal)
        m_src_cfg.type = DiffSignalSource::DC;
        m_src_cfg.amplitude = 0.0;  // No input signal
        m_src_cfg.vcm = 0.0;
        
        // VDD: with ripple
        m_vdd_cfg.mode = VddSource::SINUSOIDAL;
        m_vdd_cfg.nominal = 1.0;
        m_vdd_cfg.ripple_amp = 0.1;  // 100mV ripple
        m_vdd_cfg.ripple_freq = 100e6;  // 100 MHz
        
        m_sim_duration = 100e-9;
    }
    
    void configure_imbalance() {
        std::cout << "Configuring IMBALANCE_TEST..." << std::endl;
        
        // Driver parameters with imbalance
        m_params.dc_gain = 1.0;
        m_params.vswing = 0.8;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "none";
        m_params.poles.clear();
        m_params.imbalance.gain_mismatch = 10.0;  // 10% mismatch
        m_params.imbalance.skew = 0.0;
        
        // Source: sine wave
        m_src_cfg.type = DiffSignalSource::SINE;
        m_src_cfg.amplitude = 0.3;
        m_src_cfg.frequency = 1e9;
        m_src_cfg.vcm = 0.0;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 10e-9;
    }
    
    void configure_slew_rate() {
        std::cout << "Configuring SLEW_RATE_TEST..." << std::endl;
        
        // Driver parameters with slew rate limiting
        m_params.dc_gain = 1.0;
        m_params.vswing = 0.8;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "none";
        m_params.poles.clear();
        m_params.slew_rate.enable = true;
        m_params.slew_rate.max_slew_rate = 5e10;  // 50 V/ns
        
        // Source: step input
        m_src_cfg.type = DiffSignalSource::STEP;
        m_src_cfg.amplitude = 0.4;
        m_src_cfg.step_time = 5e-9;
        m_src_cfg.vcm = 0.0;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 50e-9;
    }
    
    void configure_prbs_eye() {
        std::cout << "Configuring PRBS_EYE_TEST..." << std::endl;
        
        // Driver parameters for eye diagram
        m_params.dc_gain = 0.5;
        m_params.vswing = 0.8;
        m_params.vcm_out = 0.6;
        m_params.output_impedance = 50.0;
        m_params.sat_mode = "soft";
        m_params.vlin = 0.5;
        m_params.poles = {40e9};
        
        // Source: PRBS7 at 10 Gbps
        m_src_cfg.type = DiffSignalSource::PRBS;
        m_src_cfg.amplitude = 0.5;
        m_src_cfg.frequency = 10e9;  // 10 Gbps
        m_src_cfg.vcm = 0.0;
        m_src_cfg.prbs_order = 7;
        m_src_cfg.prbs_seed = 0x7F;
        
        // VDD: constant
        m_vdd_cfg.mode = VddSource::CONSTANT;
        m_vdd_cfg.nominal = 1.0;
        
        m_sim_duration = 500e-9;  // Longer for eye diagram
    }
    
    void create_modules() {
        std::cout << "Creating test modules..." << std::endl;
        
        // Create source
        src = new DiffSignalSource("src", m_src_cfg);
        
        // Create VDD source
        vdd_src = new VddSource("vdd_src", m_vdd_cfg);
        
        // Create DUT
        dut = new TxDriverTdf("dut", m_params);
        
        // Create monitor with CSV output
        monitor = new SignalMonitor("monitor", m_output_file, true);
        
        // Connect signals
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        
        vdd_src->out(sig_vdd);
        
        dut->in_p(sig_in_p);
        dut->in_n(sig_in_n);
        dut->vdd(sig_vdd);
        dut->out_p(sig_out_p);
        dut->out_n(sig_out_n);
        
        monitor->in_p(sig_out_p);
        monitor->in_n(sig_out_n);
    }
    
    double get_sim_duration() const {
        return m_sim_duration;
    }
    
    void print_results() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "TX Driver Transient Test Results" << std::endl;
        std::cout << "Scenario: " << get_scenario_name(m_scenario) << std::endl;
        std::cout << "========================================" << std::endl;
        
        monitor->print_summary();
        
        std::cout << "Output file: " << m_output_file << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

// ============================================================================
// Main
// ============================================================================

int sc_main(int argc, char* argv[]) {
    // Parse command line arguments
    TestScenario scenario = BASIC_FUNCTION;
    
    if (argc > 1) {
        std::string arg = argv[1];
        
        // Check for help
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [scenario]\n\n";
            std::cout << "Available scenarios:\n";
            std::cout << "  basic      - Basic step response\n";
            std::cout << "  bandwidth  - Bandwidth measurement\n";
            std::cout << "  saturation - Saturation characterization\n";
            std::cout << "  psrr       - Power supply rejection test\n";
            std::cout << "  imbalance  - Gain mismatch test\n";
            std::cout << "  slew       - Slew rate limiting test\n";
            std::cout << "  prbs       - PRBS eye diagram\n";
            return 0;
        }
        
        // Find scenario
        auto it = scenario_map.find(arg);
        if (it != scenario_map.end()) {
            scenario = it->second;
        } else {
            std::cerr << "Unknown scenario: " << arg << std::endl;
            std::cerr << "Use -h for help" << std::endl;
            return 1;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "TX Driver Transient Testbench" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Create testbench
    TxDriverTransientTestbench tb("tb");
    
    // Configure scenario
    tb.configure_scenario(scenario);
    
    // Create and connect modules
    tb.create_modules();
    
    // Run simulation
    std::cout << "Starting simulation for " 
              << tb.get_sim_duration() * 1e9 << " ns..." << std::endl;
    
    sc_core::sc_start(tb.get_sim_duration(), sc_core::SC_SEC);
    
    // Print results
    tb.print_results();
    
    std::cout << "Simulation completed successfully." << std::endl;
    
    return 0;
}
