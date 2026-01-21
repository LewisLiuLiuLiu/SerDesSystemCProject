/**
 * @file adaption_tran_tb.cpp
 * @brief Adaption transient simulation testbench
 * 
 * Supports 8 test scenarios as defined in adaption.md:
 * - BASIC_FUNCTION: All algorithms combined
 * - AGC_TEST: AGC automatic gain control
 * - DFE_TEST: DFE tap update (LMS/Sign-LMS)
 * - THRESHOLD_TEST: Threshold adaptation
 * - CDR_PI_TEST: CDR PI controller
 * - FREEZE_ROLLBACK: Freeze and rollback mechanism
 * - MULTI_RATE: Multi-rate scheduling
 * - SCENARIO_SWITCH: Multi-scenario hot switching
 */

#include <systemc>
#include "ams/adaption.h"
#include "common/parameters.h"
#include "adaption_helpers.h"
#include <iostream>
#include <string>
#include <iomanip>

using namespace serdes;
using namespace serdes::tb;

// ============================================================================
// Test Scenario Enumeration
// ============================================================================
enum TestScenario {
    BASIC_FUNCTION = 0,   // Basic function test (all algorithms combined)
    AGC_TEST = 1,         // AGC automatic gain control
    DFE_TEST = 2,         // DFE tap update
    THRESHOLD_TEST = 3,   // Threshold adaptation
    CDR_PI_TEST = 4,      // CDR PI controller
    FREEZE_ROLLBACK = 5,  // Freeze and rollback mechanism
    MULTI_RATE = 6,       // Multi-rate scheduling
    SCENARIO_SWITCH = 7   // Multi-scenario hot switching
};

// ============================================================================
// Adaption Transient Testbench
// ============================================================================
SC_MODULE(AdaptionTransientTestbench) {
    // Module instances
    AdaptionSignalSource* src;
    AdaptionDe* adaption;
    AdaptionMonitor* monitor;
    
    // Signal connections - inputs to Adaption
    sc_core::sc_signal<double> sig_phase_error;
    sc_core::sc_signal<double> sig_amplitude_rms;
    sc_core::sc_signal<int> sig_error_count;
    sc_core::sc_signal<double> sig_isi_metric;
    sc_core::sc_signal<int> sig_mode;
    sc_core::sc_signal<bool> sig_reset;
    sc_core::sc_signal<double> sig_scenario_switch;
    
    // Signal connections - outputs from Adaption
    sc_core::sc_signal<double> sig_vga_gain;
    sc_core::sc_signal<double> sig_ctle_zero;
    sc_core::sc_signal<double> sig_ctle_pole;
    sc_core::sc_signal<double> sig_ctle_dc_gain;
    sc_core::sc_signal<double> sig_dfe_tap1;
    sc_core::sc_signal<double> sig_dfe_tap2;
    sc_core::sc_signal<double> sig_dfe_tap3;
    sc_core::sc_signal<double> sig_dfe_tap4;
    sc_core::sc_signal<double> sig_dfe_tap5;
    sc_core::sc_signal<double> sig_dfe_tap6;
    sc_core::sc_signal<double> sig_dfe_tap7;
    sc_core::sc_signal<double> sig_dfe_tap8;
    sc_core::sc_signal<double> sig_sampler_threshold;
    sc_core::sc_signal<double> sig_sampler_hysteresis;
    sc_core::sc_signal<double> sig_phase_cmd;
    sc_core::sc_signal<int> sig_update_count;
    sc_core::sc_signal<bool> sig_freeze_flag;
    
    // Test configuration
    TestScenario m_scenario;
    AdaptionParams m_params;
    double m_sim_duration;
    double m_UI;
    std::string m_output_filename;
    
    AdaptionTransientTestbench(sc_core::sc_module_name nm,
                               TestScenario scenario = BASIC_FUNCTION)
        : sc_core::sc_module(nm)
        , m_scenario(scenario)
        , m_sim_duration(10e-6)  // 10 us default
        , m_UI(2.5e-11)          // 25 ps (40 Gbps)
    {
        // Configure parameters based on scenario
        configure_scenario(scenario);
        
        // Create signal source
        src = new AdaptionSignalSource("src", m_params.fast_update_period);
        configure_source(scenario);
        
        // Create Adaption module
        adaption = new AdaptionDe("adaption", m_params);
        
        // Create monitor
        m_output_filename = get_output_filename(scenario);
        monitor = new AdaptionMonitor("monitor", m_output_filename, 
                                      m_params.fast_update_period, 
                                      m_params.dfe.num_taps);
        
        // Connect source to Adaption inputs
        src->phase_error(sig_phase_error);
        src->amplitude_rms(sig_amplitude_rms);
        src->error_count(sig_error_count);
        src->isi_metric(sig_isi_metric);
        src->mode(sig_mode);
        src->reset(sig_reset);
        src->scenario_switch(sig_scenario_switch);
        
        adaption->phase_error(sig_phase_error);
        adaption->amplitude_rms(sig_amplitude_rms);
        adaption->error_count(sig_error_count);
        adaption->isi_metric(sig_isi_metric);
        adaption->mode(sig_mode);
        adaption->reset(sig_reset);
        adaption->scenario_switch(sig_scenario_switch);
        
        // Connect Adaption outputs
        adaption->vga_gain(sig_vga_gain);
        adaption->ctle_zero(sig_ctle_zero);
        adaption->ctle_pole(sig_ctle_pole);
        adaption->ctle_dc_gain(sig_ctle_dc_gain);
        adaption->dfe_tap1(sig_dfe_tap1);
        adaption->dfe_tap2(sig_dfe_tap2);
        adaption->dfe_tap3(sig_dfe_tap3);
        adaption->dfe_tap4(sig_dfe_tap4);
        adaption->dfe_tap5(sig_dfe_tap5);
        adaption->dfe_tap6(sig_dfe_tap6);
        adaption->dfe_tap7(sig_dfe_tap7);
        adaption->dfe_tap8(sig_dfe_tap8);
        adaption->sampler_threshold(sig_sampler_threshold);
        adaption->sampler_hysteresis(sig_sampler_hysteresis);
        adaption->phase_cmd(sig_phase_cmd);
        adaption->update_count(sig_update_count);
        adaption->freeze_flag(sig_freeze_flag);
        
        // Connect monitor inputs
        monitor->vga_gain(sig_vga_gain);
        monitor->dfe_tap1(sig_dfe_tap1);
        monitor->dfe_tap2(sig_dfe_tap2);
        monitor->dfe_tap3(sig_dfe_tap3);
        monitor->dfe_tap4(sig_dfe_tap4);
        monitor->dfe_tap5(sig_dfe_tap5);
        monitor->dfe_tap6(sig_dfe_tap6);
        monitor->dfe_tap7(sig_dfe_tap7);
        monitor->dfe_tap8(sig_dfe_tap8);
        monitor->sampler_threshold(sig_sampler_threshold);
        monitor->sampler_hysteresis(sig_sampler_hysteresis);
        monitor->phase_cmd(sig_phase_cmd);
        monitor->update_count(sig_update_count);
        monitor->freeze_flag(sig_freeze_flag);
        monitor->phase_error(sig_phase_error);
        monitor->amplitude_rms(sig_amplitude_rms);
        monitor->error_count(sig_error_count);
    }
    
    ~AdaptionTransientTestbench() {
        delete src;
        delete adaption;
        delete monitor;
    }
    
    void configure_scenario(TestScenario scenario) {
        // Common parameters
        m_params.Fs = 80e9;
        m_params.UI = m_UI;
        m_params.seed = 12345;
        m_params.update_mode = "multi-rate";
        m_params.fast_update_period = 2.5e-10;  // 10 UI
        m_params.slow_update_period = 2.5e-7;   // 10000 UI
        
        switch (scenario) {
            case BASIC_FUNCTION:
                // All algorithms enabled with default parameters
                m_params.agc.enabled = true;
                m_params.agc.target_amplitude = 0.4;
                m_params.agc.kp = 0.1;
                m_params.agc.ki = 100.0;
                m_params.agc.gain_min = 0.5;
                m_params.agc.gain_max = 8.0;
                m_params.agc.initial_gain = 2.0;
                
                m_params.dfe.enabled = true;
                m_params.dfe.num_taps = 5;
                m_params.dfe.algorithm = "sign-lms";
                m_params.dfe.mu = 1e-4;
                
                m_params.threshold.enabled = true;
                m_params.threshold.initial = 0.0;
                m_params.threshold.hysteresis = 0.02;
                
                m_params.cdr_pi.enabled = true;
                m_params.cdr_pi.kp = 0.01;
                m_params.cdr_pi.ki = 1e-4;
                m_params.cdr_pi.phase_range = 0.5 * m_UI;
                
                m_sim_duration = 10e-6;
                break;
                
            case AGC_TEST:
                // AGC test with amplitude step
                m_params.agc.enabled = true;
                m_params.agc.target_amplitude = 0.4;
                m_params.agc.kp = 0.1;
                m_params.agc.ki = 100.0;
                m_params.agc.rate_limit = 10.0;
                
                m_params.dfe.enabled = false;
                m_params.threshold.enabled = false;
                m_params.cdr_pi.enabled = false;
                
                m_sim_duration = 10e-6;
                break;
                
            case DFE_TEST:
                // DFE test with strong ISI
                m_params.agc.enabled = false;
                
                m_params.dfe.enabled = true;
                m_params.dfe.num_taps = 8;
                m_params.dfe.algorithm = "sign-lms";
                m_params.dfe.mu = 1e-4;
                m_params.dfe.leakage = 1e-6;
                
                m_params.threshold.enabled = false;
                m_params.cdr_pi.enabled = false;
                
                m_sim_duration = 10e-6;
                break;
                
            case THRESHOLD_TEST:
                // Threshold adaptation test
                m_params.agc.enabled = false;
                m_params.dfe.enabled = false;
                
                m_params.threshold.enabled = true;
                m_params.threshold.initial = 0.0;
                m_params.threshold.adapt_step = 0.001;
                m_params.threshold.drift_threshold = 0.05;
                
                m_params.cdr_pi.enabled = false;
                
                m_sim_duration = 10e-6;
                break;
                
            case CDR_PI_TEST:
                // CDR PI test
                m_params.agc.enabled = false;
                m_params.dfe.enabled = false;
                m_params.threshold.enabled = false;
                
                m_params.cdr_pi.enabled = true;
                m_params.cdr_pi.kp = 0.01;
                m_params.cdr_pi.ki = 1e-4;
                m_params.cdr_pi.phase_resolution = 1e-12;
                m_params.cdr_pi.phase_range = 5e-11;
                m_params.cdr_pi.anti_windup = true;
                
                m_sim_duration = 10e-6;
                break;
                
            case FREEZE_ROLLBACK:
                // Freeze and rollback test
                m_params.agc.enabled = true;
                m_params.dfe.enabled = true;
                m_params.threshold.enabled = true;
                m_params.cdr_pi.enabled = true;
                
                m_params.safety.freeze_on_error = true;
                m_params.safety.rollback_enable = true;
                m_params.safety.snapshot_interval = 1e-6;
                m_params.safety.error_burst_threshold = 100;
                
                m_sim_duration = 10e-6;
                break;
                
            case MULTI_RATE:
                // Multi-rate scheduling test
                m_params.fast_update_period = 2.5e-11;  // 1 UI
                m_params.slow_update_period = 2.5e-9;   // 100 UI
                
                m_params.agc.enabled = true;
                m_params.dfe.enabled = true;
                m_params.threshold.enabled = true;
                m_params.cdr_pi.enabled = true;
                
                m_sim_duration = 10e-6;
                break;
                
            case SCENARIO_SWITCH:
                // Scenario switch test
                m_params.agc.enabled = true;
                m_params.dfe.enabled = true;
                m_params.threshold.enabled = true;
                m_params.cdr_pi.enabled = true;
                
                m_sim_duration = 9e-6;
                break;
        }
    }
    
    void configure_source(TestScenario scenario) {
        switch (scenario) {
            case BASIC_FUNCTION:
                src->set_stimulus_type(AdaptionSignalSource::CONSTANT);
                src->set_amplitude_target(0.3);
                src->set_phase_error_init(0.5 * m_UI);
                break;
                
            case AGC_TEST:
                src->set_stimulus_type(AdaptionSignalSource::STEP);
                src->configure_amplitude_step(0.2, 0.6, 2e-6);
                src->set_phase_error_init(0.0);
                break;
                
            case DFE_TEST:
                src->set_stimulus_type(AdaptionSignalSource::CONSTANT);
                src->set_amplitude_target(0.4);
                src->set_error_rate(10);
                break;
                
            case THRESHOLD_TEST:
                src->set_stimulus_type(AdaptionSignalSource::SINUSOIDAL);
                src->set_amplitude_target(0.4);
                src->set_amplitude_noise(0.05);
                break;
                
            case CDR_PI_TEST:
                src->set_stimulus_type(AdaptionSignalSource::CONSTANT);
                src->set_phase_error_init(0.5 * m_UI);
                src->set_phase_error_noise(0.001 * m_UI);
                break;
                
            case FREEZE_ROLLBACK:
                src->set_stimulus_type(AdaptionSignalSource::FAULT_INJECTION);
                src->configure_fault_injection(3e-6, 0.5e-6, 0);  // Error burst at 3us
                break;
                
            case MULTI_RATE:
                src->set_stimulus_type(AdaptionSignalSource::CONSTANT);
                src->set_amplitude_target(0.35);
                src->set_phase_error_init(0.3 * m_UI);
                break;
                
            case SCENARIO_SWITCH:
                src->set_stimulus_type(AdaptionSignalSource::STEP);
                src->set_amplitude_target(0.3);
                src->set_step_time(3e-6);
                break;
        }
    }
    
    std::string get_output_filename(TestScenario scenario) {
        switch (scenario) {
            case BASIC_FUNCTION:   return "adaption_basic.csv";
            case AGC_TEST:         return "adaption_agc.csv";
            case DFE_TEST:         return "adaption_dfe.csv";
            case THRESHOLD_TEST:   return "adaption_threshold.csv";
            case CDR_PI_TEST:      return "adaption_cdr.csv";
            case FREEZE_ROLLBACK:  return "adaption_safety.csv";
            case MULTI_RATE:       return "adaption_multirate.csv";
            case SCENARIO_SWITCH:  return "adaption_switch.csv";
            default:               return "adaption_output.csv";
        }
    }
    
    const char* get_scenario_name() {
        switch (m_scenario) {
            case BASIC_FUNCTION:   return "BASIC_FUNCTION";
            case AGC_TEST:         return "AGC_TEST";
            case DFE_TEST:         return "DFE_TEST";
            case THRESHOLD_TEST:   return "THRESHOLD_TEST";
            case CDR_PI_TEST:      return "CDR_PI_TEST";
            case FREEZE_ROLLBACK:  return "FREEZE_ROLLBACK";
            case MULTI_RATE:       return "MULTI_RATE";
            case SCENARIO_SWITCH:  return "SCENARIO_SWITCH";
            default:               return "UNKNOWN";
        }
    }
    
    double get_sim_duration() const { return m_sim_duration; }
    
    void print_results() {
        AdaptionStats stats = monitor->get_stats(m_UI);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Adaption Testbench - " << get_scenario_name() << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "Simulation Configuration:" << std::endl;
        std::cout << "  Symbol rate: " << (1.0 / m_UI / 1e9) << " Gbps (UI = " 
                  << (m_UI * 1e12) << " ps)" << std::endl;
        std::cout << "  Simulation duration: " << (m_sim_duration * 1e6) << " us ("
                  << static_cast<int>(m_sim_duration / m_UI) << " UI)" << std::endl;
        std::cout << "  Update mode: " << m_params.update_mode << std::endl;
        std::cout << "  Fast path period: " << (m_params.fast_update_period * 1e12) << " ps ("
                  << static_cast<int>(m_params.fast_update_period / m_UI) << " UI)" << std::endl;
        std::cout << "  Slow path period: " << (m_params.slow_update_period * 1e9) << " ns ("
                  << static_cast<int>(m_params.slow_update_period / m_UI) << " UI)" << std::endl;
        
        if (m_params.agc.enabled) {
            std::cout << "\nAGC Statistics:" << std::endl;
            std::cout << "  Initial gain: " << m_params.agc.initial_gain << std::endl;
            std::cout << "  Final gain: " << stats.final_gain << std::endl;
            std::cout << "  Target amplitude: " << m_params.agc.target_amplitude << " V" << std::endl;
            if (stats.agc_convergence_time > 0) {
                std::cout << "  Convergence time: " << (stats.agc_convergence_time * 1e9) << " ns ("
                          << static_cast<int>(stats.agc_convergence_time / m_UI) << " UI)" << std::endl;
            } else {
                std::cout << "  Convergence time: Not converged" << std::endl;
            }
            std::cout << "  Steady-state error: " << stats.agc_steady_error << " %" << std::endl;
        }
        
        if (m_params.dfe.enabled) {
            std::cout << "\nDFE Statistics:" << std::endl;
            std::cout << "  Number of taps: " << m_params.dfe.num_taps << std::endl;
            std::cout << "  Algorithm: " << m_params.dfe.algorithm << std::endl;
            std::cout << "  Step size: " << m_params.dfe.mu << std::endl;
            if (stats.dfe_convergence_time > 0) {
                std::cout << "  Convergence time: " << (stats.dfe_convergence_time * 1e9) << " ns ("
                          << static_cast<int>(stats.dfe_convergence_time / m_UI) << " UI)" << std::endl;
            } else {
                std::cout << "  Convergence time: Not converged" << std::endl;
            }
        }
        
        if (m_params.cdr_pi.enabled) {
            std::cout << "\nCDR PI Statistics:" << std::endl;
            std::cout << "  Kp: " << m_params.cdr_pi.kp << std::endl;
            std::cout << "  Ki: " << m_params.cdr_pi.ki << std::endl;
            std::cout << "  Phase range: " << (m_params.cdr_pi.phase_range * 1e12) << " ps" << std::endl;
            std::cout << "  Final phase command: " << (stats.final_phase_cmd * 1e12) << " ps" << std::endl;
            if (stats.cdr_lock_time > 0) {
                std::cout << "  Lock time: " << (stats.cdr_lock_time * 1e9) << " ns ("
                          << static_cast<int>(stats.cdr_lock_time / m_UI) << " UI)" << std::endl;
            } else {
                std::cout << "  Lock time: Not locked" << std::endl;
            }
            std::cout << "  Steady-state RMS: " << stats.cdr_steady_rms << " UI" << std::endl;
        }
        
        if (m_params.threshold.enabled) {
            std::cout << "\nThreshold Adaptation Statistics:" << std::endl;
            std::cout << "  Initial threshold: " << m_params.threshold.initial << " V" << std::endl;
            std::cout << "  Final threshold: " << stats.final_threshold << " V" << std::endl;
            std::cout << "  Hysteresis: " << m_params.threshold.hysteresis << " V" << std::endl;
        }
        
        std::cout << "\nSafety Mechanism Statistics:" << std::endl;
        std::cout << "  Freeze events: " << stats.freeze_events << std::endl;
        std::cout << "  Rollback events: " << stats.rollback_events << std::endl;
        
        std::cout << "\nUpdate Statistics:" << std::endl;
        std::cout << "  Total updates: " << stats.total_updates << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Output file: " << m_output_filename << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

// ============================================================================
// Parse scenario from command line
// ============================================================================
TestScenario parse_scenario(const std::string& arg) {
    if (arg == "basic" || arg == "0") return BASIC_FUNCTION;
    if (arg == "agc" || arg == "1") return AGC_TEST;
    if (arg == "dfe" || arg == "2") return DFE_TEST;
    if (arg == "threshold" || arg == "3") return THRESHOLD_TEST;
    if (arg == "cdr_pi" || arg == "cdr" || arg == "4") return CDR_PI_TEST;
    if (arg == "safety" || arg == "freeze" || arg == "5") return FREEZE_ROLLBACK;
    if (arg == "multirate" || arg == "6") return MULTI_RATE;
    if (arg == "switch" || arg == "7") return SCENARIO_SWITCH;
    
    // Default
    return BASIC_FUNCTION;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [scenario]\n" << std::endl;
    std::cout << "Scenarios:" << std::endl;
    std::cout << "  basic / 0      Basic function test (all algorithms)" << std::endl;
    std::cout << "  agc / 1        AGC automatic gain control test" << std::endl;
    std::cout << "  dfe / 2        DFE tap update test" << std::endl;
    std::cout << "  threshold / 3  Threshold adaptation test" << std::endl;
    std::cout << "  cdr_pi / 4     CDR PI controller test" << std::endl;
    std::cout << "  safety / 5     Freeze and rollback mechanism test" << std::endl;
    std::cout << "  multirate / 6  Multi-rate scheduling test" << std::endl;
    std::cout << "  switch / 7     Multi-scenario hot switching test" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================
int sc_main(int argc, char* argv[]) {
    // Parse command line arguments
    TestScenario scenario = BASIC_FUNCTION;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        scenario = parse_scenario(arg);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Adaption Transient Testbench" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Create testbench
    AdaptionTransientTestbench* tb = new AdaptionTransientTestbench("tb", scenario);
    
    // Run simulation
    double sim_duration = tb->get_sim_duration();
    std::cout << "Running simulation for " << (sim_duration * 1e6) << " us..." << std::endl;
    
    sc_core::sc_start(sim_duration, sc_core::SC_SEC);
    
    // Print results
    tb->print_results();
    
    // Cleanup
    delete tb;
    
    return 0;
}
