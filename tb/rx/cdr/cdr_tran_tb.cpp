/**
 * @file cdr_tran_tb.cpp
 * @brief CDR Transient Simulation Testbench
 * 
 * Supports multiple test scenarios as defined in cdr.md documentation:
 * - Scenario 0: PHASE_LOCK_BASIC - Basic phase locking test
 * - Scenario 1: FREQUENCY_OFFSET - Frequency offset capture test
 * - Scenario 2: JITTER_TOLERANCE - Jitter tolerance test (JTOL)
 * - Scenario 3: PHASE_TRACKING - Dynamic phase tracking test
 * - Scenario 4: LOOP_BANDWIDTH - Loop bandwidth measurement
 * 
 * @version 0.2
 * @date 2026-01-20
 */

#include <systemc-ams>
#include "ams/rx_cdr.h"
#include "ams/rx_sampler.h"
#include "common/parameters.h"
#include "cdr_helpers.h"
#include <iostream>
#include <string>
#include <iomanip>

using namespace serdes;
using namespace serdes::tb;

// ============================================================================
// Test Scenario Enumeration (Document Order)
// ============================================================================

enum TestScenario {
    PHASE_LOCK_BASIC = 0,    // Basic phase locking test
    FREQUENCY_OFFSET = 1,    // Frequency offset capture test
    JITTER_TOLERANCE = 2,    // Jitter tolerance test (JTOL)
    PHASE_TRACKING = 3,      // Dynamic phase tracking test
    LOOP_BANDWIDTH = 4       // Loop bandwidth measurement
};

// ============================================================================
// CDR Transient Testbench Module
// ============================================================================

SC_MODULE(CdrTransientTestbench) {
    // Module instances
    DataSource* src;
    SimpleSampler* sampler;
    RxCdrTdf* cdr;
    CdrMonitor* monitor;

    // Signal connections
    sca_tdf::sca_signal<double> sig_data;
    sca_tdf::sca_signal<double> sig_phase_offset;
    sca_tdf::sca_signal<double> sig_sampled;
    sca_tdf::sca_signal<double> sig_phase_out;

    TestScenario m_scenario;
    CdrParams m_params;
    double m_data_rate;
    double m_ui;

    CdrTransientTestbench(sc_core::sc_module_name nm,
                         TestScenario scenario = PHASE_LOCK_BASIC)
        : sc_module(nm)
        , m_scenario(scenario)
        , m_data_rate(10e9)  // Default 10Gbps
        , m_ui(1.0 / m_data_rate)
    {
        // Configure CDR parameters (default values from document)
        m_params.pi.kp = 0.01;
        m_params.pi.ki = 1e-4;
        m_params.pi.edge_threshold = 0.5;
        m_params.pi.adaptive_threshold = false;
        m_params.pai.resolution = 1e-12;
        m_params.pai.range = 5e-11;
        m_params.debug_enable = false;

        // Configure based on test scenario
        configure_scenario(scenario);

        // Create modules
        cdr = new RxCdrTdf("cdr", m_params);
        sampler = new SimpleSampler("sampler", m_data_rate);
        monitor = new CdrMonitor("monitor", get_output_filename(scenario), m_data_rate);

        // Connect modules
        src->out(sig_data);

        sampler->in(sig_data);
        sampler->phase_offset(sig_phase_offset);
        sampler->out(sig_sampled);

        cdr->in(sig_sampled);  // CDR gets data from sampler output
        cdr->phase_out(sig_phase_out);

        // Feedback connection: CDR phase output -> Sampler phase offset
        sig_phase_offset(sig_phase_out);

        monitor->phase_in(sig_phase_out);
        monitor->data_in(sig_sampled);
    }

    void configure_scenario(TestScenario scenario) {
        switch (scenario) {
            case PHASE_LOCK_BASIC:
                // Basic phase lock test - PRBS-15, no jitter, no frequency offset
                src = new DataSource("src",
                                     DataSource::PRBS15,
                                     1.0,           // Amplitude
                                     m_data_rate,
                                     100e9,         // Sample rate
                                     0.0,           // No random jitter
                                     0.0,           // No periodic jitter frequency
                                     0.0,           // No periodic jitter amplitude
                                     0.0);          // No frequency offset
                break;

            case FREQUENCY_OFFSET:
                // Frequency offset capture test - PRBS-7 + frequency offset
                src = new DataSource("src",
                                     DataSource::PRBS7,
                                     1.0,
                                     m_data_rate,
                                     100e9,
                                     0.0,
                                     0.0,
                                     0.0,
                                     100.0);        // 100ppm frequency offset
                // Increase PAI range for frequency offset tracking
                m_params.pai.range = 1e-10;  // ±100ps
                break;

            case JITTER_TOLERANCE:
                // Jitter tolerance test - PRBS-31 + RJ + SJ
                src = new DataSource("src",
                                     DataSource::PRBS31,
                                     1.0,
                                     m_data_rate,
                                     100e9,
                                     2e-12,         // 2ps random jitter (sigma)
                                     1e6,           // 1MHz periodic jitter frequency
                                     10e-12,        // 10ps periodic jitter amplitude
                                     0.0);
                break;

            case PHASE_TRACKING:
                // Dynamic phase tracking test - Alternating pattern + phase modulation
                src = new DataSource("src",
                                     DataSource::ALTERNATING,
                                     1.0,
                                     m_data_rate,
                                     100e9,
                                     0.0,
                                     10e6,          // 10MHz phase modulation
                                     20e-12,        // 20ps modulation amplitude
                                     0.0);
                break;

            case LOOP_BANDWIDTH:
                // Loop bandwidth measurement - Sine wave + small signal modulation
                src = new DataSource("src",
                                     DataSource::SINE,
                                     1.0,
                                     m_data_rate,
                                     100e9,
                                     0.0,
                                     5e6,           // 5MHz modulation (near expected BW)
                                     20e-12,        // 20ps amplitude (small signal)
                                     0.0);
                break;
        }
    }

    std::string get_output_filename(TestScenario scenario) {
        switch (scenario) {
            case PHASE_LOCK_BASIC:   return "cdr_tran_lock.csv";
            case FREQUENCY_OFFSET:   return "cdr_tran_freq.csv";
            case JITTER_TOLERANCE:   return "cdr_tran_jtol.csv";
            case PHASE_TRACKING:     return "cdr_tran_track.csv";
            case LOOP_BANDWIDTH:     return "cdr_tran_bw.csv";
            default:                 return "cdr_tran_output.csv";
        }
    }

    ~CdrTransientTestbench() {
        delete src;
        delete sampler;
        delete cdr;
        delete monitor;
    }

    void print_results() {
        PhaseStats stats = monitor->get_phase_stats(m_ui);

        std::cout << "\n";
        std::cout << "===============================================================================\n";
        std::cout << "  CDR Transient Simulation Results (" << get_scenario_name() << ")\n";
        std::cout << "===============================================================================\n";
        
        std::cout << "\n[Phase Adjustment Statistics]\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Mean:           " << std::setw(10) << stats.mean * 1e12 << " ps\n";
        std::cout << "  RMS:            " << std::setw(10) << stats.rms * 1e12 << " ps\n";
        std::cout << "  Peak-to-Peak:   " << std::setw(10) << stats.peak_to_peak * 1e12 << " ps\n";
        std::cout << "  Min:            " << std::setw(10) << stats.min_value * 1e12 << " ps\n";
        std::cout << "  Max:            " << std::setw(10) << stats.max_value * 1e12 << " ps\n";
        std::cout << "  Lock Time:      " << std::setw(10) << stats.lock_time * 1e9 << " ns";
        std::cout << " (" << stats.lock_time / m_ui << " UI)\n";
        std::cout << "  Steady-State RMS: " << std::setw(8) << stats.steady_state_rms * 1e12 << " ps\n";
        std::cout << "  Lock Status:    " << (monitor->is_locked() ? "LOCKED" : "NOT LOCKED") << "\n";

        std::cout << "\n[CDR Parameters]\n";
        std::cout << std::scientific << std::setprecision(2);
        std::cout << "  Kp:             " << m_params.pi.kp << "\n";
        std::cout << "  Ki:             " << m_params.pi.ki << "\n";
        std::cout << "  Edge Threshold: " << m_params.pi.edge_threshold << "\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  PAI Resolution: " << m_params.pai.resolution * 1e12 << " ps\n";
        std::cout << "  PAI Range:      " << m_params.pai.range * 1e12 << " ps\n";

        // Calculate theoretical loop parameters
        double bw_theory = LoopBandwidthAnalyzer::calculate_theoretical_bandwidth(
            m_params.pi.kp, m_params.pi.ki, m_data_rate);
        double zeta_theory = LoopBandwidthAnalyzer::calculate_damping_factor(
            m_params.pi.kp, m_params.pi.ki, m_data_rate);
        double pm_theory = LoopBandwidthAnalyzer::calculate_phase_margin(
            m_params.pi.kp, m_params.pi.ki, m_data_rate);

        std::cout << "\n[Theoretical Loop Parameters]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Loop Bandwidth: " << bw_theory / 1e6 << " MHz\n";
        std::cout << "  Damping Factor: " << zeta_theory << "\n";
        std::cout << "  Phase Margin:   " << pm_theory << " deg\n";

        std::cout << "\n[Output File]\n";
        std::cout << "  " << get_output_filename(m_scenario) << "\n";

        // Scenario-specific analysis
        analyze_scenario_results(stats);
        
        std::cout << "===============================================================================\n";
    }

    const char* get_scenario_name() {
        switch (m_scenario) {
            case PHASE_LOCK_BASIC:   return "PHASE_LOCK_BASIC";
            case FREQUENCY_OFFSET:   return "FREQUENCY_OFFSET";
            case JITTER_TOLERANCE:   return "JITTER_TOLERANCE";
            case PHASE_TRACKING:     return "PHASE_TRACKING";
            case LOOP_BANDWIDTH:     return "LOOP_BANDWIDTH";
            default:                 return "UNKNOWN";
        }
    }

    void analyze_scenario_results(const PhaseStats& stats) {
        std::cout << "\n[Scenario Analysis]\n";
        
        switch (m_scenario) {
            case PHASE_LOCK_BASIC:
                if (monitor->is_locked()) {
                    std::cout << "  [PASS] CDR successfully locked\n";
                    std::cout << "  Lock time: " << stats.lock_time / m_ui << " UI\n";
                    if (stats.lock_time / m_ui < 5000) {
                        std::cout << "  [PASS] Lock time within spec (< 5000 UI)\n";
                    } else {
                        std::cout << "  [WARN] Lock time exceeds typical spec (> 5000 UI)\n";
                    }
                } else {
                    std::cout << "  [FAIL] CDR did not lock\n";
                }
                if (stats.steady_state_rms * 1e12 < 5.0) {
                    std::cout << "  [PASS] Steady-state jitter within spec (< 5ps RMS)\n";
                } else {
                    std::cout << "  [WARN] Steady-state jitter exceeds typical spec (> 5ps RMS)\n";
                }
                break;

            case FREQUENCY_OFFSET:
                if (monitor->is_locked()) {
                    std::cout << "  [PASS] CDR successfully tracking frequency offset\n";
                    if (std::abs(stats.mean) < m_params.pai.range) {
                        std::cout << "  [PASS] Phase adjustment within range\n";
                    } else {
                        std::cout << "  [FAIL] Phase adjustment exceeds PAI range\n";
                    }
                } else {
                    std::cout << "  [WARN] CDR may still be tracking frequency offset\n";
                }
                std::cout << "  Frequency offset: 100 ppm\n";
                std::cout << "  Expected phase drift rate: 0.01 ps/UI\n";
                break;

            case JITTER_TOLERANCE:
                if (monitor->is_locked()) {
                    std::cout << "  [PASS] CDR tolerates injected jitter\n";
                }
                std::cout << "  Injected jitter: 2ps RJ (sigma) + 1MHz 10ps SJ\n";
                std::cout << "  Measured phase jitter RMS: " << stats.rms * 1e12 << " ps\n";
                break;

            case PHASE_TRACKING: {
                double bw_theory = LoopBandwidthAnalyzer::calculate_theoretical_bandwidth(
                    m_params.pi.kp, m_params.pi.ki, m_data_rate);
                std::cout << "  Modulation frequency: 10 MHz\n";
                std::cout << "  Theoretical loop bandwidth: " << bw_theory / 1e6 << " MHz\n";
                if (10e6 > bw_theory) {
                    std::cout << "  [WARN] Modulation frequency exceeds loop bandwidth\n";
                    std::cout << "         CDR may not fully track the phase modulation\n";
                } else {
                    std::cout << "  [INFO] Modulation frequency within loop bandwidth\n";
                }
                break;
            }

            case LOOP_BANDWIDTH: {
                double bw_theory = LoopBandwidthAnalyzer::calculate_theoretical_bandwidth(
                    m_params.pi.kp, m_params.pi.ki, m_data_rate);
                std::cout << "  Theoretical bandwidth: " << bw_theory / 1e6 << " MHz\n";
                std::cout << "  Test modulation frequency: 5 MHz\n";
                std::cout << "  [INFO] For accurate bandwidth measurement, analyze the\n";
                std::cout << "         output file with frequency sweep post-processing\n";
                break;
            }

            default:
                break;
        }
    }
};

// ============================================================================
// SystemC Main Function
// ============================================================================

int sc_main(int argc, char* argv[]) {
    // Parse command line arguments
    TestScenario scenario = PHASE_LOCK_BASIC;

    if (argc > 1) {
        std::string arg = argv[1];
        
        // Support both string and numeric arguments
        if (arg == "PHASE_LOCK_BASIC" || arg == "lock" || arg == "0") {
            scenario = PHASE_LOCK_BASIC;
        } else if (arg == "FREQUENCY_OFFSET" || arg == "freq" || arg == "1") {
            scenario = FREQUENCY_OFFSET;
        } else if (arg == "JITTER_TOLERANCE" || arg == "jtol" || arg == "2") {
            scenario = JITTER_TOLERANCE;
        } else if (arg == "PHASE_TRACKING" || arg == "track" || arg == "3") {
            scenario = PHASE_TRACKING;
        } else if (arg == "LOOP_BANDWIDTH" || arg == "bw" || arg == "4") {
            scenario = LOOP_BANDWIDTH;
        } else {
            std::cout << "CDR Transient Testbench\n";
            std::cout << "======================\n\n";
            std::cout << "Usage: " << argv[0] << " [scenario]\n\n";
            std::cout << "Scenarios (per cdr.md document order):\n";
            std::cout << "  0, lock, PHASE_LOCK_BASIC   - Basic phase locking test (default)\n";
            std::cout << "  1, freq, FREQUENCY_OFFSET   - Frequency offset capture test\n";
            std::cout << "  2, jtol, JITTER_TOLERANCE   - Jitter tolerance test\n";
            std::cout << "  3, track, PHASE_TRACKING    - Dynamic phase tracking test\n";
            std::cout << "  4, bw, LOOP_BANDWIDTH       - Loop bandwidth measurement\n";
            return 1;
        }
    }

    // Create testbench
    CdrTransientTestbench tb("tb", scenario);

    // Determine simulation time based on scenario
    double sim_time = 1e-6;  // Default 1us (10,000 UI @ 10Gbps)
    
    switch (scenario) {
        case PHASE_LOCK_BASIC:
            sim_time = 1e-6;   // 10,000 UI - sufficient for lock
            break;
        case FREQUENCY_OFFSET:
            sim_time = 5e-6;   // 50,000 UI - track frequency drift
            break;
        case JITTER_TOLERANCE:
            sim_time = 10e-6;  // 100,000 UI - statistical significance
            break;
        case PHASE_TRACKING:
            sim_time = 5e-6;   // 50,000 UI - multiple modulation cycles
            break;
        case LOOP_BANDWIDTH:
            sim_time = 1e-6;   // 10,000 UI - frequency response
            break;
    }

    // Print simulation header
    std::cout << "\n";
    std::cout << "===============================================================================\n";
    std::cout << "  CDR Transient Simulation Starting\n";
    std::cout << "===============================================================================\n";
    std::cout << "  Scenario:        " << tb.get_scenario_name() << "\n";
    std::cout << "  Data Rate:       10 Gbps\n";
    std::cout << "  UI Period:       100 ps\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Simulation Time: " << sim_time * 1e6 << " us (" 
              << sim_time / tb.m_ui << " UI)\n";
    std::cout << "===============================================================================\n";

    // Run transient simulation
    sc_core::sc_start(sim_time, sc_core::SC_SEC);

    // Print results
    tb.print_results();

    std::cout << "\nSimulation completed successfully!\n";
    return 0;
}
