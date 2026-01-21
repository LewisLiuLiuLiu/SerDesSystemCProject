/**
 * @file wave_gen_tran_tb.cpp
 * @brief Wave Generation transient testbench
 * 
 * Test scenarios (per waveGen.md section 4.2):
 * - Scenario 0 (PRBS_BASIC): Basic PRBS generation test
 * - Scenario 1 (PULSE_TEST): Single-bit pulse timing test
 * - Scenario 2 (RJ_TEST): Random jitter injection test
 * - Scenario 3 (SJ_TEST): Sinusoidal jitter injection test
 * - Scenario 4 (STATS_TEST): Statistical characteristics test
 * 
 * Usage:
 *   ./wave_gen_tran_tb [scenario]
 *   
 *   scenario can be:
 *     0, prbs    - PRBS basic test
 *     1, pulse   - Single pulse test
 *     2, rj      - Random jitter test
 *     3, sj      - Sinusoidal jitter test
 *     4, stats   - Statistics test
 */

#include <systemc-ams>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include "ams/wave_generation.h"
#include "common/parameters.h"
#include "wave_gen_helpers.h"

using namespace serdes;

// ============================================================================
// Test Scenario Enumeration
// ============================================================================

enum TestScenario {
    PRBS_BASIC = 0,
    PULSE_TEST = 1,
    RJ_TEST = 2,
    SJ_TEST = 3,
    STATS_TEST = 4
};

// ============================================================================
// Testbench Top Module
// ============================================================================

SC_MODULE(WaveGenTranTestbench) {
    // Modules
    WaveGenerationTdf* wave_gen;
    WaveMonitor* monitor;
    
    // Signals
    sca_tdf::sca_signal<double> sig_wave;
    
    // Parameters
    WaveGenParams wave_params;
    double sample_rate;
    unsigned int seed;
    sc_core::sc_time sim_duration;
    std::string output_filename;
    TestScenario scenario;
    
    SC_CTOR(WaveGenTranTestbench)
        : sample_rate(80e9)
        , seed(12345)
        , sim_duration(1, sc_core::SC_US)
        , output_filename("wave_output.csv")
        , scenario(PRBS_BASIC)
    {}
    
    void configure(TestScenario sc) {
        scenario = sc;
        
        switch (scenario) {
            case PRBS_BASIC:
                configure_prbs_basic();
                break;
            case PULSE_TEST:
                configure_pulse_test();
                break;
            case RJ_TEST:
                configure_rj_test();
                break;
            case SJ_TEST:
                configure_sj_test();
                break;
            case STATS_TEST:
                configure_stats_test();
                break;
            default:
                configure_prbs_basic();
                break;
        }
    }
    
    void configure_prbs_basic() {
        std::cout << "=== Scenario: PRBS_BASIC ===" << std::endl;
        wave_params.type = PRBSType::PRBS31;
        wave_params.single_pulse = 0.0;
        wave_params.jitter.RJ_sigma = 0.0;
        wave_params.jitter.SJ_freq.clear();
        wave_params.jitter.SJ_pp.clear();
        sample_rate = 80e9;
        sim_duration = sc_core::sc_time(1, sc_core::SC_US);
        output_filename = "wave_prbs.csv";
    }
    
    void configure_pulse_test() {
        std::cout << "=== Scenario: PULSE_TEST ===" << std::endl;
        wave_params.type = PRBSType::PRBS31;
        wave_params.single_pulse = 100e-12;  // 100 ps pulse
        wave_params.jitter.RJ_sigma = 0.0;
        wave_params.jitter.SJ_freq.clear();
        wave_params.jitter.SJ_pp.clear();
        sample_rate = 80e9;
        sim_duration = sc_core::sc_time(1, sc_core::SC_NS);
        output_filename = "wave_pulse.csv";
    }
    
    void configure_rj_test() {
        std::cout << "=== Scenario: RJ_TEST ===" << std::endl;
        wave_params.type = PRBSType::PRBS31;
        wave_params.single_pulse = 0.0;
        wave_params.jitter.RJ_sigma = 5e-12;  // 5 ps RJ sigma
        wave_params.jitter.SJ_freq.clear();
        wave_params.jitter.SJ_pp.clear();
        sample_rate = 80e9;
        sim_duration = sc_core::sc_time(1, sc_core::SC_US);
        output_filename = "wave_rj.csv";
    }
    
    void configure_sj_test() {
        std::cout << "=== Scenario: SJ_TEST ===" << std::endl;
        wave_params.type = PRBSType::PRBS31;
        wave_params.single_pulse = 0.0;
        wave_params.jitter.RJ_sigma = 0.0;
        wave_params.jitter.SJ_freq.clear();
        wave_params.jitter.SJ_freq.push_back(5e6);  // 5 MHz
        wave_params.jitter.SJ_pp.clear();
        wave_params.jitter.SJ_pp.push_back(20e-12);  // 20 ps peak-to-peak
        sample_rate = 80e9;
        sim_duration = sc_core::sc_time(1, sc_core::SC_US);
        output_filename = "wave_sj.csv";
    }
    
    void configure_stats_test() {
        std::cout << "=== Scenario: STATS_TEST ===" << std::endl;
        wave_params.type = PRBSType::PRBS31;
        wave_params.single_pulse = 0.0;
        wave_params.jitter.RJ_sigma = 0.0;
        wave_params.jitter.SJ_freq.clear();
        wave_params.jitter.SJ_pp.clear();
        sample_rate = 80e9;
        sim_duration = sc_core::sc_time(10, sc_core::SC_US);  // Longer for stats
        output_filename = "wave_stats.csv";
    }
    
    void build() {
        // Calculate max samples based on simulation duration
        double sim_duration_sec = sim_duration.to_seconds();
        size_t max_samples = static_cast<size_t>(sim_duration_sec * sample_rate * 1.1);
        
        // Create modules
        wave_gen = new WaveGenerationTdf("wave_gen", wave_params, sample_rate, seed);
        monitor = new WaveMonitor("monitor", sample_rate, max_samples);
        
        // Connect signals
        wave_gen->out(sig_wave);
        monitor->in(sig_wave);
        
        std::cout << "Testbench built:" << std::endl;
        std::cout << "  Sample rate: " << sample_rate / 1e9 << " GHz" << std::endl;
        std::cout << "  Simulation duration: " << sim_duration << std::endl;
        std::cout << "  Max samples: " << max_samples << std::endl;
        std::cout << "  Output file: " << output_filename << std::endl;
        if (wave_params.single_pulse > 0) {
            std::cout << "  Mode: Single-bit pulse (" << wave_params.single_pulse * 1e12 << " ps)" << std::endl;
        } else {
            std::cout << "  Mode: PRBS" << std::endl;
        }
        if (wave_params.jitter.RJ_sigma > 0) {
            std::cout << "  RJ sigma: " << wave_params.jitter.RJ_sigma * 1e12 << " ps" << std::endl;
        }
        if (!wave_params.jitter.SJ_freq.empty()) {
            std::cout << "  SJ: " << wave_params.jitter.SJ_freq[0] / 1e6 << " MHz, " 
                      << wave_params.jitter.SJ_pp[0] * 1e12 << " ps pp" << std::endl;
        }
    }
    
    void run() {
        std::cout << "\nStarting simulation..." << std::endl;
        sc_core::sc_start(sim_duration);
        std::cout << "Simulation completed." << std::endl;
    }
    
    void analyze() {
        std::cout << "\n=== Analysis Results ===" << std::endl;
        
        // Get statistics
        WaveformStats stats = monitor->calculate_stats();
        stats.print();
        
        // Scenario-specific verification
        bool pass = true;
        
        switch (scenario) {
            case PRBS_BASIC:
                pass = verify_prbs_basic(stats);
                break;
            case PULSE_TEST:
                pass = verify_pulse_test(stats);
                break;
            case RJ_TEST:
                pass = verify_rj_test(stats);
                break;
            case SJ_TEST:
                pass = verify_sj_test(stats);
                break;
            case STATS_TEST:
                pass = verify_stats_test(stats);
                break;
        }
        
        std::cout << "\n=== Verification: " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;
        
        // Save waveform
        monitor->save_csv(output_filename);
    }
    
    bool verify_prbs_basic(const WaveformStats& stats) {
        bool pass = true;
        
        // Check mean is close to 0
        if (std::abs(stats.mean) >= 0.05) {
            std::cout << "FAIL: Mean " << stats.mean << " >= 0.05 V" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Mean within range" << std::endl;
        }
        
        // Check peak-to-peak is ~2.0V
        if (stats.peak_to_peak < 1.95 || stats.peak_to_peak > 2.05) {
            std::cout << "FAIL: Peak-to-peak " << stats.peak_to_peak << " not ~2.0 V" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Peak-to-peak within range" << std::endl;
        }
        
        // Check code balance < 1%
        if (stats.balance >= 0.01) {
            std::cout << "FAIL: Balance " << (stats.balance * 100) << "% >= 1%" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Code balance within range" << std::endl;
        }
        
        // Verify NRZ levels
        if (!StatisticsAnalyzer::verify_nrz_levels(monitor->get_samples())) {
            std::cout << "FAIL: Non-NRZ levels detected" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: NRZ levels correct" << std::endl;
        }
        
        return pass;
    }
    
    bool verify_pulse_test(const WaveformStats& stats) {
        bool pass = true;
        
        const std::vector<double>& samples = monitor->get_samples();
        const std::vector<double>& timestamps = monitor->get_timestamps();
        
        // Expected pulse width
        double expected_pulse = wave_params.single_pulse;
        double timestep = 1.0 / sample_rate;
        int expected_samples = static_cast<int>(expected_pulse / timestep);
        
        // Measure actual pulse width
        double pulse_width = StatisticsAnalyzer::measure_pulse_width(samples, timestamps);
        
        if (pulse_width < 0) {
            std::cout << "FAIL: Could not measure pulse width" << std::endl;
            pass = false;
        } else {
            double error = std::abs(pulse_width - expected_pulse);
            double tolerance = timestep * 1.5;  // Allow 1.5 timestep tolerance
            
            if (error > tolerance) {
                std::cout << "FAIL: Pulse width " << pulse_width * 1e12 << " ps, expected " 
                          << expected_pulse * 1e12 << " ps" << std::endl;
                pass = false;
            } else {
                std::cout << "PASS: Pulse width correct (" << pulse_width * 1e12 << " ps)" << std::endl;
            }
        }
        
        // Verify first samples are high (+1.0)
        int high_count = 0;
        for (size_t i = 0; i < samples.size() && i < static_cast<size_t>(expected_samples); ++i) {
            if (samples[i] > 0) high_count++;
        }
        
        if (high_count < expected_samples - 1) {
            std::cout << "FAIL: Expected " << expected_samples << " high samples, got " << high_count << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Pulse high samples correct" << std::endl;
        }
        
        // Verify samples after pulse are low (-1.0)
        int low_count = 0;
        for (size_t i = expected_samples; i < samples.size(); ++i) {
            if (samples[i] < 0) low_count++;
        }
        
        if (samples.size() > static_cast<size_t>(expected_samples) && 
            low_count < static_cast<int>(samples.size() - expected_samples - 1)) {
            std::cout << "FAIL: Samples after pulse should be low" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Post-pulse samples correct" << std::endl;
        }
        
        return pass;
    }
    
    bool verify_rj_test(const WaveformStats& stats) {
        bool pass = true;
        
        // Basic PRBS checks still apply
        if (std::abs(stats.mean) >= 0.1) {
            std::cout << "FAIL: Mean " << stats.mean << " outside range" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Mean within range" << std::endl;
        }
        
        // Note: Current implementation doesn't truly apply jitter to output
        // This is a placeholder for when jitter is properly implemented
        std::cout << "INFO: RJ injection is demonstration-only in current implementation" << std::endl;
        
        return pass;
    }
    
    bool verify_sj_test(const WaveformStats& stats) {
        bool pass = true;
        
        // Basic PRBS checks still apply
        if (std::abs(stats.mean) >= 0.1) {
            std::cout << "FAIL: Mean " << stats.mean << " outside range" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Mean within range" << std::endl;
        }
        
        // Note: Current implementation doesn't truly apply jitter to output
        // This is a placeholder for when jitter is properly implemented
        std::cout << "INFO: SJ injection is demonstration-only in current implementation" << std::endl;
        
        return pass;
    }
    
    bool verify_stats_test(const WaveformStats& stats) {
        bool pass = true;
        
        // Need more samples for stats test
        if (stats.sample_count < 100000) {
            std::cout << "WARN: Only " << stats.sample_count << " samples (expected >100000)" << std::endl;
        }
        
        // Check mean close to 0
        if (std::abs(stats.mean) >= 0.02) {
            std::cout << "FAIL: Mean " << stats.mean << " >= 0.02 V" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Mean within tight range" << std::endl;
        }
        
        // Check RMS close to 1.0
        if (std::abs(stats.rms - 1.0) >= 0.05) {
            std::cout << "FAIL: RMS " << stats.rms << " not ~1.0 V" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: RMS correct" << std::endl;
        }
        
        // Check code balance < 1%
        if (stats.balance >= 0.01) {
            std::cout << "FAIL: Balance " << (stats.balance * 100) << "% >= 1%" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: Code balance excellent" << std::endl;
        }
        
        return pass;
    }
    
    ~WaveGenTranTestbench() {
        delete wave_gen;
        delete monitor;
    }
};

// ============================================================================
// Main Function
// ============================================================================

TestScenario parse_scenario(const char* arg) {
    if (strcmp(arg, "0") == 0 || strcmp(arg, "prbs") == 0) {
        return PRBS_BASIC;
    } else if (strcmp(arg, "1") == 0 || strcmp(arg, "pulse") == 0) {
        return PULSE_TEST;
    } else if (strcmp(arg, "2") == 0 || strcmp(arg, "rj") == 0) {
        return RJ_TEST;
    } else if (strcmp(arg, "3") == 0 || strcmp(arg, "sj") == 0) {
        return SJ_TEST;
    } else if (strcmp(arg, "4") == 0 || strcmp(arg, "stats") == 0) {
        return STATS_TEST;
    }
    return PRBS_BASIC;  // Default
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [scenario]" << std::endl;
    std::cout << std::endl;
    std::cout << "Scenarios:" << std::endl;
    std::cout << "  0, prbs   - PRBS basic test (default)" << std::endl;
    std::cout << "  1, pulse  - Single-bit pulse test" << std::endl;
    std::cout << "  2, rj     - Random jitter test" << std::endl;
    std::cout << "  3, sj     - Sinusoidal jitter test" << std::endl;
    std::cout << "  4, stats  - Statistical characteristics test" << std::endl;
}

int sc_main(int argc, char* argv[]) {
    std::cout << "=== Wave Generation Transient Testbench ===" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    TestScenario scenario = PRBS_BASIC;
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        scenario = parse_scenario(argv[1]);
    }
    
    // Create and configure testbench
    WaveGenTranTestbench tb("tb");
    tb.configure(scenario);
    tb.build();
    
    // Run simulation
    tb.run();
    
    // Analyze results
    tb.analyze();
    
    std::cout << "\n=== Testbench completed ===" << std::endl;
    
    return 0;
}
