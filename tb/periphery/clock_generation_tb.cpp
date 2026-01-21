/**
 * @file clock_generation_tb.cpp
 * @brief Testbench for Clock Generation module
 * 
 * This testbench provides interactive testing of the ClockGenerationTdf module
 * with multiple test scenarios:
 * - IDEAL clock generation
 * - Phase continuity verification
 * - Different frequency configurations
 * - Long-term numerical stability
 * 
 * Output: clock_generation.dat - Tabular format trace file
 * 
 * @version 0.1
 * @date 2026-01-21
 */

#include <systemc-ams>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include "ams/clock_generation.h"
#include "common/parameters.h"
#include "common/types.h"

using namespace serdes;

// ============================================================================
// Phase Analyzer Module
// ============================================================================

/**
 * @brief Phase analyzer module for detailed analysis
 * Collects and analyzes clock phase output
 */
class PhaseAnalyzer : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> phase_in;
    
    // Statistics
    double m_min_phase;
    double m_max_phase;
    double m_sum_phase;
    double m_sum_phase_sq;
    int m_sample_count;
    int m_wrap_count;
    double m_prev_phase;
    
    // Phase increment statistics
    double m_sum_increment;
    double m_sum_increment_sq;
    int m_increment_count;
    
    PhaseAnalyzer(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , phase_in("phase_in")
        , m_min_phase(1e10)
        , m_max_phase(-1e10)
        , m_sum_phase(0.0)
        , m_sum_phase_sq(0.0)
        , m_sample_count(0)
        , m_wrap_count(0)
        , m_prev_phase(0.0)
        , m_sum_increment(0.0)
        , m_sum_increment_sq(0.0)
        , m_increment_count(0)
    {}
    
    void set_attributes() {
        phase_in.set_rate(1);
    }
    
    void processing() {
        double phase = phase_in.read();
        
        // Update basic statistics
        if (phase < m_min_phase) m_min_phase = phase;
        if (phase > m_max_phase) m_max_phase = phase;
        m_sum_phase += phase;
        m_sum_phase_sq += phase * phase;
        
        // Count phase wraps and calculate increments
        if (m_sample_count > 0) {
            double delta = phase - m_prev_phase;
            if (delta < -M_PI) {
                delta += 2.0 * M_PI;
                m_wrap_count++;
            }
            m_sum_increment += delta;
            m_sum_increment_sq += delta * delta;
            m_increment_count++;
        }
        
        m_prev_phase = phase;
        m_sample_count++;
    }
    
    // Analysis methods
    double get_mean_phase() const {
        return m_sample_count > 0 ? m_sum_phase / m_sample_count : 0.0;
    }
    
    double get_phase_std() const {
        if (m_sample_count < 2) return 0.0;
        double mean = get_mean_phase();
        double variance = m_sum_phase_sq / m_sample_count - mean * mean;
        return variance > 0 ? std::sqrt(variance) : 0.0;
    }
    
    double get_mean_increment() const {
        return m_increment_count > 0 ? m_sum_increment / m_increment_count : 0.0;
    }
    
    double get_increment_std() const {
        if (m_increment_count < 2) return 0.0;
        double mean = get_mean_increment();
        double variance = m_sum_increment_sq / m_increment_count - mean * mean;
        return variance > 0 ? std::sqrt(variance) : 0.0;
    }
    
    void print_statistics() const {
        std::cout << "\n=== Phase Analysis Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Total samples: " << m_sample_count << std::endl;
        std::cout << "  Phase range: [" << m_min_phase << ", " << m_max_phase << "] rad" << std::endl;
        std::cout << "  Phase mean: " << get_mean_phase() << " rad (expected: " << M_PI << ")" << std::endl;
        std::cout << "  Phase std: " << get_phase_std() << " rad" << std::endl;
        std::cout << "  Cycle count: " << m_wrap_count << std::endl;
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Phase increment mean: " << get_mean_increment() << " rad" << std::endl;
        std::cout << "  Phase increment std: " << get_increment_std() << " rad" << std::endl;
        std::cout << "==============================\n" << std::endl;
    }
};

// ============================================================================
// Clock Generation Testbench
// ============================================================================

SC_MODULE(ClockGenerationTestbench) {
    // Modules
    ClockGenerationTdf* clk_gen;
    PhaseAnalyzer* analyzer;
    
    // Signals
    sca_tdf::sca_signal<double> sig_phase;
    
    // Trace file
    sca_util::sca_trace_file* tf;
    
    // Parameters
    ClockParams m_params;
    double m_sim_duration;
    std::string m_output_filename;
    
    SC_CTOR(ClockGenerationTestbench)
        : tf(nullptr)
        , m_sim_duration(1e-6)
        , m_output_filename("clock_generation.dat")
    {}
    
    void configure(const ClockParams& params, double duration, const std::string& filename) {
        m_params = params;
        m_sim_duration = duration;
        m_output_filename = filename;
    }
    
    void setup() {
        std::cout << "\n=== Clock Generation Testbench ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Clock type: " << ClockTypeToString(m_params.type) << std::endl;
        std::cout << "  Frequency: " << m_params.frequency / 1e9 << " GHz" << std::endl;
        std::cout << "  Simulation duration: " << m_sim_duration * 1e6 << " us" << std::endl;
        std::cout << "  Output file: " << m_output_filename << std::endl;
        
        // Create modules
        std::cout << "\nCreating modules..." << std::endl;
        clk_gen = new ClockGenerationTdf("clk_gen", m_params);
        analyzer = new PhaseAnalyzer("analyzer");
        
        // Connect signals
        std::cout << "Connecting signals..." << std::endl;
        clk_gen->clk_phase(sig_phase);
        analyzer->phase_in(sig_phase);
        
        // Create trace file
        std::cout << "Creating trace file..." << std::endl;
        tf = sca_util::sca_create_tabular_trace_file(m_output_filename.c_str());
        sca_util::sca_trace(tf, sig_phase, "clk_phase");
    }
    
    void run() {
        std::cout << "\nStarting simulation..." << std::endl;
        sc_core::sc_start(m_sim_duration, sc_core::SC_SEC);
        std::cout << "Simulation completed." << std::endl;
    }
    
    void analyze() {
        analyzer->print_statistics();
        
        // Calculate expected values
        double expected_cycles = m_params.frequency * m_sim_duration;
        double expected_increment = 2.0 * M_PI / 100.0;
        
        std::cout << "=== Verification ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Expected cycles: " << expected_cycles << std::endl;
        std::cout << "  Actual cycles: " << analyzer->m_wrap_count << std::endl;
        std::cout << "  Cycle error: " 
                  << std::abs(analyzer->m_wrap_count - expected_cycles) / expected_cycles * 100.0 
                  << " %" << std::endl;
        
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Expected increment: " << expected_increment << " rad" << std::endl;
        std::cout << "  Actual increment: " << analyzer->get_mean_increment() << " rad" << std::endl;
        std::cout << "  Increment error: " 
                  << std::abs(analyzer->get_mean_increment() - expected_increment) 
                  << " rad" << std::endl;
        
        // Pass/Fail criteria
        bool phase_range_ok = (analyzer->m_min_phase >= 0.0) && 
                              (analyzer->m_max_phase < 2.0 * M_PI + 0.01);
        bool cycle_count_ok = std::abs(analyzer->m_wrap_count - expected_cycles) < 2;
        bool increment_ok = std::abs(analyzer->get_mean_increment() - expected_increment) < 1e-10;
        
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "  Phase range: " << (phase_range_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Cycle count: " << (cycle_count_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Phase increment: " << (increment_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Overall: " << ((phase_range_ok && cycle_count_ok && increment_ok) ? "PASS" : "FAIL") << std::endl;
        std::cout << "====================\n" << std::endl;
    }
    
    void cleanup() {
        if (tf) {
            sca_util::sca_close_tabular_trace_file(tf);
            tf = nullptr;
        }
        delete clk_gen;
        delete analyzer;
        
        std::cout << "Trace file saved: " << m_output_filename << std::endl;
    }
};

// ============================================================================
// Test Scenarios
// ============================================================================

void run_scenario_ideal_basic(ClockGenerationTestbench& tb) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Scenario: IDEAL Clock Basic Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 40e9;  // 40 GHz
    
    tb.configure(params, 1e-9, "clock_ideal_basic.dat");  // 1 ns
    tb.setup();
    tb.run();
    tb.analyze();
    tb.cleanup();
}

void run_scenario_frequency_sweep(ClockGenerationTestbench& tb) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Scenario: Frequency Sweep Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<double> frequencies = {10e9, 20e9, 40e9, 80e9};
    
    for (double freq : frequencies) {
        std::cout << "\n--- Testing " << freq / 1e9 << " GHz ---" << std::endl;
        
        ClockParams params;
        params.type = ClockType::IDEAL;
        params.frequency = freq;
        
        std::string filename = "clock_" + std::to_string(static_cast<int>(freq / 1e9)) + "GHz.dat";
        
        // Run for 10 cycles
        double duration = 10.0 / freq;
        
        tb.configure(params, duration, filename);
        tb.setup();
        tb.run();
        tb.analyze();
        tb.cleanup();
    }
}

void run_scenario_long_simulation(ClockGenerationTestbench& tb) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Scenario: Long Simulation Stability Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    ClockParams params;
    params.type = ClockType::IDEAL;
    params.frequency = 10e9;  // 10 GHz (lower freq for longer time)
    
    tb.configure(params, 100e-9, "clock_long_sim.dat");  // 100 ns = 1000 cycles
    tb.setup();
    tb.run();
    tb.analyze();
    tb.cleanup();
}

void run_scenario_clock_type_comparison(ClockGenerationTestbench& tb) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Scenario: Clock Type Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<ClockType> types = {ClockType::IDEAL, ClockType::PLL, ClockType::ADPLL};
    std::vector<std::string> type_names = {"ideal", "pll", "adpll"};
    
    for (size_t i = 0; i < types.size(); ++i) {
        std::cout << "\n--- Testing " << type_names[i] << " mode ---" << std::endl;
        
        ClockParams params;
        params.type = types[i];
        params.frequency = 40e9;
        
        std::string filename = "clock_" + type_names[i] + ".dat";
        
        tb.configure(params, 1e-9, filename);
        tb.setup();
        tb.run();
        tb.analyze();
        tb.cleanup();
    }
}

// ============================================================================
// Main Function
// ============================================================================

int sc_main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Clock Generation Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    int scenario = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "basic" || arg == "0") scenario = 0;
        else if (arg == "sweep" || arg == "1") scenario = 1;
        else if (arg == "long" || arg == "2") scenario = 2;
        else if (arg == "compare" || arg == "3") scenario = 3;
        else if (arg == "all" || arg == "4") scenario = 4;
        else {
            std::cout << "Usage: " << argv[0] << " [scenario]" << std::endl;
            std::cout << "  basic  (0) - Basic IDEAL clock test" << std::endl;
            std::cout << "  sweep  (1) - Frequency sweep test" << std::endl;
            std::cout << "  long   (2) - Long simulation stability test" << std::endl;
            std::cout << "  compare(3) - Clock type comparison" << std::endl;
            std::cout << "  all    (4) - Run all scenarios" << std::endl;
            return 1;
        }
    }
    
    // Create testbench
    ClockGenerationTestbench tb("tb");
    
    // Run selected scenario
    switch (scenario) {
        case 0:
            run_scenario_ideal_basic(tb);
            break;
        case 1:
            run_scenario_frequency_sweep(tb);
            break;
        case 2:
            run_scenario_long_simulation(tb);
            break;
        case 3:
            run_scenario_clock_type_comparison(tb);
            break;
        case 4:
            run_scenario_ideal_basic(tb);
            run_scenario_frequency_sweep(tb);
            run_scenario_long_simulation(tb);
            run_scenario_clock_type_comparison(tb);
            break;
        default:
            run_scenario_ideal_basic(tb);
            break;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
