/**
 * @file channel_sparam_tb.cpp
 * @brief Channel S-Parameter Testbench for End-to-End Integration Testing
 * 
 * This testbench validates the ChannelSParamTdf module by:
 * 1. Generating a PRBS test signal (broadband for frequency response coverage)
 * 2. Passing it through the channel model
 * 3. Recording input and output waveforms
 * 4. Saving data for Python verification script
 * 
 * Usage:
 *   ./channel_sparam_tb [config_file] [duration_ns]
 * 
 * Examples:
 *   ./channel_sparam_tb                          # Use default simple channel
 *   ./channel_sparam_tb config/channel_config.json  # Use S-parameter config
 *   ./channel_sparam_tb config/channel_config.json 1000  # 1000 ns simulation
 */

#include <systemc-ams>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "ams/channel_sparam.h"
#include "ams/wave_generation.h"
#include "common/parameters.h"

using namespace serdes;

// ============================================================================
// Signal Recorder for Channel Verification
// ============================================================================

/**
 * @brief Records input and output signals for channel verification
 * 
 * Captures:
 * - Input signal (from WaveGen)
 * - Output signal (from Channel)
 * - Timestamps
 * 
 * Output format compatible with Python verification script
 */
class ChannelSignalRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_signal;
    sca_tdf::sca_in<double> out_signal;
    
    ChannelSignalRecorder(sc_core::sc_module_name nm, double sample_rate)
        : sca_tdf::sca_module(nm)
        , in_signal("in_signal")
        , out_signal("out_signal")
        , m_sample_rate(sample_rate)
    {}
    
    void set_attributes() override {
        set_timestep(1.0 / m_sample_rate, sc_core::SC_SEC);
        in_signal.set_rate(1);
        out_signal.set_rate(1);
    }
    
    void processing() override {
        m_time.push_back(sc_core::sc_time_stamp().to_seconds());
        m_input.push_back(in_signal.read());
        m_output.push_back(out_signal.read());
    }
    
    /**
     * @brief Save recorded data to CSV file
     * @param filename Output file path
     */
    void save_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        // Write header
        file << "time_s,input_V,output_V\n";
        file << std::scientific << std::setprecision(12);
        
        // Write data
        for (size_t i = 0; i < m_time.size(); ++i) {
            file << m_time[i] << "," 
                 << m_input[i] << ","
                 << m_output[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved " << m_time.size() << " samples to " << filename << std::endl;
    }
    
    /**
     * @brief Save data in EyeAnalyzer-compatible format (.dat)
     * @param filename Output file path
     */
    void save_dat(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        // EyeAnalyzer format: time voltage
        file << std::scientific << std::setprecision(12);
        
        for (size_t i = 0; i < m_time.size(); ++i) {
            file << m_time[i] << " " << m_output[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved EyeAnalyzer format to " << filename << std::endl;
    }
    
    /**
     * @brief Get recorded time array
     */
    const std::vector<double>& get_time_array() const { return m_time; }
    
    /**
     * @brief Get recorded input signal
     */
    const std::vector<double>& get_input() const { return m_input; }
    
    /**
     * @brief Get recorded output signal
     */
    const std::vector<double>& get_output() const { return m_output; }
    
    /**
     * @brief Get sample rate
     */
    double get_sample_rate() const { return m_sample_rate; }
    
private:
    double m_sample_rate;
    std::vector<double> m_time;
    std::vector<double> m_input;
    std::vector<double> m_output;
};

// ============================================================================
// Channel Testbench Top Module
// ============================================================================

SC_MODULE(ChannelSparamTestbench) {
    // Sub-modules
    WaveGenerationTdf* wave_gen;
    ChannelSParamTdf* channel;
    ChannelSignalRecorder* recorder;
    
    // Signals
    sca_tdf::sca_signal<double> sig_input;
    sca_tdf::sca_signal<double> sig_output;
    
    // Parameters
    WaveGenParams m_wave_params;
    ChannelParams m_channel_params;
    ChannelExtendedParams m_ext_params;
    
    double m_sample_rate;
    double m_ui;
    double m_sim_duration;
    std::string m_config_file;
    std::string m_output_prefix;
    bool m_use_sparam_config;
    
    SC_CTOR(ChannelSparamTestbench)
        : sig_input("sig_input")
        , sig_output("sig_output")
        , m_sample_rate(100e9)      // 100 GHz default
        , m_ui(100e-12)             // 100 ps (10 Gbps)
        , m_sim_duration(1000e-9)   // 1 us default
        , m_config_file("")
        , m_output_prefix("channel_sparam")
        , m_use_sparam_config(false)
    {}
    
    // ========================================================================
    // Configuration Methods
    // ========================================================================
    
    void configure(int argc, char* argv[]) {
        // Parse command line arguments
        if (argc > 1) {
            m_config_file = argv[1];
            m_use_sparam_config = true;
        }
        
        if (argc > 2) {
            // Duration in ns
            m_sim_duration = std::atof(argv[2]) * 1e-9;
        }
        
        // Configure wave generation
        configure_wavegen();
        
        // Configure channel (this detects method from config)
        configure_channel();
        
        // Set output prefix based on method
        if (m_use_sparam_config) {
            switch (m_ext_params.method) {
                case ChannelMethod::POLE_RESIDUE:
                    m_output_prefix = "channel_pr";
                    break;
                case ChannelMethod::IMPULSE:
                    m_output_prefix = "channel_impulse";
                    break;
                case ChannelMethod::RATIONAL:
                    m_output_prefix = "channel_rational";
                    break;
                default:
                    m_output_prefix = "channel_sparam_config";
                    break;
            }
        }
        
        std::cout << "Testbench Configuration:" << std::endl;
        std::cout << "  Sample rate: " << m_sample_rate / 1e9 << " GHz" << std::endl;
        std::cout << "  UI: " << m_ui * 1e12 << " ps" << std::endl;
        std::cout << "  Data rate: " << 1.0 / m_ui / 1e9 << " Gbps" << std::endl;
        std::cout << "  Duration: " << m_sim_duration * 1e9 << " ns" << std::endl;
        std::cout << "  Config file: " << (m_use_sparam_config ? m_config_file : "none (simple model)") << std::endl;
        std::cout << "  Method: " << get_method_string(m_ext_params.method) << std::endl;
    }
    
    /**
     * @brief Get string representation of channel method
     */
    const char* get_method_string(ChannelMethod method) {
        switch (method) {
            case ChannelMethod::SIMPLE: return "SIMPLE";
            case ChannelMethod::RATIONAL: return "RATIONAL";
            case ChannelMethod::IMPULSE: return "IMPULSE";
            case ChannelMethod::POLE_RESIDUE: return "POLE_RESIDUE";
            default: return "UNKNOWN";
        }
    }
    
    void configure_wavegen() {
        // PRBS31 for broadband frequency coverage
        m_wave_params.type = PRBSType::PRBS31;
        m_wave_params.single_pulse = 0.0;  // PRBS mode, not pulse
        m_wave_params.jitter.RJ_sigma = 0.0;  // No jitter for clean measurement
        m_wave_params.jitter.SJ_freq.clear();
        m_wave_params.jitter.SJ_pp.clear();
    }
    
    void configure_channel() {
        // Default simple channel parameters
        m_channel_params.touchstone = "";
        m_channel_params.ports = 2;
        m_channel_params.crosstalk = false;
        m_channel_params.bidirectional = false;
        m_channel_params.attenuation_db = 6.0;    // 6 dB attenuation
        m_channel_params.bandwidth_hz = 15e9;     // 15 GHz bandwidth
        
        // Extended parameters for S-parameter modeling
        if (m_use_sparam_config) {
            // Detect method from config file content
            m_ext_params.method = detect_method_from_config(m_config_file);
            m_ext_params.config_file = m_config_file;
            m_ext_params.fs = m_sample_rate;
        } else {
            m_ext_params.method = ChannelMethod::SIMPLE;    // Simple low-pass model
        }
    }
    
    /**
     * @brief Detect channel method from config file
     */
    ChannelMethod detect_method_from_config(const std::string& config_file) {
        // Read config file to detect method
        std::ifstream file(config_file);
        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open config file " << config_file 
                      << ", using RATIONAL method" << std::endl;
            return ChannelMethod::RATIONAL;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();
        
        // Check for method field
        if (content.find("\"method\": \"POLE_RESIDUE\"") != std::string::npos ||
            content.find("\"method\":\"POLE_RESIDUE\"") != std::string::npos) {
            std::cout << "  Detected POLE_RESIDUE method from config" << std::endl;
            return ChannelMethod::POLE_RESIDUE;
        } else if (content.find("\"method\": \"IMPULSE\"") != std::string::npos ||
                   content.find("\"method\":\"IMPULSE\"") != std::string::npos) {
            std::cout << "  Detected IMPULSE method from config" << std::endl;
            return ChannelMethod::IMPULSE;
        } else if (content.find("\"pole_residue\"") != std::string::npos) {
            std::cout << "  Detected POLE_RESIDUE method from pole_residue field" << std::endl;
            return ChannelMethod::POLE_RESIDUE;
        } else if (content.find("\"impulse_response\"") != std::string::npos) {
            std::cout << "  Detected IMPULSE method from impulse_response field" << std::endl;
            return ChannelMethod::IMPULSE;
        } else if (content.find("\"rational\"") != std::string::npos ||
                   content.find("\"numerator\"") != std::string::npos) {
            std::cout << "  Detected RATIONAL method from rational/numerator field" << std::endl;
            return ChannelMethod::RATIONAL;
        }
        
        std::cout << "  Using default RATIONAL method" << std::endl;
        return ChannelMethod::RATIONAL;
    }
    
    // ========================================================================
    // Build Method
    // ========================================================================
    
    void build() {
        std::cout << "\nBuilding channel testbench..." << std::endl;
        
        // Create WaveGen
        wave_gen = new WaveGenerationTdf("wave_gen", m_wave_params, m_sample_rate, m_ui, 12345);
        
        // Create Channel with appropriate constructor
        if (m_use_sparam_config) {
            std::cout << "  Using S-parameter configuration: " << m_config_file << std::endl;
            channel = new ChannelSParamTdf("channel", m_channel_params, m_ext_params);
        } else {
            std::cout << "  Using simple channel model" << std::endl;
            channel = new ChannelSParamTdf("channel", m_channel_params);
        }
        
        // Create recorder
        recorder = new ChannelSignalRecorder("recorder", m_sample_rate);
        
        // Connect signals
        wave_gen->out(sig_input);
        channel->in(sig_input);
        channel->out(sig_output);
        recorder->in_signal(sig_input);
        recorder->out_signal(sig_output);
        
        std::cout << "Testbench built successfully." << std::endl;
    }
    
    // ========================================================================
    // Run Method
    // ========================================================================
    
    void run() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Channel S-Parameter Testbench" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Running simulation for " << m_sim_duration * 1e9 << " ns..." << std::endl;
        
        sc_core::sc_start(m_sim_duration, sc_core::SC_SEC);
        
        std::cout << "Simulation completed." << std::endl;
    }
    
    // ========================================================================
    // Save Results
    // ========================================================================
    
    void save_results() {
        std::cout << "\nSaving results..." << std::endl;
        
        // Save CSV for Python analysis
        std::string csv_file = m_output_prefix + "_waveform.csv";
        recorder->save_csv(csv_file);
        
        // Save .dat file for EyeAnalyzer compatibility
        std::string dat_file = m_output_prefix + "_output.dat";
        recorder->save_dat(dat_file);
        
        // Save metadata JSON
        save_metadata();
    }
    
    void save_metadata() {
        std::string meta_file = m_output_prefix + "_metadata.json";
        std::ofstream file(meta_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open metadata file " << meta_file << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(12);
        file << "{\n";
        file << "  \"testbench\": \"channel_sparam\",\n";
        file << "  \"sample_rate_hz\": " << m_sample_rate << ",\n";
        file << "  \"ui_s\": " << m_ui << ",\n";
        file << "  \"data_rate_bps\": " << (1.0 / m_ui) << ",\n";
        file << "  \"simulation_duration_s\": " << m_sim_duration << ",\n";
        file << "  \"num_samples\": " << recorder->get_time_array().size() << ",\n";
        file << "  \"waveform_file\": \"" << m_output_prefix << "_waveform.csv\",\n";
        file << "  \"output_dat_file\": \"" << m_output_prefix << "_output.dat\",\n";
        file << "  \"config_file\": \"" << (m_use_sparam_config ? m_config_file : "") << "\",\n";
        file << "  \"channel_model\": \"" << (m_use_sparam_config ? get_method_string(m_ext_params.method) : "simple_lpf") << "\",\n";
        
        // Channel parameters
        file << "  \"channel\": {\n";
        file << "    \"attenuation_db\": " << m_channel_params.attenuation_db << ",\n";
        file << "    \"bandwidth_hz\": " << m_channel_params.bandwidth_hz << "\n";
        file << "  }\n";
        
        file << "}\n";
        file.close();
        
        std::cout << "Saved metadata to " << meta_file << std::endl;
    }
    
    // ========================================================================
    // Print Summary
    // ========================================================================
    
    void print_summary() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Channel Testbench Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        
        const auto& input = recorder->get_input();
        const auto& output = recorder->get_output();
        
        if (input.empty() || output.empty()) {
            std::cout << "No data recorded." << std::endl;
            return;
        }
        
        // Calculate statistics
        double in_min = *std::min_element(input.begin(), input.end());
        double in_max = *std::max_element(input.begin(), input.end());
        double out_min = *std::min_element(output.begin(), output.end());
        double out_max = *std::max_element(output.begin(), output.end());
        
        double in_pp = in_max - in_min;
        double out_pp = out_max - out_min;
        
        // Calculate RMS
        double in_rms = 0.0, out_rms = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            in_rms += input[i] * input[i];
            out_rms += output[i] * output[i];
        }
        in_rms = std::sqrt(in_rms / input.size());
        out_rms = std::sqrt(out_rms / output.size());
        
        std::cout << "Input Signal:" << std::endl;
        std::cout << "  Peak-to-peak: " << in_pp * 1000 << " mV" << std::endl;
        std::cout << "  RMS: " << in_rms * 1000 << " mV" << std::endl;
        
        std::cout << "\nOutput Signal:" << std::endl;
        std::cout << "  Peak-to-peak: " << out_pp * 1000 << " mV" << std::endl;
        std::cout << "  RMS: " << out_rms * 1000 << " mV" << std::endl;
        
        // Estimate attenuation
        if (in_rms > 0) {
            double gain = out_rms / in_rms;
            double gain_db = 20.0 * std::log10(gain);
            std::cout << "\nEstimated Channel Gain: " << gain_db << " dB" << std::endl;
        }
        
        std::cout << "\nOutput Files:" << std::endl;
        std::cout << "  " << m_output_prefix << "_waveform.csv" << std::endl;
        std::cout << "  " << m_output_prefix << "_output.dat" << std::endl;
        std::cout << "  " << m_output_prefix << "_metadata.json" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    ~ChannelSparamTestbench() {
        delete wave_gen;
        delete channel;
        delete recorder;
    }
};

// ============================================================================
// Main Function
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [config_file] [duration_ns]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  config_file   Path to JSON config file (optional)" << std::endl;
    std::cout << "  duration_ns   Simulation duration in nanoseconds (default: 1000)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog << "                           # Use simple channel model" << std::endl;
    std::cout << "  " << prog << " config/channel_config.json  # Use S-parameter config" << std::endl;
    std::cout << "  " << prog << " config/channel_config.json 500  # 500 ns simulation" << std::endl;
}

int sc_main(int argc, char* argv[]) {
    // Disable SystemC deprecation warnings
    sc_core::sc_report_handler::set_actions(
        "/IEEE_Std_1666/deprecated",
        sc_core::SC_DO_NOTHING
    );
    
    // Check for help
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Channel S-Parameter Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create and run testbench
    ChannelSparamTestbench tb("tb");
    tb.configure(argc, argv);
    tb.build();
    tb.run();
    tb.save_results();
    tb.print_summary();
    
    sc_core::sc_stop();
    
    std::cout << "\nTestbench completed successfully." << std::endl;
    return 0;
}
