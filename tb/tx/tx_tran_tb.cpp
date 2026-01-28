/**
 * @file tx_tran_tb.cpp
 * @brief Transient testbench for TX Top module
 * 
 * This testbench performs complete TX chain simulation:
 * - WaveGen → FFE → Mux → Driver → Output
 * - Optional channel processing using S-parameter model
 * 
 * Test scenarios:
 * 0. BASIC_OUTPUT     - Basic output waveform
 * 1. FFE_SWEEP        - FFE parameter sweep
 * 2. CHANNEL_TEST     - TX + Channel model
 * 
 * Usage:
 *   ./tx_tran_tb [scenario]
 * 
 * Examples:
 *   ./tx_tran_tb basic
 *   ./tx_tran_tb channel
 */

#include <systemc-ams>
#include "ams/tx_top.h"
#include "ams/wave_generation.h"
#include "ams/channel_sparam.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <cstring>

using namespace serdes;

// ============================================================================
// Test Scenarios
// ============================================================================

enum TestScenario {
    BASIC_OUTPUT = 0,
    FFE_SWEEP,
    CHANNEL_TEST
};

std::map<std::string, TestScenario> scenario_map = {
    {"basic", BASIC_OUTPUT},
    {"ffe_sweep", FFE_SWEEP},
    {"channel", CHANNEL_TEST},
    {"0", BASIC_OUTPUT},
    {"1", FFE_SWEEP},
    {"2", CHANNEL_TEST}
};

std::string get_scenario_name(TestScenario scenario) {
    switch (scenario) {
        case BASIC_OUTPUT: return "basic";
        case FFE_SWEEP: return "ffe_sweep";
        case CHANNEL_TEST: return "channel";
        default: return "unknown";
    }
}

// ============================================================================
// Constant VDD Source
// ============================================================================

class ConstVddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    ConstVddSource(sc_core::sc_module_name nm, double voltage = 1.0)
        : sca_tdf::sca_module(nm), out("out"), m_voltage(voltage) {}
    
    void set_attributes() override {
        set_timestep(10.0, sc_core::SC_PS);
    }
    
    void processing() override {
        out.write(m_voltage);
    }
    
private:
    double m_voltage;
};

// ============================================================================
// Differential to Single-ended Converter (for channel input)
// ============================================================================

class DiffToSingleTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    sca_tdf::sca_out<double> out;
    
    DiffToSingleTdf(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in_p("in_p"), in_n("in_n"), out("out") {}
    
    void set_attributes() override {}
    
    void processing() override {
        out.write(in_p.read() - in_n.read());
    }
};

// ============================================================================
// Signal Recorder
// ============================================================================

class SignalRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    std::vector<double> time_stamps;
    std::vector<double> samples_p;
    std::vector<double> samples_n;
    std::vector<double> samples_diff;
    
    SignalRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in_p("in_p"), in_n("in_n") {}
    
    void set_attributes() override {
        set_timestep(10.0, sc_core::SC_PS);
    }
    
    void processing() override {
        double vp = in_p.read();
        double vn = in_n.read();
        
        time_stamps.push_back(get_time().to_seconds());
        samples_p.push_back(vp);
        samples_n.push_back(vn);
        samples_diff.push_back(vp - vn);
    }
    
    void save_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        file << "time_s,out_p_V,out_n_V,out_diff_V\n";
        file << std::scientific << std::setprecision(9);
        
        for (size_t i = 0; i < time_stamps.size(); ++i) {
            file << time_stamps[i] << ","
                 << samples_p[i] << ","
                 << samples_n[i] << ","
                 << samples_diff[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved " << time_stamps.size() << " samples to " << filename << std::endl;
    }
};

// ============================================================================
// Single-ended Signal Recorder (for channel output)
// ============================================================================

class SingleRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    std::vector<double> time_stamps;
    std::vector<double> samples;
    
    SingleRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in("in") {}
    
    void set_attributes() override {
        set_timestep(10.0, sc_core::SC_PS);
    }
    
    void processing() override {
        time_stamps.push_back(get_time().to_seconds());
        samples.push_back(in.read());
    }
    
    void save_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        file << "time_s,rx_input_V\n";
        file << std::scientific << std::setprecision(9);
        
        for (size_t i = 0; i < time_stamps.size(); ++i) {
            file << time_stamps[i] << "," << samples[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved " << time_stamps.size() << " samples to " << filename << std::endl;
    }
};

// ============================================================================
// TX Transient Testbench (without channel)
// ============================================================================

SC_MODULE(TxTransientTestbench) {
    // Sub-modules
    WaveGenerationTdf* wavegen;
    ConstVddSource* vdd_src;
    TxTopModule* tx;
    SignalRecorder* recorder;
    
    // Signals
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_tx_out_p;
    sca_tdf::sca_signal<double> sig_tx_out_n;
    
    // Configuration
    WaveGenParams m_wave_params;
    TxParams m_tx_params;
    double m_sim_duration;
    std::string m_output_file;
    
    SC_CTOR(TxTransientTestbench)
        : sig_wavegen_out("sig_wavegen_out")
        , sig_vdd("sig_vdd")
        , sig_tx_out_p("sig_tx_out_p")
        , sig_tx_out_n("sig_tx_out_n")
        , m_sim_duration(1000e-9)
        , m_output_file("tx_output.csv")
    {
    }
    
    void configure(TestScenario scenario) {
        // Default wave parameters
        m_wave_params = WaveGenParams();
        m_wave_params.type = PRBSType::PRBS31;
        m_wave_params.single_pulse = 0.0;
        
        // Default TX parameters
        m_tx_params = TxParams();
        m_tx_params.ffe.taps = {0.0, 1.0, -0.25};  // De-emphasis
        m_tx_params.mux_lane = 0;
        m_tx_params.driver.dc_gain = 0.8;
        m_tx_params.driver.vswing = 0.8;
        m_tx_params.driver.vcm_out = 0.6;
        m_tx_params.driver.sat_mode = "soft";
        m_tx_params.driver.vlin = 0.5;
        m_tx_params.driver.poles = {50e9};
        
        std::string scenario_name = get_scenario_name(scenario);
        m_output_file = "tx_output_" + scenario_name + ".csv";
        
        switch (scenario) {
            case BASIC_OUTPUT:
                configure_basic();
                break;
            case FFE_SWEEP:
                configure_ffe_sweep();
                break;
            default:
                configure_basic();
                break;
        }
    }
    
    void configure_basic() {
        std::cout << "Configuring BASIC_OUTPUT test..." << std::endl;
        m_sim_duration = 2000e-9;  // 2us
    }
    
    void configure_ffe_sweep() {
        std::cout << "Configuring FFE_SWEEP test..." << std::endl;
        // Use heavier de-emphasis
        m_tx_params.ffe.taps = {0.0, 1.0, -0.35};
        m_sim_duration = 2000e-9;
    }
    
    void build() {
        // Create WaveGen
        wavegen = new WaveGenerationTdf("wavegen", m_wave_params, 100e9, 12345);
        
        // Create VDD source
        vdd_src = new ConstVddSource("vdd_src", 1.0);
        
        // Create TX
        tx = new TxTopModule("tx", m_tx_params);
        
        // Create recorder
        recorder = new SignalRecorder("recorder");
        
        // Connect signals
        wavegen->out(sig_wavegen_out);
        vdd_src->out(sig_vdd);
        
        tx->in(sig_wavegen_out);
        tx->vdd(sig_vdd);
        tx->out_p(sig_tx_out_p);
        tx->out_n(sig_tx_out_n);
        
        recorder->in_p(sig_tx_out_p);
        recorder->in_n(sig_tx_out_n);
    }
    
    void run() {
        std::cout << "Running simulation for " << m_sim_duration * 1e9 << " ns..." << std::endl;
        sc_core::sc_start(m_sim_duration, sc_core::SC_SEC);
    }
    
    void save_results() {
        recorder->save_to_csv(m_output_file);
        save_config("tx_config.json");
    }
    
    void save_config(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open config file " << filename << std::endl;
            return;
        }
        
        file << "{\n";
        file << "  \"wave\": {\n";
        file << "    \"type\": \"PRBS31\",\n";
        file << "    \"single_pulse\": " << m_wave_params.single_pulse << "\n";
        file << "  },\n";
        file << "  \"tx\": {\n";
        file << "    \"ffe\": {\n";
        file << "      \"taps\": [";
        for (size_t i = 0; i < m_tx_params.ffe.taps.size(); ++i) {
            file << m_tx_params.ffe.taps[i];
            if (i < m_tx_params.ffe.taps.size() - 1) file << ", ";
        }
        file << "]\n";
        file << "    },\n";
        file << "    \"mux_lane\": " << m_tx_params.mux_lane << ",\n";
        file << "    \"driver\": {\n";
        file << "      \"dc_gain\": " << m_tx_params.driver.dc_gain << ",\n";
        file << "      \"vswing\": " << m_tx_params.driver.vswing << ",\n";
        file << "      \"vcm_out\": " << m_tx_params.driver.vcm_out << ",\n";
        file << "      \"sat_mode\": \"" << m_tx_params.driver.sat_mode << "\",\n";
        file << "      \"vlin\": " << m_tx_params.driver.vlin << ",\n";
        file << "      \"poles\": [";
        for (size_t i = 0; i < m_tx_params.driver.poles.size(); ++i) {
            file << m_tx_params.driver.poles[i];
            if (i < m_tx_params.driver.poles.size() - 1) file << ", ";
        }
        file << "]\n";
        file << "    }\n";
        file << "  },\n";
        file << "  \"simulation\": {\n";
        file << "    \"duration_s\": " << m_sim_duration << ",\n";
        file << "    \"sample_rate_Hz\": 100e9\n";
        file << "  }\n";
        file << "}\n";
        
        file.close();
        std::cout << "Saved configuration to " << filename << std::endl;
    }
    
    void print_summary() {
        if (recorder->samples_diff.empty()) return;
        
        // Calculate statistics
        size_t start = recorder->samples_diff.size() / 10;
        double min_val = recorder->samples_diff[start];
        double max_val = recorder->samples_diff[start];
        double sum = 0.0;
        
        for (size_t i = start; i < recorder->samples_diff.size(); ++i) {
            if (recorder->samples_diff[i] < min_val) min_val = recorder->samples_diff[i];
            if (recorder->samples_diff[i] > max_val) max_val = recorder->samples_diff[i];
            sum += recorder->samples_diff[i];
        }
        
        double mean = sum / (recorder->samples_diff.size() - start);
        double pp = max_val - min_val;
        
        std::cout << "\n=== TX Output Summary ===" << std::endl;
        std::cout << "  Peak-to-peak (diff): " << pp * 1000 << " mV" << std::endl;
        std::cout << "  Max (diff): " << max_val * 1000 << " mV" << std::endl;
        std::cout << "  Min (diff): " << min_val * 1000 << " mV" << std::endl;
        std::cout << "  Mean (diff): " << mean * 1000 << " mV" << std::endl;
        std::cout << "  Samples recorded: " << recorder->samples_diff.size() << std::endl;
    }
};

// ============================================================================
// TX + Channel Testbench
// ============================================================================

SC_MODULE(TxChannelTestbench) {
    // Sub-modules
    WaveGenerationTdf* wavegen;
    ConstVddSource* vdd_src;
    TxTopModule* tx;
    SignalRecorder* tx_recorder;
    DiffToSingleTdf* d2s;
    ChannelSParamTdf* channel;
    SingleRecorder* rx_recorder;
    
    // Signals
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_tx_out_p;
    sca_tdf::sca_signal<double> sig_tx_out_n;
    sca_tdf::sca_signal<double> sig_channel_in;
    sca_tdf::sca_signal<double> sig_channel_out;
    
    // Configuration
    WaveGenParams m_wave_params;
    TxParams m_tx_params;
    ChannelParams m_channel_params;
    double m_sim_duration;
    
    SC_CTOR(TxChannelTestbench)
        : sig_wavegen_out("sig_wavegen_out")
        , sig_vdd("sig_vdd")
        , sig_tx_out_p("sig_tx_out_p")
        , sig_tx_out_n("sig_tx_out_n")
        , sig_channel_in("sig_channel_in")
        , sig_channel_out("sig_channel_out")
        , m_sim_duration(2000e-9)
    {
    }
    
    void configure() {
        std::cout << "Configuring TX + Channel test..." << std::endl;
        
        // Wave parameters
        m_wave_params = WaveGenParams();
        m_wave_params.type = PRBSType::PRBS31;
        
        // TX parameters
        m_tx_params = TxParams();
        m_tx_params.ffe.taps = {0.0, 1.0, -0.25};
        m_tx_params.driver.dc_gain = 0.8;
        m_tx_params.driver.vswing = 0.8;
        m_tx_params.driver.vcm_out = 0.6;
        m_tx_params.driver.sat_mode = "soft";
        m_tx_params.driver.poles = {50e9};
        
        // Channel parameters (simple model)
        m_channel_params = ChannelParams();
        m_channel_params.attenuation_db = 10.0;
        m_channel_params.bandwidth_hz = 20e9;
    }
    
    void build() {
        // Create WaveGen
        wavegen = new WaveGenerationTdf("wavegen", m_wave_params, 100e9, 12345);
        
        // Create VDD source
        vdd_src = new ConstVddSource("vdd_src", 1.0);
        
        // Create TX
        tx = new TxTopModule("tx", m_tx_params);
        
        // Create TX recorder
        tx_recorder = new SignalRecorder("tx_recorder");
        
        // Create differential to single-ended converter
        d2s = new DiffToSingleTdf("d2s");
        
        // Create channel
        channel = new ChannelSParamTdf("channel", m_channel_params);
        
        // Create RX recorder
        rx_recorder = new SingleRecorder("rx_recorder");
        
        // Connect signals
        wavegen->out(sig_wavegen_out);
        vdd_src->out(sig_vdd);
        
        tx->in(sig_wavegen_out);
        tx->vdd(sig_vdd);
        tx->out_p(sig_tx_out_p);
        tx->out_n(sig_tx_out_n);
        
        tx_recorder->in_p(sig_tx_out_p);
        tx_recorder->in_n(sig_tx_out_n);
        
        d2s->in_p(sig_tx_out_p);
        d2s->in_n(sig_tx_out_n);
        d2s->out(sig_channel_in);
        
        channel->in(sig_channel_in);
        channel->out(sig_channel_out);
        
        rx_recorder->in(sig_channel_out);
    }
    
    void run() {
        std::cout << "Running TX + Channel simulation for " << m_sim_duration * 1e9 << " ns..." << std::endl;
        sc_core::sc_start(m_sim_duration, sc_core::SC_SEC);
    }
    
    void save_results() {
        tx_recorder->save_to_csv("tx_output_channel.csv");
        rx_recorder->save_to_csv("rx_input.csv");
    }
    
    void print_summary() {
        if (tx_recorder->samples_diff.empty() || rx_recorder->samples.empty()) return;
        
        // TX output statistics
        size_t start = tx_recorder->samples_diff.size() / 10;
        double tx_min = tx_recorder->samples_diff[start];
        double tx_max = tx_recorder->samples_diff[start];
        for (size_t i = start; i < tx_recorder->samples_diff.size(); ++i) {
            if (tx_recorder->samples_diff[i] < tx_min) tx_min = tx_recorder->samples_diff[i];
            if (tx_recorder->samples_diff[i] > tx_max) tx_max = tx_recorder->samples_diff[i];
        }
        
        // RX input statistics
        start = rx_recorder->samples.size() / 10;
        double rx_min = rx_recorder->samples[start];
        double rx_max = rx_recorder->samples[start];
        for (size_t i = start; i < rx_recorder->samples.size(); ++i) {
            if (rx_recorder->samples[i] < rx_min) rx_min = rx_recorder->samples[i];
            if (rx_recorder->samples[i] > rx_max) rx_max = rx_recorder->samples[i];
        }
        
        std::cout << "\n=== TX + Channel Summary ===" << std::endl;
        std::cout << "TX Output:" << std::endl;
        std::cout << "  Peak-to-peak: " << (tx_max - tx_min) * 1000 << " mV" << std::endl;
        std::cout << "RX Input (after channel):" << std::endl;
        std::cout << "  Peak-to-peak: " << (rx_max - rx_min) * 1000 << " mV" << std::endl;
        std::cout << "  Attenuation: " << 20 * std::log10((rx_max - rx_min) / (tx_max - tx_min)) << " dB" << std::endl;
    }
};

// ============================================================================
// Main Function
// ============================================================================

int sc_main(int argc, char* argv[]) {
    // Parse command line
    TestScenario scenario = BASIC_OUTPUT;
    
    if (argc > 1) {
        std::string arg = argv[1];
        auto it = scenario_map.find(arg);
        if (it != scenario_map.end()) {
            scenario = it->second;
        } else {
            std::cout << "Unknown scenario: " << arg << std::endl;
            std::cout << "Available scenarios: basic, ffe_sweep, channel" << std::endl;
            return 1;
        }
    }
    
    std::cout << "=== TX Transient Testbench ===" << std::endl;
    std::cout << "Scenario: " << get_scenario_name(scenario) << std::endl;
    
    if (scenario == CHANNEL_TEST) {
        // TX + Channel testbench
        TxChannelTestbench tb("tb");
        tb.configure();
        tb.build();
        tb.run();
        tb.save_results();
        tb.print_summary();
    } else {
        // TX only testbench
        TxTransientTestbench tb("tb");
        tb.configure(scenario);
        tb.build();
        tb.run();
        tb.save_results();
        tb.print_summary();
    }
    
    sc_core::sc_stop();
    
    std::cout << "\nTestbench completed successfully." << std::endl;
    return 0;
}
