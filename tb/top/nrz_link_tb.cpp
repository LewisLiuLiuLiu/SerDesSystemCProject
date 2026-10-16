/**
 * @file nrz_link_tb.cpp
 * @brief 10Gbps NRZ 差分链路测试台
 * 
 * 特点:
 * - 使用 NrzLinkConfig 统一参数结构
 * - 差分信道直连 (无 DiffToSingle 转换)
 * - 测试台内直接组装子模块
 */

#include <systemc-ams>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "nrz_link_config.h"

#include "ams/wave_generation.h"
#include "ams/tx_top.h"
#include "ams/channel_sparam.h"
#include "ams/rx_top.h"
#include "de/config_loader.h"

using namespace serdes;

// ============================================================================
// Signal Recorder (眼图数据记录)
// ============================================================================

class EyeDataRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    EyeDataRecorder(sc_core::sc_module_name nm, const std::string& name)
        : sca_tdf::sca_module(nm)
        , in_p("in_p"), in_n("in_n")
        , m_name(name) {}
    
    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
    }
    
    void processing() override {
        double time = get_time().to_seconds();
        double vp = in_p.read();
        double vn = in_n.read();
        double vdiff = vp - vn;
        
        m_time.push_back(time);
        m_voltage_p.push_back(vp);
        m_voltage_n.push_back(vn);
        m_voltage_diff.push_back(vdiff);
    }
    
    void save_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }
        
        file << std::scientific << std::setprecision(9);
        file << "time_s,voltage_p,voltage_n,voltage_diff\n";
        
        for (size_t i = 0; i < m_time.size(); ++i) {
            file << m_time[i] << "," 
                 << m_voltage_p[i] << "," 
                 << m_voltage_n[i] << ","
                 << m_voltage_diff[i] << "\n";
        }
        
        file.close();
        std::cout << "Saved " << m_time.size() << " samples to " << filename << std::endl;
    }
    
    SignalStats get_diff_stats() const {
        SignalStats stats;
        if (m_voltage_diff.empty()) return stats;
        
        size_t start = m_voltage_diff.size() / 10;  // 跳过前10%
        
        stats.min_val = *std::min_element(m_voltage_diff.begin() + start, m_voltage_diff.end());
        stats.max_val = *std::max_element(m_voltage_diff.begin() + start, m_voltage_diff.end());
        stats.peak_to_peak = stats.max_val - stats.min_val;
        stats.sample_count = m_voltage_diff.size() - start;
        
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = start; i < m_voltage_diff.size(); ++i) {
            sum += m_voltage_diff[i];
            sum_sq += m_voltage_diff[i] * m_voltage_diff[i];
        }
        stats.mean_val = sum / stats.sample_count;
        stats.rms_val = std::sqrt(sum_sq / stats.sample_count);
        
        return stats;
    }
    
private:
    std::string m_name;
    std::vector<double> m_time;
    std::vector<double> m_voltage_p;
    std::vector<double> m_voltage_n;
    std::vector<double> m_voltage_diff;
};

// ============================================================================
// Data Output Recorder (数据输出记录)
// ============================================================================

class DataRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;

    DataRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), in("in") {}

    void set_attributes() override { in.set_rate(1); }

    void processing() override {
        m_time.push_back(get_time().to_seconds());
        m_data.push_back(in.read());
    }

    void save_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "time_s,data\n";
        file << std::scientific << std::setprecision(9);
        for (size_t i = 0; i < m_time.size(); ++i) {
            file << m_time[i] << "," << m_data[i] << "\n";
        }
        file.close();
        std::cout << "Saved " << m_time.size() << " data samples to " << filename << std::endl;
    }

private:
    std::vector<double> m_time;
    std::vector<double> m_data;
};

// ============================================================================
// DE to TDF Bridge for DFE Tap Monitoring (DE域到TDF域桥接)
// Uses sca_de::sca_in to read DE signals from within TDF module
// ============================================================================

class DeToTdfTapBridge : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out1;
    sca_tdf::sca_out<double> out2;
    sca_tdf::sca_out<double> out3;
    sca_tdf::sca_out<double> out4;
    sca_tdf::sca_out<double> out5;

    // DE inputs for reading DFE tap values (using sca_tdf::sca_de namespace)
    sca_tdf::sca_de::sca_in<double> de_tap1;
    sca_tdf::sca_de::sca_in<double> de_tap2;
    sca_tdf::sca_de::sca_in<double> de_tap3;
    sca_tdf::sca_de::sca_in<double> de_tap4;
    sca_tdf::sca_de::sca_in<double> de_tap5;

    DeToTdfTapBridge(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , out1("out1"), out2("out2"), out3("out3"), out4("out4"), out5("out5")
        , de_tap1("de_tap1"), de_tap2("de_tap2"), de_tap3("de_tap3")
        , de_tap4("de_tap4"), de_tap5("de_tap5") {}

    void set_attributes() override {
        // Set timestep to match other TDF modules (2 ps)
        this->set_timestep(2.0, sc_core::SC_PS);
        out1.set_rate(1);
        out2.set_rate(1);
        out3.set_rate(1);
        out4.set_rate(1);
        out5.set_rate(1);
    }

    void processing() override {
        // Read DE domain signals and write to TDF domain outputs
        out1.write(de_tap1.read());
        out2.write(de_tap2.read());
        out3.write(de_tap3.read());
        out4.write(de_tap4.read());
        out5.write(de_tap5.read());
    }
};

// ============================================================================
// Multi-Channel Data Recorder (多通道数据记录 - 用于DFE抽头、CDR相位等)
// ============================================================================

class MultiChannelRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in1;
    sca_tdf::sca_in<double> in2;
    sca_tdf::sca_in<double> in3;
    sca_tdf::sca_in<double> in4;
    sca_tdf::sca_in<double> in5;

    MultiChannelRecorder(sc_core::sc_module_name nm, const std::string& name,
                         const std::vector<std::string>& channel_names)
        : sca_tdf::sca_module(nm)
        , in1("in1"), in2("in2"), in3("in3"), in4("in4"), in5("in5")
        , m_name(name)
        , m_channel_names(channel_names)
        , m_num_channels(channel_names.size()) {}

    void set_attributes() override {
        in1.set_rate(1);
        in2.set_rate(1);
        in3.set_rate(1);
        in4.set_rate(1);
        in5.set_rate(1);
    }

    void processing() override {
        m_time.push_back(get_time().to_seconds());
        m_data[0].push_back(in1.read());
        if (m_num_channels > 1) m_data[1].push_back(in2.read());
        if (m_num_channels > 2) m_data[2].push_back(in3.read());
        if (m_num_channels > 3) m_data[3].push_back(in4.read());
        if (m_num_channels > 4) m_data[4].push_back(in5.read());
    }

    void save_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }

        file << std::scientific << std::setprecision(9);

        // Header
        file << "time_s";
        for (const auto& name : m_channel_names) {
            file << "," << name;
        }
        file << "\n";

        // Data
        for (size_t i = 0; i < m_time.size(); ++i) {
            file << m_time[i];
            for (size_t ch = 0; ch < m_num_channels; ++ch) {
                file << "," << m_data[ch][i];
            }
            file << "\n";
        }

        file.close();
        std::cout << "Saved " << m_time.size() << " samples (" << m_num_channels
                  << " channels) to " << filename << std::endl;
    }

private:
    std::string m_name;
    std::vector<std::string> m_channel_names;
    size_t m_num_channels;
    std::vector<double> m_time;
    std::vector<std::vector<double>> m_data{5};  // Max 5 channels
};

// ============================================================================
// NRZ Link Testbench
// ============================================================================

SC_MODULE(NrzLinkTb) {
    // 子模块
    ConstVddSource* vdd_src;
    ConstVddSource* dummy1_src;
    ConstVddSource* dummy2_src;
    ConstVddSource* dummy3_src;
    ConstVddSource* dummy4_src;
    WaveGenerationTdf* wavegen;
    DeToTdfTapBridge* dfe_tap_bridge;
    TxTopModule* tx;
    ChannelSParamTdf* channel;
    RxTopModule* rx;
    
    // 记录器
    EyeDataRecorder* rec_tx;
    EyeDataRecorder* rec_channel;
    EyeDataRecorder* rec_dfe;
    EyeDataRecorder* rec_ctle;
    EyeDataRecorder* rec_vga;
    DataRecorder* rec_data;
    MultiChannelRecorder* rec_dfe_taps;
    MultiChannelRecorder* rec_cdr_phase;
    
    // 内部信号
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_tx_out_p;
    sca_tdf::sca_signal<double> sig_tx_out_n;
    sca_tdf::sca_signal<double> sig_channel_out_p;
    sca_tdf::sca_signal<double> sig_channel_out_n;
    sca_tdf::sca_signal<double> sig_data_out;

    // DFE Tap signals (converted from DE to TDF for recording)
    sca_tdf::sca_signal<double> sig_dfe_tap1;
    sca_tdf::sca_signal<double> sig_dfe_tap2;
    sca_tdf::sca_signal<double> sig_dfe_tap3;
    sca_tdf::sca_signal<double> sig_dfe_tap4;
    sca_tdf::sca_signal<double> sig_dfe_tap5;

    // CDR Phase signal
    sca_tdf::sca_signal<double> sig_cdr_phase;

    // Dummy signals for unused recorder inputs
    sca_tdf::sca_signal<double> sig_dummy1;
    sca_tdf::sca_signal<double> sig_dummy2;
    sca_tdf::sca_signal<double> sig_dummy3;
    sca_tdf::sca_signal<double> sig_dummy4;
    
    // 配置
    NrzLinkConfig m_config;
    
    SC_CTOR(NrzLinkTb)
        : vdd_src(nullptr)
        , dummy1_src(nullptr), dummy2_src(nullptr), dummy3_src(nullptr), dummy4_src(nullptr)
        , wavegen(nullptr)
        , dfe_tap_bridge(nullptr)
        , tx(nullptr)
        , channel(nullptr), rx(nullptr)
        , rec_tx(nullptr), rec_channel(nullptr), rec_dfe(nullptr)
        , rec_ctle(nullptr), rec_vga(nullptr), rec_data(nullptr)
        , rec_dfe_taps(nullptr), rec_cdr_phase(nullptr)
        , sig_vdd("sig_vdd")
        , sig_wavegen_out("sig_wavegen_out")
        , sig_tx_out_p("sig_tx_out_p")
        , sig_tx_out_n("sig_tx_out_n")
        , sig_channel_out_p("sig_channel_out_p")
        , sig_channel_out_n("sig_channel_out_n")
        , sig_data_out("sig_data_out")
        , sig_dfe_tap1("sig_dfe_tap1"), sig_dfe_tap2("sig_dfe_tap2")
        , sig_dfe_tap3("sig_dfe_tap3"), sig_dfe_tap4("sig_dfe_tap4")
        , sig_dfe_tap5("sig_dfe_tap5")
        , sig_cdr_phase("sig_cdr_phase")
        , sig_dummy1("sig_dummy1"), sig_dummy2("sig_dummy2")
        , sig_dummy3("sig_dummy3"), sig_dummy4("sig_dummy4")
    {}
    
    void configure(const NrzLinkConfig& config) {
        m_config = config;
        m_config.sync_ui();  // 确保 UI 同步
    }
    
    void build() {
        std::cout << "\n=== Building NRZ Link Testbench ===" << std::endl;
        m_config.print_summary();
        
        double ui = m_config.ui();
        double fs = m_config.sample_rate();
        
        // ====== 创建子模块 ======
        
        std::cout << "[Build] Creating VDD source..." << std::endl;
        vdd_src = new ConstVddSource("vdd_src", 1.0, m_config.timestep_ps());

        std::cout << "[Build] Creating DE-to-TDF bridge for DFE taps..." << std::endl;
        dfe_tap_bridge = new DeToTdfTapBridge("dfe_tap_bridge");

        dummy1_src = new ConstVddSource("dummy1_src", 0.0, m_config.timestep_ps());
        dummy2_src = new ConstVddSource("dummy2_src", 0.0, m_config.timestep_ps());
        dummy3_src = new ConstVddSource("dummy3_src", 0.0, m_config.timestep_ps());
        dummy4_src = new ConstVddSource("dummy4_src", 0.0, m_config.timestep_ps());
        
        // Bind dummy sources to signals
        dummy1_src->out(sig_dummy1);
        dummy2_src->out(sig_dummy2);
        dummy3_src->out(sig_dummy3);
        dummy4_src->out(sig_dummy4);
        
        std::cout << "[Build] Creating WaveGen (PRBS)..." << std::endl;
        wavegen = new WaveGenerationTdf("wavegen", 
                                        m_config.wave, 
                                        fs, 
                                        ui, 
                                        m_config.seed);
        
        std::cout << "[Build] Creating TX (FFE + Driver)..." << std::endl;
        tx = new TxTopModule("tx", m_config.tx);
        
        std::cout << "[Build] Creating Channel (Differential MIMO)..." << std::endl;
        channel = new ChannelSParamTdf("channel", 
                                       m_config.channel,
                                       m_config.channel_ext);
        
        std::cout << "[Build] Creating RX (CTLE + VGA + DFE + CDR)..." << std::endl;
        rx = new RxTopModule("rx", m_config.rx, m_config.adaption);
        
        std::cout << "[Build] Creating recorders..." << std::endl;
        rec_tx = new EyeDataRecorder("rec_tx", "tx_out");
        rec_channel = new EyeDataRecorder("rec_channel", "channel_out");
        rec_dfe = new EyeDataRecorder("rec_dfe", "dfe_out");
        rec_ctle = new EyeDataRecorder("rec_ctle", "ctle_out");
        rec_vga = new EyeDataRecorder("rec_vga", "vga_out");
        rec_data = new DataRecorder("rec_data");

        // DFE taps recorder (5 channels) - NOTE: DFE taps are DE domain signals
        // For now we record constant 0 since DE-to-TDF conversion requires sca_de::sca_in/out bridge
        // The actual tap values can be obtained from the final summary or by reading DE signals directly
        rec_dfe_taps = new MultiChannelRecorder("rec_dfe_taps", "dfe_taps",
            std::vector<std::string>{"tap1", "tap2", "tap3", "tap4", "tap5"});

        // CDR phase recorder (1 channel for phase, can be extended)
        rec_cdr_phase = new MultiChannelRecorder("rec_cdr_phase", "cdr_phase",
            std::vector<std::string>{"phase"});


        
        // ====== 连接信号链 ======
        
        std::cout << "[Build] Connecting signal chain..." << std::endl;
        
        // VDD
        vdd_src->out(sig_vdd);
        
        // WaveGen -> TX
        wavegen->out(sig_wavegen_out);
        tx->in(sig_wavegen_out);
        tx->vdd(sig_vdd);
        tx->out_p(sig_tx_out_p);
        tx->out_n(sig_tx_out_n);
        
        // TX -> Channel (差分直连)
        // 注意: sc_vector 端口需要显式索引
        channel->in[0](sig_tx_out_p);
        channel->in[1](sig_tx_out_n);
        channel->out[0](sig_channel_out_p);
        channel->out[1](sig_channel_out_n);
        
        // Channel -> RX
        rx->in_p(sig_channel_out_p);
        rx->in_n(sig_channel_out_n);
        rx->vdd(sig_vdd);
        rx->data_out(sig_data_out);
        
        // 连接记录器
        rec_tx->in_p(sig_tx_out_p);
        rec_tx->in_n(sig_tx_out_n);

        rec_channel->in_p(sig_channel_out_p);
        rec_channel->in_n(sig_channel_out_n);

        // CTLE 输出 (从 RX 内部获取)
        rec_ctle->in_p(const_cast<sca_tdf::sca_signal<double>&>(rx->get_ctle_out_p_signal()));
        rec_ctle->in_n(const_cast<sca_tdf::sca_signal<double>&>(rx->get_ctle_out_n_signal()));

        // VGA 输出 (从 RX 内部获取)
        rec_vga->in_p(const_cast<sca_tdf::sca_signal<double>&>(rx->get_vga_out_p_signal()));
        rec_vga->in_n(const_cast<sca_tdf::sca_signal<double>&>(rx->get_vga_out_n_signal()));

        // DFE 输出 (从 RX 内部获取)
        rec_dfe->in_p(const_cast<sca_tdf::sca_signal<double>&>(rx->get_dfe_out_p_signal()));
        rec_dfe->in_n(const_cast<sca_tdf::sca_signal<double>&>(rx->get_dfe_out_n_signal()));

        // CDR 相位 - 连接到真实的 CDR 相位输出
        rec_cdr_phase->in1(const_cast<sca_tdf::sca_signal<double>&>(rx->get_cdr_phase_signal()));
        // 其他通道暂时不用
        rec_cdr_phase->in2(sig_dummy1);
        rec_cdr_phase->in3(sig_dummy2);
        rec_cdr_phase->in4(sig_dummy3);
        rec_cdr_phase->in5(sig_dummy4);

        // DFE Taps - connect via DE-to-TDF bridge
        // Connect DE inputs from RX to bridge
        dfe_tap_bridge->de_tap1(rx->get_dfe_tap_signal(1));
        dfe_tap_bridge->de_tap2(rx->get_dfe_tap_signal(2));
        dfe_tap_bridge->de_tap3(rx->get_dfe_tap_signal(3));
        dfe_tap_bridge->de_tap4(rx->get_dfe_tap_signal(4));
        dfe_tap_bridge->de_tap5(rx->get_dfe_tap_signal(5));

        // Connect TDF outputs from bridge to recorder
        dfe_tap_bridge->out1(sig_dfe_tap1);
        dfe_tap_bridge->out2(sig_dfe_tap2);
        dfe_tap_bridge->out3(sig_dfe_tap3);
        dfe_tap_bridge->out4(sig_dfe_tap4);
        dfe_tap_bridge->out5(sig_dfe_tap5);

        rec_dfe_taps->in1(sig_dfe_tap1);
        rec_dfe_taps->in2(sig_dfe_tap2);
        rec_dfe_taps->in3(sig_dfe_tap3);
        rec_dfe_taps->in4(sig_dfe_tap4);
        rec_dfe_taps->in5(sig_dfe_tap5);

        rec_data->in(sig_data_out);
        
        std::cout << "[Build] NRZ Link built successfully (differential direct connection)" << std::endl;
    }
    
    void run() {
        std::cout << "\n=== Running NRZ Link Simulation ===" << std::endl;
        std::cout << "Duration: " << m_config.sim_duration * 1e6 << " us ("
                  << m_config.sim_ui_count() << " UI)" << std::endl;
        
        sc_core::sc_start(m_config.sim_duration, sc_core::SC_SEC);
        
        std::cout << "Simulation completed." << std::endl;
    }
    
    void save_results() {
        std::cout << "\n=== Saving Results ===" << std::endl;

        std::string prefix = m_config.output_prefix;

        rec_tx->save_csv(prefix + "_tx.csv");
        rec_channel->save_csv(prefix + "_channel.csv");
        rec_ctle->save_csv(prefix + "_ctle.csv");
        rec_vga->save_csv(prefix + "_vga.csv");
        rec_dfe->save_csv(prefix + "_dfe.csv");
        rec_data->save_csv(prefix + "_data.csv");

        // Save DFE taps evolution
        rec_dfe_taps->save_csv(prefix + "_dfe_taps.csv");

        // Save CDR phase evolution
        rec_cdr_phase->save_csv(prefix + "_cdr_phase.csv");

        // 保存配置元数据
        save_metadata(prefix + "_metadata.json");
    }
    
    void save_metadata(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "{\n";
        file << "  \"data_rate_gbps\": " << m_config.data_rate / 1e9 << ",\n";
        file << "  \"ui_ps\": " << m_config.ui() * 1e12 << ",\n";
        file << "  \"sample_rate_ghz\": " << m_config.sample_rate() / 1e9 << ",\n";
        file << "  \"sim_duration_us\": " << m_config.sim_duration * 1e6 << ",\n";
        file << "  \"channel_method\": \"" 
             << (m_config.channel_ext.method == ChannelMethod::SIMPLE ? "SIMPLE" : "STATE_SPACE") 
             << "\",\n";
        file << "  \"channel_attenuation_db\": " << m_config.channel.attenuation_db << ",\n";
        file << "  \"ffe_taps\": " << m_config.tx.ffe.taps.size() << ",\n";
        file << "  \"dfe_taps\": " << m_config.rx.dfe_summer.tap_coeffs.size() << ",\n";
        file << "  \"adaption_enabled\": " << (m_config.adaption.agc.enabled ? "true" : "false") << "\n";
        file << "}\n";
        file.close();
        
        std::cout << "Saved metadata to " << filename << std::endl;
    }
    
    void print_summary() {
        std::cout << "\n+----------------------------------------------+" << std::endl;
        std::cout << "|           Simulation Summary                 |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;

        // TX 统计
        SignalStats tx_stats = rec_tx->get_diff_stats();
        std::cout << "| TX Output:                                   |" << std::endl;
        std::cout << "|   Peak-to-Peak: " << std::setw(10) << tx_stats.peak_to_peak * 1000
                  << " mV            |" << std::endl;

        // Channel 统计
        SignalStats ch_stats = rec_channel->get_diff_stats();
        double attn = 20 * std::log10(ch_stats.peak_to_peak / tx_stats.peak_to_peak);
        std::cout << "| Channel Output:                              |" << std::endl;
        std::cout << "|   Peak-to-Peak: " << std::setw(10) << ch_stats.peak_to_peak * 1000
                  << " mV            |" << std::endl;
        std::cout << "|   Attenuation:  " << std::setw(10) << attn
                  << " dB            |" << std::endl;

        // CTLE 统计
        SignalStats ctle_stats = rec_ctle->get_diff_stats();
        std::cout << "| CTLE Output:                                 |" << std::endl;
        std::cout << "|   Peak-to-Peak: " << std::setw(10) << ctle_stats.peak_to_peak * 1000
                  << " mV            |" << std::endl;

        // VGA 统计
        SignalStats vga_stats = rec_vga->get_diff_stats();
        std::cout << "| VGA Output:                                  |" << std::endl;
        std::cout << "|   Peak-to-Peak: " << std::setw(10) << vga_stats.peak_to_peak * 1000
                  << " mV            |" << std::endl;

        // DFE 统计
        SignalStats dfe_stats = rec_dfe->get_diff_stats();
        std::cout << "| DFE Output:                                  |" << std::endl;
        std::cout << "|   Peak-to-Peak: " << std::setw(10) << dfe_stats.peak_to_peak * 1000
                  << " mV            |" << std::endl;

        // DFE Tap 系数
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| DFE Tap Coefficients (Final):                |" << std::endl;
        for (int i = 1; i <= 5; ++i) {
            double tap = get_dfe_tap(i);
            std::cout << "|   Tap " << i << ": " << std::setw(12) << std::setprecision(6) << tap
                      << "                 |" << std::endl;
        }

        // CDR Phase 信息
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| CDR Status:                                  |" << std::endl;
        std::cout << "|   Phase:        " << std::setw(12) << std::setprecision(6)
                  << get_cdr_phase() * 1e12 << " ps          |" << std::endl;
        std::cout << "|   Integral:     " << std::setw(12) << std::setprecision(6)
                  << get_cdr_integral_state() << "               |" << std::endl;

        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Output Files:                                |" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_tx.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_channel.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_ctle.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_vga.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_dfe.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_data.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_dfe_taps.csv" << std::endl;
        std::cout << "|   " << m_config.output_prefix << "_cdr_phase.csv" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
    }
    
    double get_dfe_tap(int index) const {
        if (!rx) return 0.0;
        return rx->get_dfe_tap_signal(index).read();
    }

    double get_cdr_phase() const {
        if (!rx) return 0.0;
        return rx->get_cdr_phase();
    }

    double get_cdr_integral_state() const {
        if (!rx) return 0.0;
        return rx->get_cdr_integral_state();
    }
    
    ~NrzLinkTb() {
        delete vdd_src;
        delete dummy1_src;
        delete dummy2_src;
        delete dummy3_src;
        delete dummy4_src;
        delete wavegen;
        delete dfe_tap_bridge;
        delete tx;
        delete channel;
        delete rx;
        delete rec_tx;
        delete rec_channel;
        delete rec_ctle;
        delete rec_vga;
        delete rec_dfe;
        delete rec_data;
        delete rec_dfe_taps;
        delete rec_cdr_phase;
    }
};

// ============================================================================
// Main
// ============================================================================

int sc_main(int argc, char* argv[]) {
    // 设置 1 fs 时间精度（SystemC 最小精度）
    // 注意：1.5625 ps = 1562.5 fs 仍不能被 1 fs 整除
    //       100 ps / 64 不是整数 fs，会被舍入为 1562 fs 或 1563 fs
    sc_core::sc_set_time_resolution(1, sc_core::SC_FS);
    
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", 
                                             sc_core::SC_DO_NOTHING);
    
    std::cout << "+----------------------------------------------+" << std::endl;
    std::cout << "|     10Gbps NRZ Link Testbench                |" << std::endl;
    std::cout << "|     (Differential Direct Connection)         |" << std::endl;
    std::cout << "+----------------------------------------------+" << std::endl;
    
    // 默认配置
    NrzLinkConfig config;
    
    // 命令行参数处理
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "long") {
            std::cout << "Using LONG CHANNEL configuration..." << std::endl;
            config.configure_long_channel();
        }
        else if (arg == "short") {
            std::cout << "Using SHORT CHANNEL configuration..." << std::endl;
            config.configure_short_channel();
        }
        else if (arg == "no-adapt") {
            std::cout << "Disabling adaption..." << std::endl;
            config.disable_adaption();
        }
        else if (arg == "ss" && i + 1 < argc) {
            std::string config_file = argv[++i];
            std::cout << "Using STATE_SPACE channel: " << config_file << std::endl;
            config.use_state_space_channel(config_file);
        }
        else if (arg == "-d" && i + 1 < argc) {
            int ui_count = std::atoi(argv[++i]);
            std::cout << "Setting duration to " << ui_count << " UI..." << std::endl;
            config.set_duration_ui(ui_count);
        }
        else if (arg == "-o" && i + 1 < argc) {
            config.output_prefix = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc) {
            std::string cfg_file = argv[++i];
            std::cout << "Loading SystemParams config: " << cfg_file << std::endl;
            SystemParams sp;
            if (!ConfigLoader::loadFromFile(cfg_file, sp)) {
                std::cerr << "配置加载失败: " << cfg_file << std::endl;
                return 1;
            }
            config.apply_system_params(sp);
            std::cout << "Config applied to link (timing fields re-derived by link)." << std::endl;
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "\nUsage: nrz_link_tb [options]\n" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  long        Use long channel (20dB loss)" << std::endl;
            std::cout << "  short       Use short channel (3dB loss)" << std::endl;
            std::cout << "  no-adapt    Disable adaption" << std::endl;
            std::cout << "  ss <file>   Use State Space channel from JSON" << std::endl;
            std::cout << "  -d <ui>     Set duration in UI count" << std::endl;
            std::cout << "  -o <prefix> Set output file prefix" << std::endl;
            std::cout << "  --config <f> Load SystemParams JSON (rx/adaption/tx/wave/clock 子节覆盖；"
                         "链路时序字段仍由 data_rate×oversampling 派生)" << std::endl;
            return 0;
        }
    }
    
    // 创建测试台
    NrzLinkTb tb("tb");
    tb.configure(config);
    tb.build();
    tb.run();
    tb.save_results();
    tb.print_summary();
    
    sc_core::sc_stop();
    
    std::cout << "\nTestbench completed successfully." << std::endl;
    return 0;
}
