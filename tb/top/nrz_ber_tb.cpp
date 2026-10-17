/**
 * @file nrz_ber_tb.cpp
 * @brief T-S3 / A3 minimal BER testbench (offline decision export).
 *
 * Signal chain is identical to nrz_link_tb (WaveGen -> TxTop -> ChannelSParamTdf
 * -> RxTop); only the per-UI data decisions (CDR data strobe) and a small
 * metadata JSON are written. This keeps 1e6-UI runs at ~O(10 MB) instead of
 * hundreds of GB of waveform CSVs.
 *
 * CLI: nrz_ber_tb [ss <channel.json>] [--config <cfg.json>] [-d <UI>] [-o <prefix>]
 */

#include <systemc-ams>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
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
// Per-UI decision recorder
// CDR data_trigger (delayed 1 step, same delay as the sampler trigger) marks
// the data sampling instant. The recorder writes the sampler output one step
// later (held value, robust to intra-step scheduling order).
// ============================================================================
class UiDecisionRecorder : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<bool> data_strobe;
    sca_tdf::sca_in<double> data;

    UiDecisionRecorder(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), data_strobe("data_strobe"), data("data") {}

    void set_attributes() override {
        data_strobe.set_rate(1);
        data.set_rate(1);
    }

    void processing() override {
        if (m_pending) {
            m_bits.push_back(data.read() > 0.5 ? 1 : 0);
            m_pending = false;
        }
        if (data_strobe.read()) {
            m_pending = true;
        }
    }

    void save_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }
        file << "ui_index,bit\n";
        for (size_t i = 0; i < m_bits.size(); ++i) {
            file << i << "," << m_bits[i] << "\n";
        }
        file.close();
        std::cout << "Saved " << m_bits.size() << " decisions to " << filename << std::endl;
    }

    size_t size() const { return m_bits.size(); }

private:
    std::vector<int> m_bits;
    bool m_pending = false;
};

// ============================================================================
// Minimal BER testbench
// ============================================================================
SC_MODULE(NrzBerTb) {
    ConstVddSource* vdd_src;
    WaveGenerationTdf* wavegen;
    TxTopModule* tx;
    ChannelSParamTdf* channel;
    RxTopModule* rx;
    UiDecisionRecorder* rec;

    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_tx_out_p;
    sca_tdf::sca_signal<double> sig_tx_out_n;
    sca_tdf::sca_signal<double> sig_channel_out_p;
    sca_tdf::sca_signal<double> sig_channel_out_n;
    sca_tdf::sca_signal<double> sig_data_out;

    NrzLinkConfig m_config;

    SC_CTOR(NrzBerTb)
        : vdd_src(nullptr), wavegen(nullptr), tx(nullptr), channel(nullptr), rx(nullptr), rec(nullptr)
        , sig_vdd("sig_vdd"), sig_wavegen_out("sig_wavegen_out")
        , sig_tx_out_p("sig_tx_out_p"), sig_tx_out_n("sig_tx_out_n")
        , sig_channel_out_p("sig_channel_out_p"), sig_channel_out_n("sig_channel_out_n")
        , sig_data_out("sig_data_out") {}

    void configure(const NrzLinkConfig& config) {
        m_config = config;
        m_config.sync_ui();
    }

    void build() {
        std::cout << "\n=== Building Minimal BER Testbench ===" << std::endl;
        m_config.print_summary();

        vdd_src = new ConstVddSource("vdd_src", 1.0, m_config.timestep_ps());
        if (m_config.wave.jitter.enable) {
            m_config.wave.jitter.edge_csv = m_config.output_prefix + "_edge_jitter.csv";
        }
        wavegen = new WaveGenerationTdf("wavegen", m_config.wave,
                                        m_config.sample_rate(), m_config.ui(), m_config.seed);
        tx = new TxTopModule("tx", m_config.tx);
        channel = new ChannelSParamTdf("channel", m_config.channel, m_config.channel_ext);
        rx = new RxTopModule("rx", m_config.rx, m_config.adaption);
        rec = new UiDecisionRecorder("rec_decisions");

        vdd_src->out(sig_vdd);

        wavegen->out(sig_wavegen_out);
        tx->in(sig_wavegen_out);
        tx->vdd(sig_vdd);
        tx->out_p(sig_tx_out_p);
        tx->out_n(sig_tx_out_n);

        channel->in[0](sig_tx_out_p);
        channel->in[1](sig_tx_out_n);
        channel->out[0](sig_channel_out_p);
        channel->out[1](sig_channel_out_n);

        rx->in_p(sig_channel_out_p);
        rx->in_n(sig_channel_out_n);
        rx->vdd(sig_vdd);
        rx->data_out(sig_data_out);

        rec->data(sig_data_out);
        rec->data_strobe(rx->get_data_trigger_signal());
    }

    void run() {
        std::cout << "\n=== Running (" << m_config.sim_ui_count() << " UI) ===" << std::endl;
        sc_core::sc_start(m_config.sim_duration, sc_core::SC_SEC);
    }

    void save_results() {
        rec->save_csv(m_config.output_prefix + "_decisions.csv");
        save_metadata(m_config.output_prefix + "_metadata.json");
    }

    void save_metadata(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        file << "{\n";
        file << "  \"data_rate_gbps\": " << m_config.data_rate / 1e9 << ",\n";
        file << "  \"ui_ps\": " << m_config.ui() * 1e12 << ",\n";
        file << "  \"sample_rate_ghz\": " << m_config.sample_rate() / 1e9 << ",\n";
        file << "  \"ui_count_requested\": " << m_config.sim_ui_count() << ",\n";
        file << "  \"decisions\": " << rec->size() << ",\n";
        file << "  \"channel_method\": \""
             << (m_config.channel_ext.method == ChannelMethod::SIMPLE ? "SIMPLE" : "STATE_SPACE")
             << "\",\n";
        file << "  \"ffe_taps\": " << m_config.tx.ffe.taps.size() << ",\n";
        file << "  \"sampler_noise_enable\": " << (m_config.rx.sampler.noise_enable ? "true" : "false") << ",\n";
        file << "  \"sampler_noise_sigma\": " << m_config.rx.sampler.noise_sigma << ",\n";
        file << "  \"cdr_phase_ps\": " << rx->get_cdr_phase() * 1e12 << ",\n";
        file << "  \"adaption_agc\": " << (m_config.adaption.agc.enabled ? "true" : "false") << ",\n";
        file << "  \"adaption_dfe\": " << (m_config.adaption.dfe.enabled ? "true" : "false") << "\n";
        file << "}\n";
        file.close();
        std::cout << "Saved metadata to " << filename << std::endl;
    }

    void print_summary() {
        std::cout << "\n+----------------------------------------------+" << std::endl;
        std::cout << "| Minimal BER TB Summary                       |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Decisions: " << std::setw(10) << rec->size() << "                    |" << std::endl;
        std::cout << "| CDR phase: " << std::setw(10) << rx->get_cdr_phase() * 1e12 << " ps               |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
    }

    ~NrzBerTb() {
        delete vdd_src;
        delete wavegen;
        delete tx;
        delete channel;
        delete rx;
        delete rec;
    }
};

// ============================================================================
// Main
// ============================================================================
int sc_main(int argc, char* argv[]) {
    sc_core::sc_set_time_resolution(1, sc_core::SC_FS);
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1660/deprecated", sc_core::SC_DO_NOTHING);
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", sc_core::SC_DO_NOTHING);

    NrzLinkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "ss" && i + 1 < argc) {
            std::string cfg = argv[++i];
            std::cout << "Using STATE_SPACE channel: " << cfg << std::endl;
            config.use_state_space_channel(cfg);
        } else if (arg == "-d" && i + 1 < argc) {
            int ui_count = std::atoi(argv[++i]);
            config.set_duration_ui(ui_count);
        } else if (arg == "-o" && i + 1 < argc) {
            config.output_prefix = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            std::string cfg_file = argv[++i];
            std::cout << "Loading SystemParams config: " << cfg_file << std::endl;
            SystemParams sp;
            if (!ConfigLoader::loadFromFile(cfg_file, sp)) {
                std::cerr << "配置加载失败: " << cfg_file << std::endl;
                return 1;
            }
            config.apply_system_params(sp);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "\nUsage: nrz_ber_tb [options]\n"
                      << "  ss <file>    Use State Space channel from JSON\n"
                      << "  --config <f> Load SystemParams JSON overlay\n"
                      << "  -d <ui>      Duration in UI count\n"
                      << "  -o <prefix>  Output prefix (writes <prefix>_decisions.csv/_metadata.json)\n"
                      << std::endl;
            return 0;
        }
    }

    NrzBerTb tb("tb");
    tb.configure(config);
    tb.build();
    tb.run();
    tb.save_results();
    tb.print_summary();

    sc_core::sc_stop();
    std::cout << "\nMinimal BER testbench completed successfully." << std::endl;
    return 0;
}
