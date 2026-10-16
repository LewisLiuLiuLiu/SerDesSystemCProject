#include <systemc>
#include <systemc-ams>
#include "ams/wave_generation.h"
#include "ams/tx_ffe.h"
#include "ams/tx_mux.h"
#include "ams/tx_driver.h"
#include "ams/channel_sparam.h"
#include "ams/rx_ctle.h"
#include "ams/rx_vga.h"
#include "ams/rx_sampler.h"
#include "ams/rx_dfe.h"
#include "ams/rx_cdr.h"
#include "de/config_loader.h"
#include "common/parameters.h"
#include <iostream>

// ============================================================================
// 常量源（用于差分负端/供电端口绑定）
// ============================================================================
class ConstSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;

    ConstSource(sc_core::sc_module_name nm, double value, double sample_rate)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_value(value)
        , m_timestep(1.0 / sample_rate)
    {}

    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }

    void processing() {
        out.write(m_value);
    }

private:
    double m_value;
    double m_timestep;
};

int sc_main(int argc, char* argv[]) {
    using namespace serdes;
    
    std::cout << "=== SerDes SystemC-AMS Simple Link Testbench ===" << std::endl;
    
    // 加载默认配置
    // 加载配置：可选 [config.json]，缺省使用内置默认值
    SystemParams params;
    if (argc > 1) {
        if (!ConfigLoader::loadFromFile(argv[1], params)) {
            std::cerr << "配置加载失败，退出: " << argv[1] << std::endl;
            return 1;
        }
    } else {
        params = ConfigLoader::load_default();
    }
    
    std::cout << "Configuration loaded:" << std::endl;
    std::cout << "  Sampling rate: " << params.global.Fs / 1e9 << " GHz" << std::endl;
    std::cout << "  Data rate: " << 1.0 / params.global.UI / 1e9 << " Gbps" << std::endl;
    std::cout << "  Simulation time: " << params.global.duration * 1e6 << " us" << std::endl;
    
    // 创建信号连接
    sca_tdf::sca_signal<double> sig_wave_out;
    sca_tdf::sca_signal<double> sig_ffe_out;
    sca_tdf::sca_signal<double> sig_mux_out;
    sca_tdf::sca_signal<double> sig_driver_out;
    sca_tdf::sca_signal<double> sig_channel_out;
    sca_tdf::sca_signal<double> sig_ctle_out;
    sca_tdf::sca_signal<double> sig_vga_out;
    sca_tdf::sca_signal<double> sig_sampler_out;
    sca_tdf::sca_signal<double> sig_dfe_out;
    sca_tdf::sca_signal<double> sig_cdr_phase_out;
    sca_tdf::sca_signal<double> sig_zero;  // 用于差分输入的0V
    sca_tdf::sca_signal<double> sig_vdd;   // 供电（vdd 端口绑定）
    sca_tdf::sca_signal<double> sig_ctle_out_n;    // CTLE 差分负端输出（未使用）
    sca_tdf::sca_signal<double> sig_driver_out_n;  // TX Driver 差分负端输出（未使用）
    sca_tdf::sca_signal<double> sig_vga_out_n;     // VGA 差分负端输出（未使用）
    sca_tdf::sca_signal<bool> sig_sampling_trigger;  // CDR -> Sampler 采样触发
    sc_core::sc_signal<bool> sig_sampler_de_dummy;   // Sampler DE 输出（未使用）
    sc_core::sc_signal<double> sig_vga_gain_cmd_de;  // VGA AGC 增益命令（未使用 → 0 = 静态增益）
    
    // 常量源（差分负端 / vdd 供电）
    ConstSource v_zero("v_zero", 0.0, params.global.Fs);
    ConstSource v_vdd("v_vdd", 1.0, params.global.Fs);
    
    // 创建 TX 模块
    std::cout << "\nCreating TX modules..." << std::endl;
    WaveGenerationTdf wave_gen("wave_gen", params.wave, params.global.Fs, params.global.UI);
    TxFfeTdf tx_ffe("tx_ffe", params.tx.ffe);
    TxMuxTdf tx_mux("tx_mux", params.tx.mux_lane);
    TxDriverTdf tx_driver("tx_driver", params.tx.driver);
    
    // 创建 Channel 模块
    std::cout << "Creating Channel module..." << std::endl;
    ChannelSParamTdf channel("channel", params.channel);
    
    // 创建 RX 模块
    std::cout << "Creating RX modules..." << std::endl;
    RxCtleTdf rx_ctle("rx_ctle", params.rx.ctle);
    RxVgaTdf rx_vga("rx_vga", params.rx.vga);
    RxSamplerTdf rx_sampler("rx_sampler", params.rx.sampler);
    RxDfeParams rx_dfe_params;  // 固定抽头 DFE（RxParams 现由 dfe_summer 提供配置）
    RxDfeTdf rx_dfe("rx_dfe", rx_dfe_params);
    RxCdrTdf rx_cdr("rx_cdr", params.cdr);
    
    // 常量源
    v_zero.out(sig_zero);
    v_vdd.out(sig_vdd);
    
    // 连接 TX 链路
    std::cout << "Connecting TX chain..." << std::endl;
    wave_gen.out(sig_wave_out);
    tx_ffe.in(sig_wave_out);
    tx_ffe.out(sig_ffe_out);
    tx_mux.in(sig_ffe_out);
    tx_mux.out(sig_mux_out);
    tx_driver.in_p(sig_mux_out);
    tx_driver.in_n(sig_zero);  // 单端转差分
    tx_driver.vdd(sig_vdd);
    tx_driver.out_p(sig_driver_out);
    tx_driver.out_n(sig_driver_out_n);
    
    // 连接 Channel（MIMO 端口向量，SISO 模式使用索引 0）
    std::cout << "Connecting Channel..." << std::endl;
    channel.in[0](sig_driver_out);
    channel.out[0](sig_channel_out);
    
    // 连接 RX 链路
    std::cout << "Connecting RX chain..." << std::endl;
    rx_ctle.in_p(sig_channel_out);
    rx_ctle.in_n(sig_zero);  // 单端转差分
    rx_ctle.vdd(sig_vdd);
    rx_ctle.out_p(sig_ctle_out);
    rx_ctle.out_n(sig_ctle_out_n);
    rx_vga.in_p(sig_ctle_out);
    rx_vga.in_n(sig_zero);
    rx_vga.vdd(sig_vdd);
    rx_vga.out_p(sig_vga_out);
    rx_vga.out_n(sig_vga_out_n);
    rx_vga.gain_cmd_de(sig_vga_gain_cmd_de);
    rx_sampler.in_p(sig_vga_out);
    rx_sampler.in_n(sig_zero);
    rx_sampler.clk_sample(sig_zero);        // 未使用（触发式采样），绑定以满足精化要求
    rx_sampler.sampling_trigger(sig_sampling_trigger);
    rx_sampler.data_out(sig_sampler_out);
    rx_sampler.data_out_de(sig_sampler_de_dummy);
    rx_dfe.in(sig_sampler_out);
    rx_dfe.out(sig_dfe_out);
    rx_cdr.in(sig_dfe_out);
    rx_cdr.phase_out(sig_cdr_phase_out);
    rx_cdr.sampling_trigger(sig_sampling_trigger);  // CDR 采样触发 -> Sampler
    
    // 创建 trace 文件
    std::cout << "\nCreating trace file..." << std::endl;
    sca_util::sca_trace_file* tf = sca_util::sca_create_tabular_trace_file("simple_link");
    
    // 追踪关键信号
    sca_util::sca_trace(tf, sig_wave_out, "wave_out");
    sca_util::sca_trace(tf, sig_ffe_out, "ffe_out");
    sca_util::sca_trace(tf, sig_driver_out, "driver_out");
    sca_util::sca_trace(tf, sig_channel_out, "channel_out");
    sca_util::sca_trace(tf, sig_ctle_out, "ctle_out");
    sca_util::sca_trace(tf, sig_vga_out, "vga_out");
    sca_util::sca_trace(tf, sig_sampler_out, "sampler_out");
    sca_util::sca_trace(tf, sig_cdr_phase_out, "cdr_phase_out");
    sca_util::sca_trace(tf, sig_zero, "zero");
    
    // 运行仿真
    std::cout << "\nStarting simulation..." << std::endl;
    sc_core::sc_start(params.global.duration, sc_core::SC_SEC);
    
    // 关闭 trace 文件
    sca_util::sca_close_tabular_trace_file(tf);
    
    std::cout << "\n=== Simulation completed successfully! ===" << std::endl;
    std::cout << "Trace file: simple_link.dat" << std::endl;
    
    return 0;
}
