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

int sc_main(int argc, char* argv[]) {
    using namespace serdes;
    
    std::cout << "=== SerDes SystemC-AMS Simple Link Testbench ===" << std::endl;
    
    // 加载默认配置
    SystemParams params = ConfigLoader::load_default();
    
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
    sca_tdf::sca_signal<double> sig_cdr_out;
    
    // 创建 TX 模块
    std::cout << "\nCreating TX modules..." << std::endl;
    WaveGenerationTdf wave_gen("wave_gen", params.wave);
    TxFfeTdf tx_ffe("tx_ffe", params.tx.ffe);
    TxMuxTdf tx_mux("tx_mux", params.tx.mux_lane);
    TxDriverTdf tx_driver("tx_driver", params.tx.driver);
    
    // 创建 Channel 模块
    std::cout << "Creating Channel module..." << std::endl;
    ChannelSparamTdf channel("channel", params.channel);
    
    // 创建 RX 模块
    std::cout << "Creating RX modules..." << std::endl;
    RxCtleTdf rx_ctle("rx_ctle", params.rx.ctle);
    RxVgaTdf rx_vga("rx_vga", params.rx.vga);
    RxSamplerTdf rx_sampler("rx_sampler", params.rx.sampler);
    RxDfeTdf rx_dfe("rx_dfe", params.rx.dfe);
    RxCdrTdf rx_cdr("rx_cdr", params.cdr);
    
    // 连接 TX 链路
    std::cout << "Connecting TX chain..." << std::endl;
    wave_gen.out(sig_wave_out);
    tx_ffe.in(sig_wave_out);
    tx_ffe.out(sig_ffe_out);
    tx_mux.in(sig_ffe_out);
    tx_mux.out(sig_mux_out);
    tx_driver.in(sig_mux_out);
    tx_driver.out(sig_driver_out);
    
    // 连接 Channel
    std::cout << "Connecting Channel..." << std::endl;
    channel.in(sig_driver_out);
    channel.out(sig_channel_out);
    
    // 连接 RX 链路
    std::cout << "Connecting RX chain..." << std::endl;
    rx_ctle.in(sig_channel_out);
    rx_ctle.out(sig_ctle_out);
    rx_vga.in(sig_ctle_out);
    rx_vga.out(sig_vga_out);
    rx_sampler.in(sig_vga_out);
    rx_sampler.out(sig_sampler_out);
    rx_dfe.in(sig_sampler_out);
    rx_dfe.out(sig_dfe_out);
    rx_cdr.in(sig_dfe_out);
    rx_cdr.out(sig_cdr_out);
    
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
    sca_util::sca_trace(tf, sig_cdr_out, "cdr_out");
    
    // 运行仿真
    std::cout << "\nStarting simulation..." << std::endl;
    sc_core::sc_start(params.global.duration, sc_core::SC_SEC);
    
    // 关闭 trace 文件
    sca_util::sca_close_tabular_trace_file(tf);
    
    std::cout << "\n=== Simulation completed successfully! ===" << std::endl;
    std::cout << "Trace file: simple_link.dat" << std::endl;
    
    return 0;
}
