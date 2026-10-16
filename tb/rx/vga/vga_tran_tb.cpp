// VGA瞬态仿真测试平台 - 支持多种测试场景
#include <systemc-ams>
#include "ams/rx_vga.h"
#include "common/parameters.h"
#include "vga_helpers.h"
#include <iostream>
#include <string>

using namespace serdes;
using namespace serdes::tb;

// 测试场景枚举
enum TestScenario {
    BASIC_PRBS,           // 基本PRBS测试
    FREQUENCY_RESPONSE,   // 频率响应测试
    PSRR_TEST,            // PSRR测试
    CMRR_TEST,            // CMRR测试
    SATURATION_TEST       // 饱和测试
};

// VGA瞬态仿真顶层模块 - 支持多种测试场景
SC_MODULE(VgaTransientTestbench) {
    // 模块实例
    DiffSignalSource* src;
    VddSource* vdd_src;
    RxVgaTdf* vga;
    SignalMonitor* monitor;
    
    // 信号连接
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_out_p;
    sca_tdf::sca_signal<double> sig_out_n;
    sc_core::sc_signal<double> sig_gain_cmd;  // AGC 增益命令（本 TB 不用 → 0 = 静态增益）
    
    TestScenario m_scenario;
    RxVgaParams m_params;
    
    VgaTransientTestbench(sc_core::sc_module_name nm, 
                           TestScenario scenario = BASIC_PRBS)
        : sc_module(nm)
        , m_scenario(scenario)
    {
        // 配置VGA参数
        m_params.zeros = {1e9};       // 1 GHz zero
        m_params.poles = {10e9, 20e9};      // 10 GHz, 20 GHz poles
        m_params.dc_gain = 2.0;
        m_params.vcm_out = 0.6;
        m_params.offset_enable = false;
        m_params.noise_enable = false;
        
        // 根据测试场景配置
        configure_scenario(scenario);
        
        // 创建VGA
        vga = new RxVgaTdf("vga", m_params);
        
        // 创建监测器 - 保存到文件
        std::string filename = get_output_filename(scenario);
        monitor = new SignalMonitor("monitor", filename);
        
        // 连接模块
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        vdd_src->vdd(sig_vdd);
        
        vga->in_p(sig_in_p);
        vga->in_n(sig_in_n);
        vga->vdd(sig_vdd);
        vga->out_p(sig_out_p);
        vga->out_n(sig_out_n);
        vga->gain_cmd_de(sig_gain_cmd);
        
        monitor->in_p(sig_out_p);
        monitor->in_n(sig_out_n);
    }
    
    void configure_scenario(TestScenario scenario) {
        switch (scenario) {
            case BASIC_PRBS:
                // 基本PRBS测试 - 标准配置
                src = new DiffSignalSource("src", 
                                           DiffSignalSource::PRBS,
                                           0.1,      // 100mV amplitude
                                           10e9,     // 10 GHz
                                           0.6);     // 0.6V common mode
                vdd_src = new VddSource("vdd_src", 1.0);
                break;
                
            case FREQUENCY_RESPONSE:
                // 频率响应测试 - 使用正弦波
                src = new DiffSignalSource("src", 
                                           DiffSignalSource::SINE,
                                           0.1,      // 100mV amplitude
                                           5e9,      // 5 GHz test frequency
                                           0.6);
                vdd_src = new VddSource("vdd_src", 1.0);
                break;
                
            case PSRR_TEST:
                // PSRR测试 - VDD有噪声
                m_params.psrr.enable = true;
                m_params.psrr.gain = 0.01;  // -40dB PSRR
                m_params.psrr.poles = {1e6};
                m_params.psrr.vdd_nom = 1.0;
                
                src = new DiffSignalSource("src", 
                                           DiffSignalSource::DC,
                                           0.0,      // 无差分信号
                                           0.0,
                                           0.6);
                // VDD有正弦波噪声
                vdd_src = new VddSource("vdd_src", 1.0, 100e9,
                                        VddSource::SINUSOIDAL,
                                        0.1,       // 100mV VDD ripple
                                        1e6);      // 1 MHz ripple frequency
                break;
                
            case CMRR_TEST:
                // CMRR测试 - 共模输入变化
                m_params.cmrr.enable = true;
                m_params.cmrr.gain = 0.001;  // -60dB CMRR
                m_params.cmrr.poles = {10e6};
                
                src = new DiffSignalSource("src", 
                                           DiffSignalSource::DC,
                                           0.1,      // 小差分信号
                                           0.0,
                                           0.6);     // 后续可扩展为共模变化
                vdd_src = new VddSource("vdd_src", 1.0);
                break;
                
            case SATURATION_TEST:
                // 饱和测试 - 大信号输入
                src = new DiffSignalSource("src", 
                                           DiffSignalSource::SQUARE,
                                           0.5,      // 500mV大幅度
                                           1e9,      // 1 GHz
                                           0.6);
                vdd_src = new VddSource("vdd_src", 1.0);
                break;
        }
    }
    
    std::string get_output_filename(TestScenario scenario) {
        switch (scenario) {
            case BASIC_PRBS:        return "vga_tran_prbs.csv";
            case FREQUENCY_RESPONSE: return "vga_tran_freq.csv";
            case PSRR_TEST:          return "vga_tran_psrr.csv";
            case CMRR_TEST:          return "vga_tran_cmrr.csv";
            case SATURATION_TEST:    return "vga_tran_sat.csv";
            default:                 return "vga_tran_output.csv";
        }
    }
    
    ~VgaTransientTestbench() {
        delete src;
        delete vdd_src;
        delete vga;
        delete monitor;
    }
    
    void print_results() {
        SignalStats diff_stats = monitor->get_diff_stats();
        SignalStats cm_stats = monitor->get_cm_stats();
        
        std::cout << "\n=== VGA瞬态仿真结果 (" << get_scenario_name() << ") ===" << std::endl;
        std::cout << "差分信号统计:" << std::endl;
        std::cout << "  均值: " << diff_stats.mean << " V" << std::endl;
        std::cout << "  RMS: " << diff_stats.rms << " V" << std::endl;
        std::cout << "  峰峰值: " << diff_stats.peak_to_peak << " V" << std::endl;
        std::cout << "  最小值: " << diff_stats.min_value << " V" << std::endl;
        std::cout << "  最大值: " << diff_stats.max_value << " V" << std::endl;
        
        std::cout << "\n共模信号统计:" << std::endl;
        std::cout << "  均值: " << cm_stats.mean << " V" << std::endl;
        std::cout << "  RMS: " << cm_stats.rms << " V" << std::endl;
        std::cout << "  峰峰值: " << cm_stats.peak_to_peak << " V" << std::endl;
        
        std::cout << "\n输出波形已保存到: " << get_output_filename(m_scenario) << std::endl;
        
        // 场景特定分析
        analyze_scenario_results(diff_stats, cm_stats);
    }
    
    const char* get_scenario_name() {
        switch (m_scenario) {
            case BASIC_PRBS:        return "基本PRBS测试";
            case FREQUENCY_RESPONSE: return "频率响应测试";
            case PSRR_TEST:          return "PSRR测试";
            case CMRR_TEST:          return "CMRR测试";
            case SATURATION_TEST:    return "饱和测试";
            default:                 return "未知场景";
        }
    }
    
    // 根据测试场景返回对应的仿真时间(纳秒)
    double get_simulation_time_ns() {
        switch (m_scenario) {
            case FREQUENCY_RESPONSE:
                // 频率响应测试: 5GHz正弦波, 1μs可覆盖5000个周期
                return 1000.0;  // 1 μs
            case PSRR_TEST:
                // PSRR测试: VDD纹波1MHz, 需要至少3个完整周期
                return 3000.0;  // 3 μs
            case CMRR_TEST:
                // CMRR测试: 低频效应分析, 需要更长观测时间
                return 3000.0;  // 3 μs
            case BASIC_PRBS:
            case SATURATION_TEST:
            default:
                // 高频测试: 100ns足够观察多个周期
                return 100.0;   // 100 ns
        }
    }
    
    void analyze_scenario_results(const SignalStats& diff_stats, 
                                  const SignalStats& cm_stats) {
        switch (m_scenario) {
            case BASIC_PRBS:
                std::cout << "\n[分析] DC增益 ≈ " 
                          << diff_stats.peak_to_peak / 0.1 << "x" << std::endl;
                break;
                
            case PSRR_TEST:
                if (diff_stats.peak_to_peak > 0.001) {
                    std::cout << "\n[分析] PSRR测试: VDD噪声已耦合到输出" << std::endl;
                    std::cout << "  输出差分纹波: " << diff_stats.peak_to_peak * 1000 << " mV" << std::endl;
                } else {
                    std::cout << "\n[分析] PSRR测试: VDD噪声被有效抑制" << std::endl;
                }
                break;
                
            case SATURATION_TEST:
                std::cout << "\n[分析] 饱和测试:" << std::endl;
                std::cout << "  输入幅度: 500mV" << std::endl;
                std::cout << "  输出峰峰值: " << diff_stats.peak_to_peak * 1000 << " mV" << std::endl;
                if (diff_stats.peak_to_peak < 0.75 * m_params.dc_gain) {
                    std::cout << "  状态: 已进入饱和区" << std::endl;
                }
                break;
                
            default:
                break;
        }
    }
};

// SystemC主函数
int sc_main(int argc, char* argv[]) {
    // 解析命令行参数
    TestScenario scenario = BASIC_PRBS;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "prbs" || arg == "0") scenario = BASIC_PRBS;
        else if (arg == "freq" || arg == "1") scenario = FREQUENCY_RESPONSE;
        else if (arg == "psrr" || arg == "2") scenario = PSRR_TEST;
        else if (arg == "cmrr" || arg == "3") scenario = CMRR_TEST;
        else if (arg == "sat" || arg == "4") scenario = SATURATION_TEST;
        else {
            std::cout << "用法: " << argv[0] << " [scenario]" << std::endl;
            std::cout << "场景选项:" << std::endl;
            std::cout << "  prbs, 0  - 基本PRBS测试 (默认)" << std::endl;
            std::cout << "  freq, 1  - 频率响应测试" << std::endl;
            std::cout << "  psrr, 2  - PSRR测试" << std::endl;
            std::cout << "  cmrr, 3  - CMRR测试" << std::endl;
            std::cout << "  sat, 4   - 饱和测试" << std::endl;
            return 1;
        }
    }
    
    // 创建测试平台
    VgaTransientTestbench tb("tb", scenario);
    
    // 根据测试场景获取仿真时间
    double sim_time_ns = tb.get_simulation_time_ns();
    
    // 运行瞬态仿真
    std::cout << "开始VGA瞬态仿真 (" << tb.get_scenario_name() << ")..." << std::endl;
    std::cout << "仿真时间: " << sim_time_ns << " ns" << std::endl;
    sc_core::sc_start(sim_time_ns, sc_core::SC_NS);
    
    // 打印结果
    tb.print_results();
    
    return 0;
}
