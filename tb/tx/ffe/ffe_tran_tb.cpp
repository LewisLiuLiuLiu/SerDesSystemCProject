// FFE 瞬态仿真测试平台 - 支持多种测试场景
#include <systemc-ams>
#include "ams/tx_ffe.h"
#include "common/parameters.h"
#include "ffe_helpers.h"
#include <iostream>
#include <string>
#include <iomanip>

using namespace serdes;
using namespace serdes::tb;

// 测试场景枚举
enum TestScenario {
    BASIC_PRBS,           // 基本 PRBS 测试
    IMPULSE_RESPONSE,     // 冲激响应测试
    STEP_RESPONSE,        // 阶跃响应测试
    FREQUENCY_RESPONSE,   // 频率响应测试
    PREEMPHASIS_TEST      // 预加重效果测试
};

// FFE 瞬态仿真顶层模块
SC_MODULE(FfeTransientTestbench) {
    // 模块实例
    SignalSource* src;
    TxFfeTdf* ffe;
    SignalMonitor* monitor;
    
    // 信号连接
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    TestScenario m_scenario;
    TxFfeParams m_params;
    
    FfeTransientTestbench(sc_core::sc_module_name nm, 
                          TestScenario scenario = BASIC_PRBS)
        : sc_module(nm)
        , m_scenario(scenario)
    {
        // 配置 FFE 默认参数
        m_params.taps = {0.2, 0.6, 0.2};
        
        // 根据测试场景配置
        configure_scenario(scenario);
        
        // 创建 FFE
        ffe = new TxFfeTdf("ffe", m_params);
        
        // 创建监测器 - 保存到文件
        std::string filename = get_output_filename(scenario);
        monitor = new SignalMonitor("monitor", filename);
        
        // 连接模块
        src->out(sig_in);
        ffe->in(sig_in);
        ffe->out(sig_out);
        monitor->in(sig_out);
    }
    
    void configure_scenario(TestScenario scenario) {
        switch (scenario) {
            case BASIC_PRBS:
                // 基本 PRBS 测试 - 标准配置
                m_params.taps = {0.2, 0.6, 0.2};
                src = new SignalSource("src", 
                                       SignalSource::PRBS,
                                       1.0,       // 1V 幅度
                                       10e9,      // 10 GHz 符号率
                                       100e9);    // 100 GHz 采样率
                break;
                
            case IMPULSE_RESPONSE:
                // 冲激响应测试
                m_params.taps = {0.2, 0.6, 0.2};
                src = new SignalSource("src", 
                                       SignalSource::IMPULSE,
                                       1.0,
                                       10e9,
                                       100e9);
                break;
                
            case STEP_RESPONSE:
                // 阶跃响应测试
                m_params.taps = {0.2, 0.6, 0.2};
                src = new SignalSource("src", 
                                       SignalSource::STEP,
                                       1.0,
                                       10e9,
                                       100e9);
                break;
                
            case FREQUENCY_RESPONSE:
                // 频率响应测试 - 使用正弦波
                m_params.taps = {0.2, 0.6, 0.2};
                src = new SignalSource("src", 
                                       SignalSource::SINE,
                                       1.0,
                                       5e9,       // 5 GHz 测试频率
                                       100e9);
                break;
                
            case PREEMPHASIS_TEST:
                // 预加重效果测试 - 使用去加重配置
                m_params.taps = {0.0, 1.0, -0.25};  // PCIe Gen3 风格
                src = new SignalSource("src", 
                                       SignalSource::SQUARE,
                                       1.0,
                                       5e9,       // 5 GHz 方波
                                       100e9);
                break;
        }
    }
    
    std::string get_output_filename(TestScenario scenario) {
        switch (scenario) {
            case BASIC_PRBS:        return "ffe_tran_prbs.csv";
            case IMPULSE_RESPONSE:  return "ffe_tran_impulse.csv";
            case STEP_RESPONSE:     return "ffe_tran_step.csv";
            case FREQUENCY_RESPONSE: return "ffe_tran_freq.csv";
            case PREEMPHASIS_TEST:   return "ffe_tran_preemph.csv";
            default:                 return "ffe_tran_output.csv";
        }
    }
    
    ~FfeTransientTestbench() {
        delete src;
        delete ffe;
        delete monitor;
    }
    
    void print_results() {
        SignalStats stats = monitor->get_stats();
        
        std::cout << "\n=== FFE 瞬态仿真结果 (" << get_scenario_name() << ") ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "输出信号统计:" << std::endl;
        std::cout << "  均值: " << stats.mean << " V" << std::endl;
        std::cout << "  RMS: " << stats.rms << " V" << std::endl;
        std::cout << "  峰峰值: " << stats.peak_to_peak << " V" << std::endl;
        std::cout << "  最小值: " << stats.min_value << " V" << std::endl;
        std::cout << "  最大值: " << stats.max_value << " V" << std::endl;
        
        // 打印抽头配置
        std::cout << "\nFFE 抽头配置:" << std::endl;
        std::cout << "  抽头数: " << m_params.taps.size() << std::endl;
        std::cout << "  系数: [";
        for (size_t i = 0; i < m_params.taps.size(); ++i) {
            std::cout << m_params.taps[i];
            if (i < m_params.taps.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 计算抽头系数之和
        double tap_sum = 0.0;
        for (double t : m_params.taps) tap_sum += t;
        std::cout << "  抽头系数和: " << tap_sum << std::endl;
        
        std::cout << "\n输出波形已保存到: " << get_output_filename(m_scenario) << std::endl;
        
        // 场景特定分析
        analyze_scenario_results(stats);
    }
    
    const char* get_scenario_name() {
        switch (m_scenario) {
            case BASIC_PRBS:        return "基本 PRBS 测试";
            case IMPULSE_RESPONSE:  return "冲激响应测试";
            case STEP_RESPONSE:     return "阶跃响应测试";
            case FREQUENCY_RESPONSE: return "频率响应测试";
            case PREEMPHASIS_TEST:   return "预加重效果测试";
            default:                 return "未知场景";
        }
    }
    
    void analyze_scenario_results(const SignalStats& stats) {
        switch (m_scenario) {
            case BASIC_PRBS:
                std::cout << "\n[分析] PRBS 测试:" << std::endl;
                std::cout << "  输入幅度: 1.0 V" << std::endl;
                std::cout << "  输出峰峰值: " << stats.peak_to_peak << " V" << std::endl;
                break;
                
            case IMPULSE_RESPONSE:
                std::cout << "\n[分析] 冲激响应:" << std::endl;
                std::cout << "  冲激响应应显示抽头系数序列" << std::endl;
                {
                    const std::vector<double>& samples = monitor->get_samples();
                    std::cout << "  前 " << std::min(size_t(10), samples.size()) 
                              << " 个输出样本: [";
                    for (size_t i = 0; i < std::min(size_t(10), samples.size()); ++i) {
                        std::cout << std::setprecision(4) << samples[i];
                        if (i < std::min(size_t(10), samples.size()) - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
                break;
                
            case STEP_RESPONSE:
                std::cout << "\n[分析] 阶跃响应:" << std::endl;
                {
                    double tap_sum = 0.0;
                    for (double t : m_params.taps) tap_sum += t;
                    std::cout << "  预期稳态值: " << tap_sum << " V" << std::endl;
                    std::cout << "  实测最终值: " << monitor->get_last() << " V" << std::endl;
                }
                break;
                
            case FREQUENCY_RESPONSE:
                std::cout << "\n[分析] 频率响应 (5 GHz):" << std::endl;
                {
                    double gain = FirFrequencyAnalyzer::gain_linear(m_params.taps, 5e9, 10e9);
                    double gain_db = FirFrequencyAnalyzer::gain_db(m_params.taps, 5e9, 10e9);
                    double phase = FirFrequencyAnalyzer::phase_deg(m_params.taps, 5e9, 10e9);
                    std::cout << "  理论增益: " << gain << " (" << gain_db << " dB)" << std::endl;
                    std::cout << "  理论相位: " << phase << " 度" << std::endl;
                    std::cout << "  实测 RMS: " << stats.rms << " V" << std::endl;
                }
                break;
                
            case PREEMPHASIS_TEST:
                std::cout << "\n[分析] 预加重/去加重效果:" << std::endl;
                {
                    double deemph_db = EmphasisAnalyzer::deemphasis_db(m_params.taps);
                    bool is_deemph = EmphasisAnalyzer::is_deemphasis_mode(m_params.taps);
                    std::cout << "  模式: " << (is_deemph ? "去加重" : "预加重/混合") << std::endl;
                    std::cout << "  去加重量: " << deemph_db << " dB" << std::endl;
                    std::cout << "  输出峰峰值: " << stats.peak_to_peak << " V" << std::endl;
                }
                break;
                
            default:
                break;
        }
    }
    
    double get_simulation_time_ns() {
        switch (m_scenario) {
            case IMPULSE_RESPONSE:
                return 10.0;   // 10 ns
            case FREQUENCY_RESPONSE:
                return 200.0;  // 200 ns (多个周期)
            default:
                return 100.0;  // 100 ns
        }
    }
};

// SystemC 主函数
int sc_main(int argc, char* argv[]) {
    // 解析命令行参数
    TestScenario scenario = BASIC_PRBS;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "prbs" || arg == "0") scenario = BASIC_PRBS;
        else if (arg == "impulse" || arg == "1") scenario = IMPULSE_RESPONSE;
        else if (arg == "step" || arg == "2") scenario = STEP_RESPONSE;
        else if (arg == "freq" || arg == "3") scenario = FREQUENCY_RESPONSE;
        else if (arg == "preemph" || arg == "4") scenario = PREEMPHASIS_TEST;
        else {
            std::cout << "用法: " << argv[0] << " [scenario]" << std::endl;
            std::cout << "场景选项:" << std::endl;
            std::cout << "  prbs, 0    - 基本 PRBS 测试 (默认)" << std::endl;
            std::cout << "  impulse, 1 - 冲激响应测试" << std::endl;
            std::cout << "  step, 2    - 阶跃响应测试" << std::endl;
            std::cout << "  freq, 3    - 频率响应测试" << std::endl;
            std::cout << "  preemph, 4 - 预加重效果测试" << std::endl;
            return 1;
        }
    }
    
    // 创建测试平台
    FfeTransientTestbench tb("tb", scenario);
    
    // 运行瞬态仿真
    double sim_time = tb.get_simulation_time_ns();
    std::cout << "开始 FFE 瞬态仿真 (" << tb.get_scenario_name() << ")..." << std::endl;
    std::cout << "仿真时间: " << sim_time << " ns" << std::endl;
    sc_core::sc_start(sim_time, sc_core::SC_NS);
    
    // 打印结果
    tb.print_results();
    
    return 0;
}
