// Sampler瞬态仿真测试平台 - 支持多种测试场景
#include <systemc-ams>
#include "ams/rx_sampler.h"
#include "common/parameters.h"
#include "sampler_helpers.h"
#include <iostream>
#include <string>
#include <stdexcept>

using namespace serdes;
using namespace serdes::tb;

// 测试场景枚举
enum TestScenario {
    BASIC_FUNCTION,         // 基本功能测试
    CDR_INTEGRATION,        // CDR集成测试
    FUZZY_DECISION,         // 模糊判决测试
    PARAMETER_VALIDATION,   // 参数验证测试
    BER_MEASUREMENT         // BER测量测试
};

// Sampler瞬态仿真顶层模块 - 支持多种测试场景
SC_MODULE(SamplerTransientTestbench) {
    // 模块实例
    DiffSignalSource* src = nullptr;
    ClockSource* clk_src = nullptr;
    PhaseOffsetSource* phase_src = nullptr;
    TriggerSource* trigger_src = nullptr;
    RxSamplerTdf* sampler = nullptr;
    SamplerSignalMonitor* monitor = nullptr;
    
    // 信号连接
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_clk;
    sca_tdf::sca_signal<double> sig_phase;
    sca_tdf::sca_signal<double> sig_out;
    sca_tdf::sca_signal<bool> sig_sampling_trigger;   // 采样触发（TDF 域）
    sc_core::sc_signal<bool> sig_data_out_de;         // Sampler DE 输出 -> Monitor DE 桥接输入
    
    TestScenario m_scenario;
    RxSamplerParams m_params;
    std::string m_output_file;
    
    SamplerTransientTestbench(sc_core::sc_module_name nm, 
                             TestScenario scenario = BASIC_FUNCTION,
                             const std::string& output_file = "")
        : sc_module(nm)
        , m_scenario(scenario)
        , m_output_file(output_file)
    {
        // 根据场景配置参数
        configure_parameters();
        
        // 创建模块实例
        create_modules();
        
        // 连接信号
        connect_signals();
    }
    
    ~SamplerTransientTestbench() {
        delete src;
        delete clk_src;
        delete phase_src;
        delete trigger_src;
        delete sampler;
        delete monitor;
    }
    
    void configure_parameters() {
        // 基本参数配置
        m_params.threshold = 0.0;
        m_params.hysteresis = 0.02;
        m_params.resolution = 0.05;
        m_params.sample_delay = 0.0;
        m_params.offset_enable = false;
        m_params.offset_value = 0.0;
        m_params.noise_enable = false;
        m_params.noise_sigma = 0.0;
        m_params.noise_seed = 12345;
        
        switch (m_scenario) {
            case BASIC_FUNCTION:
                m_params.phase_source = "clock";
                break;
                
            case CDR_INTEGRATION:
                m_params.phase_source = "phase";
                break;
                
            case FUZZY_DECISION:
                m_params.phase_source = "clock";
                m_params.resolution = 0.02;
                m_params.hysteresis = 0.01;
                break;
                
            case PARAMETER_VALIDATION:
                // 配置无效参数用于测试
                m_params.phase_source = "clock";
                m_params.hysteresis = 0.1;
                m_params.resolution = 0.05;  // 无效：hysteresis > resolution
                break;
                
            case BER_MEASUREMENT:
                m_params.phase_source = "clock";
                m_params.noise_enable = true;
                m_params.noise_sigma = 0.01;  // 10mV noise
                break;
        }
    }
    
    void create_modules() {
        // 根据场景创建不同类型的信号源
        double amplitude = 0.2;  // 200mV differential
        double frequency = 1e9;  // 1GHz
        
        switch (m_scenario) {
            case BASIC_FUNCTION:
                // 基本功能测试使用方波信号
                src = new DiffSignalSource("src", DiffSignalSource::SQUARE, 
                                          amplitude, frequency, 0.6, 100e9);
                break;
                
            case CDR_INTEGRATION:
                // CDR集成测试使用PRBS信号
                src = new DiffSignalSource("src", DiffSignalSource::PRBS, 
                                          amplitude, frequency, 0.6, 100e9);
                break;
                
            case FUZZY_DECISION:
                // 模糊判决测试使用小幅度正弦波
                src = new DiffSignalSource("src", DiffSignalSource::SINE, 
                                          0.03, frequency, 0.6, 100e9);
                break;
                
            case PARAMETER_VALIDATION:
                // 参数验证测试使用DC信号
                src = new DiffSignalSource("src", DiffSignalSource::DC, 
                                          amplitude, frequency, 0.6, 100e9);
                break;
                
            case BER_MEASUREMENT:
                // BER测试使用PRBS信号
                src = new DiffSignalSource("src", DiffSignalSource::PRBS, 
                                          amplitude, frequency, 0.6, 100e9);
                break;
        }
        
        // 创建时钟源和相位源
        clk_src = new ClockSource("clk_src", 10e9, 1.0, 0.5, 100e9);
        phase_src = new PhaseOffsetSource("phase_src", 0.0, 100e9);
        trigger_src = new TriggerSource("trigger_src", 10e9, 100e9);
        
        // 创建采样器实例
        try {
            sampler = new RxSamplerTdf("sampler", m_params);
        } catch (const std::exception& e) {
            std::cerr << "Error creating sampler: " << e.what() << std::endl;
            if (m_scenario == PARAMETER_VALIDATION) {
                std::cout << "Parameter validation test PASSED - exception correctly thrown" << std::endl;
                sc_core::sc_stop();
            }
            return;
        }
        
        // 创建监测器
        monitor = new SamplerSignalMonitor("monitor", m_output_file, 100e9);
    }
    
    void connect_signals() {
        // 参数验证场景下 sampler 构造失败（预期行为），跳过连接
        if (sampler == nullptr || monitor == nullptr) {
            return;
        }
        
        // 连接信号源到采样器
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        
        // 连接时钟和相位源
        clk_src->clk_out(sig_clk);
        phase_src->phase_out(sig_phase);
        
        // 采样触发源 -> Sampler（被动式采样：每 UI 一次触发）
        trigger_src->trigger_out(sig_sampling_trigger);
        sampler->sampling_trigger(sig_sampling_trigger);
        
        // 连接采样器
        sampler->in_p(sig_in_p);
        sampler->in_n(sig_in_n);
        sampler->clk_sample(sig_clk);
        // Note: phase_offset port removed from RxSamplerTdf
        sampler->data_out(sig_out);
        sampler->data_out_de(sig_data_out_de);
        
        // 连接监测器
        monitor->in_p(sig_in_p);
        monitor->in_n(sig_in_n);
        monitor->data_out(sig_out);
        monitor->data_out_de(sig_data_out_de);
    }
};

// 命令行参数解析函数
TestScenario parse_scenario(const std::string& arg) {
    if (arg == "basic" || arg == "0") {
        return BASIC_FUNCTION;
    } else if (arg == "cdr" || arg == "1") {
        return CDR_INTEGRATION;
    } else if (arg == "fuzzy" || arg == "2") {
        return FUZZY_DECISION;
    } else if (arg == "validate" || arg == "3") {
        return PARAMETER_VALIDATION;
    } else if (arg == "ber" || arg == "4") {
        return BER_MEASUREMENT;
    } else {
        std::cerr << "Invalid scenario: " << arg << std::endl;
        std::cerr << "Valid scenarios: basic/0, cdr/1, fuzzy/2, validate/3, ber/4" << std::endl;
        throw std::invalid_argument("Invalid scenario");
    }
}

// 主函数 - 解析命令行参数并运行测试
int sc_main(int argc, char* argv[]) {
    try {
        // 默认配置
        TestScenario scenario = BASIC_FUNCTION;
        std::string output_file = "sampler_tran_basic.csv";
        
        // 解析命令行参数
        if (argc >= 2) {
            scenario = parse_scenario(argv[1]);
            
            // 根据场景设置默认输出文件名
            switch (scenario) {
                case BASIC_FUNCTION:
                    output_file = "sampler_tran_basic.csv";
                    break;
                case CDR_INTEGRATION:
                    output_file = "sampler_tran_cdr.csv";
                    break;
                case FUZZY_DECISION:
                    output_file = "sampler_tran_fuzzy.csv";
                    break;
                case PARAMETER_VALIDATION:
                    output_file = "";
                    break;
                case BER_MEASUREMENT:
                    output_file = "sampler_tran_ber.csv";
                    break;
            }
        }
        
        if (argc >= 3) {
            output_file = argv[2];
        }
        
        std::cout << "Running Sampler Transient Testbench with scenario: " << scenario << std::endl;
        
        // 创建并运行测试平台
        SamplerTransientTestbench* tb = new SamplerTransientTestbench("tb", scenario, output_file);
        
        // 运行仿真
        sc_core::sc_start(100, sc_core::SC_NS);
        
        std::cout << "Simulation completed successfully" << std::endl;
        
        // 清理资源
        delete tb;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
