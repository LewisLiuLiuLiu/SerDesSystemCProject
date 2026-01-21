#include "de/config_loader.h"
#include <fstream>
#include <iostream>

namespace serdes {

bool ConfigLoader::loadFromFile(const std::string& filepath, SystemParams& params) {
    // 检测文件格式
    if (filepath.find(".json") != std::string::npos) {
        return loadJSON(filepath, params);
    } else if (filepath.find(".yaml") != std::string::npos || filepath.find(".yml") != std::string::npos) {
        return loadYAML(filepath, params);
    }
    
    std::cerr << "Unknown file format: " << filepath << std::endl;
    return false;
}

bool ConfigLoader::loadJSON(const std::string& filepath, SystemParams& params) {
    // TODO: 实现 JSON 解析
    // 目前使用默认配置
    std::cout << "Loading JSON config from: " << filepath << std::endl;
    params = load_default();
    return true;
}

bool ConfigLoader::loadYAML(const std::string& filepath, SystemParams& params) {
    // TODO: 实现 YAML 解析
    // 目前使用默认配置
    std::cout << "Loading YAML config from: " << filepath << std::endl;
    params = load_default();
    return true;
}

SystemParams ConfigLoader::load_default() {
    SystemParams params;
    
    // Global parameters
    params.global.Fs = 80e9;         // 80 GHz sampling rate
    params.global.UI = 2.5e-11;      // 25 ps unit interval (40 Gbps)
    params.global.duration = 1e-6;   // 1 us simulation
    params.global.seed = 12345;
    
    // Wave generation
    params.wave.type = PRBSType::PRBS31;
    params.wave.poly = "x^31 + x^28 + 1";
    params.wave.init = "0x7FFFFFFF";
    params.wave.jitter.RJ_sigma = 0.0;  // No jitter by default
    
    // TX FFE
    params.tx.ffe.taps = {-0.1, 1.0, -0.1};  // 3-tap FFE
    
    // TX Driver
    params.tx.driver.dc_gain = 1.0;
    params.tx.driver.vswing = 0.8;           // 800 mV
    params.tx.driver.vcm_out = 0.6;          // 600 mV common mode
    params.tx.driver.output_impedance = 50.0;
    params.tx.driver.poles = {50e9};         // 50 GHz pole
    params.tx.driver.sat_mode = "soft";
    params.tx.driver.vlin = 1.0;             // 1 V linear range
    
    // Channel
    params.channel.attenuation_db = 10.0;  // 10 dB loss
    params.channel.bandwidth_hz = 20e9;    // 20 GHz bandwidth
    
    // RX CTLE
    params.rx.ctle.zeros = {2e9};      // 2 GHz zero
    params.rx.ctle.poles = {30e9};     // 30 GHz pole
    params.rx.ctle.dc_gain = 1.5;
    
    // RX VGA
    params.rx.vga.dc_gain = 4.0;
    
    // RX Sampler  
    params.rx.sampler.threshold = 0.0;
    params.rx.sampler.hysteresis = 0.02;
    
    // RX DFE
    params.rx.dfe.taps = {-0.05, -0.02, 0.01};
    
    // CDR
    params.cdr.pi.kp = 0.05;
    params.cdr.pi.ki = 0.001;
    params.cdr.pai.resolution = 1e-12;
    params.cdr.pai.range = 5e-11;
    
    // Clock
    params.clock.type = ClockType::IDEAL;
    params.clock.frequency = 40e9;  // 40 GHz
    
    return params;
}

} // namespace serdes
