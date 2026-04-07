/**
 * @file nrz_link_config.h
 * @brief 10Gbps NRZ 链路统一参数配置
 * 
 * 特点:
 * - 所有时序参数由 data_rate 和 oversampling 派生
 * - UI 自动同步到所有子模块
 * - 支持 SIMPLE 和 STATE_SPACE 两种信道模型
 * - 差分信道直连，无需 DiffToSingle 转换
 */

#ifndef NRZ_LINK_CONFIG_H
#define NRZ_LINK_CONFIG_H

#include "common/parameters.h"
#include "ams/channel_sparam.h"
#include <systemc-ams>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>

namespace serdes {

// ============================================================================
// Signal Statistics (信号统计结构)
// ============================================================================

struct SignalStats {
    double min_val = 0.0;
    double max_val = 0.0;
    double mean_val = 0.0;
    double rms_val = 0.0;
    double peak_to_peak = 0.0;
    size_t sample_count = 0;
};

// ============================================================================
// Constant VDD Source (恒压源)
// ============================================================================

class ConstVddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    ConstVddSource(sc_core::sc_module_name nm, double voltage = 1.0, double timestep_ps = 1.5625)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_voltage(voltage)
        , m_timestep_ps(timestep_ps) {}
    
    void set_attributes() override {
        set_timestep(m_timestep_ps, sc_core::SC_PS);
    }
    
    void processing() override {
        out.write(m_voltage);
    }
    
private:
    double m_voltage;
    double m_timestep_ps;
};

/**
 * @brief NRZ 链路统一配置结构
 */
struct NrzLinkConfig {
    // ========================================================================
    // 核心时序参数 (其他参数由此派生)
    // ========================================================================
    double data_rate;              ///< 数据速率 (Hz), e.g., 10e9
    int oversampling;              ///< 过采样倍数, e.g., 64
    
    // ========================================================================
    // 派生计算方法
    // ========================================================================
    double ui() const { return 1.0 / data_rate; }
    double sample_rate() const { return data_rate * oversampling; }
    double timestep_s() const { return 1.0 / sample_rate(); }
    double timestep_ps() const { return timestep_s() * 1e12; }
    double nyquist_freq() const { return data_rate / 2.0; }
    
    // ========================================================================
    // 仿真控制
    // ========================================================================
    double sim_duration;           ///< 仿真时长 (s)
    unsigned int seed;             ///< 随机种子
    std::string output_prefix;     ///< 输出文件前缀
    
    int sim_ui_count() const { 
        return static_cast<int>(sim_duration / ui()); 
    }
    
    // ========================================================================
    // 子模块参数
    // ========================================================================
    WaveGenParams wave;
    TxParams tx;
    ChannelParams channel;
    ChannelExtendedParams channel_ext;
    RxParams rx;
    AdaptionParams adaption;
    ClockParams clock;
    
    // ========================================================================
    // 构造函数: 10Gbps NRZ 默认配置
    // ========================================================================
    NrzLinkConfig()
        : data_rate(10e9)
        , oversampling(50)         // 50x: 500 GHz, 2 ps timestep (exact fs resolution)
        , sim_duration(2e-6)       // 2µs = 20000 UI
        , seed(12345)
        , output_prefix("nrz_10g")
    {
        init_10g_defaults();
        sync_ui();
    }
    
    /**
     * @brief 同步 UI 到所有子模块
     * 
     * 关键: 确保所有需要 UI 的模块使用一致的时间基准
     */
    void sync_ui() {
        double ui_val = ui();
        double fs = sample_rate();
        
        // DFE Summer 需要 UI 计算延迟
        rx.dfe_summer.ui = ui_val;
        
        // CDR 需要 UI 进行相位归一化
        rx.cdr.ui = ui_val;
        
        // Adaption 需要 UI 和采样率
        adaption.UI = ui_val;
        adaption.Fs = fs;
        adaption.fast_update_period = 2.0 * ui_val;      // 2 UI
        adaption.slow_update_period = 2000.0 * ui_val;   // 2000 UI
        
        // Clock 频率
        clock.frequency = data_rate;
    }
    
    /**
     * @brief 10Gbps NRZ 默认参数初始化
     */
    void init_10g_defaults() {
        // ====== Wave Generation ======
        wave.type = PRBSType::PRBS31;
        wave.single_pulse = 0.0;  // PRBS模式
        
        // ====== TX ======
        tx.ffe.taps = {1.0};                    // 无预加重 (单抽头)
        tx.mux_lane = 0;
        tx.driver.dc_gain = 1.0;
        tx.driver.vswing = 0.8;                 // 800mVpp 差分
        tx.driver.vcm_out = 0.6;                // 600mV 共模
        tx.driver.output_impedance = 50.0;
        tx.driver.poles = {25e9};               // 25GHz 带宽
        tx.driver.sat_mode = "soft";
        tx.driver.vlin = 0.5;
        
        // ====== Channel (差分直连: 2输入 2输出) ======
        channel.touchstone = "";
        channel.ports = 4;                      // 物理端口数 (s4p)
        channel.crosstalk = false;
        channel.bidirectional = false;
        channel.attenuation_db = 6.0;           // 6dB 损耗
        channel.bandwidth_hz = 20e9;            // 20GHz 带宽
        channel_ext.method = ChannelMethod::SIMPLE;
        channel_ext.config_file = "";
        
        // ====== RX CTLE ======
        // 高频补偿: zero < pole, 在 Nyquist (5GHz) 附近提升
        rx.ctle.zeros = {1e9};                  // 1GHz zero
        rx.ctle.poles = {3e9, 15e9};            // 3GHz, 15GHz poles
        rx.ctle.dc_gain = 1.0;
        rx.ctle.vcm_out = 0.6;
        rx.ctle.sat_min = -0.8;
        rx.ctle.sat_max = 0.8;
        rx.ctle.offset_enable = false;
        rx.ctle.noise_enable = false;
        
        // ====== RX VGA ======
        rx.vga.zeros = {};                      // 纯增益级
        rx.vga.poles = {25e9};                  // 25GHz 带宽
        rx.vga.dc_gain = 2.0;
        rx.vga.vcm_out = 0.6;
        rx.vga.sat_min = -0.8;
        rx.vga.sat_max = 0.8;
        rx.vga.offset_enable = false;
        rx.vga.noise_enable = false;
        
        // ====== RX DFE Summer ======
        rx.dfe_summer.tap_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};  // Initial taps, will be updated by adaptation
        rx.dfe_summer.vcm_out = 0.0;
        rx.dfe_summer.vtap = 0.05;  // 50mV per tap - reasonable for DFE feedback
        rx.dfe_summer.map_mode = "pm1";
        rx.dfe_summer.enable = true;
        rx.dfe_summer.sat_enable = false;
        
        // ====== RX Sampler ======
        rx.sampler.threshold = 0.0;
        rx.sampler.hysteresis = 0.01;           // < resolution
        rx.sampler.resolution = 0.02;
        rx.sampler.sample_delay = 0.0;
        rx.sampler.phase_source = "phase";      // CDR 相位驱动
        rx.sampler.offset_enable = false;
        rx.sampler.noise_enable = false;
        
        // ====== RX CDR ======
        rx.cdr.pi.kp = 0.01;
        rx.cdr.pi.ki = 1e-4;
        rx.cdr.pi.edge_threshold = 0.5;
        rx.cdr.pai.resolution = 1e-12;
        rx.cdr.pai.range = 5e-11;
        rx.cdr.sample_point = 0.5;              // UI 中心采样
        rx.cdr.debug_enable = false;
        
        // ====== Adaption ======
        adaption.update_mode = "multi-rate";
        adaption.seed = seed;
        
        adaption.agc.enabled = true;
        adaption.agc.target_amplitude = 0.4;
        adaption.agc.kp = 0.1;
        adaption.agc.ki = 100.0;
        adaption.agc.gain_min = 0.5;
        adaption.agc.gain_max = 8.0;
        adaption.agc.initial_gain = 2.0;
        
        adaption.dfe.enabled = true;
        adaption.dfe.num_taps = 5;
        adaption.dfe.algorithm = "sign-lms";
        adaption.dfe.mu = 1e-5;
        adaption.dfe.mu_startup = 0.00001;   // Reduced from 0.0001 for finer convergence
        adaption.dfe.mu_acquire = 0.000001;  // Reduced from 0.00001
        adaption.dfe.mu_track = 0.0000001;   // Reduced from 0.000001
        adaption.dfe.leakage = 1e-6;         // Reset to default
        adaption.dfe.tap_min = -1.0;
        adaption.dfe.tap_max = 1.0;
        // FIX: Tap 1 should be the largest (compensates most recent ISI)
        // Typical DFE tap profile: exponential decay
        adaption.dfe.initial_taps = {0.30, 0.15, 0.08, 0.04, 0.02};  // Tap1 largest!
        adaption.dfe.stats_period = 4096;   // Increased from 2048 for better statistics

        adaption.vref_adapt.enabled = false;
        adaption.vref_adapt.vref_pos = 0.35;  // ~50% of VGA output amplitude (±0.72V)
        adaption.vref_adapt.vref_neg = -0.35;

        adaption.cdr_pi.enabled = true;
        adaption.cdr_pi.kp = 0.01;
        adaption.cdr_pi.ki = 1e-4;
        
        adaption.safety.freeze_on_error = false;
        adaption.safety.rollback_enable = false;
        
        // ====== Clock ======
        clock.type = ClockType::IDEAL;
    }
    
    // ========================================================================
    // 配置变体
    // ========================================================================
    
    /**
     * @brief 使用 State Space 信道模型 (从JSON加载)
     */
    void use_state_space_channel(const std::string& config_file) {
        channel_ext.method = ChannelMethod::STATE_SPACE;
        channel_ext.config_file = config_file;
    }
    
    /**
     * @brief 长信道配置 (高损耗场景)
     */
    void configure_long_channel() {
        channel.attenuation_db = 20.0;
        channel.bandwidth_hz = 10e9;
        
        // 启用 FFE 预加重
        tx.ffe.taps = {-0.1, 1.0, -0.3};
        
        // 增加 RX 增益补偿
        rx.ctle.dc_gain = 2.0;
        rx.vga.dc_gain = 4.0;
        
        output_prefix = "nrz_10g_long";
    }
    
    /**
     * @brief 短信道配置 (低损耗场景)
     */
    void configure_short_channel() {
        channel.attenuation_db = 3.0;
        channel.bandwidth_hz = 30e9;
        
        // 无预加重
        tx.ffe.taps = {1.0};
        
        // 降低增益避免饱和
        rx.ctle.dc_gain = 0.8;
        rx.vga.dc_gain = 1.5;
        
        output_prefix = "nrz_10g_short";
    }
    
    /**
     * @brief 禁用自适应 (固定参数仿真)
     */
    void disable_adaption() {
        adaption.agc.enabled = false;
        adaption.dfe.enabled = false;
        adaption.vref_adapt.enabled = false;
        adaption.cdr_pi.enabled = false;
    }
    
    /**
     * @brief 设置仿真时长 (按UI数)
     */
    void set_duration_ui(int ui_count) {
        sim_duration = ui_count * ui();
    }
    
    /**
     * @brief 打印配置摘要
     */
    void print_summary() const {
        std::cout << "\n+----------------------------------------------+" << std::endl;
        std::cout << "|       NRZ Link Configuration Summary         |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Timing:                                      |" << std::endl;
        std::cout << "|   Data Rate:     " << std::setw(10) << data_rate / 1e9 
                  << " Gbps        |" << std::endl;
        std::cout << "|   UI:            " << std::setw(10) << ui() * 1e12 
                  << " ps          |" << std::endl;
        std::cout << "|   Sample Rate:   " << std::setw(10) << sample_rate() / 1e9 
                  << " GHz         |" << std::endl;
        std::cout << "|   Timestep:      " << std::setw(10) << timestep_ps() 
                  << " ps          |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Simulation:                                  |" << std::endl;
        std::cout << "|   Duration:      " << std::setw(10) << sim_duration * 1e6 
                  << " us          |" << std::endl;
        std::cout << "|   UI Count:      " << std::setw(10) << sim_ui_count() 
                  << "             |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Channel:                                     |" << std::endl;
        std::cout << "|   Method:        " << std::setw(10) 
                  << (channel_ext.method == ChannelMethod::SIMPLE ? "SIMPLE" : "STATE_SPACE") 
                  << "             |" << std::endl;
        std::cout << "|   Attenuation:   " << std::setw(10) << channel.attenuation_db 
                  << " dB          |" << std::endl;
        std::cout << "|   Bandwidth:     " << std::setw(10) << channel.bandwidth_hz / 1e9 
                  << " GHz         |" << std::endl;
        std::cout << "+----------------------------------------------+" << std::endl;
        std::cout << "| Equalization:                                |" << std::endl;
        std::cout << "|   FFE Taps:      " << std::setw(10) << tx.ffe.taps.size() 
                  << "             |" << std::endl;
        std::cout << "|   DFE Taps:      " << std::setw(10) << rx.dfe_summer.tap_coeffs.size() 
                  << "             |" << std::endl;
        std::cout << "|   Adaption:      " << std::setw(10) 
                  << (adaption.agc.enabled ? "ENABLED" : "DISABLED") 
                  << "             |" << std::endl;
        std::cout << "+----------------------------------------------+\n" << std::endl;
    }
};

} // namespace serdes

#endif // NRZ_LINK_CONFIG_H
