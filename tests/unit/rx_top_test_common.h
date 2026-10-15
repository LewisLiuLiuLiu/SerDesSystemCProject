/**
 * @file rx_top_test_common.h
 * @brief Common test infrastructure for RxTopModule unit tests
 */

#ifndef SERDES_TESTS_RX_TOP_TEST_COMMON_H
#define SERDES_TESTS_RX_TOP_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_top.h"
#include <cmath>
#include <string>
#include <vector>

namespace serdes {
namespace test {

// ============================================================================
// Differential signal source for RX input
// ============================================================================

class RxDifferentialSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    enum WaveformType { DC, SINE, STEP, SQUARE, PRBS };
    
    RxDifferentialSource(sc_core::sc_module_name nm,
                         WaveformType type = DC,
                         double amplitude = 0.5,
                         double frequency = 10e9,
                         double vcm = 0.0,
                         double step_time = 1e-9)
        : sca_tdf::sca_module(nm)
        , out_p("out_p")
        , out_n("out_n")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_vcm(vcm)
        , m_step_time(step_time)
        , m_prbs_state(0x7FFFFFFF)
    {}
    
    void set_attributes() override {
        out_p.set_rate(1);
        out_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double v_diff = 0.0;
        
        switch (m_type) {
            case DC:
                v_diff = m_amplitude;
                break;
            case SINE:
                v_diff = m_amplitude * std::sin(2.0 * M_PI * m_frequency * t);
                break;
            case STEP:
                v_diff = (t >= m_step_time) ? m_amplitude : -m_amplitude;
                break;
            case SQUARE:
                v_diff = (std::fmod(t * m_frequency, 1.0) < 0.5) ? m_amplitude : -m_amplitude;
                break;
            case PRBS:
                v_diff = generate_prbs_sample();
                break;
        }
        
        // Differential output: out_p = vcm + v_diff/2, out_n = vcm - v_diff/2
        out_p.write(m_vcm + v_diff * 0.5);
        out_n.write(m_vcm - v_diff * 0.5);
    }
    
    void set_amplitude(double amp) { m_amplitude = amp; }
    void set_frequency(double freq) { m_frequency = freq; }
    void set_vcm(double vcm) { m_vcm = vcm; }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_vcm;
    double m_step_time;
    unsigned int m_prbs_state;
    
    double generate_prbs_sample() {
        // PRBS-31: x^31 + x^28 + 1
        bool bit = ((m_prbs_state >> 30) ^ (m_prbs_state >> 27)) & 1;
        m_prbs_state = (m_prbs_state << 1) | bit;
        return bit ? m_amplitude : -m_amplitude;
    }
};

// ============================================================================
// Constant VDD source
// ============================================================================

class RxConstantVddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    RxConstantVddSource(sc_core::sc_module_name nm, double voltage = 1.0)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_voltage(voltage)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        out.write(m_voltage);
    }
    
    void set_voltage(double v) { m_voltage = v; }
    
private:
    double m_voltage;
};

// ============================================================================
// Data output monitor for capturing RX sampler output
// ============================================================================

class RxDataMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    std::vector<double> samples;
    std::vector<double> time_stamps;
    
    RxDataMonitor(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , in("in")
    {}
    
    void set_attributes() override {
        in.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        samples.push_back(in.read());
        time_stamps.push_back(get_time().to_seconds());
    }
    
    void clear() {
        samples.clear();
        time_stamps.clear();
    }
    
    // Count number of '1' bits (positive samples)
    size_t count_ones() const {
        size_t count = 0;
        for (double s : samples) {
            if (s > 0.5) ++count;
        }
        return count;
    }
    
    // Count number of '0' bits (negative/zero samples)
    size_t count_zeros() const {
        size_t count = 0;
        for (double s : samples) {
            if (s < 0.5) ++count;
        }
        return count;
    }
    
    // Count transitions (edge count)
    size_t count_transitions() const {
        if (samples.size() < 2) return 0;
        size_t count = 0;
        for (size_t i = 1; i < samples.size(); ++i) {
            bool prev = (samples[i-1] > 0.5);
            bool curr = (samples[i] > 0.5);
            if (prev != curr) ++count;
        }
        return count;
    }
    
    // Get last N samples
    std::vector<double> get_last_samples(size_t n) const {
        if (samples.size() <= n) return samples;
        return std::vector<double>(samples.end() - n, samples.end());
    }
    
    // Check if output is valid (has both 0s and 1s)
    bool is_valid_data() const {
        return count_ones() > 0 && count_zeros() > 0;
    }
};

// ============================================================================
// RX Top Testbench
// ============================================================================

class RxTopTestbench {
public:
    RxDifferentialSource* src;
    RxConstantVddSource* vdd_src;
    RxTopModule* dut;
    RxDataMonitor* monitor;
    
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_data_out;
    
    RxTopTestbench(const RxParams& params,
                   const AdaptionParams& adaption_params,
                   RxDifferentialSource::WaveformType src_type = RxDifferentialSource::DC,
                   double src_amplitude = 0.5,
                   double src_frequency = 10e9,
                   double vdd_nominal = 1.0,
                   const std::string& prefix = "tb")
        : sig_in_p((prefix + "_sig_in_p").c_str())
        , sig_in_n((prefix + "_sig_in_n").c_str())
        , sig_vdd((prefix + "_sig_vdd").c_str())
        , sig_data_out((prefix + "_sig_data_out").c_str())
    {
        // Create differential signal source
        src = new RxDifferentialSource((prefix + "_src").c_str(), src_type, src_amplitude, src_frequency);
        
        // Create VDD source
        vdd_src = new RxConstantVddSource((prefix + "_vdd_src").c_str(), vdd_nominal);
        
        // Create DUT with adaption parameters
        dut = new RxTopModule((prefix + "_dut").c_str(), params, adaption_params);
        
        // Create monitor
        monitor = new RxDataMonitor((prefix + "_monitor").c_str());
        
        // Connect signals
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        vdd_src->out(sig_vdd);
        
        dut->in_p(sig_in_p);
        dut->in_n(sig_in_n);
        dut->vdd(sig_vdd);
        dut->data_out(sig_data_out);
        
        monitor->in(sig_data_out);
    }
    
    ~RxTopTestbench() {
        delete src;
        delete vdd_src;
        delete dut;
        delete monitor;
    }
    
    // Convenience accessors
    const std::vector<double>& get_output_samples() const { 
        return monitor->samples; 
    }
    
    size_t get_sample_count() const { 
        return monitor->samples.size(); 
    }
    
    double get_cdr_phase() const { 
        return dut->get_cdr_phase(); 
    }
    
    double get_cdr_integral_state() const { 
        return dut->get_cdr_integral_state(); 
    }
};

// ============================================================================
// Default test parameters factory
// ============================================================================

inline RxParams get_default_rx_params() {
    RxParams params;
    
    // CTLE: moderate boost
    params.ctle.zeros = {2e9};
    params.ctle.poles = {30e9};
    params.ctle.dc_gain = 1.5;
    params.ctle.vcm_out = 0.0;
    
    // VGA: unity gain
    params.vga.zeros = {1e9};
    params.vga.poles = {20e9};
    params.vga.dc_gain = 2.0;
    params.vga.vcm_out = 0.0;
    
    // DFE Summer: 3 taps (差分架构)
    params.dfe_summer.tap_coeffs = {-0.05, -0.02, 0.01};
    params.dfe_summer.ui = 100e-12;
    params.dfe_summer.vcm_out = 0.0;
    params.dfe_summer.vtap = 1.0;
    params.dfe_summer.map_mode = "pm1";
    params.dfe_summer.enable = true;
    
    // Sampler: phase-driven mode (will be forced by RxTopModule)
    params.sampler.phase_source = "phase";
    params.sampler.threshold = 0.0;
    params.sampler.hysteresis = 0.01;
    params.sampler.resolution = 0.02;
    
    // CDR: moderate gains
    params.cdr.pi.kp = 0.01;
    params.cdr.pi.ki = 1e-4;
    params.cdr.pi.edge_threshold = 0.5;
    params.cdr.pai.resolution = 1e-12;
    params.cdr.pai.range = 5e-11;
    params.cdr.ui = 100e-12;  // 10 Gbps
    
    return params;
}

// ============================================================================
// Default Adaption parameters factory
// ============================================================================

inline AdaptionParams get_default_adaption_params() {
    AdaptionParams params;
    
    // 基本参数
    params.Fs = 80e9;
    params.UI = 100e-12;  // 10 Gbps
    params.seed = 12345;
    params.update_mode = "multi-rate";
    params.fast_update_period = 2.5e-10;
    params.slow_update_period = 2.5e-7;
    
    // AGC 参数（禁用，使用固定增益）
    params.agc.enabled = false;
    params.agc.initial_gain = 2.0;
    
    // DFE 自适应参数（禁用，使用静态抽头）
    params.dfe.enabled = false;
    params.dfe.num_taps = 3;
    params.dfe.algorithm = "sign-lms";
    params.dfe.initial_taps = {-0.05, -0.02, 0.01};
    
    // Vref 自适应参数（禁用）
    params.vref_adapt.enabled = false;
    
    // CDR PI 参数（禁用，使用 RxCdrTdf 内部 PI）
    params.cdr_pi.enabled = false;
    
    // 安全参数（禁用测试模式下的冻结功能）
    params.safety.freeze_on_error = false;
    params.safety.rollback_enable = false;
    
    return params;
}

// ============================================================================
// Test parameter variants
// ============================================================================

inline RxParams get_high_gain_rx_params() {
    RxParams params = get_default_rx_params();
    params.ctle.dc_gain = 3.0;
    params.vga.dc_gain = 4.0;
    return params;
}

inline RxParams get_aggressive_cdr_params() {
    RxParams params = get_default_rx_params();
    params.cdr.pi.kp = 0.05;
    params.cdr.pi.ki = 5e-4;
    return params;
}

inline RxParams get_no_dfe_params() {
    RxParams params = get_default_rx_params();
    params.dfe_summer.tap_coeffs.clear();  // Empty taps = passthrough
    params.dfe_summer.enable = false;
    return params;
}

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_RX_TOP_TEST_COMMON_H
