/**
 * @file tx_top_test_common.h
 * @brief Common test infrastructure for TxTopModule unit tests
 */

#ifndef SERDES_TESTS_TX_TOP_TEST_COMMON_H
#define SERDES_TESTS_TX_TOP_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/tx_top.h"
#include "ams/wave_generation.h"
#include <cmath>
#include <vector>

namespace serdes {
namespace test {

// ============================================================================
// Single-ended signal source for TX input
// ============================================================================

class TxSignalSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    enum WaveformType { DC, SINE, STEP, SQUARE, PRBS };
    
    TxSignalSource(sc_core::sc_module_name nm,
                   WaveformType type = DC,
                   double amplitude = 1.0,
                   double frequency = 10e9,
                   double step_time = 1e-9)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_step_time(step_time)
        , m_prbs_state(0x7FFFFFFF)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double v = 0.0;
        
        switch (m_type) {
            case DC:
                v = m_amplitude;
                break;
            case SINE:
                v = m_amplitude * std::sin(2.0 * M_PI * m_frequency * t);
                break;
            case STEP:
                v = (t >= m_step_time) ? m_amplitude : -m_amplitude;
                break;
            case SQUARE:
                v = (std::fmod(t * m_frequency, 1.0) < 0.5) ? m_amplitude : -m_amplitude;
                break;
            case PRBS:
                v = generate_prbs_sample();
                break;
        }
        
        out.write(v);
    }
    
    void set_amplitude(double amp) { m_amplitude = amp; }
    void set_frequency(double freq) { m_frequency = freq; }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
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

class TxConstantVddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    TxConstantVddSource(sc_core::sc_module_name nm, double voltage = 1.0)
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
// Differential signal monitor for capturing TX output
// ============================================================================

class TxDifferentialMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    std::vector<double> samples_p;
    std::vector<double> samples_n;
    std::vector<double> samples_diff;
    std::vector<double> samples_cm;
    std::vector<double> time_stamps;
    
    TxDifferentialMonitor(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , in_p("in_p")
        , in_n("in_n")
    {}
    
    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double vp = in_p.read();
        double vn = in_n.read();
        
        samples_p.push_back(vp);
        samples_n.push_back(vn);
        samples_diff.push_back(vp - vn);
        samples_cm.push_back(0.5 * (vp + vn));
        time_stamps.push_back(get_time().to_seconds());
    }
    
    void clear() {
        samples_p.clear();
        samples_n.clear();
        samples_diff.clear();
        samples_cm.clear();
        time_stamps.clear();
    }
    
    // Get DC value of differential output (skip initial transient)
    double get_dc_diff() const {
        if (samples_diff.empty()) return 0.0;
        size_t start = samples_diff.size() / 10;
        if (start < 1) start = 1;
        double sum = 0.0;
        for (size_t i = start; i < samples_diff.size(); ++i) {
            sum += samples_diff[i];
        }
        return sum / (samples_diff.size() - start);
    }
    
    // Get DC value of common-mode output
    double get_dc_cm() const {
        if (samples_cm.empty()) return 0.0;
        size_t start = samples_cm.size() / 10;
        if (start < 1) start = 1;
        double sum = 0.0;
        for (size_t i = start; i < samples_cm.size(); ++i) {
            sum += samples_cm[i];
        }
        return sum / (samples_cm.size() - start);
    }
    
    // Get RMS value of differential output
    double get_rms_diff() const {
        if (samples_diff.empty()) return 0.0;
        size_t start = samples_diff.size() / 10;
        if (start < 1) start = 1;
        double sum_sq = 0.0;
        for (size_t i = start; i < samples_diff.size(); ++i) {
            sum_sq += samples_diff[i] * samples_diff[i];
        }
        return std::sqrt(sum_sq / (samples_diff.size() - start));
    }
    
    // Get peak-to-peak value of differential output
    double get_pp_diff() const {
        if (samples_diff.empty()) return 0.0;
        size_t start = samples_diff.size() / 10;
        if (start < 1) start = 1;
        double min_val = samples_diff[start];
        double max_val = samples_diff[start];
        for (size_t i = start; i < samples_diff.size(); ++i) {
            if (samples_diff[i] < min_val) min_val = samples_diff[i];
            if (samples_diff[i] > max_val) max_val = samples_diff[i];
        }
        return max_val - min_val;
    }
    
    // Check if output is symmetric (|out_p| ≈ |out_n|)
    bool is_symmetric(double tolerance = 0.05) const {
        if (samples_p.empty() || samples_n.empty()) return false;
        size_t start = samples_p.size() / 10;
        if (start < 1) start = 1;
        for (size_t i = start; i < samples_p.size(); ++i) {
            double p_from_cm = samples_p[i] - samples_cm[i];
            double n_from_cm = samples_n[i] - samples_cm[i];
            if (std::abs(p_from_cm + n_from_cm) > tolerance * std::max(std::abs(p_from_cm), std::abs(n_from_cm))) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// TX Top Testbench
// ============================================================================

class TxTopTestbench {
public:
    TxSignalSource* src;
    TxConstantVddSource* vdd_src;
    TxTopModule* dut;
    TxDifferentialMonitor* monitor;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_out_p;
    sca_tdf::sca_signal<double> sig_out_n;
    
    TxTopTestbench(const TxParams& params,
                   TxSignalSource::WaveformType src_type = TxSignalSource::DC,
                   double src_amplitude = 1.0,
                   double src_frequency = 10e9,
                   double vdd_nominal = 1.0)
        : sig_in("sig_in")
        , sig_vdd("sig_vdd")
        , sig_out_p("sig_out_p")
        , sig_out_n("sig_out_n")
    {
        // Create signal source
        src = new TxSignalSource("src", src_type, src_amplitude, src_frequency);
        
        // Create VDD source
        vdd_src = new TxConstantVddSource("vdd_src", vdd_nominal);
        
        // Create DUT
        dut = new TxTopModule("dut", params);
        
        // Create monitor
        monitor = new TxDifferentialMonitor("monitor");
        
        // Connect signals
        src->out(sig_in);
        vdd_src->out(sig_vdd);
        
        dut->in(sig_in);
        dut->vdd(sig_vdd);
        dut->out_p(sig_out_p);
        dut->out_n(sig_out_n);
        
        monitor->in_p(sig_out_p);
        monitor->in_n(sig_out_n);
    }
    
    ~TxTopTestbench() {
        // SystemC modules are automatically managed by the simulator
    }
    
    // Convenience accessors
    const std::vector<double>& get_output_diff() const { return monitor->samples_diff; }
    const std::vector<double>& get_output_p() const { return monitor->samples_p; }
    const std::vector<double>& get_output_n() const { return monitor->samples_n; }
    const std::vector<double>& get_time() const { return monitor->time_stamps; }
};

// ============================================================================
// TX Top Testbench with WaveGen (for more realistic testing)
// ============================================================================

class TxTopTestbenchWithWaveGen {
public:
    WaveGenerationTdf* wavegen;
    TxConstantVddSource* vdd_src;
    TxTopModule* dut;
    TxDifferentialMonitor* monitor;
    
    sca_tdf::sca_signal<double> sig_wavegen_out;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_out_p;
    sca_tdf::sca_signal<double> sig_out_n;
    
    TxTopTestbenchWithWaveGen(const WaveGenParams& wave_params,
                              const TxParams& tx_params,
                              double sample_rate = 100e9,
                              unsigned int seed = 12345,
                              double vdd_nominal = 1.0)
        : sig_wavegen_out("sig_wavegen_out")
        , sig_vdd("sig_vdd")
        , sig_out_p("sig_out_p")
        , sig_out_n("sig_out_n")
    {
        // Create WaveGen
        wavegen = new WaveGenerationTdf("wavegen", wave_params, sample_rate, seed);
        
        // Create VDD source
        vdd_src = new TxConstantVddSource("vdd_src", vdd_nominal);
        
        // Create DUT
        dut = new TxTopModule("dut", tx_params);
        
        // Create monitor
        monitor = new TxDifferentialMonitor("monitor");
        
        // Connect signals
        wavegen->out(sig_wavegen_out);
        vdd_src->out(sig_vdd);
        
        dut->in(sig_wavegen_out);
        dut->vdd(sig_vdd);
        dut->out_p(sig_out_p);
        dut->out_n(sig_out_n);
        
        monitor->in_p(sig_out_p);
        monitor->in_n(sig_out_n);
    }
    
    ~TxTopTestbenchWithWaveGen() {
        // SystemC modules are automatically managed by the simulator
    }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_TX_TOP_TEST_COMMON_H
