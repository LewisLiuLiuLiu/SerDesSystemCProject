/**
 * @file sampler_test_common.h
 * @brief Common test infrastructure for RxSamplerTdf unit tests
 */

#ifndef SERDES_TESTS_SAMPLER_TEST_COMMON_H
#define SERDES_TESTS_SAMPLER_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include "ams/rx_sampler.h"
#include "common/parameters.h"

namespace serdes {
namespace test {

// ============================================================================
// Differential signal source module for testing
// ============================================================================

class DifferentialSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    double m_amplitude;
    double m_vcm;
    
    DifferentialSource(sc_core::sc_module_name nm, double amplitude, double vcm)
        : sca_tdf::sca_module(nm)
        , out_p("out_p")
        , out_n("out_n")
        , m_amplitude(amplitude)
        , m_vcm(vcm)
    {}
    
    void set_attributes() {
        out_p.set_rate(1);
        out_n.set_rate(1);
        out_p.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
        out_n.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() {
        out_p.write(m_vcm + 0.5 * m_amplitude);
        out_n.write(m_vcm - 0.5 * m_amplitude);
    }
};

// ============================================================================
// Clock source module for testing
// ============================================================================

class ClockSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> clk_out;
    
    double m_frequency;
    double m_amplitude;
    double m_vcm;
    int m_step_count;
    
    ClockSource(sc_core::sc_module_name nm, double frequency, double amplitude, double vcm)
        : sca_tdf::sca_module(nm)
        , clk_out("clk_out")
        , m_frequency(frequency)
        , m_amplitude(amplitude)
        , m_vcm(vcm)
        , m_step_count(0)
    {}
    
    void set_attributes() {
        clk_out.set_rate(1);
        clk_out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() {
        double t = m_step_count * 1.0 / 100e9;
        double clk = m_vcm + 0.5 * m_amplitude * sin(2.0 * M_PI * m_frequency * t);
        clk_out.write(clk);
        m_step_count++;
    }
};

// ============================================================================
// Constant value source for phase offset
// ============================================================================

class ConstantSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    double m_value;
    
    ConstantSource(sc_core::sc_module_name nm, double value)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_value(value)
    {}
    
    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() {
        out.write(m_value);
    }
};

// ============================================================================
// Periodic sampling trigger source (replaces removed phase_offset input)
// ============================================================================
class TriggerSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<bool> trigger_out;
    
    double m_timestep;
    int m_period_samples;
    unsigned long m_step_count;
    
    TriggerSource(sc_core::sc_module_name nm, double trigger_rate = 10e9, double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , trigger_out("trigger_out")
        , m_timestep(1.0 / sample_rate)
        , m_period_samples(static_cast<int>(std::lround(sample_rate / trigger_rate)))
        , m_step_count(0)
    {
        if (m_period_samples < 1) m_period_samples = 1;
    }
    
    void set_attributes() {
        trigger_out.set_rate(1);
        trigger_out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        trigger_out.write((m_step_count % m_period_samples) == 0);
        m_step_count++;
    }
};

// ============================================================================
// Testbench for sampler basic functionality
// ============================================================================

SC_MODULE(SamplerBasicTestbench) {
    DifferentialSource* src;
    ClockSource* clk_src;
    ConstantSource* phase_src;
    TriggerSource* trigger_src;
    RxSamplerTdf* sampler;
    
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_clk;
    sca_tdf::sca_signal<double> sig_phase;
    sca_tdf::sca_signal<double> sig_out;
    sca_tdf::sca_signal<bool> sig_trigger;
    sc_core::sc_signal<bool> sig_out_de;  // DE domain signal for sca_tdf::sca_de::sca_out
    
    RxSamplerParams params;
    double input_amplitude;
    
    SamplerBasicTestbench(sc_core::sc_module_name nm, const RxSamplerParams& p, double amp)
        : sc_core::sc_module(nm)
        , params(p)
        , input_amplitude(amp)
    {
        // Create modules
        src = new DifferentialSource("src", input_amplitude, 0.6);
        clk_src = new ClockSource("clk_src", 10e9, 1.0, 0.5);
        phase_src = new ConstantSource("phase_src", 0.0);  // Zero phase offset
        trigger_src = new TriggerSource("trigger_src", 10e9, 100e9);
        sampler = new RxSamplerTdf("sampler", params);
        
        // Connect signals
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        clk_src->clk_out(sig_clk);
        phase_src->out(sig_phase);
        trigger_src->trigger_out(sig_trigger);
        
        sampler->in_p(sig_in_p);
        sampler->in_n(sig_in_n);
        sampler->clk_sample(sig_clk);
        sampler->sampling_trigger(sig_trigger);
        sampler->data_out(sig_out);
        sampler->data_out_de(sig_out_de);
    }
    
    ~SamplerBasicTestbench() {
        // SystemC modules are automatically managed by the simulator
        // Do not delete them manually
    }
    
    double get_output() {
        return sig_out.read(0);
    }
    
    bool get_output_de() {
        return sig_out_de.read();
    }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_SAMPLER_TEST_COMMON_H
