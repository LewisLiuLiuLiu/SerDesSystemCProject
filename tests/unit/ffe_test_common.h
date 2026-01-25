/**
 * @file ffe_test_common.h
 * @brief Common test infrastructure for TxFfeTdf unit tests
 */

#ifndef SERDES_TESTS_FFE_TEST_COMMON_H
#define SERDES_TESTS_FFE_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include <complex>
#include <vector>
#include "ams/tx_ffe.h"
#include "common/parameters.h"

namespace serdes {
namespace test {

// ============================================================================
// Signal Source Module
// ============================================================================

class SignalSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    enum WaveformType {
        DC,
        SINE,
        SQUARE,
        IMPULSE,
        STEP,
        PRBS
    };
    
    SignalSource(sc_core::sc_module_name nm, 
                 WaveformType type = DC,
                 double amplitude = 1.0,
                 double frequency = 1e9,
                 double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_timestep(1.0 / sample_rate)
        , m_step_count(0)
    {}
    
    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        double signal = 0.0;
        double t = m_step_count * m_timestep;
        
        switch (m_type) {
            case DC:
                signal = m_amplitude;
                break;
            case SINE:
                signal = m_amplitude * sin(2.0 * M_PI * m_frequency * t);
                break;
            case SQUARE:
                signal = m_amplitude * (sin(2.0 * M_PI * m_frequency * t) > 0 ? 1.0 : -1.0);
                break;
            case IMPULSE:
                signal = (m_step_count == 0) ? m_amplitude : 0.0;
                break;
            case STEP:
                signal = m_amplitude;
                break;
            case PRBS:
                signal = m_amplitude * ((m_step_count % 127) < 64 ? 1.0 : -1.0);
                break;
        }
        
        out.write(signal);
        m_step_count++;
    }
    
    void reset() { m_step_count = 0; }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_timestep;
    unsigned long m_step_count;
};

// ============================================================================
// Signal Sink Module
// ============================================================================

class SignalSink : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    SignalSink(sc_core::sc_module_name nm, double sample_rate = 100e9)
        : sca_tdf::sca_module(nm)
        , in("in")
        , m_timestep(1.0 / sample_rate)
    {}
    
    void set_attributes() {
        in.set_rate(1);
        in.set_timestep(m_timestep, sc_core::SC_SEC);
    }
    
    void processing() {
        m_samples.push_back(in.read());
    }
    
    const std::vector<double>& get_samples() const { return m_samples; }
    
    double get_last() const { 
        return m_samples.empty() ? 0.0 : m_samples.back(); 
    }
    
    double get_mean() const {
        if (m_samples.empty()) return 0.0;
        double sum = 0.0;
        for (double v : m_samples) sum += v;
        return sum / m_samples.size();
    }
    
    double get_rms() const {
        if (m_samples.empty()) return 0.0;
        double sum_sq = 0.0;
        for (double v : m_samples) sum_sq += v * v;
        return sqrt(sum_sq / m_samples.size());
    }
    
    double get_max() const {
        if (m_samples.empty()) return 0.0;
        double max_val = m_samples[0];
        for (double v : m_samples) if (v > max_val) max_val = v;
        return max_val;
    }
    
    double get_min() const {
        if (m_samples.empty()) return 0.0;
        double min_val = m_samples[0];
        for (double v : m_samples) if (v < min_val) min_val = v;
        return min_val;
    }
    
    void clear() { m_samples.clear(); }
    
private:
    double m_timestep;
    std::vector<double> m_samples;
};

// ============================================================================
// FFE Testbench
// ============================================================================

SC_MODULE(FfeBasicTestbench) {
    SignalSource* src;
    TxFfeTdf* ffe;
    SignalSink* sink;
    
    sca_tdf::sca_signal<double> sig_in;
    sca_tdf::sca_signal<double> sig_out;
    
    TxFfeParams params;
    
    FfeBasicTestbench(sc_core::sc_module_name nm, 
                      const TxFfeParams& p,
                      SignalSource::WaveformType waveform = SignalSource::DC,
                      double amplitude = 1.0,
                      double frequency = 1e9)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SignalSource("src", waveform, amplitude, frequency);
        ffe = new TxFfeTdf("ffe", params);
        sink = new SignalSink("sink");
        
        src->out(sig_in);
        ffe->in(sig_in);
        ffe->out(sig_out);
        sink->in(sig_out);
    }
    
    ~FfeBasicTestbench() {
        // SystemC modules are automatically managed by the simulator
    }
    
    const std::vector<double>& get_output_samples() const {
        return sink->get_samples();
    }
    
    double get_output_last() const { return sink->get_last(); }
    double get_output_mean() const { return sink->get_mean(); }
    double get_output_rms() const { return sink->get_rms(); }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_FFE_TEST_COMMON_H
