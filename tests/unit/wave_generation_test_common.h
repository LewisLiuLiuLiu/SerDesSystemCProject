/**
 * @file wave_generation_test_common.h
 * @brief Common test infrastructure for WaveGenerationTdf unit tests
 */

#ifndef SERDES_TESTS_WAVE_GENERATION_TEST_COMMON_H
#define SERDES_TESTS_WAVE_GENERATION_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include <vector>
#include "ams/wave_generation.h"
#include "common/parameters.h"

namespace serdes {
namespace test {

// ============================================================================
// Test Helper: Simple Receiver Module
// ============================================================================

class SimpleReceiver : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    
    std::vector<double> m_samples;
    size_t m_max_samples;
    
    SimpleReceiver(sc_core::sc_module_name nm, size_t max_samples = 1000)
        : sca_tdf::sca_module(nm)
        , in("in")
        , m_max_samples(max_samples)
    {}
    
    void set_attributes() {
        in.set_rate(1);
    }
    
    void processing() {
        if (m_samples.size() < m_max_samples) {
            m_samples.push_back(in.read());
        }
    }
    
    const std::vector<double>& get_samples() const { return m_samples; }
    void clear_samples() { m_samples.clear(); }
};

// ============================================================================
// Test Helper: Testbench Module
// ============================================================================

SC_MODULE(WaveGenTestbench) {
    WaveGenerationTdf* wave_gen;
    SimpleReceiver* receiver;
    
    sca_tdf::sca_signal<double> sig_wave;
    
    WaveGenParams params;
    double sample_rate;
    unsigned int seed;
    
    WaveGenTestbench(sc_core::sc_module_name nm, 
                     const WaveGenParams& p,
                     double sr = 80e9,
                     unsigned int s = 12345,
                     size_t max_samples = 1000)
        : sc_core::sc_module(nm)
        , params(p)
        , sample_rate(sr)
        , seed(s)
    {
        wave_gen = new WaveGenerationTdf("wave_gen", params, sample_rate, seed);
        receiver = new SimpleReceiver("receiver", max_samples);
        
        wave_gen->out(sig_wave);
        receiver->in(sig_wave);
    }
    
    ~WaveGenTestbench() {
        // SystemC modules are automatically managed by the simulator
    }
    
    const std::vector<double>& get_samples() const {
        return receiver->get_samples();
    }
    
    unsigned int get_lfsr_state() const {
        return wave_gen->get_lfsr_state();
    }
    
    double get_current_time() const {
        return wave_gen->get_current_time();
    }
    
    bool is_pulse_mode() const {
        return wave_gen->is_pulse_mode();
    }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_WAVE_GENERATION_TEST_COMMON_H
