/**
 * @file cdr_test_common.h
 * @brief Common test infrastructure for RxCdrTdf unit tests
 */

#ifndef SERDES_TESTS_CDR_TEST_COMMON_H
#define SERDES_TESTS_CDR_TEST_COMMON_H

#include <gtest/gtest.h>
#include <systemc-ams>
#include <cmath>
#include "ams/rx_cdr.h"
#include "common/parameters.h"

namespace serdes {
namespace test {

// ============================================================================
// Test Helper: Simple Data Source
// ============================================================================

class SimpleDataSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;

    std::vector<double> m_data_pattern;
    size_t m_index;

    SimpleDataSource(sc_core::sc_module_name nm, const std::vector<double>& pattern)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_data_pattern(pattern)
        , m_index(0)
    {}

    void set_attributes() {
        out.set_rate(1);
        out.set_timestep(1.0 / 10e9, sc_core::SC_SEC);  // 10 GHz
    }

    void processing() {
        if (!m_data_pattern.empty()) {
            out.write(m_data_pattern[m_index % m_data_pattern.size()]);
            m_index++;
        } else {
            out.write(0.0);
        }
    }
};

// ============================================================================
// Test Helper: Testbench Module
// ============================================================================

SC_MODULE(CdrBasicTestbench) {
    SimpleDataSource* src;
    RxCdrTdf* cdr;

    sca_tdf::sca_signal<double> sig_data;
    sca_tdf::sca_signal<double> sig_phase;
    sca_tdf::sca_signal<bool> sig_trigger;  // CDR sampling_trigger 输出（本测试不使用触发）
    sc_core::sc_signal<double> sig_phase_error_de;  // BBPD 误差 DE 输出（本测试不监测）

    CdrParams params;

    CdrBasicTestbench(sc_core::sc_module_name nm, const CdrParams& p, const std::vector<double>& pattern)
        : sc_core::sc_module(nm)
        , params(p)
    {
        src = new SimpleDataSource("src", pattern);
        cdr = new RxCdrTdf("cdr", params);

        src->out(sig_data);
        cdr->in(sig_data);
        cdr->phase_out(sig_phase);
        cdr->sampling_trigger(sig_trigger);
        cdr->phase_error_de(sig_phase_error_de);
    }

    ~CdrBasicTestbench() {
        // SystemC modules are automatically managed by the simulator
        // Do not delete them manually
    }

    double get_phase_output() {
        return sig_phase.read(0);
    }
    
    double get_integral_state() {
        return cdr->get_integral_state();
    }
    
    double get_phase_error() {
        return cdr->get_phase_error();
    }
};

} // namespace test
} // namespace serdes

#endif // SERDES_TESTS_CDR_TEST_COMMON_H
