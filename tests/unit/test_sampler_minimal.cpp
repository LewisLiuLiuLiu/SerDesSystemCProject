/**
 * @file test_sampler_minimal.cpp
 * @brief Minimal test to debug Sampler crash
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_sampler.h"
#include "common/parameters.h"

using namespace serdes;

namespace {
// Drives all sampler inputs (quiet, no trigger) so the TDF channels have drivers.
class ZeroDriver : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> p, n, clk;
    sca_tdf::sca_out<bool> trig;
    ZeroDriver(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), p("p"), n("n"), clk("clk"), trig("trig") {}
    void set_attributes() override {
        p.set_rate(1); n.set_rate(1); clk.set_rate(1); trig.set_rate(1);
        set_timestep(1e-11, sc_core::SC_SEC);
    }
    void processing() override {
        p.write(0.0); n.write(0.0); clk.write(0.0); trig.write(false);
    }
};
}

// Minimal test: Sampler with no trigger connection
TEST(SamplerMinimalTest, NoTrigger) {
    RxSamplerParams params;
    params.resolution = 0.1;
    params.hysteresis = 0.02;
    
    // Create minimal signals
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_clk;
    sca_tdf::sca_signal<bool> sig_trigger;
    sca_tdf::sca_signal<double> sig_out;
    sc_core::sc_signal<bool> sig_out_de;
    sc_core::sc_signal<double> sig_vref_thr_dummy;  // 动态门限占位（0 = 静态门限）
    
    // Create sampler and quiet drivers
    ZeroDriver drv("drv");
    drv.p(sig_in_p);
    drv.n(sig_in_n);
    drv.clk(sig_clk);
    drv.trig(sig_trigger);
    RxSamplerTdf* sampler = new RxSamplerTdf("sampler", params);
    
    // Connect only required ports
    sampler->in_p(sig_in_p);
    sampler->in_n(sig_in_n);
    sampler->clk_sample(sig_clk);
    sampler->sampling_trigger(sig_trigger);
    sampler->vref_threshold_de(sig_vref_thr_dummy);
    sampler->data_out(sig_out);
    sampler->data_out_de(sig_out_de);
    
    // Run short simulation
    sc_core::sc_start(1, sc_core::SC_NS);

    // No trigger ever fires: the sample-and-hold must stay at the initial 0
    EXPECT_DOUBLE_EQ(sig_out.read(0), 0.0);
    EXPECT_FALSE(sig_out_de.read());
}
