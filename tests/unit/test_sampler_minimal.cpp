/**
 * @file test_sampler_minimal.cpp
 * @brief Minimal test to debug Sampler crash
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/rx_sampler.h"
#include "common/parameters.h"

using namespace serdes;

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
    
    // Create sampler
    RxSamplerTdf* sampler = new RxSamplerTdf("sampler", params);
    
    // Connect only required ports
    sampler->in_p(sig_in_p);
    sampler->in_n(sig_in_n);
    sampler->clk_sample(sig_clk);
    sampler->sampling_trigger(sig_trigger);
    sampler->data_out(sig_out);
    sampler->data_out_de(sig_out_de);
    
    // Run short simulation
    sc_core::sc_start(1, sc_core::SC_NS);
    
    SUCCEED() << "Sampler initialized and ran without crash";
}

// SystemC main function
int sc_main(int argc, char** argv) {
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", sc_core::SC_DO_NOTHING);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
