#include <gtest/gtest.h>
#include <systemc-ams>
#include <iostream>
#include "ams/rx_sampler.h"
#include "common/parameters.h"

using namespace serdes;

class TestSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    sca_tdf::sca_out<bool> trigger;
    
    TestSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out_p("out_p"), out_n("out_n"), trigger("trigger")
    {}
    
    void set_attributes() {
        set_timestep(10.0, sc_core::SC_PS);
        out_p.set_rate(1);
        out_n.set_rate(1);
        trigger.set_rate(1);
    }
    
    void processing() {
        out_p.write(0.7);
        out_n.write(0.5);
        trigger.write(true);
    }
};

class ClockSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    ClockSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out("out")
    {}
    
    void set_attributes() {
        set_timestep(10.0, sc_core::SC_PS);
        out.set_rate(1);
    }
    
    void processing() {
        out.write(1.0);
    }
};

SC_MODULE(SamplerTestbench) {
    TestSource* src;
    ClockSource* clk_src;
    RxSamplerTdf* sampler;
    
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_clk;
    sca_tdf::sca_signal<double> sig_out;
    sca_tdf::sca_signal<bool> sig_trigger;
    sc_core::sc_signal<bool> sig_out_de;
    
    SamplerTestbench(sc_core::sc_module_name nm, const RxSamplerParams& params)
        : sc_core::sc_module(nm)
    {
        src = new TestSource("src");
        clk_src = new ClockSource("clk_src");
        sampler = new RxSamplerTdf("sampler", params);
        
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        src->trigger(sig_trigger);
        
        clk_src->out(sig_clk);
        
        sampler->in_p(sig_in_p);
        sampler->in_n(sig_in_n);
        sampler->clk_sample(sig_clk);
        sampler->sampling_trigger(sig_trigger);
        sampler->data_out(sig_out);
        sampler->data_out_de(sig_out_de);
    }
    
    double get_output() {
        return sig_out.read(0);
    }
};

TEST(SamplerDebug, Basic) {
    std::cout << "[DEBUG] Starting test" << std::endl;
    
    RxSamplerParams params;
    params.resolution = 0.1;
    params.hysteresis = 0.02;
    params.offset_enable = false;
    params.noise_enable = false;
    
    std::cout << "[DEBUG] Creating testbench" << std::endl;
    SamplerTestbench* tb = new SamplerTestbench("tb", params);
    
    std::cout << "[DEBUG] Starting simulation" << std::endl;
    sc_core::sc_start(100, sc_core::SC_PS);
    
    std::cout << "[DEBUG] Reading output" << std::endl;
    double output = tb->get_output();
    std::cout << "[DEBUG] Output = " << output << std::endl;
    
    EXPECT_NEAR(output, 1.0, 0.1);
    std::cout << "[DEBUG] Test complete" << std::endl;
}

int sc_main(int argc, char** argv) {
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", sc_core::SC_DO_NOTHING);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
