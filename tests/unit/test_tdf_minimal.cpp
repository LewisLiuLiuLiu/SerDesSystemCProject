#include <gtest/gtest.h>
#include <systemc-ams>
#include <iostream>

using namespace sca_tdf;

class Source : public sca_module {
public:
    sca_out<double> out;
    Source(sc_core::sc_module_name nm) : sca_module(nm), out("out") {}
    void set_attributes() {
        std::cout << "[SOURCE] set_attributes" << std::endl;
        set_timestep(100.0, sc_core::SC_PS);
        out.set_rate(1);
        std::cout << "[SOURCE] set_attributes done" << std::endl;
    }
    void initialize() {
        std::cout << "[SOURCE] initialize" << std::endl;
    }
    void processing() {
        std::cout << "[SOURCE] processing" << std::endl;
        out.write(1.0);
    }
};

class Sink : public sca_module {
public:
    sca_in<double> in;
    Sink(sc_core::sc_module_name nm) : sca_module(nm), in("in") {}
    void set_attributes() {
        std::cout << "[SINK] set_attributes" << std::endl;
        in.set_rate(1);
        std::cout << "[SINK] set_attributes done" << std::endl;
    }
    void initialize() {
        std::cout << "[SINK] initialize" << std::endl;
    }
    void processing() {
        std::cout << "[SINK] processing" << std::endl;
        double val = in.read();
        std::cout << "Sink received: " << val << std::endl;
    }
};

SC_MODULE(Testbench) {
    Source* src;
    Sink* sink;
    sca_signal<double> sig;
    
    Testbench(sc_core::sc_module_name nm) : sc_core::sc_module(nm) {
        src = new Source("src");
        sink = new Sink("sink");
        src->out(sig);
        sink->in(sig);
    }
};

TEST(TdfMinimal, Basic) {
    std::cout << "[TEST] Creating testbench" << std::endl;
    Testbench* tb = new Testbench("tb");
    
    std::cout << "[TEST] Starting simulation" << std::endl;
    sc_core::sc_start(500, sc_core::SC_PS);
    
    std::cout << "[TEST] Done" << std::endl;
    SUCCEED();
}

int sc_main(int argc, char** argv) {
    sc_core::sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", sc_core::SC_DO_NOTHING);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
