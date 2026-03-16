// Test sca_ltf_nd coefficient format
#include <systemc-ams>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace sca_tdf;

SC_MODULE(TestSource) {
    sca_tdf::sca_out<double> out;
    
    void processing() {
        double t = get_time().to_seconds();
        // Multi-tone signal
        double signal = 0.0;
        for (int i = 1; i <= 10; ++i) {
            signal += std::sin(2 * M_PI * i * 1e9 * t);  // 1-10 GHz
        }
        out.write(signal);
    }
    
    void set_attributes() {
        set_timestep(10e-12, SC_SEC);  // 100 GHz
        out.set_rate(1);
    }
    
    SC_CTOR(TestSource) : out("out") {}
};

SC_MODULE(TestSink) {
    sca_tdf::sca_in<double> in;
    std::ofstream file;
    
    void processing() {
        double t = get_time().to_seconds();
        file << t << "," << in.read() << std::endl;
    }
    
    void set_attributes() {
        set_timestep(10e-12, SC_SEC);
        in.set_rate(1);
    }
    
    void start_of_simulation() {
        file.open("ltf_test_output.csv");
        file << "time,output" << std::endl;
    }
    
    void end_of_simulation() {
        file.close();
    }
    
    SC_CTOR(TestSink) : in("in") {}
};

int sc_main(int argc, char* argv[]) {
    // Test: H(s) = 1/(1 + s/1e10) (10 GHz lowpass)
    // num = [1] (constant)
    // den = [1, 1e-10] ??? Let's test both interpretations
    
    sc_signal<double> sig_in, sig_out;
    
    TestSource src("src");
    src.out(sig_in);
    
    sca_ltf_nd ltf("ltf");
    ltf.in(sig_in);
    ltf.out(sig_out);
    
    // Interpretation A: ascending [constant, s, s^2, ...]
    // H(s) = 1 / (1 + s/1e10) = 1 / (1 + 1e-10*s)
    // num = [1]
    // den = [1, 1e-10]
    sca_util::sca_vector<double> num, den;
    num.resize(1);
    num(0) = 1.0;
    den.resize(2);
    den(0) = 1.0;        // constant term
    den(1) = 1e-10;      // s term coefficient (1/1e10)
    
    // Note: sca_ltf_nd doesn't have direct set_coeff method
    // We need to use it in processing, but for now just print
    std::cout << "Testing sca_ltf_nd coefficient interpretation" << std::endl;
    std::cout << "H(s) = 1 / (1 + s/1e10)" << std::endl;
    std::cout << "num[0] = " << num(0) << std::endl;
    std::cout << "den[0] = " << den(0) << ", den[1] = " << den(1) << std::endl;
    std::cout << "Expected DC gain: 1.0" << std::endl;
    std::cout << "Expected at 10 GHz: 0.707" << std::endl;
    
    // We can't easily test without full simulation setup
    // For now, just return
    std::cout << "\nNote: Full test requires simulation" << std::endl;
    
    return 0;
}
