// Simple test to verify sca_ltf_nd behavior
#include <systemc-ams>
#include <iostream>

using namespace sca_tdf;

int sc_main(int argc, char* argv[]) {
    // Test 1: Simple RC filter H(s) = 1/(1 + s/omega_c)
    // At DC: H(0) = 1
    // At omega_c: H(omega_c) = 1/sqrt(2) = 0.707
    
    sca_ltf_nd ltf;
    sca_util::sca_vector<double> num, den;
    
    // H(s) = 1 / (1 + s/100e9) = 100e9 / (s + 100e9)
    // Coefficients for sca_ltf_nd (ascending order: [constant, s, s^2, ...])
    // num = [100e9] (constant term)
    // den = [100e9, 1] (constant term, s term)
    
    num.resize(1);
    num(0) = 100e9;
    
    den.resize(2);
    den(0) = 100e9;  // constant term
    den(1) = 1;      // s term
    
    // Test at different frequencies
    std::cout << "Testing sca_ltf_nd with H(s) = 100e9/(s + 100e9)" << std::endl;
    std::cout << "Expected DC gain: 1.0" << std::endl;
    std::cout << "Expected at 100e9 rad/s: 0.707" << std::endl;
    
    // Note: sca_ltf_nd needs to be called during simulation
    // For now, just print the coefficients
    std::cout << "num[0] = " << num(0) << std::endl;
    std::cout << "den[0] = " << den(0) << ", den[1] = " << den(1) << std::endl;
    
    return 0;
}
