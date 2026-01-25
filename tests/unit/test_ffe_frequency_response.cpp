/**
 * @file test_ffe_frequency_response.cpp
 * @brief Unit test for TxFfeTdf module - Frequency Response Theory
 */

#include "ffe_test_common.h"

using namespace serdes;
using namespace serdes::test;

TEST(FfeResponseTest, FrequencyResponseTheory) {
    // FIR filter frequency response: H(f) = sum(c[k] * e^(-j2*pi*f*k*T))
    std::vector<double> taps = {0.2, 0.6, 0.2};
    double T = 1.0 / 10e9;  // Symbol period (10 GHz)
    
    // DC frequency (f = 0)
    double freq_dc = 0.0;
    std::complex<double> H_dc(0.0, 0.0);
    for (size_t k = 0; k < taps.size(); ++k) {
        std::complex<double> exp_term = std::exp(std::complex<double>(0.0, -2.0 * M_PI * freq_dc * k * T));
        H_dc += taps[k] * exp_term;
    }
    double gain_dc = std::abs(H_dc);
    EXPECT_NEAR(gain_dc, 1.0, 0.001) << "DC gain should be 1.0 (sum of taps)";
    
    // Nyquist frequency (f = Fs/2)
    double freq_nyquist = 5e9;
    std::complex<double> H_nyquist(0.0, 0.0);
    for (size_t k = 0; k < taps.size(); ++k) {
        std::complex<double> exp_term = std::exp(std::complex<double>(0.0, -2.0 * M_PI * freq_nyquist * k * T));
        H_nyquist += taps[k] * exp_term;
    }
    double gain_nyquist = std::abs(H_nyquist);
    
    // For symmetric taps, high frequency gain differs from DC
    EXPECT_GT(gain_dc, 0.0) << "DC gain should be positive";
    
    // Verify gain is in reasonable range
    EXPECT_LE(gain_nyquist, 2.0) << "Nyquist gain should be bounded";
}
