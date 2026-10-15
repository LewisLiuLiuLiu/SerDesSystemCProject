/**
 * @file test_channel_sparam.cpp
 * @brief Unit tests for ChannelSParamTdf module
 * 
 * Tests cover:
 * 1. Simple model (v0.4 compatible)
 * 2. State-space model data structures (unified VF entry)
 * 3. Configuration loading (JSON parsing)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <sstream>

#include "common/parameters.h"
#include "ams/channel_sparam.h"
#include "../third_party/json.hpp"

// Note: SystemC-AMS headers are included via the channel_sparam.h
// These tests focus on non-simulation aspects that can be unit tested
// Full integration tests require SystemC simulation infrastructure

namespace serdes {
namespace test {

/**
 * Test fixture for channel parameter tests
 */
class ChannelParamTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup default parameters
        basic_params.touchstone = "";
        basic_params.ports = 2;
        basic_params.crosstalk = false;
        basic_params.bidirectional = false;
        basic_params.attenuation_db = 6.0;
        basic_params.bandwidth_hz = 25e9;
    }
    
    ChannelParams basic_params;
};

/**
 * Test default parameter values
 */
TEST_F(ChannelParamTest, DefaultValues) {
    ChannelParams p;
    EXPECT_EQ(p.touchstone, "");
    EXPECT_EQ(p.ports, 2);
    EXPECT_FALSE(p.crosstalk);
    EXPECT_FALSE(p.bidirectional);
    EXPECT_DOUBLE_EQ(p.attenuation_db, 10.0);
    EXPECT_DOUBLE_EQ(p.bandwidth_hz, 20e9);
}

/**
 * Test attenuation calculation
 */
TEST_F(ChannelParamTest, AttenuationLinear) {
    // Test various dB values
    struct TestCase {
        double db;
        double expected_linear;
    };
    
    std::vector<TestCase> cases = {
        {0.0, 1.0},
        {6.0, 0.5012},    // ~0.5 (6 dB = half voltage)
        {20.0, 0.1},      // 20 dB = 0.1
        {40.0, 0.01},     // 40 dB = 0.01
    };
    
    for (const auto& tc : cases) {
        double linear = std::pow(10.0, -tc.db / 20.0);
        EXPECT_NEAR(linear, tc.expected_linear, 0.001) 
            << "Failed for " << tc.db << " dB";
    }
}

/**
 * Test extended channel method enumeration
 */
TEST(ChannelMethodTest, EnumValues) {
    // 当前仅保留两种方法：SIMPLE 与 STATE_SPACE（统一 VF 入口）
    EXPECT_NE(static_cast<int>(ChannelMethod::SIMPLE), 
              static_cast<int>(ChannelMethod::STATE_SPACE));
}

/**
 * Test extended parameters defaults
 */
TEST(ChannelExtendedParamsTest, DefaultValues) {
    // 扩展参数现仅包含方法选择与配置文件路径（统一 VF/状态空间入口）
    ChannelExtendedParams ext;
    
    EXPECT_EQ(ext.method, ChannelMethod::SIMPLE);
    EXPECT_TRUE(ext.config_file.empty());
}

/**
 * Test state-space data structure defaults
 */
TEST(StateSpaceDataTest, DefaultValues) {
    StateSpaceData ss;
    
    EXPECT_EQ(ss.n_states, 0);
    EXPECT_EQ(ss.n_inputs, 1);
    EXPECT_EQ(ss.n_outputs, 1);
}

/**
 * Test full-model data structure defaults
 */
TEST(FullModelDataTest, DefaultValues) {
    FullModelData fm;
    
    EXPECT_EQ(fm.n_diff_ports, 0);
    EXPECT_EQ(fm.n_outputs, 0);
    EXPECT_EQ(fm.n_states, 0);
    EXPECT_TRUE(fm.port_pairs.empty());
    EXPECT_TRUE(fm.delays.empty());
}

/**
 * Test JSON configuration parsing (mock test without full SystemC)
 */
TEST(ConfigParsingTest, ValidJSON) {
    // Create a valid JSON config string
    std::string json_config = R"({
        "version": "1.0",
        "fs": 100e9,
        "method": "rational",
        "filters": {
            "S21": {
                "num": [0.7943, 1.2e-10],
                "den": [1.0, 1.8e-10],
                "order": 8,
                "dc_gain": 0.7943,
                "mse": 1.2e-4
            }
        }
    })";
    
    // Parse using nlohmann json directly
    try {
        auto config = nlohmann::json::parse(json_config);
        
        EXPECT_EQ(config["version"], "1.0");
        EXPECT_EQ(config["method"], "rational");
        EXPECT_TRUE(config.contains("filters"));
        EXPECT_TRUE(config["filters"].contains("S21"));
        
        auto& s21 = config["filters"]["S21"];
        EXPECT_EQ(s21["order"], 8);
        EXPECT_NEAR(s21["dc_gain"].get<double>(), 0.7943, 1e-4);
        
    } catch (const std::exception& e) {
        FAIL() << "JSON parsing failed: " << e.what();
    }
}

/**
 * Test invalid JSON handling
 */
TEST(ConfigParsingTest, InvalidJSON) {
    std::string invalid_json = "{ invalid json }";
    
    EXPECT_THROW({
        auto config = nlohmann::json::parse(invalid_json);
    }, nlohmann::json::parse_error);
}

/**
 * Test first-order filter coefficient calculation
 */
TEST(FilterCoefficientTest, FirstOrderLowpass) {
    // Test bilinear transform approximation for first-order lowpass
    double bandwidth_hz = 20e9;
    double fs = 100e9;
    double dt = 1.0 / fs;
    
    double omega_c = 2.0 * M_PI * bandwidth_hz;
    double alpha = omega_c * dt / (1.0 + omega_c * dt);
    
    // Alpha should be between 0 and 1
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 1.0);
    
    // For bandwidth = 20 GHz and fs = 100 GHz
    // alpha should be approximately 0.56
    EXPECT_NEAR(alpha, 0.56, 0.1);
}

/**
 * Test DC gain calculation from impulse response
 */
TEST(ImpulseResponseTest, DCGainFromSum) {
    // DC gain = integral of impulse response = sum(h) * dt
    std::vector<double> impulse = {0.1, 0.3, 0.4, 0.15, 0.05};
    double dt = 1e-11;  // 10 ps
    
    double dc_gain = 0.0;
    for (double h : impulse) {
        dc_gain += h;
    }
    dc_gain *= dt;
    
    // Expected: (0.1 + 0.3 + 0.4 + 0.15 + 0.05) * 1e-11 = 1e-11
    EXPECT_NEAR(dc_gain, 1e-11, 1e-13);
}

/**
 * Test circular buffer index calculation
 */
TEST(ConvolutionTest, CircularBufferIndex) {
    int L = 5;  // Buffer length
    
    // Test various index calculations
    for (int delay_idx = 0; delay_idx < L; ++delay_idx) {
        for (int k = 0; k < L; ++k) {
            int buf_pos = (delay_idx - k + L) % L;
            
            // Should always be in range [0, L)
            EXPECT_GE(buf_pos, 0);
            EXPECT_LT(buf_pos, L);
        }
    }
}

/**
 * Test direct convolution calculation
 */
TEST(ConvolutionTest, DirectConvolution) {
    // Simple test: delta function convolution
    std::vector<double> h = {1.0, 0.0, 0.0};  // Delta at t=0
    std::vector<double> x = {0.5, 1.0, 0.5, 0.0};  // Input signal
    
    int L = h.size();
    int N = x.size();
    std::vector<double> y(N, 0.0);
    
    // Direct convolution y[n] = sum_k h[k] * x[n-k]
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < L; ++k) {
            int idx = n - k;
            if (idx >= 0 && idx < N) {
                y[n] += h[k] * x[idx];
            }
        }
    }
    
    // For delta function, output should equal input
    EXPECT_NEAR(y[0], 0.5, 1e-10);
    EXPECT_NEAR(y[1], 1.0, 1e-10);
    EXPECT_NEAR(y[2], 0.5, 1e-10);
    EXPECT_NEAR(y[3], 0.0, 1e-10);
}

/**
 * Test FFT size calculation
 */
TEST(FFTTest, SizeCalculation) {
    // FFT size should be next power of 2 >= 2*L
    std::vector<std::pair<int, int>> test_cases = {
        {100, 256},    // 2*100 = 200, next power of 2 = 256
        {256, 512},    // 2*256 = 512
        {512, 1024},   // 2*512 = 1024
        {1000, 2048},  // 2*1000 = 2000, next power of 2 = 2048
    };
    
    for (const auto& tc : test_cases) {
        int L = tc.first;
        int expected_fft_size = tc.second;
        
        int fft_size = 1;
        while (fft_size < 2 * L) {
            fft_size *= 2;
        }
        
        EXPECT_EQ(fft_size, expected_fft_size) 
            << "Failed for L=" << L;
    }
}

} // namespace test
} // namespace serdes
