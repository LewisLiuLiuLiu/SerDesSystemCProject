/**
 * @file test_tx_driver.cpp
 * @brief Unit tests for TX Driver module
 * 
 * Tests the TxDriverTdf module for:
 * - DC gain accuracy
 * - Common-mode voltage control
 * - Soft/hard saturation
 * - Bandwidth limiting
 * - PSRR (power supply rejection ratio)
 * - Gain mismatch
 * - Slew rate limiting
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/tx_driver.h"
#include <cmath>
#include <vector>

using namespace serdes;

// ============================================================================
// Test Helper Modules
// ============================================================================

/**
 * @brief Differential signal source for testing
 */
class DifferentialSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;
    
    enum WaveformType { DC, SINE, STEP };
    
    DifferentialSource(sc_core::sc_module_name nm,
                       WaveformType type = DC,
                       double amplitude = 0.5,
                       double frequency = 1e9,
                       double vcm = 0.0,
                       double step_time = 1e-9)
        : sca_tdf::sca_module(nm)
        , out_p("out_p")
        , out_n("out_n")
        , m_type(type)
        , m_amplitude(amplitude)
        , m_frequency(frequency)
        , m_vcm(vcm)
        , m_step_time(step_time)
    {}
    
    void set_attributes() override {
        out_p.set_rate(1);
        out_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double v_diff = 0.0;
        
        switch (m_type) {
            case DC:
                v_diff = m_amplitude;
                break;
            case SINE:
                v_diff = m_amplitude * std::sin(2.0 * M_PI * m_frequency * t);
                break;
            case STEP:
                v_diff = (t >= m_step_time) ? m_amplitude : 0.0;
                break;
        }
        
        out_p.write(m_vcm + 0.5 * v_diff);
        out_n.write(m_vcm - 0.5 * v_diff);
    }
    
private:
    WaveformType m_type;
    double m_amplitude;
    double m_frequency;
    double m_vcm;
    double m_step_time;
};

/**
 * @brief Constant VDD source
 */
class ConstantVddSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    ConstantVddSource(sc_core::sc_module_name nm, double voltage = 1.0)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_voltage(voltage)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        out.write(m_voltage);
    }
    
private:
    double m_voltage;
};

/**
 * @brief VDD source with sinusoidal ripple
 */
class VddWithRipple : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    VddWithRipple(sc_core::sc_module_name nm, 
                  double nominal = 1.0,
                  double ripple_amplitude = 0.01,
                  double ripple_frequency = 100e6)
        : sca_tdf::sca_module(nm)
        , out("out")
        , m_nominal(nominal)
        , m_ripple_amp(ripple_amplitude)
        , m_ripple_freq(ripple_frequency)
    {}
    
    void set_attributes() override {
        out.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double t = get_time().to_seconds();
        double ripple = m_ripple_amp * std::sin(2.0 * M_PI * m_ripple_freq * t);
        out.write(m_nominal + ripple);
    }
    
private:
    double m_nominal;
    double m_ripple_amp;
    double m_ripple_freq;
};

/**
 * @brief Signal monitor/sink for capturing output
 */
class SignalMonitor : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;
    
    std::vector<double> samples_p;
    std::vector<double> samples_n;
    std::vector<double> samples_diff;
    std::vector<double> samples_cm;
    std::vector<double> time_stamps;
    
    SignalMonitor(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm)
        , in_p("in_p")
        , in_n("in_n")
    {}
    
    void set_attributes() override {
        in_p.set_rate(1);
        in_n.set_rate(1);
        set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    
    void processing() override {
        double vp = in_p.read();
        double vn = in_n.read();
        
        samples_p.push_back(vp);
        samples_n.push_back(vn);
        samples_diff.push_back(vp - vn);
        samples_cm.push_back(0.5 * (vp + vn));
        time_stamps.push_back(get_time().to_seconds());
    }
    
    void clear() {
        samples_p.clear();
        samples_n.clear();
        samples_diff.clear();
        samples_cm.clear();
        time_stamps.clear();
    }
    
    double get_dc_diff() const {
        if (samples_diff.empty()) return 0.0;
        // Skip first 10% for settling
        size_t start = samples_diff.size() / 10;
        double sum = 0.0;
        for (size_t i = start; i < samples_diff.size(); ++i) {
            sum += samples_diff[i];
        }
        return sum / (samples_diff.size() - start);
    }
    
    double get_dc_cm() const {
        if (samples_cm.empty()) return 0.0;
        size_t start = samples_cm.size() / 10;
        double sum = 0.0;
        for (size_t i = start; i < samples_cm.size(); ++i) {
            sum += samples_cm[i];
        }
        return sum / (samples_cm.size() - start);
    }
    
    double get_rms_diff() const {
        if (samples_diff.empty()) return 0.0;
        size_t start = samples_diff.size() / 10;
        double sum_sq = 0.0;
        for (size_t i = start; i < samples_diff.size(); ++i) {
            sum_sq += samples_diff[i] * samples_diff[i];
        }
        return std::sqrt(sum_sq / (samples_diff.size() - start));
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class TxDriverTestbench {
public:
    DifferentialSource* src;
    sca_tdf::sca_module* vdd_src;
    TxDriverTdf* dut;
    SignalMonitor* monitor;
    
    sca_tdf::sca_signal<double> sig_in_p;
    sca_tdf::sca_signal<double> sig_in_n;
    sca_tdf::sca_signal<double> sig_vdd;
    sca_tdf::sca_signal<double> sig_out_p;
    sca_tdf::sca_signal<double> sig_out_n;
    
    TxDriverTestbench(const TxDriverParams& params,
                      DifferentialSource::WaveformType src_type = DifferentialSource::DC,
                      double src_amplitude = 0.5,
                      double src_frequency = 1e9,
                      double vdd_nominal = 1.0,
                      bool use_vdd_ripple = false,
                      double ripple_amp = 0.01,
                      double ripple_freq = 100e6)
        : sig_in_p("sig_in_p")
        , sig_in_n("sig_in_n")
        , sig_vdd("sig_vdd")
        , sig_out_p("sig_out_p")
        , sig_out_n("sig_out_n")
    {
        // Create source
        src = new DifferentialSource("src", src_type, src_amplitude, src_frequency);
        
        // Create VDD source
        if (use_vdd_ripple) {
            vdd_src = new VddWithRipple("vdd_src", vdd_nominal, ripple_amp, ripple_freq);
        } else {
            vdd_src = new ConstantVddSource("vdd_src", vdd_nominal);
        }
        
        // Create DUT
        dut = new TxDriverTdf("dut", params);
        
        // Create monitor
        monitor = new SignalMonitor("monitor");
        
        // Connect signals
        src->out_p(sig_in_p);
        src->out_n(sig_in_n);
        
        if (use_vdd_ripple) {
            static_cast<VddWithRipple*>(vdd_src)->out(sig_vdd);
        } else {
            static_cast<ConstantVddSource*>(vdd_src)->out(sig_vdd);
        }
        
        dut->in_p(sig_in_p);
        dut->in_n(sig_in_n);
        dut->vdd(sig_vdd);
        dut->out_p(sig_out_p);
        dut->out_n(sig_out_n);
        
        monitor->in_p(sig_out_p);
        monitor->in_n(sig_out_n);
    }
    
    ~TxDriverTestbench() {
        delete src;
        delete vdd_src;
        delete dut;
        delete monitor;
    }
};

// ============================================================================
// Test Cases
// ============================================================================

/**
 * @brief Test DC gain accuracy
 */
TEST(TxDriverTest, DCGainTest) {
    TxDriverParams params;
    params.dc_gain = 0.5;
    params.vswing = 1.0;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";  // Disable saturation for linear test
    params.poles.clear();      // Disable bandwidth filtering
    
    double input_diff = 0.4;  // 400mV differential input
    
    TxDriverTestbench tb(params, DifferentialSource::DC, input_diff);
    
    // Run simulation
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // Expected: output_diff = input_diff * dc_gain * voltage_division
    // voltage_division = Z0 / (Zout + Z0) = 50 / (50 + 50) = 0.5
    double expected_diff = input_diff * params.dc_gain * 0.5;
    double actual_diff = tb.monitor->get_dc_diff();
    
    EXPECT_NEAR(actual_diff, expected_diff, 0.01);
    
    sc_core::sc_stop();
}

/**
 * @brief Test common-mode voltage control
 */
TEST(TxDriverTest, CommonModeTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";
    params.poles.clear();
    
    TxDriverTestbench tb(params, DifferentialSource::DC, 0.2);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // Common mode should be vcm_out * voltage_division
    double expected_cm = params.vcm_out * 0.5;  // 0.6 * 0.5 = 0.3
    double actual_cm = tb.monitor->get_dc_cm();
    
    EXPECT_NEAR(actual_cm, expected_cm, 0.02);
    
    sc_core::sc_stop();
}

/**
 * @brief Test soft saturation (tanh)
 */
TEST(TxDriverTest, SoftSaturationTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;  // Vsat = 0.4V
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "soft";
    params.vlin = 0.4;  // Linear range
    params.poles.clear();
    
    // Large input that should saturate
    double input_diff = 2.0;  // Much larger than vswing
    
    TxDriverTestbench tb(params, DifferentialSource::DC, input_diff);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // With soft saturation, output should approach but not exceed Vsat
    // tanh(2.0/0.4) = tanh(5) ≈ 0.9999
    // So output_diff ≈ 0.4 * 0.9999 * 0.5 (voltage division) ≈ 0.2
    double actual_diff = std::abs(tb.monitor->get_dc_diff());
    double max_output = (params.vswing / 2.0) * 0.5;  // 0.2V after division
    
    // Should be close to max but slightly less (tanh never reaches 1)
    EXPECT_LT(actual_diff, max_output);
    EXPECT_GT(actual_diff, max_output * 0.95);
    
    sc_core::sc_stop();
}

/**
 * @brief Test hard saturation (clipping)
 */
TEST(TxDriverTest, HardSaturationTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;  // Vsat = 0.4V
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "hard";
    params.poles.clear();
    
    // Large input that should clip
    double input_diff = 2.0;
    
    TxDriverTestbench tb(params, DifferentialSource::DC, input_diff);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // With hard saturation, output should be exactly at Vsat * voltage_division
    double expected_diff = (params.vswing / 2.0) * 0.5;  // 0.2V
    double actual_diff = std::abs(tb.monitor->get_dc_diff());
    
    EXPECT_NEAR(actual_diff, expected_diff, 0.01);
    
    sc_core::sc_stop();
}

/**
 * @brief Test bandwidth limiting (pole filtering)
 */
TEST(TxDriverTest, BandwidthTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 1.0;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";
    params.poles = {10e9};  // 10 GHz pole
    
    // Test with sine wave at pole frequency (should be -3dB)
    double input_amp = 0.2;
    double test_freq = 10e9;
    
    TxDriverTestbench tb(params, DifferentialSource::SINE, input_amp, test_freq);
    
    // Run for several cycles
    sc_core::sc_start(10, sc_core::SC_NS);
    
    // At pole frequency, gain should be reduced by ~3dB (factor of ~0.707)
    // Expected RMS = input_amp * dc_gain * voltage_div * 0.707 / sqrt(2)
    double expected_rms = input_amp * params.dc_gain * 0.5 * 0.707 / std::sqrt(2.0);
    double actual_rms = tb.monitor->get_rms_diff();
    
    // Allow 20% tolerance for numerical accuracy
    EXPECT_NEAR(actual_rms, expected_rms, expected_rms * 0.2);
    
    sc_core::sc_stop();
}

/**
 * @brief Test PSRR (power supply rejection)
 */
TEST(TxDriverTest, PSRRTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";
    params.poles.clear();
    
    // Enable PSRR
    params.psrr.enable = true;
    params.psrr.gain = 0.01;  // -40dB PSRR
    params.psrr.poles = {1e9};
    params.psrr.vdd_nom = 1.0;
    
    // DC input, VDD with ripple
    double input_diff = 0.0;  // No signal input
    double vdd_ripple = 0.1;  // 100mV ripple
    double ripple_freq = 100e6;
    
    TxDriverTestbench tb(params, DifferentialSource::DC, input_diff, 1e9,
                         1.0, true, vdd_ripple, ripple_freq);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // PSRR coupling should add ripple to differential output
    // Expected: ripple * psrr_gain * voltage_div = 0.1 * 0.01 * 0.5 = 0.0005V RMS
    double actual_rms = tb.monitor->get_rms_diff();
    double expected_rms = vdd_ripple * params.psrr.gain * 0.5 / std::sqrt(2.0);
    
    // Allow larger tolerance due to filter dynamics
    EXPECT_NEAR(actual_rms, expected_rms, expected_rms * 0.5);
    
    sc_core::sc_stop();
}

/**
 * @brief Test gain mismatch
 */
TEST(TxDriverTest, GainMismatchTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";
    params.poles.clear();
    params.imbalance.gain_mismatch = 10.0;  // 10% mismatch
    
    double input_diff = 0.4;
    
    TxDriverTestbench tb(params, DifferentialSource::DC, input_diff);
    
    sc_core::sc_start(100, sc_core::SC_NS);
    
    // With 10% mismatch:
    // gain_p = 1.0 + 10/200 = 1.05
    // gain_n = 1.0 - 10/200 = 0.95
    // The difference in single-ended gains should be visible
    
    // Get steady-state values (skip first 10%)
    size_t start = tb.monitor->samples_p.size() / 10;
    double avg_p = 0.0, avg_n = 0.0;
    for (size_t i = start; i < tb.monitor->samples_p.size(); ++i) {
        avg_p += tb.monitor->samples_p[i];
        avg_n += tb.monitor->samples_n[i];
    }
    avg_p /= (tb.monitor->samples_p.size() - start);
    avg_n /= (tb.monitor->samples_n.size() - start);
    
    // Check that out_p is larger than out_n (for positive input)
    // The mismatch should cause asymmetric swing
    double vcm = (avg_p + avg_n) / 2.0;
    double swing_p = avg_p - vcm;
    double swing_n = vcm - avg_n;
    
    // swing_p should be ~5% larger than swing_n
    double mismatch_ratio = swing_p / swing_n;
    EXPECT_NEAR(mismatch_ratio, 1.05 / 0.95, 0.05);
    
    sc_core::sc_stop();
}

/**
 * @brief Test slew rate limiting
 */
TEST(TxDriverTest, SlewRateLimitTest) {
    TxDriverParams params;
    params.dc_gain = 1.0;
    params.vswing = 0.8;
    params.vcm_out = 0.6;
    params.output_impedance = 50.0;
    params.sat_mode = "none";
    params.poles.clear();
    params.slew_rate.enable = true;
    params.slew_rate.max_slew_rate = 1e11;  // 100 V/ns = 0.1 V/ps
    
    // Step input
    double step_amp = 0.4;
    
    TxDriverTestbench tb(params, DifferentialSource::STEP, step_amp, 1e9, 1.0);
    
    sc_core::sc_start(50, sc_core::SC_NS);
    
    // Find the rise time (10% to 90%)
    double final_value = step_amp * params.dc_gain * 0.5;  // After voltage division
    double v_10 = 0.1 * final_value;
    double v_90 = 0.9 * final_value;
    
    double t_10 = 0.0, t_90 = 0.0;
    bool found_10 = false, found_90 = false;
    
    for (size_t i = 0; i < tb.monitor->samples_diff.size(); ++i) {
        double v = tb.monitor->samples_diff[i];
        if (!found_10 && v >= v_10) {
            t_10 = tb.monitor->time_stamps[i];
            found_10 = true;
        }
        if (!found_90 && v >= v_90) {
            t_90 = tb.monitor->time_stamps[i];
            found_90 = true;
            break;
        }
    }
    
    if (found_10 && found_90) {
        double rise_time = t_90 - t_10;
        // Expected rise time: 0.8 * final_value / slew_rate
        double expected_rise = 0.8 * final_value / params.slew_rate.max_slew_rate;
        
        // Allow 50% tolerance due to discrete time steps
        EXPECT_NEAR(rise_time, expected_rise, expected_rise * 0.5);
    }
    
    sc_core::sc_stop();
}
