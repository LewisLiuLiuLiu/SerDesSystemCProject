/**
 * @file test_amplitude_estimator.cpp
 * @brief Unit test for AmplitudeEstimatorTdf (AGC detector)
 *
 * Verifies:
 * - EMA(|v_p - v_n|) converges to the differential amplitude
 * - saturated_de asserts when the EMA approaches the configured clamp level
 *
 * Uses two sc_start phases in a single elaboration (same instances).
 */

#include <gtest/gtest.h>
#include <systemc-ams>
#include "ams/amplitude_estimator.h"

using namespace serdes;

namespace {

// 两级幅度源：0-100ns: 0.3V；100ns 之后: 0.6V（差分幅度）
class TwoLevelDiffSource : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out_p;
    sca_tdf::sca_out<double> out_n;

    TwoLevelDiffSource(sc_core::sc_module_name nm)
        : sca_tdf::sca_module(nm), out_p("out_p"), out_n("out_n") {}

    void set_attributes() override {
        out_p.set_rate(1);
        out_n.set_rate(1);
        out_p.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
        out_n.set_timestep(1.0 / 100e9, sc_core::SC_SEC);
    }
    void processing() override {
        double t = get_time().to_seconds();
        double amp = (t < 100e-9) ? 0.3 : 0.6;
        out_p.write(0.5 * amp);
        out_n.write(-0.5 * amp);
    }
};

} // namespace

TEST(AmplitudeEstimatorTest, EmaConvergenceAndSaturationFlag) {
    TwoLevelDiffSource src("src");

    // tau = 6.4ns；sat_level = 0.5V（VGA 默认限幅）；DE 刷新 2.5ns
    AmplitudeEstimatorTdf est("est", 6.4e-9, 0.5, 2.5e-9);

    sca_tdf::sca_signal<double> s_in_p, s_in_n, s_amp_tdf;
    sc_core::sc_signal<double> s_amp_de;
    sc_core::sc_signal<bool> s_sat;

    src.out_p(s_in_p);
    src.out_n(s_in_n);
    est.in_p(s_in_p);
    est.in_n(s_in_n);
    est.amplitude(s_amp_tdf);
    est.amplitude_de(s_amp_de);
    est.saturated_de(s_sat);

    // 相位1：幅度 0.3V
    sc_core::sc_start(100, sc_core::SC_NS);
    EXPECT_NEAR(s_amp_de.read(), 0.3, 0.01)
        << "EMA must converge to |v_p - v_n| = 0.3V";
    EXPECT_FALSE(s_sat.read())
        << "0.3V is below 0.95 * 0.5V clamp level";

    // 相位2：幅度 0.6V → EMA 收敛到 0.6 > 0.475 → 饱和标志置位
    sc_core::sc_start(100, sc_core::SC_NS);
    EXPECT_NEAR(s_amp_de.read(), 0.6, 0.02)
        << "EMA must converge to |v_p - v_n| = 0.6V";
    EXPECT_TRUE(s_sat.read())
        << "0.6V exceeds 0.95 * 0.5V clamp level";

    sc_core::sc_stop();
}
