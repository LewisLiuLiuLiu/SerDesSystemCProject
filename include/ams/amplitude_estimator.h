#ifndef SERDES_AMPLITUDE_ESTIMATOR_H
#define SERDES_AMPLITUDE_ESTIMATOR_H

#include <systemc-ams>

namespace serdes {

/**
 * @brief Amplitude Estimator TDF Module (AGC detector)
 *
 * Estimates the signal amplitude at the monitored stage (e.g. VGA output)
 * as an exponential moving average (EMA) of the instantaneous differential
 * magnitude |v_p - v_n|, and exports it to the DE domain for the AGC
 * controller in AdaptionDe.
 *
 * Target semantics (confirmed 2026-10):
 *   AGC target amplitude = EMA(|v_p - v_n|) at the VGA output.
 *
 * Outputs:
 * - amplitude / amplitude_de : EMA(|v_p - v_n|) in Volts
 * - saturated_de             : near-saturation flag; true when the EMA has
 *   climbed to within `sat_margin` of the monitored stage's differential
 *   clamp level (`sat_level`). Gain increases can no longer raise the
 *   amplitude in this state, so the AGC must stop increasing gain
 *   (wind-up interlock, v1).
 *
 * The DE outputs are refreshed every `de_update_interval` seconds instead
 * of every timestep, keeping the DE event rate low (same pattern as
 * DfeAdaptTdf::write_tap_outputs).
 */
class AmplitudeEstimatorTdf : public sca_tdf::sca_module {
public:
    // TDF inputs (tap the monitored differential signal)
    sca_tdf::sca_in<double> in_p;
    sca_tdf::sca_in<double> in_n;

    // TDF output (monitoring / eye tools)
    sca_tdf::sca_out<double> amplitude;

    // DE outputs (to AdaptionDe AGC)
    sca_tdf::sca_de::sca_out<double> amplitude_de;   // EMA(|v_p - v_n|), V
    sca_tdf::sca_de::sca_out<bool> saturated_de;     // near-saturation flag

    /**
     * @brief Constructor
     * @param nm Module name
     * @param ema_tau EMA time constant (s), must be > 0
     * @param sat_level Differential clamp level of the monitored stage (V);
     *                  <= 0 disables the saturation flag
     * @param de_update_interval DE output refresh interval (s), must be > 0
     * @param sat_margin Saturation flag margin (0-1), flag when EMA > margin * sat_level
     */
    AmplitudeEstimatorTdf(sc_core::sc_module_name nm,
                          double ema_tau,
                          double sat_level,
                          double de_update_interval,
                          double sat_margin = 0.95);

    void set_attributes() override;
    void initialize() override;
    void processing() override;

    /// Current EMA value (for debug)
    double get_amplitude() const { return m_ema; }

private:
    double m_tau;             // EMA time constant (s)
    double m_sat_level;       // clamp level (V), <=0 = flag disabled
    double m_de_interval;     // DE refresh interval (s)
    double m_sat_margin;      // flag margin

    double m_alpha;           // EMA coefficient per timestep (lazy init)
    double m_ema;             // current EMA
    bool m_initialized;       // first sample primes the EMA
    bool m_attr_ready;        // lazy timestep-derived constants ready

    long m_steps;             // timestep counter
    long m_de_steps;          // DE refresh every N steps
};

} // namespace serdes

#endif // SERDES_AMPLITUDE_ESTIMATOR_H
