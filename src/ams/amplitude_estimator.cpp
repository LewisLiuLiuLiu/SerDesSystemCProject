#include "ams/amplitude_estimator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace serdes {

AmplitudeEstimatorTdf::AmplitudeEstimatorTdf(
    sc_core::sc_module_name nm,
    double ema_tau,
    double sat_level,
    double de_update_interval,
    double sat_margin)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , amplitude("amplitude")
    , amplitude_de("amplitude_de")
    , saturated_de("saturated_de")
    , m_tau(ema_tau)
    , m_sat_level(sat_level)
    , m_de_interval(de_update_interval)
    , m_sat_margin(sat_margin)
    , m_alpha(1.0)
    , m_ema(0.0)
    , m_initialized(false)
    , m_attr_ready(false)
    , m_steps(0)
    , m_de_steps(1)
{
    if (!(ema_tau > 0.0)) {
        throw std::invalid_argument(
            "AmplitudeEstimator: ema_tau must be > 0");
    }
    if (!(de_update_interval > 0.0)) {
        throw std::invalid_argument(
            "AmplitudeEstimator: de_update_interval must be > 0");
    }
}

void AmplitudeEstimatorTdf::set_attributes() {
    in_p.set_rate(1);
    in_n.set_rate(1);
    amplitude.set_rate(1);
    // The module inherits the timestep of the TDF cluster it taps.
}

void AmplitudeEstimatorTdf::initialize() {
    // alpha and the DE decimation need the resolved timestep, which is
    // only guaranteed once the cluster has been elaborated; derive them
    // lazily on the first processing() call.
}

void AmplitudeEstimatorTdf::processing() {
    // ---- Lazy derivation of timestep-dependent constants ----
    if (!m_attr_ready) {
        double ts = get_timestep().to_seconds();
        if (ts > 0.0) {
            m_alpha = std::min(1.0, ts / m_tau);
            double steps = m_de_interval / ts;
            m_de_steps = (steps < 1.0) ? 1 : static_cast<long>(steps + 0.5);
        }
        m_attr_ready = true;
    }

    // ---- EMA of the instantaneous differential magnitude ----
    double d = std::fabs(in_p.read() - in_n.read());
    if (!m_initialized) {
        m_ema = d;  // prime with the first sample
        m_initialized = true;
    } else {
        m_ema += m_alpha * (d - m_ema);
    }

    amplitude.write(m_ema);

    // ---- Decimated DE output refresh ----
    if ((m_steps % m_de_steps) == 0) {
        amplitude_de.write(m_ema);

        // Near-saturation detection on the smoothed amplitude
        bool sat = (m_sat_level > 0.0) && (m_ema > m_sat_margin * m_sat_level);
        saturated_de.write(sat);
    }
    ++m_steps;
}

} // namespace serdes
