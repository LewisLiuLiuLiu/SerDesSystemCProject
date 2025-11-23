#include "ams/wave_generation.h"
#include <cmath>
#include <random>

namespace serdes {

WaveGenerationTdf::WaveGenerationTdf(sc_core::sc_module_name nm, const WaveGenParams& params)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_params(params)
    , m_lfsr_state(0x7FFFFFFF)
    , m_sample_rate(80e9)
    , m_time(0.0)
{
}

void WaveGenerationTdf::set_attributes() {
    out.set_rate(1);
    out.set_timestep(1.0 / m_sample_rate, sc_core::SC_SEC);
}

void WaveGenerationTdf::processing() {
    // Simple PRBS generation (PRBS31 polynomial: x^31 + x^28 + 1)
    unsigned int feedback = ((m_lfsr_state >> 30) ^ (m_lfsr_state >> 27)) & 0x1;
    m_lfsr_state = ((m_lfsr_state << 1) | feedback) & 0x7FFFFFFF;
    
    // Convert to NRZ signal (-1 or +1)
    double bit_value = (m_lfsr_state & 0x1) ? 1.0 : -1.0;
    
    // Output
    out.write(bit_value);
    
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
