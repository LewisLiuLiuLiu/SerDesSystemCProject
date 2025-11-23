#include "ams/clock_generation.h"
#include <cmath>

namespace serdes {

ClockGenerationTdf::ClockGenerationTdf(sc_core::sc_module_name nm, const ClockParams& params)
    : sca_tdf::sca_module(nm)
    , clk_out("clk_out")
    , m_params(params)
    , m_time(0.0)
    , m_sample_rate(80e9)
{
}

void ClockGenerationTdf::set_attributes() {
    clk_out.set_rate(1);
    clk_out.set_timestep(1.0 / m_sample_rate, sc_core::SC_SEC);
}

void ClockGenerationTdf::processing() {
    // 理想时钟生成：方波
    double period = 1.0 / m_params.frequency;
    double phase = std::fmod(m_time, period) / period;
    
    // 生成方波（0或1）
    double clk = (phase < 0.5) ? 0.0 : 1.0;
    
    clk_out.write(clk);
    
    m_time += 1.0 / m_sample_rate;
}

} // namespace serdes
