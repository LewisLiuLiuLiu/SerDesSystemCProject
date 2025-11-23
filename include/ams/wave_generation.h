#ifndef SERDES_WAVE_GENERATION_H
#define SERDES_WAVE_GENERATION_H

#include <systemc-ams>
#include "common/parameters.h"

namespace serdes {

class WaveGenerationTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> out;
    
    WaveGenerationTdf(sc_core::sc_module_name nm, const WaveGenParams& params);
    
    void set_attributes();
    void processing();
    
private:
    WaveGenParams m_params;
    unsigned int m_lfsr_state;
    double m_sample_rate;
    double m_time;
};

} // namespace serdes

#endif // SERDES_WAVE_GENERATION_H
