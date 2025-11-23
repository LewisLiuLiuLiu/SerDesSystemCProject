#ifndef SERDES_CLOCK_GENERATION_H
#define SERDES_CLOCK_GENERATION_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class ClockGenerationTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_out<double> clk_phase;
    ClockGenerationTdf(sc_core::sc_module_name nm, const ClockParams& params);
    void set_attributes();
    void processing();
private:
    ClockParams m_params;
    double m_phase;
    double m_frequency;
};
}
#endif
