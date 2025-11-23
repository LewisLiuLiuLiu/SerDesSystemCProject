#ifndef SERDES_RX_SAMPLER_H
#define SERDES_RX_SAMPLER_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxSamplerTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_in<double> clk;
    sca_tdf::sca_out<double> out;
    RxSamplerTdf(sc_core::sc_module_name nm, const RxSamplerParams& params);
    void set_attributes();
    void processing();
private:
    RxSamplerParams m_params;
};
}
#endif
