#ifndef SERDES_RX_DFE_H
#define SERDES_RX_DFE_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxDfeTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxDfeTdf(sc_core::sc_module_name nm, const RxDfeParams& params);
    void set_attributes();
    void processing();
private:
    RxDfeParams m_params;
    std::vector<double> m_taps;
};
}
#endif
