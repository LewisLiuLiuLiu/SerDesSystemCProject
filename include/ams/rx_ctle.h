#ifndef SERDES_RX_CTLE_H
#define SERDES_RX_CTLE_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxCtleTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxCtleTdf(sc_core::sc_module_name nm, const RxCtleParams& params);
    void set_attributes();
    void processing();
private:
    RxCtleParams m_params;
};
}
#endif
