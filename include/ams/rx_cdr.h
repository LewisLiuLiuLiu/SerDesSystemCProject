#ifndef SERDES_RX_CDR_H
#define SERDES_RX_CDR_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxCdrTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> phase_out;
    RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params);
    void set_attributes();
    void processing();
private:
    CdrParams m_params;
    double m_phase;
};
}
#endif
