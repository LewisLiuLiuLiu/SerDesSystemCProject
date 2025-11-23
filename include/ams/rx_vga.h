#ifndef SERDES_RX_VGA_H
#define SERDES_RX_VGA_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class RxVgaTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    RxVgaTdf(sc_core::sc_module_name nm, const RxVgaParams& params);
    void set_attributes();
    void processing();
private:
    RxVgaParams m_params;
};
}
#endif
