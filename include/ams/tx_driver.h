#ifndef SERDES_TX_DRIVER_H
#define SERDES_TX_DRIVER_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class TxDriverTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    TxDriverTdf(sc_core::sc_module_name nm, const TxDriverParams& params);
    void set_attributes();
    void processing();
private:
    TxDriverParams m_params;
};
}
#endif
