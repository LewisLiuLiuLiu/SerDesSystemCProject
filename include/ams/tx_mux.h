#ifndef SERDES_TX_MUX_H
#define SERDES_TX_MUX_H
#include <systemc-ams>
namespace serdes {
class TxMuxTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    TxMuxTdf(sc_core::sc_module_name nm);
    void set_attributes();
    void processing();
};
}
#endif
