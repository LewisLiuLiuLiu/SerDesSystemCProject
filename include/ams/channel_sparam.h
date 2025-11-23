#ifndef SERDES_CHANNEL_SPARAM_H
#define SERDES_CHANNEL_SPARAM_H
#include <systemc-ams>
#include "common/parameters.h"
namespace serdes {
class ChannelSParamTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
    void set_attributes();
    void processing();
private:
    ChannelParams m_params;
    std::vector<double> m_buffer;
};
}
#endif
