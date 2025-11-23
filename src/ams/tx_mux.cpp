#include "ams/tx_mux.h"

namespace serdes {

TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int lane_sel)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_lane_sel(lane_sel)
{
}

void TxMuxTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void TxMuxTdf::processing() {
    // 简单透传模式（单通道）
    double x_in = in.read();
    out.write(x_in);
}

} // namespace serdes
