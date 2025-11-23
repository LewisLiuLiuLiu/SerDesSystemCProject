#include "ams/rx_vga.h"

namespace serdes {

RxVgaTdf::RxVgaTdf(sc_core::sc_module_name nm, const RxVgaParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
{
}

void RxVgaTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void RxVgaTdf::processing() {
    // 读取输入
    double x_in = in.read();
    
    // 应用增益
    double y = m_params.gain * x_in;
    
    // 输出
    out.write(y);
}

} // namespace serdes
