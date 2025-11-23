#include "ams/rx_ctle.h"
#include <cmath>

namespace serdes {

RxCtleTdf::RxCtleTdf(sc_core::sc_module_name nm, const RxCtleParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_filter_state(0.0)
{
}

void RxCtleTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void RxCtleTdf::processing() {
    // 读取输入
    double x_in = in.read();
    
    // 简化CTLE：高通特性（提升高频）
    // 使用简单的差分 + 增益
    double x_diff = x_in - m_filter_state;
    m_filter_state = x_in;
    
    // 应用DC增益
    double y = m_params.dc_gain * x_in + 0.5 * x_diff;
    
    // 输出
    out.write(y);
}

} // namespace serdes
