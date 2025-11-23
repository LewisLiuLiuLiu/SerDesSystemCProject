#include "ams/tx_driver.h"
#include <cmath>

namespace serdes {

TxDriverTdf::TxDriverTdf(sc_core::sc_module_name nm, const TxDriverParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_filter_state(0.0)
{
}

void TxDriverTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void TxDriverTdf::processing() {
    // 读取输入
    double x_in = in.read();
    
    // 1. 摆幅缩放
    double v_scaled = (m_params.swing / 2.0) * x_in;
    
    // 2. 带宽限制（简化的一阶低通滤波）
    // 使用简单的IIR滤波器近似
    double alpha = 0.5;  // 简化系数，实际应根据带宽计算
    m_filter_state = alpha * v_scaled + (1.0 - alpha) * m_filter_state;
    double v_filtered = m_filter_state;
    
    // 3. 非线性饱和（硬限幅）
    double v_out = v_filtered;
    if (v_out > m_params.sat) {
        v_out = m_params.sat;
    } else if (v_out < -m_params.sat) {
        v_out = -m_params.sat;
    }
    
    // 输出
    out.write(v_out);
}

} // namespace serdes
