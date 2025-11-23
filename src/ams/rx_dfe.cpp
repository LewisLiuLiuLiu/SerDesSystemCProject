#include "ams/rx_dfe.h"

namespace serdes {

RxDfeTdf::RxDfeTdf(sc_core::sc_module_name nm, const RxDfeParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
{
    // 初始化反馈历史缓冲区
    m_history.resize(params.taps.size(), 0.0);
}

void RxDfeTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void RxDfeTdf::processing() {
    // 读取输入
    double x_in = in.read();
    
    // 计算DFE反馈补偿
    double feedback_sum = 0.0;
    for (size_t i = 0; i < m_params.taps.size(); ++i) {
        feedback_sum += m_params.taps[i] * m_history[i];
    }
    
    // 应用DFE补偿
    double y = x_in - feedback_sum;
    
    // 更新历史（移位）
    for (size_t i = m_history.size() - 1; i > 0; --i) {
        m_history[i] = m_history[i-1];
    }
    if (!m_history.empty()) {
        m_history[0] = y;  // 存储当前输出作为下次反馈
    }
    
    // 输出
    out.write(y);
}

} // namespace serdes
