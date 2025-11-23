#include "ams/rx_cdr.h"
#include <cmath>

namespace serdes {

RxCdrTdf::RxCdrTdf(sc_core::sc_module_name nm, const CdrParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_phase(0.0)
    , m_prev_bit(0.0)
{
}

void RxCdrTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void RxCdrTdf::processing() {
    // 读取输入
    double current_bit = in.read();
    
    // Bang-Bang相位检测（简化）
    double phase_error = 0.0;
    if (std::abs(current_bit - m_prev_bit) > 0.5) {  // 检测到边沿
        if (current_bit > m_prev_bit) {
            phase_error = 1.0;  // 时钟晚，需要提前
        } else {
            phase_error = -1.0;  // 时钟早，需要延后
        }
    }
    
    // PI控制器
    m_phase += m_params.pi.kp * phase_error + m_params.pi.ki * phase_error;
    
    // 限制相位范围
    if (m_phase > m_params.pai.range) {
        m_phase = m_params.pai.range;
    } else if (m_phase < -m_params.pai.range) {
        m_phase = -m_params.pai.range;
    }
    
    m_prev_bit = current_bit;
    
    // 输出恢复的时钟相位（简化，直接输出数据）
    out.write(current_bit);
}

} // namespace serdes
