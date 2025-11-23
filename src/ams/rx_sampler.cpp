#include "ams/rx_sampler.h"

namespace serdes {

RxSamplerTdf::RxSamplerTdf(sc_core::sc_module_name nm, const RxSamplerParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_prev_bit(false)
{
}

void RxSamplerTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}

void RxSamplerTdf::processing() {
    // 读取输入电压
    double v_in = in.read();
    
    // 判决：与阈值比较（带迟滞）
    bool bit_out;
    if (v_in > m_params.threshold + m_params.hysteresis/2.0) {
        bit_out = true;
    } else if (v_in < m_params.threshold - m_params.hysteresis/2.0) {
        bit_out = false;
    } else {
        bit_out = m_prev_bit;  // 保持上一状态（迟滞）
    }
    
    m_prev_bit = bit_out;
    
    // 输出（转换为double: false->-1.0, true->+1.0）
    double y = bit_out ? 1.0 : -1.0;
    out.write(y);
}

} // namespace serdes
