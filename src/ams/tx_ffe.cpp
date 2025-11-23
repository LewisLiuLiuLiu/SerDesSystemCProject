#include "ams/tx_ffe.h"
#include <cmath>

namespace serdes {

TxFfeTdf::TxFfeTdf(sc_core::sc_module_name nm, const TxFfeParams& params)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_params(params)
    , m_buffer_ptr(0)
{
    // 初始化循环缓冲区
    size_t num_taps = params.taps.size();
    if (num_taps == 0) {
        num_taps = 1;  // 至少需要1个抽头
    }
    m_buffer.resize(num_taps, 0.0);
}

void TxFfeTdf::set_attributes() {
    // 设置输入输出采样率相同
    in.set_rate(1);
    out.set_rate(1);
}

void TxFfeTdf::processing() {
    // 读取输入样本
    double x_in = in.read();
    
    // 将当前输入存入循环缓冲区
    m_buffer[m_buffer_ptr] = x_in;
    
    // 计算FIR卷积输出
    double y = 0.0;
    size_t num_taps = m_params.taps.size();
    
    for (size_t i = 0; i < num_taps; ++i) {
        // 计算循环索引
        size_t idx = (m_buffer_ptr + num_taps - i) % num_taps;
        // 累加卷积结果
        y += m_params.taps[i] * m_buffer[idx];
    }
    
    // 更新缓冲区指针
    m_buffer_ptr = (m_buffer_ptr + 1) % num_taps;
    
    // 输出
    out.write(y);
}

} // namespace serdes
