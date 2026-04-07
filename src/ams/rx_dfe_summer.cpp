/**
 * @file rx_dfe_summer.cpp
 * @brief RX DFE Summer 模块实现
 */

#include "ams/rx_dfe_summer.h"
#include <iostream>
#include <algorithm>

namespace serdes {

RxDfeSummerTdf::RxDfeSummerTdf(sc_core::sc_module_name nm, 
                               const RxDfeSummerParams& params)
    : sca_tdf::sca_module(nm)
    , in_p("in_p")
    , in_n("in_n")
    , data_in("data_in")
    , out_p("out_p")
    , out_n("out_n")
    , tap1_de("tap1_de")
    , tap2_de("tap2_de")
    , tap3_de("tap3_de")
    , tap4_de("tap4_de")
    , tap5_de("tap5_de")
    , m_params(params)
    , m_last_feedback(0.0)
    , m_de_ports_connected(false)
    , m_last_bit(-1.0)
    , m_de_taps_initialized(false)
{
    // 初始化抽头系数
    m_tap_coeffs = m_params.tap_coeffs;
    
    // 初始化历史缓冲区（全零）
    size_t tap_count = m_tap_coeffs.size();
    m_history_bits.resize(tap_count, 0.0);
}

void RxDfeSummerTdf::set_attributes()
{
    // 注意：不设置固定时间步，让 SystemC-AMS 自动同步
    // DFE Summer 将继承其他模块的时间步长
    
    // 设置端口速率
    in_p.set_rate(1);
    in_n.set_rate(1);
    data_in.set_rate(1);
    data_in.set_delay(1);  // 添加 1 个样本延迟，打破反馈环路
    out_p.set_rate(1);
    out_n.set_rate(1);
}

void RxDfeSummerTdf::processing()
{
    // ========================================================================
    // 步骤 1: 读取差分输入
    // ========================================================================
    double v_in_p = in_p.read();
    double v_in_n = in_n.read();
    double v_main = v_in_p - v_in_n;  // 差分值
    
    // ========================================================================
    // 步骤 2: 使能检查
    // ========================================================================
    if (!m_params.enable) {
        // 直通模式：输出 = 输入（经共模合成）
        out_p.write(m_params.vcm_out + 0.5 * v_main);
        out_n.write(m_params.vcm_out - 0.5 * v_main);
        return;
    }
    
    // ========================================================================
    // 步骤 3: 读取历史判决并更新缓冲区
    // ========================================================================
    double new_bit = data_in.read();
    update_history(new_bit);
    
    // ========================================================================
    // 步骤 4: 尝试从 DE 端口读取抽头更新
    // ========================================================================
    read_de_tap_updates();
    
    // ========================================================================
    // 步骤 5: 计算反馈电压
    // ========================================================================
    double v_fb = compute_feedback();
    m_last_feedback = v_fb;
    
    // ========================================================================
    // 步骤 6: 差分求和
    // ========================================================================
    double v_eq = v_main - v_fb;
    
    // ========================================================================
    // 步骤 7: 可选软饱和限幅
    // ========================================================================
    if (m_params.sat_enable) {
        v_eq = soft_saturate(v_eq);
    }
    
    // ========================================================================
    // 步骤 8: 共模合成输出
    // ========================================================================
    out_p.write(m_params.vcm_out + 0.5 * v_eq);
    out_n.write(m_params.vcm_out - 0.5 * v_eq);
}

double RxDfeSummerTdf::bit_map(double bit_val) const
{
    if (m_params.map_mode == "pm1") {
        // ±1 映射：0 → -1, 1 → +1
        return (bit_val > 0.5) ? 1.0 : -1.0;
    } else {
        // 0/1 映射：0 → 0, 1 → 1
        return (bit_val > 0.5) ? 1.0 : 0.0;
    }
}

double RxDfeSummerTdf::compute_feedback() const
{
    double v_fb = 0.0;
    size_t tap_count = std::min(m_tap_coeffs.size(), m_history_bits.size());
    
    for (size_t k = 0; k < tap_count; ++k) {
        double bit_mapped = bit_map(m_history_bits[k]);
        v_fb += m_tap_coeffs[k] * bit_mapped * m_params.vtap;
    }
    
    return v_fb;
}

double RxDfeSummerTdf::soft_saturate(double v_in) const
{
    // 软饱和：使用 tanh 函数
    double v_sat = 0.5 * (m_params.sat_max - m_params.sat_min);
    double v_center = 0.5 * (m_params.sat_max + m_params.sat_min);
    
    return v_center + v_sat * std::tanh(v_in / v_sat);
}

void RxDfeSummerTdf::update_history(double new_bit)
{
    // FIFO 队列：移位并插入新比特
    // history[0] = b[n-1] (最近), history[N-1] = b[n-N] (最远)
    
    if (m_history_bits.empty()) return;
    
    // 右移所有元素
    for (size_t i = m_history_bits.size() - 1; i > 0; --i) {
        m_history_bits[i] = m_history_bits[i - 1];
    }
    
    // 插入新比特到位置 0
    m_history_bits[0] = new_bit;
}

void RxDfeSummerTdf::read_de_tap_updates()
{
    // 尝试从 DE 端口读取抽头更新
    // 注意：只有当端口被连接且值有效时才读取
    // SystemC-AMS 的 sca_de 端口在未连接时会返回默认值
    
    // 读取各抽头更新（只有当值非零且有限时才更新）
    // 使用m_de_taps_initialized标志来区分"初始零值"和"DE端口返回的零值"
    if (m_tap_coeffs.size() >= 1) {
        double tap1_val = tap1_de.read();
        if (std::isfinite(tap1_val)) {
            if (!m_de_taps_initialized && tap1_val != 0.0) {
                m_de_taps_initialized = true;
            }
            if (m_de_taps_initialized) {
                m_tap_coeffs[0] = tap1_val;
            }
        }
    }
    if (m_tap_coeffs.size() >= 2) {
        double tap2_val = tap2_de.read();
        if (std::isfinite(tap2_val)) {
            if (!m_de_taps_initialized && tap2_val != 0.0) {
                m_de_taps_initialized = true;
            }
            if (m_de_taps_initialized) {
                m_tap_coeffs[1] = tap2_val;
            }
        }
    }
    if (m_tap_coeffs.size() >= 3) {
        double tap3_val = tap3_de.read();
        if (std::isfinite(tap3_val)) {
            if (!m_de_taps_initialized && tap3_val != 0.0) {
                m_de_taps_initialized = true;
            }
            if (m_de_taps_initialized) {
                m_tap_coeffs[2] = tap3_val;
            }
        }
    }
    if (m_tap_coeffs.size() >= 4) {
        double tap4_val = tap4_de.read();
        if (std::isfinite(tap4_val)) {
            if (!m_de_taps_initialized && tap4_val != 0.0) {
                m_de_taps_initialized = true;
            }
            if (m_de_taps_initialized) {
                m_tap_coeffs[3] = tap4_val;
            }
        }
    }
    if (m_tap_coeffs.size() >= 5) {
        double tap5_val = tap5_de.read();
        if (std::isfinite(tap5_val)) {
            if (!m_de_taps_initialized && tap5_val != 0.0) {
                m_de_taps_initialized = true;
            }
            if (m_de_taps_initialized) {
                m_tap_coeffs[4] = tap5_val;
            }
        }
    }
}

} // namespace serdes
