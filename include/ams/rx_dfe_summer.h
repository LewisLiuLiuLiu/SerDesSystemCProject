/**
 * @file rx_dfe_summer.h
 * @brief RX DFE Summer 模块 - 判决反馈均衡求和器
 * 
 * 差分输入/输出架构，从主路径信号减去基于历史判决的反馈信号，
 * 抵消后游 ISI，增大眼图开度。
 * 
 * 文档参考：docs/modules/dfesummer.md
 */

#ifndef SERDES_AMS_RX_DFE_SUMMER_H
#define SERDES_AMS_RX_DFE_SUMMER_H

#include <systemc-ams>
#include "common/parameters.h"
#include <vector>
#include <cmath>

namespace serdes {

/**
 * @class RxDfeSummerTdf
 * @brief 差分 DFE Summer TDF 模块
 * 
 * 功能：
 * - 接收差分输入信号 (in_p, in_n)
 * - 基于历史判决计算反馈电压
 * - 输出均衡后的差分信号 (out_p, out_n)
 * - 支持 DE 域抽头系数动态更新
 */
class RxDfeSummerTdf : public sca_tdf::sca_module {
public:
    // ========================================================================
    // TDF 域差分输入端口
    // ========================================================================
    sca_tdf::sca_in<double> in_p;       ///< 差分输入正端 (来自 VGA)
    sca_tdf::sca_in<double> in_n;       ///< 差分输入负端 (来自 VGA)
    
    // ========================================================================
    // TDF 域历史判决输入端口
    // ========================================================================
    sca_tdf::sca_in<double> data_in;    ///< 历史判决输入 (来自 Sampler，0.0/1.0)
    
    // ========================================================================
    // TDF 域差分输出端口
    // ========================================================================
    sca_tdf::sca_out<double> out_p;     ///< 差分输出正端 (送往 Sampler)
    sca_tdf::sca_out<double> out_n;     ///< 差分输出负端 (送往 Sampler)
    
    // ========================================================================
    // DE→TDF 桥接端口 (抽头系数动态更新，可选)
    // ========================================================================
    sca_tdf::sca_de::sca_in<double> tap1_de;  ///< DE 域抽头1更新
    sca_tdf::sca_de::sca_in<double> tap2_de;  ///< DE 域抽头2更新
    sca_tdf::sca_de::sca_in<double> tap3_de;  ///< DE 域抽头3更新
    sca_tdf::sca_de::sca_in<double> tap4_de;  ///< DE 域抽头4更新
    sca_tdf::sca_de::sca_in<double> tap5_de;  ///< DE 域抽头5更新
    
    // ========================================================================
    // 构造函数
    // ========================================================================
    /**
     * @brief 构造函数
     * @param nm 模块名称
     * @param params DFE Summer 参数
     */
    RxDfeSummerTdf(sc_core::sc_module_name nm, const RxDfeSummerParams& params);
    
    // ========================================================================
    // TDF 回调函数
    // ========================================================================
    void set_attributes() override;
    void processing() override;
    
    // ========================================================================
    // 公共接口
    // ========================================================================
    /**
     * @brief 获取当前抽头系数
     */
    const std::vector<double>& get_tap_coeffs() const { return m_tap_coeffs; }
    
    /**
     * @brief 获取最近一次反馈电压
     */
    double get_last_feedback() const { return m_last_feedback; }
    
private:
    // ========================================================================
    // 参数
    // ========================================================================
    RxDfeSummerParams m_params;
    
    // ========================================================================
    // 内部状态
    // ========================================================================
    std::vector<double> m_tap_coeffs;     ///< 当前抽头系数
    std::vector<double> m_history_bits;   ///< 历史判决缓冲区
    double m_last_feedback;               ///< 上一次反馈电压（调试用）
    bool m_de_ports_connected;            ///< DE端口是否连接标志
    double m_last_bit;                    ///< 上一个UI的判决值（用于检测新采样）
    bool m_de_taps_initialized;           ///< 是否已从DE端口接收到有效抽头值
    
    // ========================================================================
    // 内部方法
    // ========================================================================
    /**
     * @brief 比特映射函数
     * @param bit_val 输入比特值 (0.0 或 1.0)
     * @return 映射后的值 (pm1模式: -1/+1, 01模式: 0/1)
     */
    double bit_map(double bit_val) const;
    
    /**
     * @brief 计算反馈电压
     * @return 反馈电压 v_fb
     */
    double compute_feedback() const;
    
    /**
     * @brief 软饱和限幅
     * @param v_in 输入电压
     * @return 限幅后的电压
     */
    double soft_saturate(double v_in) const;
    
    /**
     * @brief 更新历史缓冲区
     * @param new_bit 新的判决比特
     */
    void update_history(double new_bit);
    
    /**
     * @brief 从 DE 端口读取抽头更新（如果连接）
     */
    void read_de_tap_updates();
};

} // namespace serdes

#endif // SERDES_AMS_RX_DFE_SUMMER_H
