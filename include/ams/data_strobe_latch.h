#ifndef SERDES_DATA_STROBE_LATCH_H
#define SERDES_DATA_STROBE_LATCH_H

#include <systemc-ams>

namespace serdes {

/**
 * @brief 数据频闪锁存器（TDF）
 *
 * 主采样器在 CDR 的双相触发（edge+data，2/UI）下采样并保持，其原始输出
 * 流是"边沿样本 + 数据样本"的混合流：每 UI 夹着一个采在跳变沿上的
 * 过渡值（伪比特）。本模块在 CDR 的 data_trigger 频闪时刻（每 UI 1 个）
 * 锁存混合流，对外输出干净的 1/UI 数据判决流（保持到下一频闪）。
 *
 * 时序：CDR data_trigger 端口延迟 1 步；本模块 in/data_strobe 各再延迟
 * 1 步（共 2 步），确保锁存到的是采样器已写入的数据判决（而非边沿样本）。
 */
class DataStrobeLatchTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in;             ///< 混合保持流（主采样器输出）
    sca_tdf::sca_in<bool> data_strobe;      ///< CDR data_trigger（1/UI）
    sca_tdf::sca_out<double> out;           ///< 干净判决流（0/1，1/UI 更新）

    DataStrobeLatchTdf(sc_core::sc_module_name nm);

    void set_attributes() override;
    void initialize() override;
    void processing() override;

    /// 最近锁存的判决值（调试）
    double get_held() const { return m_held; }

private:
    double m_held;
};

} // namespace serdes

#endif // SERDES_DATA_STROBE_LATCH_H
