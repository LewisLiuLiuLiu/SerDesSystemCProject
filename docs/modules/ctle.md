# CTLE 模块文档

级别：AMS 子模块（RX）

## 概述
差分输入/差分输出的连续时间线性均衡器（CTLE），采用多零点/多极点的传递函数。支持输入偏移、输入端噪声注入（可开关）、输出饱和限制，以及可配置的差分输出共模电压。

## 接口
- 端口（TDF）：
  - 输入：`sca_tdf::sca_in<double> in_p`, `sca_tdf::sca_in<double> in_n`
  - 输入（可选，PSRR 开启时）：`sca_tdf::sca_in<double> vdd`（电源节点电压）
  - 输出：`sca_tdf::sca_out<double> out_p`, `sca_tdf::sca_out<double> out_n`
- 配置键：
  - `dc_gain`（double）：直流增益（线性倍数）
  - `zeros`（[double] Hz）：零点频率列表
  - `poles`（[double] Hz）：极点频率列表
  - `vcm_out`（double）：差分输出共模电压（V），同时为 CMFB 目标值
  - `offset_enable`（bool）：是否启用输入偏移
  - `vos`（double）：输入偏移电压（V），当 `offset_enable=true` 生效
  - `noise_enable`（bool）：是否启用噪声
  - `vnoise_sigma`（double）：噪声标准差（V），当 `noise_enable=true` 生效
  - `sat_min`（double）：输出最小电压（V）
  - `sat_max`（double）：输出最大电压（V）
  - `psrr`（object）：电源抑制路径配置
    - `enable`（bool）
    - `gain`（double）：PSRR 路径增益（线性倍数）
    - `zeros`（[double] Hz）
    - `poles`（[double] Hz）
    - `vdd_nom`（double）：名义电源电压（V）
  - `cmfb`（object）：共模反馈环路配置
    - `enable`（bool）
    - `bandwidth`（double）：环路带宽（Hz），一阶近似
    - `loop_gain`（double）：环路增益（线性倍数）
    - 说明：CMFB 的目标共模由全局 `vcm_out` 提供
  - `cmrr`（object）：输入共模到差分的泄漏配置
    - `enable`（bool）
    - `gain`（double）：CM→DIFF 路径直流比例因子（线性倍数）
    - `zeros`（[double] Hz）
    - `poles`（[double] Hz）

## 参数
- 主传递函数：多零点与多极点（`zeros`/`poles`），配合 `dc_gain` 构成 `H_ctle(s)`
- 差分输出共模：`vcm_out` 决定 `out_p/out_n` 的共模电平（若开启 CMFB 则由环路调节到该目标）
- 输入偏移：`vos` 加到差分输入上（在 `offset_enable` 为真时）
- 噪声：在输入端注入高斯噪声（在 `noise_enable` 为真时，幅度由 `vnoise_sigma` 决定）
- 饱和：使用双曲正切 `tanh` 软限制，在生成 `out_p/out_n` 前对差分信号进行压制；`sat_min/sat_max` 定义输出允许范围，对应等效饱和值 `Vsat = 0.5*(sat_max - sat_min)`
- PSRR：
  - `psrr.enable`: 默认 false
  - `psrr.gain`: 默认 0.0（关闭时无馈通）
  - `psrr.zeros`/`psrr.poles`: 可选，构造 `H_psrr(s)`
  - `psrr.vdd_nom`: 名义电源，用于计算纹波
- CMFB：
  - `cmfb.enable`: 默认 false
  - `cmfb.bandwidth`: 默认 1e6（Hz）
  - `cmfb.loop_gain`: 默认 1.0
  - 目标共模：使用全局 `vcm_out` 作为设定点
- CMRR：
  - `cmrr.enable`: 默认 false
  - `cmrr.gain`: 默认 0.0（理想无泄漏）
  - `cmrr.zeros`/`cmrr.poles`: 可选，构造 `H_cmrr(s)`

## 行为模型
1. 差分输入、偏移与噪声：`vin_diff = (in_p - in_n) + (offset_enable ? vos : 0) + (noise_enable ? N(0, vnoise_sigma) : 0)`；输入共模：`vin_cm = 0.5*(in_p + in_n)`
2. 主线性滤波：根据 `zeros, poles, dc_gain` 构造 `H_ctle(s)`，使用 `sca_tdf::sca_ltf_nd` 进行时域滤波，得到 `vout_diff_l`。
3. 饱和限制：`Vsat = 0.5*(sat_max - sat_min)`，`vout_diff = tanh(vout_diff_l / Vsat) * Vsat`（软限制，映射到输出范围）
4. PSRR 路径（可选）：`vdd_ripple = vdd - vdd_nom`；构造 `H_psrr(s)`；`vout_psrr_diff = ltf_nd(H_psrr){ vdd_ripple }`
5. CMRR 路径（可选）：构造 `H_cmrr(s)`；`vout_cmrr_diff = ltf_nd(H_cmrr){ vin_cm }`
6. 差分输出合成：`vout_total = vout_diff + (psrr.enable ? vout_psrr_diff : 0) + (cmrr.enable ? vout_cmrr_diff : 0)`
7. 共模与 CMFB（可选）：
   - 无 CMFB：`vcm_eff = vcm_out`
   - 有 CMFB：测量共模 `vcm_meas = LPF{0.5*(out_p_prev + out_n_prev)}`；误差 `e_cm = vcm_out - vcm_meas`；控制（近似一阶）`Δvcm = ltf_nd(loop_gain / (1 + s/(2π*bandwidth))){ e_cm }`；应用 `vcm_eff = vcm_out + Δvcm`
8. 差分输出与共模：
   - `out_p = vcm_eff + 0.5 * vout_total`
   - `out_n = vcm_eff - 0.5 * vout_total`

## 依赖
- 必须：SystemC‑AMS 2.3.x
- 数值与稳定性：零极点与时间步需匹配（`fs ≫ f_max`，经验 ≥ 20–50×），避免不稳定或过高阶近似；闭环（CMFB）在 TDF 中实现时须避免代数环（引入测量低通或一步延迟）
- 规范遵循：噪声与偏移必须由 `noise_enable` 与 `offset_enable` 控制开关
- 建模域建议：CTLE/PSRR/CMRR 推荐使用 TDF 的 `sca_tdf::sca_ltf_nd`；CMFB 闭环如需更自然的连续实现可采用 LSF/ELN

## 使用示例
1. 在 RX 的 CTLE 阶段配置：`dc_gain`, `zeros`, `poles`, `vcm_out`, `sat_min/sat_max`
2. 如需偏移/噪声：设置 `offset_enable=true` 与 `vos`；`noise_enable=true` 与 `vnoise_sigma`
3. 开启 PSRR：`psrr.enable=true, psrr.gain=0.05, psrr.zeros=[1e6], psrr.poles=[1e3, 1e7], psrr.vdd_nom=1.0`；连接 `vdd` 注入单频或随机纹波
4. 开启 CMFB：`vcm_out=0.6, cmfb.enable=true, cmfb.bandwidth=1e6, cmfb.loop_gain=2.0`；对输出共模施加阶跃扰动观察稳定性
5. 开启 CMRR：`cmrr.enable=true, cmrr.gain=1e-3, cmrr.poles=[1e5]`；在输入叠加共模扫频信号，观察差分端残余

## 测试验证
- 频响一致性：Bode 幅相响应与目标零极点模型一致
- 共模正确性：`out_p/out_n` 共模为 `vcm_out`（无 CMFB）或 `vcm_out + Δvcm`（有 CMFB）
- 偏移开关：`offset_enable` 控制下，输出偏移符合 `vos`
- 噪声开关：`noise_enable` 控制下，输出噪底与 `vnoise_sigma` 对应
- 饱和行为：输出在 `sat_min/sat_max` 范围内有效限制
- PSRR 验证：在 `vdd` 注入单频纹波，测量差分输出的抑制曲线，与 `gain/zeros/poles` 一致
- CMFB 验证：施加共模阶跃/慢变扰动，观察输出共模回到 `vcm_out` 的时间常数与稳定性
- CMRR 验证：在输入叠加共模扫频信号，测得差分残余曲线，与 `cmrr.gain` 与配置的 `zeros/poles` 一致

## 变更历史
- v0.3（2025-10-22）：统一以 `vcm_out` 作为共模设定点与 CMFB 目标；补充 PSRR/CMFB/CMRR 的接口与行为模型；加入 TDF/LSF/ELN 建模建议；移除本地路径链接
- v0.2：新增 PSRR/CMFB/CMRR 的接口、参数与行为模型；增加 `vdd` 输入引脚与验证方案
- v0.1 初始模板，按需求与规范补充差分与噪声/偏移、饱和与零极点配置
