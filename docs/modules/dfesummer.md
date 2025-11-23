# DFE Summer 模块小文档

级别：AMS 子模块（RX）

## 概述
DFE Summer（判决反馈均衡求和器）位于 RX 接收链的 CTLE/VGA 之后、Sampler 之前，作用是将主路径的差分信号与“基于历史判决比特生成的反馈信号”进行求和（通常为相减），从而抵消后游符号间干扰（post‑cursor ISI），增大眼图开度并降低误码率。

核心原则：
- 反馈至少延迟 1 UI（使用 b[n−1], b[n−2], …），避免零延迟代数环。
- tap_coeffs（tap 的系数）为各后游系数；init_bits 用于启动阶段预填充历史比特，保证反馈寄存器有定义的初值。
- 当 tap_coeffs 全为 0 时，DFE summer 等效为直通（v_fb=0），不会引入环路或改变主路径输出。

应用场景：高速串行链路（SERDES）RX 端的后游 ISI 取消，与 Sampler/CDR/Adaption 联合工作。
## 接口
- 端口（TDF）：
  - 输入：`sca_tdf::sca_in<double> in_p`, `sca_tdf::sca_in<double> in_n`（主路径差分）
  - 输入：`sca_tdf::sca_in<int> data_in`（来自采样器的判决比特，0/1；如为 DE 域可通过桥接到 TDF）
  - 输出：`sca_tdf::sca_out<double> out_p`, `sca_tdf::sca_out<double> out_n`（均衡后的差分输出）
- 配置键：
  - `tap_coeffs`（[double]）：后游 tap 的系数列表，按 k=1…N 顺序
  - `ui`（double，秒）：单位间隔，用于 TDF 步长与反馈更新节拍
  - `vcm_out`（double，V）：差分输出共模电压
  - `vtap`（double）：比特映射后的反馈电压缩放（单位与 CTLE/VGA 输出匹配）
  - `map_mode`（string）：比特映射模式，`"pm1"`（0→−1，1→+1）或 `"01"`（0→0，1→1）
  - `sat_min`/`sat_max`（double，V，可选）：输出限幅范围（软/硬限制）
  - `init_bits`（[double]，可选）：历史比特初始化值（长度建议与 `taps` 一致）
  - `enable`（bool）：模块使能，默认 true（关闭时直通）
## 参数
- `tap_coeffs`：默认 `[]` 或 `[0,...,0]`，当全部为 0 时等效直通
- `ui`：默认 `2.5e-11`（秒），与系统 UI 保持一致
- `vcm_out`：默认 `0.0`（V），与前级输出共模一致更为稳定
- `vtap`：默认 `1.0`（线性倍数），用于把比特映射量级匹配到主路径幅度
- `map_mode`：默认 `"pm1"`（推荐，抗直流偏置更稳健）
- `sat_min/sat_max`：默认不限制；如需，建议设置为物理输出范围以抑制过补偿与噪声放大
- `init_bits`：默认按 `tap_coeffs.size()` 填充为 0（在 `pm1` 映射下表示“无反馈”）；也可用训练序列对应的 ±1 预填充
- `enable`：默认 true；当 false 时，输出为主路径直通

单位与映射约定：
- `map_mode="pm1"` 时，0/1 → −1/+1；`vtap` 把 ±1 转换到伏特等效幅度
- `map_mode="01"` 时，0/1 → 0/1；`vtap` 把 0/1 转换到目标幅度
## 行为模型
1. 差分输入：`v_main = in_p - in_n`
2. 历史比特与反馈：
   - 启动阶段：`hist = init_bits`（长度 ≈ `tap_coeffs.size()`）
   - 每个 UI 使用历史比特计算反馈电压：`v_fb = Σ_{k=1..N} tap_coeffs[k-1] * map(hist[k-1]) * vtap`
3. 求和与输出：
   - 差分均衡：`v_eq = v_main - v_fb`
   - 可选限幅：软限制 `v_eq_sat = tanh(v_eq / Vsat) * Vsat`（`Vsat = 0.5*(sat_max - sat_min)`），或硬裁剪到 `[sat_min, sat_max]`
   - 差分/共模合成：`out_p = vcm_out + 0.5*v_eq`，`out_n = vcm_out - 0.5*v_eq`
4. 历史队列更新（因果性保障）：
   - 在完成本拍输出后，读取当前判决比特 `bit = data_in.read()`
   - 映射：`map(bit)`（如 `pm1`：0→−1，1→+1）
   - 入队：`hist.push_front(map(bit))`，若超长则 `pop_back()`；确保反馈至少延迟 1 UI
5. 零延迟环路说明：
   - 若把当前比特 `b[n]`直接用于当前输出的反馈，会形成代数环（当前输出依赖当前比特，当前比特又依赖当前输出）
   - 后果：数值不稳定、步长缩小、仿真停滞，且物理上出现“瞬时完美抵消”的非真实行为
   - 规避：严格使用 `b[n−k] (k≥1)`，必要时插入显式 1 UI 延迟或先算输出后更新历史队列
6. tap_coeffs=0 的特例：
   - 当 `tap_coeffs` 全为 0 时，`v_fb=0`，模块等效直通；`init_bits` 对输出无影响，但仍建议保持队列逻辑的因果顺序以兼容后续自适应启用
## 依赖
- 必须：SystemC‑AMS 2.3.x（TDF 域实现）
- 时间步：`set_timestep(ui)`，与 CDR/Sampler 的 UI 对齐
- 数值稳定性：避免代数环；`fs ≫ 链路最高特征频率`，经验 ≥ 20–50×
- 互联建议：若 Sampler 在 DE 域输出比特，可使用官方桥接端口转换到 TDF；前级 CTLE/VGA 推荐使用 `sca_tdf::sca_ltf_nd` 等线性滤波器
- 规范：参数开关（如 `enable`）需明确控制；限幅与映射需与系统单位一致
## 使用示例
1. 基本直通（DFE 关闭或 `tap_coeffs=0`）：
   - 配置：`enable=true`，`tap_coeffs=[0,0,0]`，`ui=2.5e-11`，`vcm_out` 与前级一致
   - 连接：CTLE/VGA → `in_p/in_n`；Sampler → `data_in`；输出 → Sampler 前级
2. 开启 3‑tap DFE：
   - 配置：`tap_coeffs=[0.05, 0.03, 0.02]`，`map_mode="pm1"`，`vtap=1.0`，`init_bits=[0,0,0]`
   - 如需限幅：设置 `sat_min/sat_max` 到物理范围（例如 `0.0/1.2`）
3. 与自适应联动：
   - Adaption 模块根据误差 `e[n]` 更新 `tap_coeffs`（例如 Sign‑LMS：`c_k ← c_k + μ·sign(e[n])·b[n−k]`）
   - DFE summer 仅负责按当前 `tap_coeffs` 求和与抵消，不内置自适应算法
## 测试验证
- 直通一致性：`tap_coeffs=0` 或 `enable=false` 时，`out_p/out_n` 与主路径差分一致（考虑共模合成）
- 因果性验证：检查反馈是否至少延迟 1 UI（波形对齐或在代码中查看更新顺序）
- 眼图开度：对比 `taps=0` 与 `taps>0` 情况下的眼图，验证后游 ISI 取消效果
- 限幅行为：当设置 `sat_min/sat_max`，输出应在范围内软/硬限制，避免过补偿
- init_bits 影响：在不同初始填充下启动瞬态的差异（0 填充稳妥；训练序列填充更快收敛）
- 抗零延迟环路：故意将当前比特用于反馈（仅测试环境），应观察到数值问题或不合理行为；恢复因果更新后应正常
- BER 评估：在 PRBS 输入下，统计误码率变化；验证 DFE 后的 BER 下降
## 变更历史
- v0.2（2025-10-22）：将配置键 `taps` 重命名为 `tap_coeffs`，并明确“DFE 求和依据 tap_coeffs（tap 的系数）进行”，同步更新行为模型、示例与测试描述。
- v0.1（2025-10-22）：首次完整文档；明确零延迟环路的风险与规避、`init_bits` 的作用与设置、`taps=0` 的直通特性；补充接口/参数/行为模型/测试方案与依赖说明。
