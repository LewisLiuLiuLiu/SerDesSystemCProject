# Adaption 模块小文档

级别：DE 顶层模块

## 概述
Adaption 为 SystemC DE 域顶层模块，承载链路运行时自适应算法库，包括 AGC（自动增益控制）、DFE 抽头更新、CDR PI 策略、阈值自适应等。通过 DE‑TDF 桥接机制对 AMS 域模块（CTLE、VGA、Sampler、DFE Summer、CDR）的参数进行在线更新与控制，提升链路在不同通道、速率与噪声条件下的稳态性能（眼图开口、抖动抑制、误码率）与动态响应（锁定时间、收敛速度）。

**设计目标**：
- 在线估计与策略优化：依据采样误差、幅度统计、相位误差等实时指标，动态调整增益、抽头、阈值与相位命令
- 鲁棒性与安全性：提供饱和钳位、速率限制、泄漏、冻结/回退机制，防止算法发散或参数异常
- 多速率与跨域协同：支持快路径（CDR/阈值，高更新频率）与慢路径（AGC/DFE，低更新频率），确保 DE‑TDF 时序对齐与参数原子更新

**与其他模块关系**：
- **与 RX 模块**：驱动 CTLE/VGA/Sampler/DFE 参数；接收采样信号、误差指标、判决结果
- **与 CDR 模块**：输出 PI 控制器的相位命令（phaseCmd），接收相位误差（phaseError）
- **与 SystemConfiguration**：读取场景参数、初值、阈值与门限；支持运行时场景切换（可选）
- **与 EyeAnalyzer（Python）**：离线后处理用于验证 Adaption 效果，但不直接交互

## 接口

### DE 域输入端口（`sca_de::sca_in`）
- **采样与误差指标**（来自 RX/CDR）：
  - `sca_de::sca_in<double> phase_error`：相位误差（CDR 用，单位：秒或归一化 UI）
  - `sca_de::sca_in<double> amplitude_rms`：幅度 RMS 或峰值（AGC 用）
  - `sca_de::sca_in<int> error_count`：误码计数或误差累积（阈值自适应/DFE 用）
  - `sca_de::sca_in<double> isi_metric`：ISI 指标（可选，用于 DFE 更新策略）

- **控制与场景**（来自 SystemConfiguration）：
  - `sca_de::sca_in<int> mode`：运行模式（0=初始化，1=训练，2=数据，3=冻结）
  - `sca_de::sca_in<bool> reset`：全局复位或参数重置信号
  - `sca_de::sca_in<double> scenario_switch`：场景切换事件（可选）

### DE 域输出端口（`sca_de::sca_out`）
- **增益与滤波器参数**（到 RX）：
  - `sca_de::sca_out<double> vga_gain`：VGA 增益设定
  - `sca_de::sca_out<double> ctle_zero`：CTLE 零点频率（可选，支持在线调整）
  - `sca_de::sca_out<double> ctle_pole`：CTLE 极点频率（可选）
  - `sca_de::sca_out<double> ctle_dc_gain`：CTLE 直流增益（可选）

- **DFE 抽头系数**（到 DFE Summer）：
  - `sca_de::sca_out<std::vector<double>> dfe_taps`：DFE 抽头系数数组（tap1, tap2, ..., tapN）
  - 或单独端口：`sca_de::sca_out<double> dfe_tap1/tap2/...`（根据实现选择）

- **采样阈值与迟滞**（到 Sampler）：
  - `sca_de::sca_out<double> sampler_threshold`：采样阈值（V）
  - `sca_de::sca_out<double> sampler_hysteresis`：迟滞窗口（V）

- **CDR 相位命令**（到 CDR PI/相位插值器）：
  - `sca_de::sca_out<double> phase_cmd`：相位插值器命令（秒或归一化步长）

- **诊断与监控**（可选，到 trace 或上层）：
  - `sca_de::sca_out<int> update_count`：更新次数计数
  - `sca_de::sca_out<bool> freeze_flag`：冻结/回退状态标志

### 桥接机制
- **DE‑TDF 桥接**：使用 `sca_de::sca_in/out` 与 TDF 模块的 `sca_tdf::sca_de::sca_in/out` 端口连接
- **时序对齐**：DE 事件驱动或周期驱动更新，参数在下一 TDF 采样周期生效；避免读写竞争与跨域延迟不确定性
- **数据同步**：缓冲机制或时间戳标记，确保参数原子更新（多参数同时切换时）

## 参数

### 全局配置
- `Fs`（double）：系统采样率（Hz），影响更新周期与时序对齐
- `UI`（double）：单位间隔（秒），用于归一化相位误差与抖动指标
- `seed`（int）：随机种子（用于仿真可重复性，与算法随机化扰动相关）
- `update_mode`（string）：调度模式，"event"（事件驱动）| "periodic"（周期驱动）| "multi-rate"（多速率）
- `fast_update_period`（double）：快路径更新周期（秒，用于 CDR/阈值）
- `slow_update_period`（double）：慢路径更新周期（秒，用于 AGC/DFE）

### AGC（自动增益控制）
- `agc.enabled`（bool）：是否启用 AGC
- `agc.target_amplitude`（double）：目标幅度（V 或归一化）
- `agc.kp`（double）：比例系数
- `agc.ki`（double）：积分系数
- `agc.gain_min`（double）：最小增益（dB 或线性）
- `agc.gain_max`（double）：最大增益
- `agc.rate_limit`（double）：增益变化速率限制（dB/s 或 linear/s）
- `agc.initial_gain`（double）：初始增益

### DFE 抽头更新
- `dfe.enabled`（bool）：是否启用 DFE 在线更新
- `dfe.num_taps`（int）：抽头数量（通常 3‑8）
- `dfe.algorithm`（string）：更新算法，"lms" | "sign-lms" | "nlms"
- `dfe.mu`（double）：步长系数（LMS/Sign‑LMS）
- `dfe.leakage`（double）：泄漏系数（0‑1，防止发散）
- `dfe.initial_taps`（array<double>）：初始抽头系数
- `dfe.tap_min`（double）：单个抽头最小值（饱和约束）
- `dfe.tap_max`（double）：单个抽头最大值
- `dfe.freeze_threshold`（double）：误差超过此阈值时冻结更新

### 阈值自适应
- `threshold.enabled`（bool）：是否启用阈值自适应
- `threshold.initial`（double）：初始阈值（V）
- `threshold.hysteresis`（double）：迟滞窗口（V）
- `threshold.adapt_step`（double）：调整步长（V/更新）
- `threshold.target_ber`（double）：目标 BER（用于阈值优化目标，可选）
- `threshold.drift_threshold`（double）：电平漂移阈值（超过时触发调整）

### CDR PI 控制器
- `cdr_pi.enabled`（bool）：是否启用 PI 控制
- `cdr_pi.kp`（double）：比例系数
- `cdr_pi.ki`（double）：积分系数
- `cdr_pi.phase_resolution`（double）：相位命令分辨率（秒）
- `cdr_pi.phase_range`（double）：相位命令范围（±秒）
- `cdr_pi.anti_windup`（bool）：是否启用抗积分饱和
- `cdr_pi.initial_phase`（double）：初始相位命令

### 安全与回退
- `safety.freeze_on_error`（bool）：误差超限时是否冻结所有更新
- `safety.rollback_enable`（bool）：是否支持参数回滚至上次稳定快照
- `safety.snapshot_interval`（double）：稳定快照保存间隔（秒）
- `safety.error_burst_threshold`（int）：误码暴涨阈值（触发冻结/回退）

## 行为模型

### 调度模式

#### 事件驱动（Event-Driven）
- 当输入指标（phase_error、amplitude_rms、error_count）超过门限或变化率达到阈值时，触发对应算法更新
- 优点：响应快速、计算开销低（仅在必要时更新）
- 缺点：需要精细设计门限，避免过度触发或漏触发

#### 周期驱动（Periodic）
- 按固定时间间隔（fast_update_period/slow_update_period）周期性执行估计与更新
- 优点：实现简单、时序可预测
- 缺点：可能在稳态浪费计算资源

#### 多速率（Multi-Rate）
- 快路径（高频更新）：CDR PI、阈值自适应（如每 10‑100 UI 更新一次）
- 慢路径（低频更新）：AGC、DFE 抽头（如每 1000‑10000 UI 更新一次）
- 优点：平衡性能与开销，符合实际硬件分层控制策略

### 算法要点

#### AGC（自动增益控制）
1. **幅度估计**：
   - 从 amplitude_rms 读取当前幅度（RMS/峰值/滑窗平均）
   - 计算误差：`amp_error = target_amplitude - current_amplitude`

2. **PI 控制器更新**：
   - 比例项：`P = kp * amp_error`
   - 积分项：`I += ki * amp_error * dt`
   - 输出增益：`gain = P + I`

3. **约束与钳位**：
   - 增益范围：`gain = clamp(gain, gain_min, gain_max)`
   - 速率限制：`delta_gain = clamp(gain - gain_prev, -rate_limit*dt, rate_limit*dt)`
   - 更新：`gain_out = gain_prev + delta_gain`

4. **输出**：写入 `vga_gain` 端口，下一 TDF 周期生效

#### DFE 抽头更新（LMS/Sign-LMS）
1. **误差获取**：从 error_count 或专用误差端口读取当前误差 `e(n)`

2. **LMS 更新**：
   - 对每个抽头 `i`：`tap[i] = tap[i] + mu * e(n) * x[n-i]`
   - 其中 `x[n-i]` 为延迟 `i` 的符号判决值（需从 RX 获取）

3. **Sign-LMS 简化**：
   - `tap[i] = tap[i] + mu * sign(e(n)) * sign(x[n-i])`
   - 降低乘法复杂度，适合硬件实现

4. **泄漏与约束**：
   - 泄漏：`tap[i] = (1 - leakage) * tap[i]`（防止发散）
   - 饱和：`tap[i] = clamp(tap[i], tap_min, tap_max)`

5. **冻结条件**：若 `|e(n)| > freeze_threshold`，暂停更新（避免异常噪声干扰）

6. **输出**：写入 `dfe_taps` 数组或单独端口，DFE Summer 在下一周期使用新系数

#### 阈值自适应
1. **电平分布估计**：
   - 从采样信号统计高/低电平均值与方差
   - 或使用误码趋势（error_count 变化率）

2. **阈值调整**：
   - 目标：最小化误码或最大化眼图开口
   - 策略：梯度下降或二分查找，向误差减小方向调整
   - `threshold += adapt_step * sign(gradient)`

3. **迟滞更新**：
   - 根据噪声强度动态调整迟滞窗口，平衡抗噪与灵敏度

4. **输出**：写入 `sampler_threshold` 与 `sampler_hysteresis`

#### CDR PI 控制器
1. **相位误差获取**：从 `phase_error` 端口读取（早/晚采样差值或相位检测器输出）

2. **PI 更新**：
   - 比例项：`P = kp * phase_error`
   - 积分项：`I += ki * phase_error * dt`
   - 相位命令：`phase_cmd = P + I`

3. **抗饱和（Anti-Windup）**：
   - 若 `phase_cmd` 超出 `±phase_range`，钳位并停止积分累积
   - `phase_cmd = clamp(phase_cmd, -phase_range, phase_range)`

4. **量化与分辨率**：
   - 按 `phase_resolution` 量化命令：`phase_cmd_q = round(phase_cmd / phase_resolution) * phase_resolution`

5. **输出**：写入 `phase_cmd`，相位插值器根据命令调整采样时刻

### 稳定性与回退机制

#### 冻结策略
- 检测条件：
  - 误码暴涨：`error_count > error_burst_threshold`
  - 幅度异常：`amplitude_rms` 超出预期范围
  - 相位失锁：`|phase_error|` 持续超限
- 动作：暂停所有参数更新，维持当前值，等待条件恢复

#### 回退策略
- 快照保存：每隔 `snapshot_interval` 保存一次当前参数（增益、抽头、阈值、相位）
- 触发条件：冻结持续时间超过阈值或关键指标持续劣化
- 动作：恢复至上次稳定快照参数，重新启动训练

#### 历史记录
- 维护最近 N 次更新的参数与指标历史（用于诊断与回归分析）
- 输出到 trace（update_count、freeze_flag、参数快照）

## 依赖

### SystemC 库
- **必须**：SystemC 2.3.4（DE 域基础）
- **必须**：SystemC‑AMS 2.3.4（DE‑TDF 桥接机制，`sca_de::sca_in/out`）

### 外部模块
- **RX 模块**：提供采样误差、幅度统计、判决结果
- **CDR 模块**：提供相位误差、接收相位命令
- **SystemConfiguration**：提供场景参数与初值

### 算法库（可选）
- 标准 C++ 库：`<vector>`, `<cmath>`, `<algorithm>`（用于数组、饱和、统计）
- 无需外部数学库（基础 LMS/PI 可用标准库实现）

## 时序与跨域桥接

### DE 更新周期与 TDF 对齐
- **DE 域时钟**：Adaption 模块在 DE 域运行，事件驱动或按固定周期（如每 N 个 TDF 周期触发一次）
- **TDF 采样率**：AMS 模块（CTLE/VGA/Sampler/DFE）以系统 Fs 运行，参数在每个 TDF 时间步读取
- **对齐原则**：DE 输出参数在当前事件完成后，下一 TDF 采样周期生效；避免同一 TDF 步内读写竞争

### 跨域数据同步
- **缓冲机制**：DE 输出通过 `sca_de::sca_out` 写入缓冲，TDF 模块在采样时刻读取最新值
- **时间戳标记**：对于多参数同时更新（如 CTLE 零/极/增益），确保原子性（同一时刻生效）
- **延迟处理**：DE→TDF 桥接可能有 1 个 TDF 周期延迟，算法设计需考虑此延迟对稳定性影响

### 多场景/热切换
- **场景切换事件**：通过 `scenario_switch` 或 `mode` 信号触发
- **参数原子切换**：同时更新所有相关参数，避免过渡期参数不一致
- **防抖策略**：切换后进入短暂训练期，冻结误码统计，避免瞬态误触发冻结/回退

## 数据流与追踪

### 输入数据流（RX/Channel → Adaption）
- 采样误差/相位误差：实时或周期性报告
- 幅度统计：RMS/峰值，可由 RX 内部统计模块提供或 Adaption 从原始采样计算
- 误码计数：Sampler 判决结果与理想比特序列比较（需 PRBS 同步）

### 输出参数流（Adaption → RX/CDR）
- 增益/抽头/阈值/相位命令：通过 DE‑TDF 桥接写入对应模块
- 参数更新频率：快路径（CDR/阈值）每 10‑100 UI，慢路径（AGC/DFE）每 1000‑10000 UI

### Trace 输出（用于后处理/回归）
- 记录关键信号：
  - `vga_gain(t)`、`dfe_taps(t)`、`sampler_threshold(t)`、`phase_cmd(t)`
  - `update_count(t)`、`freeze_flag(t)`、`error_count(t)`
- 使用 `sc_trace()` 或 `sca_trace()`（根据信号类型）写入 `.dat` 或 VCD
- 后处理：Python 读取 trace，绘制参数收敛曲线、更新频率分布、冻结事件时间线

### 与 Python EyeAnalyzer 协同
- Adaption 以在线控制为主，不直接调用 Python
- Python 后处理用于离线验证 Adaption 效果：
  - 对比 Adaption 开启/关闭的眼图开口、TJ、BER
  - 分析参数收敛速度与稳定性
  - 生成回归报告（眼高/眼宽改善百分比、锁定时间等）

## 使用示例

### 配置示例（JSON 片段）
```json
{
  "adaption": {
    "update_mode": "multi-rate",
    "fast_update_period": 2.5e-10,
    "slow_update_period": 2.5e-7,
    "agc": {
      "enabled": true,
      "target_amplitude": 0.4,
      "kp": 0.1,
      "ki": 100.0,
      "gain_min": 0.5,
      "gain_max": 8.0,
      "rate_limit": 10.0,
      "initial_gain": 2.0
    },
    "dfe": {
      "enabled": true,
      "num_taps": 5,
      "algorithm": "sign-lms",
      "mu": 1e-4,
      "leakage": 1e-6,
      "initial_taps": [-0.05, -0.02, 0.01, 0.005, 0.002],
      "tap_min": -0.5,
      "tap_max": 0.5,
      "freeze_threshold": 0.3
    },
    "threshold": {
      "enabled": true,
      "initial": 0.0,
      "hysteresis": 0.02,
      "adapt_step": 0.001,
      "drift_threshold": 0.05
    },
    "cdr_pi": {
      "enabled": true,
      "kp": 0.01,
      "ki": 1e-4,
      "phase_resolution": 1e-12,
      "phase_range": 5e-11,
      "anti_windup": true,
      "initial_phase": 0.0
    },
    "safety": {
      "freeze_on_error": true,
      "rollback_enable": true,
      "snapshot_interval": 1e-5,
      "error_burst_threshold": 100
    }
  }
}
```

### SystemC 实例化示例
```cpp
// 创建 Adaption 模块
AdaptionDe adaption("adaption");
adaption.load_config("config/scene_base.json");

// 连接输入（来自 RX/CDR）
adaption.phase_error(cdr.phase_error_out);
adaption.amplitude_rms(rx.amplitude_stat);
adaption.error_count(rx.error_counter);
adaption.mode(sys_config.mode);
adaption.reset(sys_config.reset);

// 连接输出（到 RX/CDR）
adaption.vga_gain(rx.vga_gain_in);
adaption.dfe_taps(rx.dfe_taps_in);
adaption.sampler_threshold(rx.sampler_threshold_in);
adaption.sampler_hysteresis(rx.sampler_hysteresis_in);
adaption.phase_cmd(cdr.phase_cmd_in);

// Trace 输出
sc_trace(tf, adaption.vga_gain, "vga_gain");
sc_trace(tf, adaption.phase_cmd, "phase_cmd");
sc_trace(tf, adaption.update_count, "update_count");
sc_trace(tf, adaption.freeze_flag, "freeze_flag");
```

### 运行流程
1. **初始化**（mode=0）：
   - 加载配置参数，初始化增益/抽头/阈值/相位命令为初值
   - 复位积分器与历史缓冲

2. **训练阶段**（mode=1）：
   - 启用所有自适应算法（AGC/DFE/阈值/CDR PI）
   - 高频更新（快路径）与低频更新（慢路径）并行运行
   - 监控冻结条件，必要时暂停更新

3. **数据阶段**（mode=2）：
   - 维持训练后的参数，或继续低频微调（可选）
   - 统计误码率与眼图指标

4. **冻结阶段**（mode=3）：
   - 停止所有参数更新，维持当前值
   - 用于诊断或切换场景前的稳定期

5. **Trace 输出**：
   - 仿真结束后生成 `results.dat`，包含参数时间序列
   - Python 后处理分析收敛曲线与 Adaption 效果

## 测试验证

### 单元测试

#### AGC 测试
- **阶跃响应**：输入幅度从 0.2V 阶跃至 0.6V，验证增益收敛至目标、无超调
- **稳态误差**：恒定输入下，增益稳定后幅度误差 < 5%
- **速率限制**：快速幅度变化时，增益变化率不超过 `rate_limit`

#### DFE 更新测试
- **ISI 场景**：典型长通道（S 参数定义），注入 PRBS31
- **收敛性**：抽头在 1000‑10000 UI 内收敛至稳定值
- **误码改善**：DFE 开启后误码率下降 > 10x
- **泄漏稳定性**：长时间运行（1e6 UI）抽头不发散

#### 阈值自适应测试
- **电平偏移**：输入信号直流偏移 ±50mV，阈值自动跟踪
- **噪声注入**：RJ sigma=5ps，阈值与迟滞自动调整，误码率最小化
- **鲁棒性**：异常噪声暴涨时不误触发极端阈值

#### CDR PI 测试
- **锁定时间**：初始相位误差 ±0.5 UI，锁定时间 < 1000 UI
- **稳态抖动**：锁定后相位误差 RMS < 0.01 UI
- **抗饱和**：大相位扰动下积分器不溢出，恢复后正常工作
- **噪声容忍度**：相位噪声注入（SJ 5MHz, 2ps），环路稳定

### 集成仿真

#### 标准链路场景
- **短通道**（S21 插损 < 5dB）：
  - AGC/DFE/阈值/CDR 联合工作
  - 眼图开口 > 80% UI，TJ@1e-12 < 0.3 UI
- **长通道**（S21 插损 > 15dB）：
  - DFE 抽头数增加至 8，收敛时间 < 10000 UI
  - 眼图开口 > 50% UI，误码率 < 1e-9

#### 串扰场景
- **强串扰**（NEXT > -30dB）：
  - 阈值自适应补偿串扰引起的电平偏移
  - DFE 抑制 FEXT 导致的 ISI
  - 眼图开口下降 < 20%（vs 无串扰）

#### 双向传输场景
- 启用 Channel 双向开关（S12/S11/S22）
- AGC 与 DFE 应对反射与回波
- 验证参数不发散、误码率在可接受范围

#### 抖动与调制组合
- PRBS31 + RJ(5ps) + SJ(5MHz, 2ps) + DJ
- CDR PI 滤除 SJ，RJ 通过阈值与 DFE 部分抑制
- TJ@1e-12 分解：RJ/DJ 比例符合预期

### 回归指标

- **眼图改善**：
  - Adaption 开启 vs 关闭（固定参数）
  - 眼高改善 > 20%，眼宽改善 > 15%
  - 开口面积改善 > 30%

- **抖动抑制**：
  - TJ@1e-12 降低 > 10%
  - RJ sigma 降低 > 5%（通过 DFE 与阈值优化）

- **锁定时间**：
  - CDR 锁定时间 < 1000 UI（95% 场景）
  - AGC 收敛时间 < 5000 UI
  - DFE 收敛时间 < 10000 UI

- **误码率**：
  - 目标 BER < 1e-12 场景下，实际 BER < 1e-11（含安全裕量）
  - 误码暴涨时冻结/回退机制正常触发，恢复时间 < 2000 UI

- **稳定性**：
  - 1e6 UI 长时间仿真，参数无发散
  - 冻结/回退事件 < 5 次（正常场景）
  - 参数更新次数符合预期（快路径 > 1000 次，慢路径 > 10 次）

## 性能与数值稳定性

### 复杂度评估
- **AGC**：O(1) 每次更新（PI 计算 + 钳位）
- **DFE**：O(num_taps) 每次更新（LMS/Sign-LMS，num_taps 通常 3‑8）
- **阈值自适应**：O(1) 每次更新（梯度估计 + 调整）
- **CDR PI**：O(1) 每次更新（PI 计算 + 量化）
- **总开销**：快路径每 10‑100 UI 更新一次，慢路径每 1000‑10000 UI 更新一次；对仿真性能影响 < 5%

### 数值稳定性
- **步长选择**：
  - AGC：kp/ki 根据环路带宽与阻尼系数设计（典型 kp=0.01‑0.1，ki=10‑1000）
  - DFE：mu 根据信号功率与噪声调整（典型 1e-5 ‑ 1e-3）
  - CDR PI：kp/ki 根据锁定时间与稳定性权衡（典型 kp=0.001‑0.1，ki=1e-5 ‑ 1e-3）

- **定点/浮点**：
  - SystemC 仿真使用 double（浮点），无溢出风险
  - 硬件实现需考虑定点量化与饱和处理（本文档仅涵盖行为模型）

- **饱和与钳位**：
  - 所有输出参数必须钳位至合理范围（gain_min/max、tap_min/max、phase_range）
  - 积分器使用抗饱和策略，防止长时间累积导致溢出

- **泄漏**：
  - DFE 抽头泄漏（1e-6 ‑ 1e-4）防止噪声累积导致发散
  - AGC/CDR 积分器可选泄漏（通常不需要，依赖饱和约束）

### 并发性与竞态
- **多输入指标合并**：
  - 若多个输入信号同时触发更新（phase_error + amplitude_rms），需定义优先级或合并策略
  - 建议：快路径（CDR/阈值）优先，慢路径（AGC/DFE）延后

- **跨域竞态**：
  - DE 写 + TDF 读：通过 SystemC‑AMS 桥接机制保证原子性，无需额外锁
  - 多参数同时更新：确保在同一 DE 事件内完成，TDF 在下一周期统一读取

## 变更历史

### v0.1（初稿，2025-10-30）
- 建立模块框架：概述、接口、参数、行为模型、依赖、使用示例、测试验证
- 定义四大算法：AGC、DFE 抽头更新、阈值自适应、CDR PI 控制器
- 提出多速率调度与冻结/回退机制
- 明确 DE‑TDF 桥接时序与跨域同步策略
- 提供 JSON 配置示例与 SystemC 实例化代码
- 定义单元测试、集成仿真与回归指标

### 后续计划
- v0.2：补充 CTLE 参数在线调整接口与策略（零/极/增益自适应）
- v0.3：增加多场景热切换详细流程与防抖策略
- v0.4：与实际 RX/CDR 模块接口联调，更新端口定义与信号名称
- v0.5：完善回归测试套件，补充边界条件与异常场景（如输入信号丢失、配置缺失）
