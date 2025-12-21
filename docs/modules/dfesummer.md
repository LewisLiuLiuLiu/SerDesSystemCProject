# DFE Summer 模块技术文档

**级别**：AMS 子模块（RX）  
**类名**：`RxDfeSummerTdf`  
**当前版本**：v0.5 (2025-12-21)  
**状态**：开发中

---

## 1. 概述

DFE Summer（判决反馈均衡求和器）位于 RX 接收链的 CTLE/VGA 之后、Sampler 之前，是实现判决反馈均衡（Decision Feedback Equalization）的核心模块。其主要功能是将主路径的差分信号与基于历史判决比特生成的反馈信号进行求和（减法），从而抵消后游符号间干扰（post-cursor ISI），增大眼图开度并降低误码率。

### 1.1 设计原理

DFE 的核心设计思想是利用已判决的历史符号来预测并抵消当前符号受到的后游 ISI：

- **后游 ISI 来源**：高速信道的频率相关衰减和群延迟导致每个发送符号在时域上"拖尾"，影响后续符号的采样点电压
- **反馈补偿机制**：已判决的历史符号（b[n-1], b[n-2], ...）通过 FIR 滤波器结构生成反馈电压，从当前输入信号中减去
- **因果性约束**：反馈路径必须至少延迟 1 UI，避免形成代数环（当前判决依赖当前输出，当前输出又依赖当前判决）

反馈电压的数学表达式为：
```
v_fb = Σ_{k=1}^{N} c_k × map(b[n-k]) × vtap
```
其中：
- c_k：第 k 个抽头系数（tap coefficient）
- b[n-k]：第 n-k 个 UI 的判决比特（0 或 1）
- map()：比特映射函数（0→-1, 1→+1 或 0→0, 1→1）
- vtap：电压缩放因子，将比特映射值转换为伏特

均衡后的输出为：`v_eq = v_main - v_fb`

### 1.2 核心特性

- **差分架构**：完整的差分信号路径，与前级 CTLE/VGA 和后级 Sampler 兼容
- **多抽头支持**：支持 1-9 个抽头（典型配置为 3-5 个），可根据信道特性灵活配置
- **比特映射模式**：支持 ±1 映射（推荐，抗直流偏置更稳健）和 0/1 映射
- **自适应接口**：通过 DE→TDF 桥接端口接收来自 Adaption 模块的实时抽头更新
- **软饱和机制**：可选的输出限幅，防止过补偿导致的信号失真
- **历史比特接口**：通过 data_in 端口接收外部维护的历史判决数组，简化模块职责

### 1.3 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2025-10-22 | 初始版本，基本 DFE 求和功能 |
| v0.2 | 2025-10-22 | 配置键 `taps` 重命名为 `tap_coeffs` |
| v0.3 | 2025-12-18 | 新增 DE→TDF 抽头更新端口，与 Adaption 模块对接 |
| v0.4 | 2025-12-18 | 改进 `data_in` 接口为数组形式，明确长度约束 |
| v0.5 | 2025-12-21 | 完善文档结构，新增测试平台架构和仿真分析章节 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

| 端口名 | 方向 | 类型 | 说明 |
|--------|------|------|------|
| `in_p` | 输入 | double | 差分输入正端（来自 VGA） |
| `in_n` | 输入 | double | 差分输入负端（来自 VGA） |
| `data_in` | 输入 | vector&lt;int&gt; | 历史判决数据数组 |
| `out_p` | 输出 | double | 差分输出正端（送往 Sampler） |
| `out_n` | 输出 | double | 差分输出负端（送往 Sampler） |

**DE→TDF 参数更新端口**（可选）：

| 端口名 | 方向 | 类型 | 说明 |
|--------|------|------|------|
| `tap_coeffs_de` | 输入 | vector&lt;double&gt; | 来自 Adaption 的抽头系数更新 |

> **关于 data_in 端口**：
> - 数组长度由 `tap_coeffs` 的长度 N 决定
> - `data_in[0]` 为最近一次判决 b[n-1]，`data_in[1]` 为 b[n-2]，...，`data_in[N-1]` 为 b[n-N]
> - 数组由 RX 顶层模块或 Sampler 维护更新，DFE Summer 只读取不修改

### 2.2 参数配置（RxDfeSummerParams）

#### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tap_coeffs` | vector&lt;double&gt; | [] | 后游抽头系数列表，按 k=1...N 顺序 |
| `ui` | double | 2.5e-11 | 单位间隔（秒），用于 TDF 时间步 |
| `vcm_out` | double | 0.0 | 差分输出共模电压（V） |
| `vtap` | double | 1.0 | 比特映射电压缩放因子 |
| `map_mode` | string | "pm1" | 比特映射模式："pm1"（±1）或 "01" |
| `enable` | bool | true | 模块使能，false 时为直通模式 |

#### 饱和限幅参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sat_enable` | bool | false | 启用输出限幅 |
| `sat_min` | double | -0.5 | 输出最小电压（V） |
| `sat_max` | double | 0.5 | 输出最大电压（V） |

#### 初始化参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `init_bits` | vector&lt;double&gt; | [0,...] | 历史比特初始化值，长度必须等于 N |

#### 派生参数

| 参数 | 说明 |
|------|------|
| `tap_count` | 抽头数量 N，等于 `tap_coeffs.size()`，决定 `data_in` 数组长度 |

#### 比特映射模式说明

- **pm1 模式**（推荐）：0 → -1，1 → +1
  - 反馈电压对称，抗直流偏置
  - 当所有历史比特为 0 时，反馈电压为负值

- **01 模式**：0 → 0，1 → 1
  - 反馈电压非对称
  - 需要额外的直流偏置补偿

---

## 3. 核心实现机制

### 3.1 信号处理流程

DFE Summer 模块的 `processing()` 方法采用严格的多步骤处理架构：

```
输入读取 → 使能检查 → 历史数据验证 → 反馈计算 → 差分求和 → 可选限幅 → 共模合成 → 输出
```

**步骤 1 - 输入读取**：从差分输入端口读取信号，计算差分分量 `v_main = in_p - in_n`。

**步骤 2 - 使能检查**：若 `enable=false`，直接将输入信号进行共模合成后输出（直通模式）。

**步骤 3 - 历史数据验证**：读取 `data_in` 数组，验证长度是否等于 `tap_count`。若不匹配：
- 长度不足：用 0 填充
- 长度过多：截断
- 应输出警告日志

**步骤 4 - 反馈计算**：遍历所有抽头，计算总反馈电压：
```cpp
v_fb = 0.0;
for (int k = 0; k < tap_count; k++) {
    double bit_val = map(data_in[k], map_mode);  // 比特映射
    v_fb += tap_coeffs[k] * bit_val * vtap;
}
```

**步骤 5 - 差分求和**：从主路径信号减去反馈电压：`v_eq = v_main - v_fb`

**步骤 6 - 可选限幅**：若启用 `sat_enable`，使用软饱和函数：
```cpp
if (sat_enable) {
    double Vsat = 0.5 * (sat_max - sat_min);
    v_eq = tanh(v_eq / Vsat) * Vsat;
}
```

**步骤 7 - 共模合成**：基于共模电压生成差分输出：
```
out_p = vcm_out + 0.5 * v_eq
out_n = vcm_out - 0.5 * v_eq
```

### 3.2 抽头更新机制

DFE Summer 支持两种抽头配置模式：

#### 静态模式（无自适应）

- 内部 `tap_coeffs` 保持为配置文件中的初始值
- 适用于信道特性已知且稳定的场景
- 抽头系数可通过离线训练或信道仿真预先确定

#### 动态模式（与 Adaption 联动）

1. **初始化阶段**：DFE Summer 按配置中的 `tap_coeffs` 初始化内部系数
2. **运行时更新**：
   - Adaption 模块在 DE 域执行 LMS/Sign-LMS 等算法
   - 新抽头通过 `tap_coeffs_de` 端口传入
   - DFE Summer 在每个 TDF 周期读取最新系数
   - 更新在下一 UI 开始生效

**长度一致性约束**：运行时更新的抽头数组长度必须与初始配置一致，若不匹配应报错或截断/填充。

### 3.3 零延迟环路规避

**问题本质**：
若当前比特 b[n] 直接用于当前输出的反馈计算，会形成代数环：
- 当前输出 v_eq[n] 依赖反馈 v_fb[n]
- 反馈 v_fb[n] 依赖判决 b[n]
- 判决 b[n] 依赖采样值，而采样值来自 v_eq[n]

**后果**：
- 数值不稳定、仿真步长急剧缩小
- 可能导致仿真停滞或发散
- 物理上出现"瞬时完美抵消"的非真实行为

**规避方案**：
- 严格使用历史符号 b[n-k] (k≥1) 进行反馈计算
- `data_in` 数组机制天然保证这一点：`data_in[0]` 最早也是 b[n-1]
- 数组更新由外部模块在当前 UI 判决完成后、下一 UI 开始前执行

### 3.4 直通模式设计

当满足以下任一条件时，DFE Summer 等效为直通：

1. **显式禁用**：`enable = false`
2. **全零抽头**：`tap_coeffs` 所有元素均为 0
3. **空抽头配置**：`tap_coeffs` 为空数组

直通模式下：
- `v_fb = 0`，输出 = 输入（经共模合成）
- `data_in` 数组的值不影响输出
- 但 `data_in` 仍应保持有效长度，以兼容后续自适应启用

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

DFE Summer 测试平台（`DfeSummerTransientTestbench`）采用模块化设计，核心理念：

1. **场景驱动**：通过枚举类型选择不同测试场景，每个场景自动配置信号源、抽头系数和历史比特
2. **组件复用**：差分信号源、历史比特生成器、信号监控器等辅助模块可复用
3. **眼图对比**：重点验证 DFE 开启前后的眼图开度变化

### 4.2 测试场景定义

| 场景 | 命令行参数 | 测试目标 | 输出文件 |
|------|----------|---------|----------|
| BYPASS_TEST | `bypass` / `0` | 验证直通模式一致性 | dfe_summer_bypass.csv |
| BASIC_DFE | `basic` / `1` | 基本 DFE 反馈功能 | dfe_summer_basic.csv |
| MULTI_TAP | `multi` / `2` | 多抽头配置测试 | dfe_summer_multi.csv |
| ADAPTATION | `adapt` / `3` | 自适应抽头更新 | dfe_summer_adapt.csv |
| SATURATION | `sat` / `4` | 大信号饱和测试 | dfe_summer_sat.csv |

### 4.3 场景配置详解

#### BYPASS_TEST - 直通模式测试

验证当 DFE 禁用或抽头全零时，输出与输入保持一致。

- **信号源**：PRBS-7 伪随机序列
- **输入幅度**：100mV
- **抽头配置**：`tap_coeffs = [0, 0, 0]` 或 `enable = false`
- **验证点**：`out_diff ≈ in_diff`（容许微小数值误差）

#### BASIC_DFE - 基本 DFE 测试

验证单抽头或少数抽头配置下的基本反馈功能。

- **信号源**：带 ISI 的 PRBS 信号（通过 ISI 注入模块模拟信道影响）
- **抽头配置**：`tap_coeffs = [0.1]`（单抽头）
- **历史比特**：与输入 PRBS 同步生成
- **验证点**：
  - 反馈电压符合公式计算
  - 输出 ISI 减少

#### MULTI_TAP - 多抽头测试

验证典型 3-5 抽头配置的性能。

- **信号源**：带多游标 ISI 的 PRBS 信号
- **抽头配置**：`tap_coeffs = [0.08, 0.05, 0.03]`
- **验证点**：
  - 各抽头独立生效
  - 总反馈电压正确累加

#### ADAPTATION - 自适应更新测试

验证与 Adaption 模块的联动功能。

- **初始抽头**：`tap_coeffs = [0, 0, 0]`
- **更新序列**：通过 `tap_coeffs_de` 端口逐步注入新抽头值
- **验证点**：
  - 抽头更新在下一 UI 生效
  - 更新过程无毛刺

#### SATURATION - 饱和测试

验证大信号输入下的限幅行为。

- **信号源**：大幅度方波（500mV）
- **抽头配置**：大系数 `tap_coeffs = [0.3, 0.2, 0.1]`
- **饱和配置**：`sat_min = -0.4V, sat_max = 0.4V`
- **验证点**：输出幅度受限于 sat_min/sat_max 范围

### 4.4 信号连接拓扑

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  DiffSignalSrc  │       │  RxDfeSummerTdf │       │  SignalMonitor  │
│                 │       │                 │       │                 │
│  out_p ─────────┼───────▶ in_p            │       │                 │
│  out_n ─────────┼───────▶ in_n            │       │                 │
└─────────────────┘       │                 │       │                 │
                          │  out_p ─────────┼───────▶ in_p            │
┌─────────────────┐       │  out_n ─────────┼───────▶ in_n            │
│  HistoryBitGen  │       │                 │       │  → 统计分析      │
│                 │       │                 │       │  → CSV保存       │
│  data ──────────┼───────▶ data_in         │       └─────────────────┘
└─────────────────┘       │                 │
                          │                 │
┌─────────────────┐       │                 │
│  AdaptionMock   │       │                 │
│  (DE域)         │       │                 │
│  taps ──────────┼───────▶ tap_coeffs_de   │
└─────────────────┘       └─────────────────┘
```

### 4.5 辅助模块说明

#### DiffSignalSource - 差分信号源

与 CTLE 测试平台复用，支持：
- DC、SINE、SQUARE、PRBS 波形
- 可配置幅度、频率、共模电压

#### HistoryBitGenerator - 历史比特生成器

生成与输入信号同步的历史判决数组：
- 输入：当前比特流（参考 PRBS）
- 输出：长度为 N 的历史比特数组
- 功能：维护 FIFO 队列，每 UI 移位更新

#### ISIInjector - ISI 注入模块

为输入信号添加可控的后游 ISI：
- 参数：各游标的 ISI 系数
- 用于模拟信道效应，验证 DFE 取消能力

#### AdaptionMock - 自适应模拟器

模拟 Adaption 模块行为：
- 按预设时间表输出抽头更新
- 用于验证 DE→TDF 端口功能

---

## 5. 仿真结果分析

### 5.1 统计指标说明

#### 通用统计指标

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| 均值 (mean) | 所有采样点的算术平均 | 反映信号的直流分量 |
| RMS | 均方根 $\sqrt{\frac{1}{N}\sum v_i^2}$ | 反映信号的有效值/功率 |
| 峰峰值 (peak_to_peak) | 最大值 - 最小值 | 反映信号的动态范围 |
| 最大/最小值 | 极值统计 | 用于判断饱和等边界行为 |

#### DFE 专用性能指标

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| 眼高 (eye_height) | 眼图中心垂直开口 | 反映信号质量，DFE 后应增大 |
| 眼宽 (eye_width) | 眼图中心水平开口 | 反映定时裕量 |
| ISI 残余 | DFE 后输出的后游 ISI 分量 | 应接近零，反映均衡效果 |
| 反馈电压误差 | 实际 v_fb 与理论 $\sum c_k \cdot \text{map}(b_{n-k}) \cdot V_{tap}$ 的差异 | 验证实现正确性 |
| 均衡增益 | (DFE后眼高 - DFE前眼高) / DFE前眼高 | 量化均衡改善效果 |

#### 指标计算公式

**眼高计算**：
```
eye_height = min(V_1_low) - max(V_0_high)
```
其中 V_1_low 是逻辑"1"采样点的下边界分布，V_0_high 是逻辑"0"采样点的上边界分布。

**ISI 残余计算**：
```
ISI_residual = Σ|h_k - c_k| for k = 1 to N
```
其中 h_k 是信道后游响应，c_k 是 DFE 抽头系数。理想情况下 c_k = h_k，残余为零。

### 5.2 典型测试结果解读

#### BYPASS 测试结果示例

配置：`tap_coeffs = [0, 0, 0]`，输入 100mV PRBS

期望结果：
- 差分输出峰峰值 ≈ 输入峰峰值（200mV）
- 输出波形与输入波形完全一致
- 任何可测量的差异应 < 1μV（数值精度范围）

分析方法：
- 计算输入输出差分信号的互相关系数，应 > 0.9999
- 绘制输入输出差值波形，验证无系统性偏移

#### BASIC_DFE 测试结果解读

配置：单抽头 `tap_coeffs = [0.1]`，带 10% h1 ISI 的 PRBS

假设场景：
- 输入信号：`v_in[n] = v_data[n] + 0.1 × v_data[n-1]`（后游 ISI）
- DFE 反馈：`v_fb[n] = 0.1 × map(b[n-1])`

期望结果：
- 若抽头系数与 ISI 系数匹配，后游 ISI 应被完全抵消
- 眼图垂直开口应增大约 10%
- 输出差分信号的峰峰值标准差应减小

分析方法：
- 对比 DFE 开启前后的眼图叠加
- 计算眼高改善百分比：`(eye_height_after - eye_height_before) / eye_height_before × 100%`

#### MULTI_TAP 测试结果解读

配置：3 抽头 `tap_coeffs = [0.08, 0.05, 0.03]`，带多游标 ISI 的 PRBS

假设场景：
- 信道脉冲响应：h0=1.0, h1=0.08, h2=0.05, h3=0.03
- 输入信号包含 3 个后游 ISI 分量

期望结果：
- 第 1 抽头抵消 h1 ISI
- 第 2 抽头抵消 h2 ISI
- 第 3 抽头抵消 h3 ISI
- 总 ISI 残余 < 5%（考虑量化和噪声）

分析方法：
- 分别验证各抽头的独立贡献：依次置零单个抽头，观察对应 ISI 分量的恢复
- 绘制脉冲响应对比图：信道响应 vs DFE 后响应

数值示例：
| 配置 | 眼高 (mV) | 眼高改善 |
|------|-----------|----------|
| 无 DFE | 160 | - |
| 单抽头 (c1=0.08) | 176 | +10% |
| 3 抽头 (c1/c2/c3) | 192 | +20% |

#### ADAPTATION 测试结果解读

初始抽头全零，t=100ns 时更新为 `[0.05, 0.03]`

期望结果：
- t < 100ns：输出 = 直通（无 DFE 效应）
- t ≥ 100ns + 1 UI：新抽头生效，开始 DFE 补偿
- 过渡期间无输出毛刺或不连续

分析方法：
- 在抽头更新时刻前后各取 10 UI 的波形
- 验证更新生效的延迟 = 1 UI（因果性保障）
- 检查过渡点的波形连续性

验证要点：
- 更新前后的反馈电压变化符合公式计算
- 若更新时刻恰好有符号转换，不应出现异常脉冲

#### SATURATION 测试结果解读

配置：大系数 `tap_coeffs = [0.3, 0.2, 0.1]`，`sat_min = -0.4V, sat_max = 0.4V`

假设场景：
- 输入信号：500mV 方波
- 理论输出（无饱和）：可能超过 ±0.6V

期望结果：
- 输出差分信号被限制在 ±0.4V 范围内
- 限幅采用软饱和（tanh），波形无硬切割痕迹
- 接近饱和区时增益压缩，远离饱和区时线性

分析方法：
- 绘制输入-输出传递曲线，验证软饱和特性
- 测量输出峰峰值，确认 ≤ 0.8V（sat_max - sat_min）

数值示例：
| 输入差分 (mV) | 无限幅输出 (mV) | 限幅后输出 (mV) |
|---------------|-----------------|-----------------|
| 300 | 300 | 295 |
| 400 | 400 | 375 |
| 500 | 500 | 395 |
| 600 | 600 | 399 |

### 5.3 波形数据文件格式

CSV 输出格式：
```
时间(s),输入差分(V),输出差分(V),反馈电压(V),历史比特
0.000000e+00,0.100000,0.100000,0.000000,"[0,0,0]"
2.500000e-11,0.095000,0.085000,0.010000,"[1,0,0]"
5.000000e-11,0.102000,0.092000,0.010000,"[1,1,0]"
...
```

#### 列定义

| 列名 | 单位 | 说明 |
|------|------|------|
| 时间 | 秒 (s) | 仿真时间戳 |
| 输入差分 | 伏特 (V) | in_p - in_n |
| 输出差分 | 伏特 (V) | out_p - out_n |
| 反馈电压 | 伏特 (V) | v_fb = Σ c_k × map(b[n-k]) × vtap |
| 历史比特 | - | JSON 格式数组，如 "[1,0,1]" |

#### 采样策略

- **默认采样率**：每 UI 一个数据点
- **高分辨率模式**：每 UI 10-20 个采样点（用于眼图绘制）
- **文件大小估算**：1μs 仿真 @ 40Gbps ≈ 40,000 行 ≈ 2MB

#### 数据后处理建议

```python
import pandas as pd
import numpy as np

# 读取 CSV
df = pd.read_csv('dfe_summer_basic.csv')

# 计算眼高（简化版）
v_diff = df['输出差分(V)']
eye_height = np.percentile(v_diff[v_diff > 0], 5) - np.percentile(v_diff[v_diff < 0], 95)
print(f"眼高: {eye_height*1000:.2f} mV")

# 计算均衡增益
v_in = df['输入差分(V)']
eye_height_in = np.percentile(v_in[v_in > 0], 5) - np.percentile(v_in[v_in < 0], 95)
eq_gain = (eye_height - eye_height_in) / eye_height_in * 100
print(f"均衡增益: {eq_gain:.1f}%")
```

---

## 6. 运行指南

### 6.1 环境配置

运行测试前需要配置环境变量：

```bash
source scripts/setup_env.sh
```

或手动设置：
```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
```

验证环境配置：
```bash
echo $SYSTEMC_HOME       # 应输出 SystemC 安装路径
echo $SYSTEMC_AMS_HOME   # 应输出 SystemC-AMS 安装路径
```

### 6.2 构建与运行

```bash
# 创建构建目录并编译
mkdir -p build && cd build
cmake ..
make dfe_summer_tran_tb

# 运行测试（在 tb 目录下）
cd tb
./dfe_summer_tran_tb [scenario]
```

场景参数：
| 参数 | 编号 | 说明 |
|------|------|------|
| `bypass` | `0` | 直通模式测试（默认） |
| `basic` | `1` | 基本 DFE 单抽头测试 |
| `multi` | `2` | 多抽头配置测试 |
| `adapt` | `3` | 自适应抽头更新测试 |
| `sat` | `4` | 大信号饱和测试 |

运行示例：
```bash
# 运行基本 DFE 测试
./dfe_summer_tran_tb basic

# 运行多抽头测试
./dfe_summer_tran_tb 2

# 运行全部场景（批量测试）
for i in 0 1 2 3 4; do ./dfe_summer_tran_tb $i; done
```

### 6.3 参数配置示例

DFE Summer 支持通过 JSON 配置文件进行参数化。以下是针对不同应用场景的快速启动配置。

#### 快速验证配置（单抽头）

适用于初步功能验证和调试：

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.1],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  }
}
```

#### 典型应用配置（3抽头）

适用于中等 ISI 信道的常规应用：

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.08, 0.05, 0.03],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true,
    "sat_enable": false
  }
}
```

#### 高性能配置（5抽头+限幅）

适用于严重 ISI 信道，需要更多抽头和输出保护：

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.10, 0.07, 0.05, 0.03, 0.02],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true,
    "sat_enable": true,
    "sat_min": -0.5,
    "sat_max": 0.5,
    "init_bits": [0, 0, 0, 0, 0]
  }
}
```

#### 配置加载方式

```bash
# 使用命令行参数指定配置文件
./dfe_summer_tran_tb basic --config config/dfe_3tap.json

# 或通过环境变量
export SERDES_CONFIG=config/dfe_5tap.json
./dfe_summer_tran_tb multi
```

### 6.4 结果查看

测试完成后，控制台输出统计结果，波形数据保存到 CSV 文件。

#### 控制台输出示例

```
=== DFE Summer Transient Simulation ===
Scenario: BASIC_DFE
Duration: 1.0 us
Tap count: 1
Tap coeffs: [0.10]

--- Statistics ---
Input  diff: mean=0.000 mV, pp=200.0 mV, rms=70.7 mV
Output diff: mean=0.000 mV, pp=185.2 mV, rms=65.3 mV
Feedback:    mean=0.000 mV, pp=100.0 mV, rms=35.4 mV
Eye height improvement: +8.5%

Output file: dfe_summer_basic.csv
```

#### 波形可视化

```bash
# 基本波形查看
python scripts/plot_dfe_waveform.py dfe_summer_basic.csv

# 输入输出对比
python scripts/plot_dfe_waveform.py dfe_summer_basic.csv --compare

# 眼图叠加绘制
python scripts/plot_eye.py dfe_summer_basic.csv --samples-per-ui 20
```

#### 眼图对比分析

对比 DFE 开启前后的眼图变化，量化均衡效果：

```bash
# 生成对比报告
python scripts/compare_eye_dfe.py \
    --before dfe_summer_bypass.csv \
    --after dfe_summer_basic.csv \
    --output report/dfe_comparison.html

# 批量对比多抽头配置
python scripts/compare_eye_dfe.py \
    --before dfe_summer_bypass.csv \
    --after dfe_summer_basic.csv dfe_summer_multi.csv \
    --labels "1-tap" "3-tap" \
    --output report/dfe_multi_comparison.html
```

#### 结果文件汇总

| 场景 | 输出文件 | 主要分析内容 |
|------|----------|--------------|
| BYPASS | dfe_summer_bypass.csv | 基线参考，无 DFE 效应 |
| BASIC_DFE | dfe_summer_basic.csv | 单抽头眼高改善 |
| MULTI_TAP | dfe_summer_multi.csv | 多抽头累积效果 |
| ADAPTATION | dfe_summer_adapt.csv | 抽头更新过渡行为 |
| SATURATION | dfe_summer_sat.csv | 限幅传递曲线 |

---

## 7. 技术要点

### 7.1 因果性保障

DFE 反馈必须严格保证至少 1 UI 的延迟。本设计通过以下机制实现：

1. **data_in 接口设计**：`data_in[0]` 对应 b[n-1]，而非 b[n]
2. **外部更新责任**：历史比特数组由 RX 顶层更新，DFE Summer 只读取
3. **更新时序**：数组更新发生在当前 UI 判决完成后、下一 UI 处理前

若误将当前比特用于反馈，SystemC-AMS 可能报告代数环错误或仿真异常缓慢。

### 7.2 抽头系数范围

- **典型范围**：0.01 ~ 0.3（取决于信道 ISI 特性）
- **符号约定**：正系数用于抵消同相 ISI，负系数用于抵消反相 ISI
- **稳定性约束**：抽头绝对值之和应 < 1，避免反馈发散

建议通过信道仿真或自适应算法确定最优抽头值。

### 7.3 抽头数量选择

| 抽头数量 | 适用场景 | 复杂度 |
|----------|----------|--------|
| 1-2 | 短信道、低 ISI | 低 |
| 3-5 | 中等信道、典型应用 | 中 |
| 6-9 | 长信道、严重 ISI | 高 |

更多抽头可以抵消更多后游 ISI，但增加：
- 功耗和面积
- 自适应收敛时间
- 误差传播风险

### 7.4 与自适应模块的协同

DFE Summer 本身不包含自适应算法，职责分离如下：

| 模块 | 职责 |
|------|------|
| DFE Summer | 读取抽头和历史比特，计算反馈，执行求和 |
| Adaption | 读取误差信号，执行 LMS 算法，输出新抽头 |
| RX 顶层 | 维护历史比特数组，协调各模块时序 |

这种分离简化了各模块设计，便于独立测试和替换。

### 7.5 比特映射模式对比

| 特性 | pm1 模式 | 01 模式 |
|------|----------|---------|
| 映射规则 | 0→-1, 1→+1 | 0→0, 1→1 |
| 反馈对称性 | 对称 | 非对称 |
| 直流分量 | 无（平均为 0） | 有（平均为 0.5×vtap） |
| 推荐场景 | 通用（默认） | 特定协议要求 |

### 7.6 时间步设置

- TDF 时间步应设为 UI：`set_timestep(ui)`
- 与 CDR/Sampler 的 UI 保持一致
- 过小的时间步会增加计算开销，过大则丢失信号细节

---

## 8. 参考信息

### 8.1 相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 参数定义 | `/include/common/parameters.h` | RxDfeSummerParams 结构体 |
| 头文件 | `/include/ams/rx_dfe_summer.h` | RxDfeSummerTdf 类声明 |
| 实现文件 | `/src/ams/rx_dfe_summer.cpp` | RxDfeSummerTdf 类实现 |
| 测试平台 | `/tb/rx/dfe_summer/dfe_summer_tran_tb.cpp` | 瞬态仿真测试 |
| 测试辅助 | `/tb/rx/dfe_summer/dfe_summer_helpers.h` | 信号源和监控器 |
| 单元测试 | `/tests/unit/test_dfe_summer_basic.cpp` | GoogleTest 单元测试 |
| 自适应文档 | `/docs/modules/adaption.md` | Adaption 模块接口说明 |

### 8.2 依赖项

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 标准
- GoogleTest 1.12.1（单元测试）

### 8.3 配置示例

基本配置（3 抽头 DFE）：
```json
{
  "dfe_summer": {
    "tap_coeffs": [0.08, 0.05, 0.03],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  }
}
```

带限幅的配置：
```json
{
  "dfe_summer": {
    "tap_coeffs": [0.1, 0.06, 0.04, 0.02],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "sat_enable": true,
    "sat_min": -0.5,
    "sat_max": 0.5,
    "init_bits": [0, 0, 0, 0],
    "enable": true
  }
}
```

与自适应联动的配置：
```json
{
  "dfe_summer": {
    "tap_coeffs": [0, 0, 0],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  },
  "adaption": {
    "dfe_enable": true,
    "dfe_mu": 0.01,
    "dfe_algorithm": "sign_lms"
  }
}
```

---

**文档版本**：v0.5  
**最后更新**：2025-12-21  
**作者**：SerDes Design Team