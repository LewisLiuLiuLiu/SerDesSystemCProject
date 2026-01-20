# Wave Generation 模块技术文档

**级别**：AMS 顶层模块  
**类名**：`WaveGenerationTdf`  
**当前版本**：v0.1 (2026-01-20)  
**状态**：开发中

---

## 1. 概述

波形生成模块（Wave Generation）是 SerDes 系统的信号源，负责生成测试激励信号，支持 PRBS 伪随机序列、单比特脉冲、抖动注入和多种调制方式，为后续 TX、Channel、RX 链路提供输入激励。

### 1.1 设计原理

Wave Generation 模块的核心设计思想是提供多种信号生成模式，并通过多种机制注入真实物理效应：

- **PRBS 生成**：基于线性反馈移位寄存器（LFSR）的反馈移位机制，通过特定抽头位置的异或运算生成最大长度伪随机序列
- **单比特脉冲（SBR）**：生成单个脉冲信号，用于测试系统对阶跃响应和瞬态特性的响应能力
- **抖动建模**：在时间域注入随机抖动（RJ）和周期性抖动（SJ），模拟实际链路的时序不确定性
- **NRZ 调制**：将二进制比特映射为双电平信号（+1.0V/-1.0V），符合高速串行链路的 NRZ 编码规范

LFSR 的反馈函数形式为：
```
feedback = (LFSR[tap1] XOR LFSR[tap2])
LFSR[t+1] = (LFSR[t] << 1) | feedback
```
其中 tap1 和 tap2 是根据 PRBS 标准多项式确定的抽头位置。

单比特脉冲的数学形式为：
```
v(t) = { +1.0V,  0 ≤ t < T_pulse
       { -1.0V,  t ≥ T_pulse
```
其中 T_pulse 是脉冲宽度，由 `single_pulse` 参数控制。

### 1.2 核心特性

- **标准 PRBS 支持**：支持 PRBS7、PRBS9、PRBS15、PRBS23、PRBS31 五种标准序列
- **单比特脉冲（SBR）**：生成可配置宽度的单个脉冲，用于瞬态响应测试
- **抖动注入**：
  - 随机抖动（RJ）：高斯分布时延扰动
  - 周期性抖动（SJ）：多音正弦相位调制
- **可配置输出**：支持自定义多项式和初始状态
- **可复现性**：通过随机种子控制，确保仿真结果可重复
- **高效实现**：基于位操作的 LFSR，低计算开销
- **TDF 域建模**：使用 SystemC-AMS TDF 域实现精确时序控制

### 1.3 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2026-01-20 | 初始版本，支持标准 PRBS 生成、单比特脉冲、RJ/SJ 抖动注入、NRZ 调制 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `out` | 输出 | double | 波形输出端口（NRZ 调制的双电平信号） |

> **说明**：Wave Generation 模块作为信号源，仅提供单输出端口，直接连接到 TX 模块的输入端。

### 2.2 参数配置（WaveGenParams）

#### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | PRBSType | PRBS31 | 信号类型（PRBS7/9/15/23/31） |
| `poly` | string | "x^31 + x^28 + 1" | 自定义多项式表达式（当 type=custom 时使用） |
| `init` | string | "0x7FFFFFFF" | LFSR 初始状态（十六进制字符串） |
| `single_pulse` | double | 0.0 | 单比特脉冲宽度（秒，type=PRBS 时忽略） |

**PRBSType 枚举值**：
- `PRBS7`：7 位 LFSR，周期 2⁷-1 = 127
- `PRBS9`：9 位 LFSR，周期 2⁹-1 = 511
- `PRBS15`：15 位 LFSR，周期 2¹⁵-1 = 32767
- `PRBS23`：23 位 LFSR，周期 2²³-1 = 8388607
- `PRBS31`：31 位 LFSR，周期 2³¹-1 = 2147483647（最常用）

**单比特脉冲模式**：
- 当 `single_pulse > 0` 时，模块进入单比特脉冲模式
- 在仿真开始时输出 +1.0V，持续 `single_pulse` 时间后切换为 -1.0V
- 用于测试系统的瞬态响应和阶跃响应特性
- 与 PRBS 模式互斥，单比特脉冲模式下 LFSR 不工作

#### JitterParams 子结构

抖动参数结构，用于在时间域注入随机和周期性抖动，模拟实际链路的时序不确定性。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `RJ_sigma` | double | 0.0 | 随机抖动标准差（秒） |
| `SJ_freq` | vector&lt;double&gt; | [] | 周期性抖动频率数组（Hz） |
| `SJ_pp` | vector&lt;double&gt; | [] | 周期性抖动峰峰值数组（秒） |

**工作原理**：
- **随机抖动（RJ）**：通过高斯分布采样生成时延偏移，`RJ_sigma` 控制抖动的统计特性。RJ 通常由热噪声、散粒噪声等物理过程产生，服从高斯分布。
- **周期性抖动（SJ）**：支持多个正弦抖动分量叠加，每个分量由 `SJ_freq[i]` 和 `SJ_pp[i]` 定义。SJ 通常由电源纹波、串扰等周期性干扰产生，在频域表现为离散的谱线。

**实现方式**：
```cpp
// 随机抖动
std::normal_distribution<double> dist(0.0, RJ_sigma);
double jitter_rj = dist(m_rng);

// 周期性抖动（多音叠加）
double jitter_sj = 0.0;
for (size_t i = 0; i < SJ_freq.size(); ++i) {
    double phase = 2.0 * M_PI * SJ_freq[i] * m_time;
    jitter_sj += SJ_pp[i] * std::sin(phase);
}
```

#### ModulationParams 子结构

调制参数结构，用于配置幅度和相位调制。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `AM` | double | 0.0 | 幅度调制指数（0-1） |
| `PM` | double | 0.0 | 相位调制指数（弧度） |

**工作原理**：
- **幅度调制（AM）**：通过调制指数 `AM` 控制输出信号的幅度变化。调制后的信号为 `v_out = v_bit × (1 + AM × m(t))`，其中 `m(t)` 是调制信号。
- **相位调制（PM）**：通过调制指数 `PM` 控制输出信号的相位变化。调制后的信号为 `v_out = v_bit × cos(ωt + PM × m(t))`。

**注意**：当前实现主要支持 NRZ 调制（二电平信号），AM/PM 参数为预留接口，用于未来扩展多电平调制（PAM-N）或复杂调制方案。

---

## 3. 核心实现机制

### 3.1 信号处理流程

Wave Generation 模块的 `processing()` 方法支持两种生成模式，采用模式选择的多分支架构，确保信号生成的正确性和可维护性：

```
模式选择 → [PRBS 模式 / 单比特脉冲模式] → NRZ 调制 → 抖动注入 → 输出
```

#### PRBS 模式

**步骤1-PRBS 生成**：根据配置的 PRBS 类型，使用线性反馈移位寄存器（LFSR）生成下一个比特。从预定义的 `PRBS_CONFIGS` 表中获取对应类型的抽头位置（tap1、tap2）和掩码，计算反馈位：
```cpp
int prbs_index = static_cast<int>(m_params.type)
const PRBSConfig& config = PRBS_CONFIGS[prbs_index]
feedback = ((m_lfsr_state >> config.tap1) ^ (m_lfsr_state >> config.tap2)) & 0x1
m_lfsr_state = ((m_lfsr_state << 1) | feedback) & config.mask
```

**步骤2-NRZ 调制（PRBS 模式）**：将 LFSR 生成的二进制比特映射为双电平信号：
```cpp
bool bit = (m_lfsr_state & 0x1)
double bit_value = bit ? 1.0 : -1.0
```

#### 单比特脉冲模式

**步骤1-脉冲生成**：根据当前时间判断输出状态：
```cpp
if (m_time < m_params.single_pulse) {
    bit_value = 1.0  // 脉冲高电平
} else {
    bit_value = -1.0 // 脉冲低电平
}
```

单比特脉冲的时序特性：
- 0 ≤ t < T_pulse：输出 +1.0V（上升沿）
- t ≥ T_pulse：输出 -1.0V（下降沿）
- T_pulse 由 `single_pulse` 参数控制

**步骤2-NRZ 调制（单比特脉冲模式）**：单比特脉冲本身已为双电平信号，无需额外调制。

#### 通用步骤

**步骤3-抖动注入**：在两种模式下均支持抖动注入，在时间域注入随机抖动（RJ）和周期性抖动（SJ）：
```cpp
double jitter_offset = 0.0

// 随机抖动（高斯分布）
if (m_params.jitter.RJ_sigma > 0.0) {
    std::normal_distribution<double> dist(0.0, m_params.jitter.RJ_sigma)
    jitter_offset += dist(m_rng)
}

// 周期性抖动（多音叠加）
for (size_t i = 0; i < m_params.jitter.SJ_freq.size(); ++i) {
    double sj_phase = 2.0 * M_PI * m_params.jitter.SJ_freq[i] * m_time
    jitter_offset += m_params.jitter.SJ_pp[i] * std::sin(sj_phase)
}
```

**步骤4-输出**：将调制后的比特值输出到端口，并更新时间戳：
```cpp
out.write(bit_value)
m_time += 1.0 / m_sample_rate
```

### 3.2 关键算法/机制

#### LFSR 反馈机制（PRBS 模式）

线性反馈移位寄存器（LFSR）是 PRBS 生成的核心算法，其数学原理基于有限域 GF(2) 上的多项式运算。

**反馈多项式**：
对于 n 位 LFSR，反馈多项式为：
```
P(x) = x^n + x^k + 1
```
其中 k 是抽头位置。当 LFSR 的状态向量为 S = [s_{n-1}, s_{n-2}, ..., s_0] 时，反馈位为：
```
feedback = s_{n-1} XOR s_{n-1-k}
```

**标准 PRBS 多项式**：

| PRBS 类型 | 多项式 | 抽头位置 | 周期 |
|-----------|--------|----------|------|
| PRBS7 | x⁷ + x⁶ + 1 | 6, 5 | 2⁷-1 = 127 |
| PRBS9 | x⁹ + x⁴ + 1 | 8, 4 | 2⁹-1 = 511 |
| PRBS15 | x¹⁵ + x¹⁴ + 1 | 14, 13 | 2¹⁵-1 = 32767 |
| PRBS23 | x²³ + x¹⁸ + 1 | 22, 17 | 2²³-1 = 8388607 |
| PRBS31 | x³¹ + x²⁸ + 1 | 30, 27 | 2³¹-1 = 2147483647 |

**实现方式**：
使用位操作实现高效的 LFSR 更新，避免浮点运算和复杂的数学库调用：
```cpp
// 提取抽头位的值
unsigned int tap1_bit = (m_lfsr_state >> config.tap1) & 0x1
unsigned int tap2_bit = (m_lfsr_state >> config.tap2) & 0x1

// 异或运算生成反馈
unsigned int feedback = tap1_bit ^ tap2_bit

// 左移并插入反馈位
m_lfsr_state = ((m_lfsr_state << 1) | feedback) & config.mask
```

#### 单比特脉冲生成机制

**时序控制**：
单比特脉冲的生成基于精确的时间比较，利用 TDF 域的离散时间特性：
```cpp
if (m_time < m_params.single_pulse) {
    bit_value = 1.0  // 脉冲高电平期间
} else {
    bit_value = -1.0 // 脉冲结束后
}
```

**上升沿和下降沿**：
- 上升沿：t = 0 时刻，输出从 -1.0V 切换到 +1.0V
- 下降沿：t = T_pulse 时刻，输出从 +1.0V 切换到 -1.0V

**应用场景**：
1. **阶跃响应测试**：验证系统对阶跃信号的响应时间和超调
2. **瞬态特性分析**：测量系统的上升时间、下降时间和建立时间
3. **眼图基础测试**：作为眼图分析的基准信号
4. **调试和诊断**：快速验证信号链路的连通性和极性

#### 抖动生成机制

**随机抖动（RJ）**：
RJ 由热噪声、散粒噪声等随机物理过程产生，服从高斯分布。使用 C++11 标准库的 `std::normal_distribution` 生成高斯随机数：
```cpp
std::mt19937 m_rng  // Mersenne Twister 随机数生成器
std::normal_distribution<double> dist(0.0, RJ_sigma)
double jitter_rj = dist(m_rng)
```

**周期性抖动（SJ）**：
SJ 由电源纹波、串扰等周期性干扰产生，在频域表现为离散的谱线。支持多音正弦叠加：
```cpp
double jitter_sj = 0.0
for (size_t i = 0; i < SJ_freq.size(); ++i) {
    double phase = 2.0 * M_PI * SJ_freq[i] * m_time
    jitter_sj += SJ_pp[i] * std::sin(phase)
}
```

**抖动叠加**：
总抖动为 RJ 和 SJ 的线性叠加：
```
jitter_total = jitter_rj + jitter_sj
```

### 3.3 设计决策说明

#### 为什么选择 LFSR 而非其他 PRBS 生成方法？

**LFSR 的优势**：
1. **计算效率高**：仅使用位运算（移位、异或、与），无浮点运算，适合高速仿真
2. **内存占用小**：仅需存储一个 32 位整数状态，无需预先生成序列
3. **周期可控**：通过多项式选择可精确控制序列周期，满足不同测试需求
4. **标准兼容**：符合 ITU-T O.150 标准定义的 PRBS 序列

**替代方案对比**：
- **预生成序列表**：需要存储大量数据，内存开销大，灵活性差
- **软件伪随机数生成器**：通常产生浮点数，需要额外的量化步骤，效率较低
- **硬件描述语言实现**：虽然精确但移植性差，不适合跨平台仿真

#### 为什么增加单比特脉冲模式？

**单比特脉冲的价值**：
1. **瞬态响应测试**：提供标准的阶跃信号，用于测量系统的上升时间、下降时间和建立时间
2. **调试便利性**：简单的脉冲信号便于快速验证信号链路的连通性和极性
3. **眼图基准**：作为眼图分析的基准信号，帮助理解系统对理想信号的响应
4. **低复杂度**：无需复杂的 LFSR 计算，计算开销极小

**与 PRBS 的互补性**：
- PRBS：用于长期统计特性测试（眼图、BER、抖动分析）
- 单比特脉冲：用于瞬态特性测试（阶跃响应、带宽测量）
- 两种模式配合使用，提供全面的系统验证能力

#### 为什么使用 NRZ 调制？

**NRZ 的优势**：
1. **简单高效**：二电平映射（+1.0V/-1.0V），无需复杂的编码/解码逻辑
2. **带宽利用率高**：每个符号携带 1 比特信息，无冗余开销
3. **SerDes 标准兼容**：符合 PCIe、SATA、SAS 等高速串行接口标准
4. **易于分析**：眼图、抖动等指标在 NRZ 下最直观

**未来扩展**：
文档中预留了 AM/PM 调制参数，为未来支持 PAM-N（多电平调制）等高级调制方案提供接口。

#### 为什么使用 Mersenne Twister 随机数生成器？

**Mersenne Twister 的优势**：
1. **周期极长**：周期为 2¹⁹⁹³⁷-1，远超实际仿真需求
2. **统计特性优良**：通过 Diehard 测试，随机性质量高
3. **速度快**：相比其他高质量随机数生成器，性能较好
4. **可复现性**：通过固定种子确保仿真结果可重复

**注意**：当前实现中抖动仅作为演示性实现，未真正修改采样时间戳。完整的抖动建模需要在 TDF 域中动态调整时间步长或使用 DE-TDF 桥接实现精确的时间调制。

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

Wave Generation 测试平台采用模块化设计，支持多种信号生成模式的统一验证。核心设计理念：

1. **模式驱动**：通过参数配置选择 PRBS 或单比特脉冲模式，每种模式自动配置相应的验证方法
2. **统计验证**：通过 PSD（功率谱密度）和 PDF（概率密度函数）分析验证信号质量
3. **抖动验证**：针对 RJ 和 SJ 抖动，提供专门的统计验证方法
4. **组件复用**：信号监控器、统计分析器等辅助模块可复用

### 4.2 测试场景定义

测试平台支持五种核心测试场景：

| 场景 | 命令行参数 | 测试目标 | 输出文件 |
|------|----------|---------|----------|
| PRBS_BASIC | `prbs` / `0` | PRBS 基本生成和统计特性 | wave_prbs.csv |
| PULSE_TEST | `pulse` / `1` | 单比特脉冲时序特性 | wave_pulse.csv |
| RJ_TEST | `rj` / `2` | 随机抖动注入验证 | wave_rj.csv |
| SJ_TEST | `sj` / `3` | 周期性抖动注入验证 | wave_sj.csv |
| STATS_TEST | `stats` / `4` | PSD/PDF 统计分析 | wave_stats.csv |

### 4.3 场景配置详解

#### PRBS_BASIC - PRBS 基本测试

验证 PRBS 序列生成的基本正确性和统计特性。

- **信号类型**：PRBS31
- **采样率**：80 GHz
- **仿真时长**：1 μs
- **验证点**：
  - 输出电平：+1.0V 或 -1.0V，无中间电平
  - 序列周期：符合 PRBS31 的 2³¹-1 周期
  - 码流平衡：0 和 1 的数量接近相等
  - 统计特性：均值 ≈ 0，峰峰值 = 2.0V

#### PULSE_TEST - 单比特脉冲测试

验证单比特脉冲的时序特性和阶跃响应。

- **信号类型**：单比特脉冲
- **脉冲宽度**：100 ps（可配置）
- **采样率**：80 GHz
- **仿真时长**：1 ns（足够覆盖脉冲上升沿和下降沿）
- **验证点**：
  - 上升沿：t = 0 时刻，输出从 -1.0V 切换到 +1.0V
  - 下降沿：t = T_pulse 时刻，输出从 +1.0V 切换到 -1.0V
  - 脉冲宽度：高电平持续时间 = T_pulse
  - 电平稳定性：高电平和低电平期间无抖动

#### RJ_TEST - 随机抖动测试

验证随机抖动（RJ）注入的有效性。

- **信号类型**：PRBS31 + RJ
- **RJ 标准差**：5 ps（可配置）
- **采样率**：80 GHz
- **仿真时长**：1 μs
- **验证点**：
  - PDF 分布：输出电平的 PDF 应符合高斯分布
  - 统计特性：RMS 值应反映 RJ 的影响
  - 频谱特性：PSD 噪底应因 RJ 而抬升

#### SJ_TEST - 周期性抖动测试

验证周期性抖动（SJ）注入的有效性。

- **信号类型**：PRBS31 + SJ
- **SJ 频率**：5 MHz
- **SJ 峰峰值**：20 ps（可配置）
- **采样率**：80 GHz
- **仿真时长**：1 μs（覆盖至少 5 个 SJ 周期）
- **验证点**：
  - 频谱特性：PSD 在 5 MHz 处应出现明显峰值
  - 周期性：时域波形应呈现周期性调制
  - 峰值幅度：PSD 峰值应与 SJ_pp 成正比

#### STATS_TEST - 统计特性测试

综合验证 PRBS 信号的统计特性，包括 PSD 和 PDF。

- **信号类型**：PRBS31
- **采样率**：80 GHz
- **仿真时长**：10 μs（足够长的数据用于统计分析）
- **验证点**：
  - PSD：应符合 NRZ 信号的频谱特性（sinc² 形状）
  - PDF：双峰分布，峰值对应 +1.0V 和 -1.0V
  - 码流平衡：0 和 1 的数量差 < 1%
  - 自相关：自相关函数在非零延迟处应接近零

### 4.4 信号连接拓扑

测试平台的模块连接关系如下：

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ WaveGeneration  │       │  SignalMonitor  │       │  StatsAnalyzer  │
│     Tdf          │       │                   │       │                   │
│                   │       │                   │       │                   │
│  out ─────────────┼───────▶ in               │       │                   │
└─────────────────┘       │                   │       │  → PSD 计算       │
                            │  waveform ────────┼───────▶  → PDF 计算       │
                            │                   │       │  → 统计指标       │
                            │  → CSV 保存       │       │                   │
                            └─────────────────┘       └─────────────────┘
```

**拓扑说明**：
- WaveGenerationTdf：信号源模块，根据配置生成 PRBS 或单比特脉冲
- SignalMonitor：信号监控器，实时记录波形数据并保存为 CSV 文件
- StatsAnalyzer：统计分析器（Python 后处理），计算 PSD、PDF 和统计指标

### 4.5 辅助模块说明

#### SignalMonitor - 信号监控器

功能：
- 实时记录波形数据（时间戳和电压值）
- 计算基本统计信息（均值、RMS、峰峰值、最大/最小值）
- 输出 CSV 格式波形文件

输出格式：
```
时间(s),波形(V)
0.000000e+00,1.000000
1.250000e-11,-1.000000
2.500000e-11,1.000000
...
```

#### StatsAnalyzer - 统计分析器（Python）

功能：
- 读取 CSV 波形文件
- 计算功率谱密度（PSD）：使用 Welch 方法
- 计算概率密度函数（PDF）：使用直方图估计
- 计算统计指标：均值、标准差、峰峰值、码流平衡度

Python 实现示例：
```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 读取波形数据
t, x = np.loadtxt('wave_prbs.csv', unpack=True)

# PSD 计算
Fs = 80e9
f, Pxx = signal.welch(x, fs=Fs, nperseg=1<<14)
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V²/Hz)')
plt.title('Power Spectral Density')
plt.show()

# PDF 计算
hist, bins = np.histogram(x, bins=256, density=True)
centers = 0.5 * (bins[1:] + bins[:-1])
plt.plot(centers, hist)
plt.xlabel('Amplitude (V)')
plt.ylabel('Probability Density')
plt.title('Probability Density Function')
plt.show()
```

---

## 5. 仿真结果分析

### 5.1 统计指标说明

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| 均值 (mean) | 所有采样点的算术平均 | 反映信号的直流分量，PRBS 应接近 0 |
| RMS | 均方根 | 反映信号的有效值/功率，NRZ 信号应为 1.0V |
| 峰峰值 (peak_to_peak) | 最大值 - 最小值 | 反映信号的动态范围，NRZ 应为 2.0V |
| 码流平衡度 | \|count(1) - count(0)\| / total | 衡量 0 和 1 的数量平衡性，应接近 0 |
| 自相关峰值 | 非零延迟处的最大自相关值 | 衡量序列的随机性，理想 PRBS 应接近 0 |

### 5.2 典型测试结果解读

#### PRBS_BASIC 测试结果示例

配置：PRBS31，采样率 80 GHz，仿真时长 1 μs

期望结果：
- 输出电平：仅 +1.0V 或 -1.0V，无中间电平
- 差分输出均值 ≈ 0（PRBS 信号平均应为零）
- 差分输出峰峰值 = 2.0V（+1.0V - (-1.0V)）
- RMS = 1.0V（NRZ 信号的有效值）
- 码流平衡度 < 1%（0 和 1 的数量差 < 1%）

分析方法：
```python
# Python 分析示例
t, x = np.loadtxt('wave_prbs.csv', unpack=True)
mean = np.mean(x)
rms = np.sqrt(np.mean(x**2))
pp = np.max(x) - np.min(x)
balance = abs(np.sum(x > 0) - np.sum(x < 0)) / len(x)
print(f"均值: {mean:.6f} V")
print(f"RMS: {rms:.6f} V")
print(f"峰峰值: {pp:.6f} V")
print(f"码流平衡度: {balance:.2%}")
```

#### PULSE_TEST 测试结果解读

配置：单比特脉冲，脉冲宽度 100 ps，采样率 80 GHz，仿真时长 1 ns

期望结果：
- 上升沿：t = 0 时刻，输出从 -1.0V 切换到 +1.0V
- 脉冲高电平持续时间 ≈ 100 ps（8 个采样点）
- 下降沿：t = 100 ps 时刻，输出从 +1.0V 切换到 -1.0V
- 脉冲后电平稳定在 -1.0V

分析方法：
```python
# 检测上升沿和下降沿
t, x = np.loadtxt('wave_pulse.csv', unpack=True)
rise_edge_idx = np.where(np.diff(x) > 0)[0][0]
fall_edge_idx = np.where(np.diff(x) < 0)[0][0]
pulse_width = t[fall_edge_idx] - t[rise_edge_idx]
print(f"上升沿时间: {t[rise_edge_idx]:.3e} s")
print(f"下降沿时间: {t[fall_edge_idx]:.3e} s")
print(f"脉冲宽度: {pulse_width:.3e} s")
```

#### RJ_TEST 测试结果解读

配置：PRBS31 + RJ，RJ 标准差 5 ps，采样率 80 GHz，仿真时长 1 μs

期望结果：
- PDF 分布：输出电平的 PDF 应呈现双峰分布（+1.0V 和 -1.0V），但由于 RJ 影响，峰值会略微展宽
- 统计特性：RMS 值应略高于 1.0V（由于 RJ 引入的额外能量）
- 频谱特性：PSD 噪底应因 RJ 而抬升，呈现平坦的噪声底

分析方法：
```python
# PDF 分析
t, x = np.loadtxt('wave_rj.csv', unpack=True)
hist, bins = np.histogram(x, bins=256, density=True)
centers = 0.5 * (bins[1:] + bins[:-1])
peak_1 = centers[np.argmax(hist[:128])]
peak_2 = centers[128 + np.argmax(hist[128:])]
print(f"PDF 峰值 1: {peak_1:.6f} V")
print(f"PDF 峰值 2: {peak_2:.6f} V")

# PSD 分析
Fs = 80e9
f, Pxx = signal.welch(x, fs=Fs, nperseg=1<<14)
noise_floor = np.mean(Pxx[int(len(Pxx)*0.5):])
print(f"PSD 噪底: {noise_floor:.3e} V²/Hz")
```

#### SJ_TEST 测试结果解读

配置：PRBS31 + SJ，SJ 频率 5 MHz，SJ 峰峰值 20 ps，采样率 80 GHz，仿真时长 1 μs

期望结果：
- 频谱特性：PSD 在 5 MHz 处应出现明显峰值，峰值幅度与 SJ_pp 成正比
- 周期性：时域波形应呈现周期性调制，调制频率为 5 MHz
- 峰值幅度：PSD 峰值应比噪底高至少 20 dB（取决于 SJ_pp）

分析方法：
```python
# PSD 峰值检测
t, x = np.loadtxt('wave_sj.csv', unpack=True)
Fs = 80e9
f, Pxx = signal.welch(x, fs=Fs, nperseg=1<<14)
peak_idx = np.argmax(Pxx)
peak_freq = f[peak_idx]
peak_power = Pxx[peak_idx]
noise_floor = np.mean(Pxx[int(len(Pxx)*0.5):])
print(f"PSD 峰值频率: {peak_freq/1e6:.2f} MHz")
print(f"PSD 峰值功率: {peak_power:.3e} V²/Hz")
print(f"PSD 噪底: {noise_floor:.3e} V²/Hz")
print(f"信噪比: {10*np.log10(peak_power/noise_floor):.2f} dB")
```

#### STATS_TEST 测试结果解读

配置：PRBS31，采样率 80 GHz，仿真时长 10 μs

期望结果：
- PSD：应符合 NRZ 信号的频谱特性，呈现 sinc² 形状（主瓣 + 旁瓣）
- PDF：双峰分布，峰值对应 +1.0V 和 -1.0V，峰值尖锐
- 码流平衡：0 和 1 的数量差 < 1%
- 自相关：自相关函数在非零延迟处应接近零（理想 PRBS 的白噪声特性）

分析方法：
```python
# 综合统计分析
t, x = np.loadtxt('wave_stats.csv', unpack=True)
Fs = 80e9

# PSD 分析
f, Pxx = signal.welch(x, fs=Fs, nperseg=1<<14)

# PDF 分析
hist, bins = np.histogram(x, bins=256, density=True)
centers = 0.5 * (bins[1:] + bins[:-1])

# 码流平衡
balance = abs(np.sum(x > 0) - np.sum(x < 0)) / len(x)

# 自相关分析
autocorr = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr_peak = np.max(np.abs(autocorr[1:]))

print(f"码流平衡度: {balance:.2%}")
print(f"自相关峰值: {autocorr_peak:.6f}")
```

### 5.3 波形数据文件格式

CSV 输出格式：
```
时间(s),波形(V)
0.000000e+00,1.000000
1.250000e-11,-1.000000
2.500000e-11,1.000000
3.750000e-11,-1.000000
...
```

**格式说明**：
- 第一列：时间戳（秒），科学计数法表示
- 第二列：波形电压值（V），NRZ 信号为 +1.0V 或 -1.0V
- 列分隔符：逗号（,）
- 无表头行

**采样点数计算**：
```
采样点数 = 仿真时长 / 时间步长
```
例如：仿真时长 1 μs，时间步长 12.5 ps（80 GHz 采样率），则采样点数为 80,000。

**Python 读取示例**：
```python
import numpy as np

# 读取波形数据
t, x = np.loadtxt('wave_prbs.csv', unpack=True, delimiter=',')

# t: 时间数组（秒）
# x: 波形数组（伏特）
```

---

## 6. 运行指南

### 6.1 环境配置

运行测试前需要配置环境变量，确保 SystemC 和 SystemC-AMS 库可用：

```bash
source scripts/setup_env.sh
```

**环境变量说明**：
- `SYSTEMC_HOME`：SystemC 2.3.4 安装路径
- `SYSTEMC_AMS_HOME`：SystemC-AMS 2.3.4 安装路径
- `LD_LIBRARY_PATH`：动态库搜索路径

**手动配置环境变量**（如果 `setup_env.sh` 不可用）：
```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib:$SYSTEMC_AMS_HOME/lib:$LD_LIBRARY_PATH
```

**依赖检查**：
- C++14 编译器（Clang/GCC）
- Python 3.x（用于后处理和可视化）
- NumPy、SciPy、Matplotlib（Python 依赖库）

### 6.2 构建与运行

#### 使用 Makefile 构建

```bash
# 进入项目根目录
cd /mnt/d/systemCProjects/SerDesSystemCProject

# 清理旧的构建文件（可选）
make clean

# 构建库和测试平台
make all

# 仅构建库
make lib

# 仅构建测试平台
make tb

# 查看构建信息
make info
```

**构建输出**：
- 静态库：`build/lib/libserdes.a`
- 测试平台：`build/bin/simple_link_tb`

#### 运行简单链路测试

Wave Generation 模块作为信号源集成在简单链路测试平台中：

```bash
cd build/bin
./simple_link_tb
```

**输出说明**：
- 控制台：显示配置信息和仿真进度
- 波形文件：`simple_link.dat`（SystemC-AMS 表格格式）

**配置文件修改**：

要修改 Wave Generation 模块的参数，编辑配置文件：

```bash
# 编辑 YAML 配置文件
vim ../config/default.yaml
```

**常用配置修改示例**：

1. **切换 PRBS 类型**：
```yaml
wave:
  type: PRBS7  # 改为 PRBS7/9/15/23/31
```

2. **启用单比特脉冲模式**：
```yaml
wave:
  type: PRBS31
  single_pulse: 100e-12  # 100 ps 脉冲宽度
```

3. **注入随机抖动（RJ）**：
```yaml
wave:
  jitter:
    RJ_sigma: 5e-13  # 0.5 ps 标准差
```

4. **注入周期性抖动（SJ）**：
```yaml
wave:
  jitter:
    SJ_freq: [5e6, 10e6]  # 5 MHz 和 10 MHz
    SJ_pp: [2e-12, 1e-12]  # 2 ps 和 1 ps 峰峰值
```

5. **修改随机种子（可复现性）**：
```yaml
global:
  seed: 12345  # 固定种子确保结果可重复
```

修改配置后需要重新构建和运行：

```bash
cd ../..
make clean && make all
cd build/bin
./simple_link_tb
```

#### 使用 CMake 构建（可选）

```bash
# 创建构建目录
mkdir build && cd build

# 配置（Debug 或 Release）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j4

# 运行测试
./bin/simple_link_tb
```

### 6.3 结果查看

#### 控制台输出

运行测试后，控制台会显示以下信息：

```
=== SerDes SystemC-AMS Simple Link Testbench ===
Configuration loaded:
  Sampling rate: 80 GHz
  Data rate: 40 Gbps
  Simulation time: 1 us

Creating TX modules...
Creating Channel module...
Creating RX modules...
Connecting TX chain...
Connecting Channel...
Connecting RX chain...

Creating trace file...
Starting simulation...

=== Simulation completed successfully! ===
Trace file: simple_link.dat
```

#### 波形数据文件格式

输出文件 `simple_link.dat` 使用 SystemC-AMS 表格格式：

```
时间(s)  wave_out(V)  ffe_out(V)  driver_out(V)  channel_out(V)  ctle_out(V)  vga_out(V)  sampler_out(V)  cdr_out(V)
0.0      1.0          0.2         0.16          0.08            0.12         0.48        1.0             1.0
1.25e-11 -1.0         -0.2        -0.16         -0.08           -0.12        -0.48       -1.0            -1.0
2.50e-11 1.0          0.2         0.16          0.08            0.12         0.48        1.0             1.0
...
```

**说明**：
- 第一列：时间戳（秒）
- `wave_out`：Wave Generation 模块输出（NRZ 信号）
- 后续列：链路各模块输出

#### Python 后处理与可视化

使用 Python 脚本读取和分析波形数据：

**基本波形绘制**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 读取波形数据
data = np.loadtxt('simple_link.dat', skiprows=1)
t = data[:, 0]          # 时间（秒）
wave_out = data[:, 1]   # Wave Generation 输出
ctle_out = data[:, 5]   # CTLE 输出

# 绘制波形
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, wave_out, 'b-', linewidth=0.8)
plt.xlabel('时间 (ns)')
plt.ylabel('电压 (V)')
plt.title('Wave Generation 输出')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(t * 1e9, ctle_out, 'r-', linewidth=0.8)
plt.xlabel('时间 (ns)')
plt.ylabel('电压 (V)')
plt.title('CTLE 输出')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('waveform_analysis.png', dpi=150)
plt.show()
```

**PSD（功率谱密度）分析**：

```python
from scipy import signal

Fs = 80e9  # 采样率
f, Pxx = signal.welch(wave_out, fs=Fs, nperseg=1<<14)

plt.figure(figsize=(10, 6))
plt.semilogy(f / 1e9, Pxx)
plt.xlabel('频率 (GHz)')
plt.ylabel('PSD (V²/Hz)')
plt.title('Wave Generation 输出功率谱密度')
plt.grid(True, alpha=0.3)
plt.xlim(0, 40)
plt.savefig('psd_analysis.png', dpi=150)
plt.show()
```

**PDF（概率密度函数）分析**：

```python
# PDF 计算
hist, bins = np.histogram(wave_out, bins=256, density=True)
centers = 0.5 * (bins[1:] + bins[:-1])

plt.figure(figsize=(10, 6))
plt.plot(centers, hist, 'b-', linewidth=1.5)
plt.xlabel('电压 (V)')
plt.ylabel('概率密度')
plt.title('Wave Generation 输出概率密度函数')
plt.grid(True, alpha=0.3)
plt.savefig('pdf_analysis.png', dpi=150)
plt.show()
```

**统计指标计算**：

```python
# 计算统计指标
mean = np.mean(wave_out)
rms = np.sqrt(np.mean(wave_out**2))
pp = np.max(wave_out) - np.min(wave_out)
balance = abs(np.sum(wave_out > 0) - np.sum(wave_out < 0)) / len(wave_out)

print(f"Wave Generation 输出统计:")
print(f"  均值: {mean:.6f} V")
print(f"  RMS: {rms:.6f} V")
print(f"  峰峰值: {pp:.6f} V")
print(f"  码流平衡度: {balance:.2%}")
```

**眼图绘制**（使用 Python）：

```python
# 眼图绘制
UI = 2.5e-11  # 单位间隔（40 Gbps）
phi = (t % UI) / UI  # 归一化相位 [0, 1]

H, xe, ye = np.histogram2d(phi, wave_out, bins=[128, 128], density=True)

plt.figure(figsize=(10, 6))
plt.imshow(H.T, origin='lower', aspect='auto',
           extent=[0, 1, ye[0], ye[-1]], cmap='hot')
plt.xlabel('UI 相位')
plt.ylabel('电压 (V)')
plt.title('Wave Generation 输出眼图')
plt.colorbar(label='密度')
plt.savefig('eye_diagram.png', dpi=150)
plt.show()
```

#### 使用现有脚本

项目提供了 CTLE 波形分析脚本，可以适配用于 Wave Generation 输出分析：

```bash
# 复制并修改脚本以支持 Wave Generation 输出
cp scripts/plot_ctle_waveform.py scripts/plot_wave_gen.py

# 修改脚本以读取 simple_link.dat 并提取 wave_out 列
# 然后运行分析
python scripts/plot_wave_gen.py
```

**注意事项**：
- Wave Generation 模块通常作为链路的一部分运行，输出在 `simple_link.dat` 文件中
- 要单独测试 Wave Generation 模块，需要创建专门的测试平台（当前未提供）
- 仿真时间较长时，建议使用 CMake Release 模式构建以提高性能
- Python 后处理需要安装 NumPy、SciPy、Matplotlib 库

**性能优化建议**：
- 对于长时间仿真（> 10 μs），使用 Release 构建模式
- 减少追踪信号数量以降低文件大小
- 使用 Python 的 `np.memmap` 处理大型数据文件
- 适当降低采样率（`Fs`）以减少数据量，但需满足奈奎斯特准则

---

## 7. 技术要点

### 7.1 LFSR 周期与序列长度

**问题**：不同 PRBS 类型的周期差异巨大，从 127 到 21 亿不等，如何选择合适的类型？

**选择原则**：
- **PRBS7/9**：适用于快速验证和调试（周期短，易于观察重复模式）
- **PRBS15**：适用于中等长度仿真（周期 32767，平衡了随机性和计算效率）
- **PRBS23/31**：适用于长期统计特性测试（周期极长，避免序列重复）

**注意事项**：
- 仿真时长应远小于 PRBS 周期，否则会出现序列重复，影响统计特性
- 对于 PRBS31（周期 21 亿），即使仿真 1ms（80,000 个 UI），序列重复概率仍可忽略
- 如果仿真时长接近 PRBS 周期，应考虑使用更长的 PRBS 类型

### 7.2 单比特脉冲模式的时序精度

**问题**：单比特脉冲的上升沿和下降沿时间精度如何保证？

**解决方案**：
- 单比特脉冲的时序由 TDF 域的时间步长决定，默认为 12.5 ps（80 GHz 采样率）
- 脉冲宽度 `single_pulse` 应为时间步长的整数倍，否则会出现量化误差
- 例如：`single_pulse = 100e-12`（100 ps）对应 8 个时间步，精确可控

**设计考虑**：
- 单比特脉冲模式不使用 LFSR，避免了 LFSR 更新带来的额外计算开销
- 脉冲生成基于简单的时间比较，计算效率极高
- 适用于需要精确控制脉冲宽度的应用场景

### 7.3 抖动建模的局限性

**当前实现的限制**：
- 抖动值（RJ/SJ）仅作为演示性实现，未真正修改采样时间戳
- 在 TDF 域中，每个时间步的输出值是固定的，无法在采样点之间插入抖动
- 抖动值被计算但未应用到输出，因此实际输出波形不受抖动影响

**完整抖动建模的挑战**：
- **方案 1**：动态调整时间步长，但 SystemC-AMS TDF 域要求固定时间步长
- **方案 2**：使用 DE-TDF 桥接，在 DE 域中生成抖动时间戳，再转换为 TDF 域信号
- **方案 3**：在输出端插入插值滤波器，模拟抖动对采样点的影响

**当前建议**：
- 抖动参数可用于验证配置加载和参数传递机制
- 如需完整抖动建模，建议使用专门的抖动注入模块或扩展当前实现

### 7.4 随机数生成器的可复现性

**问题**：如何确保仿真结果的可重复性？

**解决方案**：
- 使用 Mersenne Twister 随机数生成器（`std::mt19937`），支持固定种子
- 通过配置文件中的 `seed` 参数控制随机数序列
- 相同的种子和配置将产生完全相同的 PRBS 序列和抖动值

**注意事项**：
- 随机种子应在仿真开始时初始化，避免中途修改
- 多线程环境下，每个线程应有独立的随机数生成器实例
- 如果使用多个 Wave Generation 模块，应确保每个模块使用不同的种子

### 7.5 NRZ 调制的电平选择

**设计决策**：为什么选择 +1.0V/-1.0V 作为 NRZ 电平？

**原因分析**：
- **归一化设计**：±1.0V 是归一化的电平，便于后续模块的增益调整
- **对称性**：正负电平对称，均值为零，符合 NRZ 信号的统计特性
- **灵活性**：后续模块（如 TX Driver）可以重新映射到实际摆幅（如 0.8V）

**替代方案对比**：
- **0V/1.0V**：不对称，均值不为零，需要额外的直流偏移
- **0V/0.8V**：符合实际驱动器输出，但缺乏灵活性
- **±0.5V**：摆幅较小，信噪比降低

### 7.6 PRBS 与单比特脉冲的互斥性

**设计决策**：为什么 PRBS 模式和单比特脉冲模式互斥？

**原因**：
- **逻辑清晰**：两种模式的目标不同，互斥设计避免了参数冲突
- **计算效率**：单比特脉冲模式不需要 LFSR，节省计算资源
- **代码简洁**：模式选择使用简单的条件分支，易于维护

**实现方式**：
```cpp
if (m_params.single_pulse > 0.0) {
    // 单比特脉冲模式
    bit_value = (m_time < m_params.single_pulse) ? 1.0 : -1.0;
} else {
    // PRBS 模式
    // LFSR 更新和 NRZ 调制
}
```

### 7.7 时间步长与采样率的关系

**关键原则**：采样率应远高于信号带宽，满足奈奎斯特准则。

**当前配置**：
- 默认采样率：80 GHz（时间步长 12.5 ps）
- 数据率：40 Gbps（UI = 25 ps）
- 采样率/数据率 = 2（每个 UI 2 个采样点）

**建议**：
- 对于高速信号（> 10 Gbps），建议采样率 ≥ 2 × 数据率
- 对于需要精确测量上升时间的场景，建议采样率 ≥ 10 × 信号带宽
- 降低采样率可以减少数据量，但可能丢失高频信息

### 7.8 配置文件的加载机制

**工作流程**：
1. 测试平台启动时调用 `ConfigLoader::load_default()`
2. 从 `config/default.yaml` 读取配置
3. 解析 YAML 格式，填充 `SystemParams` 结构体
4. 将 `params.wave` 传递给 `WaveGenerationTdf` 构造函数

**注意事项**：
- 配置文件路径相对于项目根目录
- YAML 格式要求严格，注意缩进和语法
- 如果配置文件不存在或格式错误，程序会抛出异常

### 7.9 与其他信号源的对比

| 特性 | Wave Generation | 标准信号发生器 | 硬件 PRBS 生成器 |
|------|-----------------|----------------|------------------|
| **灵活性** | 高（可配置参数） | 中（预设波形） | 低（固定多项式） |
| **精度** | 受限于时间步长 | 高（硬件精度） | 高（硬件精度） |
| **速度** | 受限于仿真速度 | 高（实时输出） | 高（实时输出） |
| **成本** | 低（软件实现） | 高（硬件设备） | 高（硬件设备） |
| **可复现性** | 完全可控 | 依赖设备 | 完全可控 |
| **应用场景** | 仿真验证 | 实验室测试 | 芯片测试 |

**Wave Generation 的优势**：
- 完全可配置，支持多种 PRBS 类型和抖动模式
- 与 SystemC-AMS 仿真环境无缝集成
- 支持长时间仿真和统计分析
- 无需额外硬件，成本低

### 7.10 已知限制和未来改进方向

**当前限制**：
1. **抖动建模不完整**：抖动值未真正应用到输出波形
2. **仅支持 NRZ 调制**：不支持 PAM-N 等多电平调制
3. **无独立的测试平台**：需要通过简单链路测试平台运行
4. **不支持实时控制**：参数在仿真开始后不可修改

**未来改进方向**：
1. **完整抖动建模**：使用 DE-TDF 桥接实现精确的时间调制
2. **PAM-N 支持**：扩展调制参数，支持 4 电平、8 电平等
3. **独立测试平台**：创建专门的 Wave Generation 测试平台
4. **实时参数调整**：支持通过 DE 域接口动态修改参数
5. **更多 PRBS 类型**：支持自定义多项式和 PRBS-58 等

### 7.11 性能优化建议

**计算效率优化**：
- 使用位操作实现 LFSR，避免浮点运算
- 单比特脉冲模式跳过 LFSR 更新，减少计算开销
- 抖动计算仅在启用时执行，避免不必要的计算

**内存优化**：
- LFSR 状态仅占用 4 字节（32 位整数）
- 不需要预生成序列表，节省内存
- 随机数生成器状态占用约 2.5 KB

**仿真速度优化**：
- 使用 Release 模式构建（`-O2` 优化）
- 减少追踪信号数量，降低 I/O 开销
- 适当降低采样率，减少数据处理量

### 7.12 调试和诊断技巧

**常见问题排查**：

1. **输出电平不正确**：
   - 检查 `type` 参数是否正确设置
   - 验证 LFSR 初始状态 `init` 是否有效
   - 确认 NRZ 调制逻辑是否正确

2. **序列重复过早**：
   - 检查仿真时长是否接近 PRBS 周期
   - 考虑使用更长的 PRBS 类型（如 PRBS31）

3. **抖动未生效**：
   - 确认抖动参数（`RJ_sigma`、`SJ_freq`、`SJ_pp`）是否非零
   - 注意当前实现中抖动仅作为演示，未真正应用

4. **单比特脉冲宽度不准确**：
   - 检查 `single_pulse` 是否为时间步长的整数倍
   - 验证采样率配置是否正确

**调试建议**：
- 使用 `sca_util::sca_trace()` 追踪关键信号
- 在 `processing()` 方法中添加调试输出（仅用于调试）
- 使用 Python 脚本分析输出波形，验证统计特性

---

## 8. 参考信息

### 8.1 相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 参数定义 | `/include/common/parameters.h` | WaveGenParams 结构体 |
| 头文件 | `/include/ams/wave_generation.h` | WaveGenerationTdf 类声明 |
| 实现文件 | `/src/ams/wave_generation.cpp` | WaveGenerationTdf 类实现 |
| 测试平台 | `/tb/simple_link_tb.cpp` | 简单链路测试（集成测试） |
| 配置加载 | `/include/de/config_loader.h` | 配置加载器声明 |
| 配置实现 | `/src/de/config_loader.cpp` | 配置加载器实现 |
| 配置文件 | `/config/default.yaml` | 默认配置（YAML） |
| 配置文件 | `/config/default.json` | 默认配置（JSON） |

### 8.2 依赖项

- **SystemC** 2.3.4
- **SystemC-AMS** 2.3.4
- **C++14** 标准
- **YAML-cpp**（配置文件解析）
- **nlohmann/json**（JSON 配置支持）
- **Python 3.x**（后处理和可视化）
- **NumPy**（数值计算）
- **SciPy**（信号处理）
- **Matplotlib**（绘图）

### 8.3 配置示例

**基本 PRBS 配置**：
```yaml
wave:
  type: PRBS31
  poly: "x^31 + x^28 + 1"
  init: "0x7FFFFFFF"
  single_pulse: 0.0
  jitter:
    RJ_sigma: 0.0
    SJ_freq: []
    SJ_pp: []
  modulation:
    AM: 0.0
    PM: 0.0
```

**单比特脉冲配置**：
```yaml
wave:
  type: PRBS31
  single_pulse: 100e-12  # 100 ps 脉冲宽度
```

**带抖动的 PRBS 配置**：
```yaml
wave:
  type: PRBS31
  jitter:
    RJ_sigma: 5e-13      # 0.5 ps 随机抖动
    SJ_freq: [5e6, 10e6] # 5 MHz 和 10 MHz 周期性抖动
    SJ_pp: [2e-12, 1e-12] # 2 ps 和 1 ps 峰峰值
```

**JSON 格式配置**：
```json
{
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF",
    "single_pulse": 0.0,
    "jitter": {
      "RJ_sigma": 5e-13,
      "SJ_freq": [5e6],
      "SJ_pp": [2e-12]
    },
    "modulation": {
      "AM": 0.0,
      "PM": 0.0
    }
  }
}
```

---

**文档版本**：v0.1  
**最后更新**：2026-01-20  
**作者**：Yizhe Liu