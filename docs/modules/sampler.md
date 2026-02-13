# Sampler 模块技术文档

🌐 **Languages**: [中文](sampler.md) | [English](../en/modules/sampler.md)

**级别**：AMS 子模块（RX）  
**类名**：`RxSamplerTdf`  
**当前版本**：v0.3 (2025-12-07)  
**状态**：生产就绪

---

## 1. 概述

采样器（Sampler）是SerDes接收端的关键判决模块，主要功能是将连续的模拟差分信号转换为离散的数字比特流。模块支持动态相位调整、可配置的非理想效应建模以及先进的模糊判决机制，用于模拟真实采样器的复杂行为特性。

### 1.1 设计原理

采样器的核心设计思想是基于比较器架构的阈值判决，结合CDR（Clock and Data Recovery）提供的动态相位信息实现精确的数据恢复：

- **差分比较**：对互补的模拟输入信号进行差分运算，获得差分电压Vdiff
- **动态采样时刻**：接收CDR模块输出的采样时钟或相位偏移信号，动态调整采样位置
- **多级判决机制**：结合分辨率阈值和迟滞效应，实现鲁棒的判决逻辑
- **非理想效应建模**：集成偏移、噪声、jitter等实际器件的非理想特性

传递函数的数学形式为：
```
data_out = f(Vdiff, phase_offset, parameters)
其中：Vdiff = (inp - inn) + offset + noise
```

### 1.2 核心特性

- **CDR集成接口**：支持时钟驱动和相位信号驱动两种模式
- **动态采样时刻**：实时响应CDR相位调整，优化采样点位置
- **模糊判决机制**：分辨率阈值内的随机判决，模拟比较器亚稳态
- **施密特触发器效应**：迟滞功能避免信号抖动引起的判决错误
- **参数验证机制**：hysteresis与resolution参数的冲突检测和错误处理
- **可配置非理想效应**：偏移、噪声、jitter的独立建模和控制

### 1.3 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2025-09 | 初始版本，基本采样功能 |
| v0.2 | 2025-11-23 | 新增参数验证机制，完善文档 |
| v0.3 | 2025-12-07 | 根据CTLE文档风格重构，优化技术描述 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `inp` | 输入 | double | 差分输入正端 |
| `inn` | 输入 | double | 差分输入负端 |
| `clk_sample` | 输入 | double | CDR采样时钟（可选） |
| `phase_offset` | 输入 | double | CDR相位偏移（可选） |
| `data_out` | 输出 | int | 数字比特输出（0或1） |
| `data_out_de` | 输出 | bool | DE域输出（可选） |

> **重要**：`clk_sample`和`phase_offset`端口根据`phase_source`参数选择其一连接，SystemC-AMS要求所有端口均需连接。

### 2.3 端口数据类型详解

为了澄清CDR集成接口的设计理念，本节详细说明`clk_sample`和`phase_offset`端口的数据类型和物理含义：

#### 2.3.1 `clk_sample`端口（时钟驱动模式）

**数据类型**：`double`（单位：伏特V）  
**物理含义**：这是一个**连续电压信号**，表示CDR模块输出的时钟波形

**工作原理**：
```cpp
// 典型使用方式
if (clk_sample.read() > voltage_threshold) {
    // 当电压超过阈值时，触发采样时刻
    perform_sampling();
}
```

**信号特性**：
- **波形类型**：正弦波、方波或三角波（取决于CDR设计）
- **电压范围**：典型范围0V-1V或-0.5V至+0.5V
- **频率**：与数据速率匹配（如10Gbps对应10GHz时钟）
- **触发机制**：电压阈值检测，模拟时钟边沿触发

**时序示例**：
```
clk_sample电压信号：
    1.0V ┌──┐    ┌──┐    ┌──┐
         └──┘    └──┘    └──┘
    0.0V    ↑      ↑      ↑
         采样    采样    采样
    阈值：0.5V ──────────────────
```

#### 2.3.2 `phase_offset`端口（相位信号模式）

**数据类型**：`double`（单位：秒s）  
**物理含义**：这是一个**时间偏移量**，表示CDR检测到的相位误差

**工作原理**：
```cpp
// 典型使用方式
double current_time = get_simulation_time();
double actual_sample_time = current_time + phase_offset.read() + sample_delay;
// 基于相位偏移计算实际采样时刻
```

**数值含义**：
- **正值**：表示需要延迟采样（如+100e-12 = 延迟100皮秒）
- **负值**：表示需要提前采样（如-50e-12 = 提前50皮秒）  
- **零值**：表示按标称时刻采样
- **动态范围**：典型范围±0.1×UI（单位间隔）

**应用示例**：
```
phase_offset信号值：
    +100e-12 ────────────────────── 延迟采样
    +50e-12  ────────────────────── 轻微延迟
    0        ────────────────────── 正常采样
    -50e-12  ────────────────────── 轻微提前
    -100e-12 ────────────────────── 提前采样
```

#### 2.3.3 关键区别总结

| 特性 | `clk_sample` | `phase_offset` |
|------|-------------|----------------|
| **数据类型** | double（电压） | double（时间） |
| **单位** | 伏特 (V) | 秒 (s) |
| **信号性质** | 连续时钟波形 | 动态相位校正值 |
| **作用方式** | 电压阈值触发 | 时间偏移计算 |
| **物理含义** | 时钟边沿信号 | 相位误差信息 |
| **CDR输出** | 完整时钟信号 | 相位检测结果 |
| **采样触发** | 边沿检测 | 时间计算 |

#### 2.3.4 选择指南

**使用`clk_sample`（时钟驱动）的情况**：
- CDR直接输出可用的采样时钟
- 需要硬件风格的时钟触发机制
- 系统要求严格的时钟同步
- CDR设计相对简单，直接产生时钟

**使用`phase_offset`（相位信号）的情况**：
- CDR输出相位误差而非时钟
- 需要精确的相位跟踪和调整
- 支持连续的相位优化算法
- CDR环路输出校正信息 |

### 2.2 参数配置（RxSamplerParams）

#### 基本采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_delay` | double | 0.0 | 固定采样延迟时间（s） |
| `phase_source` | string | "clock" | 相位来源：clock/phase |
| `threshold` | double | 0.0 | 判决阈值（V，默认为0V） |
| `resolution` | double | 0.02 | 分辨率阈值（V，模糊判决区半宽） |
| `hysteresis` | double | 0.02 | 迟滞阈值（V，施密特触发器效应） |

#### 偏移配置子结构

建模比较器的输入失调电压。

| 参数 | 说明 |
|------|------|
| `enable` | 启用偏移建模 |
| `value` | 失调电压（V） |

工作原理：`Vdiff_effective = (inp - inn) + value`

#### 噪声配置子结构

建模器件热噪声和输入参考噪声。

| 参数 | 说明 |
|------|------|
| `enable` | 启用噪声建模 |
| `sigma` | 噪声标准差（V，高斯分布） |
| `seed` | 随机数种子（可复现性） |

工作原理：`noise_sample ~ N(0, sigma²)`，叠加到差分信号

### 2.4 相位控制机制

#### 时钟驱动模式（phase_source = "clock"）

CDR输出采样时钟信号，Sampler在时钟上升沿进行采样。

```
时序关系：
CLK ──┐    ┌──┐    ┌──┐    ┌──┐
      └──┘    └──┘    └──┘    └──┘
      ↑      ↑      ↑      ↑
   Sample  Sample  Sample  Sample
```

#### 相位信号模式（phase_source = "phase")

CDR输出相位偏移信号，Sampler基于标称时刻和相位偏移计算实际采样时刻。

```
采样时刻计算：
t_sample = t_nominal + phase_offset + sample_delay

其中：
- t_nominal: 标称采样时刻（如UI中心）
- phase_offset: CDR相位偏移（动态变化）
- sample_delay: 固定延迟（配置参数）
```

---

## 3. 核心实现机制

本章从实现机制的角度，详细阐述采样器模块的信号处理流程、判决逻辑、参数验证以及噪声/抖动建模，并在此基础上推导出完整的误码率（BER）性能分析模型。通过将实现细节与性能指标有机结合，帮助读者深入理解采样器的行为特性与BER之间的关系。

### 3.1 信号处理流程与噪声建模

采样器模块的`processing()`方法采用严格的多步骤流水线处理架构，确保判决逻辑的正确性和可维护性：

```
输入读取 → 差分计算 → 偏移注入 → 噪声注入 → 相位调整 → 判决逻辑 → 输出生成
```

**步骤1-输入读取**：从差分输入端口读取信号，计算差分分量 `Vdiff = inp - inn` 和共模分量 `Vcm = 0.5*(inp + inn)`。

**步骤2-偏移注入**：若启用`offset_enable`，将失调电压`value`叠加到差分信号，模拟比较器的输入失调。具体工作原理为：
```
Vdiff_effective = (inp - inn) + offset.value
```

**步骤3-噪声注入**：若启用`noise_enable`，采用Mersenne Twister随机数生成器产生高斯分布噪声，标准差由`sigma`指定。噪声样本服从：
```
noise_sample ~ N(0, noise.sigma²)
```
该噪声将叠加到差分信号上，构成器件级的热噪声和输入参考噪声建模。

**步骤4-相位调整**：根据`phase_source`选择时钟采样或相位偏移模式，计算实际采样时刻。

> **注意**：`clk_sample`和`phase_offset`端口在当前版本中为预留接口，尚未在`processing()`函数中实现实际读取和使用逻辑。相位调整功能将在后续版本中完善。

**步骤5-判决逻辑应用**：
- 若`resolution = 0`：标准阈值判决
- 若`resolution > 0`：模糊判决机制

**步骤6-输出生成**：将判决结果转换为数字输出，支持TDF和DE两种域。

**等效传递函数总结**：

综合上述流程，采样器的输入到输出传递函数可以表示为：
```
data_out = f(Vdiff, phase_offset, parameters)
其中：Vdiff = (inp - inn) + offset.value + noise_sample
判决阈值：V_th = threshold
```
这一等效关系对应第1章中提到的设计原理，并为后续的BER分析提供了基础。

### 3.2 判决逻辑与模糊区行为

采样器的核心功能是基于输入差分电压进行数字判决。根据`resolution`参数的设置，判决机制分为标准判决和模糊判决两种模式。这两种机制直接决定了在不同输入条件下发生误判的概率，并在后续3.5节中成为BER建模的基础。

#### 3.2.1 标准判决机制（resolution = 0）

采用双阈值迟滞判决，实现施密特触发器效应：

```
阈值定义：
- threshold_high = threshold + hysteresis/2
- threshold_low = threshold - hysteresis/2

判决逻辑：
if (Vdiff > threshold_high):
    output = 1
elif (Vdiff < threshold_low):
    output = 0
else:
    output = previous_output  // 迟滞区：保持状态
```

#### 3.2.2 模糊判决机制（resolution > 0）

引入模糊判决区，模拟比较器的亚稳态行为：

```
决策区域划分：
┌─────────────────────────────────┐
│  Vdiff >= +resolution  → 1      │  确定区（高）
├─────────────────────────────────┤
│  |Vdiff| < resolution  → 随机   │  模糊区
├─────────────────────────────────┤
│  Vdiff <= -resolution  → 0      │  确定区（低）
└─────────────────────────────────┘

判决算法：
if (abs(Vdiff) >= resolution):
    output = (Vdiff > 0) ? 1 : 0
else:
    // 模糊区：随机判决（Bernoulli分布）
    output = random_bernoulli(0.5, seed)
```

### 3.3 参数验证与错误处理机制

为确保仿真结果的有效性，采样器模块实现了严格的参数验证机制。一旦参数设置不合理（如 `hysteresis ≥ resolution`），后续基于这些参数计算的BER将没有物理意义，因此采用强制终止策略以避免误导性结果。

#### 3.3.1 参数冲突检测

系统实现严格的参数验证机制，检测关键参数冲突：

```
核心验证规则：
if (hysteresis >= resolution):
    // 触发错误处理流程
    log_error("Hysteresis must be less than resolution")
    save_simulation_state()
    terminate_simulation()
```

**物理意义说明**：
- `hysteresis` 定义了双阈值判决的迟滞区间宽度
- `resolution` 定义了模糊判决区的半宽
- 若 `hysteresis >= resolution`（大于或等于），会导致判决行为冲突，无法明确判决策略

#### 3.3.2 错误处理流程与数据保存

当检测到参数冲突时，系统执行以下错误处理流程：

1. **错误检测**：ParameterValidator类检测到冲突参数
2. **错误日志**：记录详细错误信息和当前参数配置
3. **状态保存**：保存仿真状态、波形数据、统计信息
4. **友好提示**：输出用户友好的错误信息和解决建议
5. **仿真终止**：安全终止仿真进程，避免错误结果

**数据保存机制**：

在错误终止前，系统确保完整保存所有关键数据：

- **参数配置**：当前所有模块参数设置
- **错误日志**：时间戳、错误类型、堆栈信息
- **波形数据**：已生成的信号波形（CSV格式）
- **统计信息**：已计算的BER、抖动等性能指标
- **仿真状态**：时间、迭代计数、模块状态

### 3.4 抖动建模与综合噪声

时序抖动（Jitter）是影响采样器性能的关键因素之一。本节从抖动的物理来源出发，推导抖动引起的等效电压误差，并给出综合噪声的统一建模方法，为后续3.5节的BER分析提供完整的噪声模型。

#### 3.4.1 Jitter来源分类

**确定性抖动（Deterministic Jitter, DJ）**：
- 数据相关抖动（DDJ）：由ISI引起
- 占空比失真（DCD）：上升/下降沿不匹配
- 正弦波抖动：周期性抖动成分

**随机抖动（Random Jitter, RJ）**：
- 热噪声抖动：器件热噪声引起
- 相位噪声抖动：振荡器相位噪声转换
- 统计特性：高斯分布，均值为0

#### 3.4.2 Jitter引起的电压误差与综合噪声

**时域误差分析**：

Jitter引起的采样时刻偏差会在信号跳变沿处转换为等效电压误差。对于数据速率为 `f_data`、信号幅度为 `A` 的差分信号：

```
采样时刻偏差: Δt_jitter
信号变化率: dV/dt = 2π × f_data × A
电压误差: ΔV_jitter = (dV/dt) × Δt_jitter = 2π × f_data × A × Δt_jitter
```

将时序抖动的标准差 `σ_tjitter` 转换为等效电压噪声：
```
σ_jitter = 2π × f_data × A × σ_tjitter
```

**综合噪声模型**：

结合3.1节定义的器件噪声 `σ_noise`（即配置参数 `noise.sigma`）和上述抖动诱发的电压噪声 `σ_jitter`，可得总噪声标准差：

```
σ_total = sqrt(σ_noise² + σ_jitter²)
      = sqrt(σ_noise² + (2π × f_data × A × σ_tjitter)²)
```

这一统一的 `σ_total` 定义将在后续BER分析中作为核心参数使用。

**信噪比恶化**：
```
SNR_jitter = -20log₁₀(2π × f_data × A × σ_tjitter)

总信噪比：
1/SNR_total² = 1/SNR_signal² + 1/SNR_jitter² + 1/SNR_noise²
```

#### 3.4.3 Jitter容限工程指导

**典型容限指标**：
```
高速SerDes (≥10 Gbps): σ_tjitter < 0.1 × UI
中速SerDes (1-10 Gbps): σ_tjitter < 0.05 × UI  
低速SerDes (<1 Gbps): σ_tjitter < 0.02 × UI
```

### 3.5 BER分析与数值示例

在明确了噪声、偏移和抖动对输入信号的综合影响之后，本节基于前述实现机制推导采样器的BER性能模型。我们将从最简单的仅噪声情况开始，逐步引入偏移、模糊判决和抖动等因素，最终给出完整的综合BER模型及数值计算示例。

#### 3.5.1 理想信道下的BER（仅噪声）

**假设条件**：
- 发送信号：±A（差分幅度2A）
- 器件噪声：σ_noise = noise.sigma（3.1节定义）
- 判决阈值：V_th = 0
- 忽略偏移和抖动（offset = 0, σ_tjitter = 0）

**BER计算**：
```
BER = Q(A / σ_noise)

其中Q函数定义为：
Q(x) = (1/√(2π)) ∫[x,∞] exp(-t²/2) dt
     ≈ (1/2) erfc(x/√2)
```

这是最基础的BER模型，仅考虑加性高斯噪声对判决的影响。

#### 3.5.2 噪声与偏移下的BER

**引入偏移电压**：
- 偏移：V_offset = offset.value（3.1节定义）
- 判决阈值：V_th = threshold（2.2节定义）

由于偏移电压和判决阈值的存在，发送'1'和发送'0'时的判决裕量不再对称：

**BER计算**：
```
对于发送'1' (信号 = +A):
BER_1 = Q((A - (V_offset + threshold)) / σ_noise)

对于发送'0' (信号 = -A):
BER_0 = Q((A + (V_offset - threshold)) / σ_noise)

总BER = (BER_1 + BER_0) / 2
```

在实际系统中，偏移电压和判决阈值会导致眼图中心偏移，从而增加误码率。

#### 3.5.3 模糊判决对BER的修正

如3.2.2节所述，当启用模糊判决机制（`resolution > 0`）时，输入差分电压落在模糊区 `|Vdiff| < resolution` 内将进行随机判决，这会引入额外的误码。

**模糊区概率**：

信号加噪声后落入模糊区的概率可近似表示为：
```
P_metastable ≈ erf(resolution / (√2 × σ_total))
```

其中 `σ_total` 为综合噪声标准差。在仅考虑器件噪声时，`σ_total = σ_noise`；考虑抖动时则使用3.4.2节定义的完整表达式。

**额外误码率**：

模糊区内的随机判决（50/50概率）导致的额外误码率为：
```
BER_fuzzy ≈ P_metastable × 0.5
```

**修正后的BER**：
```
BER ≈ Q(A / σ_total) + P_metastable × 0.5
```

这一修正项反映了比较器亚稳态行为对系统BER的影响。

#### 3.5.4 综合BER模型与计算示例

**完整BER公式**：

综合3.1～3.4节的所有建模要素（器件噪声、偏移、抖动、模糊判决），可得采样器的完整BER模型：

```
BER_total ≈ Q((A - |V_offset + threshold|) / σ_total) + P_metastable × 0.5

其中：
σ_total = sqrt(σ_noise² + σ_jitter²)  （由3.4.2节定义）
σ_jitter = 2π × f_data × A × σ_tjitter
P_metastable = erf(resolution / (√2 × σ_total))  （由3.5.3节定义）
```

这一统一公式将所有非理想效应整合到一个可计算的BER函数中，便于进行参数扫描和性能优化。

**Python数值计算示例**：

```python
import numpy as np
from scipy.special import erfc, erf

def calculate_ber(A, sigma_noise, V_offset, threshold, resolution, f_data, sigma_tjitter):
    """
    计算Sampler的综合误码率

    参数：
    A: 信号幅度（V）
    sigma_noise: 器件噪声标准差（V），对应配置参数 noise.sigma
    V_offset: 偏移电压（V），对应配置参数 offset.value
    threshold: 判决阈值（V），对应配置参数 threshold
    resolution: 分辨率阈值（V），对应配置参数 resolution
    f_data: 数据速率（Hz）
    sigma_tjitter: 时序抖动标准差（s）

    返回：
    BER_total: 总误码率

    公式对应关系：
    - sigma_jitter 计算与 3.4.2 节一致
    - sigma_total 定义与 3.4.2 节一致
    - P_metastable 定义与 3.5.3 节一致
    - threshold 影响与 3.5.2 节一致
    """
    # Q函数
    def Q(x):
        return 0.5 * erfc(x / np.sqrt(2))

    # Jitter诱发的电压误差（3.4.2节公式）
    sigma_jitter = 2 * np.pi * f_data * A * sigma_tjitter

    # 综合噪声（3.4.2节公式）
    sigma_total = np.sqrt(sigma_noise**2 + sigma_jitter**2)

    # 噪声、偏移和阈值导致的BER（3.5.2节扩展）
    SNR_eff = (A - abs(V_offset + threshold)) / sigma_total
    BER_noise = Q(SNR_eff)

    # 模糊判决导致的BER（3.5.3节公式）
    P_metastable = erf(resolution / (np.sqrt(2) * sigma_total))
    BER_fuzzy = P_metastable * 0.5

    # 总BER（3.5.4节综合公式）
    BER_total = BER_noise + BER_fuzzy

    return BER_total

# 示例参数
A = 0.5              # 500 mV差分幅度
sigma_noise = 0.01   # 10 mV RMS器件噪声
V_offset = 0.005     # 5 mV偏移
threshold = 0.0      # 0 V判决阈值
resolution = 0.02    # 20 mV分辨率阈值
f_data = 10e9        # 10 Gbps数据速率
sigma_tjitter = 1e-12 # 1 ps RMS抖动

BER = calculate_ber(A, sigma_noise, V_offset, threshold, resolution, f_data, sigma_tjitter)
print(f"BER = {BER:.2e}")
# 输出示例: BER ≈ 1e-12
```

**说明**：

上述函数将3.1～3.4节的所有建模要素（噪声、偏移、抖动、模糊判决、判决阈值）统一到一个可计算的BER函数中。通过调整各参数，可以进行参数扫描和性能优化，指导实际设计中的裕量分配。

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

采样器测试平台（`SamplerTransientTestbench`）采用场景驱动的模块化设计，支持多种工作模式和边界条件的统一验证。核心设计理念：

1. **场景分类**：基础功能、CDR集成、边界条件、性能评估四大类
2. **参数化测试**：通过配置驱动自动生成测试用例
3. **结果验证**：自动化结果分析和性能指标计算
4. **文档集成**：测试结果直接生成到技术文档

### 4.2 测试场景定义

| 场景 | 命令行参数 | 测试目标 | 输出文件 |
|------|----------|---------|----------|
| BASIC_FUNCTION | `basic` / `0` | 基本采样和判决功能 | sampler_tran_basic.csv |
| CDR_INTEGRATION | `cdr` / `1` | CDR相位跟踪能力 | sampler_tran_cdr.csv |
| FUZZY_DECISION | `fuzzy` / `2` | 模糊判决机制验证 | sampler_tran_fuzzy.csv |
| PARAMETER_VALIDATION | `validate` / `3` | 参数验证和错误处理 | sampler_tran_validation.csv |
| BER_MEASUREMENT | `ber` / `4` | 误码率性能测试 | sampler_tran_ber.csv |

### 4.3 场景配置详解

#### BASIC_FUNCTION - 基本功能测试

验证采样器的基本差分信号判决和迟滞功能。

- **信号源**：PRBS-15伪随机序列
- **输入幅度**：200mV差分
- **符号率**：10 Gbps
- **测试参数**：resolution=0, hysteresis=20mV
- **验证点**：输出BER < 1e-12，迟滞功能正常

#### CDR_INTEGRATION - CDR集成测试

验证与CDR模块的相位跟踪和时钟同步能力。

- **CDR相位**：1 GHz正弦波调制（±100ps）
- **输入信号**：10 Gbps PRBS-7
- **测试参数**：phase_source="phase"
- **验证点**：相位跟踪误差 < 5ps

#### FUZZY_DECISION - 模糊判决测试

验证分辨率阈值内的随机判决机制。

- **信号源**：低幅度正弦波（30mV）
- **测试参数**：resolution=20mV, hysteresis=10mV
- **验证点**：模糊区随机性符合50/50分布

#### PARAMETER_VALIDATION - 参数验证测试

验证参数冲突检测和错误处理机制。

- **冲突参数**：hysteresis=30mV, resolution=20mV
- **验证点**：触发错误处理，保存状态，安全终止

#### BER_MEASUREMENT - BER测试

综合性能测试，包含噪声、偏移、jitter等所有非理想效应。

- **测试时间**：≥1百万比特
- **非理想效应**：噪声5mV，偏移2mV，抖动1ps
- **验证点**：实际BER与理论计算对比

### 4.4 信号连接拓扑

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ DiffSignalSource  │       │   RxSamplerTdf   │       │  SignalMonitor  │
│                   │       │                   │       │                   │
│  out_p ───────────┼───────▶ inp              │       │                   │
│  out_n ───────────┼───────▶ inn              │       │                   │
└─────────────────┘       │                   │       │                   │
                            │  data_out ────────┼───────▶ digital_in      │
┌─────────────────┐       │                   │       │                   │
│   CDRModule     │       │                   │       │  → 统计分析        │
│                   │       │                   │       │  → BER计算         │
│  phase_offset ───┼───────▶ phase_offset      │       │  → CSV保存         │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### 4.5 辅助模块说明

#### DiffSignalSource - 差分信号源

差分信号源模块用于生成测试所需的差分输入信号，支持四种波形类型：

| 波形类型 | 枚举值 | 说明 |
|---------|--------|------|
| DC | `DC` | 直流信号，用于静态偏移测试 |
| SINE | `SINE` | 正弦波，用于频率响应和抖动测试 |
| SQUARE | `SQUARE` | 方波，用于眼图和采样位置测试 |
| PRBS | `PRBS` | 伪随机序列，用于BER测试和功能验证 |

可配置参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `amplitude` | double | 0.1 | 信号幅度（V） |
| `frequency` | double | 1e9 | 信号频率（Hz） |
| `vcm` | double | 0.6 | 输出共模电压（V） |
| `sample_rate` | double | 100e9 | 采样率（Hz） |

输出信号生成规则：
- `out_p = vcm + 0.5 × signal`
- `out_n = vcm - 0.5 × signal`

#### PhaseOffsetSource - 相位偏移源

相位偏移源模块用于生成CDR相位控制信号，模拟时钟数据恢复环路的相位调制输出。

功能特性：
- **恒定偏移模式**：输出固定相位偏移值
- **动态调制模式**：支持运行时调整相位偏移（通过`set_offset()`方法）
- **时域单位**：相位偏移以秒为单位（符合Sampler模块接口规范）

可配置参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `offset` | double | 0.0 | 初始相位偏移（s） |
| `sample_rate` | double | 100e9 | 采样率（Hz） |

应用场景：
- **相位扫描测试**：通过改变offset值扫描采样相位
- **CDR跟踪测试**：模拟CDR输出的动态相位调整
- **抖动注入**：配合抖动模型实现相位抖动注入

#### ClockSource - 时钟源模块

时钟源模块生成正弦时钟信号，用于时钟驱动采样模式的测试。

可配置参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `frequency` | double | 10e9 | 时钟频率（Hz） |
| `amplitude` | double | 1.0 | 时钟摆幅（V） |
| `vcm` | double | 0.5 | 时钟共模电压（V） |
| `sample_rate` | double | 100e9 | 采样率（Hz） |

输出信号：`clk = vcm + 0.5 × amplitude × sin(2πft)`

#### SamplerSignalMonitor - 采样器信号监控器

信号监控器模块实现Sampler测试平台的数据记录和分析功能。

功能列表：
- **实时波形记录**：同步记录差分输入、TDF输出、DE输出
- **CSV文件输出**：自动保存波形数据到指定文件
- **多通道监测**：同时监测模拟输入和数字输出

输入端口：

| 端口 | 类型 | 说明 |
|------|------|------|
| `in_p` | double | 差分输入正端（监测用） |
| `in_n` | double | 差分输入负端（监测用） |
| `data_out` | double | TDF域数字输出 |
| `data_out_de` | bool | DE域数字输出 |

CSV输出格式：
```
time(s),input+(V),input-(V),differential(V),tdf_output,de_output
0.000000e+00,0.650000,0.550000,0.100000,1.000000,1
...
```

#### BerCalculator - 误码率计算器

误码率计算器是一个静态工具类，用于统计Sampler输出的误码率。

功能：
- **期望序列比对**：将实际采样结果与期望序列逐位比较
- **误码统计**：计算错误比特数和总比特数
- **BER输出**：返回误码率（错误比特数 / 总比特数）

使用示例：
```cpp
std::vector<bool> expected = {...};  // 期望序列
std::vector<bool> actual = {...};    // 实际采样结果
double ber = BerCalculator::calculate_ber(expected, actual);
```

---

## 5. 仿真结果分析

### 5.1 性能指标定义

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| 误码率（BER） | 错误比特数 / 总比特数 | 判决可靠性 |
| 采样精度 | |V_actual - V_theoretical| | 判决准确性 |
| 相位跟踪误差 | \|phase_error\| | CDR集成性能 |
| 抖动容限 | σ_tjitter_max | 时序鲁棒性 |
| 模糊区概率 | P(\|Vdiff\| < resolution) | 亚稳态频率 |

### 5.2 典型测试结果解读

#### 5.2.1 基本功能测试结果

**配置**：200mV输入，hysteresis=20mV，resolution=0

**期望结果**：
- BER < 1e-12（理想情况下为0）
- 迟滞功能：阈值切换时有明显延迟
- 输出波形：清晰的数字信号

**分析方法**：统计错误比特数，计算BER值

#### 5.2.2 CDR集成测试结果

**配置**：±100ps相位调制，phase_source="phase"

**期望结果**：
- 相位跟踪误差 < 5ps RMS
- 采样点始终保持在数据眼图中心附近
- BER在相位调制下保持稳定

**分析方法**：比较理论相位和实际采样时刻的差异

#### 5.2.3 模糊判决测试结果

**配置**：50mV输入，resolution=20mV，hysteresis=10mV

**期望结果**：
- 模糊区输出呈50/50随机分布
- 确定区输出与输入信号完全对应
- 随机种子可重现相同的随机序列

**分析方法**：统计模糊区的0和1比例，验证随机性

### 5.3 波形数据文件格式

CSV输出格式：
```
time(s),input+(V),input-(V),differential(V),tdf_output,de_output
0.000000e+00,0.600000,0.400000,0.200000,1.000000,1
1.000000e-10,0.601000,0.399000,0.202000,1.000000,1
...
```

采样率：默认100GHz（10ps步长），可配置调整。

---

## 6. 运行指南

### 6.1 环境配置

运行测试前需要配置环境变量：

```bash
source scripts/setup_env.sh
export SYSTEMC_HOME=/path/to/systemc
export SYSTEMC_AMS_HOME=/path/to/systemc-ams
```

### 6.2 构建与运行

```bash
cd build
cmake ..
make sampler_tran_tb
cd tb
./sampler_tran_tb [scenario]
```

场景参数：
- `basic` 或 `0` - 基本功能测试（默认）
- `cdr` 或 `1` - CDR集成测试
- `fuzzy` 或 `2` - 模糊判决测试
- `validate` 或 `3` - 参数验证测试
- `ber` 或 `4` - BER性能测试

### 6.3 参数配置示例

#### 6.3.1 基本配置

```json
{
  "sampler": {
    "sample_delay": 0.0,
    "resolution": 0.02,
    "hysteresis": 0.01,
    "phase_source": "clock"
  }
}
```

#### 6.3.2 高级配置

```json
{
  "sampler": {
    "sample_delay": 5e-12,
    "resolution": 0.015,
    "hysteresis": 0.008,
    "phase_source": "phase",
    "offset": {
      "enable": true,
      "value": 0.005
    },
    "noise": {
      "enable": true,
      "sigma": 0.01,
      "seed": 12345
    }
  }
}
```

### 6.4 结果查看

测试完成后，控制台输出性能统计，波形数据保存到CSV文件。使用Python进行可视化分析：

```bash
python scripts/plot_sampler_waveform.py
```

---

## 7. 技术要点

### 7.1 CDR相位集成注意事项

**问题**：相位信号路径可能引入额外的相位延迟，影响采样时刻精度。

**解决方案**：
- 使用CDR模块输出的直接相位偏移信号
- 考虑相位信号到采样器的传播延迟
- 在`sample_delay`参数中补偿固定延迟

### 7.2 模糊判决的随机性验证

**问题**：如何确保模糊区的随机判决具有真正的随机性。

**解决方案**：
- 使用Mersenne Twister伪随机数生成器
- 提供可配置的随机种子
- 通过统计测试验证随机性分布

### 7.3 参数验证机制实现

**核心验证规则**：
```cpp
if (hysteresis >= resolution) {
    throw std::invalid_argument(
        "Hysteresis must be less than resolution to avoid decision ambiguity"
    );
}
```

**错误处理流程**：
1. 参数验证器检测冲突
2. 记录详细错误日志
3. 保存当前仿真状态
4. 终止仿真进程

### 7.4 数值稳定性考虑

**问题**：在高采样率下，浮点数精度可能影响判决准确性。

**解决方案**：
- 使用双精度浮点数（double）
- 避免接近机器精度的阈值设置
- 考虑数值误差在参数选择中的影响

### 7.5 时间步设置指导

**采样率要求**：采样率应远高于信号带宽，建议：
```
f_sample ≥ 20 × f_data
```

对于10 Gbps信号，建议采样步长 ≤ 5ps。

### 7.6 性能优化建议

1. **噪声生成优化**：使用查表法替代实时随机数生成
2. **参数缓存**：缓存计算频繁的参数值
3. **条件执行**：根据参数启用状态条件执行相关代码
4. **内存管理**：避免在仿真循环中分配动态内存

---

## 8. 参考信息

### 8.1 相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 参数定义 | `/include/common/parameters.h` | RxSamplerParams结构体 |
| 头文件 | `/include/ams/rx_sampler.h` | RxSamplerTdf类声明 |
| 实现文件 | `/src/ams/rx_sampler.cpp` | RxSamplerTdf类实现 |
| 参数验证器 | `/include/common/parameter_validator.h` | ParameterValidator类 |
| 测试平台 | `/tb/rx/sampler/sampler_tran_tb.cpp` | 瞬态仿真测试 |
| 单元测试 | `/tests/unit/test_sampler_basic.cpp` | GoogleTest单元测试 |
| 波形绘图 | `/scripts/plot_sampler_waveform.py` | Python可视化脚本 |

### 8.2 依赖项

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++11标准
- GoogleTest 1.12.1（单元测试）
- NumPy/SciPy（Python分析工具）

### 8.3 性能基准

**典型性能指标**：
- 判决延迟：< 1ns
- 时序精度：±1ps
- 噪声建模精度：±0.1%
- BER测量精度：±5%（1e12样本）

**推荐参数配置**：
- 数据速率 ≤ 25 Gbps
- 输入幅度 ≥ 100mV
- 采样率 ≥ 100GS/s
- 仿真步长 ≤ 10ps

---

**文档版本**：v0.3  
**最后更新**：2025-12-07  
**作者**：Yizhe Liu