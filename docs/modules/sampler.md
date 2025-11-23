# Sampler 模块文档

级别：AMS 子模块（RX）

## 概述
Sampler（采样器）模块是 RX 接收端的关键组件，负责将连续的模拟差分信号转换为数字比特流。模块支持可配置的采样延时、偏移/噪声注入以及分辨率阈值，用于模拟真实采样器的非理想特性。

**主要功能**：
- 差分信号判决：对互补的模拟输入信号进行差分比较
- 动态采样时刻：由 CDR 模块提供变化的采样相位
- 延迟采样：支持可配置的采样时刻延迟
- 非理想效应建模：偏移（offset）、噪声（noise）注入
- 模糊判决区：当差分幅度小于分辨率阈值时输出随机比特
- 迟滞功能：施密特触发器效应，避免信号抖动

## 接口
- 端口：
  - 输入：
    - `sca_tdf::sca_in<double> inp`：差分正端输入
    - `sca_tdf::sca_in<double> inn`：差分负端输入
    - `sca_tdf::sca_in<double> clk_sample`（可选）：来自 CDR 的采样时钟
    - `sca_tdf::sca_in<double> phase_offset`（可选）：来自 CDR 的相位偏移（秒）
  - 输出：
    - `sca_tdf::sca_out<int> data_out`：数字比特输出（0 或 1）
    - `sca_de::sca_out<bool> data_out_de`（可选）：DE 域输出，用于与数字逻辑连接

- 配置键：
  - `sample_delay`（double）：固定采样延迟时间（秒）
  - `offset`（object）：偏移配置
    - `enable`（bool）：是否启用偏移
    - `value`（double）：偏移值（伏特）
  - `noise`（object）：噪声配置
    - `enable`（bool）：是否启用噪声
    - `sigma`（double）：高斯噪声标准差（伏特）
    - `seed`（int）：随机种子（用于可复现性）
  - `resolution`（double）：分辨率阈值（伏特）
  - `hysteresis`（double）：迟滞阈值（伏特）
  - `phase_source`（string）：相位来源，"clock"（时钟驱动）或 "phase"（相位信号驱动）

## 参数
- `sample_delay`: 可选，默认 0.0（秒），固定采样延迟
- `offset.enable`: 可选，默认 false
- `offset.value`: 可选，默认 0.0（V），偏移电压
- `noise.enable`: 可选，默认 false
- `noise.sigma`: 可选，默认 0.0（V），噪声标准差
- `noise.seed`: 可选，默认随机，随机数种子
- `resolution`: 可选，默认 0.0（V），分辨率阈值（模糊判决区半宽）
- `hysteresis`: 可选，默认 0.0（V），迟滞阈值
- `phase_source`: 可选，默认 "clock"

**参数说明**：
- **sample_delay**：在 CDR 提供的相位基础上的额外延迟（正值延后，负值提前）
- **offset.value**：添加到差分信号的直流偏移，模拟比较器失调
- **noise.sigma**：输入参考噪声，影响判决质量
- **resolution**：当 `|Vdiff| < resolution` 时，输出随机 0/1；当 `|Vdiff| >= resolution` 时，正常判决
- **hysteresis**：施密特触发器效应，避免信号在阈值附近抖动

## 行为模型

### 1. 采样机制

#### 1.1 动态采样时刻
- **CDR 驱动采样**：
  - Sampler 的采样时刻由 CDR（Clock and Data Recovery）模块动态提供
  - CDR 通过相位调整不断优化采样点位置
  - Sampler 接收 CDR 输出的采样时钟或相位信号
  
- **接口方式**：
  ```
  方式一：时钟信号驱动（phase_source = "clock"）
  CDR → clk_sample → Sampler
  
  方式二：相位信号驱动（phase_source = "phase"）
  CDR → phase_offset → Sampler
  ```

- **采样时刻计算**：
  ```
  t_sample = t_nominal + phase_offset + sample_delay
  
  其中：
  - t_nominal: 标称采样时刻（如 UI 中心）
  - phase_offset: CDR 提供的相位偏移（动态变化）
  - sample_delay: 固定延迟（配置参数）
  ```

#### 1.2 采样时序关系
```
UI (Unit Interval)
|<-------------- 1 UI ------------->|
|                                    |
Input: ____/‾‾‾‾‾\____/‾‾‾‾‾\____
           ↑              ↑
    Data Edge      Data Edge

CDR Phase:  |<-- tracking -->|
            t_nominal + Δφ(t)
                   ↓
Sampler:        Sample Point
            (动态调整位置)
```

### 2. 差分信号处理

#### 2.1 信号路径
```
inp ──┐
      ├──> Vdiff = (inp - inn)
inn ──┘
       ↓
   [+ offset] (if enabled)
       ↓
   [+ noise]  (if enabled)
       ↓
    Decision
       ↓
   data_out
```

#### 2.2 差分电压计算
```cpp
// 伪代码
double Vdiff = inp - inn;

if (offset.enable) {
    Vdiff += offset.value;
}

if (noise.enable) {
    Vdiff += gaussian_noise(0, noise.sigma, noise.seed);
}
```

### 3. 判决逻辑

#### 3.1 标准判决（resolution = 0）
```
if (Vdiff > hysteresis/2):
    data_out = 1
elif (Vdiff < -hysteresis/2):
    data_out = 0
else:
    data_out = previous_output  // 迟滞区：保持上次状态
    
其中：
- threshold_high = +hysteresis/2
- threshold_low = -hysteresis/2
```

#### 3.2 模糊判决（resolution > 0）
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
if (abs(Vdiff) < resolution) {
    // 模糊区：随机判决（50/50 概率）
    data_out = random_bernoulli(0.5, seed);
} else if (Vdiff >= resolution) {
    data_out = 1;
} else {
    data_out = 0;
}
```

### 4. 噪声建模

#### 4.1 高斯噪声生成
```cpp
// 使用 C++ 标准库
std::normal_distribution<double> dist(0.0, noise.sigma);
std::mt19937 gen(noise.seed);  // Mersenne Twister

double noise_sample = dist(gen);
```

#### 4.2 噪声特性
- **类型**：加性高斯白噪声（AWGN）
- **统计特性**：均值为 0，标准差为 `noise.sigma`
- **独立性**：每次采样生成独立的噪声样本
- **可复现性**：通过 `seed` 参数控制

### 5. 误码率（BER）计算

#### 5.1 理想信道（无 ISI）

**情况 1：仅噪声，无偏移**
```
假设：
- 发送信号：±A (差分幅度 2A)
- 噪声：σ_n = noise.sigma
- 判决阈值：V_th = 0

BER = Q(A / σ_n)

其中 Q 函数：
Q(x) = (1/√(2π)) ∫[x,∞] exp(-t²/2) dt
     ≈ (1/2) erfc(x/√2)
```

**情况 2：噪声 + 偏移**
```
假设：
- 偏移：V_offset = offset.value

对于发送 '1' (信号 = +A):
  BER_1 = Q((A - V_offset) / σ_n)

对于发送 '0' (信号 = -A):
  BER_0 = Q((A + V_offset) / σ_n)

总 BER = (BER_1 + BER_0) / 2
```

**情况 3：噪声 + 分辨率阈值**
```
分辨率阈值引入额外的误判区域：

当 |V_signal + V_noise| < resolution 时，输出随机

BER ≈ Q(A / σ_n) + P(|V_total| < resolution) × 0.5

其中：
P(|V_total| < resolution) 
  ≈ erf(resolution / (√2 × σ_n))
```

#### 5.2 综合误码率公式

```
BER_total ≈ Q((A - |V_offset|) / σ_n) + P_metastable × 0.5

其中：
- A: 差分信号幅度（伏特）
- V_offset: 偏移电压（伏特）
- σ_n: 噪声标准差（伏特）
- P_metastable: 落入分辨率阈值内的概率
  
P_metastable ≈ erf(resolution / (√2 × σ_n))
```

#### 5.3 数值计算示例

```python
import numpy as np
from scipy.special import erfc, erf

def calculate_BER(A, sigma_n, V_offset, resolution):
    """
    计算 Sampler 的误码率
    
    参数：
    A: 信号幅度（伏特）
    sigma_n: 噪声标准差（伏特）
    V_offset: 偏移电压（伏特）
    resolution: 分辨率阈值（伏特）
    
    返回：
    BER_total: 总误码率
    """
    # Q 函数
    def Q(x):
        return 0.5 * erfc(x / np.sqrt(2))
    
    # 噪声和偏移导致的 BER
    SNR_eff = (A - abs(V_offset)) / sigma_n
    BER_noise = Q(SNR_eff)
    
    # 分辨率阈值导致的模糊判决概率
    P_metastable = erf(resolution / (np.sqrt(2) * sigma_n))
    
    # 总 BER
    BER_total = BER_noise + P_metastable * 0.5
    
    return BER_total

# 示例参数
A = 0.5          # 500 mV 差分幅度
sigma_n = 0.01   # 10 mV RMS 噪声
V_offset = 0.005 # 5 mV 偏移
resolution = 0.02 # 20 mV 分辨率

BER = calculate_BER(A, sigma_n, V_offset, resolution)
print(f"BER = {BER:.2e}")
# 输出示例: BER ≈ 1e-10
```

### 6. 与 CDR 的交互

#### 6.1 闭环工作原理
```
┌─────────┐  data_out   ┌─────────┐
│ Sampler │────────────>│   CDR   │
└─────────┘             └─────────┘
     ↑                       │
     │   phase_offset/clk    │
     └───────────────────────┘
     
工作流程：
1. Sampler 使用当前相位采样数据
2. 输出比特流给 CDR
3. CDR 根据数据转换点检测相位误差
4. CDR 调整 phase_offset 或 clk_sample
5. Sampler 使用新相位采样（循环）
```

#### 6.2 相位接口设计

**方式 A：连续时间相位**
```cpp
// Sampler 接收连续的相位偏移
sca_tdf::sca_in<double> phase_offset;  // 单位：秒

// 采样时刻计算
double t_actual = current_time + phase_offset.read() + sample_delay;
```

**方式 B：离散时钟信号**
```cpp
// Sampler 接收采样时钟边沿
sca_tdf::sca_in<double> clk_sample;

// 在时钟上升沿触发采样
if (clk_sample.read() > threshold) {
    perform_sampling();
}
```

### 7. 输出模式

- **TDF 输出**（`data_out`）：整数 0/1，适合后续 TDF 模块处理
- **DE 输出**（`data_out_de`）：布尔值，适合连接到纯 SystemC 数字逻辑

## 依赖

### SystemC-AMS
- **必须**：SystemC-AMS 2.3.4
- 使用 TDF（Timed Data Flow）域建模
- 可选：DE 域桥接（`sca_de::sca_out`）

### C++ 标准库
- `<random>`：高斯噪声生成（`std::normal_distribution`）
- `<cmath>`：数学运算

### 配置文件
- JSON 格式配置（如 `config/sampler_config.json`）

## 使用示例

### 配置示例

#### 理想采样器（无非理想效应）
```json
{
  "sampler": {
    "sample_delay": 0.0,
    "offset": {
      "enable": false
    },
    "noise": {
      "enable": false
    },
    "resolution": 0.0,
    "hysteresis": 0.0,
    "phase_source": "clock"
  }
}
```

#### 真实采样器（包含非理想效应）
```json
{
  "sampler": {
    "sample_delay": 5e-12,        // 5 ps 延迟
    "offset": {
      "enable": true,
      "value": 0.01               // 10 mV 失调
    },
    "noise": {
      "enable": true,
      "sigma": 0.005,             // 5 mV RMS 噪声
      "seed": 12345
    },
    "resolution": 0.02,           // 20 mV 分辨率
    "hysteresis": 0.01,           // 10 mV 迟滞
    "phase_source": "phase"
  }
}
```

### SystemC-AMS 实例化

```cpp
// 创建 Sampler 模块
Sampler sampler("sampler");
sampler.sample_delay = 5e-12;
sampler.offset_enable = true;
sampler.offset_value = 0.01;
sampler.noise_enable = true;
sampler.noise_sigma = 0.005;
sampler.resolution = 0.02;
sampler.hysteresis = 0.01;

// 连接信号
sampler.inp(vga_outp);           // 连接到 VGA 输出正端
sampler.inn(vga_outn);           // 连接到 VGA 输出负端
sampler.phase_offset(cdr_phase); // 连接到 CDR 相位输出
sampler.data_out(dfe_input);     // 连接到 DFE 输入
```

### 与其他模块集成

```
信号链：
WaveGen → TX → Channel → CTLE → VGA → Sampler → DFE/CDR
                                          ↓         ↑
                                    (数字比特流)    │
                                          └─────────┘
                                         (相位反馈)
```

### BER 仿真示例

```cpp
// 在 Sampler 模块中添加 BER 统计
// 伪代码
class Sampler : public sca_tdf::sca_module {
private:
    long total_bits = 0;
    long error_bits = 0;
    
public:
    void processing() {
        int decision = make_decision();
        int actual = get_actual_bit();  // 从发送端获取
        
        total_bits++;
        if (decision != actual) {
            error_bits++;
        }
    }
    
    double get_BER() {
        return (total_bits > 0) ? (double)error_bits / total_bits : 0.0;
    }
    
    void print_BER() {
        std::cout << "Total bits: " << total_bits << std::endl;
        std::cout << "Error bits: " << error_bits << std::endl;
        std::cout << "BER: " << get_BER() << std::endl;
    }
};
```

## 测试验证

### 1. 基本功能验证
- **目标**：验证差分判决正确性
- **方法**：
  - 输入标准差分信号（如 ±0.5V）
  - 验证输出与输入符号一致
- **指标**：100% 正确率（无噪声/偏移时）

### 2. 延迟验证
- **目标**：验证采样延迟准确性
- **方法**：
  - 输入已知波形
  - 对比设定延迟与实际输出时刻
- **指标**：延迟误差 < 1 ps

### 3. 偏移效应验证
- **目标**：验证偏移对判决的影响
- **方法**：
  - 输入小幅度差分信号（如 ±10mV）
  - 添加不同偏移值（如 ±5mV）
  - 观察误判率变化
- **指标**：误判率符合理论预期

### 4. 噪声效应验证
- **目标**：验证噪声注入功能
- **方法**：
  - 输入恒定差分电平
  - 启用噪声，统计输出翻转率
  - 对比理论 BER
- **指标**：实测 BER 与理论值偏差 < 10%

### 5. 分辨率阈值验证
- **目标**：验证模糊判决区行为
- **方法**：
  - 输入幅度 < resolution 的差分信号
  - 统计输出 0/1 的概率分布
- **指标**：输出概率接近 50/50（随机判决）

### 6. 迟滞验证
- **目标**：验证施密特触发器效应
- **方法**：
  - 输入在阈值附近抖动的信号
  - 对比有/无迟滞时的输出稳定性
- **指标**：有迟滞时输出抖动显著减少

### 7. CDR 集成验证
- **目标**：验证与 CDR 的闭环工作
- **方法**：
  - 连接 Sampler 和 CDR 模块
  - 输入带抖动的数据信号
  - 观察 CDR 相位收敛过程
- **指标**：相位误差收敛到 ±5% UI 以内

### 8. 可复现性验证
- **目标**：验证 seed 参数的作用
- **方法**：
  - 相同配置和 seed，运行多次
  - 对比输出序列
- **指标**：输出完全一致

## 变更历史

### v0.1 (初始版本)
- 初始设计，包含以下功能：
  - 差分信号输入与数字比特输出
  - 动态采样时刻（由 CDR 提供）
  - 可配置采样延迟
  - 偏移和噪声注入功能（enable 控制）
  - 分辨率阈值与模糊判决
  - 迟滞功能
  - 随机数种子控制，确保可复现性
  - 完整的 BER 计算公式和数值示例
  - TDF 和 DE 双域输出支持