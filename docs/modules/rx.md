# RX 接收端模块技术文档

**级别**：AMS 顶层模块  
**当前版本**：v1.0 (2026-01-27)  
**状态**：生产就绪

---

## 1. 概述

SerDes接收端（RX）是高速串行链路的核心组成部分，负责将经过信道衰减和失真的模拟差分信号恢复为原始数字比特流。RX通过多级均衡、自动增益控制、判决反馈和时钟数据恢复等技术，实现对信道损伤的全面补偿。

### 1.1 设计原理

RX接收端的核心设计思想是采用分级级联架构，每个子模块专注于特定的信号处理任务：

```
信道输出 → CTLE → VGA → DFE Summer → Sampler → 数字输出
                                ↑           ↓
                            历史判决    采样数据
                                ↑           ↓
                            data_out  ←  CDR  ← phase_offset
```

**信号流处理逻辑**：

1. **CTLE（连续时间线性均衡器）**：频域均衡，通过零极点传递函数提升高频增益，补偿信道的频率相关损耗
2. **VGA（可变增益放大器）**：幅度调整，配合AGC算法动态控制信号摆幅到最优范围
3. **DFE Summer（判决反馈均衡器）**：时域均衡，利用已判决符号的反馈抵消后游标码间干扰（ISI）
4. **Sampler（采样器）**：阈值判决，在CDR指定的最佳采样时刻进行二值化判决
5. **CDR（时钟数据恢复）**：相位跟踪，从数据跃变中提取时钟信息，动态调整采样相位

**分级均衡策略**：

- **CTLE处理线性ISI**：通过频域连续时间均衡，补偿信道的高频衰减，但会放大高频噪声
- **DFE处理非线性ISI**：通过时域判决反馈均衡，消除后游标ISI，不放大噪声但存在误差传播风险
- **增益分配平衡**：CTLE和VGA的总增益需使Sampler输入信号摆幅达到最优判决范围（通常200-600mV）

### 1.2 核心特性

- **五级级联架构**：CTLE → VGA → DFE → Sampler → CDR，覆盖接收端完整信号链
- **差分信号路径**：全程差分传输，CTLE/VGA/Sampler均采用差分输入输出，共模抑制比>40dB
- **闭环时钟恢复**：CDR提供采样相位反馈，Sampler接收相位调整信号，形成相位锁定环路
- **多域协同**：TDF域模拟模块（CTLE/VGA/DFE/Sampler/CDR）与DE域自适应模块（Adaption）协同工作
- **可配置均衡深度**：
  - CTLE：多零极点传递函数，频域均衡
  - VGA：可变增益放大，动态范围控制
  - DFE：1-8抽头可配置，后游标ISI抵消
- **自适应优化**：支持LMS/Sign-LMS/NLMS等自适应算法动态优化均衡参数
- **非理想效应建模**：集成偏移、噪声、PSRR、CMFB、CMRR、饱和等实际器件特性

### 1.3 子模块概览

| 模块 | 类名 | 功能 | 关键参数 | 独立文档 |
|------|------|------|---------|---------|
| **CTLE** | `RxCtleTdf` | 连续时间线性均衡器 | zeros, poles, dc_gain | ctle.md |
| **VGA** | `RxVgaTdf` | 可变增益放大器 | dc_gain(可调), zeros, poles | vga.md |
| **DFE Summer** | `RxDfeSummerTdf` | 判决反馈均衡求和器 | tap_coeffs, vtap, map_mode | dfesummer.md |
| **Sampler** | `RxSamplerTdf` | 采样器/判决器 | resolution, hysteresis, phase_source | sampler.md |
| **CDR** | `RxCdrTdf` | 时钟数据恢复 | kp, ki, resolution, range | cdr.md |

### 1.4 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-01-27 | 初始版本，整合五个子模块的顶层文档 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

#### 2.1.1 顶层输入输出端口

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `in_p` | 输入 | double | 来自信道的差分输入正端 |
| `in_n` | 输入 | double | 来自信道的差分输入负端 |
| `vdd` | 输入 | double | 电源电压（PSRR建模用） |
| `data_out` | 输出 | int | 恢复的数字比特流（0/1） |

> **重要**：即使不启用PSRR功能，`vdd`端口也必须连接（SystemC-AMS要求所有端口均需连接）。

#### 2.1.2 内部模块级联关系

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RX 接收端顶层模块                                    │
│                                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐                  │
│  │  CTLE   │    │   VGA   │    │ DFE Summer  │    │ Sampler  │                  │
│  │         │    │         │    │             │    │          │                  │
│  │ in_p ←──┼────┼─ out_p  │    │             │    │          │                  │
│  │ in_n ←──┼────┼─ out_n  │    │             │    │          │                  │
│  │         │    │         │    │             │    │          │                  │
│  │ out_p ──┼────┼→ in_p   │    │             │    │          │     data_out     │
│  │ out_n ──┼────┼→ in_n   │    │ in_p ←──────┼────┼─ out_p   │    ──────────→   │
│  │         │    │         │    │ in_n ←──────┼────┼─ out_n   │                  │
│  │ vdd ←───┼────┼─ vdd    │    │             │    │          │                  │
│  └─────────┘    │ out_p ──┼────┼→ in_p       │    │ inp ←────┼── out_p         │
│       ↑         │ out_n ──┼────┼→ in_n       │    │ inn ←────┼── out_n         │
│       │         │         │    │             │    │          │                  │
│      VDD        │ vdd ←───┼────┼─────────────┼────┼──────────┼── vdd           │
│                 └─────────┘    │             │    │          │                  │
│                                │ data_in ←───┼────┼──────────┼── data_out      │
│                                │ (历史判决)  │    │          │      │           │
│                                │             │    │          │      │           │
│                                └─────────────┘    │ phase ←──┼──────┼───┐       │
│                                                   │ _offset  │      │   │       │
│                                                   └──────────┘      │   │       │
│                                                                     │   │       │
│                                                        ┌────────────┘   │       │
│                                                        ↓                │       │
│                                                   ┌─────────┐           │       │
│                                                   │   CDR   │           │       │
│                                                   │         │           │       │
│                                                   │ in ←────┼───────────┘       │
│                                                   │         │                   │
│                                                   │ phase ──┼→ phase_offset     │
│                                                   │ _out    │                   │
│                                                   └─────────┘                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**关键信号流**：

- **前向路径**：`in_p/in_n` → CTLE → VGA → DFE Summer → Sampler → `data_out`
- **DFE反馈路径**：Sampler.data_out（历史判决）→ DFE.data_in（抽头输入）
- **CDR闭环路径**：Sampler.data_out → CDR.in → CDR.phase_out → Sampler.phase_offset

### 2.2 参数配置（RxParams结构）

#### 2.2.1 总体参数结构

```cpp
struct RxParams {
    RxCtleParams ctle;          // CTLE参数
    RxVgaParams vga;            // VGA参数
    RxSamplerParams sampler;    // Sampler参数
    RxDfeParams dfe;            // DFE参数
};

// CDR参数独立定义
struct CdrParams {
    CdrPiParams pi;             // PI控制器参数
    CdrPaiParams pai;           // 相位插值器参数
    double ui;                  // 单位间隔 (s)
    bool debug_enable;          // 调试输出使能
};
```

#### 2.2.2 各子模块参数汇总

| 子模块 | 关键参数 | 默认配置 | 调整目的 |
|--------|---------|---------|---------|
| CTLE | `zeros=[2e9]`, `poles=[30e9]`, `dc_gain=1.5` | 单零点单极点 | 高频提升，带宽限制 |
| VGA | `zeros=[1e9]`, `poles=[20e9]`, `dc_gain=2.0` | 可变增益 | AGC动态调整 |
| DFE | `taps=[-0.05,-0.02,0.01]`, `mu=1e-4` | 3抽头 | 后游标ISI抵消 |
| Sampler | `resolution=0.02`, `hysteresis=0.02` | 模糊判决 | 亚稳态建模 |
| CDR | `kp=0.01`, `ki=1e-4`, `resolution=1e-12` | PI控制器 | 相位跟踪 |

#### 2.2.3 配置示例（JSON格式）

```json
{
  "rx": {
    "ctle": {
      "zeros": [2e9],
      "poles": [30e9],
      "dc_gain": 1.5,
      "vcm_out": 0.6,
      "psrr": {"enable": false},
      "cmfb": {"enable": true, "bandwidth": 1e6},
      "cmrr": {"enable": false}
    },
    "vga": {
      "zeros": [1e9],
      "poles": [20e9],
      "dc_gain": 2.0,
      "vcm_out": 0.6
    },
    "dfe": {
      "taps": [-0.05, -0.02, 0.01],
      "update": "sign-lms",
      "mu": 1e-4
    },
    "sampler": {
      "threshold": 0.0,
      "resolution": 0.02,
      "hysteresis": 0.02,
      "phase_source": "phase"
    }
  },
  "cdr": {
    "pi": {"kp": 0.01, "ki": 1e-4, "edge_threshold": 0.5},
    "pai": {"resolution": 1e-12, "range": 5e-11},
    "ui": 1e-10
  }
}
```

---

## 3. 核心实现机制

### 3.1 信号处理流程

RX接收端的完整信号处理流程包含7个关键步骤：

```
步骤1: 信道输出读取 → 差分信号 in_p/in_n
步骤2: CTLE均衡    → 频域补偿，高频提升
步骤3: VGA放大     → 幅度调整，动态增益控制
步骤4: DFE反馈抵消 → 减去历史判决反馈的ISI分量
步骤5: Sampler判决 → 在最优相位时刻进行阈值判决
步骤6: 更新DFE历史 → 将当前判决推入历史缓冲区
步骤7: CDR相位更新 → 检测边沿，调整下一个采样相位
```

**时序约束**：

- 所有TDF模块必须运行在相同采样率（通常为10-20倍符号速率）
- DFE反馈延迟 ≥ 1 UI（避免代数环）
- CDR相位更新延迟通常为1-2 UI

### 3.2 CTLE-VGA级联设计

#### 3.2.1 增益分配策略

总增益需求计算：

```
G_total = V_sampler_min / V_channel_out
```

其中：
- `V_sampler_min`：Sampler最小输入摆幅（通常100-200mV）
- `V_channel_out`：信道输出摆幅（取决于信道损耗）

**推荐增益分配**：

| 信道损耗 | CTLE增益 | VGA增益 | 总增益 |
|---------|---------|---------|-------|
| 轻度（5dB） | 1.2 | 1.5 | 1.8 |
| 中度（10dB） | 1.5 | 2.0 | 3.0 |
| 重度（15dB） | 2.0 | 3.0 | 6.0 |
| 极重（20dB） | 2.5 | 4.0 | 10.0 |

#### 3.2.2 共模电压管理

- CTLE输出共模 = VGA输入共模 = 0.6V（可配置）
- 两级均采用独立的CMFB环路稳定共模电压
- 避免级间共模失配导致的非线性失真

#### 3.2.3 带宽匹配

- CTLE极点频率（30GHz）>> VGA极点频率（20GHz）
- VGA作为二级滤波，进一步抑制高频噪声
- 系统总带宽由最低极点频率决定

### 3.3 DFE反馈环路设计

#### 3.3.1 历史判决维护

```cpp
// 伪代码示例
std::vector<int> history_bits(N_taps);  // N_taps = DFE抽头数

void update_history(int new_bit) {
    // 移位操作：新判决进入位置0，旧判决依次后移
    for (int i = N_taps - 1; i > 0; i--) {
        history_bits[i] = history_bits[i - 1];
    }
    history_bits[0] = new_bit;
}

// DFE Summer在每个UI读取history_bits计算反馈电压
double compute_feedback() {
    double feedback = 0.0;
    for (int i = 0; i < N_taps; i++) {
        // 将0/1映射为-1/+1
        int symbol = (history_bits[i] == 1) ? +1 : -1;
        feedback += taps[i] * symbol * vtap;
    }
    return feedback;
}
```

#### 3.3.2 抽头系数计算

- **方法1**：信道脉冲响应测量 + 后游标采样 + 比例缩放
- **方法2**：LMS自适应算法在线优化
- **归一化约束**：`Σ|tap_coeffs[k]| < 0.5`（避免过补偿）

#### 3.3.3 稳定性考虑

- DFE抽头系数过大会导致误差传播
- 需要配合CDR相位对齐确保判决时刻最优
- 自适应算法需要收敛性验证

### 3.4 Sampler-CDR闭环机制

#### 3.4.1 相位检测与调整

1. **Sampler**：根据`phase_offset`信号动态调整采样时刻
2. **CDR**：检测数据边沿与当前相位关系，计算相位误差
3. **PI控制器**：根据误差更新相位累积量
4. **相位量化**：按PAI分辨率量化后输出给Sampler

#### 3.4.2 Bang-Bang相位检测算法

```cpp
// CDR中的相位检测逻辑
double bit_diff = current_bit - prev_bit;
double phase_error = 0.0;

if (std::abs(bit_diff) > edge_threshold) {
    if (bit_diff > 0)
        phase_error = +1.0;   // 上升沿：时钟晚，需提前
    else
        phase_error = -1.0;   // 下降沿：时钟早，需延迟
}

// PI控制器更新
integral += ki * phase_error;
double prop_term = kp * phase_error;
double pi_output = prop_term + integral;
phase = pi_output * ui;  // 缩放到秒
```

#### 3.4.3 锁定过程

- **初始阶段**：相位误差较大，PI控制器快速调整
- **收敛阶段**：误差逐渐减小，相位抖动收敛到Bang-Bang PD固有水平（1-5ps RMS）
- **稳态阶段**：相位锁定，跟踪频偏和低频抖动

#### 3.4.4 闭环带宽

- 典型值：1-10MHz（远低于数据速率）
- 作用：跟踪低频抖动，抑制高频噪声
- 调整方法：修改CDR的Kp/Ki参数

### 3.5 自适应优化机制（可选）

#### 3.5.1 自适应目标

| 模块 | 自适应参数 | 算法 | 更新周期 |
|------|-----------|------|---------|
| VGA | dc_gain | AGC | 100-1000 UI |
| DFE | tap_coeffs | LMS/Sign-LMS/NLMS | 每UI |
| CDR | kp, ki | 可选自适应 | 慢速 |
| Sampler | threshold | 阈值跟踪 | 1000 UI |

#### 3.5.2 LMS算法实现

```cpp
// DFE抽头自适应更新
for (int i = 0; i < N_taps; i++) {
    // error = 判决前信号 - 判决后信号（软判决误差）
    // 或使用BER反馈
    taps[i] += mu * error * history_bits[i];
    
    // 抽头限幅
    taps[i] = std::clamp(taps[i], tap_min, tap_max);
}
```

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

RX测试平台需要闭环集成设计：

- **TX侧**：信号源（PRBS）+ 信道模型
- **RX侧**：CTLE + VGA + DFE + Sampler + CDR级联
- **性能评估**：BER统计 + 眼图采集 + 相位误差测量

与子模块测试平台的区别：
- 子模块测试（如ctle_tran_tb）：单模块开环测试
- RX顶层测试：全链路闭环测试，包含反馈路径

### 4.2 测试场景定义

| 场景 | 命令行参数 | 测试目标 | 输出文件 |
|------|----------|---------|----------|
| BASIC_PRBS | `prbs` / `0` | 基本链路建立和锁定 | rx_tran_prbs.csv |
| CHANNEL_SWEEP | `ch_sweep` / `1` | 不同信道损耗下的BER | rx_ber_sweep.csv |
| ADAPTION_TEST | `adapt` / `2` | 自适应算法收敛性 | rx_adaption.csv |
| JITTER_TOLERANCE | `jtol` / `3` | 系统级JTOL测试 | rx_jtol.csv |
| EYE_SCAN | `eye` / `4` | 2D眼图扫描 | rx_eye_2d.csv |

### 4.3 场景配置详解

#### BASIC_PRBS - 基本链路测试

- **信号源**：PRBS-31, 10Gbps
- **信道**：中等损耗（10dB @ Nyquist）
- **RX配置**：默认参数
- **仿真时间**：≥100,000 UI
- **验证点**：
  - CDR锁定时间 < 5000 UI
  - 锁定后BER < 1e-12
  - 相位稳定性 < 5ps RMS

#### CHANNEL_SWEEP - 信道损耗扫描

- **信道变化**：5dB, 10dB, 15dB, 20dB @ Nyquist
- **RX配置**：固定参数 或 自适应开启
- **验证点**：绘制BER vs 损耗曲线，确定链路裕量

#### ADAPTION_TEST - 自适应收敛测试

- **初始状态**：DFE抽头系数为零
- **自适应算法**：LMS, 步长μ=0.001
- **监控信号**：抽头系数时域演化 + BER收敛曲线
- **验证点**：
  - 收敛时间 < 50,000 UI
  - 稳态BER达到最优值

#### JITTER_TOLERANCE - 系统级抖动容限

- 与CDR单独测试的区别：包含CTLE/VGA/DFE的影响
- 抖动注入位置：信道输出端
- 测试方法：扫描抖动频率（1kHz-100MHz），记录BER

#### EYE_SCAN - 二维眼图扫描

- **X轴**：相位扫描（-0.5UI ~ +0.5UI）
- **Y轴**：阈值扫描（-Vswing ~ +Vswing）
- **每点统计**：≥10,000 UI的BER测量
- **输出**：眼图热图，标注眼高/眼宽

### 4.4 信号连接拓扑

```
┌──────────┐   ┌────────┐   ┌──────┐   ┌─────┐   ┌─────────┐   ┌────────┐   ┌─────────┐
│ PRBS Gen │→→→│ Channel│→→→│ CTLE │→→→│ VGA │→→→│DFE Summ │→→→│Sampler │→→→│ BER Mon │
└──────────┘   └────────┘   └──────┘   └─────┘   └─────────┘   └────────┘   └─────────┘
                                                       ↑            ↓    ↓
                                                       │            │    └──→┌─────┐
                                                       │            │        │ CDR │
                                                       └────────────┴────────└─────┘
                                                      DFE反馈     CDR相位反馈
```

### 4.5 辅助模块说明

| 模块 | 功能 | 配置参数 |
|------|------|---------|
| **Channel Model** | S参数导入或解析式损耗模型 | touchstone, attenuation_db |
| **PRBS Generator** | 支持PRBS-7/15/31，可配置抖动注入 | type, jitter |
| **BER Monitor** | 实时统计误码率，支持眼图采集 | measure_length, eye_params |
| **Adaption Controller** | DE域自适应算法控制器 | agc, dfe, threshold |
| **Performance Analyzer** | 眼高/眼宽/Q-factor分析 | ui_bins, amp_bins |

---

## 5. 仿真结果分析

### 5.1 统计指标说明

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| **BER** | 误码数 / 总比特数 | 系统可靠性核心指标 |
| **眼高** | min(信号高电平) - max(信号低电平) | 噪声裕量 |
| **眼宽** | 最优采样相位范围（UI） | 时序裕量 |
| **Q-factor** | √2 × erfc⁻¹(2×BER) | 信噪比等效指标 |
| **锁定时间** | CDR相位误差 < 5ps的时刻 | 链路建立速度 |

### 5.2 典型测试结果解读

#### BASIC_PRBS测试结果示例

**配置**：10Gbps, 中等信道（10dB @ Nyquist）

**期望结果**：
```
=== RX Performance Summary ===
CDR Lock Time:        2345 UI (234.5 ns)
BER (after lock):     0.0 (no errors in 1e7 bits)
Eye Height:           450 mV (对应Q=7.2, BER=1e-12理论值)
Eye Width:            0.65 UI (65 ps)
Phase Jitter (RMS):   2.1 ps
CTLE Output Swing:    300 mV (增益1.5×输入200mV)
VGA Output Swing:     600 mV (增益2.0×CTLE输出)
DFE Tap Coeffs:       [-0.08, -0.03, 0.01] (自适应收敛值)
```

**波形特征**：
- CTLE输出：高频提升明显，边沿变陡
- VGA输出：幅度放大，保持差分特性
- DFE输出：ISI明显减小，眼图开度增加
- Sampler输出：清晰的数字跳变，极少误码

#### CHANNEL_SWEEP结果解读

**BER vs 信道损耗曲线**：

| 损耗(dB@Nyq) | BER(无DFE) | BER(有DFE) | 改善(dB) |
|-------------|-----------|-----------|---------|
| 5 | 1e-15 | 1e-15 | 0（余量充足）|
| 10 | 1e-9 | 1e-13 | 4dB |
| 15 | 1e-5 | 1e-11 | 6dB |
| 20 | 1e-3 | 1e-9 | 6dB |
| 25 | >1e-1 | 1e-6 | >5dB |

**分析要点**：
- DFE在高损耗信道（>15dB）中效果显著
- 20dB损耗接近系统极限，需要配合更强的CTLE/VGA
- 25dB损耗可能需要启用更多DFE抽头（5-7抽头）

### 5.3 波形数据文件格式

**rx_tran_prbs.csv**：
```csv
时间(s),CTLE_out_diff(V),VGA_out_diff(V),DFE_out_diff(V),Sampler_out,CDR_phase(ps),BER
0.0e0,0.000,0.000,0.000,0,0.0,N/A
1.0e-10,0.150,0.300,0.280,1,2.5,N/A
2.0e-10,-0.145,-0.290,-0.275,0,2.3,N/A
...
1.0e-6,0.148,0.296,0.283,1,1.8,1.2e-13
```

---

## 6. 运行指南

### 6.1 环境配置

运行测试前需要配置环境变量：

```bash
source scripts/setup_env.sh
```

确保以下依赖已正确安装：
- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14兼容编译器

### 6.2 构建与运行

```bash
cd build
cmake ..
make rx_tran_tb
cd tb
./rx_tran_tb [scenario]
```

场景参数：
- `prbs` 或 `0` - 基本PRBS测试（默认）
- `ch_sweep` 或 `1` - 信道损耗扫描
- `adapt` 或 `2` - 自适应收敛测试
- `jtol` 或 `3` - 抖动容限测试
- `eye` 或 `4` - 眼图扫描

### 6.3 参数调优流程

**步骤1：信道表征**
```bash
python scripts/analyze_channel.py channel.s4p
# 输出：损耗@Nyquist, 群延迟, 建议CTLE零极点
```

**步骤2：CTLE/VGA基础配置**
- 根据信道分析结果设置CTLE零极点
- VGA增益初步设为1.5-2.0

**步骤3：DFE初始化**
- 方法A：抽头系数设为零，启用自适应
- 方法B：根据信道脉冲响应预设初值

**步骤4：CDR参数选择**
- 根据cdr.md第8.4节公式估算Kp/Ki
- 目标带宽：数据速率/1000 ~ 数据速率/10000

**步骤5：运行仿真验证**
```bash
./rx_tran_tb prbs
# 检查BER, 眼图, 锁定时间
```

**步骤6：迭代优化**
- 若BER不达标：增加DFE抽头数 或 优化CTLE参数
- 若CDR不锁定：调整Kp/Ki 或 增大PAI range
- 若眼图闭合：检查饱和限制 或 噪声配置

### 6.4 结果查看

测试完成后，控制台输出统计结果，波形数据保存到CSV文件。使用Python进行可视化：

```bash
# 波形可视化
python scripts/plot_rx_waveforms.py rx_tran_prbs.csv

# 眼图绘制
python scripts/plot_eye_diagram.py rx_eye_2d.csv

# BER曲线
python scripts/plot_ber_sweep.py rx_ber_sweep.csv
```

---

## 7. 技术要点

### 7.1 级联增益分配原则

**总增益需求**：
```
G_total = V_sampler_min / V_channel_out
```

**分配策略**：
- **CTLE**：提供1.5-2.0倍增益 + 频域整形
- **VGA**：提供1.5-5.0倍可调增益 + 二次滤波
- **DFE**：不改变平均增益，仅抵消ISI

**饱和管理**：
- 每级输出限幅范围应匹配下级输入范围
- 避免中间级提前饱和导致非线性失真
- 软饱和（tanh）优于硬饱和（clamp）

### 7.2 DFE反馈延迟的代数环问题

**问题描述**：
若DFE的反馈延迟为0 UI，会形成代数环：
```
当前输出 → Sampler判决 → DFE反馈 → 当前输出（循环依赖）
```

**解决方案**：
- DFE的`data_in`端口读取**上一个UI的判决**
- 在RX顶层模块维护判决历史缓冲区
- 确保信号流的因果性

**实现示例**：
```cpp
// 在RX顶层processing()函数
int current_bit = sampler.data_out.read();
dfe.data_in.write(history_buffer);  // 使用历史判决
history_buffer.push_front(current_bit);
history_buffer.pop_back();
```

### 7.3 Sampler-CDR时序协调

**相位更新时序**：
- CDR在第n个UI检测相位误差
- PI控制器计算新的相位偏移
- Sampler在第n+1个UI应用新相位

**延迟影响**：
- 1 UI延迟不影响稳定性（环路带宽远低于数据速率）
- 但会增加锁定时间（约10-20%）

### 7.4 共模电压级联管理

**各级共模要求**：
- CTLE输出：0.6V（典型）
- VGA输出：0.6V（与CTLE匹配）
- DFE输出：0.6V 或 0.0V（取决于Sampler要求）
- Sampler输入：需在差分对管的共模输入范围内

**失配处理**：
- 若级间共模不匹配，可插入AC耦合电容
- AC耦合会引入低频滚降，需权衡

### 7.5 自适应算法稳定性

**LMS算法收敛条件**：
```
0 < μ < 2 / λ_max
```

其中：
- μ：步长
- λ_max：输入信号自相关矩阵最大特征值

**实际建议**：
- 保守取值：μ = 0.001 ~ 0.01
- 监控抽头系数是否振荡或发散
- 必要时采用归一化LMS（NLMS）提高稳定性

### 7.6 时间步长与采样率设置

**一致性要求**：
所有TDF模块必须设置相同的采样率：

```cpp
// 全局配置
double Fs = 100e9;  // 100 GHz（符号速率10Gbps × 10倍过采样）
double Ts = 1.0 / Fs;

// 各模块set_attributes()
ctle.set_timestep(Ts);
vga.set_timestep(Ts);
dfe.set_timestep(Ts);
sampler.set_timestep(Ts);
cdr.set_timestep(Ts);
```

**过采样考虑**：
- 最小采样率 = 2 × 最高频率分量（Nyquist准则）
- 推荐采样率 = 5-10 × 符号速率（保证波形保真度）

---

## 8. 参考信息

### 8.1 相关文件

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| CTLE头文件 | `/include/ams/rx_ctle.h` | RxCtleTdf类声明 |
| CTLE实现 | `/src/ams/rx_ctle.cpp` | RxCtleTdf类实现 |
| VGA头文件 | `/include/ams/rx_vga.h` | RxVgaTdf类声明 |
| VGA实现 | `/src/ams/rx_vga.cpp` | RxVgaTdf类实现 |
| DFE头文件 | `/include/ams/rx_dfe.h` | RxDfeTdf类声明 |
| DFE实现 | `/src/ams/rx_dfe.cpp` | RxDfeTdf类实现 |
| Sampler头文件 | `/include/ams/rx_sampler.h` | RxSamplerTdf类声明 |
| Sampler实现 | `/src/ams/rx_sampler.cpp` | RxSamplerTdf类实现 |
| CDR头文件 | `/include/ams/rx_cdr.h` | RxCdrTdf类声明 |
| CDR实现 | `/src/ams/rx_cdr.cpp` | RxCdrTdf类实现 |
| 参数定义 | `/include/common/parameters.h` | RxParams/CdrParams结构体 |
| CTLE文档 | `/docs/modules/ctle.md` | CTLE详细技术文档 |
| VGA文档 | `/docs/modules/vga.md` | VGA详细技术文档 |
| DFE文档 | `/docs/modules/dfesummer.md` | DFE Summer详细技术文档 |
| Sampler文档 | `/docs/modules/sampler.md` | Sampler详细技术文档 |
| CDR文档 | `/docs/modules/cdr.md` | CDR详细技术文档 |
| 自适应文档 | `/docs/modules/adaption.md` | 自适应控制器文档 |

### 8.2 依赖项

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14标准
- GoogleTest 1.12.1（单元测试）

### 8.3 性能指标总结

| 指标 | 典型值 | 说明 |
|------|-------|------|
| 最大数据速率 | 56 Gbps | 取决于信道和工艺 |
| BER目标 | < 1e-12 | 配合FEC可达1e-15 |
| 锁定时间 | 1-5 μs | CDR收敛时间 |
| 相位抖动 | < 5 ps RMS | 锁定后CDR抖动 |
| CTLE增益范围 | 1.0-3.0 | 可配置 |
| VGA增益范围 | 1.0-10.0 | 配合AGC动态调整 |
| DFE抽头数 | 1-8 | 典型3-5抽头 |
| 眼高目标 | > 200 mV | Sampler输入端 |
| 眼宽目标 | > 0.5 UI | 时序裕量要求 |

---

**文档版本**：v1.0  
**最后更新**：2026-01-27  
**作者**：Yizhe Liu
