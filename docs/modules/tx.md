# TX 发送端模块技术文档

🌐 **Languages**: [中文](tx.md) | [English](../en/modules/tx.md)

**级别**：AMS 顶层模块  
**当前版本**：v1.0 (2026-01-27)  
**状态**：生产就绪

---

## 1. 概述

SerDes发送端（TX）是高速串行链路的起始模块，负责将数字比特流转换为具有预均衡的高摆幅模拟差分信号，通过传输线驱动到信道。TX通过前馈均衡（FFE）预补偿信道损耗，通过驱动器（Driver）提供足够的驱动能力和阻抗匹配。

### 1.1 设计原理

TX发送端的核心设计思想是采用级联架构，在发送端提前补偿信道引入的码间干扰（ISI），降低接收端均衡器的负担：

```
数字输入 → WaveGen → FFE → Mux → Driver → 差分输出 → Channel
          (数字→模拟) (FIR均衡) (通道选择) (驱动&匹配)
```

**信号流处理逻辑**：

1. **WaveGen（波形生成器）**：将数字比特流（0/1）转换为模拟NRZ波形（如±1V），支持PRBS码型和抖动注入
2. **FFE（前馈均衡器）**：通过FIR滤波器对信号进行预失真处理，实现预加重或去加重，补偿信道高频衰减
3. **Mux（多路复用器）**：通道选择和时分复用功能，支持多通道系统的灵活配置
4. **Driver（驱动器）**：最终输出缓冲级，提供足够的驱动能力、阻抗匹配和摆幅控制

**预均衡策略**：

- **预加重（Pre-emphasis）**：在跳变边沿注入额外能量，增强高频分量
- **去加重（De-emphasis）**：衰减非跳变符号的幅度，相对提升边沿能量比例
- **混合模式**：同时使用前置抽头和后置抽头，平衡前后游标ISI补偿

### 1.2 核心特性

- **四级级联架构**：WaveGen → FFE → Mux → Driver，覆盖发送端完整信号链
- **预均衡能力**：FFE提供3-7抽头FIR滤波器，支持预加重和去加重配置
- **差分输出**：Driver采用完整差分架构，输出阻抗可配置（典型50Ω）
- **可配置摆幅**：Driver输出摆幅可调（典型800-1200mV峰峰值）
- **带宽限制建模**：Driver多极点传递函数模拟寄生效应和封装影响
- **非线性效应**：支持软饱和（tanh）和硬饱和（clamp）两种饱和模式
- **PSRR建模**：电源纹波通过可配置传递函数耦合到输出
- **压摆率限制**：可选的输出边沿速率约束，模拟真实器件特性

### 1.3 子模块概览

| 模块 | 类名 | 功能 | 关键参数 | 独立文档 |
|------|------|------|---------|---------|
| **WaveGen** | `WaveGenTdf` | 数字比特流生成器 | pattern, jitter | waveGen.md |
| **FFE** | `TxFfeTdf` | 前馈均衡器 | taps | ffe.md |
| **Mux** | `TxMuxTdf` | 多路复用器 | lane_sel | mux.md |
| **Driver** | `TxDriverTdf` | 输出驱动器 | dc_gain, poles, vswing | driver.md |

### 1.4 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-01-27 | 初始版本，整合四个子模块的顶层文档 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

#### 2.1.1 顶层输入输出端口

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `data_in` | 输入 | int | 数字比特流输入（来自编码器） |
| `out_p` | 输出 | double | 差分输出正端（驱动信道） |
| `out_n` | 输出 | double | 差分输出负端 |
| `vdd` | 输入 | double | 电源电压（PSRR建模用） |

> **重要**：即使不启用PSRR功能，`vdd`端口也必须连接（SystemC-AMS要求所有端口均需连接）。

#### 2.1.2 内部模块级联关系

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TX 发送端顶层模块                                    │
│                                                                                  │
│  data_in                                                                         │
│     │                                                                            │
│     ↓                                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│  │ WaveGen │    │   FFE   │    │   Mux   │    │ Driver  │                       │
│  │         │    │         │    │         │    │         │                       │
│  │ data ←──┼────┼─        │    │         │    │         │                       │
│  │ _in     │    │         │    │         │    │         │     out_p             │
│  │         │    │ in ←────┼────┼─ out    │    │         │    ─────────→         │
│  │ out ────┼────┼→ in     │    │         │    │ in_p ←──┼── out_p              │
│  │         │    │ out ────┼────┼→ in     │    │ in_n ←──┼── out_n              │
│  │         │    │         │    │ out ────┼────┼→        │     out_n             │
│  │         │    │         │    │         │    │         │    ─────────→         │
│  └─────────┘    └─────────┘    │ lane ←──┼────┼── sel   │                       │
│                                │ _sel    │    │         │                       │
│                                └─────────┘    │ vdd ←───┼── vdd                 │
│                                               │         │                       │
│                                               │ out_p ──┼→ out_p               │
│                                               │ out_n ──┼→ out_n               │
│                                               └─────────┘                       │
│                                                    ↑                            │
│                                                   VDD                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**关键信号流**：

- **主信号路径**：`data_in` → WaveGen.out → FFE.in → FFE.out → Mux.in → Mux.out → Driver.in → `out_p/out_n`
- **控制信号**：lane_sel（通道选择），vdd（电源）

### 2.2 参数配置（TxParams结构）

#### 2.2.1 总体参数结构

```cpp
struct TxParams {
    TxFfeParams ffe;            // FFE参数
    int mux_lane;               // Mux通道选择
    TxDriverParams driver;      // Driver参数
    
    TxParams() : mux_lane(0) {}
};

// WaveGen参数独立定义
struct WaveGenParams {
    PRBSType type;              // PRBS类型
    std::string poly;           // 多项式表达式
    std::string init;           // 初始状态
    double single_pulse;        // 单脉冲宽度
    JitterParams jitter;        // 抖动参数
    ModulationParams modulation;// 调制参数
};
```

#### 2.2.2 各子模块参数汇总

| 子模块 | 关键参数 | 默认配置 | 调整目的 |
|--------|---------|---------|---------|
| WaveGen | `type=PRBS31`, `jitter.RJ_sigma=0` | PRBS序列 | 数据源生成 |
| FFE | `taps=[0.2, 0.6, 0.2]` | 3抽头对称 | 预补偿信道ISI |
| Mux | `mux_lane=0` | Bypass模式 | 单通道系统 |
| Driver | `dc_gain=1.0`, `poles=[50e9]`, `vswing=0.8` | 标准配置 | 驱动&匹配 |

#### 2.2.3 配置示例（JSON格式）

```json
{
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF",
    "single_pulse": 0.0,
    "jitter": {
      "RJ_sigma": 0.0,
      "SJ_freq": [],
      "SJ_pp": []
    }
  },
  "tx": {
    "ffe": {
      "taps": [0.0, 1.0, -0.25]
    },
    "mux_lane": 0,
    "driver": {
      "dc_gain": 1.0,
      "vswing": 0.8,
      "vcm_out": 0.6,
      "output_impedance": 50.0,
      "poles": [50e9],
      "sat_mode": "soft",
      "vlin": 1.0,
      "psrr": {"enable": false},
      "imbalance": {"gain_mismatch": 0.0, "skew": 0.0},
      "slew_rate": {"enable": false}
    }
  }
}
```

---

## 3. 核心实现机制

### 3.1 信号处理流程

TX发送端的完整信号处理流程包含5个关键步骤：

```
步骤1: 数字比特流 → WaveGen（数字→模拟NRZ波形）
步骤2: WaveGen输出 → FFE（FIR预均衡，预加重/去加重）
步骤3: FFE输出 → Mux（通道选择，可选延迟/抖动）
步骤4: Mux输出 → Driver（增益、带宽限制、饱和）
步骤5: Driver输出 → 差分信号输出到信道
```

**时序约束**：

- WaveGen采样率 = 数据速率（如10Gbps对应10GHz）
- FFE延迟线间隔 = 1 UI
- Mux工作在符号速率
- Driver采样率与上级一致

### 3.2 WaveGen-FFE级联设计

#### 3.2.1 波形生成

WaveGen将数字比特（0/1）映射为NRZ模拟波形：

- **NRZ编码**：`0 → -amplitude`，`1 → +amplitude`
- **典型幅度**：±1V（2V峰峰值）
- **支持的PRBS类型**：PRBS-7, PRBS-15, PRBS-23, PRBS-31

#### 3.2.2 FFE均衡策略

FFE采用FIR滤波器实现预补偿，数学表达式为：

```
y[n] = Σ c[k] × x[n-k]
       k=0 to N-1
```

**去加重配置示例**（PCIe Gen3）：

| 配置 | 抽头系数 | 去加重量 | 应用场景 |
|------|---------|---------|---------|
| 3.5dB | [0, 1.0, -0.25] | 3.5 dB | 短信道 |
| 6dB | [0, 1.0, -0.35] | 6 dB | 中等信道 |
| 9.5dB | [0, 1.0, -0.5] | 9.5 dB | 长信道 |

**预加重配置示例**：

| 配置 | 抽头系数 | 说明 |
|------|---------|------|
| 平衡模式 | [0.15, 0.7, 0.15] | 前后对称 |
| 混合模式 | [0.05, 0.85, -0.2] | 轻度预加重+去加重 |

#### 3.2.3 抽头系数约束

- **归一化约束**：`Σ|c[k]| ≈ 1`（保持功率）
- **主抽头最大**：主抽头通常为最大值
- **物理可实现性**：抽头系数应在硬件实现范围内

### 3.3 Mux通道选择与延迟建模

#### 3.3.1 单通道模式（Bypass）

- `lane_sel = 0`
- 直通模式，用于简单系统或调试
- 无额外延迟

#### 3.3.2 多通道模式（扩展）

- 在完整N:1并串转换系统中，Mux选择N个并行Lane之一
- 本行为模型简化为单输入单输出
- 通过`lane_sel`参数指定通道索引

#### 3.3.3 延迟与抖动建模

- **传播延迟**：10-50ps（可配置）
- **抖动注入**：支持DCD（占空比失真）和RJ（随机抖动）
- **应用场景**：测试CDR的抖动容限能力

### 3.4 Driver输出级设计

#### 3.4.1 增益与阻抗匹配

**阻抗匹配分压效应**：

```
开路电压: Voc = Vin × dc_gain
信道入口电压: Vchannel = Voc × Z0/(Zout + Z0)

对于理想匹配(Zout = Z0 = 50Ω):
Vchannel = Voc / 2
```

**参数配置示例**：

假设输入为±1V（2V峰峰值），期望信道入口800mV峰峰值，理想匹配：

```
驱动器内部开路摆幅需求: 800mV × 2 = 1600mV
配置: dc_gain = 1600mV / 2000mV = 0.8
```

#### 3.4.2 带宽限制

多极点传递函数模拟驱动器的频率响应：

```
H(s) = Gdc × ∏(1 + s/ωp_j)^(-1)
```

**极点频率选择**：

| 数据速率 | 推荐极点频率 | 说明 |
|---------|-------------|------|
| 10 Gbps | 10-15 GHz | 1.5-2×奈奎斯特 |
| 25 Gbps | 25-35 GHz | 1.5-2×奈奎斯特 |
| 56 Gbps | 40-50 GHz | 1.5-2×奈奎斯特 |

#### 3.4.3 非线性饱和

**软饱和（推荐）**：

```
Vout = Vswing × tanh(Vin / Vlin)
```

- 连续导数，谐波失真小
- Vlin参数决定线性区范围
- 推荐：`Vlin = Vswing / 1.2`

**硬饱和**：

```
Vout = clamp(Vin, -Vswing/2, +Vswing/2)
```

- 计算简单但导数不连续
- 适用于快速功能验证

#### 3.4.4 PSRR建模

电源纹波对输出的影响：

```cpp
struct PsrrParams {
    bool enable;                     // 启用PSRR建模
    double gain;                     // PSRR路径增益（如0.01 = -40dB）
    std::vector<double> poles;       // 低通滤波器极点
    double vdd_nom;                  // 标称电源电压
};
```

工作原理：`vdd_ripple = vdd - vdd_nom` → PSRR传递函数 → 耦合到差分输出

#### 3.4.5 差分不平衡

```cpp
struct ImbalanceParams {
    double gain_mismatch;            // 增益失配（%）
    double skew;                     // 相位偏移（s）
};
```

增益失配效应：
- `gain_p = 1 + mismatch/200`
- `gain_n = 1 - mismatch/200`

### 3.5 输出共模电压控制

**共模设置**：

- Driver的`vcm_out`参数设置差分输出共模电压（典型0.6V）
- 确保满足信道和接收端输入范围要求

**AC耦合链路**：

- 若信道有AC耦合电容，共模由信道的DC阻断特性决定
- TX侧共模不影响RX，但需避免Driver输出超出其线性范围

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

TX测试平台通常为开环设计：

- **TX侧**：WaveGen + FFE + Mux + Driver级联
- **负载**：理想50Ω阻抗 或 完整信道模型
- **性能评估**：眼图测量、频谱分析、输出摆幅统计

与RX测试的区别：
- TX测试为开环（无反馈路径）
- 主要关注输出信号质量，而非误码率

### 4.2 测试场景定义

| 场景 | 命令行参数 | 测试目标 | 输出文件 |
|------|----------|---------|----------|
| BASIC_OUTPUT | `basic` / `0` | 基本输出波形和摆幅 | tx_tran_basic.csv |
| FFE_SWEEP | `ffe_sweep` / `1` | 不同FFE系数下的眼图 | tx_eye_ffe_*.csv |
| DRIVER_SATURATION | `sat` / `2` | 大信号饱和特性 | tx_sat.csv |
| FREQUENCY_RESPONSE | `freq` / `3` | TX链路频率响应 | tx_freq_resp.csv |
| JITTER_INJECTION | `jitter` / `4` | WaveGen抖动注入效果 | tx_jitter.csv |

### 4.3 场景配置详解

#### BASIC_OUTPUT - 基本输出测试

- **信号源**：PRBS-31, 10Gbps
- **FFE**：默认去加重 [0, 1.0, -0.25]
- **Driver**：标准配置，Vswing=800mV
- **负载**：50Ω理想阻抗
- **验证点**：
  - 输出摆幅 ≈ 400mV（考虑50%分压）
  - 眼高 > 80%摆幅
  - 眼宽 > 0.6 UI

#### FFE_SWEEP - FFE参数扫描

- **抽头系数变化**：
  - 配置1：[0, 1.0, 0]（无均衡）
  - 配置2：[0, 1.0, -0.2]（3.5dB去加重）
  - 配置3：[0, 1.0, -0.35]（6dB去加重）
  - 配置4：[0.05, 0.9, -0.25]（混合模式）
- **验证点**：对比不同配置下的眼图开度和频谱

#### DRIVER_SATURATION - 饱和测试

- **输入幅度变化**：0.5×标称 → 2×标称
- **FFE**：无均衡（避免干扰）
- **验证点**：
  - 线性区：输出 ∝ 输入
  - 饱和区：输出钳位在sat_max/sat_min

#### FREQUENCY_RESPONSE - 频响测试

- **信号源**：正弦扫频（100MHz ~ 50GHz）
- **测量**：输出/输入幅度比
- **验证点**：
  - -3dB带宽应接近极点频率
  - 高频滚降斜率 ≈ -20dB/decade/极点

#### JITTER_INJECTION - 抖动注入测试

- **WaveGen抖动配置**：RJ_sigma=0.5ps, DCD=2%
- **测量**：输出眼图抖动增加量
- **验证点**：抖动传递与配置一致

### 4.4 信号连接拓扑

```
┌──────────┐   ┌─────┐   ┌─────┐   ┌────────┐   ┌────────┐   ┌──────────┐
│ WaveGen  │→→→│ FFE │→→→│ Mux │→→→│ Driver │→→→│ Channel│→→→│ Eye Mon  │
│(PRBS-31) │   │(FIR)│   │(Sel)│   │(Amp&BW)│   │(50Ω/S4P│   │(眼图采集) │
└──────────┘   └─────┘   └─────┘   └────────┘   └────────┘   └──────────┘
```

### 4.5 辅助模块说明

| 模块 | 功能 | 配置参数 |
|------|------|---------|
| **Ideal Load** | 50Ω纯阻性负载 | impedance |
| **Channel Model** | S参数导入，模拟真实信道 | touchstone |
| **Eye Monitor** | 采集眼图数据，计算眼高/眼宽/抖动 | ui_bins, amp_bins |
| **Spectrum Analyzer** | FFT分析输出频谱 | fft_size |

---

## 5. 仿真结果分析

### 5.1 统计指标说明

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| **输出摆幅** | max(out_p - out_n) - min(out_p - out_n) | 驱动能力 |
| **眼高** | min(高电平) - max(低电平) | 噪声裕量 |
| **眼宽** | 最优采样窗口（UI） | 时序裕量 |
| **上升/下降时间** | 10%-90%电平跨越时间 | 边沿速度 |
| **抖动（RMS）** | 边沿位置标准差 | 时序精度 |
| **频谱纯度** | 谐波失真度（dBc） | 非线性失真 |

### 5.2 典型测试结果解读

#### BASIC_OUTPUT测试结果示例

**配置**：10Gbps, PRBS-31, FFE=[0, 1.0, -0.25], 50Ω负载

**期望结果**：
```
=== TX Performance Summary ===
Output Swing (Diff):   420 mV (目标400mV，考虑分压)
Eye Height:            360 mV (85%摆幅)
Eye Width:             0.68 UI (68 ps)
Rise Time (20%-80%):   25 ps
Fall Time (20%-80%):   27 ps
Jitter (RMS):          1.2 ps (无WaveGen抖动)
FFE主抽头能量:         75%
FFE后抽头能量:         -25%
```

**波形特征**：
- WaveGen输出：标准NRZ方波，±1V
- FFE输出：跳变处有预加重"尖峰"，平坦区电平降低
- Driver输出：边沿变缓（带宽限制），摆幅符合预期

#### FFE_SWEEP结果解读

**眼图对比（信道入口处）**：

| 配置 | 后抽头系数 | 眼高(mV) | 眼宽(UI) | 说明 |
|------|----------|---------|---------|------|
| 无均衡 | 0 | 280 | 0.55 | 基线，ISI最严重 |
| 3.5dB去加重 | -0.2 | 340 | 0.65 | 标准配置 |
| 6dB去加重 | -0.35 | 370 | 0.70 | 高损耗信道优化 |
| 混合模式 | 前0.05，后-0.25 | 360 | 0.68 | 平衡前后游标 |

**频谱对比**：
- 无均衡：低频能量高，高频衰减大
- 6dB去加重：高频提升+6dB，低频降低，频谱更平坦

#### DRIVER_SATURATION结果解读

**输入-输出特性曲线**：
```
输出摆幅(mV)
800 |         ┌─────────  饱和区(Vswing限制)
    |       ┌─┘
600 |     ┌─┘  线性区(增益=0.8)
    |   ┌─┘
400 | ┌─┘
    |─┘
0   └────────────────────▶ 输入摆幅(mV)
    0   500  1000  1500  2000
```

**分析要点**：
- 输入<1000mV：线性放大，输出=输入×0.8
- 输入>1500mV：进入饱和，输出钳位在800mV
- 软饱和曲线更平滑，谐波失真小于硬饱和

### 5.3 波形数据文件格式

**tx_tran_basic.csv**：
```csv
时间(s),WaveGen_out(V),FFE_out(V),Driver_out_diff(V)
0.0e0,0.000,0.000,0.000
1.0e-11,1.000,1.000,0.380
2.0e-11,1.000,0.750,0.370
1.0e-10,1.000,0.750,0.375
1.1e-10,-1.000,-1.250,-0.485
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
make tx_tran_tb         # 构建TX顶层测试平台
make ffe_tran_tb        # 构建FFE单模块测试
make tx_driver_tran_tb  # 构建Driver单模块测试
cd tb
./tx_tran_tb [scenario]
```

场景参数：
- `basic` 或 `0` - 基本输出测试（默认）
- `ffe_sweep` 或 `1` - FFE参数扫描
- `sat` 或 `2` - 饱和测试
- `freq` 或 `3` - 频率响应测试
- `jitter` 或 `4` - 抖动注入测试

### 6.3 参数调优流程

**步骤1：确定输出摆幅需求**

- 查阅接口标准（PCIe/USB/Ethernet）
- 考虑信道损耗预算
- 设置Driver的Vswing参数

**步骤2：配置FFE抽头系数**

- 方法A：根据标准推荐值（如PCIe规范）
- 方法B：根据信道S参数仿真优化
- 方法C：启用自适应算法（扩展功能）

**步骤3：设置Driver增益和带宽**

- `dc_gain`：考虑阻抗匹配分压，设为期望摆幅的2倍（对于50Ω匹配）
- `poles`：根据数据速率设置，fp ≈ 1.5-2×(Bitrate/2)

**步骤4：运行仿真验证**

```bash
./tx_tran_tb basic
# 检查输出摆幅, 眼图, 上升时间
```

**步骤5：迭代优化**

- 若眼图闭合：优化FFE系数 或 增加Driver带宽
- 若饱和：降低输入幅度 或 调整Driver增益
- 若抖动过大：检查WaveGen配置 或 降低PSRR影响

### 6.4 结果查看

测试完成后，控制台输出统计结果，波形数据保存到CSV文件。使用Python进行可视化：

```bash
# 波形可视化
python scripts/plot_tx_waveforms.py tx_tran_basic.csv

# 眼图绘制
python scripts/plot_eye_diagram.py tx_eye_ffe_*.csv

# 频响曲线
python scripts/plot_freq_response.py tx_freq_resp.csv
```

---

## 7. 技术要点

### 7.1 FFE系数设计方法

#### 7.1.1 标准推荐值

| 标准 | 抽头系数 | 去加重量 |
|------|---------|---------|
| PCIe Gen3 | [0, 1.0, -0.25] | 3.5dB |
| PCIe Gen4 | [0, 1.0, -0.35] | 6dB |
| USB 3.2 | [0, 1.0, -0.2] | 可选 |

优点：快速配置，兼容性好

#### 7.1.2 信道逆滤波

```
FFE传递函数 F(f) 设计为 H_channel(f) 的逆
目标: F(f) × H_channel(f) ≈ 常数
```

- 在频域求解，转换回时域得抽头系数
- 优点：理论最优，适应信道特性

#### 7.1.3 时域优化

- 测量信道脉冲响应h[n]
- 目标：最小化 Σ(h_eq[n])² for n≠0（抑制ISI）
- 约束：Σ|c[k]| ≤ 1（功率限制）

### 7.2 Driver阻抗匹配原理

**分压效应**：
```
开路电压: Voc = Vin × dc_gain
信道入口电压: Vchannel = Voc × Z0/(Zout + Z0)

对于理想匹配(Zout = Z0 = 50Ω):
Vchannel = Voc / 2
```

**dc_gain设置**：
```
若期望信道入口800mV峰峰值，输入2V:
Voc需求 = 800mV × 2 = 1600mV
dc_gain = 1600mV / 2000mV = 0.8
```

**失配影响**：
- `Zout > Z0`：反射系数>0，信号反弹回TX
- `Zout < Z0`：反射系数<0，信号衰减过度
- 容差要求：`|Zout - Z0| / Z0 < 10%`

### 7.3 带宽限制的权衡

**极点频率选择**：
- **过高**：带宽过宽，高频噪声通过，EMI增加
- **过低**：带宽不足，ISI增加，眼图闭合
- **推荐**：fp = 1.5-2 × (Bitrate/2)

**多极点建模**：
- 真实Driver有多个寄生极点（管子Cgs、负载Cload、封装Lpkg）
- 单极点模型简化但可能低估高频衰减
- 关键链路建议使用2-3个极点精确建模

### 7.4 软饱和vs硬饱和

**软饱和（tanh）**：
- 优点：连续导数，谐波失真小，收敛性好
- 缺点：计算稍复杂
- 适用：生产仿真，精度要求高

**硬饱和（clamp）**：
- 优点：计算简单，极限情况清晰
- 缺点：导数不连续，可能引入高次谐波
- 适用：快速验证，最坏情况分析

**Vlin参数调优**：
```
Vlin = Vswing / α
α越大 → 线性区越窄 → 饱和越早
α越小 → 线性区越宽 → 信号裕量大
推荐: α = 1.2-1.5
```

### 7.5 FFE与Driver级联的功率管理

**问题**：FFE预加重会增加峰值摆幅

```
例: FFE系数[0, 1.0, -0.3]
跳变时刻: y[n] = 1.0×(+1) + (-0.3)×(-1) = 1.3 (增加30%)
```

**解决方案1：归一化FFE输出**
```
scale_factor = 1 / max(|y[n]|)
FFE输出 × scale_factor
```

**解决方案2：Driver预留裕量**
```
Driver的sat_max设置为标称摆幅的1.3-1.5倍
```

**解决方案3：自适应功率控制**
```
监控Driver输出，若接近饱和则降低FFE系数幅度
```

### 7.6 抖动建模的实用性

**DCD（占空比失真）**：
- 物理来源：时钟占空比≠50%
- 影响：奇偶UI宽度不等，产生确定性抖动
- 建模：在WaveGen对奇偶符号施加±DCD/2的时间偏移

**RJ（随机抖动）**：
- 物理来源：PLL相位噪声、热噪声
- 影响：数据边沿时刻随机波动
- 建模：在每个符号时刻叠加高斯分布时移

**抖动传递**：
- TX侧注入抖动 → 信道传输 → RX侧CDR需要跟踪
- 可用于测试CDR的JTOL能力

### 7.7 时间步长与采样率设置

**一致性要求**：
所有TDF模块必须设置相同的采样率：

```cpp
// 全局配置
double Fs = 100e9;  // 100 GHz
double Ts = 1.0 / Fs;

// 各模块set_attributes()
wavegen.set_timestep(Ts);
ffe.set_timestep(Ts);
mux.set_timestep(Ts);
driver.set_timestep(Ts);
```

**采样率选择**：
- 最小：2×最高频率（Nyquist）
- 推荐：5-10×符号速率
- 过高：仿真时间长，文件大
- 过低：波形失真，带宽建模不准

---

## 8. 参考信息

### 8.1 相关文件

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| WaveGen头文件 | `/include/ams/wave_gen.h` | WaveGenTdf类声明 |
| WaveGen实现 | `/src/ams/wave_gen.cpp` | WaveGenTdf类实现 |
| FFE头文件 | `/include/ams/tx_ffe.h` | TxFfeTdf类声明 |
| FFE实现 | `/src/ams/tx_ffe.cpp` | TxFfeTdf类实现 |
| Mux头文件 | `/include/ams/tx_mux.h` | TxMuxTdf类声明 |
| Mux实现 | `/src/ams/tx_mux.cpp` | TxMuxTdf类实现 |
| Driver头文件 | `/include/ams/tx_driver.h` | TxDriverTdf类声明 |
| Driver实现 | `/src/ams/tx_driver.cpp` | TxDriverTdf类实现 |
| 参数定义 | `/include/common/parameters.h` | TxParams/WaveGenParams结构体 |
| WaveGen文档 | `/docs/modules/waveGen.md` | WaveGen详细技术文档 |
| FFE文档 | `/docs/modules/ffe.md` | FFE详细技术文档 |
| Mux文档 | `/docs/modules/mux.md` | Mux详细技术文档 |
| Driver文档 | `/docs/modules/driver.md` | Driver详细技术文档 |
| FFE测试平台 | `/tb/tx/ffe_tran_tb.cpp` | FFE瞬态测试 |
| Driver测试平台 | `/tb/tx/tx_driver_tran_tb.cpp` | Driver瞬态测试 |
| FFE单元测试 | `/tests/unit/test_ffe_*.cpp` | FFE单元测试集 |
| Driver单元测试 | `/tests/unit/test_tx_driver_*.cpp` | Driver单元测试集 |

### 8.2 依赖项

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14标准
- GoogleTest 1.12.1（单元测试）

### 8.3 性能指标总结

| 指标 | 典型值 | 说明 |
|------|-------|------|
| 最大数据速率 | 56 Gbps | 取决于工艺和封装 |
| 输出摆幅 | 800-1200 mV | 符合PCIe/USB标准 |
| 上升/下降时间 | 20-40 ps | @ 10Gbps |
| 输出抖动 | < 2 ps RMS | 不含信道影响 |
| FFE抽头数 | 3-7 | 典型3抽头（PCIe） |
| 输出阻抗 | 50 Ω | 差分100Ω |
| PSRR | > 40 dB | @ 1MHz |

### 8.4 接口标准参考

| 标准 | 数据速率 | 摆幅要求 | FFE配置 |
|------|---------|---------|---------|
| PCIe Gen3 | 8 Gbps | 800-1200mV | 3-tap, 3.5dB或6dB去加重 |
| PCIe Gen4 | 16 Gbps | 800-1200mV | 3-tap, 强制去加重 |
| USB 3.2 Gen2 | 10 Gbps | 800-1000mV | 3-tap, 可选均衡 |
| 10G Ethernet | 10.3125 Gbps | 500-800mV | 5-tap, 混合模式 |
| 25G Ethernet | 25.78125 Gbps | 400-800mV | 5-7 tap, 强均衡 |

---

**文档版本**：v1.0  
**最后更新**：2026-01-27  
**作者**：Yizhe Liu
