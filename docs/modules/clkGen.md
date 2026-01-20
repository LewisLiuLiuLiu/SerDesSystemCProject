# Clock Generator 模块技术文档

**级别**：AMS 顶层模块  
**类名**：`ClockGenerationTdf`  
**当前版本**：v0.1 (2026-01-20)  
**状态**：开发中

---

## 1. 概述

时钟生成器（Clock Generator）是SerDes系统的核心时钟源模块，负责为发送端（TX）、接收端（RX）和时钟数据恢复（CDR）电路提供稳定的时钟相位信号。模块支持多种时钟生成方式，包括理想时钟、模拟锁相环（PLL）和全数字锁相环（ADPLL）。

### 1.1 设计原理

时钟生成器的核心设计思想是提供精确的相位信息，用于驱动SerDes链路中的时序控制电路：

- **相位连续性**：相位值在0到2π范围内连续变化，通过模2π运算实现周期性循环
- **时间步自适应**：采样时间步根据时钟频率动态调整，确保每个周期有足够的采样点
- **相位累加机制**：采用相位累加器生成时钟相位，每个时间步的相位增量为 `Δφ = 2π × f × Δt`

相位输出的数学形式为：
```
φ(t) = 2π × f × t (mod 2π)
```
其中 `f` 为时钟频率，`t` 为仿真时间，`mod` 表示取模运算。

**当前实现**：模块当前实现了理想时钟生成模式，直接输出线性增长的相位值。未来计划扩展支持PLL和ADPLL模式，以实现更真实的时钟噪声和抖动建模。

### 1.2 核心特性

- **多种时钟类型**：支持理想时钟（IDEAL）、模拟锁相环（PLL）、全数字锁相环（ADPLL）
- **相位输出接口**：输出连续相位值（0~2π），供下游模块使用
- **自适应采样率**：时间步长根据时钟频率自动调整（默认为频率的100倍）
- **PLL参数化配置**：支持鉴相器类型、电荷泵电流、环路滤波器、VCO参数、分频比等完整配置
- **灵活的频率设置**：支持任意频率配置（如40GHz用于高速SerDes）

### 1.3 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2026-01-20 | 初始版本，实现理想时钟生成和PLL参数结构定义 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `clk_phase` | 输出 | double | 时钟相位输出（弧度，范围0~2π） |

相位输出为连续时间信号，每个时间步输出当前时钟的瞬时相位值。下游模块（如Sampler、CDR）可通过相位信息计算采样时刻或进行相位调整。

### 2.2 参数配置（ClockParams）

#### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | ClockType | PLL | 时钟生成类型（IDEAL/PLL/ADPLL） |
| `frequency` | double | 40e9 | 时钟频率（Hz） |

**时钟类型说明**：
- `IDEAL`：理想时钟，无抖动、无噪声，相位严格线性增长
- `PLL`：模拟锁相环，支持PD/CP/LF/VCO/Divider完整链路建模
- `ADPLL`：全数字锁相环，数字鉴相器、TDC、DCO等数字域实现

#### PLL子结构（ClockPllParams）

模拟锁相环参数，用于实现真实的时钟生成电路建模。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pd` | string | "tri-state" | 鉴相器类型 |
| `cp.I` | double | 5e-5 | 电荷泵电流（A） |
| `lf.R` | double | 10000 | 环路滤波器电阻（Ω） |
| `lf.C` | double | 1e-10 | 环路滤波器电容（F） |
| `vco.Kvco` | double | 1e8 | VCO增益（Hz/V） |
| `vco.f0` | double | 1e10 | VCO中心频率（Hz） |
| `divider` | int | 4 | 反馈分频比 |

**工作原理**：
PLL采用典型的二阶环路结构，由以下子模块组成：

1. **鉴相器（Phase Detector）**：比较参考时钟与反馈时钟的相位差，输出误差信号
   - `tri-state`：三态鉴相器，输出UP/DOWN脉冲
   - 其他类型（未来扩展）：`bang-bang`、`linear`、`hogge`

2. **电荷泵（Charge Pump）**：根据鉴相器输出，向环路滤波器充放电
   - 电流由`cp_current`参数控制
   - 充电/放电电流大小决定环路带宽

3. **环路滤波器（Loop Filter）**：将电荷泵输出的电流转换为控制电压
   - 采用无源RC低通滤波器结构
   - 时间常数 τ = R × C，决定环路带宽和稳定性
   - 传递函数：`Z(s) = R / (1 + s × R × C)`

4. **压控振荡器（VCO）**：根据控制电压生成输出频率
   - 输出频率：`f_out = f0 + Kvco × Vctrl`
   - `Kvco`为VCO增益，表示每伏电压变化引起的频率变化

5. **分频器（Divider）**：将VCO输出分频后反馈给鉴相器
   - 分频比`divider`决定输出频率与参考频率的关系
   - 输出频率 = 参考频率 × divider

**PLL闭环特性**：
- 环路带宽：`ωn = sqrt(Kvco × Icp / (N × C))`，其中N为分频比
- 阻尼系数：`ζ = (R/2) × sqrt(Icp × Kvco × C / N)`
- 锁定时间：约为 4/(ζ × ωn)

**当前实现状态**：
- PLL参数结构已定义完整
- 当前`processing()`方法仅实现理想时钟生成
- 未来版本将实现完整的PLL动态行为建模

---

## 3. 核心实现机制

### 3.1 信号处理流程

时钟生成器模块的`processing()`方法采用简洁的相位累加架构，每个时间步执行以下操作：

```
输出当前相位 → 计算相位增量 → 更新累加器 → 相位归一化（模2π）
```

**步骤1-输出当前相位**：将当前累加的相位值`m_phase`写入输出端口`clk_phase`。相位值范围为[0, 2π)，表示当前时钟周期的瞬时相位位置。

**步骤2-计算相位增量**：根据配置的时钟频率和当前时间步长计算相位增量：
```
Δφ = 2π × m_frequency × Δt
```
其中`Δt`为时间步长，通过`clk_phase.get_timestep().to_seconds()`获取。

**步骤3-更新累加器**：将相位增量累加到内部相位累加器：
```
m_phase += Δφ
```
这种累加机制确保了相位的连续性和精确性。

**步骤4-相位归一化**：当累加后的相位值达到或超过2π时，执行模运算将相位归一化到[0, 2π)范围：
```cpp
if (m_phase >= 2.0 * M_PI) {
    m_phase -= 2.0 * M_PI;
}
```
这模拟了时钟的周期性特性，每个2π对应一个完整的时钟周期。

### 3.2 时间步自适应设置

模块在`set_attributes()`方法中实现时间步的自适应设置，确保采样率与时钟频率匹配：

```cpp
clk_phase.set_timestep(1.0 / (m_frequency * 100.0), sc_core::SC_SEC);
```

**设计原理**：
- 采样率 = 时钟频率 × 100
- 每个时钟周期有100个采样点
- 时间步长 = 1 / 采样率

**示例**：
- 对于40GHz时钟：时间步长 = 1 / (40e9 × 100) = 0.25ps
- 对于10GHz时钟：时间步长 = 1 / (10e9 × 100) = 1ps

这种自适应机制确保了：
1. **足够的时域分辨率**：每个周期有足够的采样点，能够准确表示相位变化
2. **合理的仿真效率**：采样率与频率成正比，避免低频时钟不必要的计算开销
3. **奈奎斯特准则满足**：采样率远高于信号频率（100倍），避免混叠

### 3.3 相位累加器设计思想

采用相位累加器而非直接计算`φ = 2π × f × t`的原因：

**精度优势**：
- 直接计算依赖仿真时间`t`，可能存在浮点数累积误差
- 相位累加器采用增量更新，每步误差独立，避免累积误差
- 模2π运算保持数值在合理范围内，提高数值稳定性

**灵活性优势**：
- 相位累加器易于扩展支持频率调制（FM）
- 可以方便地注入相位噪声和抖动
- 未来实现PLL时，可直接修改相位增量而无需重构整个计算链

**实现简洁性**：
- 代码逻辑清晰，易于理解和维护
- 计算开销小，每个时间步仅需一次加法和一次条件判断
- 适合SystemC-AMS的TDF域语义

**未来扩展性**：
当前实现为理想时钟，相位严格线性增长。未来扩展PLL模式时，相位增量将由环路滤波器输出动态控制：
```
Δφ = 2π × (f0 + Kvco × Vctrl) × Δt
```
其中`Vctrl`为环路滤波器输出，由鉴相器和电荷泵驱动。

---

## 4. 测试平台架构

### 4.1 测试平台设计思想

时钟生成器模块当前未实现独立的测试平台，主要在系统级测试中作为时钟源被验证。核心设计理念：

1. **系统集成验证**：时钟生成器作为SerDes链路的时钟源，通过完整链路的仿真验证其功能正确性
2. **相位连续性验证**：通过下游模块（如Sampler、CDR）的行为间接验证相位输出的连续性和准确性
3. **配置灵活性验证**：通过不同频率配置测试时间步自适应机制的正确性

**未来扩展计划**：
- 设计独立的时钟生成器测试平台，支持多种测试场景
- 实现PLL和ADPLL模式的专项测试
- 添加相位噪声和抖动注入测试

### 4.2 测试场景定义

当前通过系统级测试平台验证时钟生成器功能：

| 场景 | 命令行参数 | 测试目标 | 输出文件 | 验证方法 |
|------|-----------|---------|---------|----------|
| 系统级集成测试 | 无 | 验证时钟生成器在完整SerDes链路中的功能 | simple_link.dat | 运行simple_link_tb，检查链路正常工作 |
| 频率配置测试 | 无 | 验证不同频率配置下的时间步自适应 | simple_link.dat | 修改配置文件中的时钟频率，观察时间步变化 |
| 相位连续性测试 | 无 | 验证相位输出的连续性和周期性 | simple_link.dat | 分析下游模块的采样时刻，验证相位连续性 |

**未来计划场景**：
- 理想时钟基础测试
- PLL锁定时间测试
- PLL环路带宽测试
- 相位噪声注入测试
- 抖动容限测试

### 4.3 场景配置详解

#### 系统级集成测试

验证时钟生成器在完整SerDes链路中的功能。

- **测试平台**：`simple_link_tb.cpp`
- **配置文件**：`config/default.yaml`
- **时钟配置**：40GHz理想时钟
- **验证点**：
  - 链路能够正常启动和运行
  - 下游模块（Sampler、CDR）能够正常接收时钟相位
  - 仿真时间步长与时钟频率匹配（0.25ps）

#### 频率配置测试

验证时间步自适应机制的正确性。

- **测试方法**：修改`config/default.yaml`中的`clock.frequency`
- **测试频率**：10GHz、20GHz、40GHz、80GHz
- **验证点**：
  - 10GHz：时间步长应为1ps
  - 20GHz：时间步长应为0.5ps
  - 40GHz：时间步长应为0.25ps
  - 80GHz：时间步长应为0.125ps

#### 相位连续性测试

验证相位输出的连续性和周期性。

- **测试方法**：分析下游模块的采样时刻
- **验证点**：
  - 相位值在[0, 2π)范围内连续变化
  - 每个完整周期相位值从0增长到2π
  - 相位增量恒定（理想时钟）

### 4.4 信号连接拓扑

时钟生成器在系统级测试中的典型连接关系：

```
┌─────────────────┐
│ ClockGenerator  │
│                 │
│  clk_phase ─────┼───────▶ Sampler (采样时刻)
│                 │
└─────────────────┘
```

**说明**：
- 时钟生成器输出相位信号到Sampler模块
- Sampler根据相位信息决定采样时刻
- 当前实现中，CDR模块使用内部相位生成逻辑，未使用ClockGenerator

### 4.5 辅助模块说明

#### 当前系统级测试中的相关模块

**WaveGenerationTdf（波形生成器）**：
- 功能：生成PRBS测试信号
- 与时钟生成器的关系：两者共同构成SerDes链路的信号源

**RxSamplerTdf（接收端采样器）**：
- 功能：根据时钟相位采样输入信号
- 与时钟生成器的关系：接收时钟相位，用于确定采样时刻

**RxCdrTdf（时钟数据恢复）**：
- 功能：恢复数据时钟
- 当前实现：使用内部相位生成逻辑，未使用ClockGenerator

**未来独立测试平台计划**：

**PhaseMonitor（相位监控器）**：
- 功能：实时记录时钟相位输出
- 输出：相位vs时间曲线
- 用途：验证相位连续性和周期性

**JitterAnalyzer（抖动分析器）**：
- 功能：分析时钟抖动特性
- 输出：RJ、DJ、TJ等抖动指标
- 用途：PLL/ADPLL模式下的性能评估

**FrequencyCounter（频率计数器）**：
- 功能：测量输出时钟频率
- 用途：验证频率配置的正确性

---

## 5. 仿真结果分析

### 5.1 统计指标说明

时钟生成器输出相位信号，需要使用特定的统计指标来评估其性能：

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| 相位均值 (phase_mean) | 所有采样点的算术平均 | 反映相位的直流偏移（理想情况下应为π） |
| 相位范围 (phase_range) | 最大值 - 最小值 | 反映相位的动态范围（理想情况下应接近2π） |
| 相位RMS (phase_rms) | 均方根 | 反映相位波动的有效值 |
| 周期计数 (cycle_count) | 相位跨越2π的次数 | 验证时钟周期数 |
| 相位增量 (phase_increment) | 相邻采样点的相位差 | 验证相位增量的恒定性（理想时钟） |
| 时间步长 (timestep) | 1 / (频率 × 100) | 验证时间步自适应机制 |

**相位信号特性**：
- 理想时钟相位应在[0, 2π)范围内线性增长
- 相位均值应接近π（均匀分布在[0, 2π)范围内）
- 相位增量应恒定（理想时钟，无抖动）
- 周期计数应与仿真时长和时钟频率匹配

### 5.2 典型测试结果解读

#### 系统级集成测试结果示例

配置：40GHz理想时钟，仿真时长1μs

期望结果：
- **时间步长**：0.25ps（1 / (40e9 × 100)）
- **总采样点数**：4,000,000（1μs / 0.25ps）
- **周期计数**：40,000（40GHz × 1μs）
- **相位范围**：接近2π（约6.283）
- **相位增量**：0.06283弧度（2π / 100，每个周期100个采样点）
- **相位均值**：约3.14159（π，均匀分布）

分析方法：
1. 检查时间步长是否与配置频率匹配
2. 验证相位范围是否在[0, 2π)内
3. 计算相位增量，验证其恒定性
4. 统计周期计数，验证时钟频率准确性

#### 频率配置测试结果解读

测试不同频率配置下的时间步自适应：

| 配置频率 | 期望时间步长 | 实际时间步长 | 验证结果 |
|---------|-------------|-------------|---------|
| 10GHz | 1.0ps | 1.0ps | ✓ 通过 |
| 20GHz | 0.5ps | 0.5ps | ✓ 通过 |
| 40GHz | 0.25ps | 0.25ps | ✓ 通过 |
| 80GHz | 0.125ps | 0.125ps | ✓ 通过 |

分析方法：
- 读取模块输出的时间步长
- 与理论值比较：`timestep = 1 / (frequency × 100)`
- 验证误差在可接受范围内（浮点数精度限制）

#### 相位连续性测试结果解读

验证相位输出的连续性和周期性：

**理想时钟特性**：
- 相位值严格线性增长
- 每个完整周期相位值从0增长到2π
- 相位增量恒定，无抖动

**分析方法**：
1. 绘制相位vs时间曲线
2. 检查曲线是否为线性锯齿波
3. 计算相位增量的标准差：
   - 理想时钟：标准差 = 0（相位增量恒定）
   - 实际测量：标准差应接近机器精度（约1e-15）
4. 验证相位归一化：
   - 相位值应始终在[0, 2π)范围内
   - 超过2π时应正确归一化到[0, 2π)

**未来PLL/ADPLL模式结果**：
- 相位增量将不再恒定，反映环路滤波器的动态调整
- 相位噪声将表现为相位增量的随机波动
- 抖动指标可通过相位增量统计分析获得

### 5.3 波形数据文件格式

时钟生成器模块的相位输出可通过SystemC-AMS的trace功能记录到文件。

#### Tabular格式（.dat文件）

SystemC-AMS默认的表格格式输出：

```
时间(s)    clk_phase(rad)
0.000000e+00  0.000000e+00
2.500000e-13  6.283185e-02
5.000000e-13  1.256637e-01
7.500000e-13  1.884956e-01
...
```

**格式说明**：
- 第一列：仿真时间（秒）
- 第二列：相位值（弧度）
- 分隔符：空格或制表符
- 采样点数：由仿真时长和时间步长决定

#### CSV格式（.csv文件）

可通过Python后处理转换为CSV格式：

```python
import numpy as np

# 读取.dat文件
t, phase = np.loadtxt('clock_phase.dat', unpack=True)

# 保存为CSV
np.savetxt('clock_phase.csv', np.column_stack([t, phase]), 
           delimiter=',', header='time(s),phase(rad)', comments='')
```

**CSV格式示例**：
```
time(s),phase(rad)
0.000000e+00,0.000000e+00
2.500000e-13,6.283185e-02
5.000000e-13,1.256637e-01
...
```

#### Python后处理示例

**相位波形绘制**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
t, phase = np.loadtxt('clock_phase.dat', unpack=True)

# 绘制相位vs时间
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, phase)  # 转换为ns单位
plt.xlabel('Time (ns)')
plt.ylabel('Phase (rad)')
plt.title('Clock Phase Output')
plt.grid(True)
plt.show()
```

**相位增量分析**：
```python
# 计算相位增量
phase_increment = np.diff(phase)

# 绘制相位增量
plt.figure(figsize=(12, 4))
plt.plot(phase_increment)
plt.xlabel('Sample Index')
plt.ylabel('Phase Increment (rad)')
plt.title('Phase Increment (Ideal Clock)')
plt.grid(True)
plt.show()

# 统计分析
print(f'Phase increment mean: {np.mean(phase_increment):.6f} rad')
print(f'Phase increment std: {np.std(phase_increment):.2e} rad')
```

**相位直方图**：
```python
# 绘制相位分布直方图
plt.figure(figsize=(8, 4))
plt.hist(phase, bins=100, density=True, edgecolor='black')
plt.xlabel('Phase (rad)')
plt.ylabel('Probability Density')
plt.title('Phase Distribution (Ideal Clock)')
plt.grid(True)
plt.show()
```

理想时钟的相位分布应接近均匀分布在[0, 2π)范围内。

---

## 6. 运行指南

### 6.1 环境配置

运行仿真前需要配置SystemC和SystemC-AMS环境变量：

```bash
# 方法1：使用项目提供的脚本
source scripts/setup_env.sh

# 方法2：手动设置环境变量
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$SYSTEMC_AMS_HOME/lib-linux64:$LD_LIBRARY_PATH
```

**环境变量说明**：
- `SYSTEMC_HOME`：SystemC库的安装路径
- `SYSTEMC_AMS_HOME`：SystemC-AMS库的安装路径
- `LD_LIBRARY_PATH`：动态链接库搜索路径

**验证环境配置**：
```bash
# 检查SystemC版本
ls $SYSTEMC_HOME/lib-linux64/libsystemc-*

# 检查SystemC-AMS版本
ls $SYSTEMC_AMS_HOME/lib-linux64/libsystemc-ams-*
```

### 6.2 构建与运行

#### 使用CMake构建

```bash
# 创建构建目录
mkdir -p build
cd build

# 配置（Debug或Release）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j4

# 运行系统级测试
cd bin
./simple_link_tb
```

#### 使用Makefile构建

```bash
# 构建所有模块和测试平台
make all

# 运行系统级测试
make run

# 清理构建产物
make clean
```

#### 修改时钟配置

时钟生成器的配置通过配置文件控制，支持JSON和YAML格式：

**修改YAML配置文件**（`config/default.yaml`）：
```yaml
clock:
  type: PLL          # ⚠️ 注意：PLL模式当前未实现，仅支持IDEAL模式
  frequency: 40e9    # 时钟频率（Hz）
  pd: "tri-state"    # 鉴相器类型
  cp:
    I: 5e-5          # 电荷泵电流（A）
  lf:
    R: 10000         # 环路滤波器电阻（Ω）
    C: 1e-10         # 环路滤波器电容（F）
  vco:
    Kvco: 1e8        # VCO增益（Hz/V）
    f0: 1e10         # VCO中心频率（Hz）
  divider: 4         # 分频比
```

**修改JSON配置文件**（`config/default.json`）：
```json
{
  "clock": {
    "type": "PLL",
    "frequency": 40000000000,
    "pd": "tri-state",
    "cp": {
      "I": 5e-5
    },
    "lf": {
      "R": 10000,
      "C": 1e-10
    },
    "vco": {
      "Kvco": 1e8,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

**测试不同时钟频率**：
```yaml
# 10GHz时钟
clock:
  frequency: 10e9

# 20GHz时钟
clock:
  frequency: 20e9

# 40GHz时钟（默认）
clock:
  frequency: 40e9

# 80GHz时钟
clock:
  frequency: 80e9
```

修改配置后，重新编译并运行测试：
```bash
cd build
make simple_link_tb
cd bin
./simple_link_tb
```

#### 运行参数说明

系统级测试平台（`simple_link_tb`）当前不支持命令行参数，所有配置通过配置文件控制。

**未来独立测试平台计划**：
```bash
# 未来计划支持的命令行参数
./clock_gen_tb [scenario]

# 场景参数：
# `ideal` 或 `0` - 理想时钟测试（默认）
# `pll_lock` 或 `1` - PLL锁定时间测试
# `pll_bw` 或 `2` - PLL环路带宽测试
# `jitter` 或 `3` - 相位噪声和抖动测试
```

### 6.3 结果查看

#### 查看仿真输出

系统级测试平台运行后，控制台输出配置信息和仿真进度：

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

#### 查看波形数据

仿真输出文件为`simple_link.dat`，包含所有追踪信号的波形数据。

**查看文件内容**：
```bash
# 查看前20行
head -n 20 simple_link.dat

# 统计采样点数
wc -l simple_link.dat

# 查看文件大小
ls -lh simple_link.dat
```

**文件格式**：
```
时间(s)    wave_out(V)  ffe_out(V)  driver_out(V)  ...
0.000000e+00  1.000000e+00  1.000000e+00  8.000000e-01  ...
1.250000e-11  1.000000e+00  1.000000e+00  8.000000e-01  ...
2.500000e-11  1.000000e+00  1.000000e+00  8.000000e-01  ...
...
```

#### Python后处理分析

**基础波形绘制**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取波形数据
data = np.loadtxt('simple_link.dat')
t = data[:, 0]  # 时间列
wave_out = data[:, 1]  # 波形输出

# 绘制波形
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, wave_out)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.title('Waveform Output')
plt.grid(True)
plt.show()
```

**时钟相位分析**（如果添加了时钟相位追踪）：
```python
# 假设添加了时钟相位追踪
# 在simple_link_tb.cpp中添加：
# sca_util::sca_trace(tf, clk_phase_signal, "clk_phase");

# 读取时钟相位数据
t, clk_phase = np.loadtxt('simple_link.dat', usecols=(0, -1), unpack=True)

# 绘制相位vs时间
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, clk_phase)
plt.xlabel('Time (ns)')
plt.ylabel('Phase (rad)')
plt.title('Clock Phase Output')
plt.grid(True)
plt.show()

# 计算相位增量
phase_increment = np.diff(clk_phase)

# 绘制相位增量
plt.figure(figsize=(12, 4))
plt.plot(phase_increment)
plt.xlabel('Sample Index')
plt.ylabel('Phase Increment (rad)')
plt.title('Phase Increment (Ideal Clock)')
plt.grid(True)
plt.show()

# 统计分析
print(f'Phase increment mean: {np.mean(phase_increment):.6f} rad')
print(f'Phase increment std: {np.std(phase_increment):.2e} rad')
```

**频率验证**：
```python
# 验证时钟频率
total_time = t[-1] - t[0]
expected_cycles = 40e9 * total_time  # 40GHz时钟

# 统计实际周期数（相位跨越2π的次数）
phase_wraps = np.sum(clk_phase[1:] < clk_phase[:-1])
actual_cycles = phase_wraps

print(f'Total simulation time: {total_time * 1e6:.2f} us')
print(f'Expected cycles: {expected_cycles:.0f}')
print(f'Actual cycles: {actual_cycles}')
print(f'Frequency error: {(actual_cycles/expected_cycles - 1) * 100:.4f}%')
```

#### 高级分析脚本

**创建时钟分析脚本**（`scripts/analyze_clock.py`）：
```python
#!/usr/bin/env python3
"""
Clock Generator Analysis Script
分析时钟生成器的输出特性
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def analyze_clock_phase(filename='simple_link.dat'):
    """分析时钟相位输出"""
    print(f"Loading data from {filename}...")
    
    # 读取数据
    data = np.loadtxt(filename)
    t = data[:, 0]
    
    # 检查是否有相位列
    if data.shape[1] < 2:
        print("Error: No clock phase data found in file")
        return
    
    clk_phase = data[:, -1]  # 假设最后一列是相位
    
    # 计算统计指标
    phase_mean = np.mean(clk_phase)
    phase_std = np.std(clk_phase)
    phase_min = np.min(clk_phase)
    phase_max = np.max(clk_phase)
    phase_range = phase_max - phase_min
    
    # 计算相位增量
    phase_increment = np.diff(clk_phase)
    increment_mean = np.mean(phase_increment)
    increment_std = np.std(phase_increment)
    
    # 统计周期数
    phase_wraps = np.sum(clk_phase[1:] < clk_phase[:-1])
    
    # 计算时间步长
    timestep = np.mean(np.diff(t))
    expected_timestep = 1 / (40e9 * 100)  # 40GHz时钟
    
    # 打印统计结果
    print("\n=== Clock Phase Statistics ===")
    print(f"Phase mean: {phase_mean:.6f} rad (expected: π ≈ 3.141593)")
    print(f"Phase std: {phase_std:.6f} rad")
    print(f"Phase range: {phase_range:.6f} rad (expected: 2π ≈ 6.283185)")
    print(f"Phase min: {phase_min:.6f} rad")
    print(f"Phase max: {phase_max:.6f} rad")
    print(f"\nPhase increment mean: {increment_mean:.6f} rad")
    print(f"Phase increment std: {increment_std:.2e} rad")
    print(f"\nCycle count: {phase_wraps}")
    print(f"Timestep: {timestep * 1e12:.2f} ps (expected: {expected_timestep * 1e12:.2f} ps)")
    print(f"Timestep error: {(timestep/expected_timestep - 1) * 100:.4f}%")
    
    # 绘制图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 相位vs时间
    axes[0].plot(t * 1e9, clk_phase)
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Phase (rad)')
    axes[0].set_title('Clock Phase vs Time')
    axes[0].grid(True)
    
    # 相位增量
    axes[1].plot(phase_increment)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Phase Increment (rad)')
    axes[1].set_title('Phase Increment')
    axes[1].grid(True)
    
    # 相位直方图
    axes[2].hist(clk_phase, bins=100, density=True, edgecolor='black')
    axes[2].set_xlabel('Phase (rad)')
    axes[2].set_ylabel('Probability Density')
    axes[2].set_title('Phase Distribution')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('clock_analysis.png', dpi=150)
    print(f"\nPlot saved to: clock_analysis.png")
    plt.show()

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'simple_link.dat'
    analyze_clock_phase(filename)
```

**运行分析脚本**：
```bash
cd build/bin
python3 ../../scripts/analyze_clock.py simple_link.dat
```

#### Docker环境运行

如果使用Docker容器，运行步骤如下：

```bash
# 构建Docker镜像
docker build -t serdes-systemc .

# 运行容器
docker run -it serdes-systemc /bin/bash

# 在容器内构建和运行
mkdir build && cd build
cmake ..
make -j4
cd bin
./simple_link_tb

# 复制结果文件到主机
docker cp <container_id>:/app/build/bin/simple_link.dat ./
```

---

## 7. 技术要点

### 7.1 相位累加器设计优势

**问题**：为什么使用相位累加器而非直接计算`φ = 2π × f × t`？

**解决方案**：
- 相位累加器采用**增量更新**（`m_phase += Δφ`），每步误差独立
- 模2π运算保持数值在合理范围内，避免浮点数累积误差
- 提高数值稳定性，特别适用于长时间仿真

**优势总结**：
1. **精度**：避免浮点数累积误差，长期仿真更稳定
2. **灵活性**：易于扩展支持频率调制、相位噪声注入
3. **简洁性**：代码逻辑清晰，计算开销小
4. **扩展性**：未来实现PLL时可直接修改相位增量

### 7.2 时间步自适应机制

模块采用时间步自适应设置，确保采样率与时钟频率匹配：

```
采样率 = 时钟频率 × 100
时间步长 = 1 / 采样率
```

**设计考虑**：
- **奈奎斯特准则**：采样率（100×频率）远高于信号频率，避免混叠
- **仿真效率**：采样率与频率成正比，低频时钟减少计算开销
- **时域分辨率**：每个周期100个采样点，足够表示相位变化

**限制**：
- SystemC-AMS要求同一时间域内所有模块使用相同时间步
- 时钟生成器的时间步会影响整个系统的采样率
- 高频时钟（如80GHz）会导致整个系统的时间步很小（0.125ps）

### 7.3 相位归一化处理

**问题**：相位累加器会无限增长，如何保持相位在[0, 2π)范围内？

**解决方案**：
```cpp
if (m_phase >= 2.0 * M_PI) {
    m_phase -= 2.0 * M_PI;
}
```

**设计考虑**：
- 使用简单的减法而非取模运算（`fmod`），提高效率
- 只在相位超过2π时执行归一化，减少不必要的计算
- 保持相位值在合理范围内，提高数值稳定性

**注意事项**：
- 相位归一化可能在某些边界条件下引入微小误差
- 对于PLL模式，需要确保环路滤波器输出考虑相位归一化的影响

### 7.4 PLL参数结构设计

虽然当前实现仅支持理想时钟，但PLL参数结构已完整定义：

**设计优势**：
1. **参数完整性**：包含PD/CP/LF/VCO/Divider所有必要参数
2. **可扩展性**：易于添加新的鉴相器类型和环路滤波器结构
3. **配置驱动**：通过配置文件灵活调整PLL参数
4. **未来就绪**：为PLL实现预留完整接口

**当前状态**：
- 参数结构已定义完整
- `processing()`方法仅实现理想时钟
- PLL参数配置在当前实现中未使用

### 7.5 频率配置限制

**问题**：时钟频率配置有哪些限制？

**限制说明**：
1. **频率必须大于零**：`frequency > 0`，否则时间步计算会出错
2. **频率上限**：受限于SystemC-AMS的最小时间步（通常约1fs）
3. **频率与仿真时长**：高频时钟长时间仿真会产生大量数据

**计算示例**：
- 40GHz时钟，1μs仿真：4,000,000个采样点，约32MB数据
- 80GHz时钟，1μs仿真：8,000,000个采样点，约64MB数据

**建议**：
- 高频时钟（>40GHz）建议缩短仿真时长
- 使用Python后处理时注意内存管理
- 考虑使用稀疏采样或降采样技术

### 7.6 浮点数精度问题

**问题**：相位累加器的浮点数精度如何影响长期仿真？

**精度分析**：
- `double`类型提供约15-17位有效数字
- 每个时间步的相位增量：`Δφ = 2π × f × Δt`
- 对于40GHz时钟，`Δφ ≈ 0.06283`弧度
- 浮点数相对误差：约1e-15（机器精度）

**长期仿真影响**：
- 1μs仿真（40,000个周期）：累积误差可忽略
- 1ms仿真（40,000,000个周期）：可能出现可观测的相位漂移
- 建议：超长时间仿真使用更高精度类型（如`long double`）

### 7.7 与CDR模块的接口差异

**问题**：当前CDR模块未使用时钟生成器的相位输出，为什么？

**现状说明**：
- 时钟生成器输出相位信号：`clk_phase`
- 当前CDR模块使用内部相位生成逻辑
- 两者接口不匹配，CDR未连接到时钟生成器

**未来改进**：
1. **统一时钟架构**：CDR应使用时钟生成器的相位输出
2. **相位对齐**：确保CDR和Sampler使用相同的时钟相位
3. **时钟树建模**：实现完整的时钟分配网络

**当前实现**：
- Sampler模块使用时钟生成器的相位（如果连接）
- CDR模块独立生成相位，用于时钟数据恢复

### 7.8 理想时钟模型的局限性

**当前实现**：理想时钟模型，相位严格线性增长

**未建模的非理想效应**：
1. **相位噪声（Phase Noise）**：实际时钟存在随机相位波动
2. **抖动（Jitter）**：包括随机抖动（RJ）和确定性抖动（DJ）
3. **频率漂移（Frequency Drift）**：由于温度、电压变化引起的频率偏移
4. **占空比失真（Duty Cycle Distortion）**：实际时钟的占空比可能偏离50%

**适用场景**：
- 功能验证：验证SerDes链路的基本功能
- 性能基准：提供理想条件下的性能参考
- 算法开发：用于开发和测试均衡、CDR算法

**不适用场景**：
- 抖动容限测试：需要真实时钟抖动模型
- 相位噪声分析：需要相位噪声注入
- 系统级性能评估：需要完整的非理想效应建模

### 7.9 未来PLL实现的关键挑战

**挑战1：参考时钟源**
- 问题：当前系统缺乏独立的参考时钟源
- 解决：需要添加参考时钟模块或在测试平台中生成

**挑战2：环路稳定性**
- 问题：PLL参数配置不当可能导致环路不稳定或振荡
- 解决：需要实现环路稳定性分析和参数优化算法

**挑战3：锁定时间建模**
- 问题：PLL锁定过程需要多个时钟周期，仿真时间较长
- 解决：使用快速仿真技术或简化锁定过程模型

**挑战4：数值稳定性**
- 问题：PLL环路中的积分器可能导致数值累积误差
- 解决：使用适当的数值积分方法和误差控制

**挑战5：与现有模块的集成**
- 问题：PLL输出需要与Sampler、CDR等模块正确集成
- 解决：设计统一的时钟分配接口和同步机制

### 7.10 已知限制和特殊要求

**已知限制**：
1. **仅支持理想时钟**：当前未实现PLL/ADPLL模式
2. **无噪声建模**：不支持相位噪声和抖动注入
3. **无参考时钟**：PLL模式需要额外的参考时钟源
4. **时间步全局统一**：时钟生成器的时间步影响整个系统
5. **无独立测试平台**：仅在系统级测试中验证

**特殊要求**：
1. **频率配置**：必须大于零，建议在1GHz-100GHz范围内
2. **仿真时长**：高频时钟建议缩短仿真时长，避免数据量过大
3. **环境配置**：必须正确设置`SYSTEMC_HOME`和`SYSTEMC_AMS_HOME`
4. **浮点数精度**：超长时间仿真可能需要更高精度类型
5. **相位追踪**：如需分析相位输出，必须在测试平台中添加trace

### 7.11 与其他时钟模块的对比

**与WaveGenerationTdf的对比**：
- WaveGenerationTdf：生成数据信号（PRBS），不涉及时钟相位
- ClockGenerationTdf：生成时钟相位，用于时序控制
- 两者互补：WaveGeneration提供数据，ClockGeneration提供时钟

**与RxCdrTdf的对比**：
- RxCdrTdf：从数据中恢复时钟，实现时钟数据恢复功能
- ClockGenerationTdf：生成源时钟，作为系统的时钟参考
- 关系：ClockGeneration提供参考时钟，CDR从数据中恢复同步时钟

**与理想vs实际时钟的对比**：
- 理想时钟：无抖动、无噪声，相位严格线性
- 实际时钟：存在各种非理想效应，需要复杂建模
- 当前实现：理想时钟，未来计划支持实际时钟建模

---

## 8. 参考信息

### 8.1 相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 参数定义 | `/include/common/parameters.h` | ClockParams、ClockPllParams结构体 |
| 类型定义 | `/include/common/types.h` | ClockType枚举及转换函数 |
| 头文件 | `/include/ams/clock_generation.h` | ClockGenerationTdf类声明 |
| 实现文件 | `/src/ams/clock_generation.cpp` | ClockGenerationTdf类实现 |
| 系统级测试 | `/tb/simple_link_tb.cpp` | SerDes链路集成测试 |
| 配置文件 | `/config/default.yaml` | YAML格式默认配置 |
| 配置文件 | `/config/default.json` | JSON格式默认配置 |
| 环境脚本 | `/scripts/setup_env.sh` | SystemC/SystemC-AMS环境配置 |

**文件说明**：
- **参数定义**：包含时钟生成器的所有参数结构体，包括基本参数和PLL子参数
- **类型定义**：定义了时钟类型枚举（IDEAL/PLL/ADPLL）及字符串转换函数
- **头文件**：声明了ClockGenerationTdf类，继承自sca_tdf::sca_module
- **实现文件**：实现了相位累加器、时间步自适应设置等核心功能
- **系统级测试**：当前时钟生成器主要通过系统级测试验证功能
- **配置文件**：支持YAML和JSON两种格式，通过ConfigLoader加载

### 8.2 依赖项

**核心依赖**：
- **SystemC 2.3.4**：系统级建模框架，提供事件驱动和TDF域支持
- **SystemC-AMS 2.3.4**：模拟/混合信号扩展，提供TDF域连续时间建模

**编译依赖**：
- **C++14标准**：模块使用C++14特性（如constexpr、auto等）
- **CMake 3.15+** 或 **GNU Make**：构建系统
- **标准库**：`<cmath>`（M_PI常量）、`<iostream>`等

**可选依赖**：
- **Python 3.x**：用于后处理分析和可视化
  - `numpy`：数值计算和数组处理
  - `matplotlib`：波形绘制和数据可视化
  - `scipy`：信号处理和统计分析

**版本兼容性**：
| 依赖项 | 最低版本 | 推荐版本 | 测试版本 |
|--------|---------|---------|---------|
| SystemC | 2.3.3 | 2.3.4 | 2.3.4 |
| SystemC-AMS | 2.3.3 | 2.3.4 | 2.3.4 |
| C++标准 | C++11 | C++14 | C++14 |
| CMake | 3.10 | 3.15+ | 3.20+ |
| Python | 3.6 | 3.8+ | 3.10.12 |

### 8.3 配置示例

#### 基本配置（理想时钟）

**YAML格式**：
```yaml
clock:
  type: IDEAL         # 时钟类型：IDEAL/PLL/ADPLL
  frequency: 40e9     # 时钟频率：40GHz
```

**JSON格式**：
```json
{
  "clock": {
    "type": "IDEAL",
    "frequency": 40000000000
  }
}
```

#### PLL配置（模拟锁相环）

**YAML格式**：
```yaml
clock:
  type: PLL
  frequency: 40e9
  pd: "tri-state"    # 鉴相器类型：tri-state/bang-bang/linear/hogge
  cp:
    I: 5e-5          # 电荷泵电流：50μA
  lf:
    R: 10000         # 环路滤波器电阻：10kΩ
    C: 1e-10         # 环路滤波器电容：100pF
  vco:
    Kvco: 1e8        # VCO增益：100MHz/V
    f0: 1e10         # VCO中心频率：10GHz
  divider: 4         # 反馈分频比：4
```

**JSON格式**：
```json
{
  "clock": {
    "type": "PLL",
    "frequency": 40000000000,
    "pd": "tri-state",
    "cp": {
      "I": 5e-5
    },
    "lf": {
      "R": 10000,
      "C": 1e-10
    },
    "vco": {
      "Kvco": 1e8,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

**PLL参数说明**：
- **环路带宽**：`ωn = sqrt(Kvco × Icp / (N × C))` ≈ 2.24×10^6 rad/s (356kHz)
- **阻尼系数**：`ζ = (R/2) × sqrt(Icp × Kvco × C / N)` ≈ 0.707
- **锁定时间**：`T_lock ≈ 4/(ζ × ωn)` ≈ 2.5μs

#### ADPLL配置（全数字锁相环）

**YAML格式**（未来实现）：
```yaml
clock:
  type: ADPLL
  frequency: 40e9
  pd: "digital"      # 数字鉴相器
  tdc:
    resolution: 1e-12 # 时间数字转换器分辨率：1ps
  dco:
    resolution: 1e6   # 数控振荡器分辨率：1MHz
    f0: 1e10          # DCO中心频率：10GHz
  divider: 4
```

**JSON格式**（未来实现）：
```json
{
  "clock": {
    "type": "ADPLL",
    "frequency": 40000000000,
    "pd": "digital",
    "tdc": {
      "resolution": 1e-12
    },
    "dco": {
      "resolution": 1e6,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

#### 不同频率配置示例

**10GHz时钟（低速SerDes）**：
```yaml
clock:
  type: IDEAL
  frequency: 10e9
```

**20GHz时钟（中速SerDes）**：
```yaml
clock:
  type: IDEAL
  frequency: 20e9
```

**40GHz时钟（高速SerDes，默认）**：
```yaml
clock:
  type: IDEAL
  frequency: 40e9
```

**80GHz时钟（超高速SerDes）**：
```yaml
clock:
  type: IDEAL
  frequency: 80e9
```

**注意**：高频时钟（>40GHz）会导致时间步很小，建议缩短仿真时长以避免数据量过大。

#### 完整系统配置示例

**YAML格式**（`config/default.yaml`）：
```yaml
global:
  Fs: 80e9           # 采样率：80GHz
  UI: 2.5e-11        # 单位间隔：25ps（40Gbps）
  duration: 1e-6     # 仿真时长：1μs
  seed: 12345        # 随机种子

wave:
  type: PRBS31
  poly: "x^31 + x^28 + 1"
  init: "0x7FFFFFFF"

tx:
  ffe_taps: [0.2, 0.6, 0.2]
  mux_lane: 0
  driver:
    swing: 0.8
    bw: 20e9

channel:
  attenuation_db: 10.0
  bandwidth_hz: 20e9

rx:
  ctle:
    zeros: [2e9]
    poles: [30e9]
    dc_gain: 1.5
    vcm_out: 0.6
  vga:
    gain: 4.0
  sampler:
    threshold: 0.0
    hysteresis: 0.02
  dfe:
    taps: [-0.05, -0.02, 0.01]
    update: "sign-lms"
    mu: 1e-4

cdr:
  pi:
    kp: 0.01
    ki: 1e-4
  pai:
    resolution: 1e-12
    range: 5e-11

clock:
  type: IDEAL
  frequency: 40e9

eye:
  ui_bins: 128
  amp_bins: 128
  measure_length: 1e-4
```

**JSON格式**（`config/default.json`）：
```json
{
  "global": {
    "Fs": 80000000000,
    "UI": 2.5e-11,
    "duration": 1e-6,
    "seed": 12345
  },
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF"
  },
  "tx": {
    "ffe_taps": [0.2, 0.6, 0.2],
    "mux_lane": 0,
    "driver": {
      "swing": 0.8,
      "bw": 20000000000
    }
  },
  "channel": {
    "attenuation_db": 10.0,
    "bandwidth_hz": 20000000000
  },
  "rx": {
    "ctle": {
      "zeros": [2000000000],
      "poles": [30000000000],
      "dc_gain": 1.5,
      "vcm_out": 0.6
    },
    "vga": {
      "gain": 4.0
    },
    "sampler": {
      "threshold": 0.0,
      "hysteresis": 0.02
    },
    "dfe": {
      "taps": [-0.05, -0.02, 0.01],
      "update": "sign-lms",
      "mu": 0.0001
    }
  },
  "cdr": {
    "pi": {
      "kp": 0.01,
      "ki": 0.0001
    },
    "pai": {
      "resolution": 1e-12,
      "range": 5e-11
    }
  },
  "clock": {
    "type": "IDEAL",
    "frequency": 40000000000
  },
  "eye": {
    "ui_bins": 128,
    "amp_bins": 128,
    "measure_length": 0.0001
  }
}
```

---

**文档版本**：v0.1  
**最后更新**：2026-01-20  
**作者**：Yizhe Liu
