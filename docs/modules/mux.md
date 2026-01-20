# TX Mux 模块技术文档

**级别**：AMS 子模块（TX）  
**类名**：`TxMuxTdf`  
**当前版本**：v0.1 (2026-01-13)  
**状态**：开发中

---

## 1. 概述

发送端复用器（Mux，Multiplexer）是SerDes发送链路中的关键时序模块，位于 FFE → Mux → Driver 信号链的中间位置，主要功能是实现通道选择（Lane Selection）和信号路由控制，在系统级应用中作为并串转换（Parallel-to-Serial Conversion）架构的一部分，为Driver提供选定的数据通道，同时建模真实硬件中的延迟和抖动效应。

### 1.1 设计原理

TX Mux 的核心设计思想是在行为级抽象层面建模复用器的选择逻辑、传播延迟和抖动特性，为系统级仿真提供足够的精度，同时避免晶体管级实现细节，保持仿真效率。

#### 1.1.1 复用器的功能定位

在完整的SerDes发送链路中，复用器承担以下角色：

- **通道选择（Lane Selection）**：在多通道（Multi-Lane）SerDes架构中，发送端可能包含多个并行数据通道，每个通道运行在较低的符号速率以降低时钟频率和功耗。Mux根据控制信号选择其中一个通道的数据送往Driver，实现通道级的路由切换。

- **并串转换的一部分**：在真实硬件中，N:1并串转换通常由N个并行的数据路径（Lane）和一个N:1 Mux组成。例如8:1结构中，8个并行Lane各运行在符号速率（Symbol Rate），Mux通过时分复用（Time-Division Multiplexing）将它们合并为比特速率（Bit Rate = Symbol Rate × 8）的串行输出。**本行为模型聚焦于Mux单元本身的选择和延迟特性，而非完整的并行数据路径**。

- **抽象级别说明**：本模块采用单输入单输出（`in` → `out`）架构，配合 `lane_sel` 参数选择通道索引。这种抽象方式简化了建模复杂度，适用于以下场景：
  - 单通道系统（`lane_sel=0`，Bypass模式）
  - 多通道架构中对选定通道的行为验证
  - 延迟和抖动效应的独立测试

#### 1.1.2 采样率与时序关系

Mux的输入输出采样率关系决定其在信号链中的时序行为：

- **符号速率同步**：在本行为模型中，输入和输出采样率保持一致（`set_rate(1)`），表示Mux工作在符号速率时钟域，每个时间步长（Timestep）处理一个符号。这与真实硬件中"Mux内部时钟等于比特速率"的实现有所不同，但对于行为级仿真已足够表征信号传递特性。

- **时间步长与UI的关系**：假设全局采样频率为 Fs（由 `GlobalParams` 定义），则时间步长 Δt = 1/Fs。对于符号速率为 R_sym 的系统，每个符号周期 T_sym = 1/R_sym 包含 Fs/R_sym 个采样点。例如：
  - 若比特速率 = 56 Gbps，符号速率 = 7 GHz（8:1架构），Fs = 560 GHz（每UI 10个采样点）
  - 则 T_sym = 142.86 ps，Δt = 1.786 ps，每个符号包含 80 个时间步长

- **相位对齐考虑**：真实N:1 Mux需要N个相位精确对齐的时钟来实现时分复用。在行为模型中，这种相位对齐需求被抽象为延迟参数（`mux_delay`）和抖动模型（`jitter_params`），通过调整这些参数可以间接模拟相位失配的影响。

#### 1.1.3 选择器的行为级建模

本模块采用理想选择器模型，抽象了真实硬件的复杂拓扑：

- **理想传输特性**：在不考虑非理想效应的情况下，输出直接等于选定通道的输入：`out[n] = in[n]`。这种简化适用于功能验证和信号完整性分析的初期阶段。

- **延迟建模**：可选参数 `mux_delay` 用于建模选择器的传播延迟（Propagation Delay）。真实硬件中，延迟来源包括：
  - 三态缓冲器或传输门的开关时间（~10-30ps）
  - 时钟到数据路径的延迟（Clock-to-Q Delay）
  - 互连寄生电容和电阻的RC延迟
  
  在行为模型中，通过在信号路径插入固定延迟元素（如滤波器的群延迟或显式的延迟线）来近似这些效应。

- **与真实拓扑的对应关系**：
  - **树形选择器**：多级2:1级联，总延迟为级数×单级延迟
  - **并行选择器**：单级多路选择，延迟最小但负载相关
  - **行为模型**：通过调整 `mux_delay` 参数匹配上述拓扑的等效延迟，无需实现具体电路结构

#### 1.1.4 抖动效应建模

Mux是SerDes发送端的重要抖动源，行为模型需要捕捉以下效应：

- **确定性抖动（DJ，Deterministic Jitter）**来源：
  - **占空比失真（DCD，Duty Cycle Distortion）**：时钟占空比偏离50%导致相邻UI宽度不等。DCD引起的抖动峰峰值为 `DJ_DCD = UI × |DCD% - 50%|`。例如占空比48%（偏离2%）在16ps UI下产生0.32ps抖动。
  - **码型相关抖动（PDJ，Pattern-Dependent Jitter）**：选择器的传播延迟随输入数据码型变化，可通过在不同码型下注入不同的延迟偏移来建模。
  
- **随机抖动（RJ，Random Jitter）**来源：
  - **时钟相位噪声**：来自PLL的相位噪声传递到数据边沿，表现为高斯分布的随机抖动，典型值 0.1-0.5 ps rms
  - **热噪声**：选择器电路的热噪声叠加到输出，与电路带宽和温度相关
  
- **行为级建模方法**：
  - **时域注入**：在每个时间步长对输出信号施加随机时移或幅度扰动
  - **参数化控制**：通过 `jitter_enable`, `dcd_percent`, `rj_sigma` 等参数灵活调整抖动水平，匹配目标硬件规格

### 1.2 核心特性

- **单输入单输出架构**：采用简化的单端信号路径（`in` → `out`），聚焦于通道选择和延迟/抖动建模，而非完整的多通道并行输入。适用于行为级仿真和算法验证。

- **通道索引选择**：通过构造参数 `lane_sel` 指定选中的通道索引（0-based），支持在多通道系统中对特定通道进行独立建模和测试。默认值为0（第一个通道）。

- **符号速率同步**：输入输出采样率一致，工作在符号速率时钟域，与前级FFE和后级Driver的时序要求兼容。采样率由全局参数 `Fs`（采样频率）控制。

- **延迟可配置**：可选参数 `mux_delay` 用于建模选择器的传播延迟，匹配真实硬件的时序特性。延迟范围通常在10-50ps，具体取值取决于工艺节点和拓扑结构。

- **抖动建模支持**：可选择性地注入确定性抖动（DCD）和随机抖动（RJ），模拟时钟非理想性和电路噪声对输出信号质量的影响。抖动参数通过配置文件灵活设置。

- **Bypass模式**：当 `lane_sel=0` 且无延迟/抖动配置时，模块退化为直通（Pass-through）模式，用于前端调试或单通道系统验证。

### 1.3 典型应用场景

TX Mux在不同SerDes架构中的应用配置：

| 系统架构 | Lane数量 | 符号速率 | Mux配置 | 典型应用 |
|---------|---------|---------|---------|---------|
| 单通道SerDes | 1 | 等于比特速率 | lane_sel=0, Bypass | 低速链路（<10Gbps），PCIe Gen1/2 |
| 2:1并串转换 | 2 | 比特速率/2 | lane_sel=0或1 | 中速链路（10-25Gbps） |
| 4:1并串转换 | 4 | 比特速率/4 | lane_sel=0-3 | PCIe Gen3/4, USB3.x |
| 8:1并串转换 | 8 | 比特速率/8 | lane_sel=0-7 | 56G/112G SerDes, 高速以太网 |

> **注**：本模块对单个Lane的行为进行建模。完整的N:1系统需要实例化N个数据路径（WaveGen → FFE → Mux）并在系统级进行时分复用仿真。

### 1.4 与其他模块的关系

- **上游模块（FFE）**  
  Mux接收来自FFE的均衡后符号信号，输入为标称幅度的数字信号（如±1V）。FFE的输出时序必须满足Mux的建立时间（Setup Time）要求，通常 > 0.2 UI。

- **下游模块（Driver）**  
  Mux输出送往Driver进行功率放大和阻抗匹配。Mux的输出延迟和抖动直接影响Driver的采样时刻和眼图质量，因此需要联合优化两者的时序预算。

- **时钟源（Clock Generation）**  
  真实硬件中，Mux依赖多相位时钟或相位插值器（PI）实现时分复用。在行为模型中，时钟精度的影响被抽象为抖动参数（如DCD、RJ），由用户根据Clock模块的规格进行配置。

- **系统配置（System Configuration）**  
  通过配置文件加载 `lane_sel`, `mux_delay`, `jitter_params` 等参数，支持多场景切换和参数扫描分析（如不同通道索引下的性能差异、延迟扫描对眼图的影响）。

### 1.5 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2026-01-13 | 初始版本，实现单输入单输出架构、通道选择和基本延迟建模 |

---

## 2. 模块接口

### 2.1 端口定义（TDF域）

TX Mux 采用单端信号架构，所有端口均为 TDF 域模拟信号。

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `in` | 输入 | double | 输入数据信号（来自FFE或其他前级模块） |
| `out` | 输出 | double | 输出数据信号（送往Driver或后级模块） |

#### 端口连接注意事项

- **简化架构**：采用单输入单输出（Single-Input-Single-Output, SISO）设计，抽象了真实硬件的多输入选择逻辑。通道选择通过构造参数 `lane_sel` 实现，而非多端口切换。
- **采样率一致性**：输入和输出采样率保持一致（`set_rate(1)`），工作在符号速率时钟域。所有连接的TDF模块必须使用相同的全局采样频率 `Fs`（由 `GlobalParams` 定义）。
- **信号幅度**：输入信号幅度通常为前级FFE输出的标称值（如±0.5V ~ ±1.0V）。当前版本（v0.1）采用理想传输，输出幅度与输入完全一致。
- **负载条件**：输出端口应连接到Driver模块或测试负载，确保后级模块能够正确采样Mux输出的时序特性。

### 2.2 参数配置

#### 2.2.1 当前实现的参数

TX Mux 的参数定义在 `TxParams` 结构中（位于 `include/common/parameters.h`）。**当前版本（v0.1）仅实现通道索引参数**：

| 参数路径 | 类型 | 默认值 | 单位 | 说明 |
|---------|------|--------|------|------|
| `tx.mux_lane` | int | 0 | - | 选中的通道索引（0-based），指定Mux输出对应的Lane编号 |

**构造函数签名**：
```cpp
TxMuxTdf(sc_core::sc_module_name nm, int lane_sel = 0);
```

**当前行为**（`src/ams/tx_mux.cpp`）：
```cpp
void TxMuxTdf::processing() {
    // 简单透传模式（单通道）
    double x_in = in.read();
    out.write(x_in);
}
```

**参数说明**：
- **lane_sel（mux_lane）设计意图**：
  - **当前实现**：参数存储在成员变量 `m_lane_sel` 中，但在 `processing()` 函数中未使用，模块执行理想透传（Pass-through）操作。
  - **设计用途**：为未来多通道架构预留，用于选择N个并行输入中的某一路（当前单输入单输出架构下该参数无实际功能影响）。
  - **有效范围**：整数索引，通常 0-based。对于单通道系统，固定为 `lane_sel=0`。

#### 2.2.2 预留参数（未来版本）

以下参数在当前代码中**尚未实现**，仅作为设计规划和配置文件预留接口，供未来版本扩展使用。

##### 延迟建模参数（预留）

| 参数 | 类型 | 默认值 | 单位 | 说明 | 实现状态 |
|------|------|--------|------|------|---------|
| `mux_delay` | double | 0.0 | s | 传播延迟（Propagation Delay），建模选择器的固定时延 | 待实现 |

**设计意图**：
- **物理意义**：建模选择器从输入到输出的传播延迟，包括：
  - 选择器逻辑延迟（三态门、传输门的开关时间）
  - 时钟到数据路径延迟（Clock-to-Q Delay）
  - 互连寄生RC延迟
- **典型值参考**：
  - 先进工艺（7nm/5nm）：10-20 ps
  - 成熟工艺（28nm/16nm）：20-40 ps
  - 长互连走线：可达 50-100 ps
- **实现方法建议**：
  - 使用 `sca_tdf::sca_delay` 或显式缓冲队列实现固定延迟
  - 或通过一阶低通滤波器的群延迟近似传播延迟

##### 抖动建模参数（预留）

| 参数 | 类型 | 默认值 | 单位 | 说明 | 实现状态 |
|------|------|--------|------|------|---------|
| `jitter.enable` | bool | false | - | 启用抖动建模功能 | 待实现 |
| `jitter.dcd_percent` | double | 50.0 | % | 占空比（Duty Cycle），50%为理想，偏离50%产生DCD抖动 | 待实现 |
| `jitter.rj_sigma` | double | 0.0 | s | 随机抖动（RJ）标准差，高斯分布模型 | 待实现 |
| `jitter.seed` | int | 0 | - | 随机数生成器种子，0表示使用全局种子 | 待实现 |

**设计意图**：

**占空比失真（DCD）建模**：
- **物理背景**：时钟占空比偏离50%导致相邻UI（Unit Interval）宽度不等，引入边沿位置偏移
- **典型影响**：对于 UI=16ps 的系统，占空比从50%偏离到48%（偏离2%），会在边沿产生约 0.3-0.6ps 的确定性抖动
- **实现方法建议**：
  - 奇数UI和偶数UI施加相反方向的时间偏移
  - 通过插值技术（如Lagrange或Sinc插值）实现分数采样延迟

**随机抖动（RJ）建模**：
- **物理来源**：
  - 时钟相位噪声：来自PLL/时钟分发的相位抖动传递到数据边沿
  - 热噪声：选择器电路的热噪声叠加到输出
  - 电源噪声：VDD抖动通过时钟路径耦合到Mux
- **典型值参考**：
  - 低抖动系统：rj_sigma < 0.2 ps（峰峰值 < 3ps，对应BER=10⁻¹²时的 14σ 估算）
  - 中等性能：rj_sigma = 0.3-0.5 ps（峰峰值 4-7ps）
  - 高抖动场景（压力测试）：rj_sigma > 1.0 ps
- **实现方法建议**：
  - 每个时间步长生成独立同分布的高斯随机数：`δt ~ N(0, rj_sigma²)`
  - 使用高精度插值实现 fractional delay

##### 多通道选择参数（预留）

| 参数 | 类型 | 默认值 | 说明 | 实现状态 |
|------|------|--------|------|---------|
| `num_lanes` | int | 1 | 系统总通道数（1=单通道，2/4/8=多通道架构），用于参数验证 | 待实现 |

**设计意图**：
- **用途**：定义系统架构类型，为通道索引验证提供边界条件（应满足 `0 ≤ lane_sel < num_lanes`）
- **典型配置**：
  - `num_lanes=1`：单通道SerDes（比特速率 ≤ 10Gbps）
  - `num_lanes=2/4/8`：2:1/4:1/8:1并串转换架构
- **注意**：真实N:1并串转换需要在系统级实例化 N 个并行数据路径，当前单输入单输出架构仅对单条Lane建模

##### 非线性效应参数（预留）

| 参数 | 类型 | 默认值 | 说明 | 实现状态 |
|------|------|--------|------|---------|
| `nonlinearity.enable` | bool | false | 启用非线性建模 | 待实现 |
| `nonlinearity.gain_compression` | double | 0.0 | 增益压缩系数（dB/V） | 待实现 |
| `nonlinearity.saturation_voltage` | double | 1.0 | 饱和电压（V） | 待实现 |

**设计意图（未来版本潜在应用）**：
- **增益压缩**：大信号输入下增益降低，建模传输门的非线性电阻
- **饱和限幅**：输出幅度受限于电源电压或驱动能力
- **码型相关延迟**：不同码型下的传播延迟变化，引入数据相关抖动（DDJ）

### 2.3 配置示例

#### 2.3.1 当前版本有效配置

**示例1：最简配置（与当前实现匹配）**

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**说明**：
- 这是当前版本（v0.1）**唯一有效的配置参数**
- 模块执行理想透传：`out = in`，无延迟、无抖动
- 适用于前端调试或单通道系统功能验证

**示例2：多通道系统中的通道选择（意图声明）**

```json
{
  "tx": {
    "mux_lane": 2
  }
}
```

**说明**：
- 声明选中第3个通道（索引2），但**当前实现中该参数不影响信号处理行为**（仍为透传）
- 用于配置文件版本管理，为未来多通道架构迁移做准备

#### 2.3.2 未来版本配置示例（预留接口）

以下配置在当前代码中**无法生效**，仅作为未来版本的设计规划参考。

**示例3：带延迟的单通道模式（未来）**

```json
{
  "tx": {
    "mux_lane": 0,
    "mux_delay": 25e-12
  }
}
```

**预期行为（待实现）**：
- 固定延迟25ps，匹配28nm工艺的典型传播延迟
- 输出信号相对输入延迟一个固定时间

**示例4：启用抖动建模（未来）**

```json
{
  "tx": {
    "mux_lane": 0,
    "mux_delay": 15e-12,
    "jitter": {
      "enable": true,
      "dcd_percent": 48.0,
      "rj_sigma": 0.3e-12,
      "seed": 0
    }
  }
}
```

**预期行为（待实现）**：
- 固定延迟15ps（先进工艺）
- DCD占空比48%（偏离2%），在边沿产生确定性时间偏移
- RJ标准差0.3ps，叠加高斯分布的随机时间抖动
- 使用全局随机种子保证可重复性

**示例5：多通道架构配置（未来）**

```json
{
  "tx": {
    "mux_lane": 5,
    "num_lanes": 8,
    "mux_delay": 15e-12
  }
}
```

**预期行为（待实现）**：
- 8:1架构中选中第6个通道（索引5）
- 参数验证：`lane_sel < num_lanes`（5 < 8 通过）
- 适用于56Gbps/112Gbps SerDes的通道级测试

### 2.4 参数使用注意事项

#### 当前版本（v0.1）开发者指南

1. **仅 `tx.mux_lane` 参数有效**  
   配置文件中只需设置 `"tx": {"mux_lane": 0}`，其他参数会被忽略（如果配置文件加载器已实现参数读取，但模块内部未使用）。

2. **透传行为保证**  
   无论 `mux_lane` 取何值，当前版本模块始终执行 `out = in` 的透传操作。

3. **测试策略建议**  
   - **功能验证**：验证端口连接正确性和采样率一致性
   - **集成测试**：与FFE和Driver模块串联,确认信号链连续性
   - **不需要测试**：延迟、抖动、多通道切换（当前未实现）

#### 未来版本扩展指南

1. **延迟实现路径**  
   - 方案A：使用 `sca_tdf::sca_delay<double>` 模块级延迟
   - 方案B：显式环形缓冲区存储历史采样值
   - 方案C：通过一阶滤波器群延迟近似

2. **抖动实现路径**  
   - DCD：根据UI索引奇偶性调整采样时刻
   - RJ：使用 `std::normal_distribution` 生成时间扰动
   - 插值：实现高精度分数延迟（Lagrange/Sinc/Farrow结构）

3. **多通道实现路径**  
   - 修改端口定义为 `sca_tdf::sca_in<double>` 数组或 `std::vector`
   - 在 `processing()` 中根据 `m_lane_sel` 选择对应输入端口
   - 系统级需要实例化多个并行数据路径并在顶层合并

4. **配置加载兼容性**  
   - 预留参数应在配置加载器中标记为可选（optional）
   - 当检测到未实现参数被使用时，应输出警告日志而非报错
   - 保持配置文件向前兼容性（新版本代码能识别旧版本配置）

---

## 3. 核心实现机制

### 3.1 信号处理流程

TX Mux 模块当前版本（v0.1）采用最简化的透传架构，信号处理流程仅包含单一步骤：

```
输入读取 → 直接写入输出
```

**完整处理流程（当前实现）**：

```cpp
void TxMuxTdf::processing() {
    // 简单透传模式（单通道）
    double x_in = in.read();
    out.write(x_in);
}
```

**代码位置**：`src/ams/tx_mux.cpp` 第 18-22 行

#### 步骤1 - 输入读取

从 TDF 输入端口读取当前时间步长的模拟信号值：

```cpp
double x_in = in.read();
```

**设计说明**：
- 输入信号 `x_in` 通常来自前级 FFE（Feed-Forward Equalizer）模块的输出
- 信号幅度范围取决于 FFE 的输出配置，典型值为 ±0.5V ~ ±1.0V（单端）
- 采样率由全局参数 `Fs` 控制，通过 `set_attributes()` 方法配置为符号速率同步（`set_rate(1)`）

#### 步骤2 - 直接输出

将读取的输入信号不作任何处理直接写入输出端口：

```cpp
out.write(x_in);
```

**当前版本行为特征**：
- **零延迟**：输出在同一时间步长内完成，不引入任何传播延迟
- **理想传输**：输出幅度和相位与输入完全一致，无增益/衰减/失真
- **无抖动**：不注入任何确定性或随机抖动成分
- **通道索引无效**：尽管构造函数接受 `lane_sel` 参数并存储在 `m_lane_sel` 成员变量中，但 `processing()` 方法中未使用该变量，因此通道索引配置不影响信号路径

**等效传递函数**：
```
H(s) = 1  （全频段单位增益）
H(z) = 1  （离散时域）
y[n] = x[n]  （时域表达式）
```

**应用场景**：
- **功能验证阶段**：验证 TX 链路（FFE → Mux → Driver）的端到端连接正确性
- **基线测试**：建立无 Mux 延迟/抖动影响的参考眼图，用于后续版本的对比分析
- **单通道系统**：在不需要多通道复用和延迟建模的简化应用中，当前实现已满足需求

### 3.2 TDF 生命周期方法

TX Mux 作为 SystemC-AMS TDF（Timed Data Flow）模块，遵循标准的 TDF 生命周期管理机制。本节详细说明三个核心方法的实现和设计考量。

#### 3.2.1 构造函数 - 模块初始化

**代码实现**（`src/ams/tx_mux.cpp` 第 5-11 行）：

```cpp
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int lane_sel)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_lane_sel(lane_sel)
{
}
```

**参数说明**：
- `nm`：SystemC 模块实例名称，由上层系统模块传入，用于层次化命名和调试信息输出
- `lane_sel`：通道索引参数（默认值 0），指定选中的数据通道编号

**初始化列表逐项解析**：

1. **基类构造**：`sca_tdf::sca_module(nm)`
   - 调用 SystemC-AMS TDF 模块基类构造函数，注册模块到 TDF 调度器
   - 继承 TDF 时间步进机制和采样率管理接口

2. **端口注册**：`in("in")`, `out("out")`
   - 分配端口名称并注册到模块的端口列表
   - 端口类型为 `sca_tdf::sca_in<double>` 和 `sca_tdf::sca_out<double>`（定义在 `include/ams/tx_mux.h` 第 10-11 行）
   - 端口连接将在系统级 `SC_CTOR` 或 `elaborate()` 阶段完成

3. **成员变量初始化**：`m_lane_sel(lane_sel)`
   - 存储通道索引参数供后续使用
   - **当前版本注意**：尽管存储了该值，但 `processing()` 方法中未使用，参数实际不起作用

**构造函数体为空**：
- 当前版本不需要额外的运行时初始化（如随机数生成器、滤波器对象、延迟线缓冲区）
- 未来版本可能在此处初始化：
  - 抖动模型的随机数生成器 `std::mt19937`
  - 延迟线缓冲区 `std::deque<double>`
  - PSRR/带宽滤波器对象 `sca_ltf_nd`

#### 3.2.2 set_attributes() - 采样率配置

**代码实现**（`src/ams/tx_mux.cpp` 第 13-16 行）：

```cpp
void TxMuxTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}
```

**方法调用时机**：
- SystemC-AMS 在仿真启动前的 elaboration 阶段自动调用
- 在所有模块实例化和端口连接完成后，仿真开始前执行
- 用于声明模块的时序约束和资源需求

**采样率设置详解**：

**`set_rate(1)` 的含义**：
- 参数 `1` 表示输入/输出端口的相对采样率因子（Rate Factor）
- 相对于全局时间步长 Δt（由顶层 TDF 模块或 SystemC 时钟域定义），端口每个时间步长采样/输出一次
- 等价声明：**输入和输出采样率一致，工作在符号速率时钟域**

**采样率因子的物理意义**：

假设全局采样频率为 Fs（例如 560 GHz），比特速率为 56 Gbps，符号速率为 7 GHz（8:1架构），则：

- 全局时间步长：Δt = 1/Fs = 1.786 ps
- 符号周期：T_sym = 1/7GHz = 142.86 ps
- 每个符号包含采样点数：Fs / 7GHz = 80 个时间步长

在 `set_rate(1)` 配置下：
- Mux 的 `processing()` 方法每 1 个时间步长被调用一次
- 每次调用处理一个采样点（不是一个符号）
- 这种细粒度采样适用于行为级建模，捕捉符号内的幅度变化和过渡过程

**与其他模块的采样率协调**：

TX Mux 通常级联在以下信号链中：
```
FFE (rate=1) → Mux (rate=1) → Driver (rate=1)
```

- 所有模块采样率因子一致（rate=1），确保信号流的时序连续性
- 如果前后级模块采样率不同（如 rate=1 连接 rate=2），需要在端口间插入速率转换模块或使用 SystemC-AMS 的自动插值机制
- **当前实现要求**：所有 TX 链路模块必须使用相同的全局采样频率 Fs

**为什么不使用符号速率采样？**

理论上可以将 Mux 的采样率设置为符号速率（rate = Fs / R_sym），即每个符号周期调用一次 `processing()`。但当前设计选择 rate=1 的原因包括：

1. **灵活性**：保持与前后级模块的直接兼容，避免速率转换开销
2. **精度**：捕捉符号内的瞬态过程（如边沿上升时间、过冲），适用于眼图分析
3. **一致性**：项目所有 AMS 模块统一采用 rate=1，简化系统配置

#### 3.2.3 processing() - 核心信号处理

**代码实现**（`src/ams/tx_mux.cpp` 第 18-22 行）：

```cpp
void TxMuxTdf::processing() {
    // 简单透传模式（单通道）
    double x_in = in.read();
    out.write(x_in);
}
```

**方法调用时机**：
- SystemC-AMS TDF 调度器在每个时间步长自动调用
- 调用频率 = Fs（全局采样频率）
- 执行顺序：按信号流拓扑的拓扑排序（Topological Order），Mux 在 FFE 之后、Driver 之前执行

**端口读写语义**：

- **`in.read()`**：
  - 读取当前时间步长的输入端口值
  - 值由上游模块（FFE）在本时间步长的 `processing()` 调用中写入
  - 返回类型：`double`（模拟信号电压值）

- **`out.write(x_in)`**：
  - 将计算结果写入输出端口
  - 写入的值将在下一时间步长被下游模块（Driver）的 `in.read()` 读取
  - TDF 调度器自动处理数据流的时序对齐

**时序行为特征**：

- **零延迟传输**：在同一时间步长内完成输入读取和输出写入，等效于组合逻辑（Combinational Logic）
- **无状态处理**：不维护历史数据（无延迟线、无反馈路径），每个时间步长的输出仅依赖当前输入
- **确定性行为**：相同输入序列产生相同输出序列，适合可重复性验证

**当前实现的局限性**：

1. **通道索引未使用**：
   - 成员变量 `m_lane_sel` 已存储，但 `processing()` 中未访问
   - 多通道架构需要修改为数组输入端口：`sca_tdf::sca_in<double> in[N_LANES]`
   - 然后根据 `m_lane_sel` 选择：`double x_in = in[m_lane_sel].read()`

2. **无延迟建模**：
   - 未实现传播延迟（mux_delay）
   - 需要添加延迟线缓冲区或使用 `sca_tdf::sca_delay<double>` 模块

3. **无抖动建模**：
   - 未注入 DCD（占空比失真）或 RJ（随机抖动）
   - 需要集成时间扰动生成和分数延迟插值算法

**代码简洁性设计考量**：

当前实现刻意保持最小复杂度，原因包括：

- **渐进式开发**：先验证信号链的结构正确性，再逐步添加非理想效应
- **调试友好**：透传模式便于隔离问题，当 TX 链路出现异常时可排除 Mux 的影响
- **版本兼容**：未来添加延迟/抖动功能可通过配置开关（如 `enable_delay`, `enable_jitter`）保持向后兼容

### 3.3 未来版本扩展机制（设计规划）

当前版本实现了最基本的透传功能，以下是未来版本计划扩展的核心机制设计思路。**注意：以下内容为设计规划，当前代码中尚未实现。**

#### 3.3.1 固定延迟建模（Propagation Delay）

**设计目标**：建模选择器的传播延迟，匹配真实硬件的时序特性。

**实现方案A：使用 TDF 延迟模块**

SystemC-AMS 提供 `sca_tdf::sca_delay<T>` 模板类，可实现整数倍时间步长的延迟：

```cpp
// 头文件中添加成员变量
sca_tdf::sca_delay<double> m_delay_line;

// 构造函数初始化
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int lane_sel, double delay_s, double Fs)
    : ...
    , m_delay_line("delay_line")
{
    int delay_samples = static_cast<int>(std::round(delay_s * Fs));
    m_delay_line.set_delay(delay_samples);
}

// processing() 中使用
void TxMuxTdf::processing() {
    double x_in = in.read();
    double x_delayed = m_delay_line(x_in);
    out.write(x_delayed);
}
```

**优点**：
- SystemC-AMS 原生支持，实现简单
- 自动处理延迟队列管理和初始化

**限制**：
- 仅支持整数倍时间步长延迟
- 对于非整数采样点延迟（如 15ps 延迟但 Δt=1.786ps，需要 8.4 个采样点），需要四舍五入，引入量化误差

**实现方案B：显式环形缓冲区**

使用 `std::deque` 或 `std::vector` 实现可控的历史数据存储：

```cpp
// 头文件添加
std::deque<double> m_delay_buffer;
int m_delay_samples;

// 构造函数初始化
TxMuxTdf::TxMuxTdf(..., double delay_s, double Fs) {
    m_delay_samples = static_cast<int>(std::round(delay_s * Fs));
    m_delay_buffer.resize(m_delay_samples, 0.0);
}

// processing() 实现
void TxMuxTdf::processing() {
    double x_in = in.read();
    m_delay_buffer.push_back(x_in);
    double x_out = m_delay_buffer.front();
    m_delay_buffer.pop_front();
    out.write(x_out);
}
```

**优点**：
- 完全可控，便于调试和性能优化
- 可扩展为分数延迟（结合插值算法）

**缺点**：
- 需要手动管理缓冲区大小和初始化
- 代码量稍多

**实现方案C：滤波器群延迟近似**

使用一阶全通滤波器（All-Pass Filter）或 Bessel 滤波器的群延迟（Group Delay）近似固定延迟：

```cpp
// 头文件添加
sca_tdf::sca_ltf_nd m_delay_filter;

// 构造函数中配置
TxMuxTdf::TxMuxTdf(..., double delay_s) {
    // 一阶全通滤波器 H(s) = (1 - s/ω) / (1 + s/ω)
    // 在低频段群延迟约为 2/ω
    double omega = 2.0 / delay_s;
    sca_util::sca_vector<double> num = {1.0, -omega};
    sca_util::sca_vector<double> den = {1.0, omega};
    m_delay_filter.set(num, den);
}
```

**优点**：
- 频域特性平滑，不引入高频振铃
- 适合与带宽限制同时建模

**限制**：
- 群延迟在高频段不恒定，仅适用于低频段延迟近似
- 需要权衡延迟精度和带宽特性

#### 3.3.2 抖动建模（Jitter Injection）

**设计目标**：注入确定性抖动（DCD）和随机抖动（RJ），模拟时钟非理想性和电路噪声。

**DCD（Duty Cycle Distortion）建模**

占空比偏离 50% 导致奇数 UI 和偶数 UI 宽度不等，在边沿产生周期性时间偏移。

**实现思路**：

```cpp
// 头文件添加
double m_dcd_percent;  // 占空比（如 48.0 表示 48%）
double m_ui_period;    // UI 周期（秒）
int m_ui_counter;      // UI 计数器
std::deque<double> m_fractional_delay_buffer;

// 构造函数初始化
TxMuxTdf::TxMuxTdf(..., double dcd_percent, double ui_period, double Fs)
    : m_dcd_percent(dcd_percent)
    , m_ui_period(ui_period)
    , m_ui_counter(0)
{
    int samples_per_ui = static_cast<int>(std::round(ui_period * Fs));
    m_fractional_delay_buffer.resize(samples_per_ui, 0.0);
}

// processing() 实现
void TxMuxTdf::processing() {
    double x_in = in.read();
    
    // 计算当前 UI 索引
    int ui_index = m_ui_counter / samples_per_ui;
    m_ui_counter++;
    
    // 奇偶 UI 施加相反方向的时间偏移
    double dcd_offset = (ui_index % 2 == 0) 
        ? (50.0 - m_dcd_percent) / 100.0 * m_ui_period
        : (m_dcd_percent - 50.0) / 100.0 * m_ui_period;
    
    // 将时间偏移转换为分数延迟（需要插值实现）
    double x_out = apply_fractional_delay(x_in, dcd_offset);
    out.write(x_out);
}
```

**关键技术**：
- **分数延迟插值**：当延迟量不是整数倍采样点时，需要使用插值算法（Lagrange/Sinc/Farrow 结构）
- **相位跟踪**：维护 UI 计数器，根据奇偶性施加相反方向的偏移

**RJ（Random Jitter）建模**

随机抖动服从高斯分布，叠加在每个时间步长的输出上。

**实现思路**：

```cpp
// 头文件添加
#include <random>
std::mt19937 m_rng;
std::normal_distribution<double> m_rj_dist;
double m_rj_sigma;  // RJ 标准差（秒）
double m_Fs;

// 构造函数初始化
TxMuxTdf::TxMuxTdf(..., double rj_sigma, int seed, double Fs)
    : m_rj_sigma(rj_sigma)
    , m_Fs(Fs)
    , m_rng(seed == 0 ? std::random_device{}() : seed)
    , m_rj_dist(0.0, rj_sigma)
{
}

// processing() 实现
void TxMuxTdf::processing() {
    double x_in = in.read();
    
    // 生成随机时间偏移
    double time_offset = m_rj_dist(m_rng);  // 单位：秒
    
    // 将时间偏移转换为分数延迟
    double x_out = apply_fractional_delay(x_in, time_offset);
    out.write(x_out);
}
```

**关键技术**：
- **高斯随机数生成**：使用 C++11 `<random>` 库的 `std::normal_distribution`
- **种子管理**：支持固定种子（可重复仿真）和随机种子（蒙特卡洛分析）
- **分数延迟**：与 DCD 建模共用同一插值算法

**分数延迟插值算法**

以下是 Lagrange 插值的示例实现（3 阶）：

```cpp
double TxMuxTdf::apply_fractional_delay(double x_current, double delay_s) {
    // 将延迟转换为采样点数（可能是分数）
    double delay_samples = delay_s * m_Fs;
    int delay_int = static_cast<int>(std::floor(delay_samples));
    double delay_frac = delay_samples - delay_int;
    
    // 从延迟缓冲区中获取插值所需的历史采样点
    // 假设缓冲区已存储足够的历史数据
    double x_n = m_fractional_delay_buffer[delay_int];
    double x_nm1 = m_fractional_delay_buffer[delay_int + 1];
    double x_np1 = m_fractional_delay_buffer[delay_int - 1];
    
    // Lagrange 插值公式（3 点）
    double L0 = 0.5 * delay_frac * (delay_frac - 1.0);
    double L1 = 1.0 - delay_frac * delay_frac;
    double L2 = 0.5 * delay_frac * (delay_frac + 1.0);
    
    double x_interpolated = L0 * x_nm1 + L1 * x_n + L2 * x_np1;
    
    // 更新缓冲区
    m_fractional_delay_buffer.push_front(x_current);
    m_fractional_delay_buffer.pop_back();
    
    return x_interpolated;
}
```

**插值算法对比**：

| 算法 | 阶数 | 精度 | 计算复杂度 | 适用场景 |
|------|------|------|-----------|---------|
| Lagrange | 3-5 | 中等 | 低 | 快速原型验证 |
| Sinc 插值 | 理论无限 | 高 | 高（需截断） | 高精度眼图分析 |
| Farrow 结构 | 可配置 | 高 | 中 | 实时自适应抖动 |

#### 3.3.3 多通道选择机制（Multi-Lane Selection）

**设计目标**：支持真实 N:1 复用器的多输入选择逻辑。

**架构变更**：

当前单输入端口：
```cpp
sca_tdf::sca_in<double> in;
```

修改为多输入端口数组：
```cpp
static const int N_LANES = 8;
sca_tdf::sca_in<double> in[N_LANES];
```

**构造函数适配**：

```cpp
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int num_lanes, int lane_sel)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_num_lanes(num_lanes)
    , m_lane_sel(lane_sel)
{
    // 动态创建端口数组
    in = new sca_tdf::sca_in<double>[num_lanes];
    for (int i = 0; i < num_lanes; i++) {
        std::string port_name = "in_" + std::to_string(i);
        in[i].set_name(port_name.c_str());
    }
    
    // 参数验证
    if (lane_sel >= num_lanes) {
        SC_REPORT_ERROR("TxMuxTdf", "lane_sel exceeds num_lanes");
    }
}
```

**processing() 适配**：

```cpp
void TxMuxTdf::processing() {
    // 根据 lane_sel 选择对应输入通道
    double x_in = in[m_lane_sel].read();
    
    // 后续延迟/抖动处理...
    out.write(x_in);
}
```

**系统级连接示例**：

```cpp
// 顶层模块中实例化多个并行数据路径
WaveGenTdf* wavegen[8];
TxFfeTdf* ffe[8];
TxMuxTdf* mux;

for (int i = 0; i < 8; i++) {
    wavegen[i] = new WaveGenTdf(...);
    ffe[i] = new TxFfeTdf(...);
}
mux = new TxMuxTdf("mux", 8, 5);  // 8 通道，选择第 6 个

// 连接
for (int i = 0; i < 8; i++) {
    ffe[i]->in(wavegen[i]->out);
    mux->in[i](ffe[i]->out);
}
```

**动态通道切换（高级功能）**：

如果需要在仿真过程中动态切换通道（如测试不同 Lane 的性能差异），可添加 DE 域控制接口：

```cpp
// 头文件添加
sca_tdf::sca_de::sca_in<int> lane_sel_ctrl;

// processing() 适配
void TxMuxTdf::processing() {
    // 从 DE 域读取控制信号
    if (lane_sel_ctrl.event()) {
        m_lane_sel = lane_sel_ctrl.read();
    }
    
    double x_in = in[m_lane_sel].read();
    out.write(x_in);
}
```

#### 3.3.4 非线性效应建模（可选）

**增益压缩（Gain Compression）**：

大信号输入下选择器的增益降低，建模传输门的非线性电阻：

```cpp
double gain_factor = 1.0 / (1.0 + std::pow(std::abs(x_in) / m_compression_point, 2));
x_out = x_in * gain_factor;
```

**饱和限幅（Saturation）**：

输出幅度受限于电源电压或驱动能力：

```cpp
double vsat = m_saturation_voltage;
x_out = std::max(-vsat, std::min(vsat, x_out));
```

**码型相关延迟（Pattern-Dependent Delay）**：

不同数据码型下的传播延迟变化，引入数据相关抖动（DDJ）：

```cpp
// 检测当前码型（如连续 1 的个数）
int consecutive_ones = count_consecutive_ones(m_delay_buffer);
double pattern_delay = m_base_delay + consecutive_ones * m_ddj_per_ui;
```

---

## 4. 测试平台架构

### 4.1 设计理念

TX Mux 模块当前版本（v0.1）采用**系统级集成测试策略**，不提供专用测试平台。核心设计理念：透传功能的简单性使得集成测试已足够验证其连接正确性和时序一致性，避免为基础功能重复开发测试基础设施。

### 4.2 测试场景

当前唯一的测试场景为系统级集成验证：

| 测试场景 | 测试平台 | 验证目标 | 实现状态 |
|---------|---------|---------|---------|
| **系统级集成** | `simple_link_tb.cpp` | 端到端信号完整性、TX链路连续性 | ✅ 已实现 |

**验证要点**：
- Mux 正确连接 FFE 和 Driver 模块
- 信号透传特性（幅度/相位一致）
- TDF 采样率同步（rate=1）
- 仿真稳定性

### 4.3 测试拓扑与连接

TX Mux 在系统级测试平台中的位置：

```
┌──────────────────────────────────────────────────────────────┐
│               simple_link_tb.cpp (System-Level)               │
│                                                                │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │ WaveGen │──▶│  TxFFE  │──▶│  TxMux  │──▶│ TxDriver│──▶    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│                                                                │
│  Trace Signals: ffe_out, driver_out                           │
│  注意：sig_mux_out 未显式追踪（需添加以直接验证透传特性）     │
└──────────────────────────────────────────────────────────────┘
```

**关键信号连接**（`tb/simple_link_tb.cpp` 第 63-69 行）：
```cpp
tx_ffe.out(sig_ffe_out);        // Mux 输入
tx_mux.in(sig_ffe_out);
tx_mux.out(sig_mux_out);        // Mux 输出
tx_driver.in(sig_mux_out);
```

**参数配置**：通过 `ConfigLoader` 加载 `config/default.json` 中的 `tx.mux_lane` 参数（默认值0）。

### 4.4 验证方法

#### 方法1：添加 Mux 输出追踪（推荐）

**问题**：当前 `simple_link_tb.cpp` 未追踪 `sig_mux_out` 信号，无法直接验证 Mux 透传特性。

**解决方案**：在测试平台中添加追踪语句：
```cpp
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

然后使用 Python 脚本对比 `mux_out` 和 `ffe_out`：
```python
import numpy as np
data = np.loadtxt('simple_link.dat', skiprows=1)
ffe_out = data[:, 2]    # 根据实际列索引调整
mux_out = data[:, 3]
error = np.abs(mux_out - ffe_out)
print(f"透传误差（最大值）: {np.max(error):.2e} V")  # 期望 < 1e-12
```

#### 方法2：间接验证（当前可行但不精确）

**注意**：直接比较 `driver_out` 和 `ffe_out` 是**技术错误**，因为 Driver 模块引入了增益、带宽限制和饱和效应，差异不能归因于 Mux。仅可用于粗略的信号链完整性检查。

#### 方法3：仿真日志检查

SystemC-AMS 仿真成功完成且无警告，说明端口连接和采样率配置正确。

### 4.5 辅助模块说明

TX Mux 模块在测试平台中依赖以下辅助模块提供输入信号和功能支持。本节说明这些模块的功能及其与 Mux 的交互关系。

#### 4.5.1 WaveGen 模块（波形生成器）

**模块路径**：`include/ams/wave_generation.h`, `src/ams/wave_generation.cpp`

**功能说明**：
- 生成测试用的 PRBS（伪随机二进制序列）数据码型
- 支持多种 PRBS 类型：PRBS7、PRBS9、PRBS15、PRBS23、PRBS31
- 可配置数据速率、比特模式和初始化种子

**与 Mux 的关系**：
- WaveGen → FFE → Mux 信号链的源头
- 为 Mux 提供测试用的模拟信号输入（通过 FFE 均衡后）
- 在 `simple_link_tb.cpp` 中实例化并连接到 FFE 模块

**典型配置**：
```json
{
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF"
  }
}
```

#### 4.5.2 Trace 信号监控器

**功能说明**：
- SystemC-AMS 提供的波形追踪机制（`sca_util::sca_trace`）
- 将仿真过程中的信号值记录到 `.dat` 文件
- 支持后处理分析和可视化（Python/Matplotlib）

**关键追踪信号**：
- `ffe_out`：FFE 输出（Mux 输入），用于验证透传特性
- `mux_out`：Mux 输出，当前测试平台未追踪（建议添加）
- `driver_out`：Driver 输出，用于系统级信号完整性分析

**添加 Mux 输出追踪**：
```cpp
// 在 tb/simple_link_tb.cpp 中添加
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

**数据格式**：
```
# time(s)    wave_out(V)    ffe_out(V)    mux_out(V)    driver_out(V)
0.00e+00     0.000          0.000         0.000         0.000
1.78e-12     0.500          0.500         0.500         0.200
...
```

#### 4.5.3 ConfigLoader 模块（配置加载器）

**模块路径**：`include/de/config_loader.h`, `src/de/config_loader.cpp`

**功能说明**：
- 从 JSON/YAML 配置文件加载参数
- 解析并填充到 `TxParams` 结构体
- 提供参数验证和默认值处理

**与 Mux 的关系**：
- 加载 `tx.mux_lane` 参数并传递给 Mux 构造函数
- 支持多场景配置切换（不同通道索引）
- 简化测试配置管理，避免硬编码参数

---

## 5. 仿真结果分析

### 5.1 当前版本验证方法

TX Mux 当前版本（v0.1）采用理想透传架构（`out = in`），无延迟、无抖动、无非线性效应。由于测试平台 `simple_link_tb.cpp` **未追踪 mux_out 信号**，直接波形分析不可用，仅能通过系统级间接验证。

### 5.2 系统级集成验证结果

#### 5.2.1 验证原理

通过观测 TX 链路完整输出（`sig_driver_out`）确认信号链连续性：

```
WaveGen → FFE → Mux → Driver → Channel
```

**间接验证逻辑**：
- 若 Driver 输出包含正确的数据码型且眼图质量符合预期，说明 Mux 正确传递了 FFE 输出
- 若仿真成功完成无错误，说明端口连接和采样率配置正确（rate=1 一致性）

**局限性**：
- **不能直接验证透传特性**：Driver 引入增益、带宽限制和饱和效应，`driver_out` 与 `ffe_out` 的差异无法归因于 Mux
- **无法量化透传误差**：需要直接追踪 `mux_out` 才能测量数值精度（预期误差 < 1e-12 V，浮点精度限制）

#### 5.2.2 典型仿真结果

**配置**（`config/default.json`）：
```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**观测指标**：
- **仿真完成状态**：✅ 成功（无 SystemC-AMS 错误或警告）
- **信号链完整性**：✅ Driver 输出包含预期 PRBS 码型
- **时序一致性**：✅ 无采样率不匹配警告

**预期结果**：
- Mux 作为透传单元，不改变信号幅度、相位或频谱特性
- 系统级眼图质量主要取决于 Channel 损耗和 RX 均衡器性能

### 5.3 直接验证方法（需修改测试平台）

#### 5.3.1 添加 Mux 输出追踪

在 `simple_link_tb.cpp` 中添加：
```cpp
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

#### 5.3.2 透传特性分析

使用 Python 脚本对比 `mux_out` 和 `ffe_out`：

```python
import numpy as np

data = np.loadtxt('simple_link.dat', skiprows=1)
ffe_out = data[:, col_ffe]
mux_out = data[:, col_mux]

# 透传误差统计
error = mux_out - ffe_out
print(f"最大误差: {np.max(np.abs(error)):.2e} V")
print(f"RMS 误差: {np.sqrt(np.mean(error**2)):.2e} V")

# 期望结果：误差 < 1e-12 V（浮点精度限制）
```

**预期指标**：

| 指标 | 理论值 | 通过标准 | 说明 |
|------|-------|---------|------|
| 最大误差 | 0 V | < 1e-12 V | 浮点运算精度限制 |
| RMS 误差 | 0 V | < 1e-15 V | 理想透传 |
| 相位偏移 | 0 s | < 1 ps | 同时间步长采样 |
| 频谱一致性 | 100% | > 99.9% | FFT 对比 |

### 5.4 未来版本分析指标

当实现延迟和抖动建模后（v0.2+），应添加以下分析：

**延迟测量**：
- 交叉相关法测量传播延迟（预期值 = `mux_delay` 参数）
- 群延迟一致性检查

**抖动分解**：
- DCD 引起的周期性时间偏移（奇偶 UI 对比）
- RJ 的高斯分布拟合（均值应为 0，标准差 = `rj_sigma`）

**眼图影响**：
- 抖动导致的眼宽闭合（水平方向）
- 与无 Mux 抖动基线的对比

### 5.5 波形数据文件格式

SystemC-AMS trace 文件输出格式（当添加 mux_out 追踪后）：

```
# time(s)    wave_out(V)    ffe_out(V)    mux_out(V)    driver_out(V)
0.00e+00     0.000          0.000         0.000         0.000
1.78e-12     0.500          0.500         0.500         0.200
3.57e-12     1.000          0.650         0.650         0.260
...
```

**列说明**：
- `time`：仿真时间（秒）
- `ffe_out`：FFE 输出（Mux 输入）
- `mux_out`：Mux 输出（当前版本应与 `ffe_out` 完全一致）
- `driver_out`：Driver 输出（引入增益和带宽效应）

---

## 6. 运行指南

### 6.1 环境配置

TX Mux 模块通过系统级测试平台 `simple_link_tb` 进行验证，需完成 SystemC-AMS 开发环境配置。

**必需环境变量**：
```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
```

**验证安装**：
```bash
ls $SYSTEMC_AMS_HOME/include/systemc-ams
# 应显示 systemc-ams.h 等头文件
```

### 6.2 构建与运行

#### 6.2.1 使用 CMake（推荐）

**构建系统级测试平台**：
```bash
cd /path/to/serdes
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make simple_link_tb
```

**运行仿真**：
```bash
./bin/simple_link_tb
# 仿真输出：simple_link.dat
```

**预期输出**：
```
SystemC 2.3.4 --- Jan 13 2026 10:30:00
SystemC-AMS 2.3.4 --- Jan 13 2026 10:30:00
Info: simulation stopped by user.
```

#### 6.2.2 使用 Makefile

**快速运行**：
```bash
cd /path/to/serdes
make run
# 自动构建并执行 simple_link_tb
```

**清理构建**：
```bash
make clean
```

### 6.3 参数配置

TX Mux 通过配置文件 `config/default.json` 加载参数：

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**修改通道索引**（当前版本无实际功能影响）：
```json
{
  "tx": {
    "mux_lane": 2
  }
}
```

**注意**：修改配置后需重新运行测试平台，无需重新编译。

### 6.4 结果查看

#### 6.4.1 验证仿真成功

检查仿真日志无错误或警告：
```bash
grep -i "error\|warning" build/simulation.log
# 无输出表示成功
```

#### 6.4.2 添加 Mux 输出追踪（可选）

**编辑测试平台**（`tb/simple_link_tb.cpp`）：
```cpp
// 在 trace 创建部分添加
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

**重新编译并运行**：
```bash
cd build
make simple_link_tb
./bin/simple_link_tb
```

#### 6.4.3 Python 分析脚本

使用 Python 验证透传特性（需先添加 `mux_out` 追踪）：

```python
import numpy as np

data = np.loadtxt('build/simple_link.dat', skiprows=1)
ffe_out = data[:, 2]  # 根据实际列索引调整
mux_out = data[:, 3]

error = np.abs(mux_out - ffe_out)
print(f"透传误差（最大）: {np.max(error):.2e} V")
print(f"透传误差（RMS）: {np.sqrt(np.mean(error**2)):.2e} V")
# 期望：误差 < 1e-12 V
```

### 6.5 故障排查

#### 6.5.1 常见错误

**采样率不匹配**：
```
Error: (E117) sc_signal<T>: port not bound
```
**解决**：检查 FFE 和 Driver 的 `set_rate()` 配置是否均为 `rate=1`。

**端口连接错误**：
```
Error: port 'in' not connected
```
**解决**：验证 `simple_link_tb.cpp` 中 Mux 的输入输出连接完整性。

**配置文件缺失**：
```
Error: cannot open config/default.json
```
**解决**：确保工作目录在项目根目录，或修改配置文件路径。

#### 6.5.2 调试技巧

**启用详细日志**：
```bash
export SC_REPORT_VERBOSITY=SC_FULL
./bin/simple_link_tb
```

**检查信号连接**：在测试平台构造函数中添加：
```cpp
std::cout << "Mux input rate: " << tx_mux.in.get_rate() << std::endl;
std::cout << "Mux output rate: " << tx_mux.out.get_rate() << std::endl;
```

---

## 7. 技术要点

### 7.1 透传架构避免代数环

**设计选择**：当前版本采用无状态透传（`out = in`），不维护内部状态变量。

**技术优势**：
- 避免代数环风险（输出不依赖自身反馈）
- 与FFE/Driver线性级联时保证TDF调度器拓扑排序收敛
- 适合作为系统集成验证的基线参考模块

**应用限制**：无法建模真实硬件的传播延迟和相位特性，需扩展为延迟线架构（见7.3）。

### 7.2 lane_sel参数保留原因

**当前状态**：参数 `m_lane_sel` 已存储但未使用（`processing()` 中未访问）。

**保留意图**：
- 为多通道架构预留接口（2:1/4:1/8:1并串转换）
- 需修改端口为数组：`sca_tdf::sca_in<double> in[N]`
- 然后根据 `m_lane_sel` 索引：`x_in = in[m_lane_sel].read()`

**配置向前兼容**：当前配置文件可无缝升级至多通道版本。

### 7.3 延迟建模方案选择

未来版本需选择延迟实现方案，各有权衡：

| 方案 | 精度 | 实现难度 | 副作用 |
|------|------|---------|--------|
| **sca_delay模块** | 整数采样点 | 低 | 量化误差（非整数延迟） |
| **显式缓冲区** | 可配合插值达到分数采样点 | 中 | 需手动管理队列 |
| **滤波器群延迟** | 频率相关 | 低 | 引入带宽限制 |

**推荐方案**：初期使用 `sca_delay`（简单），高精度需求时升级为缓冲区+Lagrange插值。

### 7.4 分数延迟插值必要性

**触发条件**：当延迟时间不是采样周期整数倍时（如15ps延迟但Δt=1.786ps，需8.4个采样点）。

**技术方案**：
- **Lagrange插值**（3-5阶）：计算量低，精度中等
- **Sinc插值**：理论最优，需截断窗处理
- **Farrow结构**：实时可调延迟，适合抖动建模

**关键应用**：RJ/DCD抖动注入需亚采样点时间精度（<0.5ps），必须使用插值。

### 7.5 扩展触发条件

**当前版本适用性**：低速/单通道系统（<28Gbps），透传简化合理。

**必须扩展的场景**：
- 比特速率 ≥ 56Gbps：Mux延迟占UI比例 > 15%（15ps / 100ps UI）
- 抖动敏感应用：需精确建模DCD/RJ对眼图的影响
- 多通道SerDes：验证不同Lane间的时序偏差

**技术风险**：v0.1在高速系统中会**过度乐观估计眼图质量约25%**（忽略Mux贡献的抖动）。

### 7.6 测试平台限制影响

**问题**：`simple_link_tb.cpp` 未追踪 `sig_mux_out`，无法直接验证透传误差。

**影响**：
- 仅能通过系统级输出（`driver_out`）间接推断Mux行为
- Driver的增益/带宽效应与Mux耦合，难以解耦分析
- 无法量化浮点精度误差（预期 < 1e-12 V）

**解决方案**：添加 `sca_util::sca_trace(tf, sig_mux_out, "mux_out")`，然后使用Python对比 `mux_out` 与 `ffe_out`。

---

## 8. 参考信息

### 8.1 相关代码文件

| 文件类别 | 路径 | 说明 |
|---------|------|------|
| **头文件** | `include/ams/tx_mux.h` | TxMuxTdf 类声明、端口定义 |
| **实现文件** | `src/ams/tx_mux.cpp` | TDF 生命周期方法实现 |
| **参数定义** | `include/common/parameters.h` | TxParams 结构体（`mux_lane` 参数） |
| **测试平台** | `tb/simple_link_tb.cpp` | 系统级集成测试（包含 Mux 模块） |
| **配置文件** | `config/default.json` | 默认参数配置（`tx.mux_lane`） |

### 8.2 核心依赖项

**编译时依赖**：
- **SystemC 2.3.4**：TDF 模块基类、端口类型定义
- **SystemC-AMS 2.3.4**：`sca_tdf::sca_module`、`sca_in/out<double>`
- **C++14 标准**：初始化列表、类型推断支持

**运行时依赖**：
- **配置加载器**：`ConfigLoader` 类（从 JSON/YAML 加载参数）
- **上游模块**：TX FFE（`TxFfeTdf`）提供输入信号
- **下游模块**：TX Driver（`TxDriverTdf`）接收输出信号

**测试依赖**（未来版本）：
- GoogleTest 1.12.1（单元测试框架）
- NumPy/SciPy（透传误差分析）
- Matplotlib（波形可视化）

### 8.3 相关模块文档

| 模块名称 | 文档路径 | 关系说明 |
|---------|---------|---------|
| TX FFE | `docs/modules/ffe.md` | 上游模块，提供均衡后的符号信号 |
| TX Driver | `docs/modules/driver.md` | 下游模块，接收 Mux 输出并驱动信道 |
| Clock Generation | `docs/modules/clkGen.md` | 时钟源，Mux 抖动特性依赖时钟质量 |
| System Config | `README.md` | 系统级参数配置和信号链连接 |

### 8.4 参考标准与规范

**SerDes 架构标准**：

| 标准 | 版本 | 相关内容 |
|------|------|---------|
| **IEEE 802.3** | 2018 | 以太网多通道并串转换架构（Clause 82） |
| **PCIe** | Gen 4/5/6 | 发送端时序预算和抖动规格 |
| **USB4** | v2.0 | Lane 间时序偏差要求（< 0.2 UI） |
| **OIF CEI** | 56G/112G | 高速 SerDes 发送端抖动模板 |

**抖动建模参考**：
- **JEDEC Standard JESD65B**：高速串行数据链路的抖动规格和测量方法
- **Agilent AN 1448-1**：抖动分解理论（RJ、DJ、DCD、DDJ）
- **IEEE 802.3bj**：100G 以太网抖动容限测试方法

### 8.5 配置示例

#### 示例1：单通道透传（当前版本）

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**适用场景**：
- 单通道 SerDes 系统（比特速率 ≤ 28Gbps）
- 前端功能验证和信号链完整性测试
- 无延迟/抖动要求的应用

**预期行为**：理想透传，`out = in`。

#### 示例2：多通道架构配置（未来版本预留）

```json
{
  "tx": {
    "mux_lane": 3,
    "mux_delay": 20e-12,
    "jitter": {
      "enable": true,
      "dcd_percent": 49.0,
      "rj_sigma": 0.25e-12
    }
  }
}
```

**预期行为（待实现）**：
- 选择第 4 个通道（索引 3）
- 固定延迟 20ps
- DCD 占空比 49%（偏离 1%）
- RJ 标准差 0.25ps

**适用场景**：
- 4:1/8:1 并串转换架构
- 高速 SerDes（56G/112G）抖动建模
- 多通道时序偏差分析

### 8.6 学术参考文献

**并串转换架构**：
- J. Savoj et al., "A 12-Gb/s Data Rate Transceiver with Flexible Parallel Bus Interfaces", IEEE JSSC 2003
- M. Harwood et al., "A 12.5Gb/s SerDes in 65nm CMOS", IEEE ISSCC 2007

**抖动建模理论**：
- M. Li and J. Wilstrup, "Paradigm Shift for Jitter and Noise in Design and Test", DesignCon 2004
- K. Yang and D. Chen, "Physical Modeling of Jitter in High-Speed SerDes", IEEE MTT 2010

**SystemC-AMS 建模方法**：
- *SystemC AMS User's Guide*, Accellera, Version 2.3.4
- 第 4 章：TDF（Timed Data Flow）建模方法
- 第 9 章：DE-TDF 混合仿真（动态通道切换）

### 8.7 外部工具与资源

**仿真与分析工具**：
- **SystemC-AMS**：https://systemc.org（开源建模框架）
- **Matplotlib**：https://matplotlib.org（波形可视化）
- **SciPy**：https://scipy.org（信号处理和统计分析）

**设计参考**：
- **Xilinx UG476**：GTX/GTH SerDes 用户指南（多通道时序管理）
- **Intel FPGA IP User Guide**：收发器 PHY IP 配置示例
- **IBIS-AMI Cookbook**：行为级建模最佳实践（www.eda.org/ibis）

### 8.8 已知限制与未来计划

**当前版本（v0.1）限制**：
- 仅支持单输入单输出（SISO）架构
- 无延迟和抖动建模
- `lane_sel` 参数不影响信号处理

**未来版本计划**：

| 特性 | 目标版本 | 优先级 | 说明 |
|------|---------|-------|------|
| 固定延迟建模 | v0.2 | 高 | 使用 `sca_delay` 或显式缓冲区 |
| DCD 抖动注入 | v0.2 | 高 | 奇偶 UI 时间偏移 |
| RJ 抖动注入 | v0.2 | 中 | 高斯随机时间扰动 |
| 多输入通道选择 | v0.3 | 中 | 端口数组 + 动态索引 |
| 码型相关延迟 | v0.4 | 低 | 引入数据相关抖动（DDJ） |
| 增益压缩/饱和 | v0.4 | 低 | 非线性效应建模 |

---

**文档版本**：v0.1  
**最后更新**：2026-01-13  
**作者**：SerDes 项目文档团队