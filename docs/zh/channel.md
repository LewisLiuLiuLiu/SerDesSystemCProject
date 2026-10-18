# Channel 模块技术文档

**Languages**: [中文](channel.md) | [English](../en/modules/channel.md)

**级别**：AMS 顶层模块  
**类名**：`ChannelSParamTdf`  
**当前版本**：v1.1 (2026-10-18)  
**状态**：生产就绪（完整 S参数建模）

> **2026-10 实现注记**：`full_model.delay_s` 现在**在运行时真正生效**——每个激活输出按 `round(tau/dt)` 个采样的环形缓冲施加延迟，主路径取 `port_pairs` 中 `(active_output, active_inputs[0])` 对应的条目（此前延迟仅被解析、从未使用，导致仿真信道延迟≈0ns，而真实夹具约 4ns）。MIMO 共享状态向量下各输入间延迟斜偏不可分离，未建模；信道测试平台现已驱动 input[0]、记录 output[1]（S21 路径），其余 MIMO 输入接 0V。

---

## 1. 概述

信道模块（Channel）是SerDes链路中连接发送端与接收端的关键传输路径，基于实测的S参数数据建模真实信道的频率相关衰减、相位失真、串扰耦合和反射效应。模块采用状态空间表示法实现高精度的时域建模，支持多端口差分传输和复杂拓扑场景。

### 1.1 设计原理

信道模块的核心设计思想是将频域表征的S参数（Scattering Parameters）通过Vector Fitting算法转换为时域状态空间模型：

- **频域到时域转换**：通过Vector Fitting算法将S参数频域数据拟合为有理函数形式，再转换为状态空间实现
- **状态空间表示**：采用MIMO状态空间模型 `dx/dt = A·x + B·u`, `y = C·x + D·u + E·du/dt`
- **因果性保证**：VF算法确保极点位于左半平面，满足物理因果性
- **稳定性约束**：所有极点实部为负，保证系统稳定
- **无源性保持**：可选的无源性强制确保能量守恒

模块提供两种实现方法：

**方法一：SIMPLE方法（快速验证/回退）**
- 核心思想：使用一阶低通滤波器进行快速信道近似
- 数学形式：`H(s) = A / (1 + s/ω₀)`，其中 `A = 10^(-attenuation_db/20)`，`ω₀ = 2π × bandwidth_hz`
- 时域实现：简单的一阶IIR滤波器
- 优势：无需预处理、计算开销极低、适合快速验证

**方法二：STATE_SPACE方法（VF建模唯一入口）**
- 核心思想：使用Vector Fitting算法将S参数转换为MIMO状态空间模型
- 算法来源：基于 [SINTEF Vector Fitting](https://www.sintef.no/en/software/vector-fitting/downloads/vfit3/) 的 `vectfit3.m`（Bjørn Gustavsen, SINTEF Energy Research）
- 数学形式：
  ```
  dx/dt = A·x + B·u
  y = C·x + D·u + E·du/dt
  ```
- 时域实现：利用 SystemC-AMS 的 `sca_ss` 状态空间模块
- 优势：完整MIMO支持、数值稳定性好、支持差分转换和时延提取

### 1.2 核心特性

- **双方法支持**：SIMPLE方法用于快速验证，STATE_SPACE方法用于完整S参数建模
- **MIMO建模支持**：通过 `sc_vector` 端口支持多输入多输出，可配置活跃端口子集
- **差分信号支持**：Python工具链支持将单端S参数转换为差分模式
- **时延提取与补偿**：自动提取传输时延，改善高频拟合精度
- **共享极点Vector Fitting**：多端口S参数使用共享极点，保证系统一致性
- **端口配置灵活**：通过JSON配置选择活跃输入/输出端口，支持子矩阵提取
- **数值稳定性**：使用LU分解计算DC增益，避免矩阵求逆的数值问题
- **配置驱动设计**：通过JSON配置文件管理状态空间矩阵，离线处理与在线仿真解耦

### 1.3 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.1 | 2025-09 | 初始版本，基础向量拟合框架占位 |
| v0.2 | 2025-10-16 | 双方法重构：新增有理函数拟合法和冲激响应卷积法完整说明 |
| v0.3 | 2025-10-16 | GPU加速支持：新增Metal GPU加速（Apple Silicon专属），支持直接卷积和FFT卷积 |
| v0.4 | 2025-12-07 | 完善第1章概述：重写设计原理和核心特性，对齐CTLE/VGA文档风格标准 |
| v1.0 | 2026-03-22 | **重大重构**：统一建模到State Space，移除RATIONAL/IMPULSE方法，完整MIMO支持 |

### 1.4 参考实现与致谢

**Vector Fitting 算法：**
- **原作者**：Bjørn Gustavsen（SINTEF Energy Research）
- **算法论文**：
  - [1] Gustavsen & Semlyen, "Rational approximation of frequency domain responses by Vector Fitting," *IEEE Trans. Power Delivery*, vol. 14, no. 3, pp. 1052-1061, July 1999
  - [2] Gustavsen, "Improving the pole relocating properties of vector fitting," *IEEE Trans. Power Delivery*, vol. 21, no. 3, pp. 1587-1592, July 2006
- **官方实现**：[SINTEF VFIT3](https://www.sintef.no/en/software/vector-fitting/downloads/vfit3/)
- **本项目实现**：`scripts/vector_fitting.py` - 基于 `vectfit3.m`（版本 1.0, 2008）的完整 Python 翻译

**许可证声明：**
SINTEF Vector Fitting 程序限于非商业用途（NON-COMMERCIAL use only）。本项目的 Python 实现遵循相同的非商业限制。

---

## 2. 模块接口

### 2.1 类声明与继承

```cpp
namespace serdes {
class ChannelSParamTdf : public sca_tdf::sca_module {
public:
    // TDF端口（MIMO支持）
    sc_core::sc_vector<sca_tdf::sca_in<double>> in;
    sc_core::sc_vector<sca_tdf::sca_out<double>> out;
    
    // 构造函数
    ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
    ChannelSParamTdf(sc_core::sc_module_name nm, 
                     const ChannelParams& params,
                     const ChannelExtendedParams& ext_params);
    
    // SystemC-AMS生命周期方法
    void set_attributes();
    void initialize();
    void processing();
    
    // 配置加载
    bool load_config(const std::string& config_path);
    
    // 查询接口
    ChannelMethod get_method() const;
    double get_dc_gain() const;
    int get_n_active_inputs() const;
    int get_n_active_outputs() const;
    
private:
    ChannelParams m_params;
    ChannelExtendedParams m_ext_params;
    // ... 详见实现
};
}
```

**继承层次**：
- 基类：`sca_tdf::sca_module`（SystemC-AMS TDF域模块）
- 域类型：TDF（Timed Data Flow，时间驱动数据流）

**v1.0实现说明**：
- 模块在TDF域运行，时间步从上游模块继承（如WaveGen），确保链路采样率一致
- 支持动态端口数量，通过 `sc_vector` 实现MIMO
- 活跃端口可通过JSON配置选择，支持从完整N×N矩阵中提取子矩阵

### 2.2 端口定义（TDF域）

| 端口名 | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `in[N]` | 输入 | `sc_core::sc_vector<sca_tdf::sca_in<double>>` | N个输入端口（支持差分对配置） |
| `out[M]` | 输出 | `sc_core::sc_vector<sca_tdf::sca_out<double>>` | M个输出端口（对应S参数输出端） |

**端口配对示例（4端口差分通道）**：
- 输入差分对1：`in[0]`（正）+ `in[1]`（负）
- 输入差分对2：`in[2]`（正）+ `in[3]`（负）
- 输出差分对1：`out[0]`（正）+ `out[1]`（负）
- 输出差分对2：`out[2]`（正）+ `out[3]`（负）

**活跃端口配置**：
通过JSON配置 `port_config` 选择活跃端口，支持从完整矩阵中提取子系统：
```json
{
  "port_config": {
    "active_inputs": [0, 1],
    "active_outputs": [0, 1]
  }
}
```

### 2.3 构造函数与初始化

```cpp
// 基础构造函数（SIMPLE方法，向后兼容）
ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);

// 扩展构造函数（支持STATE_SPACE方法）
ChannelSParamTdf(sc_core::sc_module_name nm, 
                 const ChannelParams& params,
                 const ChannelExtendedParams& ext_params);
```

**参数说明**：
- `nm`：SystemC模块名称，用于仿真层次标识和波形追踪
- `params`：基础信道参数结构体（`ChannelParams`），包含简化模型参数
- `ext_params`：扩展信道参数（`ChannelExtendedParams`），包含方法和配置文件

**初始化流程**：
1. 调用基类构造函数，注册模块名称
2. 存储参数到成员变量
3. 根据 `ext_params.method` 选择建模方法
4. 如使用STATE_SPACE方法，加载JSON配置文件
5. 提取活跃端口对应的状态空间子矩阵
6. 初始化 `sca_ss` 状态空间滤波器

### 2.4 参数配置

#### ChannelParams（基础参数）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `touchstone` | string | "" | S参数文件路径（.sNp格式）- 供Python工具链使用 |
| `ports` | int | 2 | 端口数量（N≥2）- 供Python工具链使用 |
| `attenuation_db` | double | 10.0 | SIMPLE方法衰减量（dB） |
| `bandwidth_hz` | double | 20e9 | SIMPLE方法带宽（Hz） |

#### ChannelExtendedParams（扩展参数）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | ChannelMethod | SIMPLE | 建模方法：SIMPLE 或 STATE_SPACE |
| `config_file` | string | "" | JSON配置文件路径（STATE_SPACE方法必需） |

**注意**：Channel模块的时间步从上游模块（如WaveGen）继承，确保整个链路采样率一致。

### 2.5 公共API方法

#### set_attributes()

设置TDF模块的时间属性和端口速率。

```cpp
void set_attributes();
```

**职责**：
- 根据活跃端口数量初始化 `in` 和 `out` 端口向量
- 声明端口速率：`in[i].set_rate(1)`, `out[i].set_rate(1)`
- 继承上游模块的时间步（如WaveGen），确保采样率一致性

#### initialize()

模块初始化，根据选择的方法设置内部状态。

```cpp
void initialize();
```

**职责**：
- **SIMPLE方法**：初始化一阶低通滤波器系数
- **STATE_SPACE方法**：加载JSON配置，提取活跃矩阵，初始化 `sca_ss` 滤波器

#### processing()

每个时间步的信号处理函数。

```cpp
void processing();
```

**职责**：
- **SIMPLE方法**：应用一阶低通滤波器
- **STATE_SPACE方法**：调用 `sca_ss` 计算输出，自动处理MIMO

#### load_config()

动态加载JSON配置文件。

```cpp
bool load_config(const std::string& config_path);
```

**参数**：
- `config_path`：JSON配置文件路径

**返回**：成功返回true，失败返回false

#### get_dc_gain()

获取信道DC增益。

```cpp
double get_dc_gain() const;
```

**实现**：
- **SIMPLE方法**：`10^(-attenuation_db/20)`
- **STATE_SPACE方法**：`D - C·A⁻¹·B`（使用LU分解求解，避免直接求逆）

---

## 3. 核心实现机制

### 3.1 SIMPLE 方法

SIMPLE方法提供快速的一阶低通滤波器建模，适用于快速验证和回退场景。

#### 3.1.1 传递函数

```
H(s) = A / (1 + s/ω₀)
```

其中：
- `A = 10^(-attenuation_db/20)`：线性幅度衰减因子
- `ω₀ = 2π × bandwidth_hz`：-3dB带宽对应的角频率

#### 3.1.2 离散实现

使用一阶IIR滤波器实现：

```cpp
y[n] = alpha * y[n-1] + (1 - alpha) * A * x[n]
```

其中 `alpha = exp(-ω₀ * Ts)`，`Ts` 为采样周期（从上游模块继承的时间步）。

#### 3.1.3 使用场景

- 快速验证链路连通性
- 无需S参数文件的简单测试
- 作为STATE_SPACE方法失败时的回退

### 3.2 STATE_SPACE 方法

STATE_SPACE方法是完整的S参数建模实现，基于Vector Fitting和状态空间表示。

#### 3.2.1 状态空间表示

MIMO系统的状态空间表示：

```
dx/dt = A·x(t) + B·u(t)
y(t) = C·x(t) + D·u(t) + E·du/dt
```

矩阵维度：
- `A`: (n_states × n_states) - 状态矩阵
- `B`: (n_states × n_inputs) - 输入矩阵
- `C`: (n_outputs × n_states) - 输出矩阵
- `D`: (n_outputs × n_inputs) - 直接传输矩阵
- `E`: (n_outputs × n_inputs) - 微分矩阵（可选）

对于N端口差分系统：
- `n_inputs = 2 × n_diff_ports`（差分对的正负端）
- `n_outputs = n_diff_ports²`（每对输入-输出组合）

#### 3.2.2 活跃矩阵提取

支持从完整模型中提取活跃端口对应的子矩阵：

```cpp
void extract_active_matrices();
```

提取逻辑：
1. 根据 `port_config.active_inputs` 选择B矩阵的列
2. 根据 `port_config.active_outputs` 选择C矩阵的行、D/E矩阵的行列
3. A矩阵保持不变（共享极点特性）

#### 3.2.3 SystemC-AMS实现

使用 `sca_tdf::sca_ss` 模块实现状态空间：

```cpp
sca_tdf::sca_ss m_ss_filter;
sca_util::sca_vector<double> m_ss_state;
```

初始化：
```cpp
// 时间步从上游模块继承
m_ss_filter.set_timestep(get_timestep());
m_ss_filter.set_model(A, B, C, D, E);
m_ss_state.init(n_states);
```

处理：
```cpp
// 读取输入
sca_util::sca_vector<double> u(n_inputs);
for (int i = 0; i < n_inputs; ++i) {
    u[i] = in[i].read();
}

// 计算输出
sca_util::sca_vector<double> y = m_ss_filter.calculate(u, m_ss_state);

// 写入输出
for (int i = 0; i < n_outputs; ++i) {
    out[i].write(y[i]);
}
```

#### 3.2.4 DC增益计算

使用LU分解求解 `A·X = B`，避免直接矩阵求逆：

```cpp
double get_dc_gain() const {
    // 求解 A * X = B
    auto LU = A;
    lu_decompose(LU, pivot);
    auto X = B;
    lu_solve(LU, pivot, X);
    
    // DC增益 = D - C * X
    // ...
}
```

---

## 4. Python预处理工具链

### 4.1 SParamModel 类

位于 `scripts/vector_fitting.py`

主要功能：
- 任意 sNp 文件加载（S4P、S12P 等）
- 自动/手动端口映射
- 差分转换（mixed-mode）
- 时延提取与补偿
- 共享极点 Vector Fitting
- MIMO 状态空间导出

### 4.2 典型使用流程

```python
from scripts.vector_fitting import SParamModel

# 1. 加载配置（可选，用于自动 fmax 计算）
model = SParamModel()
model.load_config('config/default.json', fmax_multiplier=1.5)

# 2. 加载 S参数文件
model.load_snp('channel.s4p')

# 3. 转换为差分 S参数
model.to_differential()

# 4. 执行 Vector Fitting
model.fit(
    order=14,           # 极点阶数
    extract_delay=True, # 提取时延
    fmax='auto'         # 自动计算带宽
)

# 5. 查看拟合质量
model.summary()

# 6. 导出到 JSON
model.export_json('channel.json', fs=80e9)
```

### 4.3 JSON 配置文件格式

```json
{
  "version": "3.0",
  "method": "state_space",
  "fs": 80000000000.0,
  "full_model": {
    "n_diff_ports": 2,
    "n_outputs": 4,
    "n_states": 28,
    "port_pairs": [[0,0], [0,1], [1,0], [1,1]],
    "delay_s": [4.0e-09, ...],
    "state_space": {
      "A": [[...], ...],
      "B": [[...], ...],
      "C": [[...], ...],
      "D": [...],
      "E": [...]
    }
  },
  "port_config": {
    "active_inputs": [0],
    "active_outputs": [0]
  }
}
```

字段说明：
- `version`: 配置文件版本
- `method`: 建模方法（"state_space"）
- `fs`: 采样频率（Hz）
- `full_model`: 完整MIMO模型数据
  - `n_diff_ports`: 差分端口数量
  - `n_outputs`: 输出数量（= n_diff_ports²）
  - `n_states`: 状态向量维度（= order × n_outputs）
  - `port_pairs`: 每个输出对应的 (out_port, in_port) 对
  - `delay_s`: 每个输出的传输时延（秒）
  - `state_space`: 状态空间矩阵 A, B, C, D, E
- `port_config`: 活跃端口配置
  - `active_inputs`: 活跃输入端口索引列表
  - `active_outputs`: 活跃输出端口索引列表

---

## 5. 测试平台架构

### 5.1 独立测试平台

位于 `tb/channel/channel_sparam_tb.cpp`

**测试内容**：
- SIMPLE方法功能验证
- STATE_SPACE方法功能验证
- 配置加载测试
- DC增益计算验证

**运行方式**：
```bash
cd build
make channel_sparam_tb
./bin/channel_sparam_tb
```

### 5.2 集成测试

位于 `tb/simple_link_tb.cpp`

**测试内容**：
- 完整TX→Channel→RX链路
- 眼图质量验证
- 与其他模块协同工作

---

## 6. 运行指南

### 6.1 使用SIMPLE方法

```cpp
// 基础参数
ChannelParams params;
params.attenuation_db = 10.0;
params.bandwidth_hz = 20e9;

// 创建模块（自动使用SIMPLE方法）
auto channel = std::make_unique<ChannelSParamTdf>("channel", params);
```

### 6.2 使用STATE_SPACE方法

```cpp
// 1. 使用Python工具链生成JSON配置
// python scripts/vector_fitting.py --input channel.s4p --output channel.json

// 2. C++代码中使用STATE_SPACE方法
ChannelParams params;
ChannelExtendedParams ext_params;
ext_params.method = ChannelMethod::STATE_SPACE;
ext_params.config_file = "config/channel.json";
// 注意：Channel模块的时间步从上游模块继承

auto channel = std::make_unique<ChannelSParamTdf>("channel", params, ext_params);
```

### 6.3 配置文件示例

**SIMPLE方法配置**：
```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20000000000.0
    }
  }
}
```

**STATE_SPACE方法配置**：
```json
{
  "channel": {
    "method": "state_space",
    "config_file": "config/channel_ss.json"
  }
}
```

---

## 7. 技术要点

### 7.1 Vector Fitting算法要点

**共享极点**：
- 所有S参数元素使用相同的极点集合
- 保证MIMO系统的一致性和稳定性
- 状态矩阵A在所有传输路径间共享

**时延提取**：
- 线性相位分量提取为时延
- 改善高频拟合精度
- 时延在JSON中单独存储，仿真时作为群延迟特性的一部分

**无源性强制**：
- 确保散射矩阵特征值 ≤ 1
- 防止仿真中出现能量增长

### 7.2 数值稳定性

**LU分解**：
- 使用部分主元LU分解求解线性系统
- 避免直接矩阵求逆的数值问题
- 用于DC增益计算和状态空间分析

**矩阵条件数**：
- 高阶VF可能产生病态矩阵
- 建议阶数：6-16（根据信道复杂度）

### 7.3 性能考虑

**计算复杂度**：
- SIMPLE方法：O(1) 每时间步
- STATE_SPACE方法：O(n_states²) 每时间步

**状态维度**：
- `n_states = order × n_outputs`
- 例如：14阶 × 4输出 = 56维状态向量

---

## 8. 参考信息

### 8.1 相关文件

#### 源代码文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 参数定义 | `/include/common/parameters.h` | ChannelParams结构体 |
| 头文件 | `/include/ams/channel_sparam.h` | ChannelSParamTdf类声明 |
| 实现文件 | `/src/ams/channel_sparam.cpp` | ChannelSParamTdf类实现 |
| Python工具 | `/scripts/vector_fitting.py` | Vector Fitting和预处理工具 |

#### 测试文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 独立测试 | `/tb/channel/channel_sparam_tb.cpp` | Channel模块独立测试 |
| 集成测试 | `/tb/simple_link_tb.cpp` | 完整链路集成测试 |

### 8.2 依赖项

| 依赖项 | 版本要求 | 用途 | 必需性 |
|-------|---------|------|--------|
| SystemC | 2.3.4 | SystemC核心库 | 必须 |
| SystemC-AMS | 2.3.4 | AMS扩展库 | 必须 |
| C++标准 | C++14 | 编译器标准 | 必须 |
| nlohmann/json | 3.x | JSON解析 | 必须 |
| Python | 3.7+ | 预处理工具链 | 推荐 |
| numpy | 1.19+ | 数值计算 | 推荐 |
| scipy | 1.5+ | 信号处理 | 推荐 |

### 8.3 配置示例

#### SIMPLE方法配置

```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20000000000.0
    }
  }
}
```

#### STATE_SPACE方法配置

```json
{
  "channel": {
    "method": "state_space",
    "config_file": "config/channel_state_space.json"
  }
}
```

#### 状态空间JSON格式（由Python工具生成）

```json
{
  "version": "3.0",
  "method": "state_space",
  "fs": 80000000000.0,
  "full_model": {
    "n_diff_ports": 2,
    "n_outputs": 4,
    "n_states": 28,
    "port_pairs": [[0,0], [0,1], [1,0], [1,1]],
    "delay_s": [4.0e-09, 4.0e-09, 4.0e-09, 4.0e-09],
    "state_space": {
      "A": [[...], ...],
      "B": [[...], ...],
      "C": [[...], ...],
      "D": [0.0, 0.0, 0.0, 0.0],
      "E": [0.0, 0.0, 0.0, 0.0]
    }
  },
  "port_config": {
    "active_inputs": [0],
    "active_outputs": [0]
  }
}
```

### 8.4 参数参考表

#### ChannelParams 参数

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `touchstone` | string | "" | S参数文件路径（供Python工具使用） |
| `ports` | int | 2 | 端口数量（供Python工具使用） |
| `attenuation_db` | double | 10.0 | SIMPLE方法衰减量（dB） |
| `bandwidth_hz` | double | 20e9 | SIMPLE方法带宽（Hz） |

#### ChannelExtendedParams 参数

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `method` | ChannelMethod | SIMPLE | 建模方法 |
| `config_file` | string | "" | JSON配置文件路径 |

**注意**：Channel模块的时间步从上游模块继承，不独立设置采样率。

### 8.5 常见问题（FAQ）

**Q1: 如何选择SIMPLE和STATE_SPACE方法？**

A: 
- **SIMPLE**：快速验证、无需S参数文件、计算开销低
- **STATE_SPACE**：需要精确S参数建模、支持MIMO、需要预处理

**Q2: Python工具链如何使用？**

A:
```python
from scripts.vector_fitting import SParamModel

model = SParamModel()
model.load_snp('channel.s4p')
model.to_differential()
model.fit(order=14, extract_delay=True)
model.export_json('channel.json', fs=80e9)
```

**Q3: 如何验证信道模块是否正常工作？**

A:
1. 运行 `channel_sparam_tb` 验证基本功能
2. 运行 `simple_link_tb` 验证链路集成
3. 检查输出波形的眼图质量

**Q4: 状态空间方法的计算开销如何？**

A: 计算复杂度为 O(n_states²) 每时间步。例如14阶4输出系统（56维状态）在现代CPU上可轻松支持80GSa/s采样率。

**Q5: 如何处理多端口S参数文件？**

A: Python工具链自动处理多端口文件，通过 `to_differential()` 转换为差分模式，使用共享极点VF拟合，导出完整MIMO状态空间模型。

**Q6: 可以从完整模型中提取部分端口吗？**

A: 可以。通过JSON中的 `port_config.active_inputs` 和 `port_config.active_outputs` 选择活跃端口，模块会自动提取对应的子矩阵。

---

## 参考文献

[1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency domain responses by Vector Fitting," *IEEE Transactions on Power Delivery*, vol. 14, no. 3, pp. 1052-1061, July 1999.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," *IEEE Transactions on Power Delivery*, vol. 21, no. 3, pp. 1587-1592, July 2006.

[3] B. Gustavsen, "Fast Relaxed Vector Fitting," *SINTEF Energy Research*, version 1.0, August 2008. [Online]. Available: https://www.sintef.no/en/software/vector-fitting/downloads/vfit3/

---

**文档版本**：v1.0  
**最后更新**：2026-03-22  
**作者**：Yizhe Liu
