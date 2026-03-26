# SerDes 高速串行链路仿真器 (SystemC-AMS)

[![C++](https://img.shields.io/badge/C++-11-blue.svg)](https://isocpp.org/)
[![SystemC-AMS](https://img.shields.io/badge/SystemC--AMS-2.3.4-orange.svg)](https://accellera.org/community/systemc-ams)
[![CMake](https://img.shields.io/badge/CMake-3.15+-green.svg)](https://cmake.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)

🌐 **Languages**: [English](README.md) | [中文](README_ZH.md)

基于 **SystemC-AMS** 的高速串行链路（SerDes）行为级建模与仿真平台，支持从 TX → Channel → RX 的完整信号链仿真，包含 PRBS 生成、抖动注入、均衡、时钟恢复及 Python 眼图分析。


---

## 📋 功能特性

### TX 发送端
- **FFE (前馈均衡)**：可配置抽头系数的 FIR 滤波器
- **Mux (复用器)**：Lane 选择与通道复用
- **Driver (驱动器)**：支持非线性饱和、带宽限制、差分输出

### Channel 信道
- **S 参数模型**：基于 Touchstone (.sNp) 文件
- **向量拟合**：离线有理函数拟合，确保因果稳定性
- **串扰与双向传输**：支持多端口耦合与反射

### RX 接收端
- **CTLE (连续时间线性均衡器)**：可配置零极点，支持噪声/偏移/饱和建模
- **VGA (可变增益放大器)**：可编程增益，支持 AGC
- **Sampler (采样器)**：相位可配置，支持阈值/迟滞
- **DFE (判决反馈均衡)**：FIR 结构，支持 LMS/Sign-LMS 自适应
- **CDR (时钟数据恢复)**：PI 控制环路，支持 Bang-Bang/线性相位检测

### 时钟与波形
- **Clock Generation**：理想时钟 / PLL / ADPLL 可选
- **Wave Generation**：PRBS7/9/15/23/31 与自定义多项式，支持 RJ/SJ/DJ 抖动注入

### Python EyeAnalyzer
- 眼图生成与度量（眼高、眼宽、开口面积）
- 抖动分解（RJ/DJ/TJ）
- PSD/PDF 分析与可视化

---

## 🏗️ 项目结构

```
serdes/
├── include/                    # 头文件
│   ├── ams/                    # AMS 模块 (TDF域)
│   │   ├── tx_*.h              # TX: FFE, Mux, Driver
│   │   ├── channel_sparam.h    # Channel S参数模型
│   │   ├── rx_ctle.h           # RX: CTLE, VGA, Sampler
│   │   ├── rx_dfe*.h           # DFE Summer, DAC
│   │   ├── rx_cdr.h            # CDR (PI控制器)
│   │   ├── wave_generation.h   # PRBS/波形生成
│   │   └── clock_generation.h  # 时钟生成
│   ├── common/                 # 公共类型、参数、常量
│   └── de/                     # DE 域模块
│       └── config_loader.h     # JSON/YAML 配置加载
├── src/                        # 实现文件
│   ├── ams/                    # AMS 模块实现
│   └── de/                     # DE 模块实现
├── tb/                         # Testbenches
│   ├── top/                    # 全链路仿真
│   ├── rx/, tx/, periphery/    # 子系统测试
├── tests/                      # 单元测试 (GoogleTest)
│   └── unit/                   # 139+ 测试用例
├── eye_analyzer/               # Python 眼图分析包
│   ├── core.py                 # 核心分析引擎
│   ├── jitter.py               # 抖动分解
│   └── visualization.py        # 可视化
├── scripts/                    # 脚本工具
│   ├── run_*.sh                # 测试运行脚本
│   ├── analyze_serdes_link.py  # 链路结果分析
│   └── vector_fitting.py       # S参数向量拟合
├── config/                     # 配置模板
│   ├── default.json            # 默认配置
│   └── default.yaml
└── docs/zh/modules/               # 模块文档
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本 |
|------|------|
| C++ 标准 | C++14 |
| SystemC | 2.3.4 |
| SystemC-AMS | 2.3.4 |
| CMake | ≥3.15 |
| Python | ≥3.8 |

依赖库：`numpy`, `scipy`, `matplotlib`

### 前提条件

#### 1. 安装 SystemC 和 SystemC-AMS

**SystemC**: https://github.com/accellera-official/systemc/

**SystemC-AMS**: https://www.coseda-tech.com/systemc-ams-proof-of-concept

#### 2. 设置环境变量（推荐）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export SYSTEMC_HOME=/path/to/systemc-2.3.4
export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4

# 或者临时设置
export SYSTEMC_HOME=~/systemc-2.3.4
export SYSTEMC_AMS_HOME=~/systemc-ams-2.3.4
```

> **注意**: 项目支持以下方式指定 SystemC 路径（按优先级）：
> 1. CMake 选项: `-DSYSTEMC_HOME=path -DSYSTEMC_AMS_HOME=path`
> 2. 环境变量: `SYSTEMC_HOME`, `SYSTEMC_AMS_HOME`
> 3. 自动查找标准安装路径

### 构建项目

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/serdes.git
cd serdes

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置（自动检测 SystemC 路径）
cmake ..

# 或手动指定路径（如果不使用环境变量）
# cmake -DSYSTEMC_HOME=/path/to/systemc -DSYSTEMC_AMS_HOME=/path/to/systemc-ams ..

# 4. 编译
make -j4

# 5. 运行测试（可选）
ctest
```

### 运行全链路仿真

```bash
# 使用脚本运行
./scripts/run_serdes_link.sh basic yes

# 或手动运行
cd build
./tb/serdes_link_tb basic

# Python 后处理分析
cd ..
python3 scripts/analyze_serdes_link.py basic
python3 scripts/plot_dfe_taps.py build/serdes_link_basic_dfe_taps.csv
```

### 运行单元测试

```bash
# 运行所有测试
./scripts/run_unit_tests.sh

# 或运行特定模块测试
./scripts/run_cdr_tests.sh
./scripts/run_adaption_tests.sh
```

---

## 📊 使用示例

### 配置仿真参数

编辑 `config/default.json`：

```json
{
  "global": {
    "Fs": 80e9,
    "UI": 2.5e-11,
    "duration": 1e-6,
    "seed": 12345
  },
  "wave": {
    "type": "PRBS31",
    "jitter": {
      "RJ_sigma": 5e-13,
      "SJ_freq": [5e6],
      "SJ_pp": [2e-12]
    }
  },
  "tx": {
    "ffe_taps": [0.2, 0.6, 0.2],
    "driver": { "swing": 0.8, "bw": 20e9 }
  },
  "rx": {
    "ctle": {
      "zeros": [2e9],
      "poles": [30e9],
      "dc_gain": 1.5
    },
    "dfe": { "taps": [-0.05, -0.02, 0.01] }
  },
  "cdr": {
    "pi": { "kp": 0.01, "ki": 1e-4 }
  }
}
```

### Python 眼图分析

```python
from eye_analyzer import EyeAnalyzer
import numpy as np

# 初始化分析器
analyzer = EyeAnalyzer(
    ui=2.5e-11,      # 10Gbps
    ui_bins=128,
    amp_bins=128,
    jitter_method='dual-dirac'
)

# 加载波形并分析
time, voltage = analyzer.load_waveform('waveform.csv')
metrics = analyzer.analyze(time, voltage)

# 输出结果
print(f"Eye Height: {metrics['eye_height']:.3f} V")
print(f"Eye Width: {metrics['eye_width']:.3f} UI")
print(f"TJ @ 1e-12: {metrics['tj_at_ber']:.3e} s")
```

---

## 📚 文档索引

### AMS 模块文档

| 模块 | 文档 |
|------|------|
| **TX** | [TX 系统](docs/zh/modules/tx.md) |
| └ FFE | [FFE](docs/zh/modules/ffe.md) |
| └ Mux | [Mux](docs/zh/modules/mux.md) |
| └ Driver | [Driver](docs/zh/modules/driver.md) |
| **Channel** | [Channel S参数](docs/zh/modules/channel.md) |
| **RX** | [RX 系统](docs/zh/modules/rx.md) |
| └ CTLE | [CTLE](docs/zh/modules/ctle.md) |
| └ VGA | [VGA](docs/zh/modules/vga.md) |
| └ Sampler | [Sampler](docs/zh/modules/sampler.md) |
| └ DFE Summer | [DFE Summer](docs/zh/modules/dfesummer.md) |
| └ CDR | [CDR](docs/zh/modules/cdr.md) |
| **Periphery** | WaveGen / [ClockGen](docs/zh/modules/clkGen.md) |
| **Adaption** | [Adaption](docs/zh/modules/adaption.md) |

### Python 组件

| 组件 | 文档 |
|------|------|
| EyeAnalyzer | [EyeAnalyzer](docs/zh/modules/EyeAnalyzer.md) |

---

## 🧪 测试覆盖

项目包含 **139+** 个单元测试，覆盖：

| 模块 | 测试数 | 测试内容 |
|------|--------|----------|
| Adaption | 18 | AGC、DFE LMS、CDR PI、阈值自适应 |
| CDR | 20 | PI控制器、PAI、边沿检测、模式识别 |
| ClockGen | 18 | 理想/PLL/ADPLL时钟、频率/相位测试 |
| FFE | 10 | 抽头系数、卷积、预/去加重 |
| Sampler | 16 | 判决、迟滞、噪声、偏移 |
| TX Driver | 8 | DC增益、饱和、带宽、PSRR |
| WaveGen | 21 | PRBS模式、抖动、脉冲、稳定性 |
| DFE | 3 | 抽头反馈、历史更新 |
| Channel | 3 | S参数、VF/IR一致性 |
| Top Level | 13 | TX/RX集成测试 |

---

## 🔧 技术细节

### 建模域

- **TDF (Timed Data Flow)**：主要建模域，用于模拟/混合信号模块
- **DE (Discrete Event)**：控制/算法模块，与 AMS 域通过 `sca_de::sca_in/out` 桥接

### 关键设计模式

```cpp
// TDF 模块标准结构
class RxCtleTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p, in_n;
    sca_tdf::sca_out<double> out_p, out_n;
    
    void set_attributes() override;
    void initialize() override;
    void processing() override;
};
```

### 传递函数实现

CTLE/VGA 使用零极点配置，通过 `sca_tdf::sca_ltf_nd` 实现：

```cpp
// H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
sca_util::sca_vector<double> num, den;
build_transfer_function(zeros, poles, dc_gain, num, den);
double output = m_ltf(m_num, m_den, input);
```

---

## 📄 许可证

[LICENSE](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系

如有问题或建议，请通过 GitHub Issues 联系。
