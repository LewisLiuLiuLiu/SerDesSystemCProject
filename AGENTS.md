# SerDes SystemC-AMS 行为建模平台 - 开发指南

## 项目概述

这是一个基于 **SystemC-AMS**（模拟/混合信号）的高速串行链路（SerDes）行为建模与仿真平台。支持从 TX → 通道 → RX 的完整信号链仿真，包括 PRBS 生成、抖动注入、均衡、时钟恢复以及基于 Python 的眼图分析。

**主要语言**：英文（文档也提供中文版本）

### 主要特性

- **TX 发送器**：FFE（前馈均衡）、Mux（多路复用器）、具有非线性和带宽限制的驱动器
- **通道**：基于 Touchstone（.s4p）文件的 S 参数模型，支持矢量拟合
- **RX 接收器**：CTLE（连续时间线性均衡器）、VGA（可变增益放大器）、采样器、DFE（判决反馈均衡器）、CDR（时钟数据恢复）
- **时钟与波形**：PRBS7/9/15/23/31 生成，支持 RJ/SJ/DJ 抖动注入
- **Python EyeAnalyzer**：眼图生成、抖动分解（RJ/DJ/TJ）、PSD/PDF 分析

---

## 环境配置

### SystemC 环境设置

在构建项目之前，需要设置 SystemC 和 SystemC-AMS 的环境变量：

```bash
# SystemC 2.3.4 + AMS 2.3.4
export SYSTEMC_HOME=/mnt/d/systemCProjects/systemCsrc/systemc-2.3.4-install
export SYSTEMC_AMS_HOME=/mnt/d/systemCProjects/systemCsrc/systemc-ams-install
```

将上述内容添加到 `~/.bashrc` 文件中，然后执行 `source ~/.bashrc` 使其生效。

**注意**：项目已配置 RPATH，编译后的可执行文件会自动找到库文件，**无需设置 `LD_LIBRARY_PATH`**。

### 环境验证

```bash
# 验证环境变量
echo $SYSTEMC_HOME
echo $SYSTEMC_AMS_HOME

# 验证库文件
ls $SYSTEMC_HOME/lib/libsystemc.so*
ls $SYSTEMC_AMS_HOME/lib/libsystemc-ams.so*
```

---

## 技术栈

| 组件 | 版本/要求 |
|-----------|---------------------|
| C++ 标准 | C++14 (SystemC 2.3.4 要求) |
| SystemC | 2.3.4 |
| SystemC-AMS | 2.3.4 |
| CMake | ≥3.15 |
| Python | ≥3.8 |
| GoogleTest | 用于 C++ 单元测试 |

### Python 依赖
```
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
scipy>=1.7.0
```

---

## 项目结构

```
serdes/
├── include/                    # 头文件
│   ├── ams/                    # AMS (TDF) 模块
│   │   ├── tx_*.h              # TX: FFE, Mux, Driver
│   │   ├── channel_sparam.h    # 通道 S 参数模型
│   │   ├── rx_ctle.h           # RX: CTLE, VGA, Sampler
│   │   ├── rx_dfe*.h           # DFE Summer, DAC
│   │   ├── rx_cdr.h            # CDR (PI 控制器)
│   │   ├── wave_generation.h   # PRBS/波形生成
│   │   └── clock_generation.h  # 时钟生成
│   ├── common/                 # 公共类型、参数、常量
│   │   ├── types.h             # 枚举、类型别名、工具函数
│   │   ├── constants.h         # 物理和数值常量
│   │   └── parameters.h        # 所有模块的参数结构
│   └── de/                     # DE (离散事件) 模块
│       └── config_loader.h     # JSON/YAML 配置加载器
├── src/                        # 实现文件
│   ├── ams/                    # AMS 模块实现 (19 个文件)
│   └── de/                     # DE 模块实现
├── tb/                         # 测试平台 (14 个文件)
│   ├── top/                    # 全链路仿真
│   ├── rx/, tx/, periphery/    # 子系统测试
├── tests/                      # 单元测试 (140+ 测试)
│   ├── test_main.cpp           # GoogleTest 入口
│   └── unit/                   # 独立测试文件
├── eye_analyzer/               # Python 眼图分析包
│   ├── core.py                 # 核心分析引擎 (EyeAnalyzer 类)
│   ├── analyzer.py             # UnifiedEyeAnalyzer (新)
│   ├── jitter.py               # 抖动分解
│   ├── visualization.py        # 可视化工具
│   ├── io.py                   # 数据加载工具
│   └── schemes/                # 分析方案
├── scripts/                    # 脚本工具
│   ├── run_*.sh                # 测试运行脚本
│   ├── analyze_serdes_link.py  # 链路结果分析
│   └── vector_fitting.py       # S 参数矢量拟合
├── config/                     # 配置模板
│   ├── default.json            # 默认配置
│   └── default.yaml
├── docs/                       # 文档
│   ├── modules/                # 模块文档 (英文)
│   └── zh/                     # 模块文档 (中文)
├── third_party/                # 第三方依赖
│   └── json.hpp                # nlohmann/json 单头文件库
└── cmake/                      # CMake 模块
    └── FindSystemC.cmake       # SystemC/SystemC-AMS 查找器
```

---

## 构建系统

### 环境要求

SystemC 和 SystemC-AMS 必须已安装。设置环境变量：

```bash
export SYSTEMC_HOME=/path/to/systemc-2.3.4
export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4
```

### 构建命令

```bash
# 1. 创建构建目录
mkdir build && cd build

# 2. 配置 (自动检测 SystemC 路径)
cmake ..

# 或手动指定路径
cmake -DSYSTEMC_HOME=/path/to/systemc -DSYSTEMC_AMS_HOME=/path/to/systemc-ams ..

# 3. 编译
make -j4

# 4. 安装 (可选)
make install
```

### CMake 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `BUILD_TESTING` | 构建单元测试 | OFF |
| `SYSTEMC_HOME` | SystemC 安装路径 | 自动检测 |
| `SYSTEMC_AMS_HOME` | SystemC-AMS 安装路径 | 自动检测 |

### 构建输出

- `libserdes_lib.a`：包含所有 AMS/DE 模块的静态库
- `bin/serdes_link_tb`：全链路测试平台可执行文件
- `bin/*_tb`：独立模块测试平台
- `tests/test_*`：单元测试可执行文件（当 `BUILD_TESTING=ON` 时）

---

## 测试策略

### 单元测试

项目使用 **GoogleTest**，采用特定模式：**每个测试编译为独立的可执行文件**，以避免测试间 SystemC 仿真器重置问题。

```bash
# 构建测试
cd build && cmake -DBUILD_TESTING=ON .. && make

# 通过 CTest 运行所有测试
ctest

# 运行特定测试可执行文件
./tests/test_cdr_basic_functionality

# 使用 GoogleTest 过滤器运行
./tests/test_cdr_basic_functionality --gtest_filter="*Basic*"
```

### 测试类别 (140+ 测试)

| 模块 | 测试数量 | 覆盖范围 |
|--------|------------|----------|
| Adaption | 18 | AGC, DFE LMS, CDR PI, 阈值自适应 |
| CDR | 20+ | PI 控制器, PAI, 边沿检测 |
| ClockGen | 18 | 理想/PLL/ADPLL 时钟 |
| FFE | 10 | 抽头系数, 卷积 |
| Sampler | 16 | 判决, 迟滞, 噪声 |
| TX Driver | 8 | DC 增益, 饱和, 带宽 |
| WaveGen | 21 | PRBS 模式, 抖动 |
| DFE | 3 | 抽头反馈, 历史 |
| Channel | 3 | S 参数, VF/IR |
| TX/RX Top | 21 | 集成测试 |

### 测试脚本

```bash
# 运行所有单元测试
./scripts/run_unit_tests.sh

# 运行特定模块测试
./scripts/run_cdr_tests.sh
./scripts/run_adaption_tests.sh
./scripts/run_clockgen_tests.sh
./scripts/run_ffe_tests.sh
```

### 仿真测试平台

```bash
# 运行全链路仿真
./scripts/run_serdes_link.sh basic yes

# 或手动运行
cd build
./bin/serdes_link_tb basic

# Python 后处理
python3 scripts/analyze_serdes_link.py basic
```

---

## 代码风格指南

### C++ 编码规范

1. **命名空间**：所有代码位于 `namespace serdes`

2. **头文件保护**：使用 `SERDES_<PATH>_<FILENAME>_H` 格式：
   ```cpp
   #ifndef SERDES_TX_FFE_H
   #define SERDES_TX_FFE_H
   // ...
   #endif // SERDES_TX_FFE_H
   ```

3. **类命名**：
   - TDF 模块：`XxYyTdf`（例如 `TxFfeTdf`, `RxCtleTdf`）
   - DE 模块：`XxYyDe`（例如 `ConfigLoaderDe`）

4. **成员变量**：
   - 公有：普通名称（例如 `in`, `out`）
   - 私有：`m_` 前缀（例如 `m_params`, `m_buffer`）

5. **SystemC-AMS 模块结构**：
   ```cpp
   class TxFfeTdf : public sca_tdf::sca_module {
   public:
       sca_tdf::sca_in<double> in;
       sca_tdf::sca_out<double> out;
       
       TxFfeTdf(sc_core::sc_module_name nm, const TxFfeParams& params);
       
       void set_attributes() override;
       void processing() override;
       
   private:
       TxFfeParams m_params;
       std::vector<double> m_buffer;
       size_t m_buffer_ptr;
   };
   ```

6. **包含文件**：
   - SystemC：`<systemc-ams>`（同时包含 SystemC 和 SystemC-AMS）
   - 项目头文件：使用相对路径（例如 `"ams/tx_ffe.h"`, `"common/parameters.h"`）

7. **注释**：中英文混合（遗留），新代码优先使用英文

### Python 编码规范

1. **包结构**：`eye_analyzer/` 包含 `__init__.py` 导出
2. **类命名**：PascalCase（例如 `EyeAnalyzer`, `JitterDecomposer`）
3. **函数命名**：snake_case
4. **文档字符串**：Google 风格文档字符串
5. **类型提示**：函数签名使用 Python 类型提示

---

## 配置系统

配置通过 JSON/YAML 文件加载：

```json
{
  "global": {
    "Fs": 80e9,           // 采样率 (Hz)
    "UI": 2.5e-11,        // 单位间隔 (s)
    "duration": 1e-6,     // 仿真持续时间 (s)
    "seed": 12345         // 随机种子
  },
  "tx": {
    "ffe_taps": [0.2, 0.6, 0.2],
    "driver": { "swing": 0.8, "bw": 20e9 }
  },
  "rx": {
    "ctle": { "zeros": [2e9], "poles": [30e9], "dc_gain": 1.5 },
    "dfe": { "taps": [-0.05, -0.02, 0.01], "update": "sign-lms" }
  },
  "cdr": {
    "pi": { "kp": 0.01, "ki": 1e-4 }
  }
}
```

配置结构定义在 `include/common/parameters.h` 中。

---

## 建模范畴

### TDF（定时数据流）
- 模拟/混合信号模块的主要域
- 位于 `include/ams/`, `src/ams/`
- 使用 `sca_tdf::sca_module`, `sca_tdf::sca_in<>`, `sca_tdf::sca_out<>`

### DE（离散事件）
- 控制/算法模块
- 位于 `include/de/`, `src/de/`
- 使用 `sc_core::sc_module`，通过 `sca_de::sca_in/out` 桥接到 AMS

---

## 关键设计模式

### 传递函数实现

CTLE/VGA 使用零极点配置通过 `sca_tdf::sca_ltf_nd`：

```cpp
// H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
sca_util::sca_vector<double> num, den;
build_transfer_function(zeros, poles, dc_gain, num, den);
double output = m_ltf(m_num, m_den, input);
```

### PRBS 生成

基于 LFSR 的 PRBS 生成器，支持可配置的多项式和抖动注入。

---

## Python EyeAnalyzer API

### 基本用法

```python
from eye_analyzer import EyeAnalyzer

# 初始化分析器
analyzer = EyeAnalyzer(
    ui=2.5e-11,      # 10Gbps
    ui_bins=128,
    amp_bins=128,
    jitter_method='dual-dirac'
)

# 分析波形
metrics = analyzer.analyze(time_array, voltage_array, target_ber=1e-12)

# 保存结果
analyzer.save_results(metrics, output_dir='results/')
```

### 便捷函数

```python
from eye_analyzer import analyze_eye

metrics = analyze_eye(dat_path='waveform.dat', ui=2.5e-11)
```

---

## 常见开发任务

### 添加新模块

1. 在 `include/ams/<module_name>.h` 创建头文件
2. 在 `src/ams/<module_name>.cpp` 创建实现文件
3. 如需，在 `include/common/parameters.h` 添加参数
4. 在 `tb/<category>/<name>_tb.cpp` 创建测试平台
5. 添加到 `tb/CMakeLists.txt`

### 添加单元测试

1. 在 `tests/unit/test_<feature>_<scenario>.cpp` 创建测试文件
2. 在 `tests/CMakeLists.txt` 的相应列表中添加测试名称
3. 使用 `-DBUILD_TESTING=ON` 重新构建

### 运行仿真

```bash
# 全链路仿真
cd build
./bin/serdes_link_tb <config_name> [enable_vcd]

# 输出结果：
# - serdes_link_<config>.dat (波形)
# - serdes_link_<config>_cdr.csv (CDR 数据)
# - serdes_link_<config>_dfe_taps.csv (DFE 抽头演变)
```

---

## Git 忽略说明

`.gitignore` 中的重要排除项：
- 构建产物：`build/`, `*.o`, `*.a`, `*.so`
- 仿真输出：`*.dat`, `*.vcd`, `*.csv`, `*.log`
- 生成图表：`/*.png`, `/*.svg`（仅根目录）
- Python 缓存：`__pycache__/`, `*.pyc`
- IDE 配置：`.vscode/`, `.idea/`（保留 `.qoder/` 用于 AI 编辑器配置）

---

## 故障排除

### SystemC 未找到

```bash
# 设置环境变量
export SYSTEMC_HOME=/path/to/systemc
export SYSTEMC_AMS_HOME=/path/to/systemc-ams

# 或传递给 CMake
cmake -DSYSTEMC_HOME=/path/to/systemc ..
```

### Python 导入错误

```bash
# 安装 Python 依赖
pip install -r requirements.txt
# 或
pip install -r scripts/requirements.txt
```

### 测试失败

如果测试因 "SystemC not initialized" 错误失败，确保每个测试文件使用 `tests/test_main.cpp` 中的模式，并编译为独立的可执行文件。

---

## 文档

- **英文**：`README.md`, `docs/modules/`
- **中文**：`docs/zh/`
- **模块文档**：TX, RX, Channel, CDR, Adaption, EyeAnalyzer 等的独立 `.md` 文件

### SystemC-AMS 版本不匹配

**问题**：SystemC-AMS 2.3.4 是用 SystemC 2.3.3 编译的，但项目需要 SystemC 2.3.4，这导致 TDF 模块在仿真时崩溃。

**验证**：
```bash
ldd $SYSTEMC_AMS_HOME/liblinux64/libsystemc-ams-2.3.4.so | grep systemc
```

如果显示 `libsystemc-2.3.3.so`，则需要重新编译。

**解决方案 - 重新编译 SystemC-AMS**：

```bash
# 1. 清理旧的 SystemC-AMS 安装
rm -rf $SYSTEMC_AMS_HOME

# 2. 下载并解压 SystemC-AMS 2.3.4 源码
cd /path/to/src
wget https://github.com/Minres/SystemC-AMS/archive/refs/tags/2.3.4.tar.gz
tar xzf 2.3.4.tar.gz
mv SystemC-AMS-2.3.4 systemc-ams-2.3.4

# 3. 配置并编译
cd systemc-ams-2.3.4
mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$SYSTEMC_AMS_HOME \
    -DCMAKE_PREFIX_PATH=$SYSTEMC_HOME \
    -DBUILD_SHARED_LIBS=ON

make -j4
make install

# 4. 验证
ldd $SYSTEMC_AMS_HOME/liblinux64/libsystemc-ams-2.3.4.so | grep systemc
# 应显示链接到 libsystemc-2.3.4.so

# 5. 重新编译项目
cd /path/to/SerDesSystemCProject
rm -rf build
mkdir build && cd build
cmake -DBUILD_TESTING=ON ..
make -j4
```

**替代方案**：

如果重新编译 SystemC-AMS 困难，可以降级使用 SystemC 2.3.3：

```bash
export SYSTEMC_HOME=/usr
```

但注意：SystemC 2.3.3 可能不完全兼容项目代码（需要 C++14）。
