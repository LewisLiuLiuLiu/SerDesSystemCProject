# SerDes SystemC‑AMS 建模项目说明书

本说明书用于指导基于 SystemC‑AMS 的 SerDes 行为级建模与验证，覆盖架构、模块职责、接口、数据流、配置与回归测试，以及与 Python 的后处理集成。

---

## 一、项目目标与范围
- 构建可复用的 SerDes 链路模型（以 SystemC‑AMS 为主，包含少量 SystemC DE 域模块），支持 RX、TX、Channel、Clock Generation、Wave Generation、Eye Analyzer、System Configuration 与 Adaption（信号处理算法）。
- 支持功能：
  - waveGen: PRBS波形/单比特波形生成、抖动/调制注入
  - RX: ctle、vga、sampler、DFE Summer、CDR（PI环路驱动相位插值器）
  - TX: FFE、Mux、Driver
  - Channel:基于 S 参数，支持衰减、串扰与双向传输效应
  - Clock Generation:可选 PLL/ADPLL（PD/CP/LF/VCO/Divider 等组件）
  - Python EyeAnalyzer:眼图度量、抖动分解、PSD、概率密度、线性度等（Python 后处理组件）
  - System Configuration:统一参数管理与场景切换
  - Adaption：SystemC DE 域算法库，与 AMS 域通过 DE‑TDF 桥接

---

## 二、技术栈与环境
- 语言/库：C++14，SystemC‑2.3.4，SystemC‑AMS‑2.3.4
- 建模域：TDF 为主（线性/非线性离散时间建模），必要时使用 LSF/ELN；DE 域用于控制/算法与 AMS 域桥接
- 构建：CMake/Makefile 双支持，链接已安装的 SystemC 与 SystemC‑AMS 库
- 仿真数据输出：SystemC‑AMS 默认以表格格式（tabular）存储；使用 `sca_create_tabular_trace_file()` 生成 `.dat` 文件
- Python 后处理：numpy、scipy、matplotlib（如需 Touchstone S 参数离线拟合可用 scikit‑rf）

---

## 三、项目目录结构建议
- 目录：
  - `include/`：公共头文件、类型与参数定义
  - `src/ams/`：TDF/LSF/ELN 模块（RX/TX/Channel/ClockGen/WaveGen/EyeAnalyzer/Adaption）
  - `src/de/`：SystemC DE 域模块（Adaption、CDR 控制器、配置加载器）
  - `tb/`：顶层系统、激励与监视、场景脚本
  - `tools/`：S 参数拟合、数据转换、统计分析脚本
  - `config/`：JSON/YAML 系统配置模板
  - `regression/`：回归测试配置、脚本与报告
  - `scripts/`：构建、运行、批处理
  - `build/`：编译输出（debug/release）
  - `docs/`：设计文档、API 文档、使用指南

---

## 四、最高级别类清单
- SystemConfiguration（系统配置与参数分发，DE/TDF 桥接）
- WaveGeneration（波形/PRBS 与抖动/调制生成，TDF）
- ClockGeneration（PLL/ADPLL 时钟产生，TDF/混合）
- TX（包含 FFE、Mux、Driver 的发送端，TDF）
- Channel（S 参数通道，支持衰减/串扰/双向，TDF/LSF）
- RX（包含 CTLE、VGA、Sampler、DFE、CDR 的接收端，TDF/部分 DE）
- Adoption（SystemC DE 域算法库，与 AMS 域桥接）

---

## 五、模块设计与接口

### 1) System Configuration（DE/TDF 桥接）
- 职责：集中管理所有模块参数与场景（JSON/YAML 加载，必要时支持热切换）。
- 关键参数：
  - 全局：采样率 Fs、单位间隔 UI、比特率、仿真时长、种子 Seed
  - Wave：PRBS 阶数（默认 PRBS31）、多项式、初始状态、抖动模型（RJ、DJ、SJ）、调制类型（AM/PM/FM）
  - TX：FFE 抽头系数、Mux 选择、Driver 输出摆幅/带宽/非线性参数
  - Channel：S 参数文件（.sNp）、端口拓扑、耦合矩阵、双向传输开关
  - RX：CTLE 零极点、VGA 增益、Sampler 阈值/迟滞、DFE 抽头与更新算法
  - CDR：PI 环路系数、相位插值器分辨率与范围
  - Clock：PLL/ADPLL 选择，PD/CP/LF/VCO/Divider 参数
  - Eye：采样网格、UI 窗口、统计长度
- 接口：DE 域配置模块向 TDF 模块广播参数，使用 `sca_de::sca_out` 与 TDF 端口桥接。

### 2) Wave Generation（TDF）
- 职责：生成 PRBS 及其它自定义比特序列，注入抖动/调制；输出基带或已成形的电压信号；实时统计 PSD/PDF（可选在线或离线）。
- 功能：
  - PRBS：支持 PRBS7/9/15/23/31 与自定义多项式/初值；LFSR 实现；可变 Seed
  - 抖动：随机抖动（高斯 RJ）、确定性抖动（DDJ）、周期性抖动（SJ，多音可叠加）
  - 调制：NRZ/PAM-N
  - 输出统计：在线估计 PSD（Welch/periodogram）与 PDF（直方图），或将原始波形输出由 Python 后处理
- 端口：`sca_tdf::sca_out<double> out`；可选 `sca_tdf::sca_out<double> meta_psd/meta_pdf`

### 3) Clock Generation（TDF/混合）
- 职责：提供采样时钟；支持 PLL 或 ADPLL。
- PLL 组件：PD、CP、LF、VCO、Divider；支持参考抖动、VCO 相位噪声、CP 失配/泄漏、LF 热噪。
- ADPLL 组件：数字 PD、TDC、数字 LF、DCO；量化噪声与非线性。
- 端口：`sca_tdf::sca_out<double> clk_phase` 或 `sca_tdf::sca_out<bool> clk`；通过配置选择 PLL/ADPLL。

### 4) TX（TDF）
- FFE：前置均衡，FIR 结构；抽头系数可配置；支持符号速率变化。
- Mux：Lane/通道选择与复用。
- Driver：非线性饱和、输出摆幅、带宽限制（`ltf_nd` 线性传递）、共模/差分支持。
- 端口：`sca_tdf::sca_in<double> in`；`sca_tdf::sca_out<double> out`。

### 5) Channel（TDF/LSF）
- 职责：依据 S 参数描述通道，包括衰减、串扰与双向效应。
- 实现建议：
  - 多端口 S‑参数：离线向量拟合（Vector Fitting）将 Sij(f) 近似为有理函数，得到因果稳定的 LTI 模型。
  - 时域实现：将每条 Sij 映射为传递函数系数，使用 `sca_tdf::sca_ltf_nd` 实现滤波。
  - 串扰：在 N 端口场景下，耦合矩阵对所有输入输出做线性组合。
  - 双向：S11/S22 反射、S12/S21 反向传输可开关与参数化。
- 端口：`sca_tdf::sca_in<double> in[N]`；`sca_tdf::sca_out<double> out[N]`（最小 2 端口）；支持差分对。

### 6) RX（TDF/部分 DE）
- CTLE：差分输入/差分输出；一个零点与两个极点；支持输入偏移与噪声开关（`offset_enable`/`noise_enable`），输出饱和（`sat_min`/`sat_max`），以及可配置差分输出共模电压（`vcm_out`）；`sca_tdf::sca_ltf_nd` 实现滤波。
- VGA：可编程增益，支持 AGC（可与 Adoption/DE 域交互）。
- Sampler：依据时钟相位采样；阈值与迟滞可配置；输出判决比特。
- DFE Summer：反馈均衡，FIR 结构；支持抽头在线更新（LMS/Sign‑LMS 等）。
- CDR（PI 控制环）：DE 域 PI 控制器根据采样误差调整相位插值器；TDF‑DE 桥接相位命令。
- 端口：`sca_tdf::sca_in<double> in`；`sca_tdf::sca_in<double> clk_phase`；`sca_tdf::sca_out<bool> bit_out`；内部 DFE/AGC 与 Adoption 交互。

### 7) EyeAnalyzer（Python 后处理组件）
- 职责：生成眼图与相关统计，输出眼高/眼宽、开口面积、抖动分解（RJ/DJ/Total）、线性度等。
- 实现：收集多 UI 周期样本，构建二维密度（时间相位 vs 电平）；RJ/DJ 分离可用统计/谱域方法。
- 输入/输出：读取 `results.dat` 或波形数组，输出 `eye_metrics.json` 或图像/CSV；由 Python 实现，不参与 AMS 域类建模。

### 8) Adoption（SystemC DE 域）
- 职责：承载信号处理算法（AGC、DFE 抽头更新、CDR PI 策略、阈值自适应等），以 DE 计算与 TDF 交互。
- 接口桥接：使用 `sca_de::sca_in/out` 与 TDF 域信号相连；周期性/事件驱动更新参数。

---

## 六、数据记录与 Python 后处理
- 仿真记录：使用 `sca_create_tabular_trace_file("results.dat")`；对关键信号注册 trace（波形、时钟相位、判决比特、指标）。
- Python 处理（示例流程）：
  - 读取 `.dat`（列：time,value,...）
  - PSD：`scipy.signal.welch` 或 `numpy.fft` 功率谱估计
  - PDF：`numpy.histogram` 统计电平分布；眼图二维密度可将采样点映射到相位窗
  - 可视化：`matplotlib` 绘制波形、PSD、PDF、眼图热力图；推荐使用 Surfer 查看波形
- 注意：
  - 保证采样率与 UI 匹配；使用足够长时间窗口获取稳态统计
  - 大文件按块处理或下采样以避免内存压力

---

## 七、配置示例（JSON 示意）
```json
{
  "global": { "Fs": 80e9, "UI": 2.5e-11, "duration": 1e-3, "seed": 12345 },
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF",
    "jitter": { "RJ_sigma": 5e-13, "SJ_freq": [5e6], "SJ_pp": [2e-12] },
    "modulation": { "AM": 0.1, "PM": 0.01 }
  },
  "tx": { "ffe_taps": [0.2, 0.6, 0.2], "mux_lane": 0, "driver": { "swing": 0.8, "bw": 20e9, "nonlinear": { "sat": 1.0 } } },
  "channel": { "touchstone": "chan_4port.s4p", "ports": 2, "crosstalk": true, "bidirectional": true },
  "rx": {
    "ctle": { "zeros": [2e9], "poles": [30e9], "dc_gain": 1.5 },
    "vga_gain": 4.0,
    "sampler": { "threshold": 0.0, "hysteresis": 0.02 },
    "dfe": { "taps": [-0.05, -0.02, 0.01], "update": "sign-lms", "mu": 1e-4 }
  },
  "cdr": { "pi": { "kp": 0.01, "ki": 1e-4 }, "pai": { "resolution": 1e-12, "range": 5e-11 } },
  "clock": { "type": "PLL", "pd": "tri-state", "cp": { "I": 5e-5 }, "lf": { "R": 10000, "C": 1e-10 }, "vco": { "Kvco": 1e8, "f0": 1e10 }, "divider": 4 },
  "eye": { "ui_bins": 128, "amp_bins": 128, "measure_length": 1e-4 }
}
```

---

## 八、关键实现建议
- PRBS/LFSR：位操作实现，支持多项式表驱动；保证最大长度与去同步机制。
- 抖动注入：
  - RJ：白噪高斯采样叠加相位扰动，考虑采样折返
  - SJ：正弦相位/时延调制，支持多音
  - DJ：数据相关偏移可由 FFE/DFE 与 ISI 自然产生，亦可额外注入
- 线性滤波：
  - `sca_tdf::sca_ltf_nd`：将传递函数系数提供给 SystemC‑AMS 滤波器
  - S 参数：离线向量拟合得到有理函数，确保因果与稳定；高阶模型需注意数值条件
- CDR PI 环：
  - 误差检测：早/晚采样、相位比较或 Bang‑Bang
  - PI 参数：依据环路带宽与阻尼系数设计，考虑量化与噪声
  - 相位插值器：步长分辨率与范围约束，避免环路饱和
- 眼图与抖动分解：
  - RJ/DJ 分离：QQ 图或谱域分析；总抖动 TJ 在给定 BER 下由统计模型估计
  - 面积/线性度：对密度图进行等值线/拟合，计算开口面积与线性拟合误差

---

## 九、验证策略与回归
- 单元测试：
  - Wave：不同 PRBS 与抖动/调制组合的统计一致性（均值、方差、谱峰）
  - Filter：CTLE/FFE/Channel 传递函数的幅相响应对比基准
  - CDR：阶跃/频响测试、稳定性与锁定时间
  - Eye：标准场景的眼高/眼宽阈值
- 集成仿真：
  - 标准链路场景（短/长通道、强/弱串扰、双向开关）
  - PVT 与参数扫频
- 回归：
  - `regression/tests`：场景配置集 + 期望统计指标范围
  - 自动报告（PSD 峰值、TJ@1e‑12 BER 估计、开口面积）

---

## 十、数据流与运行流程
- 步骤：
  1) 准备 config JSON/YAML
  2) 编译运行顶层 tb，SystemConfig 加载参数并分发
  3) 生成波形 → TX → Channel → RX
  4) trace 输出 `.dat`
  5) Python 读取并计算 PSD/PDF 与绘图
- 输出：
  - 波形：`results.dat`（时间/信号列）
  - 指标：`eye_metrics.json`（可选）、`psd.csv`、`pdf.csv`（可选）
- 性能与数值：
  - 平衡采样率与滤波器阶数，防止过高阶造成数值不稳
  - 长仿真分段写出 trace，降低内存占用

---

## 十一、接口与类命名建议（示例）
- WaveGenerationTdf
  - out: `sca_tdf::sca_out<double>`
  - params: `prbsOrder`, `poly`, `seed`, `jitterCfg`, `modulationCfg`
- ClockPllTdf / ClockAdpllTdf
  - out: `sca_tdf::sca_out<double> clkPhase`
- TxFfeTdf / TxMuxTdf / TxDriverTdf
  - in/out: `sca_tdf::sca_in/out<double>`
- ChannelSParamTdf
  - in[N], out[N]: `sca_tdf::sca_in/out<double>`
  - params: `sNpFile`, `couplingMatrix`, `bidirectional`
- RxCtletdf / RxVgaTdf / RxSamplerTdf / RxDfeTdf
  - in/out: `sca_tdf::sca_in/out<double>` 或 bit 流
  - extra: `sca_tdf::sca_in<double> clkPhase`
- CdrPiDe
  - in: `sca_de::sca_in<double> phaseError`
  - out: `sca_de::sca_out<double> phaseCmd`
- AdoptionDe（算法库）
  - 方法：`agcUpdate`, `dfeUpdate`, `thresholdAdapt`, `piTune`
  - 桥接：`sca_de::sca_in/out`

---

## 十二、S 参数处理与工具链
- 输入：Touchstone `.sNp` 文件（2 端口/多端口）。
- 流程：
  - 离线：`tools/sparam_fit.py` 对每个 Sij(f) 做向量拟合 → 有理函数系数（分子/分母）。
  - 生成：把系数存入 `config/channel_filters.json`。
  - 在线：`ChannelSParamTdf` 加载系数 → `sca_ltf_nd` 滤波。
- 串扰：对所有输入向量乘以耦合矩阵后再进入各自滤波器。
- 双向：依据开关启用 S12/S21 与反射 S11/S22 路径。

---

## Python EyeAnalyzer API 概览
- 函数接口（建议）：
  - `analyze_eye(dat_path: str, ui: float, ui_bins: int = 128, amp_bins: int = 128, measure_length: float | None = None, target_ber: float = 1e-12, sampling: str = 'phase-lock') -> dict`
  - 输入：`dat_path`（或波形数组）、`ui`、`ui_bins/amp_bins`、`measure_length`、`target_ber`、`sampling`
  - 输出：指标字典并可写出 `eye_metrics.json` / 图像 / CSV
- 命令行（占位建议）：
  - `python analyze_eye.py --dat results.dat --ui 2.5e-11 --ui-bins 128 --amp-bins 128 --target-ber 1e-12`

## 数据格式与 Schema
- `results.dat`（tabular）：
  - 列定义：`time`（s）, `value`（V）[, 其他信号列可选]
  - 分段写出时需记录时间基准与段索引
- `eye_metrics.json`（建议字段）：
  - `eye_height`（V）, `eye_width`（UI）, `eye_area`（V*UI）
  - `rj_sigma`（s）, `dj_pp`（s）, `tj_at_ber`（s）
  - `linearity_error`（V 或归一化）, `ui_bins`, `amp_bins`, `measure_length`
  - `hist2d`（可选，摘要或文件路径）, `pdf`/`psd`（可选）


```python
# 读取
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t, x = np.loadtxt('results.dat', unpack=True)

# PSD
Fs = 80e9
f, Pxx = signal.welch(x, fs=Fs, nperseg=1<<14)
plt.semilogy(f, Pxx); plt.title('PSD'); plt.xlabel('Freq'); plt.ylabel('Power'); plt.show()

# PDF
hist, bins = np.histogram(x, bins=256, density=True)
centers = 0.5*(bins[1:]+bins[:-1])
plt.plot(centers, hist); plt.title('PDF'); plt.xlabel('Amp'); plt.ylabel('Prob'); plt.show()

# 简单眼图（二维密度）
UI = 1/40e9
phi = (t % UI) / UI
H, xe, ye = np.histogram2d(phi, x, bins=[128,128], density=True)
plt.imshow(H.T, origin='lower', aspect='auto', extent=[0,1,ye[0],ye[-1]])
plt.title('Eye Density'); plt.xlabel('UI phase'); plt.ylabel('Amp'); plt.show()
```

---

## 十四、里程碑与交付
- M1：目录初始化与构建脚本、基础 TDF 模块骨架、trace 输出
- M2：Wave/TX/RX 基本链路与 PLL 时钟闭环、单端口 Channel
- M3：S 参数拟合与多端口串扰、双向路径
- M4：Python EyeAnalyzer 与抖动分解、Adoption 算法与 DFE/CDR 联调
- M5：配置模板、回归套件与 Python 可视化工具完善、文档与示例场景

---

如需，我可以继续提供最小可运行的模块骨架（WaveGeneration、TX‑FFE、Channel（单端口 `ltf_nd`）、RX‑CTLE、EyeAnalyzer）以及基础 CMake 配置与 Python 后处理脚本模板，帮助快速起步。

---

## AMS 模块文档索引
顶层 AMS 模块：
- [WaveGen](docs/modules/waveGen.md)
- [Channel](docs/modules/channel.md)
- [ClockGen](docs/modules/clkGen.md)
- [Adaption](docs/modules/adaption.md)

顶层系统模块：
- [RX](docs/modules/rx.md)
- [TX](docs/modules/tx.md)

RX 子模块：
- [CTLE](docs/modules/ctle.md)
- [VGA](docs/modules/vga.md)
- [Sampler](docs/modules/sampler.md)
- [DFE Summer](docs/modules/dfesummer.md)
- [CDR](docs/modules/cdr.md)

TX 子模块：
- [FFE](docs/modules/ffe.md)（待补文档）
- [Mux](docs/modules/mux.md)（待补文档）
- [Driver](docs/modules/driver.md)（待补文档）

## Python 分析组件文档索引
- [EyeAnalyzer](docs/modules/EyeAnalyzer.md)

---

附录A 规范总则与术语
- 规范用语：
  - 必须：不满足则视为不合规
  - 应：推荐遵循，必要时可例外并记录
  - 可选：根据场景采纳
- 范围：适用于所有 AMS/DE 模块、测试平台、回归与后处理脚本

附录B 接口与命名规范
- 命名：
  - 类名使用 PascalCase；TDF 类以后缀 Tdf；DE 类以后缀 De
  - 端口命名：in/out/clkPhase/bitOut 等采用小驼峰
  - 配置键使用小写蛇形或小驼峰，与 JSON 保持一致
- 接口：
  - 所有 TDF 模块必须定义明确采样率与时序约束
  - DE‑TDF 桥接必须声明驱动频率与事件触发条件
- 错误与异常：
  - 配置缺失必须失败并输出明确诊断信息
  - 参数越界应给出警告并采用安全回退值

附录C 文件与目录规范
- 按 SystemC 项目标准化目录（common/basic-examples/advanced-projects/verification/regression/tools/templates/build）
- 本项目位于 advanced-projects/serdes；公共组件放入 common/
- 配置文件放置于 config/，命名为 scene_*.json 或 *.yaml

附录D 构建与依赖规范
- CMake 工程必须链接已安装的 SystemC/SystemC‑AMS，不重复安装
- 编译标准 C++14；Clang/GCC 均支持
- 提供 Makefile 快捷目标：build/run/clean/test

附录E 测试与回归规范
- 单元测试必须覆盖：Wave/TX/RX/Channel/Clock/Eye/Adoption 的核心接口与边界条件
- 集成场景必须覆盖：短/长通道、强/弱串扰、双向开关、不同 PRBS 与抖动组合
- 回归报告必须包含：PSD 峰值、眼图开口（高/宽/面积）、TJ@目标 BER、锁定时间

附录F 数据输出与格式规范
- 所有 trace 必须通过 `sca_create_tabular_trace_file()` 输出 `.dat`
- 列顺序与名称必须在 docs/data_schema.md 说明并在代码中统一注册
- 大规模仿真应分段写出并记录段索引与时间基准

附录G 配置格式与校验
- JSON/YAML 必须通过 schema 校验（键存在性、类型、范围）
- 默认场景必须提供 PRBS31 与基础 PLL/Channel 参数
- 自定义 PRBS 与调制必须显式指定多项式、种子与频率表
