# Wave 模块小文档

级别：AMS 顶层模块

## 概述
生成 PRBS 与自定义比特序列，支持抖动（RJ/DJ/SJ）与调制（AM/PM/FM）注入，并可输出或后处理得到 PSD/PDF。

## 接口
- 端口：`sca_tdf::sca_out<double> out`
- 配置键：
  - `type`（string）：PRBS 类型（PRBS7/9/15/23/31 或 custom）; Single-bit Pulse
  - `poly`（string）：自定义多项式表示
  - `init`（string/int）：LFSR 初始状态
  - `seed`（int）：随机种子
  - `jitter.RJ_sigma`（[double]）：随机抖动标准差（s）
  - `jitter.SJ_freq`（[double]）：周期性抖动频率（Hz）
  - `jitter.SJ_pp`（[double]）：周期性抖动峰峰值（s）
  - `jitter.DCD`（[double]）：Duty Cycle Deviation （s）
  - `jitter.DJ`（[double]）：DJ （s）
  - `modulation.NRZ`/`PAM3`/`PAM4`/`PAM8`/`PAM16`（[double]）：调制深度/指数
  - `single_pulse`（[double]）：单脉冲宽度（s）
  - `rf`（[double]）：Rising and Falling 边沿速率（s）
  - `vpp`（[double]）：输出电压peak-peak范围（V）
  - `vcm`（[double]）：输出电压中心值（V）

## 参数
- `type`: 必须，默认 `PRBS31`/`SBR`/`custom`
- `poly`: 可选，自定义多项式（当 type=custom）
- `init`: 必须，默认最大长度序列非零初值
- `seed`: 可选，默认固定种子确保可复现
- `jitter.RJ_sigma`: 可选，默认 0
- `jitter.SJ_freq`/`jitter.SJ_pp`: 可选，默认空数组
- `jitter.DCD`: 可选，默认 0
- `jitter.DJ`: 可选，默认 0
- `modulation.NRZ`/`PAM3`/`PAM4`/`PAM8`/`PAM16`（[double]）：调制深度/指数；默认 NRZ
- `rf`（[double]）：Rising and Falling 边沿速率（s）
- `single_pulse`（[double]）：单脉冲宽度（s）
- `vpp`（[double]）：输出电压peak-peak范围（V）
- `vcm`（[double]）：输出电压中心值（V）

## 行为模型

### 1. 比特序列生成
- **PRBS 生成**：
  - 基于线性反馈移位寄存器（LFSR）实现
  - 支持标准多项式：PRBS7/9/15/23/31
  - 支持自定义多项式：通过 `poly` 参数以字符串形式指定（如 "x^7+x^6+1"）
  - 初始状态通过 `init` 参数配置（字符串或整数），默认为最大长度序列的非零初值
  - 位操作实现高效的序列生成

- **自定义比特序列**：
  - 当 `type=custom` 时，通过 `init` 配置完整的比特序列
  - 序列循环播放

- **单比特脉冲（SBR）**：
  - 当 `type=SBR` 时生成单个脉冲
  - 脉冲宽度由 `single_pulse` 参数控制（单位：秒）

### 2. 抖动注入
- **随机抖动（RJ）**：
  - 通过高斯分布采样生成时延偏移
  - 标准差由 `jitter.RJ_sigma` 控制（单位：秒）
  - 使用 `seed` 参数控制的随机数生成器确保可复现性

- **周期性抖动（SJ）**：
  - 支持多个正弦抖动分量叠加
  - `jitter.SJ_freq` 数组指定各分量频率（单位：Hz）
  - `jitter.SJ_pp` 数组指定对应的峰峰值（单位：秒）
  - 通过正弦函数调制相位/时延实现

- **占空比失真（DCD）**：
  - 通过 `jitter.DCD` 参数控制高低电平时间差（单位：秒）
  - 在时间戳生成时添加固定偏移

- **确定性抖动（DJ）**：
  - 通过 `jitter.DJ` 参数注入（单位：秒）
  - 可由链路 ISI/均衡效应产生，也可直接配置

### 3. 调制方式
- **NRZ（不归零码）**：
  - 默认调制方式
  - 二电平信号：高电平和低电平

- **多电平调制（PAM-N）**：
  - 支持 PAM-3、PAM-4、PAM-8、PAM-16
  - 通过 `modulation.PAM3/PAM4/PAM8/PAM16` 参数配置调制深度/指数
  - 将多个比特映射到对应电平

### 4. 波形成形
- **电压范围**：
  - `vpp`：峰峰值电压（单位：V）
  - `vcm`：中心电压/共模电压（单位：V）
  - 实际输出电压范围：[vcm - vpp/2, vcm + vpp/2]

- **边沿速率**：
  - `rf` 参数控制上升沿和下降沿的转换时间（单位：秒）
  - 实现有限带宽的真实信号特性

### 5. 统计分析
- **功率谱密度（PSD）**：
  - 可通过在线估计或 Python 离线计算
  - 用于验证 SJ 频点峰值和 RJ 噪底抬升

- **概率密度函数（PDF）**：
  - 用于抖动分布特性分析
  - 验证高斯分布特性（RJ）

### 6. 可复现性保证
- 所有随机过程通过 `seed` 参数控制
- 固定 `seed` 值确保相同配置下输出完全一致
- 便于调试和结果对比验证

## 依赖
- 必须：SystemC‑AMS 2.3.4
- 后处理：Python（numpy/scipy/matplotlib）
- 工具建议：Surfer 作为波形查看工具（macOS 体验更佳）

## 使用示例
1. 在配置中设置 `type=PRBS31`, `seed`, `jitter` 与 `modulation`
2. 连接 `out` 至 TX 输入
3. 运行仿真并记录 `results.dat`
4. 使用 Python 计算 PSD/PDF 并绘图

## 测试验证
- 统计一致性：均值/方差符合期望，码流平衡
- 谱验证：PSD 在 SJ 频点出现峰值；RJ 拉升噪底
- 可复现性：固定 `seed` 输出一致
- 性能：长序列与高采样率下的吞吐与内存占用评估

## 变更历史

### v0.2 (2025-10-16)
- **重大更新**：完善行为模型描述，新增详细的实现机制说明
  - 细化比特序列生成部分：PRBS、自定义序列、SBR 单脉冲
  - 扩展抖动注入说明：RJ、SJ、DCD、DJ 各类型的具体实现方法
  - 明确调制方式：NRZ 和 PAM-N 系列的电平映射
  - 补充波形成形细节：电压范围和边沿速率控制机制
  - 增加统计分析和可复现性保证说明
- 所有参数与接口定义保持一致
- 文档结构更加清晰，便于用户理解和使用

### v0.1 (初始版本)
- 初始模板，包含基本的概述、接口、参数等框架内容
- 占位性的行为模型描述
