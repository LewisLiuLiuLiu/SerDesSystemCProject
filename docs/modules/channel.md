# Channel 模块小文档

级别：AMS 顶层模块

## 概述
基于 Touchstone S 参数建模的通道模块，支持衰减、串扰及双向传输效应。提供两种实现方法：
1. **有理函数拟合法**（推荐）：通过向量拟合将 S 参数转换为有理函数，使用 `sca_tdf::sca_ltf_nd` 实现高效滤波
2. **冲激响应卷积法**：通过逆傅立叶变换获得时域冲激响应，使用延迟线卷积实现

## 接口
- 端口：
  - 输入：`sca_tdf::sca_in<double> in[N]`（支持差分对）
  - 输出：`sca_tdf::sca_out<double> out[N]`
- 配置键：
  - `touchstone`（string）：S 参数文件路径（.sNp）
  - `ports`（int）：端口数（N >= 2）
  - `method`（string）：处理方法，"rational"（有理函数）或 "impulse"（冲激响应）
  - `config_file`（string）：处理后的配置文件路径（如 `config/channel_filters.json`）
  - `crosstalk`（bool）：是否启用串扰耦合矩阵
  - `bidirectional`（bool）：是否启用 S12/S21 与反射 S11/S22
  - `fit`（object）：向量拟合参数（仅用于 rational 方法）
    - `order`（int）：拟合阶数，建议 6-16
    - `enforce_stable`（bool）：强制稳定性约束
    - `enforce_passive`（bool）：强制无源性约束
    - `band_limit`（float）：频段上限（Hz）
  - `impulse`（object）：冲激响应参数（仅用于 impulse 方法）
    - `time_samples`（int）：采样点数，建议 2048-8192
    - `causality`（bool）：是否应用因果性窗函数
    - `truncate_threshold`（float）：截断阈值（相对幅度）
  - `gpu_acceleration`（object）：GPU 加速配置（仅用于 impulse 方法，**仅支持 Apple Silicon**）
    - `enabled`（bool）：是否启用 GPU 加速
    - `backend`（string）：后端选择，当前仅支持 "metal"
    - `algorithm`（string）：算法选择，"direct"/"fft"/"auto"
    - `batch_size`（int）：批处理大小
    - `fft_threshold`（int）：L > threshold 时使用 FFT 卷积

## 参数
- `touchstone`: 必须，.sNp 文件路径
- `ports`: 必须，端口数（N）
- `method`: 必须，默认 "rational"
- `config_file`: 必须，离线处理生成的配置文件
- `crosstalk`: 可选，默认 false
- `bidirectional`: 可选，默认 false
- `fit.order`: 可选（rational 方法），默认 16
- `fit.enforce_stable`: 可选，默认 true
- `fit.enforce_passive`: 可选，默认 true
- `fit.band_limit`: 可选，默认使用 Touchstone 文件的最高频率
- `impulse.time_samples`: 可选（impulse 方法），默认 4096
- `impulse.causality`: 可选，默认 true
- `impulse.truncate_threshold`: 可选，默认 1e-6
- `gpu_acceleration.enabled`: 可选，默认 false（**仅 Apple Silicon 可用**）
- `gpu_acceleration.backend`: 固定为 "metal"（唯一支持的后端）
- `gpu_acceleration.algorithm`: 可选，默认 "auto"（根据 L 自动选择）
- `gpu_acceleration.batch_size`: 可选，默认 1024
- `gpu_acceleration.fft_threshold`: 可选，默认 512

## 行为模型

### 方法一：有理函数拟合法（推荐）

#### 1. 离线处理（Python）
- **向量拟合**：
  - 使用向量拟合算法（Vector Fitting）对每个 Sij(f) 进行有理函数近似
  - 拟合形式：`H(s) = Σ(r_k / (s - p_k)) + d + s*h`
  - 极点 p_k 和留数 r_k 通过迭代优化获得
  - 强制约束：
    - 稳定性：所有极点实部 < 0
    - 无源性：确保能量守恒（可选）
    - 因果性：自动满足

- **传递函数转换**：
  - 将极点-留数形式转换为分子/分母多项式
  - `H(s) = (b_n*s^n + ... + b_1*s + b_0) / (a_m*s^m + ... + a_1*s + a_0)`
  - 归一化分母首项为 1

- **配置导出**：
  - 保存为 JSON 格式：`{"filters": {"S21": {"num": [...], "den": [...]}, ...}}`
  - 包含拟合质量指标（MSE、最大误差等）

#### 2. 在线仿真（SystemC-AMS）
- **LTF 滤波器实例化**：
  - 使用 `sca_tdf::sca_ltf_nd(num, den, timestep)` 创建线性时不变滤波器
  - SystemC-AMS 自动处理状态空间实现和数值积分

- **多端口处理**：
  - 为每个 Sij 创建独立的 `sca_ltf_nd` 实例
  - N×N 端口矩阵：需要 N² 个滤波器（可根据对称性优化）

- **性能优势**：
  - 紧凑表示：8 阶滤波器仅需 ~20 个系数
  - 计算高效：O(order) 每时间步
  - 数值稳定：SystemC-AMS 内置优化

### 方法二：冲激响应卷积法

#### S 参数预处理：DC 值补全与采样频率匹配（Impulse 方法）

- **背景与必要性**：
  - Touchstone 文件（.sNp）常缺少 0 Hz（DC）点，直接 IFFT 会导致时域响应出现直流偏置和长尾振铃，破坏因果性
  - IFFT 需要在与系统采样频率 fs 一致的均匀频率网格上进行；若测量频率非均匀或上限超出 Nyquist，会出现混叠或泄漏

- **技术可行性**：
  - DC 值补全可通过向量拟合（Vector Fitting，VF）在连续 s 域估算 H(0)，并在拟合中施加稳定/无源约束，稳健性最好
  - 低频插值（对最后若干低频点进行幅相外推）方法简单，但易引入偏差与振铃；仅在数据质量较好且带宽较低时建议
  - 借助端口阻抗与等效 RLC 模型推断 DC（将 S 转 Y/Z 后估算），对通用通道泛化性不足，不作为默认方案

- **推荐实现方案（离线阶段）**：
  1. 读取 S(f)，进行带外清理：设置 band_limit ≤ fs/2（Nyquist），超出部分滚降或设为 0
  2. DC 补全：
     - 首选 VF 法：对各 Sij(f) 进行 VF，启用稳定性/无源性约束，评估 H(0) 作为 DC 点并补入
     - 备选插值法：对最低频点附近进行幅相外推，注意保持相位连续与因果性（风险较高）
  3. 构建目标 fs 的均匀频率网格：f_k = k·Δf，Δf = fs/N，0≤k≤N/2，其中 N 对应冲激长度（与 time_samples 一致或更大）
  4. 获得网格上的 Sij(f_k)：
     - 插值路径：对复数 S(f) 进行样条/分段线性插值（幅相连续、避免过度拟合）
     - VF 评估路径：直接用 VF 的有理函数在 f_k 上评估（稳健性优于插值，推荐）
  5. 负频率镜像、IFFT、因果性窗与尾部截断（truncate_threshold），得到 h(t)
  6. 验证：检查能量守恒（无源性）、相位连续性、时域零偏差与长尾抑制

- **采样频率与 VF 的关系**：
  - VF 工作在连续 s 域，参数与 fs 无关；不需要在拟合阶段"匹配采样频率"
  - 但拟合点的频率密度应覆盖到 Nyquist(fs/2)，并在高梯度区加密采样，以保证对目标 fs 的评估精度
  - 实践建议：按目标 fs 设置 band_limit ≤ fs/2；测量上限低于 fs/2 时，避免外推到 Nyquist，宁可降低 fs 或采用滚降策略

- **风险与规避**：
  - 错误 DC 导致时域直流偏移：用 VF+无源约束估计 DC；必要时仅对 S11/S22 施加更强约束
  - 插值振铃与谱泄漏：优先 VF 评估；在频域增设平滑窗或带外滚降；时域使用因果性窗并截断尾部
  - 混叠风险：严格限制 band_limit ≤ fs/2；不满足时降低 fs 或增大 N

- **预处理配置建议（文档层面，暂不改代码）**：
  - `impulse.dc_completion`: "vf" | "interp" | "none"（默认 "vf"）
  - `impulse.resample_to_fs`: true/false（默认 true）
  - `impulse.fs`: 采样频率（Hz）
  - `impulse.band_limit`: 频段上限（默认 Touchstone 最高频或设置为 ≤ fs/2）
  - `impulse.grid_points`: 频率网格点数 N（与 time_samples 对应）

#### 1. 离线处理（Python）
- **逆傅立叶变换**：
  - 读取 S 参数频域数据 Sij(f)
  - 构造双边频谱（负频率为正频率的共轭）
  - 应用 IFFT：h(t) = IFFT[Sij(f)]
  - 取实部并确保因果性

- **因果性处理**：
  - 检测峰值位置，确保 t < 0 部分能量接近零
  - 可选：应用最小相位变换
  - 可选：Hilbert 变换构造因果响应

- **截断与优化**：
  - 识别冲激响应长尾衰减阈值
  - 截断低于阈值的部分，减少卷积长度
  - 应用窗函数（如 Hamming）减少截断效应

- **配置导出**：
  - 保存时间轴、冲激响应数组和采样间隔
  - JSON 格式：`{"impulse_responses": {"S21": {"time": [...], "impulse": [...], "dt": ...}}}`

#### 2. 在线仿真（SystemC-AMS）
- **延迟线卷积**：
  - 维护输入历史：`delay_line[0..L-1]`，L 为冲激响应长度
  - 每时间步：`y(n) = Σ h(k) * x(n-k)`
  - 使用循环缓冲区优化内存访问

- **快速卷积（可选）**：
  - 对于长冲激响应（L > 512），可使用 overlap-add FFT 卷积
  - 需要外部 FFT 库（如 FFTW）
  - 块处理：缓冲输入块 → FFT → 频域乘法 → IFFT → overlap-add

- **性能考虑**：
  - 时间复杂度：O(L) 每时间步（直接卷积）或 O(L log L) 分摊（FFT）
  - 空间复杂度：O(L) 延迟线存储
  - 适用于 L < 1000 的中短通道

#### 3. GPU 加速（可选，仅 Apple Silicon）

- **系统要求**：
  - **必须**：Apple Silicon（M1/M2/M3 等 ARM64 架构）Mac 电脑
  - **不支持**：Intel Mac、Linux、Windows 系统
  - 其他 GPU 后端（CUDA、OpenCL、ROCm）在当前实现中不受支持

- **适用场景**：
  - 长冲激响应（L > 512）
  - 多端口仿真（N > 2）
  - 高采样率场景（> 100 GS/s）

- **直接卷积加速**（L < 512）：
  - 将卷积计算卸载到 GPU
  - 每个输出样本并行计算
  - 性能提升：50-100x（Metal on Apple Silicon）

- **FFT 卷积加速**（L > 512）：
  - 利用 Metal Performance Shaders（MPS）
  - 卷积定理：`y = IFFT(FFT(x) ⊙ FFT(h))`
  - 预计算冲激响应的 FFT，仅需一次
  - 性能提升：200-500x（批处理模式可达 1000x）

- **批处理策略**：
  - 收集一批输入样本（如 1024 个）
  - 一次上传到 GPU，减少延迟
  - GPU 并行计算所有输出
  - 下载结果并顺序输出

- **后端说明**：
  - **Metal**：当前唯一支持的 GPU 后端，Apple Silicon 专属优化
  - ~~OpenCL~~：暂不支持
  - ~~CUDA~~：暂不支持（需 NVIDIA GPU）
  - ~~ROCm~~：暂不支持（需 AMD GPU）

### 串扰建模
- **耦合矩阵**：
  - N 端口输入向量 `x[N]` 通过耦合矩阵 `C[N×N]` 线性组合
  - `x'[i] = Σ C[i][j] * x[j]`
  - 耦合后信号进入各自的 Sii/Sij 滤波器

- **提取方法**：
  - 从 S 参数矩阵提取交叉项 Sij (i≠j)
  - 近端串扰（NEXT）：S13, S14 等
  - 远端串扰（FEXT）：S23, S24 等

#### S 参数端口映射的标准化处理

- **问题描述**：
  - 不同来源的 .sNp 端口顺序与配对关系不统一（例如 s4p 中端口1可能对应端口2或端口3），会导致正向传输与串扰项被错误识别，进而影响 s2d/crosstalk 分析的正确性

- **技术可行性**：
  - 通过置换矩阵对端口进行重排，可对每个频点的 S 矩阵做统一标准化：对端口顺序施加同一置换 P，得到 S'(f) = P · S(f) · P^T（等价于同时对行列按一致的端口重排）
  - 手动指定与自动识别两种路径均可实现，且与现有串扰分析流程兼容

- **实现方案**：
  - **手动指定映射（推荐提供）**：
    - 在配置中允许用户明确端口分组与方向，例如差分对、输入/输出端口配对、主传输路径
    - 处理器依据配置构造置换矩阵 P，对所有频点的 S(f) 进行重排，确保后续分析的端口序一致
  - **自动识别（启发式，提供为辅助）**：
    - 计算各 Sij 的通带能量或平均幅度：Eij = ∫ |Sij(f)|^2 df，用于评估强传输路径
    - 对差分场景：依据耦合与串扰强度，识别相邻端口形成的差分对；利用 S11/S22 与互耦指标验证合理性
    - 构建加权图：节点为端口，边权为 Eij；使用最大匹配或最大权匹配，选取最可能的输入→输出配对
    - 验证准则：标准化后主路径（如 S21）应显著高于非主路径；NEXT/FEXT 分类与物理预期一致
  - **冲突与回退**：
    - 对称网络或多条强路径可能导致不唯一映射；提供置信度与候选方案，允许用户锁定或覆写部分端口
    - 映射生效后进行无源性与对称性检查，若不满足则回退到手动映射或提示用户确认

- **与串扰分析的关系**：
  - 标准化后的端口序确保 s2d/crosstalk 结果可比性与稳定性；避免因文件端口顺序差异引入的指标偏差

- **配置建议（文档层面，暂不改代码）**：
  - `port_mapping.enabled`: true/false
  - `port_mapping.mode`: "manual" | "auto"
  - `port_mapping.manual.pairs`: [[1,2],[3,4]]（差分对或端口分组）
  - `port_mapping.manual.forward`: [[1,3],[2,4]]（输入→输出配对）
  - `port_mapping.auto.criteria`: "energy" | "lowfreq" | "bandpass"
  - `port_mapping.auto.constraints`: { differential: true/false, bidirectional: true/false }

- **验证建议**：
  - 对多来源的 s4p 执行标准化后，比较主传输曲线与串扰分类的一致性；出现差异时复核自动识别的置信度并考虑手动覆写

### 双向传输
- **正向路径**：S21（端口1 → 端口2）
- **反向路径**：S12（端口2 → 端口1）
- **反射**：S11（端口1输入反射）、S22（端口2输入反射）
- **开关控制**：
  - `bidirectional=true`：启用 S12 和反射项
  - `bidirectional=false`：仅使用 S21，单向简化模型

### 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 长通道（>10 GHz 带宽） | Rational | 拟合紧凑，仿真快速 |
| 短通道（< 5 GHz） | 两者均可 | Impulse 更直观，Rational 更高效 |
| 高阶效应（非最小相位） | Impulse | 保留完整频域信息 |
| 快速参数扫描 | Rational | 重新拟合开销小 |
| 验证与调试 | 两者对比 | 交叉验证拟合精度 |
| **超长通道（L > 2048，Apple Silicon）** | **Impulse + GPU** | **Metal GPU 加速弥补计算开销** |
| **多端口高速场景（Apple Silicon）** | **Impulse + GPU FFT** | **批处理效率极高** |

### 性能对比表

假设 4 端口 S 参数，冲激响应长度 L=2048，**在 Apple Silicon Mac 上测试**：

| 实现方式 | 每秒处理样本数 | 相对速度 | 内存占用 | 系统要求 |
|---------|--------------|---------|----------|----------|
| Rational（CPU 8阶） | ~10M samples/s | **1000x** | ~1 KB | 通用 |
| Impulse（CPU 单核） | ~100K samples/s | 1x | ~16 KB | 通用 |
| Impulse（CPU 8核） | ~600K samples/s | 6x | ~16 KB | 通用 |
| **Impulse（Metal 直接）** | ~5M samples/s | **50x** | ~20 KB | **Apple Silicon** |
| **Impulse（Metal FFT）** | ~20M samples/s | **200x** | ~32 KB | **Apple Silicon** |

**注意**：GPU 加速性能数据仅适用于 Apple Silicon（M1/M2/M3）Mac 电脑。

## 依赖

### Python 工具链
- **必须**：
  - `numpy`：数值计算
  - `scipy`：信号处理、IFFT、向量拟合
  - `scikit-rf`：Touchstone 文件读取与 S 参数操作
- **可选**：
  - `vectfit3`：专业向量拟合库
  - `matplotlib`：频响/冲激响应可视化

### SystemC-AMS
- **必须**：SystemC-AMS 2.3.4
- **可选**：FFTW3（CPU 快速卷积）

### GPU 加速运行时（仅 Apple Silicon）
- **Metal**（macOS Apple Silicon）：
  - Metal Framework（系统自带）
  - Metal Performance Shaders（系统自带）
  - 支持架构：Apple M1/M2/M3 及后续芯片

**暂不支持的后端**：
- ~~OpenCL~~：未在当前实现中支持
- ~~CUDA~~（NVIDIA GPU）：不适用于 Apple Silicon
- ~~ROCm~~（AMD GPU）：不适用于 Apple Silicon

### 配置文件
- `config/channel_filters.json`（rational 方法）
- `config/channel_impulse.json`（impulse 方法）

## 使用示例

### 离线处理流程

```bash
# 1. 准备 S 参数文件
cp path/to/channel.s4p data/

# 2. 生成有理函数配置
python tools/sparam_processor.py \
  --input data/channel.s4p \
  --method rational \
  --order 8 \
  --output config/channel_filters.json

# 3. 生成冲激响应配置（可选，用于对比）
python tools/sparam_processor.py \
  --input data/channel.s4p \
  --method impulse \
  --samples 4096 \
  --output config/channel_impulse.json

# 4. 验证拟合质量
python tools/verify_channel_fit.py \
  --sparam data/channel.s4p \
  --rational config/channel_filters.json \
  --impulse config/channel_impulse.json \
  --plot results/channel_verification.png
```

### 系统配置示例

```json
{
  "channel": {
    "touchstone": "data/channel.s4p",
    "ports": 2,
    "method": "rational",
    "config_file": "config/channel_filters.json",
    "crosstalk": false,
    "bidirectional": true,
    "fit": {
      "order": 8,
      "enforce_stable": true,
      "enforce_passive": true,
      "band_limit": 25e9
    }
  }
}
```

### 系统配置示例（GPU 加速，Apple Silicon）

``json
{
  "channel": {
    "touchstone": "data/long_channel.s4p",
    "ports": 4,
    "method": "impulse",
    "config_file": "config/channel_impulse.json",
    "crosstalk": true,
    "bidirectional": true,
    "impulse": {
      "time_samples": 4096,
      "causality": true,
      "truncate_threshold": 1e-6
    },
    "gpu_acceleration": {
      "enabled": true,
      "backend": "metal",
      "algorithm": "auto",
      "batch_size": 1024,
      "fft_threshold": 512
    }
  }
}
```

**注意**：此配置仅在 Apple Silicon Mac 上有效。在其他平台上应将 `gpu_acceleration.enabled` 设为 `false`。

### SystemC-AMS 实例化

```cpp
// 创建 Channel 模块
ChannelModel channel("channel");
channel.config_file = "config/channel_filters.json";
channel.method = "rational";
channel.load_config();

// 连接信号
channel.in(tx_out);
channel.out(rx_in);
```

## 测试验证

### 1. 频响校验
- **目标**：验证时域实现与原始 S 参数频域一致性
- **方法**：
  - 输入扫频正弦信号
  - 记录幅度/相位响应
  - 与 Touchstone 文件绘图对比
- **指标**：
  - 幅度误差 < 0.5 dB（通带内）
  - 相位误差 < 5°（通带内）

### 2. 冲激响应对比
- **目标**：两种方法结果一致性
- **方法**：
  - Rational 方法：激励冲激 → 记录响应
  - Impulse 方法：直接输出预计算响应
  - 计算互相关和均方误差
- **指标**：
  - 归一化 MSE < 1%
  - 峰值时刻偏差 < 1 采样周期

### 3. 串扰场景
- **目标**：多端口耦合正确性
- **方法**：
  - 在端口1输入 PRBS，端口2观察串扰
  - 测量 NEXT/FEXT 比值
- **指标**：
  - 串扰幅度与 S13/S14 一致（±2 dB）

### 4. 双向传输
- **目标**：验证 S12/S21 和反射项
- **方法**：
  - 启用/禁用 bidirectional 开关
  - 对比输出差异
  - 测量反射系数（输入端）
- **指标**：
  - 反射系数与 S11 一致（±1 dB）

### 5. 数值稳定性
- **目标**：长时间仿真无发散
- **方法**：
  - 运行 1e6 个时间步
  - 监控输出能量
- **指标**：
  - 无 NaN/Inf
  - 输出能量 ≤ 输入能量（无源性）

### 6. 性能基准
- **目标**：仿真速度对比
- **方法**：
  - 测量每秒模拟时间（wall time）
  - Rational vs Impulse（CPU/GPU，不同冲激长度）
  - **GPU 测试平台**：Apple Silicon Mac（M1/M2/M3）
- **期望**：
  - Rational（8 阶）：> 1000x 实时
  - Impulse CPU（L=512）：> 10x 实时
  - **Impulse Metal GPU（L=512，Apple Silicon）：> 500x 实时**
  - **Impulse Metal GPU FFT（L=4096，Apple Silicon）：> 2000x 实时**

### 7. GPU 加速效果验证（仅 Apple Silicon）
- **目标**：Metal GPU 计算结果与 CPU 一致
- **系统要求**：Apple Silicon Mac（M1/M2/M3 或更新）
- **方法**：
  - 相同输入分别用 CPU 和 Metal GPU 计算
  - 逐样本对比输出
  - 计算最大绝对误差和 RMS 误差
- **指标**：
  - 最大误差 < 1e-6（单精度）或 1e-12（双精度）
  - RMS 误差 < 1e-8
  - 无数值发散现象

## 变更历史

### v0.3 (2025-10-16)
- **GPU 加速支持**：新增 Impulse 方法的 Metal GPU 加速功能（**仅 Apple Silicon**）
  - 新增 `gpu_acceleration` 配置对象
  - 支持 Metal 后端（Apple Silicon 专属）
  - 实现直接卷积和 FFT 卷积两种 GPU 算法
  - 批处理策略优化数据传输
  - 新增 Metal GPU 性能基准和验证测试
  - 补充性能对比表（CPU vs Metal GPU on Apple Silicon）
  - 更新方法选择指南，增加 Apple Silicon GPU 加速场景
  - 提供 Metal GPU 配置示例和依赖说明
  - **明确**：当前仅支持 Apple Silicon，其他 GPU 后端（CUDA/OpenCL/ROCm）暂不支持

### v0.2 (2025-10-16)
- **重大更新**：完全重写文档，新增两种实现方法
  - 新增有理函数拟合法详细说明（向量拟合 → LTF 滤波器）
  - 新增冲激响应卷积法详细说明（IFFT → 延迟线卷积）
  - 增加方法选择指南和性能对比
  - 扩展接口参数：`method`, `fit.*`, `impulse.*`
  - 详细说明 Python 离线处理流程
  - 补充 SystemC-AMS 两种实现方式
  - 新增串扰和双向传输机制说明
  - 完善测试验证策略（频响、冲激、串扰、双向、稳定性、性能）
  - 提供完整使用示例和配置模板

### v0.1 (初始版本)
- 初始模板，包含基本的向量拟合框架
- 占位性内容
