# EyeAnalyzer 模块小文档

级别：Python 分析组件

## 概述
Python 分析组件，对链路输出进行眼图与抖动分析，输出眼高/眼宽/开口面积、RJ/DJ/Total 抖动分解及线性度等指标。
## 接口
- 输入：
  - 波形数据文件（`results.dat`，tabular）或内存波形数组（time/value）
  - 可选：时钟参数（UI、采样相位估计方法）
- 输出：
  - 指标文件（`eye_metrics.json`）与可视化图像（PNG/SVG）或 CSV（PSD/PDF/密度）
- 配置键：
  - `ui_bins`/`amp_bins`（int）：二维直方图分辨率
  - `measure_length`（double）：统计时长（s）
  - `target_ber`（double）：目标 BER 下的 TJ 估计
  - `ui`（double）：单位间隔（s）
  - `sampling`（string）：采样策略（如 `peak`, `zero-cross`, `phase-lock`）
## 参数
- `ui_bins`: 必须，默认 128
- `amp_bins`: 必须，默认 128
- `measure_length`: 必须，默认 1e-4（s）
- `target_ber`: 可选，默认 1e-12
- `ui`: 必须或从仿真配置读取
- `sampling`: 可选，默认 `phase-lock`
## 行为模型
- 眼图：将时间映射到相位 (`phi=(t%UI)/UI`)，构建二维密度
- 指标：计算眼高/眼宽/开口面积与线性拟合误差
- 抖动分解：区分 RJ（高斯）与 DJ（数据相关/周期性），估计 Total Jitter@BER

## 依赖
- Python：numpy、scipy、matplotlib
- 可选：pandas 用于 CSV/JSON 处理
- 工具建议：Surfer 作为波形查看工具（macOS 体验更佳）
## API 设计
- 函数：
  - `analyze_eye(dat_path: str, ui: float, ui_bins: int = 128, amp_bins: int = 128, measure_length: float | None = None, target_ber: float = 1e-12, sampling: str = 'phase-lock') -> dict`
- 返回：指标字典（见下文 Schema），并可写出 `eye_metrics.json`、图像与 CSV
- 命令行（占位）：
  - `python analyze_eye.py --dat results.dat --ui 2.5e-11 --ui-bins 128 --amp-bins 128 --target-ber 1e-12`

## 使用示例
1. 运行 SystemC‑AMS 仿真生成 `results.dat`
2. 在 Python 中加载数据与配置（UI、bins、measure_length）
3. 计算 PSD/PDF 与二维眼图密度，输出 `eye_metrics.json` 与图像
4. 用 Surfer 或 matplotlib 查看波形与眼图
## 测试验证
- 指标稳定性：随统计时长增加指标收敛
- 抖动分解：RJ/DJ 分离合理，TJ@BER 与预期一致
- 线性度：对输出电平做线性拟合误差评估
- 文件一致性：不同块写出或采样率设置下的结果一致性
## 数据格式与Schema
- 输入 `results.dat`：tabular 格式，列为 `time`（s）与 `value`（V）；可扩展其他信号列
- 输出 `eye_metrics.json` 建议字段：
  - `eye_height`（V）, `eye_width`（UI）, `eye_area`（V*UI）
  - `rj_sigma`（s）, `dj_pp`（s）, `tj_at_ber`（s）
  - `linearity_error`（V 或归一化）, `ui_bins`, `amp_bins`, `measure_length`
  - `hist2d`（可选：摘要或文件路径）, `pdf`/`psd`（可选）

## 变更历史
- v0.1 初始模板，占位内容
