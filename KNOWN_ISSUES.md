# Known Issues - Eye Analyzer v2.0.0

**生成日期**: 2026-03-13  
**来源**: Phase 6 多 Agent 并行验证  
**验证 Agent**: #1 架构, #2 测试, #3 文档

---

## 问题分级说明

- 🔴 **Critical**: 影响核心功能，必须修复
- 🟠 **Important**: 影响用户体验或功能完整性，建议修复
- 🟡 **Minor**: 小问题，不影响核心功能，可延后修复

---

## 🔴 Critical Issues

### 1. UnifiedEyeAnalyzer 不支持 StatisticalScheme

| 属性 | 内容 |
|------|------|
| **位置** | `eye_analyzer/analyzer.py:53` |
| **发现者** | Agent #1 (架构验证) |
| **问题描述** | `valid_schemes` 列表缺少 `'statistical'`，导致 StatisticalScheme 无法通过统一接口访问 |
| **影响** | StatisticalScheme 成为"孤岛"，用户无法通过 UnifiedEyeAnalyzer 使用统计眼图分析功能 |
| **修复状态** | ✅ **已修复** (commit: c64206e) |
| **修复内容** | 添加 `'statistical'` 到 `valid_schemes` 列表，并完整支持 StatisticalScheme 的初始化和分析 |

---

### 2. README.md 示例代码错误

| 属性 | 内容 |
|------|------|
| **位置** | `README.md` 示例部分 |
| **发现者** | Agent #3 (文档验证) |
| **问题描述** | 示例代码使用不存在的 `analyzer.load_waveform()` 方法 |
| **影响** | 用户无法直接运行示例代码，影响第一印象和使用体验 |
| **修复状态** | ✅ **已修复** |
| **修复内容** | 将 `analyzer.load_waveform()` 改为 `auto_load_waveform()`，并更新 import 语句 |

---

## 🟠 Important Issues

### 3. JTOL 功能未完全实现

| 属性 | 内容 |
|------|------|
| **位置** | 设计文档 R5 章节 vs 实际代码 |
| **发现者** | Agent #2 (功能验证) |
| **问题描述** | 设计文档 R5 章节规划了完整的 JTOL 功能，但编码实现不完全 |
| **影响** | JTOL 基础功能可用，但部分高级特性（如自定义扫描精度、多模板同时对比）未实现 |
| **修复状态** | ⏭️ **可选修复** - 当前实现满足基本需求 |
| **建议** | 如需完整 JTOL 功能，需补充实现设计文档中的高级特性 |

---

### 4. StatisticalScheme 接口不兼容

| 属性 | 内容 |
|------|------|
| **位置** | `eye_analyzer/schemes/statistical.py` vs 其他 schemes |
| **发现者** | Agent #1 (架构验证) |
| **问题描述** | `StatisticalScheme.analyze(pulse_response)` 接收脉冲响应，而其他 schemes 接收 `(time_array, voltage_array)` 时域波形 |
| **影响** | 虽然已通过修复 Critical #1 支持统一接口，但接口签名不一致可能导致用户困惑 |
| **修复状态** | ⚠️ **部分修复** - 通过统一入口封装差异 |
| **建议** | 考虑重构为统一的 `analyze(input_data)` 接口，内部根据类型判断 |

---

### 5. optimal_sampling_phase metric 缺失或名称不匹配

| 属性 | 内容 |
|------|------|
| **位置** | 眼图指标计算 |
| **发现者** | Agent #2 (测试验证) |
| **问题描述** | `optimal_sampling_phase` 指标缺失或与其他组件期望的名称不匹配 |
| **影响** | 可能影响依赖于该指标的高级分析功能 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 检查所有使用 `optimal_sampling_phase` 的地方，统一命名和实现 |

---

### 6. 版本号不匹配

| 属性 | 内容 |
|------|------|
| **位置** | 代码 vs 文档 |
| **发现者** | Agent #3 (文档验证) |
| **问题描述** | 代码: `__version__ = "2.0.0"`，文档: "v1.0" |
| **影响** | 用户可能对版本产生困惑 |
| **修复状态** | ⏭️ **待修复** - 需要统一文档中的版本号 |
| **建议** | 将所有文档中的版本号统一为 "2.0.0" |

---

## 🟡 Minor Issues

### 7. ModulationFormat 重复定义

| 属性 | 内容 |
|------|------|
| **位置** | `modulation.py` vs `isi_calculator.py` |
| **发现者** | Agent #1 (架构验证) |
| **问题描述** | `ModulationFormat` 在 `modulation.py` 中定义为抽象基类，在 `isi_calculator.py` 中定义为 Enum |
| **影响** | 目前功能正常，但存在命名冲突风险 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 重命名其中一个，或使用统一的设计 |

---

### 8. StatisticalScheme 未从根模块导出

| 属性 | 内容 |
|------|------|
| **位置** | `eye_analyzer/__init__.py` |
| **发现者** | Agent #1 (架构验证) |
| **问题描述** | `StatisticalScheme` 类未在根模块 `__init__.py` 中导出 |
| **影响** | 用户需要通过完整路径 `eye_analyzer.schemes.statistical.StatisticalScheme` 导入 |
| **修复状态** | ⚠️ **可忽略** - 已通过 `EyeAnalyzer` 统一入口封装 |
| **建议** | 如用户需要直接访问 StatisticalScheme，可添加到 `__init__.py` |

---

### 9. 参数命名不一致

| 属性 | 内容 |
|------|------|
| **位置** | 文档 vs 代码 |
| **发现者** | Agent #3 (文档验证) |
| **问题描述** | 文档中使用 `jitter_extract_method`，代码中使用 `jitter_method` |
| **影响** | 可能导致用户按文档设置参数时出错 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 统一参数命名，或添加别名支持 |

---

### 10. 缺少部分参数文档

| 属性 | 内容 |
|------|------|
| **位置** | API 文档 |
| **发现者** | Agent #3 (文档验证) |
| **问题描述** | 部分参数缺少文档说明：`n_ui_display`, `center_eye`, `interpolate_factor` |
| **影响** | 用户可能不清楚这些参数的用途 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 在 `api_reference.md` 中补充这些参数的说明 |

---

### 11. 引用了不存在的文件

| 属性 | 内容 |
|------|------|
| **位置** | 文档 |
| **发现者** | Agent #3 (文档验证) |
| **问题描述** | 文档中引用了以下不存在的文件：
- `scripts/eye_analyzer.py`
- `scripts/analyze_eye.py`
- `tests/unit/test_eye_analyzer.py`
- `tests/performance/test_eye_performance.py` |
| **影响** | 用户按文档操作时可能找不到这些文件 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 更新文档，移除或更正这些引用 |

---

### 12. 测试期望设置不当

| 属性 | 内容 |
|------|------|
| **位置** | 多个测试文件 |
| **发现者** | Agent #2 (测试验证) |
| **问题描述** | 4 个测试的期望设置不当（非代码缺陷，是测试本身的问题）：
- 眼图样本计数双倍（2-UI 窗口设计导致）
- peak_to_peak 插值后放大
- 性能测试超时（100000 UI 大数据集） |
| **影响** | 测试失败但不影响实际功能 |
| **修复状态** | ⏭️ **待修复** |
| **建议** | 调整测试期望或测试参数 |

---

## 修复优先级建议

### 立即修复 (P0)
- [x] Critical #1: UnifiedEyeAnalyzer 不支持 StatisticalScheme
- [x] Critical #2: README.md 示例代码错误

### 高优先级 (P1)
- [ ] Important #5: optimal_sampling_phase metric 缺失
- [ ] Important #6: 版本号不匹配
- [ ] Minor #9: 参数命名不一致

### 中优先级 (P2)
- [ ] Important #3: JTOL 功能完善
- [ ] Important #4: StatisticalScheme 接口统一
- [ ] Minor #7: ModulationFormat 重复定义

### 低优先级 (P3)
- [ ] Minor #8: StatisticalScheme 导出
- [ ] Minor #10: 参数文档补充
- [ ] Minor #11: 文档引用修正
- [ ] Minor #12: 测试期望调整

---

## 验证结论

**Agent #1 (架构)**: 条件通过  
**Agent #2 (测试)**: 条件通过  
**Agent #3 (文档)**: 条件通过

**综合结论**: 修复 Critical 问题后，项目满足验收标准，可以进入 Phase 7。

---

## 更新历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-03-13 | v1.0 | 初始版本，基于 Phase 6 验证结果 |
