# Batch 1 进度: 基础架构 - 调制格式与 Scheme 重构

## Task 1.1: 创建调制格式抽象层 ✅ 已完成

**Task Goal:** 创建 `modulation.py` 实现调制格式抽象层，支持 PAM4 和 NRZ，预留 PAM3/PAM6/PAM8 扩展。

**完成状态:**
- ✅ 测试文件: `tests/unit/test_modulation.py` (6个测试)
- ✅ 实现文件: `eye_analyzer/modulation.py`
- ✅ 所有测试通过
- ✅ 已提交: `f7066df`

---

## Task 1.2: 重构 BaseScheme 支持 modulation 参数

**Task Goal:** 修改 `schemes/base.py`，添加 `modulation` 参数支持，使 BaseScheme 能接受字符串或 ModulationFormat 对象。

**Project Conventions:**
- 保持向后兼容（默认 'nrz'）
- 使用 Union[str, ModulationFormat] 类型
- 添加单元测试

**Steps:**

### Step 1: Write the failing test
- [ ] 创建 `tests/unit/test_schemes_base.py`
- 测试字符串 modulation 参数
- 测试对象 modulation 参数
- 测试默认值

### Step 2: Run test to verify it fails
- [ ] 运行测试，确认 TypeError

### Step 3: Write minimal implementation
- [ ] 修改 `schemes/base.py`
- 添加 modulation 参数到 __init__
- 支持字符串和对象两种传入方式

### Step 4: Run test to verify it passes
- [ ] 运行测试，全部通过

### Step 5: Commit
- [ ] git commit

**Acceptance Criteria:**
- [ ] BaseScheme 接受 modulation 参数
- [ ] 支持字符串和 ModulationFormat 对象
- [ ] 默认值为 'nrz'
- [ ] 单元测试通过

**Constraints:**
- 保持现有测试兼容
- 不要破坏现有功能
