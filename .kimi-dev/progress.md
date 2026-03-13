# Batch 2 进度: 统计眼图核心

## Batch 1 已完成 ✅

所有基础架构任务完成：
- Task 1.1: modulation.py (6 tests) ✅
- Task 1.2: BaseScheme 重构 (7 tests) ✅
- Task 1.3: GoldenCdrScheme PAM4 (5 tests) ✅
- Task 1.4: SamplerCentricScheme (2 tests) ✅

---

## Batch 2: 统计眼图核心

**Goal:** 创建统计眼图分析子系统，包括脉冲响应处理、ISI 计算、噪声/抖动注入、BER 计算。

**基于:** pystateye 算法重构

---

## Task 2.1: 创建 statistical/ 子包结构

**Files:**
- Create: `eye_analyzer/statistical/__init__.py`
- Create: `eye_analyzer/statistical/pulse_response.py`

**Steps:**

### Step 1: Create package init
- [ ] 创建 `__init__.py`，导出核心类

### Step 2: Create PulseResponseProcessor
- [ ] 实现脉冲响应预处理
- [ ] DC 去除、窗口提取、差分转换、上采样

### Step 3: Write tests
- [ ] 创建 `tests/unit/test_pulse_response.py`

### Step 4: Run tests
- [ ] 验证通过

### Step 5: Commit
- [ ] git commit

---

## Task 2.2: ISI Calculator

**Files:**
- Create: `eye_analyzer/statistical/isi_calculator.py`
- Test: `tests/unit/test_isi_calculator.py`

**Steps:**
1. Write failing test
2. Run to verify failure
3. Implement ISICalculator (convolution + brute_force)
4. Run tests
5. Commit

---

## Task 2.3: Noise Injector

**Files:**
- Create: `eye_analyzer/statistical/noise_injector.py`
- Test: `tests/unit/test_noise_injector.py`

---

## Task 2.4: Jitter Injector

**Files:**
- Create: `eye_analyzer/statistical/jitter_injector.py`
- Test: `tests/unit/test_jitter_injector.py`

---

## Task 2.5: BER Contour Calculator

**Files:**
- Create: `eye_analyzer/statistical/ber_calculator.py`
- Test: `tests/unit/test_ber_calculator.py`

---

## Task 2.6: StatisticalScheme

**Files:**
- Create: `eye_analyzer/schemes/statistical.py`
- Test: `tests/unit/test_statistical_scheme.py`

---

## Current Status

**In Progress:** Task 2.1 - 创建 statistical/ 子包
