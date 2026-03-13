# Eye Analyzer PAM4 + BER + JTol 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 eye_analyzer 扩展为支持 PAM4、统计眼图、完整 BER 分析和 Jitter Tolerance 测试的 SerDes 信号完整性分析平台。

**Architecture:** 采用 Strategy Pattern + Registry 模式实现调制格式抽象层；重构 pystateye 算法为 StatisticalScheme；新增 ber/ 子包提供 BER/JTol 功能。

**Tech Stack:** Python 3.8+, NumPy, SciPy, Matplotlib, PyYAML

---

## Batch 1: 基础架构 - 调制格式与 Scheme 重构

### Task 1.1: 创建调制格式抽象层

**Files:**
- Create: `eye_analyzer/modulation.py`
- Test: `tests/unit/test_modulation.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_modulation.py
import pytest
import numpy as np
from eye_analyzer.modulation import (
    ModulationFormat, PAM4, NRZ, 
    create_modulation, MODULATION_REGISTRY
)

def test_pam4_basic():
    pam4 = PAM4()
    assert pam4.name == 'pam4'
    assert pam4.num_levels == 4
    assert pam4.num_eyes == 3
    np.testing.assert_array_equal(pam4.get_levels(), [-3, -1, 1, 3])
    np.testing.assert_array_equal(pam4.get_thresholds(), [-2, 0, 2])
    np.testing.assert_array_equal(pam4.get_eye_centers(), [-2, 0, 2])

def test_nrz_basic():
    nrz = NRZ()
    assert nrz.name == 'nrz'
    assert nrz.num_levels == 2
    assert nrz.num_eyes == 1
    np.testing.assert_array_equal(nrz.get_levels(), [-1, 1])

def test_factory_function():
    pam4 = create_modulation('pam4')
    assert isinstance(pam4, PAM4)
    
    nrz = create_modulation('nrz')
    assert isinstance(nrz, NRZ)
    
    with pytest.raises(ValueError):
        create_modulation('invalid')
```

**Step 2: Run test to verify it fails**

```bash
cd /mnt/d/systemCProjects/SerDesSystemCProject-eye-analyzer-pam4
python -m pytest tests/unit/test_modulation.py -v
```

**Expected:** ModuleNotFoundError: No module named 'eye_analyzer.modulation'

**Step 3: Write minimal implementation**

```python
# eye_analyzer/modulation.py
"""Modulation format abstraction layer for eye_analyzer."""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union
import numpy as np


class ModulationFormat(ABC):
    """Abstract base class for modulation formats."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Format name."""
        pass
    
    @property
    @abstractmethod
    def num_levels(self) -> int:
        """Number of signal levels (M)."""
        pass
    
    @property
    def num_eyes(self) -> int:
        """Number of eye openings (M-1)."""
        return self.num_levels - 1
    
    @abstractmethod
    def get_levels(self) -> np.ndarray:
        """Return signal level array."""
        pass
    
    @abstractmethod
    def get_thresholds(self) -> np.ndarray:
        """Return decision thresholds between levels."""
        pass
    
    @abstractmethod
    def get_eye_centers(self) -> np.ndarray:
        """Return center voltage of each eye opening."""
        pass
    
    def get_level_names(self) -> List[str]:
        """Return names for each signal level."""
        return [f"LV{i}" for i in range(self.num_levels)]


class PAM4(ModulationFormat):
    """PAM4 modulation: 4 levels at -3, -1, 1, 3."""
    
    @property
    def name(self) -> str:
        return 'pam4'
    
    @property
    def num_levels(self) -> int:
        return 4
    
    def get_levels(self) -> np.ndarray:
        return np.array([-3, -1, 1, 3])
    
    def get_thresholds(self) -> np.ndarray:
        return np.array([-2, 0, 2])
    
    def get_eye_centers(self) -> np.ndarray:
        return np.array([-2, 0, 2])
    
    def get_level_names(self) -> List[str]:
        return ['LV3', 'LV2', 'LV1', 'LV0']


class NRZ(ModulationFormat):
    """NRZ/PAM2 modulation: 2 levels at -1, 1."""
    
    @property
    def name(self) -> str:
        return 'nrz'
    
    @property
    def num_levels(self) -> int:
        return 2
    
    def get_levels(self) -> np.ndarray:
        return np.array([-1, 1])
    
    def get_thresholds(self) -> np.ndarray:
        return np.array([0])
    
    def get_eye_centers(self) -> np.ndarray:
        return np.array([0])


# Registry for factory function
MODULATION_REGISTRY: Dict[str, Type[ModulationFormat]] = {
    'nrz': NRZ,
    'pam4': PAM4,
}


def create_modulation(name: str) -> ModulationFormat:
    """Factory function to create modulation format instances."""
    if name not in MODULATION_REGISTRY:
        available = list(MODULATION_REGISTRY.keys())
        raise ValueError(f"Unknown modulation '{name}'. Available: {available}")
    return MODULATION_REGISTRY[name]()
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/test_modulation.py -v
```

**Expected:** All tests pass

**Step 5: Commit**

```bash
git add eye_analyzer/modulation.py tests/unit/test_modulation.py
git commit -m "feat(modulation): add modulation format abstraction layer

- Add ModulationFormat ABC with PAM4 and NRZ implementations
- Support factory function create_modulation()
- Include unit tests for all modulation formats"
```

---

### Task 1.2: 重构 BaseScheme 支持 modulation 参数

**Files:**
- Modify: `eye_analyzer/schemes/base.py:30-57` (__init__ method)
- Modify: `eye_analyzer/schemes/base.py:111-185` (_compute_eye_height method)
- Test: `tests/unit/test_schemes_base.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_schemes_base.py
import pytest
import numpy as np
from eye_analyzer.schemes.base import BaseScheme
from eye_analyzer.modulation import PAM4, NRZ

class MockScheme(BaseScheme):
    """Mock implementation for testing."""
    def analyze(self, time_array, voltage_array, **kwargs):
        return {}
    def get_xedges(self):
        return np.linspace(0, 1, self.ui_bins + 1)
    def get_yedges(self):
        return np.linspace(-1, 1, self.amp_bins + 1)

def test_basescheme_with_string_modulation():
    scheme = MockScheme(ui=1e-12, modulation='pam4', ui_bins=64, amp_bins=128)
    assert scheme.ui == 1e-12
    assert scheme.modulation.name == 'pam4'
    assert scheme.modulation.num_levels == 4

def test_basescheme_with_object_modulation():
    pam4 = PAM4()
    scheme = MockScheme(ui=1e-12, modulation=pam4)
    assert scheme.modulation.name == 'pam4'

def test_basescheme_default_is_nrz():
    scheme = MockScheme(ui=1e-12)
    assert scheme.modulation.name == 'nrz'
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_schemes_base.py -v
```

**Expected:** TypeError - unexpected keyword argument 'modulation'

**Step 3: Write minimal implementation**

```python
# eye_analyzer/schemes/base.py - Modified __init__
from typing import Union
from ..modulation import ModulationFormat, create_modulation

class BaseScheme(ABC):
    def __init__(self, 
                 ui: float, 
                 modulation: Union[str, ModulationFormat] = 'nrz',
                 ui_bins: int = 128, 
                 amp_bins: int = 256):
        
        if ui <= 0:
            raise ValueError(f"UI must be positive, got {ui}")
        if ui_bins < 2:
            raise ValueError(f"ui_bins must be at least 2")
        if amp_bins < 2:
            raise ValueError(f"amp_bins must be at least 2")
        
        self.ui = ui
        self.ui_bins = ui_bins
        self.amp_bins = amp_bins
        
        # Support both string and object
        if isinstance(modulation, str):
            self.modulation = create_modulation(modulation)
        else:
            self.modulation = modulation
        
        self.eye_matrix = None
        self._xedges = None
        self._yedges = None
        self._v_min = 0.0
        self._v_max = 0.0
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/test_schemes_base.py -v
```

**Step 5: Commit**

```bash
git add eye_analyzer/schemes/base.py tests/unit/test_schemes_base.py
git commit -m "refactor(schemes): add modulation parameter to BaseScheme

- BaseScheme now accepts modulation as string or ModulationFormat object
- Default modulation is 'nrz' for backward compatibility during transition
- Add unit tests for modulation parameter handling"
```

---

### Task 1.3: 重构 GoldenCdrScheme 支持 PAM4

**Files:**
- Modify: `eye_analyzer/schemes/golden_cdr.py`
- Test: `tests/unit/test_golden_cdr_pam4.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_golden_cdr_pam4.py
import numpy as np
import pytest
from eye_analyzer.schemes import GoldenCdrScheme

def generate_pam4_waveform(ui, num_bits=1000, samples_per_ui=64):
    """Generate a simple PAM4 test waveform."""
    t = np.linspace(0, num_bits * ui, num_bits * samples_per_ui)
    
    # Simple PAM4 pattern: cycling through levels
    levels = [-3, -1, 1, 3]
    pattern = np.array([levels[i % 4] for i in range(num_bits)])
    
    # Create waveform with raised cosine pulse
    from scipy.signal import upfirdn, firwin
    h = firwin(samples_per_ui, 0.5)  # Simple pulse shaping
    waveform = upfirdn(pattern, h, samples_per_ui)
    
    # Truncate to match time array
    waveform = waveform[:len(t)]
    return t, waveform

def test_golden_cdr_pam4_analysis():
    ui = 2.5e-11  # 40 Gbps
    scheme = GoldenCdrScheme(ui=ui, modulation='pam4', ui_bins=128, amp_bins=256)
    
    time_array, voltage_array = generate_pam4_waveform(ui)
    
    metrics = scheme.analyze(time_array, voltage_array)
    
    # PAM4 should have 3 eyes
    assert 'eye_heights_per_eye' in metrics
    assert len(metrics['eye_heights_per_eye']) == 3
    
    # All eye heights should be positive
    for h in metrics['eye_heights_per_eye']:
        assert h > 0

def test_golden_cdr_nrz_still_works():
    ui = 2.5e-11
    scheme = GoldenCdrScheme(ui=ui, modulation='nrz')
    
    # Generate simple NRZ waveform
    t = np.linspace(0, 100 * ui, 6400)
    v = np.sign(np.sin(2 * np.pi * t / (2 * ui)))
    
    metrics = scheme.analyze(t, v)
    assert 'eye_height' in metrics
    assert metrics['eye_height'] > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_golden_cdr_pam4.py -v
```

**Expected:** AttributeError or assertion error on eye_heights_per_eye

**Step 3: Write implementation**

修改 `GoldenCdrScheme.analyze()` 方法，添加 PAM4 支持：

```python
# eye_analyzer/schemes/golden_cdr.py

def analyze(self, time_array, voltage_array, **kwargs):
    """Analyze eye diagram with PAM4 support."""
    # ... existing validation code ...
    
    # Build eye diagram
    self._build_eye_matrix(time_array, voltage_array)
    
    # Compute metrics based on modulation format
    if self.modulation.name == 'pam4':
        metrics = self._compute_metrics_pam4()
    else:
        metrics = self._compute_metrics_nrz()
    
    return metrics

def _compute_metrics_pam4(self):
    """Compute metrics for PAM4 modulation."""
    eye_heights = self._compute_eye_heights_per_eye()
    eye_widths = self._compute_eye_widths_per_eye()
    
    return {
        'eye_heights_per_eye': eye_heights,
        'eye_widths_per_eye': eye_widths,
        'eye_height_min': min(eye_heights),
        'eye_height_avg': sum(eye_heights) / len(eye_heights),
        'eye_width_min': min(eye_widths),
        'modulation': 'pam4',
        'num_eyes': 3,
    }

def _compute_eye_heights_per_eye(self):
    """Compute eye height for each of the 3 PAM4 eyes."""
    thresholds = self.modulation.get_thresholds()
    eye_heights = []
    
    for threshold in thresholds:
        height = self._compute_eye_height_at_threshold(threshold)
        eye_heights.append(height)
    
    return eye_heights

def _compute_eye_height_at_threshold(self, threshold):
    """Compute eye height at a specific decision threshold."""
    if self.eye_matrix is None or self._yedges is None:
        return 0.0
    
    # Find bin index closest to threshold
    y_centers = (self._yedges[:-1] + self._yedges[1:]) / 2
    threshold_idx = np.argmin(np.abs(y_centers - threshold))
    
    # Find optimal phase
    phase_density = np.sum(self.eye_matrix, axis=1)
    optimal_phase_idx = np.argmax(phase_density)
    
    # Get amplitude profile at optimal phase
    amplitude_profile = self.eye_matrix[optimal_phase_idx, :]
    
    # Find upper and lower eyes at this threshold
    upper_mask = y_centers > threshold
    lower_mask = y_centers < threshold
    
    # Compute eye opening
    upper_max = np.max(y_centers[upper_mask]) if np.any(upper_mask) else threshold
    lower_min = np.min(y_centers[lower_mask]) if np.any(lower_mask) else threshold
    
    return upper_max - lower_min
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/test_golden_cdr_pam4.py -v
```

**Step 5: Commit**

```bash
git add eye_analyzer/schemes/golden_cdr.py tests/unit/test_golden_cdr_pam4.py
git commit -m "feat(schemes): add PAM4 support to GoldenCdrScheme

- Add _compute_metrics_pam4() for PAM4-specific metrics
- Add per-eye eye height/width computation
- Maintain NRZ compatibility"
```

---

## Batch 2: 统计眼图核心

### Task 2.1: 创建 statistical/ 子包结构

**Files:**
- Create: `eye_analyzer/statistical/__init__.py`
- Create: `eye_analyzer/statistical/pulse_response.py`

**Step 1: Create package init**

```python
# eye_analyzer/statistical/__init__.py
"""Statistical eye diagram analysis modules."""

from .pulse_response import PulseResponseProcessor
from .isi_calculator import ISICalculator
from .noise_injector import NoiseInjector
from .jitter_injector import JitterInjector
from .ber_calculator import BERContourCalculator

__all__ = [
    'PulseResponseProcessor',
    'ISICalculator',
    'NoiseInjector',
    'JitterInjector',
    'BERContourCalculator',
]
```

**Step 2: Create PulseResponseProcessor**

```python
# eye_analyzer/statistical/pulse_response.py
"""Pulse response preprocessing for statistical eye analysis."""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple
from ..modulation import ModulationFormat


class PulseResponseProcessor:
    """Process channel pulse response for statistical eye analysis."""
    
    def process(self,
                pulse_response: np.ndarray,
                modulation: ModulationFormat,
                diff_signal: bool = True,
                upsampling: int = 16,
                interpolation_type: str = 'linear') -> np.ndarray:
        """
        Process pulse response.
        
        Args:
            pulse_response: Raw pulse response array
            modulation: Modulation format
            diff_signal: If True, multiply by 0.5 for differential signaling
            upsampling: Upsampling factor for better visualization
            interpolation_type: 'linear' or 'cubic'
            
        Returns:
            Processed pulse response
        """
        # Remove DC offset
        pulse = np.array(pulse_response)
        pulse = pulse - pulse[0]
        
        # Extract non-zero window
        window = np.where(pulse != 0)[0]
        if len(window) == 0:
            raise ValueError("Pulse response is all zeros")
        
        window_start = max(0, window[0] - 1)
        window_end = min(len(pulse), window[-1] + 2)
        pulse = pulse[window_start:window_end]
        
        # Apply differential signaling factor
        if diff_signal:
            pulse = pulse * 0.5
        
        # Upsample
        if upsampling > 1:
            x = np.linspace(0, len(pulse) - 1, len(pulse))
            f = interp1d(x, pulse, kind=interpolation_type)
            x_new = np.linspace(0, len(pulse) - 1, len(pulse) * upsampling)
            pulse = f(x_new)
        
        return pulse
    
    def find_main_cursor(self, pulse: np.ndarray) -> int:
        """Find index of main cursor (peak amplitude)."""
        return int(np.argmax(np.abs(pulse)))
```

**Step 3: Commit**

```bash
git add eye_analyzer/statistical/
git commit -m "feat(statistical): create statistical eye analysis package structure

- Add PulseResponseProcessor for pulse response preprocessing
- Include DC removal, windowing, differential signaling, upsampling"
```

---

### Task 2.2: 实现 ISI Calculator

**Files:**
- Create: `eye_analyzer/statistical/isi_calculator.py`
- Test: `tests/unit/test_isi_calculator.py`

**Step 1: Write test**

```python
# tests/unit/test_isi_calculator.py
import numpy as np
import pytest
from eye_analyzer.statistical import ISICalculator
from eye_analyzer.modulation import PAM4, NRZ

def test_isi_calculation_nrz():
    calc = ISICalculator(method='convolution')
    
    # Simple pulse response: main cursor = 1, one pre-cursor = 0.1
    pulse = np.zeros(64)
    pulse[32] = 1.0  # Main cursor
    pulse[24] = 0.1  # Pre-cursor
    
    isi_pdf = calc.calculate(
        pulse,
        modulation=NRZ(),
        samples_per_symbol=8,
        sample_size=4
    )
    
    assert isi_pdf is not None
    assert isi_pdf.shape[0] > 0
    assert np.all(isi_pdf >= 0)

def test_isi_calculation_pam4():
    calc = ISICalculator(method='brute_force')
    
    pulse = np.zeros(64)
    pulse[32] = 1.0
    
    isi_pdf = calc.calculate(
        pulse,
        modulation=PAM4(),
        samples_per_symbol=8,
        sample_size=3
    )
    
    assert isi_pdf is not None
```

**Step 2: Implement ISICalculator**

```python
# eye_analyzer/statistical/isi_calculator.py
"""ISI PDF calculation using convolution or brute force methods."""

import numpy as np
from typing import Literal
from ..modulation import ModulationFormat


class ISICalculator:
    """
    Calculate ISI (Inter-Symbol Interference) probability distribution.
    
    Two methods supported:
    - 'convolution': Fast O(N*L) using PDF convolution
    - 'brute_force': Exact O(M^N) using enumeration
    """
    
    def __init__(self, method: Literal['convolution', 'brute_force'] = 'convolution'):
        self.method = method
    
    def calculate(self,
                  pulse: np.ndarray,
                  modulation: ModulationFormat,
                  samples_per_symbol: int,
                  sample_size: int,
                  vh_size: int = 2048) -> np.ndarray:
        """
        Calculate ISI PDF.
        
        Args:
            pulse: Processed pulse response
            modulation: Modulation format
            samples_per_symbol: Samples per UI
            sample_size: Number of ISI symbols to consider
            vh_size: Voltage histogram bins
            
        Returns:
            2D array: ISI PDF [voltage_bins, time_bins]
        """
        idx_main = int(np.argmax(np.abs(pulse)))
        window_size = samples_per_symbol
        
        # Voltage range
        A_max = np.abs(pulse[idx_main])
        A_window = A_max * 2  # Multiplier for viewing window
        
        vh_edges = np.linspace(-A_window, A_window, vh_size + 1)
        vh_centers = (vh_edges[:-1] + vh_edges[1:]) / 2
        
        # Get modulation levels
        levels = modulation.get_levels().reshape(1, -1)
        
        pdf_list = []
        
        for idx in range(-window_size // 2, window_size // 2):
            idx_sampled = idx_main + idx
            
            # Sample pulse response at symbol intervals
            sampled_points = []
            i = 0
            while idx_sampled - i * samples_per_symbol >= 0:
                sampled_points.append(idx_sampled - i * samples_per_symbol)
                i += 1
            
            j = 1
            while idx_sampled + j * samples_per_symbol < len(pulse):
                sampled_points.append(idx_sampled + j * samples_per_symbol)
                j += 1
            
            sampled_points = sampled_points[:sample_size]
            sampled_amps = np.array([pulse[i] for i in sampled_points]).reshape(-1, 1)
            sampled_amps = sampled_amps @ levels  # Shape: (sample_size, num_levels)
            
            if self.method == 'convolution':
                pdf = self._calculate_by_convolution(sampled_amps, vh_edges)
            else:
                pdf = self._calculate_by_brute_force(sampled_amps, vh_edges)
            
            pdf_list.append(pdf)
        
        return np.array(pdf_list).T  # Shape: (vh_size, window_size)
    
    def _calculate_by_convolution(self, sampled_amps, vh_edges):
        """Calculate ISI PDF using convolution method."""
        pdf, _ = np.histogram(sampled_amps[0], vh_edges)
        pdf = pdf / np.sum(pdf)
        
        for j in range(1, len(sampled_amps)):
            pdf_cursor, _ = np.histogram(sampled_amps[j], vh_edges)
            pdf_cursor = pdf_cursor / np.sum(pdf_cursor)
            pdf = np.convolve(pdf, pdf_cursor, mode='same')
            pdf = pdf / np.sum(pdf)
        
        return pdf
    
    def _calculate_by_brute_force(self, sampled_amps, vh_edges):
        """Calculate ISI PDF by enumerating all combinations."""
        # Create meshgrid for all combinations
        grids = np.meshgrid(*[sampled_amps[i] for i in range(len(sampled_amps))])
        all_combs = np.array([g.flatten() for g in grids]).T
        A = np.sum(all_combs, axis=1)
        
        pdf, _ = np.histogram(A, vh_edges)
        pdf = pdf / np.sum(pdf)
        
        return pdf
```

**Step 3: Commit**

```bash
git add eye_analyzer/statistical/isi_calculator.py tests/unit/test_isi_calculator.py
git commit -m "feat(statistical): add ISI PDF calculator

- Support convolution (fast) and brute_force (exact) methods
- Handle NRZ and PAM4 modulation formats"
```

---

（由于篇幅限制，计划文件在此截断。完整计划应包含所有 Batch 的详细步骤。）

## 计划执行说明

此计划包含 5 个 Batch，每个 Batch 对应一周工作量。执行时：

1. 使用 `executing-plans` skill 逐任务执行
2. 每个 Task 按 Step 1-5 严格执行
3. 完成后更新 `.kimi-dev/status.md` 中的进度
4. 每 Batch 完成后进行审查
