# Eye Analyzer API Reference

## Table of Contents

- [Core Classes](#core-classes)
  - [EyeAnalyzer](#eyeanalyzer)
- [Modulation Formats](#modulation-formats)
  - [ModulationFormat](#modulationformat)
  - [NRZ](#nrz)
  - [PAM4](#pam4)
- [Analysis Schemes](#analysis-schemes)
  - [BaseScheme](#basescheme)
  - [StatisticalScheme](#statisticalscheme)
  - [GoldenCdrScheme](#goldencdrscheme)
  - [SamplerCentricScheme](#samplercentricscheme)
- [Statistical Analysis](#statistical-analysis)
  - [PulseResponseProcessor](#pulseresponseprocessor)
  - [ISICalculator](#isicalculator)
  - [NoiseInjector](#noiseinjector)
  - [JitterInjector](#jitterinjector)
  - [BERCalculator](#bercalculator)
- [Jitter Analysis](#jitter-analysis)
  - [JitterDecomposer](#jitterdecomposer)
  - [JitterAnalyzer](#jitteranalyzer)
- [BER Analysis](#ber-analysis)
  - [BERAnalyzer](#beranalyzer)
  - [BERContour](#bercontour)
  - [BathtubCurve](#bathtubcurve)
  - [QFactor](#qfactor)
  - [JTolTemplate](#jtoltemplate)
  - [JitterTolerance](#jittertolerance)
- [Visualization](#visualization)
  - [plot_eye_diagram](#plot_eye_diagram)
  - [plot_jtol_curve](#plot_jtol_curve)
  - [plot_bathtub_curve](#plot_bathtub_curve)
  - [create_analysis_report](#create_analysis_report)

---

## Core Classes

### EyeAnalyzer

Unified eye analyzer supporting statistical and empirical modes.

```python
class EyeAnalyzer(
    ui: float,
    modulation: str = 'nrz',
    mode: str = 'statistical',
    target_ber: float = 1e-12,
    samples_per_symbol: int = 16,
    ui_bins: int = 128,
    amp_bins: int = 256,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ui` | float | required | Unit interval in seconds (e.g., 2.5e-11 for 10Gbps) |
| `modulation` | str | 'nrz' | Modulation format: 'nrz' or 'pam4' |
| `mode` | str | 'statistical' | Analysis mode: 'statistical' or 'empirical' |
| `target_ber` | float | 1e-12 | Target bit error rate |
| `samples_per_symbol` | int | 16 | Samples per symbol for statistical mode |
| `ui_bins` | int | 128 | Number of UI bins for eye diagram |
| `amp_bins` | int | 256 | Number of amplitude bins for eye diagram |

**Example:**

```python
# Statistical mode for PAM4
analyzer = EyeAnalyzer(
    ui=2.5e-11,
    modulation='pam4',
    mode='statistical',
    target_ber=1e-12
)

# Empirical mode for NRZ
analyzer = EyeAnalyzer(
    ui=2.5e-11,
    modulation='nrz',
    mode='empirical'
)
```

#### Methods

##### analyze

```python
analyzer.analyze(
    input_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    noise_sigma: float = 0.0,
    jitter_dj: float = 0.0,
    jitter_rj: float = 0.0,
    **kwargs
) -> Dict[str, Any]
```

Perform complete eye analysis.

**Parameters:**
- `input_data`: Pulse response array (statistical) or (time_array, value_array) tuple (empirical)
- `noise_sigma`: Gaussian noise sigma in volts (statistical mode only)
- `jitter_dj`: Deterministic jitter in UI
- `jitter_rj`: Random jitter standard deviation in UI

**Returns:**
```python
{
    'eye_matrix': np.ndarray,       # 2D eye diagram data
    'xedges': np.ndarray,           # Phase bin edges
    'yedges': np.ndarray,           # Amplitude bin edges
    'ber_contour': np.ndarray,      # BER contour (statistical mode)
    'eye_metrics': {
        'eye_height': float,        # Eye height in volts
        'eye_width': float,         # Eye width in UI
        'eye_area': float,          # Eye area in V*UI
        # PAM4 additional:
        'eye_heights_per_eye': List[float],
        'eye_widths_per_eye': List[float],
        'eye_height_min': float,
        'eye_height_avg': float,
    },
    'jitter': {
        'dj': float,                # Deterministic jitter
        'rj': float,                # Random jitter
    },
    'bathtub_time': np.ndarray,     # Time bathtub curve
    'bathtub_voltage': np.ndarray,  # Voltage bathtub curve
}
```

##### analyze_jtol

```python
analyzer.analyze_jtol(
    pulse_responses: List[np.ndarray],
    sj_frequencies: List[float],
    template: str = 'ieee_802_3ck',
    **kwargs
) -> Dict[str, Any]
```

Perform jitter tolerance test.

**Parameters:**
- `pulse_responses`: List of pulse responses with different SJ amplitudes
- `sj_frequencies`: List of SJ frequencies in Hz
- `template`: JTOL template name ('ieee_802_3ck', 'oif_cei_56g', 'jedec_ddr5', 'pcie_gen6')

**Returns:**
```python
{
    'sj_frequencies': List[float],   # SJ frequencies tested
    'sj_amplitudes': List[float],    # Required SJ amplitudes
    'template': str,                 # Template used
    'pass_fail': List[bool],         # Pass/fail per frequency
    'margins': Dict[str, float],     # Margin analysis
    'overall_pass': bool,            # Overall pass/fail
}
```

##### plot_eye

```python
analyzer.plot_eye(
    ax=None,
    cmap: str = 'hot',
    title: str = 'Eye Diagram',
    show_metrics: bool = True,
    **kwargs
) -> Union[Figure, Axes]
```

Plot eye diagram.

##### plot_jtol

```python
analyzer.plot_jtol(
    jtol_results: Dict[str, Any],
    ax=None,
    **kwargs
) -> Union[Figure, Axes]
```

Plot JTOL curve.

##### plot_bathtub

```python
analyzer.plot_bathtub(
    direction: str = 'time',
    ax=None,
    **kwargs
) -> Union[Figure, Axes]
```

Plot bathtub curve.

##### create_report

```python
analyzer.create_report(
    output_file: Optional[str] = None,
    format: str = 'text',
    include_plots: bool = True,
    **kwargs
) -> str
```

Create comprehensive analysis report.

**Parameters:**
- `output_file`: Output file path (optional)
- `format`: Report format ('text', 'html', 'markdown')
- `include_plots`: Whether to include plots

---

## Modulation Formats

### ModulationFormat

Abstract base class for modulation formats.

```python
class ModulationFormat(ABC)
```

**Properties:**
- `name`: Format name
- `num_levels`: Number of signal levels (M)
- `num_eyes`: Number of eye openings (M-1)

**Methods:**
- `get_levels() -> np.ndarray`: Signal level array
- `get_thresholds() -> np.ndarray`: Decision thresholds
- `get_eye_centers() -> np.ndarray`: Eye center voltages
- `get_level_names() -> List[str]`: Level names

### NRZ

```python
class NRZ(ModulationFormat)
```

NRZ/PAM2 modulation: 2 levels at -1, 1.

**Levels:** `[-1, 1]`
**Thresholds:** `[0]`

### PAM4

```python
class PAM4(ModulationFormat)
```

PAM4 modulation: 4 levels at -3, -1, 1, 3.

**Levels:** `[-3, -1, 1, 3]`
**Thresholds:** `[-2, 0, 2]`
**Eye Centers:** `[-2, 0, 2]` (lower, middle, upper)

### create_modulation

```python
create_modulation(name: str) -> ModulationFormat
```

Factory function to create modulation format instances.

**Parameters:**
- `name`: 'nrz' or 'pam4'

---

## Analysis Schemes

### BaseScheme

```python
class BaseScheme(ABC)
```

Abstract base class for eye analysis schemes.

### StatisticalScheme

```python
class StatisticalScheme(
    ui: float,
    modulation: str = 'nrz',
    ui_bins: int = 128,
    amp_bins: int = 256,
    samples_per_symbol: int = 16,
    **kwargs
)
```

Statistical eye analysis scheme for pre-simulation channel characterization.

**Key Methods:**
- `analyze(pulse_response, dt, noise_sigma, dj, rj, target_ber) -> Dict`
- `get_ber_contour() -> np.ndarray`
- `get_xedges() -> np.ndarray`
- `get_yedges() -> np.ndarray`

### GoldenCdrScheme

```python
class GoldenCdrScheme(**kwargs)
```

Golden CDR scheme with ideal clock recovery.

### SamplerCentricScheme

```python
class SamplerCentricScheme(**kwargs)
```

Sampler-centric analysis scheme.

---

## Statistical Analysis

### PulseResponseProcessor

```python
class PulseResponseProcessor(
    ui: float,
    samples_per_symbol: int = 16
)
```

Process channel pulse responses for statistical eye analysis.

**Methods:**
- `process(pulse_response, dt) -> Dict`: Process pulse response
- `get_precursor_isi() -> np.ndarray`: Get precursor ISI samples
- `get_postcursor_isi() -> np.ndarray`: Get postcursor ISI samples

### ISICalculator

```python
class ISICalculator(modulation: ModulationFormat)
```

Calculate inter-symbol interference (ISI) effects.

**Methods:**
- `calculate_isi(pulse_response) -> Dict`: Calculate ISI components
- `get_voltage_pdf() -> Tuple[np.ndarray, np.ndarray]`: Get voltage PDF

### NoiseInjector

```python
class NoiseInjector(
    noise_sigma: float = 0.0,
    noise_type: str = 'gaussian'
)
```

Inject noise into statistical eye analysis.

**Methods:**
- `inject_noise(voltage_pdf) -> np.ndarray`: Inject noise into PDF
- `set_noise_sigma(sigma)`: Update noise sigma

### JitterInjector

```python
class JitterInjector(
    dj: float = 0.0,
    rj: float = 0.0,
    sj: Dict[float, float] = None
)
```

Inject jitter into statistical eye analysis.

**Parameters:**
- `dj`: Deterministic jitter in UI (peak-to-peak)
- `rj`: Random jitter standard deviation in UI
- `sj`: Dictionary of {frequency: amplitude} for sinusoidal jitter

### BERCalculator

```python
class BERCalculator(
    modulation: ModulationFormat,
    ber_algorithm: str = 'oif_cei'
)
```

OIF-CEI compliant BER calculator with strict conditional probability.

**Methods:**
- `calculate_ber(eye_data, threshold, phase) -> float`: Calculate BER at point
- `calculate_ber_contour(eye_matrix, levels) -> np.ndarray`: Calculate BER contour
- `bathtub_time(eye_data, threshold) -> Tuple[np.ndarray, np.ndarray]`: Time bathtub
- `bathtub_voltage(eye_data, phase) -> Tuple[np.ndarray, np.ndarray]`: Voltage bathtub

---

## Jitter Analysis

### JitterDecomposer

```python
class JitterDecomposer(
    ui: float,
    method: str = 'dual-dirac',
    psd_nperseg: int = 16384
)
```

Jitter decomposition engine supporting multiple extraction methods.

**Methods:**

- `dual-dirac`: Dual-Dirac model with Gaussian fitting
- `tail-fit`: Tail-fitting for non-bimodal DJ
- `auto`: Auto-detect best method

**Methods:**

```python
extract(
    phase_array: np.ndarray,
    value_array: np.ndarray,
    target_ber: float = 1e-12
) -> Dict[str, Any]
```

Extract jitter components.

**Returns:**
```python
{
    'rj_sigma': float,          # Random jitter (seconds)
    'dj_pp': float,             # Deterministic jitter (seconds)
    'tj_at_ber': float,         # Total jitter at target BER
    'target_ber': float,        # Target BER used
    'q_factor': float,          # Q function value
    'fit_method': str,          # Method used
    'fit_quality': float,       # R-squared (0-1)
    'pj_info': {                # Periodic jitter info
        'detected': bool,
        'frequencies': List[float],
        'amplitudes': List[float],
        'count': int
    }
}
```

### JitterAnalyzer

```python
class JitterAnalyzer(
    modulation: str = 'nrz',
    signal_amplitude: float = 1.0
)
```

Jitter analyzer supporting NRZ and PAM4 multi-eye analysis.

**Methods:**

```python
analyze(
    signal: np.ndarray,
    time: np.ndarray,
    ber: float = 1e-12
) -> Union[Dict[str, float], List[Dict[str, Any]]]
```

Analyze jitter in signal.

**Returns:**
- NRZ: `{'rj': float, 'dj': float, 'tj': float}`
- PAM4: List of per-eye results `[{'eye_id': int, 'eye_name': str, 'rj': float, 'dj': float, 'tj': float}, ...]`

---

## BER Analysis

### BERAnalyzer

```python
class BERAnalyzer(
    modulation: ModulationFormat,
    target_ber: float = 1e-12
)
```

Comprehensive BER analyzer.

**Methods:**
- `analyze(eye_data) -> Dict`: Complete BER analysis
- `eye_opening(ber_threshold) -> Dict`: Eye opening at BER threshold

### BERContour

```python
class BERContour(
    levels: List[float] = [1e-6, 1e-9, 1e-12, 1e-15]
)
```

BER contour generator.

**Methods:**
- `generate(eye_matrix) -> np.ndarray`: Generate BER contour
- `plot(ax=None, **kwargs)`: Plot contour

### BathtubCurve

```python
class BathtubCurve(
    direction: str = 'time',
    ber_range: Tuple[float, float] = (1e-18, 1e-3)
)
```

Bathtub curve generator for time or voltage direction.

**Methods:**
- `generate(eye_data, threshold=None) -> Tuple[np.ndarray, np.ndarray]`: Generate curve
- `opening_at_ber(ber) -> float`: Eye opening at specific BER

### QFactor

```python
class QFactor
```

Q-factor conversion utilities.

**Static Methods:**
- `ber_to_q(ber: float) -> float`: Convert BER to Q-factor
- `q_to_ber(q: float) -> float`: Convert Q-factor to BER
- `tj_calculation(rj: float, dj: float, ber: float) -> float`: Calculate TJ

### JTolTemplate

```python
class JTolTemplate(
    name: str,
    frequencies: List[float],
    amplitudes: List[float]
)
```

Jitter tolerance template.

**Built-in Templates:**
- `ieee_802_3ck()`: IEEE 802.3ck template
- `oif_cei_56g()`: OIF-CEI-56G template
- `jedec_ddr5()`: JEDEC DDR5 template
- `pcie_gen6()`: PCIe Gen6 template

### JitterTolerance

```python
class JitterTolerance(
    template: JTolTemplate,
    margins: Dict[str, float] = None
)
```

Jitter tolerance test runner.

**Methods:**
- `test(measured_sj: Dict[float, float]) -> Dict`: Run JTOL test
- `calculate_margins() -> Dict`: Calculate margins

---

## Visualization

### plot_eye_diagram

```python
plot_eye_diagram(
    eye_matrix: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    ber_contour: np.ndarray = None,
    ax=None,
    cmap: str = 'hot',
    title: str = 'Eye Diagram',
    **kwargs
) -> Union[Figure, Axes]
```

Plot eye diagram with optional BER contour overlay.

### plot_jtol_curve

```python
plot_jtol_curve(
    frequencies: List[float],
    amplitudes: List[float],
    template: JTolTemplate = None,
    ax=None,
    title: str = 'Jitter Tolerance',
    **kwargs
) -> Union[Figure, Axes]
```

Plot JTOL curve with optional template comparison.

### plot_bathtub_curve

```python
plot_bathtub_curve(
    x: np.ndarray,
    ber: np.ndarray,
    direction: str = 'time',
    ax=None,
    title: str = 'Bathtub Curve',
    **kwargs
) -> Union[Figure, Axes]
```

Plot bathtub curve.

### create_analysis_report

```python
create_analysis_report(
    metrics: Dict[str, Any],
    output_file: str,
    format: str = 'pdf',
    include_plots: bool = True,
    **kwargs
) -> str
```

Create comprehensive analysis report.

---

## Type Aliases

```python
from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np

# Common type aliases
EyeMatrix = np.ndarray      # 2D eye diagram histogram
BERContour = np.ndarray     # 2D BER contour
Metrics = Dict[str, Any]    # Analysis results dictionary
Waveform = Tuple[np.ndarray, np.ndarray]  # (time, value) tuple
```

---

## Exceptions

```python
class EyeAnalyzerError(Exception):
    """Base exception for eye_analyzer."""
    pass

class ValidationError(EyeAnalyzerError):
    """Input validation error."""
    pass

class ConvergenceError(EyeAnalyzerError):
    """Algorithm convergence error."""
    pass
```
