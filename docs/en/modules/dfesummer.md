# DFE Summer Module Technical Documentation

🌐 **Languages**: [中文](../../modules/dfesummer.md) | [English](dfesummer.md)

**Level**: AMS Sub-module (RX)  
**Class Name**: `RxDfeSummerTdf`  
**Current Version**: v0.5 (2025-12-21)  
**Status**: In Development

---

## 1. Overview

DFE Summer (Decision Feedback Equalization Summation) is located in the RX receive chain after CTLE/VGA and before the Sampler, and is the core module for implementing Decision Feedback Equalization (DFE). Its main function is to sum (subtract) the differential signal from the main path with the feedback signal generated based on historical decision bits, thereby canceling post-cursor inter-symbol interference (ISI), increasing eye opening, and reducing bit error rate.

### 1.1 Design Principles

The core design concept of DFE is to use already-decided historical symbols to predict and cancel the post-cursor ISI affecting the current symbol:

- **Source of Post-cursor ISI**: Frequency-dependent attenuation and group delay of high-speed channels cause each transmitted symbol to "smear" in the time domain, affecting the sampling point voltage of subsequent symbols
- **Feedback Compensation Mechanism**: Already-decided historical symbols (b[n-1], b[n-2], ...) generate feedback voltage through an FIR filter structure, which is subtracted from the current input signal
- **Causality Constraint**: The feedback path must have at least 1 UI delay to avoid forming an algebraic loop (current decision depends on current output, and current output depends on current decision)

The mathematical expression for the feedback voltage is:
```
v_fb = Σ_{k=1}^{N} c_k × map(b[n-k]) × vtap
```
Where:
- c_k: The k-th tap coefficient
- b[n-k]: The decision bit at the n-k-th UI (0 or 1)
- map(): Bit mapping function (0→-1, 1→+1 or 0→0, 1→1)
- vtap: Voltage scaling factor, converting bit mapping value to volts

The equalized output is: `v_eq = v_main - v_fb`

### 1.2 Core Features

- **Differential Architecture**: Complete differential signal path, compatible with preceding CTLE/VGA and subsequent Sampler
- **Multi-tap Support**: Supports 1-9 taps (typical configuration is 3-5), flexibly configurable according to channel characteristics
- **Bit Mapping Modes**: Supports ±1 mapping (recommended, more robust against DC offset) and 0/1 mapping
- **Adaptive Interface**: Receives real-time tap updates from the Adaption module through DE→TDF bridging ports
- **Soft Saturation Mechanism**: Optional output limiting to prevent signal distortion caused by over-compensation
- **Historical Bit Interface**: Receives externally maintained historical decision array through data_in port, simplifying module responsibilities

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-10-22 | Initial version, basic DFE summation function |
| v0.2 | 2025-10-22 | Configuration key `taps` renamed to `tap_coeffs` |
| v0.3 | 2025-12-18 | Added DE→TDF tap update port for integration with Adaption module |
| v0.4 | 2025-12-18 | Improved `data_in` interface to array form, clarified length constraints |
| v0.5 | 2025-12-21 | Improved document structure, added testbench architecture and simulation analysis chapters |

---

## 2. Module Interface

### 2.1 Port Definition (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in_p` | Input | double | Differential input positive (from VGA) |
| `in_n` | Input | double | Differential input negative (from VGA) |
| `data_in` | Input | vector&lt;int&gt; | Historical decision data array |
| `out_p` | Output | double | Differential output positive (to Sampler) |
| `out_n` | Output | double | Differential output negative (to Sampler) |

**DE→TDF Parameter Update Port** (Optional):

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `tap_coeffs_de` | Input | vector&lt;double&gt; | Tap coefficient updates from Adaption |

> **About data_in Port**:
> - Array length is determined by the length N of `tap_coeffs`
> - `data_in[0]` is the most recent decision b[n-1], `data_in[1]` is b[n-2], ..., `data_in[N-1]` is b[n-N]
> - Array is maintained and updated by RX top-level module or Sampler, DFE Summer only reads and does not modify

### 2.2 Parameter Configuration (RxDfeSummerParams)

#### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tap_coeffs` | vector&lt;double&gt; | [] | Post-cursor tap coefficient list, in k=1...N order |
| `ui` | double | 2.5e-11 | Unit interval (seconds), for TDF timestep |
| `vcm_out` | double | 0.0 | Differential output common-mode voltage (V) |
| `vtap` | double | 1.0 | Bit mapping voltage scaling factor |
| `map_mode` | string | "pm1" | Bit mapping mode: "pm1" (±1) or "01" |
| `enable` | bool | true | Module enable, false for pass-through mode |

#### Saturation Limiting Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sat_enable` | bool | false | Enable output limiting |
| `sat_min` | double | -0.5 | Minimum output voltage (V) |
| `sat_max` | double | 0.5 | Maximum output voltage (V) |

#### Initialization Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_bits` | vector&lt;double&gt; | [0,...] | Historical bit initialization values, length must equal N |

#### Derived Parameters

| Parameter | Description |
|-----------|-------------|
| `tap_count` | Number of taps N, equals `tap_coeffs.size()`, determines `data_in` array length |

#### Bit Mapping Mode Description

- **pm1 Mode** (Recommended): 0 → -1, 1 → +1
  - Feedback voltage is symmetric, resistant to DC offset
  - When all historical bits are 0, feedback voltage is negative

- **01 Mode**: 0 → 0, 1 → 1
  - Feedback voltage is asymmetric
  - Requires additional DC offset compensation

---

## 3. Core Implementation Mechanism

### 3.1 Signal Processing Flow

DFE Summer module's `processing()` method adopts a strict multi-step processing architecture:

```
Input reading → Enable check → Historical data validation → Feedback calculation → Differential summation → Optional limiting → Common-mode synthesis → Output
```

**Step 1 - Input Reading**: Read signals from differential input ports, calculate differential component `v_main = in_p - in_n`.

**Step 2 - Enable Check**: If `enable=false`, directly pass input signals through common-mode synthesis to output (pass-through mode).

**Step 3 - Historical Data Validation**: Read `data_in` array, validate if length equals `tap_count`. If mismatched:
- Length insufficient: Pad with 0
- Length excessive: Truncate
- Warning log should be output

**Step 4 - Feedback Calculation**: Iterate through all taps, calculate total feedback voltage:
```cpp
v_fb = 0.0;
for (int k = 0; k < tap_count; k++) {
    double bit_val = map(data_in[k], map_mode);  // Bit mapping
    v_fb += tap_coeffs[k] * bit_val * vtap;
}
```

**Step 5 - Differential Summation**: Subtract feedback voltage from main path signal: `v_eq = v_main - v_fb`

**Step 6 - Optional Limiting**: If `sat_enable` is enabled, use soft saturation function:
```cpp
if (sat_enable) {
    double Vsat = 0.5 * (sat_max - sat_min);
    v_eq = tanh(v_eq / Vsat) * Vsat;
}
```

**Step 7 - Common-mode Synthesis**: Generate differential output based on common-mode voltage:
```
out_p = vcm_out + 0.5 * v_eq
out_n = vcm_out - 0.5 * v_eq
```

### 3.2 Tap Update Mechanism

DFE Summer supports two tap configuration modes:

#### Static Mode (No Adaptation)

- Internal `tap_coeffs` maintains initial values from configuration file
- Applicable to scenarios where channel characteristics are known and stable
- Tap coefficients can be pre-determined through offline training or channel simulation

#### Dynamic Mode (Linked with Adaption)

1. **Initialization Phase**: DFE Summer initializes internal coefficients according to `tap_coeffs` in configuration
2. **Runtime Update**:
   - Adaption module executes LMS/Sign-LMS algorithms in DE domain
   - New taps are passed through `tap_coeffs_de` port
   - DFE Summer reads latest coefficients in each TDF period
   - Updates take effect at the start of next UI

**Length Consistency Constraint**: Runtime updated tap array length must match initial configuration, if mismatched should error or truncate/pad.

### 3.3 Zero-delay Loop Avoidance

**Problem Essence**:
If current bit b[n] is directly used for current output feedback calculation, an algebraic loop forms:
- Current output v_eq[n] depends on feedback v_fb[n]
- Feedback v_fb[n] depends on decision b[n]
- Decision b[n] depends on sampling value, which comes from v_eq[n]

**Consequences**:
- Numerical instability, simulation timestep drastically reduced
- May cause simulation stall or divergence
- Physically unrealistic "instantaneous perfect cancellation" behavior

**Avoidance Solution**:
- Strictly use historical symbols b[n-k] (k≥1) for feedback calculation
- `data_in` array mechanism naturally guarantees this: `data_in[0]` is at least b[n-1]
- Array update is performed by external module after current UI decision completes, before next UI processing

### 3.4 Pass-through Mode Design

DFE Summer is equivalent to pass-through when any of the following conditions are met:

1. **Explicit Disable**: `enable = false`
2. **All-zero Taps**: All elements of `tap_coeffs` are 0
3. **Empty Tap Configuration**: `tap_coeffs` is empty array

In pass-through mode:
- `v_fb = 0`, output = input (after common-mode synthesis)
- `data_in` array values do not affect output
- But `data_in` should still maintain valid length for compatibility with subsequent adaptive enablement

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

DFE Summer testbench (`DfeSummerTransientTestbench`) adopts modular design with core concepts:

1. **Scenario-driven**: Select different test scenarios through enumeration types, each scenario automatically configures signal source, tap coefficients, and historical bits
2. **Component Reuse**: Differential signal source, historical bit generator, signal monitor and other auxiliary modules are reusable
3. **Eye Diagram Comparison**: Focus on verifying eye opening changes before and after DFE enablement

### 4.2 Test Scenario Definitions

| Scenario | Command Line Parameter | Test Objective | Output File |
|----------|------------------------|----------------|-------------|
| BYPASS_TEST | `bypass` / `0` | Verify pass-through mode consistency | dfe_summer_bypass.csv |
| BASIC_DFE | `basic` / `1` | Basic DFE feedback function | dfe_summer_basic.csv |
| MULTI_TAP | `multi` / `2` | Multi-tap configuration test | dfe_summer_multi.csv |
| ADAPTATION | `adapt` / `3` | Adaptive tap update | dfe_summer_adapt.csv |
| SATURATION | `sat` / `4` | Large signal saturation test | dfe_summer_sat.csv |

### 4.3 Scenario Configuration Details

#### BYPASS_TEST - Pass-through Mode Test

Verify that output matches input when DFE is disabled or taps are all zero.

- **Signal Source**: PRBS-7 pseudo-random sequence
- **Input Amplitude**: 100mV
- **Tap Configuration**: `tap_coeffs = [0, 0, 0]` or `enable = false`
- **Verification Point**: `out_diff ≈ in_diff` (allowing minor numerical error)

#### BASIC_DFE - Basic DFE Test

Verify basic feedback function under single-tap or few-tap configuration.

- **Signal Source**: PRBS signal with ISI (channel effect simulated through ISI injection module)
- **Tap Configuration**: `tap_coeffs = [0.1]` (single tap)
- **Historical Bits**: Synchronized generation with input PRBS
- **Verification Points**:
  - Feedback voltage matches formula calculation
  - Output ISI reduced

#### MULTI_TAP - Multi-tap Test

Verify performance under typical 3-5 tap configuration.

- **Signal Source**: PRBS signal with multi-cursor ISI
- **Tap Configuration**: `tap_coeffs = [0.08, 0.05, 0.03]`
- **Verification Points**:
  - Each tap independently effective
  - Total feedback voltage correctly accumulated

#### ADAPTATION - Adaptive Update Test

Verify linkage function with Adaption module.

- **Initial Taps**: `tap_coeffs = [0, 0, 0]`
- **Update Sequence**: Gradually inject new tap values through `tap_coeffs_de` port
- **Verification Points**:
  - Tap updates take effect at next UI
  - Update process glitch-free

#### SATURATION - Saturation Test

Verify limiting behavior under large signal input.

- **Signal Source**: Large amplitude square wave (500mV)
- **Tap Configuration**: Large coefficients `tap_coeffs = [0.3, 0.2, 0.1]`
- **Saturation Configuration**: `sat_min = -0.4V, sat_max = 0.4V`
- **Verification Point**: Output amplitude limited to sat_min/sat_max range

### 4.4 Signal Connection Topology

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  DiffSignalSrc  │       │  RxDfeSummerTdf │       │  SignalMonitor  │
│                 │       │                 │       │                 │
│  out_p ─────────┼───────▶ in_p            │       │                 │
│  out_n ─────────┼───────▶ in_n            │       │                 │
└─────────────────┘       │                 │       │                 │
                          │  out_p ─────────┼───────▶ in_p            │
┌─────────────────┐       │  out_n ─────────┼───────▶ in_n            │
│  HistoryBitGen  │       │                 │       │  → Statistical Analysis    │
│                 │       │                 │       │  → CSV Save       │
│  data ──────────┼───────▶ data_in         │       └─────────────────┘
└─────────────────┘       │                 │
                          │                 │
┌─────────────────┐       │                 │
│  AdaptionMock   │       │                 │
│  (DE Domain)    │       │                 │
│  taps ──────────┼───────▶ tap_coeffs_de   │
└─────────────────┘       └─────────────────┘
```

### 4.5 Auxiliary Module Description

#### DiffSignalSource - Differential Signal Source

Reused with CTLE testbench, supports:
- DC, SINE, SQUARE, PRBS waveforms
- Configurable amplitude, frequency, common-mode voltage

#### HistoryBitGenerator - Historical Bit Generator

Generates historical decision array synchronized with input signal:
- Input: Current bit stream (reference PRBS)
- Output: Historical bit array of length N
- Function: Maintains FIFO queue, shifts and updates every UI

#### ISIInjector - ISI Injection Module

Adds controllable post-cursor ISI to input signal:
- Parameters: ISI coefficients for each cursor
- Used to simulate channel effects, verify DFE cancellation capability

#### AdaptionMock - Adaption Simulator

Simulates Adaption module behavior:
- Outputs tap updates according to preset schedule
- Used to verify DE→TDF port function

---

## 5. Simulation Results Analysis

### 5.1 Statistical Metrics Description

#### General Statistical Metrics

| Metric | Calculation Method | Significance |
|--------|-------------------|--------------|
| Mean | Arithmetic average of all sampling points | Reflects DC component of signal |
| RMS | Root mean square $\sqrt{\frac{1}{N}\sum v_i^2}$ | Reflects effective value/power of signal |
| Peak-to-Peak | Maximum - Minimum | Reflects dynamic range of signal |
| Max/Min | Extreme value statistics | Used to judge saturation and other boundary behaviors |

#### DFE-Specific Performance Metrics

| Metric | Calculation Method | Significance |
|--------|-------------------|--------------|
| Eye Height | Eye diagram center vertical opening | Reflects signal quality, should increase after DFE |
| Eye Width | Eye diagram center horizontal opening | Reflects timing margin |
| ISI Residual | Post-cursor ISI component of DFE output | Should be close to zero, reflecting equalization effect |
| Feedback Voltage Error | Difference between actual v_fb and theoretical $\sum c_k \cdot \text{map}(b_{n-k}) \cdot V_{tap}$ | Verifies implementation correctness |
| Equalization Gain | (Eye height after DFE - Eye height before DFE) / Eye height before DFE | Quantifies equalization improvement |

#### Metric Calculation Formulas

**Eye Height Calculation**:
```
eye_height = min(V_1_low) - max(V_0_high)
```
Where V_1_low is the lower boundary distribution of logic "1" sampling points, V_0_high is the upper boundary distribution of logic "0" sampling points.

**ISI Residual Calculation**:
```
ISI_residual = Σ|h_k - c_k| for k = 1 to N
```
Where h_k is the channel post-cursor response, c_k is the DFE tap coefficient. Ideally c_k = h_k, residual is zero.

### 5.2 Typical Test Results Interpretation

#### BYPASS Test Results Example

Configuration: `tap_coeffs = [0, 0, 0]`, 100mV PRBS input

Expected Results:
- Differential output peak-to-peak ≈ Input peak-to-peak (200mV)
- Output waveform completely matches input waveform
- Any measurable difference should be < 1μV (numerical precision range)

Analysis Method:
- Calculate cross-correlation coefficient of input and output differential signals, should be > 0.9999
- Plot input-output difference waveform, verify no systematic offset

#### BASIC_DFE Test Results Interpretation

Configuration: Single tap `tap_coeffs = [0.1]`, PRBS with 10% h1 ISI

Assumed Scenario:
- Input signal: `v_in[n] = v_data[n] + 0.1 × v_data[n-1]` (post-cursor ISI)
- DFE feedback: `v_fb[n] = 0.1 × map(b[n-1])`

Expected Results:
- If tap coefficient matches ISI coefficient, post-cursor ISI should be completely canceled
- Eye diagram vertical opening should increase by about 10%
- Peak-to-peak standard deviation of output differential signal should decrease

Analysis Method:
- Compare eye diagram overlays before and after DFE enablement
- Calculate eye height improvement percentage: `(eye_height_after - eye_height_before) / eye_height_before × 100%`

#### MULTI_TAP Test Results Interpretation

Configuration: 3 taps `tap_coeffs = [0.08, 0.05, 0.03]`, PRBS with multi-cursor ISI

Assumed Scenario:
- Channel impulse response: h0=1.0, h1=0.08, h2=0.05, h3=0.03
- Input signal contains 3 post-cursor ISI components

Expected Results:
- Tap 1 cancels h1 ISI
- Tap 2 cancels h2 ISI
- Tap 3 cancels h3 ISI
- Total ISI residual < 5% (considering quantization and noise)

Analysis Method:
- Independently verify each tap's contribution: zero out individual taps, observe corresponding ISI component recovery
- Plot pulse response comparison: channel response vs DFE response

Numerical Example:
| Configuration | Eye Height (mV) | Eye Height Improvement |
|---------------|-----------------|------------------------|
| No DFE | 160 | - |
| Single tap (c1=0.08) | 176 | +10% |
| 3 taps (c1/c2/c3) | 192 | +20% |

#### ADAPTATION Test Results Interpretation

Initial taps all zero, updated to `[0.05, 0.03]` at t=100ns

Expected Results:
- t < 100ns: Output = pass-through (no DFE effect)
- t ≥ 100ns + 1 UI: New taps take effect, DFE compensation begins
- No output glitches or discontinuities during transition

Analysis Method:
- Take 10 UI waveforms before and after tap update moment
- Verify update effective delay = 1 UI (causality guarantee)
- Check waveform continuity at transition point

Verification Points:
- Feedback voltage change before and after update matches formula calculation
- If update moment coincides with symbol transition, no abnormal pulse should appear

#### SATURATION Test Results Interpretation

Configuration: Large coefficients `tap_coeffs = [0.3, 0.2, 0.1]`, `sat_min = -0.4V, sat_max = 0.4V`

Assumed Scenario:
- Input signal: 500mV square wave
- Theoretical output (without saturation): may exceed ±0.6V

Expected Results:
- Output differential signal limited to ±0.4V range
- Limiting uses soft saturation (tanh), waveform has no hard clipping artifacts
- Gain compression near saturation region, linear away from saturation region

Analysis Method:
- Plot input-output transfer curve, verify soft saturation characteristic
- Measure output peak-to-peak, confirm ≤ 0.8V (sat_max - sat_min)

Numerical Example:
| Input Diff (mV) | Unlimited Output (mV) | Limited Output (mV) |
|-----------------|----------------------|---------------------|
| 300 | 300 | 295 |
| 400 | 400 | 375 |
| 500 | 500 | 395 |
| 600 | 600 | 399 |

### 5.3 Waveform Data File Format

CSV Output Format:
```
Time(s),Input Diff(V),Output Diff(V),Feedback Voltage(V),Historical Bits
0.000000e+00,0.100000,0.100000,0.000000,"[0,0,0]"
2.500000e-11,0.095000,0.085000,0.010000,"[1,0,0]"
5.000000e-11,0.102000,0.092000,0.010000,"[1,1,0]"
...
```

#### Column Definitions

| Column Name | Unit | Description |
|-------------|------|-------------|
| Time | seconds (s) | Simulation timestamp |
| Input Diff | volts (V) | in_p - in_n |
| Output Diff | volts (V) | out_p - out_n |
| Feedback Voltage | volts (V) | v_fb = Σ c_k × map(b[n-k]) × vtap |
| Historical Bits | - | JSON format array, e.g. "[1,0,1]" |

#### Sampling Strategy

- **Default Sampling Rate**: One data point per UI
- **High Resolution Mode**: 10-20 sampling points per UI (for eye diagram plotting)
- **File Size Estimate**: 1μs simulation @ 40Gbps ≈ 40,000 lines ≈ 2MB

#### Data Post-processing Suggestions

```python
import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv('dfe_summer_basic.csv')

# Calculate eye height (simplified)
v_diff = df['Output Diff(V)']
eye_height = np.percentile(v_diff[v_diff > 0], 5) - np.percentile(v_diff[v_diff < 0], 95)
print(f"Eye Height: {eye_height*1000:.2f} mV")

# Calculate equalization gain
v_in = df['Input Diff(V)']
eye_height_in = np.percentile(v_in[v_in > 0], 5) - np.percentile(v_in[v_in < 0], 95)
eq_gain = (eye_height - eye_height_in) / eye_height_in * 100
print(f"Equalization Gain: {eq_gain:.1f}%")
```

---

## 6. Running Guide

### 6.1 Environment Configuration

Environment variables need to be configured before running tests:

```bash
source scripts/setup_env.sh
```

Or manually set:
```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
```

Verify environment configuration:
```bash
echo $SYSTEMC_HOME       # Should output SystemC installation path
echo $SYSTEMC_AMS_HOME   # Should output SystemC-AMS installation path
```

### 6.2 Build and Run

```bash
# Create build directory and compile
mkdir -p build && cd build
cmake ..
make dfe_summer_tran_tb

# Run test (in tb directory)
cd tb
./dfe_summer_tran_tb [scenario]
```

Scenario Parameters:
| Parameter | Number | Description |
|-----------|--------|-------------|
| `bypass` | `0` | Pass-through mode test (default) |
| `basic` | `1` | Basic DFE single-tap test |
| `multi` | `2` | Multi-tap configuration test |
| `adapt` | `3` | Adaptive tap update test |
| `sat` | `4` | Large signal saturation test |

Run Examples:
```bash
# Run basic DFE test
./dfe_summer_tran_tb basic

# Run multi-tap test
./dfe_summer_tran_tb 2

# Run all scenarios (batch test)
for i in 0 1 2 3 4; do ./dfe_summer_tran_tb $i; done
```

### 6.3 Parameter Configuration Examples

DFE Summer supports parameterization through JSON configuration files. Below are quick-start configurations for different application scenarios.

#### Quick Verification Configuration (Single Tap)

Suitable for preliminary function verification and debugging:

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.1],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  }
}
```

#### Typical Application Configuration (3 Taps)

Suitable for regular applications with moderate ISI channels:

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.08, 0.05, 0.03],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true,
    "sat_enable": false
  }
}
```

#### High Performance Configuration (5 Taps + Limiting)

Suitable for severe ISI channels requiring more taps and output protection:

```json
{
  "dfe_summer": {
    "tap_coeffs": [0.10, 0.07, 0.05, 0.03, 0.02],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true,
    "sat_enable": true,
    "sat_min": -0.5,
    "sat_max": 0.5,
    "init_bits": [0, 0, 0, 0, 0]
  }
}
```

#### Configuration Loading Methods

```bash
# Use command line parameter to specify configuration file
./dfe_summer_tran_tb basic --config config/dfe_3tap.json

# Or through environment variable
export SERDES_CONFIG=config/dfe_5tap.json
./dfe_summer_tran_tb multi
```

### 6.4 Result Viewing

After test completion, console outputs statistical results, waveform data is saved to CSV file.

#### Console Output Example

```
=== DFE Summer Transient Simulation ===
Scenario: BASIC_DFE
Duration: 1.0 us
Tap count: 1
Tap coeffs: [0.10]

--- Statistics ---
Input  diff: mean=0.000 mV, pp=200.0 mV, rms=70.7 mV
Output diff: mean=0.000 mV, pp=185.2 mV, rms=65.3 mV
Feedback:    mean=0.000 mV, pp=100.0 mV, rms=35.4 mV
Eye height improvement: +8.5%

Output file: dfe_summer_basic.csv
```

#### Waveform Visualization

```bash
# Basic waveform viewing
python scripts/plot_dfe_waveform.py dfe_summer_basic.csv

# Input-output comparison
python scripts/plot_dfe_waveform.py dfe_summer_basic.csv --compare

# Eye diagram overlay plotting
python scripts/plot_eye.py dfe_summer_basic.csv --samples-per-ui 20
```

#### Eye Diagram Comparison Analysis

Compare eye diagram changes before and after DFE enablement, quantify equalization effect:

```bash
# Generate comparison report
python scripts/compare_eye_dfe.py \
    --before dfe_summer_bypass.csv \
    --after dfe_summer_basic.csv \
    --output report/dfe_comparison.html

# Batch compare multi-tap configurations
python scripts/compare_eye_dfe.py \
    --before dfe_summer_bypass.csv \
    --after dfe_summer_basic.csv dfe_summer_multi.csv \
    --labels "1-tap" "3-tap" \
    --output report/dfe_multi_comparison.html
```

#### Result File Summary

| Scenario | Output File | Main Analysis Content |
|----------|-------------|----------------------|
| BYPASS | dfe_summer_bypass.csv | Baseline reference, no DFE effect |
| BASIC_DFE | dfe_summer_basic.csv | Single-tap eye height improvement |
| MULTI_TAP | dfe_summer_multi.csv | Multi-tap cumulative effect |
| ADAPTATION | dfe_summer_adapt.csv | Tap update transition behavior |
| SATURATION | dfe_summer_sat.csv | Limiting transfer curve |

---

## 7. Technical Notes

### 7.1 Causality Guarantee

DFE feedback must strictly guarantee at least 1 UI delay. This design achieves through following mechanisms:

1. **data_in Interface Design**: `data_in[0]` corresponds to b[n-1], not b[n]
2. **External Update Responsibility**: Historical bit array is updated by RX top-level, DFE Summer only reads
3. **Update Timing**: Array update occurs after current UI decision completes, before next UI processing

If current bit is mistakenly used for feedback, SystemC-AMS may report algebraic loop errors or abnormally slow simulation.

### 7.2 Tap Coefficient Range

- **Typical Range**: 0.01 ~ 0.3 (depending on channel ISI characteristics)
- **Sign Convention**: Positive coefficients used to cancel in-phase ISI, negative coefficients used to cancel anti-phase ISI
- **Stability Constraint**: Sum of absolute tap values should be < 1 to avoid feedback divergence

It is recommended to determine optimal tap values through channel simulation or adaptive algorithms.

### 7.3 Tap Count Selection

| Tap Count | Applicable Scenario | Complexity |
|-----------|---------------------|------------|
| 1-2 | Short channel, low ISI | Low |
| 3-5 | Medium channel, typical application | Medium |
| 6-9 | Long channel, severe ISI | High |

More taps can cancel more post-cursor ISI, but increase:
- Power consumption and area
- Adaptive convergence time
- Error propagation risk

### 7.4 Coordination with Adaption Module

DFE Summer itself does not contain adaptive algorithms, responsibilities are separated as follows:

| Module | Responsibility |
|--------|----------------|
| DFE Summer | Read taps and historical bits, calculate feedback, execute summation |
| Adaption | Read error signal, execute LMS algorithm, output new taps |
| RX Top-level | Maintain historical bit array, coordinate module timing |

This separation simplifies each module design, facilitates independent testing and replacement.

### 7.5 Bit Mapping Mode Comparison

| Characteristic | pm1 Mode | 01 Mode |
|----------------|----------|---------|
| Mapping Rule | 0→-1, 1→+1 | 0→0, 1→1 |
| Feedback Symmetry | Symmetric | Asymmetric |
| DC Component | None (average is 0) | Present (average is 0.5×vtap) |
| Recommended Scenario | General (default) | Specific protocol requirements |

### 7.6 Timestep Setting

- TDF timestep should be set to UI: `set_timestep(ui)`
- Consistent with CDR/Sampler UI
- Too small timestep increases computational overhead, too large loses signal details

### 7.7 Known Limitations

This module implementation has following known limitations and constraints:

**Tap Count Limitation**:
- Current implementation supports 1-9 taps
- More than 9 taps may cause:
  - Feedback calculation delay too large, cannot complete within 1 UI
  - Error propagation accumulation, affecting adaptive convergence
- If channel ISI exceeds 9 UI, it is recommended to combine with FFE for pre-cursor equalization

**Numerical Precision Limitation**:
- Tap coefficient precision: recommended to keep 4 significant digits
- Too small tap coefficients (< 0.005) may be drowned by noise, limited practical effect
- Floating-point accumulation error: when tap count is large, accumulation order may affect last few digits of precision

**Timing Constraints**:
- `data_in` array update must be performed after current UI decision completes
- Tap update (through `tap_coeffs_de`) takes effect at start of next UI
- If tap update frequency exceeds UI period, only last update takes effect

**Simulation Performance Impact**:
- 5-tap configuration compared to pass-through mode, simulation speed reduced by about 10-15%
- Enabling saturation limiting (sat_enable) adds about 5% computational overhead
- Large-scale simulation (> 10M UI) recommended to use compilation optimization (-O2/-O3)

### 7.8 DFE vs FFE Comparison

DFE (Decision Feedback Equalizer) and FFE (Feed-Forward Equalizer) are two complementary equalization technologies with different applicable scenarios:

| Characteristic | DFE | FFE |
|----------------|-----|-----|
| Location | Receiver (RX) | Transmitter (TX) |
| Equalization Target | Post-cursor ISI | Pre-cursor + Post-cursor ISI |
| Noise Impact | Does not amplify noise (feedback based on decisions) | Amplifies high-frequency noise |
| Error Propagation | Yes (wrong decision leads to wrong feedback) | No |
| Power Consumption | Lower | Higher (requires high-speed FIR) |
| Implementation Complexity | Medium | Lower |

**Design Selection Guide**:

1. **Mild ISI Channel**: Use FFE only, 2-3 taps sufficient
2. **Moderate ISI Channel**: FFE + DFE combination, FFE handles pre-cursor + main cursor, DFE handles post-cursor
3. **Severe ISI Channel**: FFE + CTLE + DFE full-link equalization

**This Module Positioning**:

- DFE Summer focuses on post-cursor ISI cancellation
- Cooperates with TX-side FFE module
- Achieves tap adaptation through Adaption module, can be jointly optimized with FFE taps

---

## 8. Reference Information

### 8.1 Related Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | RxDfeSummerParams structure |
| Header File | `/include/ams/rx_dfe_summer.h` | RxDfeSummerTdf class declaration |
| Implementation File | `/src/ams/rx_dfe_summer.cpp` | RxDfeSummerTdf class implementation |
| Testbench | `/tb/rx/dfe_summer/dfe_summer_tran_tb.cpp` | Transient simulation test |
| Test Helpers | `/tb/rx/dfe_summer/dfe_summer_helpers.h` | Signal source and monitor |
| Unit Test | `/tests/unit/test_dfe_summer_basic.cpp` | GoogleTest unit test |
| Adaption Documentation | `/docs/modules/adaption.md` | Adaption module interface description |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 standard
- GoogleTest 1.12.1 (unit test)

### 8.3 Configuration Examples

Basic Configuration (3-tap DFE):
```json
{
  "dfe_summer": {
    "tap_coeffs": [0.08, 0.05, 0.03],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  }
}
```

Configuration with Limiting:
```json
{
  "dfe_summer": {
    "tap_coeffs": [0.1, 0.06, 0.04, 0.02],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "sat_enable": true,
    "sat_min": -0.5,
    "sat_max": 0.5,
    "init_bits": [0, 0, 0, 0],
    "enable": true
  }
}
```

Configuration with Adaptation:
```json
{
  "dfe_summer": {
    "tap_coeffs": [0, 0, 0],
    "ui": 2.5e-11,
    "vcm_out": 0.6,
    "vtap": 1.0,
    "map_mode": "pm1",
    "enable": true
  },
  "adaption": {
    "dfe_enable": true,
    "dfe_mu": 0.01,
    "dfe_algorithm": "sign_lms"
  }
}
```

---

**Document Version**: v0.5  
**Last Updated**: 2025-12-21  
**Author**: SerDes Design Team
