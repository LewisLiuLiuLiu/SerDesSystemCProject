# VGA Module Technical Documentation

🌐 **Languages**: [中文](../../modules/vga.md) | [English](vga.md)

**Level**: AMS Submodule (RX)  
**Class Name**: `RxVgaTdf`  
**Current Version**: v0.1 (2025-12-07)  
**Status**: Production Ready

---

## 1. Overview

The Variable Gain Amplifier (VGA) is a critical analog front-end module in the SerDes receiver, located after the CTLE. Its primary function is to provide configurable signal gain for dynamic signal amplitude adjustment in the Automatic Gain Control (AGC) loop, ensuring the subsequent sampler receives an appropriate signal level.

### 1.1 Design Principles

The core design concept of the VGA utilizes a zero-pole transfer function to achieve adjustable signal amplification:

- **Variable Gain**: Dynamic gain adjustment through the `dc_gain` parameter, suitable for AGC adaptive control
- **Frequency Shaping**: Frequency-selective gain characteristics can be achieved by configuring zeros and poles
- **Bandwidth Limitation**: Amplifier bandwidth is controlled through pole frequency to prevent excessive amplification of high-frequency noise

The mathematical form of the transfer function is:
```
H(s) = dc_gain × ∏(1 + s/ωz_i) / ∏(1 + s/ωp_j)
```
Where ωz = 2π×fz (zero angular frequency), ωp = 2π×fp (pole angular frequency).

### 1.2 Core Features

- **Differential Architecture**: Complete differential signal path with common-mode rejection support
- **Flexible Transfer Function**: Supports arbitrary multi-zero/multi-pole configurations
- **Variable Gain**: Gain adjustment through `dc_gain` parameter, suitable for AGC loops
- **Non-ideal Effect Modeling**: Input offset, input noise, output soft saturation
- **PSRR Modeling**: Power supply noise coupling to output through configurable transfer function
- **CMFB Loop**: Common-mode feedback loop to stabilize output common-mode voltage
- **CMRR Modeling**: Input common-mode to differential output leakage path

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-12-07 | Initial version, based on CTLE module architecture, differential interface, supports PSRR/CMFB/CMRR |

---

## 2. Module Interface

### 2.1 Port Definitions (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in_p` | Input | double | Differential input positive terminal |
| `in_n` | Input | double | Differential input negative terminal |
| `vdd` | Input | double | Supply voltage (for PSRR modeling) |
| `out_p` | Output | double | Differential output positive terminal |
| `out_n` | Output | double | Differential output negative terminal |

> **Important**: The `vdd` port must be connected even when PSRR functionality is not enabled (SystemC-AMS requires all ports to be connected).

### 2.2 Parameter Configuration (RxVgaParams)

#### Basic Parameters

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `dc_gain` | double | 2.0 | DC gain (linear multiplier) |
| `zeros` | vector&lt;double&gt; | [1e9] | List of zero frequencies (Hz) |
| `poles` | vector&lt;double&gt; | [20e9] | List of pole frequencies (Hz) |
| `vcm_out` | double | 0.6 | Differential output common-mode voltage (V) |
| `offset_enable` | bool | false | Enable input offset |
| `vos` | double | 0.0 | Input offset voltage (V) |
| `noise_enable` | bool | false | Enable input noise |
| `vnoise_sigma` | double | 0.0 | Noise standard deviation (V, Gaussian distribution) |
| `sat_min` | double | -0.5 | Output minimum voltage (V) |
| `sat_max` | double | 0.5 | Output maximum voltage (V) |

#### PSRR Sub-structure

Power Supply Rejection Ratio path, modeling the effect of VDD ripple on differential output.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable PSRR modeling |
| `gain` | PSRR path gain (e.g., 0.01 represents -40dB) |
| `poles` | Low-pass filter pole frequencies |
| `vdd_nom` | Nominal supply voltage |

Working principle: `vdd_ripple = vdd - vdd_nom` → PSRR transfer function → coupled to differential output

#### CMFB Sub-structure

Common-mode feedback loop, stabilizing output common-mode voltage to target value.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable CMFB loop |
| `bandwidth` | Loop bandwidth (Hz) |
| `loop_gain` | Loop gain |

Working principle: Measure output common-mode → Compare with target → Loop filter → Adjust common-mode

#### CMRR Sub-structure

Common-mode rejection ratio path, modeling input common-mode to differential output leakage.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable CMRR modeling |
| `gain` | CM→DIFF leakage gain |
| `poles` | Low-pass filter pole frequencies |

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The VGA module's `processing()` method adopts a strict multi-step pipeline processing architecture to ensure signal processing correctness and maintainability:

```
Input Reading → Offset Injection → Noise Injection → VGA Filtering → Soft Saturation → PSRR Path → CMRR Path → Combination → CMFB → Output
```

**Step 1 - Input Reading**: Read signals from differential input ports, calculate differential component `vin_diff = in_p - in_n` and common-mode component `vin_cm = 0.5*(in_p + in_n)`.

**Step 2 - Offset Injection**: If `offset_enable` is enabled, superimpose DC offset voltage `vos` onto the differential signal, simulating offset caused by actual amplifier mismatch.

**Step 3 - Noise Injection**: If `noise_enable` is enabled, use Mersenne Twister random number generator to produce Gaussian distributed noise, with standard deviation specified by `vnoise_sigma`.

**Step 4 - VGA Core Filtering**: This is the VGA's core function. If zeros/poles are configured, apply transfer function using SystemC-AMS's `sca_tdf::sca_ltf_nd` filter; otherwise directly apply DC gain.

**Step 5 - Soft Saturation**: Use hyperbolic tangent function `tanh(x/Vsat)*Vsat` to achieve smooth saturation, avoiding harmonic distortion from hard clipping and more realistically simulating analog circuit behavior.

**Step 6 - PSRR Path**: If enabled, calculate VDD ripple deviation from nominal value, process through PSRR transfer function, and couple to differential output.

**Step 7 - CMRR Path**: If enabled, input common-mode signal leaks to differential output through CMRR transfer function.

**Step 8 - Differential Combination**: Accumulate contributions from main path, PSRR path, and CMRR path to form total differential output.

**Step 9 - CMFB Processing**: If common-mode feedback is enabled, measure previous cycle's output common-mode (to avoid algebraic loop), compare with target common-mode, and adjust through loop filter.

**Step 10 - Output Generation**: Generate differential output based on effective common-mode voltage and differential signal: `out_p = vcm + 0.5*vdiff`, `out_n = vcm - 0.5*vdiff`.

### 3.2 Transfer Function Construction Mechanism

The module adopts a dynamic polynomial convolution method to construct transfer functions of arbitrary order:

1. **Initialization**: Numerator polynomial starts with DC gain as constant term, denominator is 1
2. **Zero Processing**: For each zero frequency fz, convolve numerator with `(1 + s/ωz)`
3. **Pole Processing**: For each pole frequency fp, convolve denominator with `(1 + s/ωp)`
4. **Coefficient Conversion**: Convert polynomial coefficients to `sca_util::sca_vector` format

Polynomial coefficient layout uses ascending power order: `[a0, a1, a2, ...]` represents `a0 + a1*s + a2*s² + ...`

### 3.3 Soft Saturation Design Concept

Traditional hard clipping introduces rich harmonic components, which does not match actual analog circuit behavior. This module uses the `tanh` function to achieve soft saturation:

- When input is much smaller than saturation voltage Vsat, output is approximately linear
- When input approaches Vsat, gain gradually compresses
- Output asymptotically approaches ±Vsat but never reaches it

This design more accurately simulates the natural limitations of output swing by transconductance stages.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Concept

The VGA testbench (`VgaTransientTestbench`) adopts a modular design, supporting unified management of multiple test scenarios. Core design concepts:

1. **Scenario-driven**: Select different test scenarios through enumeration type, with each scenario automatically configuring appropriate signal sources and VGA parameters
2. **Component Reuse**: Differential signal sources, VDD sources, signal monitors, and other auxiliary modules are reusable
3. **Result Analysis**: Automatically select appropriate analysis methods based on scenario type

### 4.2 Test Scenario Definitions

The testbench supports five core test scenarios:

| Scenario | Command Line Parameter | Test Objective | Output File |
|----------|------------------------|----------------|-------------|
| BASIC_PRBS | `prbs` / `0` | Basic signal transmission and gain characteristics | vga_tran_prbs.csv |
| FREQUENCY_RESPONSE | `freq` / `1` | Frequency response characteristics | vga_tran_freq.csv |
| PSRR_TEST | `psrr` / `2` | Power supply rejection ratio test | vga_tran_psrr.csv |
| CMRR_TEST | `cmrr` / `3` | Common-mode rejection ratio test | vga_tran_cmrr.csv |
| SATURATION_TEST | `sat` / `4` | Large signal saturation test | vga_tran_sat.csv |

### 4.3 Scenario Configuration Details

#### BASIC_PRBS - Basic PRBS Test

Verify VGA basic differential signal transmission and DC gain characteristics.

- **Signal Source**: PRBS-7 pseudo-random sequence
- **Input Amplitude**: 100mV
- **Symbol Rate**: 10 Gbps
- **Common-mode Voltage**: 0.6V
- **VDD**: 1.0V stable supply
- **Simulation Time**: 100ns
- **Verification Point**: Output amplitude ≈ Input amplitude × DC gain

#### FREQUENCY_RESPONSE - Frequency Response Test

Verify VGA response characteristics at specific frequencies.

- **Signal Source**: Sine wave
- **Test Frequency**: 5 GHz
- **Input Amplitude**: 100mV
- **Simulation Time**: 1μs (covers 5000 periods)
- **Verification Point**: Near zero/pole frequencies, gain should follow transfer function characteristics

#### PSRR_TEST - Power Supply Rejection Ratio Test

Verify the effect of VDD ripple on differential output.

- **Differential Input**: DC (no differential signal)
- **VDD Ripple**: 100mV @ 1MHz sine wave
- **PSRR Gain**: 0.01 (-40dB)
- **PSRR Pole**: 1MHz
- **Simulation Time**: Must be ≥3μs (covers 3 complete periods)
- **Verification Point**: Output differential ripple amplitude should be much smaller than VDD ripple

#### CMRR_TEST - Common-mode Rejection Ratio Test

Verify the effect of input common-mode variation on differential output.

- **Differential Input**: 100mV small differential signal
- **CMRR Gain**: 0.001 (-60dB)
- **CMRR Pole**: 10MHz
- **Simulation Time**: 3μs
- **Verification Point**: Common-mode leakage component in output should match set CMRR

#### SATURATION_TEST - Saturation Test

Verify VGA saturation behavior under large signal input.

- **Signal Source**: Square wave
- **Input Amplitude**: 500mV (large signal)
- **Frequency**: 1 GHz
- **Simulation Time**: 100ns
- **Verification Point**: Output amplitude should be limited to sat_min/sat_max range

### 4.4 Signal Connection Topology

Testbench module connection relationships are as follows:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ DiffSignalSource  │       │    RxVgaTdf      │       │  SignalMonitor  │
│                   │       │                   │       │                   │
│  out_p ───────────┼───────▶ in_p             │       │                   │
│  out_n ───────────┼───────▶ in_n             │       │                   │
└─────────────────┘       │                   │       │                   │
                            │  out_p ───────────┼───────▶ in_p            │
┌─────────────────┐       │  out_n ───────────┼───────▶ in_n            │
│    VddSource      │       │                   │       │                   │
│                   │       │                   │       │  → Statistical Analysis  │
│  vdd ─────────────┼───────▶ vdd              │       │  → CSV Save       │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### 4.5 Auxiliary Module Descriptions

#### DiffSignalSource - Differential Signal Source

Supports four waveform types:
- **DC**: DC signal
- **SINE**: Sine wave
- **SQUARE**: Square wave
- **PRBS**: Pseudo-random sequence (PRBS-7)

Configurable parameters: Amplitude, frequency, common-mode voltage, sampling rate

#### VddSource - Power Supply Module

Supports three modes:
- **CONSTANT**: Stable supply
- **SINUSOIDAL**: Supply with sine ripple (for PSRR testing)
- **RANDOM**: Supply with random noise

#### SignalMonitor - Signal Monitor

Functions:
- Real-time waveform data recording
- Statistical calculation (mean, RMS, peak-to-peak, max/min)
- CSV format waveform file output

---

## 5. Simulation Results Analysis

### 5.1 Statistical Indicators

| Indicator | Calculation Method | Significance |
|-----------|-------------------|--------------|
| Mean | Arithmetic average of all sample points | Reflects DC component of signal |
| RMS | Root mean square | Reflects effective value/power of signal |
| Peak-to-peak | Maximum - Minimum | Reflects dynamic range of signal |
| Max/Min | Extreme value statistics | Used to determine saturation, etc. |

### 5.2 Typical Test Result Interpretation

#### BASIC_PRBS Test Results Example

Configuration: Input 100mV, DC gain 2.0, zero at 1GHz, poles at 10GHz/20GHz

Expected results:
- Differential output peak-to-peak ≈ 400mV (Input 200mV peak-to-peak × 2.0 ≈ 400mV)
- Differential output mean ≈ 0 (PRBS signal average should be zero)
- Common-mode output mean ≈ 0.6V (equals vcm_out configuration value)

Analysis method: DC gain = Output peak-to-peak / Input peak-to-peak

#### PSRR Test Results Interpretation

- VDD ripple: 100mV @ 1MHz
- If output differential ripple < 1mV: VDD noise is effectively suppressed
- If output differential ripple is large: PSRR configuration is active, actual PSRR value can be calculated

PSRR calculation: `PSRR_dB = 20 * log10(Vdd_ripple / Vout_diff_ripple)`

#### Saturation Test Results Interpretation

- Input amplitude: 500mV
- If linear: Output should be 500mV × 2.0 = 1000mV
- Actual output peak-to-peak < 1000mV × some ratio: Indicates entering saturation region

### 5.3 Waveform Data File Format

CSV output format:
```
time,diff,cm
0.000000e+00,0.000000,0.600000
1.000000e-11,0.001234,0.600000
...
```

Number of sample points depends on simulation time and time step (default 10ps step, corresponding to 100GHz sampling rate).

---

## 6. Running Guide

### 6.1 Environment Configuration

Configure environment variables before running tests:

```bash
source scripts/setup_env.sh
```

### 6.2 Build and Run

```bash
cd build
cmake ..
make vga_tran_tb
cd tb
./vga_tran_tb [scenario]
```

Scenario parameters:
- `prbs` or `0` - Basic PRBS test (default)
- `freq` or `1` - Frequency response test
- `psrr` or `2` - PSRR test
- `cmrr` or `3` - CMRR test
- `sat` or `4` - Saturation test

### 6.3 Viewing Results

After test completion, console outputs statistical results, waveform data saved to CSV file. Use Python for visualization:

```bash
python scripts/plot_ctle_waveform.py  # Can reuse CTLE plotting script
```

---

## 7. Technical Highlights

### 7.1 CMFB Algebraic Loop Handling

**Problem**: If CMFB loop directly uses current cycle output for measurement, it creates an algebraic loop (output depends on output).

**Solution**:
- CMFB uses **previous cycle's output** (`m_out_p_prev`, `m_out_n_prev`) for measurement
- This introduces a time step delay, but avoids algebraic loop
- For low-frequency CMFB (bandwidth typically 10MHz), this delay is negligible

### 7.2 Multi-zero/Multi-pole Transfer Functions

Supports arbitrary number of zeros and poles, automatically handles polynomial convolution. Total number of zeros/poles recommended ≤ 10, higher order filters may cause numerical instability.

### 7.3 Soft Saturation

Uses `tanh(x/Vsat)*Vsat` to achieve smooth saturation characteristics, reducing harmonic distortion, consistent with actual circuit behavior.

### 7.4 Optional Function Independent Control

PSRR, CMFB, CMRR can all be independently enabled/disabled, no corresponding filter objects created when not enabled, saving memory and computation.

### 7.5 Time Step Setting

Default 10ps (100GHz sampling rate). Sampling rate should be much higher than highest pole frequency, recommended f_sample ≥ 20-50 × f_pole_max.

### 7.6 PSRR Test Special Requirements

For PSRR test scenarios, simulation time must be no less than 3 microseconds to ensure complete coverage of at least 3 periods of 1MHz signal variation.

### 7.7 VDD Port Must Be Connected

Even when not using PSRR functionality, the `vdd` port must be connected (SystemC-AMS requirement).

### 7.8 Main Differences from CTLE

| Item | CTLE | VGA |
|------|------|-----|
| Default DC gain | 1.5 | 2.0 |
| Default zero | 2 GHz | 1 GHz |
| Default poles | 30 GHz | 10 GHz, 20 GHz |
| CMFB loop bandwidth | 1 MHz | 10 MHz |
| CMFB loop gain | 1.0 | 10.0 |
| Primary application | Channel equalization | Variable gain/AGC |

---

## 8. Reference Information

### 8.1 Related Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | RxVgaParams structure |
| Header File | `/include/ams/rx_vga.h` | RxVgaTdf class declaration |
| Implementation File | `/src/ams/rx_vga.cpp` | RxVgaTdf class implementation |
| Testbench | `/tb/rx/vga/vga_tran_tb.cpp` | Transient simulation test |
| Test Utilities | `/tb/rx/vga/vga_helpers.h` | Signal sources and monitors |
| Unit Test | `/tests/unit/test_vga_basic.cpp` | GoogleTest unit tests |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++11 standard
- GoogleTest 1.12.1 (unit tests)

### 8.3 Configuration Example

Basic configuration:
```json
{
  "vga": {
    "zeros": [1e9],
    "poles": [10e9, 20e9],
    "dc_gain": 2.0,
    "vcm_out": 0.6
  }
}
```

---

**Document Version**: v0.1  
**Last Updated**: 2025-12-07  
**Author**: Yizhe Liu
