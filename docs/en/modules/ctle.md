# CTLE Module Technical Documentation

🌐 **Languages**: [中文](../../modules/ctle.md) | [English](ctle.md)

**Level**: AMS Submodule (RX)  
**Class Name**: `RxCtleTdf`  
**Current Version**: v0.4 (2025-12-07)  
**Status**: Production Ready

---

## 1. Overview

The Continuous-Time Linear Equalizer (CTLE) is a core analog front-end module in the SerDes receiver. Its primary function is to compensate for the high-frequency attenuation introduced by high-speed channels, restoring signal quality degraded by channel impairments through frequency-selective amplification.

### 1.1 Design Principles

The core design concept of CTLE utilizes zero-pole transfer functions to achieve frequency-dependent gain characteristics:

- **Low-frequency signals**: Channels exhibit less attenuation at low frequencies, so CTLE provides lower gain (DC gain)
- **High-frequency signals**: Channels exhibit significant attenuation at high frequencies, which CTLE compensates for by boosting high-frequency gain through its zeros
- **Very high-frequency signals**: Bandwidth is limited through poles to avoid amplifying high-frequency noise

The mathematical form of the transfer function is:
```
H(s) = dc_gain × ∏(1 + s/ωz_i) / ∏(1 + s/ωp_j)
```
where ωz = 2π×fz (zero angular frequency) and ωp = 2π×fp (pole angular frequency).

### 1.2 Core Features

- **Differential Architecture**: Complete differential signal path with common-mode rejection support
- **Flexible Transfer Function**: Supports arbitrary multi-zero/multi-pole configurations
- **Non-ideal Effects Modeling**: Input offset, input noise, and output soft saturation
- **PSRR Modeling**: Power supply noise coupling to output through configurable transfer functions
- **CMFB Loop**: Common-mode feedback loop to stabilize output common-mode voltage
- **CMRR Modeling**: Input common-mode to differential output leakage path

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-09 | Initial version, single-ended interface |
| v0.2 | 2025-10 | Added PSRR/CMFB/CMRR functionality |
| v0.3 | 2025-11-23 | Changed to differential interface, unified common-mode control |
| v0.4 | 2025-12-07 | Enhanced testbench, support for multi-scenario simulation |

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

> **Important**: Even if PSRR functionality is not enabled, the `vdd` port must be connected (SystemC-AMS requires all ports to be connected).

### 2.2 Parameter Configuration (RxCtleParams)

#### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dc_gain` | double | 1.0 | DC gain (linear scale) |
| `zeros` | vector&lt;double&gt; | [] | List of zero frequencies (Hz) |
| `poles` | vector&lt;double&gt; | [] | List of pole frequencies (Hz) |
| `vcm_out` | double | 0.6 | Differential output common-mode voltage (V) |
| `offset_enable` | bool | false | Enable input offset |
| `vos` | double | 0.0 | Input offset voltage (V) |
| `noise_enable` | bool | false | Enable input noise |
| `vnoise_sigma` | double | 0.0 | Noise standard deviation (V, Gaussian distribution) |
| `sat_min` | double | -0.5 | Output minimum voltage (V) |
| `sat_max` | double | 0.5 | Output maximum voltage (V) |

#### PSRR Substructure

Power Supply Rejection Ratio path, modeling the effect of VDD ripple on differential output.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable PSRR modeling |
| `gain` | PSRR path gain (e.g., 0.01 represents -40dB) |
| `poles` | Low-pass filter pole frequencies |
| `vdd_nom` | Nominal supply voltage |

Working principle: `vdd_ripple = vdd - vdd_nom` → PSRR transfer function → coupled to differential output

#### CMFB Substructure

Common-Mode Feedback loop, stabilizing output common-mode voltage to the target value.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable CMFB loop |
| `bandwidth` | Loop bandwidth (Hz) |
| `loop_gain` | Loop gain |

Working principle: Measure output common-mode → Compare with target → Loop filter → Adjust common-mode

#### CMRR Substructure

Common-Mode Rejection Ratio path, modeling the leakage from input common-mode to differential output.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable CMRR modeling |
| `gain` | CM→DIFF leakage gain |
| `poles` | Low-pass filter pole frequencies |

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The CTLE module's `processing()` method adopts a strict multi-step pipeline processing architecture to ensure signal processing correctness and maintainability:

```
Input Reading → Offset Injection → Noise Injection → CTLE Filtering → Soft Saturation → PSRR Path → CMRR Path → Synthesis → CMFB → Output
```

**Step 1 - Input Reading**: Read signals from differential input ports, calculate differential component `vin_diff = in_p - in_n` and common-mode component `vin_cm = 0.5*(in_p + in_n)`.

**Step 2 - Offset Injection**: If `offset_enable` is enabled, superimpose DC offset voltage `vos` onto the differential signal to simulate offset caused by actual amplifier mismatch.

**Step 3 - Noise Injection**: If `noise_enable` is enabled, use the Mersenne Twister random number generator to produce Gaussian distributed noise with standard deviation specified by `vnoise_sigma`.

**Step 4 - CTLE Core Filtering**: This is the core function of CTLE. If zeros and poles are configured, apply the transfer function using SystemC-AMS's `sca_tdf::sca_ltf_nd` filter; otherwise apply DC gain directly.

**Step 5 - Soft Saturation**: Use hyperbolic tangent function `tanh(x/Vsat)*Vsat` to achieve smooth saturation, avoiding harmonic distortion from hard clipping and more realistically simulating analog circuit behavior.

**Step 6 - PSRR Path**: If enabled, calculate ripple of VDD deviating from nominal value, process through PSRR transfer function, and couple to differential output.

**Step 7 - CMRR Path**: If enabled, input common-mode signal leaks to differential output through the CMRR transfer function.

**Step 8 - Differential Synthesis**: Accumulate contributions from main channel, PSRR path, and CMRR path to form total differential output.

**Step 9 - CMFB Processing**: If common-mode feedback is enabled, measure output common-mode from previous cycle (to avoid algebraic loop), compare with target common-mode, and adjust through loop filter.

**Step 10 - Output Generation**: Generate differential output based on effective common-mode voltage and differential signal: `out_p = vcm + 0.5*vdiff`, `out_n = vcm - 0.5*vdiff`.

### 3.2 Transfer Function Construction Mechanism

The module uses dynamic polynomial convolution to construct transfer functions of arbitrary order:

1. **Initialization**: Numerator polynomial starts with DC gain as constant term, denominator starts with 1
2. **Zero Processing**: For each zero frequency fz, convolve numerator with `(1 + s/ωz)`
3. **Pole Processing**: For each pole frequency fp, convolve denominator with `(1 + s/ωp)`
4. **Coefficient Conversion**: Convert polynomial coefficients to `sca_util::sca_vector` format

Polynomial coefficient layout uses ascending power order: `[a0, a1, a2, ...]` represents `a0 + a1*s + a2*s² + ...`

### 3.3 Soft Saturation Design Philosophy

Traditional hard clipping introduces rich harmonic components, which does not match actual analog circuit behavior. This module uses the `tanh` function to achieve soft saturation:

- When input is much smaller than saturation voltage Vsat, output is approximately linear
- When input approaches Vsat, gain is progressively compressed
- Output asymptotically approaches ±Vsat but never reaches it

This design more accurately simulates the natural output swing limitation of actual amplifiers.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The CTLE testbench (`CtleTransientTestbench`) adopts a modular design supporting unified management of multiple test scenarios. Core design principles:

1. **Scenario-Driven**: Select different test scenarios through enumeration types, with each scenario automatically configuring corresponding signal sources and CTLE parameters
2. **Component Reuse**: Differential signal source, VDD source, signal monitor and other auxiliary modules are reusable
3. **Result Analysis**: Automatically select appropriate analysis methods based on scenario type

### 4.2 Test Scenario Definitions

The testbench supports five core test scenarios:

| Scenario | Command Line Parameter | Test Objective | Output File |
|----------|------------------------|----------------|-------------|
| BASIC_PRBS | `prbs` / `0` | Basic signal transmission and gain characteristics | ctle_tran_prbs.csv |
| FREQUENCY_RESPONSE | `freq` / `1` | Frequency response characteristics | ctle_tran_freq.csv |
| PSRR_TEST | `psrr` / `2` | Power Supply Rejection Ratio test | ctle_tran_psrr.csv |
| CMRR_TEST | `cmrr` / `3` | Common-Mode Rejection Ratio test | ctle_tran_cmrr.csv |
| SATURATION_TEST | `sat` / `4` | Large signal saturation test | ctle_tran_sat.csv |

### 4.3 Scenario Configuration Details

#### BASIC_PRBS - Basic PRBS Test

Verify CTLE's basic differential signal transmission and DC gain characteristics.

- **Signal Source**: PRBS-7 pseudo-random sequence
- **Input Amplitude**: 100mV
- **Symbol Rate**: 10 Gbps
- **Common-Mode Voltage**: 0.6V
- **VDD**: 1.0V stable supply
- **Verification Point**: Output amplitude ≈ Input amplitude × DC gain

#### FREQUENCY_RESPONSE - Frequency Response Test

Verify CTLE response characteristics at specific frequencies.

- **Signal Source**: Sine wave
- **Test Frequency**: 5 GHz
- **Input Amplitude**: 100mV
- **Verification Point**: Near zero/pole frequencies, gain should be higher than DC gain

#### PSRR_TEST - Power Supply Rejection Ratio Test

Verify the effect of VDD ripple on differential output.

- **Differential Input**: DC (no differential signal)
- **VDD Ripple**: 100mV @ 1MHz sine wave
- **PSRR Gain**: 0.01 (-40dB)
- **PSRR Pole**: 1MHz
- **Simulation Time**: Must be ≥3μs (3 complete cycles)
- **Verification Point**: Output differential ripple amplitude should be much smaller than VDD ripple

#### CMRR_TEST - Common-Mode Rejection Ratio Test

Verify the effect of input common-mode variation on differential output.

- **Differential Input**: 100mV small differential signal
- **CMRR Gain**: 0.001 (-60dB)
- **CMRR Pole**: 10MHz
- **Verification Point**: Common-mode leakage component in output should match configured CMRR

#### SATURATION_TEST - Saturation Test

Verify CTLE saturation behavior under large signal input.

- **Signal Source**: Square wave
- **Input Amplitude**: 500mV (large signal)
- **Frequency**: 1 GHz
- **Verification Point**: Output amplitude should be limited to sat_min/sat_max range

### 4.4 Signal Connection Topology

The module connection relationships in the testbench are as follows:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ DiffSignalSource  │       │    RxCtleTdf     │       │  SignalMonitor  │
│                   │       │                   │       │                   │
│  out_p ───────────┼───────▶ in_p             │       │                   │
│  out_n ───────────┼───────▶ in_n             │       │                   │
└─────────────────┘       │                   │       │                   │
                            │  out_p ───────────┼───────▶ in_p            │
┌─────────────────┐       │  out_n ───────────┼───────▶ in_n            │
│    VddSource      │       │                   │       │                   │
│                   │       │                   │       │  → Statistical Analysis        │
│  vdd ─────────────┼───────▶ vdd              │       │  → CSV Save         │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### 4.5 Auxiliary Module Descriptions

#### DiffSignalSource - Differential Signal Source

Supports four waveform types:
- **DC**: DC signal
- **SINE**: Sine wave
- **SQUARE**: Square wave
- **PRBS**: Pseudo-random sequence

Configurable parameters: Amplitude, frequency, common-mode voltage

#### VddSource - Power Supply Module

Supports two modes:
- **CONSTANT**: Stable supply
- **SINUSOIDAL**: Supply with sine wave ripple (for PSRR testing)

#### SignalMonitor - Signal Monitor

Functions:
- Real-time waveform data recording
- Statistical calculation (mean, RMS, peak-to-peak, max/min)
- CSV format waveform file output

---

## 5. Simulation Result Analysis

### 5.1 Statistical Metrics Description

| Metric | Calculation Method | Significance |
|--------|-------------------|--------------|
| Mean | Arithmetic average of all sample points | Reflects DC component of signal |
| RMS | Root Mean Square | Reflects effective value/power of signal |
| Peak-to-Peak | Maximum - Minimum | Reflects dynamic range of signal |
| Max/Min | Extreme value statistics | Used for saturation detection, etc. |

### 5.2 Typical Test Result Interpretation

#### BASIC_PRBS Test Result Example

Configuration: Input 100mV, DC gain 1.5, zero 2GHz, pole 30GHz

Expected Results:
- Differential output peak-to-peak ≈ 291mV (Input 200mV peak-to-peak × 1.5 ≈ 300mV, slight difference due to frequency response characteristics)
- Differential output mean ≈ 0 (PRBS signal average should be zero)
- Common-mode output mean ≈ 0.6V (equals vcm_out configuration value)

Analysis Method: DC gain = Output peak-to-peak / Input peak-to-peak

#### PSRR Test Result Interpretation

- VDD Ripple: 100mV @ 1MHz
- If output differential ripple < 1mV: VDD noise is effectively suppressed
- If output differential ripple is larger: PSRR configuration is active, actual PSRR value can be calculated

PSRR Calculation: `PSRR_dB = 20 * log10(Vdd_ripple / Vout_diff_ripple)`

#### Saturation Test Result Interpretation

- Input Amplitude: 500mV
- If linear: Output should be 500mV × 1.5 = 750mV
- Actual output peak-to-peak < 750mV × some ratio: Indicates entry into saturation region

### 5.3 Waveform Data File Format

CSV output format:
```
Time(s),Differential Signal(V),Common-Mode Signal(V)
0.000000e+00,0.000000,0.600000
1.000000e-11,0.001234,0.600000
...
```

Number of sample points depends on simulation time and time step (default 10ps step).

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
make ctle_tran_tb
cd tb
./ctle_tran_tb [scenario]
```

Scenario parameters:
- `prbs` or `0` - Basic PRBS test (default)
- `freq` or `1` - Frequency response test
- `psrr` or `2` - PSRR test
- `cmrr` or `3` - CMRR test
- `sat` or `4` - Saturation test

### 6.3 Result Viewing

After test completion, console outputs statistical results, waveform data is saved to CSV files. Use Python for visualization:

```bash
python scripts/plot_ctle_waveform.py
```

---

## 7. Technical Points

**Problem**: If CMFB loop directly uses current cycle output for measurement, it creates an algebraic loop (output depends on output).

**Solution**:
- CMFB uses **previous cycle output** (`m_out_p_prev`, `m_out_n_prev`) for measurement
- This introduces a one time-step delay, but avoids the algebraic loop
- For low-frequency CMFB (bandwidth typically 1MHz), this delay is negligible

### 7.2 Multi-Zero/Multi-Pole Transfer Functions

Supports arbitrary number of zeros and poles, automatically handles polynomial convolution. Total number of zeros and poles recommended ≤ 10, higher order filters may cause numerical instability.

### 7.3 Soft Saturation

Uses `tanh(x/Vsat)*Vsat` to achieve smooth saturation characteristics, reducing harmonic distortion and matching actual circuit behavior.

### 7.4 Optional Function Independent Control

PSRR, CMFB, and CMRR can all be independently enabled/disabled. When not enabled, corresponding filter objects are not created, saving memory and computation.

### 7.5 Time Step Setting

Default 10ps (100GHz sampling rate). Sampling rate should be much higher than highest pole frequency, recommended f_sample ≥ 20-50 × f_pole_max.

### 7.6 PSRR Test Special Requirements

For PSRR test scenarios, simulation time must be at least 3 microseconds to ensure complete coverage of at least 3 cycles of the 1MHz signal variation.

### 7.7 VDD Port Must Be Connected

Even if PSRR functionality is not used, the `vdd` port must be connected (SystemC-AMS requirement).

---

## 8. Reference Information

### 8.1 Related Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | RxCtleParams structure |
| Header File | `/include/ams/rx_ctle.h` | RxCtleTdf class declaration |
| Implementation File | `/src/ams/rx_ctle.cpp` | RxCtleTdf class implementation |
| Testbench | `/tb/rx/ctle/ctle_tran_tb.cpp` | Transient simulation test |
| Test Helpers | `/tb/rx/ctle/ctle_helpers.h` | Signal sources and monitor |
| Unit Test | `/tests/unit/test_ctle_basic.cpp` | GoogleTest unit tests |
| Waveform Plotting | `/scripts/plot_ctle_waveform.py` | Python visualization script |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++11 Standard
- GoogleTest 1.12.1 (Unit tests)

### 8.3 Configuration Example

Basic configuration:
```json
{
  "ctle": {
    "zeros": [2e9],
    "poles": [30e9],
    "dc_gain": 1.5,
    "vcm_out": 0.6
  }
}
```

---

**Document Version**: v0.4  
**Last Updated**: 2025-12-07  
**Author**: Yizhe Liu
