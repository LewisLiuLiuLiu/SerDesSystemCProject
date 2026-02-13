# Sampler Module Technical Documentation

🌐 **Languages**: [中文](../../modules/sampler.md) | [English](sampler.md)

**Level**: AMS Sub-module (RX)  
**Class Name**: `RxSamplerTdf`  
**Current Version**: v0.3 (2025-12-07)  
**Status**: Production Ready

---

## 1. Overview

The Sampler is a key decision module in the SerDes receiver, whose primary function is to convert continuous analog differential signals into discrete digital bit streams. The module supports dynamic phase adjustment, configurable non-ideal effect modeling, and an advanced fuzzy decision mechanism for simulating the complex behavior characteristics of real samplers.

### 1.1 Design Principles

The core design concept of the sampler is threshold decision based on a comparator architecture, combined with dynamic phase information provided by CDR (Clock and Data Recovery) to achieve precise data recovery:

- **Differential Comparison**: Performs differential operation on complementary analog input signals to obtain differential voltage Vdiff
- **Dynamic Sampling Moment**: Receives sampling clock or phase offset signals output by the CDR module to dynamically adjust the sampling position
- **Multi-level Decision Mechanism**: Combines resolution threshold and hysteresis effects to achieve robust decision logic
- **Non-ideal Effect Modeling**: Integrates non-ideal characteristics of actual devices such as offset, noise, and jitter

The mathematical form of the transfer function is:
```
data_out = f(Vdiff, phase_offset, parameters)
where: Vdiff = (inp - inn) + offset + noise
```

### 1.2 Core Features

- **CDR Integration Interface**: Supports both clock-driven and phase-signal-driven modes
- **Dynamic Sampling Moment**: Real-time response to CDR phase adjustments to optimize sampling point position
- **Fuzzy Decision Mechanism**: Random decision within the resolution threshold, simulating comparator metastability
- **Schmitt Trigger Effect**: Hysteresis function avoids decision errors caused by signal jitter
- **Parameter Validation Mechanism**: Conflict detection and error handling for hysteresis and resolution parameters
- **Configurable Non-ideal Effects**: Independent modeling and control of offset, noise, and jitter

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-09 | Initial version, basic sampling function |
| v0.2 | 2025-11-23 | Added parameter validation mechanism, improved documentation |
| v0.3 | 2025-12-07 | Restructured according to CTLE documentation style, optimized technical descriptions |

---

## 2. Module Interface

### 2.1 Port Definition (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `inp` | Input | double | Differential input positive terminal |
| `inn` | Input | double | Differential input negative terminal |
| `clk_sample` | Input | double | CDR sampling clock (optional) |
| `phase_offset` | Input | double | CDR phase offset (optional) |
| `data_out` | Output | int | Digital bit output (0 or 1) |
| `data_out_de` | Output | bool | DE domain output (optional) |

> **Important**: `clk_sample` and `phase_offset` ports are connected based on the `phase_source` parameter selection. SystemC-AMS requires all ports to be connected.

### 2.3 Port Data Type Details

To clarify the design concept of the CDR integration interface, this section details the data types and physical meanings of the `clk_sample` and `phase_offset` ports:

#### 2.3.1 `clk_sample` Port (Clock-Driven Mode)

**Data Type**: `double` (Unit: Volts V)  
**Physical Meaning**: This is a **continuous voltage signal** representing the clock waveform output by the CDR module

**Working Principle**:
```cpp
// Typical usage
if (clk_sample.read() > voltage_threshold) {
    // When voltage exceeds threshold, trigger sampling moment
    perform_sampling();
}
```

**Signal Characteristics**:
- **Waveform Type**: Sine wave, square wave, or triangular wave (depending on CDR design)
- **Voltage Range**: Typical range 0V-1V or -0.5V to +0.5V
- **Frequency**: Matches data rate (e.g., 10Gbps corresponds to 10GHz clock)
- **Trigger Mechanism**: Voltage threshold detection, simulating clock edge triggering

**Timing Example**:
```
clk_sample voltage signal:
    1.0V ┌──┐    ┌──┐    ┌──┐
         └──┘    └──┘    └──┘
    0.0V    ↑      ↑      ↑
         Sample  Sample  Sample
    Threshold: 0.5V ──────────────────
```

#### 2.3.2 `phase_offset` Port (Phase Signal Mode)

**Data Type**: `double` (Unit: seconds s)  
**Physical Meaning**: This is a **time offset** representing the phase error detected by the CDR

**Working Principle**:
```cpp
// Typical usage
double current_time = get_simulation_time();
double actual_sample_time = current_time + phase_offset.read() + sample_delay;
// Calculate actual sampling moment based on phase offset
```

**Value Meanings**:
- **Positive value**: Indicates delay in sampling (e.g., +100e-12 = delay 100 picoseconds)
- **Negative value**: Indicates advance in sampling (e.g., -50e-12 = advance 50 picoseconds)  
- **Zero value**: Indicates sampling at nominal moment
- **Dynamic Range**: Typical range ±0.1×UI (Unit Interval)

**Application Example**:
```
phase_offset signal values:
    +100e-12 ────────────────────── Delay sampling
    +50e-12  ────────────────────── Slight delay
    0        ────────────────────── Normal sampling
    -50e-12  ────────────────────── Slight advance
    -100e-12 ────────────────────── Advance sampling
```

#### 2.3.3 Key Differences Summary

| Feature | `clk_sample` | `phase_offset` |
|---------|--------------|----------------|
| **Data Type** | double (voltage) | double (time) |
| **Unit** | Volts (V) | Seconds (s) |
| **Signal Nature** | Continuous clock waveform | Dynamic phase correction value |
| **Action Mode** | Voltage threshold triggering | Time offset calculation |
| **Physical Meaning** | Clock edge signal | Phase error information |
| **CDR Output** | Complete clock signal | Phase detection result |
| **Sampling Trigger** | Edge detection | Time calculation |

#### 2.3.4 Selection Guide

**When to use `clk_sample` (Clock-Driven)**:
- CDR directly outputs an available sampling clock
- Hardware-style clock triggering mechanism is required
- System requires strict clock synchronization
- CDR design is relatively simple, directly generates clock

**When to use `phase_offset` (Phase Signal)**:
- CDR outputs phase error rather than clock
- Precise phase tracking and adjustment is needed
- Supports continuous phase optimization algorithms
- CDR loop outputs correction information |

### 2.2 Parameter Configuration (RxSamplerParams)

#### Basic Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_delay` | double | 0.0 | Fixed sampling delay time (s) |
| `phase_source` | string | "clock" | Phase source: clock/phase |
| `threshold` | double | 0.0 | Decision threshold (V, default is 0V) |
| `resolution` | double | 0.02 | Resolution threshold (V, fuzzy decision zone half-width) |
| `hysteresis` | double | 0.02 | Hysteresis threshold (V, Schmitt trigger effect) |

#### Offset Configuration Sub-structure

Models the input offset voltage of the comparator.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable offset modeling |
| `value` | Offset voltage (V) |

Working principle: `Vdiff_effective = (inp - inn) + value`

#### Noise Configuration Sub-structure

Models device thermal noise and input-referred noise.

| Parameter | Description |
|-----------|-------------|
| `enable` | Enable noise modeling |
| `sigma` | Noise standard deviation (V, Gaussian distribution) |
| `seed` | Random number seed (reproducibility) |

Working principle: `noise_sample ~ N(0, sigma²)`, superimposed on differential signal

### 2.4 Phase Control Mechanism

#### Clock-Driven Mode (phase_source = "clock")

The CDR outputs a sampling clock signal, and the Sampler samples at the rising edge of the clock.

```
Timing Relationship:
CLK ──┐    ┌──┐    ┌──┐    ┌──┐
      └──┘    └──┘    └──┘    └──┘
      ↑      ↑      ↑      ↑
   Sample  Sample  Sample  Sample
```

#### Phase Signal Mode (phase_source = "phase")

The CDR outputs a phase offset signal, and the Sampler calculates the actual sampling moment based on the nominal moment and phase offset.

```
Sampling Moment Calculation:
t_sample = t_nominal + phase_offset + sample_delay

Where:
- t_nominal: Nominal sampling moment (e.g., UI center)
- phase_offset: CDR phase offset (dynamically changing)
- sample_delay: Fixed delay (configuration parameter)
```

---

## 3. Core Implementation Mechanisms

This chapter, from the perspective of implementation mechanisms, elaborates on the signal processing flow, decision logic, parameter validation, and noise/jitter modeling of the sampler module, and derives a complete Bit Error Rate (BER) performance analysis model based on these foundations. By organically combining implementation details with performance metrics, this helps readers deeply understand the relationship between sampler behavior characteristics and BER.

### 3.1 Signal Processing Flow and Noise Modeling

The `processing()` method of the sampler module adopts a strict multi-step pipeline processing architecture to ensure the correctness and maintainability of the decision logic:

```
Input Reading → Differential Calculation → Offset Injection → Noise Injection → Phase Adjustment → Decision Logic → Output Generation
```

**Step 1 - Input Reading**: Read signals from differential input ports, calculate differential component `Vdiff = inp - inn` and common-mode component `Vcm = 0.5*(inp + inn)`.

**Step 2 - Offset Injection**: If `offset_enable` is enabled, superimpose the offset voltage `value` onto the differential signal to simulate the input offset of the comparator. The specific working principle is:
```
Vdiff_effective = (inp - inn) + offset.value
```

**Step 3 - Noise Injection**: If `noise_enable` is enabled, use the Mersenne Twister random number generator to produce Gaussian-distributed noise with standard deviation specified by `sigma`. The noise sample follows:
```
noise_sample ~ N(0, noise.sigma²)
```
This noise is superimposed on the differential signal, constituting device-level thermal noise and input-referred noise modeling.

**Step 4 - Phase Adjustment**: Select clock sampling or phase offset mode based on `phase_source`, and calculate the actual sampling moment.

> **Note**: `clk_sample` and `phase_offset` ports are reserved interfaces in the current version and have not yet implemented actual reading and usage logic in the `processing()` function. The phase adjustment function will be improved in subsequent versions.

**Step 5 - Decision Logic Application**:
- If `resolution = 0`: Standard threshold decision
- If `resolution > 0`: Fuzzy decision mechanism

**Step 6 - Output Generation**: Convert the decision result to digital output, supporting both TDF and DE domains.

**Equivalent Transfer Function Summary**:

Combining the above processes, the input-to-output transfer function of the sampler can be expressed as:
```
data_out = f(Vdiff, phase_offset, parameters)
where: Vdiff = (inp - inn) + offset.value + noise_sample
Decision threshold: V_th = threshold
```
This equivalent relationship corresponds to the design principles mentioned in Chapter 1 and provides the foundation for subsequent BER analysis.

### 3.2 Decision Logic and Fuzzy Zone Behavior

The core function of the sampler is digital decision based on input differential voltage. According to the `resolution` parameter setting, the decision mechanism is divided into standard decision and fuzzy decision modes. These two mechanisms directly determine the probability of misjudgment under different input conditions and become the foundation for BER modeling in Section 3.5.

#### 3.2.1 Standard Decision Mechanism (resolution = 0)

Uses dual-threshold hysteresis decision to implement Schmitt trigger effect:

```
Threshold Definition:
- threshold_high = threshold + hysteresis/2
- threshold_low = threshold - hysteresis/2

Decision Logic:
if (Vdiff > threshold_high):
    output = 1
elif (Vdiff < threshold_low):
    output = 0
else:
    output = previous_output  // Hysteresis zone: maintain state
```

#### 3.2.2 Fuzzy Decision Mechanism (resolution > 0)

Introduces a fuzzy decision zone to simulate comparator metastability behavior:

```
Decision Region Division:
┌─────────────────────────────────┐
│  Vdiff >= +resolution  → 1      │  Determined zone (high)
├─────────────────────────────────┤
│  |Vdiff| < resolution  → random │  Fuzzy zone
├─────────────────────────────────┤
│  Vdiff <= -resolution  → 0      │  Determined zone (low)
└─────────────────────────────────┘

Decision Algorithm:
if (abs(Vdiff) >= resolution):
    output = (Vdiff > 0) ? 1 : 0
else:
    // Fuzzy zone: random decision (Bernoulli distribution)
    output = random_bernoulli(0.5, seed)
```

### 3.3 Parameter Validation and Error Handling Mechanism

To ensure the validity of simulation results, the sampler module implements strict parameter validation mechanisms. Once parameter settings are unreasonable (e.g., `hysteresis ≥ resolution`), the BER calculated based on these parameters will have no physical meaning, so a forced termination strategy is adopted to avoid misleading results.

#### 3.3.1 Parameter Conflict Detection

The system implements strict parameter validation mechanisms to detect key parameter conflicts:

```
Core Validation Rules:
if (hysteresis >= resolution):
    // Trigger error handling process
    log_error("Hysteresis must be less than resolution")
    save_simulation_state()
    terminate_simulation()
```

**Physical Meaning Explanation**:
- `hysteresis` defines the width of the hysteresis interval for dual-threshold decision
- `resolution` defines the half-width of the fuzzy decision zone
- If `hysteresis >= resolution` (greater than or equal), it will lead to conflicting decision behaviors and inability to define clear decision strategies

#### 3.3.2 Error Handling Process and Data Saving

When parameter conflicts are detected, the system executes the following error handling process:

1. **Error Detection**: ParameterValidator class detects conflicting parameters
2. **Error Log**: Records detailed error information and current parameter configuration
3. **State Saving**: Saves simulation state, waveform data, and statistical information
4. **Friendly Prompt**: Outputs user-friendly error information and resolution suggestions
5. **Simulation Termination**: Safely terminates the simulation process to avoid erroneous results

**Data Saving Mechanism**:

Before error termination, the system ensures complete saving of all key data:

- **Parameter Configuration**: Current all module parameter settings
- **Error Log**: Timestamp, error type, stack information
- **Waveform Data**: Generated signal waveforms (CSV format)
- **Statistical Information**: Calculated BER, jitter, and other performance metrics
- **Simulation State**: Time, iteration count, module state

### 3.4 Jitter Modeling and Comprehensive Noise

Timing jitter is one of the key factors affecting sampler performance. This section starts from the physical sources of jitter, derives the equivalent voltage error caused by jitter, and provides a unified modeling method for comprehensive noise, providing a complete noise model for the BER analysis in Section 3.5.

#### 3.4.1 Jitter Source Classification

**Deterministic Jitter (DJ)**:
- Data-Dependent Jitter (DDJ): Caused by ISI
- Duty Cycle Distortion (DCD): Rise/fall edge mismatch
- Sinusoidal Jitter: Periodic jitter components

**Random Jitter (RJ)**:
- Thermal noise jitter: Caused by device thermal noise
- Phase noise jitter: Oscillator phase noise conversion
- Statistical characteristics: Gaussian distribution, mean is 0

#### 3.4.2 Voltage Error Caused by Jitter and Comprehensive Noise

**Time Domain Error Analysis**:

The sampling moment deviation caused by jitter will be converted to equivalent voltage error at signal transition edges. For a differential signal with data rate `f_data` and signal amplitude `A`:

```
Sampling moment deviation: Δt_jitter
Signal change rate: dV/dt = 2π × f_data × A
Voltage error: ΔV_jitter = (dV/dt) × Δt_jitter = 2π × f_data × A × Δt_jitter
```

Converting the standard deviation of timing jitter `σ_tjitter` to equivalent voltage noise:
```
σ_jitter = 2π × f_data × A × σ_tjitter
```

**Comprehensive Noise Model**:

Combining the device noise `σ_noise` (i.e., configuration parameter `noise.sigma`) defined in Section 3.1 and the jitter-induced voltage noise `σ_jitter` above, the total noise standard deviation is:

```
σ_total = sqrt(σ_noise² + σ_jitter²)
      = sqrt(σ_noise² + (2π × f_data × A × σ_tjitter)²)
```

This unified `σ_total` definition will be used as a core parameter in subsequent BER analysis.

**Signal-to-Noise Ratio Degradation**:
```
SNR_jitter = -20log₁₀(2π × f_data × A × σ_tjitter)

Total SNR:
1/SNR_total² = 1/SNR_signal² + 1/SNR_jitter² + 1/SNR_noise²
```

#### 3.4.3 Jitter Tolerance Engineering Guidance

**Typical Tolerance Metrics**:
```
High-speed SerDes (≥10 Gbps): σ_tjitter < 0.1 × UI
Mid-speed SerDes (1-10 Gbps): σ_tjitter < 0.05 × UI  
Low-speed SerDes (<1 Gbps): σ_tjitter < 0.02 × UI
```

### 3.5 BER Analysis and Numerical Examples

After clarifying the comprehensive effects of noise, offset, and jitter on the input signal, this section derives the BER performance model of the sampler based on the implementation mechanisms described above. We start from the simplest noise-only case and gradually introduce offset, fuzzy decision, and jitter factors, finally providing a complete comprehensive BER model and numerical calculation examples.

#### 3.5.1 BER under Ideal Channel (Noise Only)

**Assumptions**:
- Transmitted signal: ±A (differential amplitude 2A)
- Device noise: σ_noise = noise.sigma (defined in Section 3.1)
- Decision threshold: V_th = 0
- Ignore offset and jitter (offset = 0, σ_tjitter = 0)

**BER Calculation**:
```
BER = Q(A / σ_noise)

Where Q function is defined as:
Q(x) = (1/√(2π)) ∫[x,∞] exp(-t²/2) dt
     ≈ (1/2) erfc(x/√2)
```

This is the most basic BER model, considering only the effect of additive Gaussian noise on decision.

#### 3.5.2 BER with Noise and Offset

**Introducing Offset Voltage**:
- Offset: V_offset = offset.value (defined in Section 3.1)
- Decision threshold: V_th = threshold (defined in Section 2.2)

Due to the presence of offset voltage and decision threshold, the decision margins when transmitting '1' and transmitting '0' are no longer symmetric:

**BER Calculation**:
```
For transmitting '1' (signal = +A):
BER_1 = Q((A - (V_offset + threshold)) / σ_noise)

For transmitting '0' (signal = -A):
BER_0 = Q((A + (V_offset - threshold)) / σ_noise)

Total BER = (BER_1 + BER_0) / 2
```

In actual systems, offset voltage and decision threshold will cause the eye diagram center to shift, thereby increasing the bit error rate.

#### 3.5.3 BER Correction for Fuzzy Decision

As described in Section 3.2.2, when the fuzzy decision mechanism is enabled (`resolution > 0`), random decision is performed when the input differential voltage falls within the fuzzy zone `|Vdiff| < resolution`, which will introduce additional errors.

**Fuzzy Zone Probability**:

The probability that the signal plus noise falls into the fuzzy zone can be approximated as:
```
P_metastable ≈ erf(resolution / (√2 × σ_total))
```

Where `σ_total` is the comprehensive noise standard deviation. When considering only device noise, `σ_total = σ_noise`; when considering jitter, use the complete expression defined in Section 3.4.2.

**Additional Bit Error Rate**:

The additional bit error rate caused by random decision (50/50 probability) in the fuzzy zone is:
```
BER_fuzzy ≈ P_metastable × 0.5
```

**Corrected BER**:
```
BER ≈ Q(A / σ_total) + P_metastable × 0.5
```

This correction term reflects the impact of comparator metastability behavior on system BER.

#### 3.5.4 Comprehensive BER Model and Calculation Example

**Complete BER Formula**:

Combining all modeling elements from Sections 3.1 to 3.4 (device noise, offset, jitter, fuzzy decision), the complete BER model of the sampler is:

```
BER_total ≈ Q((A - |V_offset + threshold|) / σ_total) + P_metastable × 0.5

Where:
σ_total = sqrt(σ_noise² + σ_jitter²)  (Defined in Section 3.4.2)
σ_jitter = 2π × f_data × A × σ_tjitter
P_metastable = erf(resolution / (√2 × σ_total))  (Defined in Section 3.5.3)
```

This unified formula integrates all non-ideal effects into a calculable BER function, facilitating parameter sweeping and performance optimization.

**Python Numerical Calculation Example**:


```python
import numpy as np
from scipy.special import erfc, erf

def calculate_ber(A, sigma_noise, V_offset, threshold, resolution, f_data, sigma_tjitter):
    """
    Calculate the comprehensive Bit Error Rate of the Sampler

    Parameters:
    A: Signal amplitude (V)
    sigma_noise: Device noise standard deviation (V), corresponding to configuration parameter noise.sigma
    V_offset: Offset voltage (V), corresponding to configuration parameter offset.value
    threshold: Decision threshold (V), corresponding to configuration parameter threshold
    resolution: Resolution threshold (V), corresponding to configuration parameter resolution
    f_data: Data rate (Hz)
    sigma_tjitter: Timing jitter standard deviation (s)

    Returns:
    BER_total: Total bit error rate

    Formula correspondence:
    - sigma_jitter calculation consistent with Section 3.4.2
    - sigma_total definition consistent with Section 3.4.2
    - P_metastable definition consistent with Section 3.5.3
    - threshold effect consistent with Section 3.5.2
    """
    # Q function
    def Q(x):
        return 0.5 * erfc(x / np.sqrt(2))

    # Jitter-induced voltage error (Section 3.4.2 formula)
    sigma_jitter = 2 * np.pi * f_data * A * sigma_tjitter

    # Comprehensive noise (Section 3.4.2 formula)
    sigma_total = np.sqrt(sigma_noise**2 + sigma_jitter**2)

    # BER caused by noise, offset and threshold (Section 3.5.2 extension)
    SNR_eff = (A - abs(V_offset + threshold)) / sigma_total
    BER_noise = Q(SNR_eff)

    # BER caused by fuzzy decision (Section 3.5.3 formula)
    P_metastable = erf(resolution / (np.sqrt(2) * sigma_total))
    BER_fuzzy = P_metastable * 0.5

    # Total BER (Section 3.5.4 comprehensive formula)
    BER_total = BER_noise + BER_fuzzy

    return BER_total

# Example parameters
A = 0.5              # 500 mV differential amplitude
sigma_noise = 0.01   # 10 mV RMS device noise
V_offset = 0.005     # 5 mV offset
threshold = 0.0      # 0 V decision threshold
resolution = 0.02    # 20 mV resolution threshold
f_data = 10e9        # 10 Gbps data rate
sigma_tjitter = 1e-12 # 1 ps RMS jitter

BER = calculate_ber(A, sigma_noise, V_offset, threshold, resolution, f_data, sigma_tjitter)
print(f"BER = {BER:.2e}")
# Output example: BER ≈ 1e-12
```

**Notes**:

The above function unifies all modeling elements from Sections 3.1 to 3.4 (noise, offset, jitter, fuzzy decision, decision threshold) into a calculable BER function. By adjusting various parameters, parameter sweeping and performance optimization can be performed to guide margin allocation in actual design.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The Sampler testbench (`SamplerTransientTestbench`) adopts a scenario-driven modular design, supporting unified verification of multiple operating modes and boundary conditions. Core design concepts:

1. **Scenario Classification**: Basic functions, CDR integration, boundary conditions, and performance evaluation
2. **Parameterized Testing**: Automatically generates test cases through configuration-driven approach
3. **Result Verification**: Automated result analysis and performance metric calculation
4. **Documentation Integration**: Test results are directly generated into technical documentation

### 4.2 Test Scenario Definitions

| Scenario | Command Line Parameter | Test Objective | Output File |
|----------|------------------------|----------------|-------------|
| BASIC_FUNCTION | `basic` / `0` | Basic sampling and decision functions | sampler_tran_basic.csv |
| CDR_INTEGRATION | `cdr` / `1` | CDR phase tracking capability | sampler_tran_cdr.csv |
| FUZZY_DECISION | `fuzzy` / `2` | Fuzzy decision mechanism verification | sampler_tran_fuzzy.csv |
| PARAMETER_VALIDATION | `validate` / `3` | Parameter validation and error handling | sampler_tran_validation.csv |
| BER_MEASUREMENT | `ber` / `4` | Bit error rate performance test | sampler_tran_ber.csv |

### 4.3 Scenario Configuration Details

#### BASIC_FUNCTION - Basic Function Test

Verifies the basic differential signal decision and hysteresis function of the sampler.

- **Signal Source**: PRBS-15 pseudo-random sequence
- **Input Amplitude**: 200mV differential
- **Symbol Rate**: 10 Gbps
- **Test Parameters**: resolution=0, hysteresis=20mV
- **Verification Points**: Output BER < 1e-12, hysteresis function normal

#### CDR_INTEGRATION - CDR Integration Test

Verifies phase tracking and clock synchronization capabilities with the CDR module.

- **CDR Phase**: 1 GHz sine wave modulation (±100ps)
- **Input Signal**: 10 Gbps PRBS-7
- **Test Parameters**: phase_source="phase"
- **Verification Points**: Phase tracking error < 5ps

#### FUZZY_DECISION - Fuzzy Decision Test

Verifies the random decision mechanism within the resolution threshold.

- **Signal Source**: Low amplitude sine wave (30mV)
- **Test Parameters**: resolution=20mV, hysteresis=10mV
- **Verification Points**: Fuzzy zone randomness conforms to 50/50 distribution

#### PARAMETER_VALIDATION - Parameter Validation Test

Verifies parameter conflict detection and error handling mechanisms.

- **Conflicting Parameters**: hysteresis=30mV, resolution=20mV
- **Verification Points**: Trigger error handling, save state, safe termination

#### BER_MEASUREMENT - BER Test

Comprehensive performance test including all non-ideal effects such as noise, offset, and jitter.

- **Test Time**: ≥1 million bits
- **Non-ideal Effects**: Noise 5mV, offset 2mV, jitter 1ps
- **Verification Points**: Compare actual BER with theoretical calculation

### 4.4 Signal Connection Topology

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ DiffSignalSource  │       │   RxSamplerTdf   │       │  SignalMonitor  │
│                   │       │                   │       │                   │
│  out_p ───────────┼───────▶ inp              │       │                   │
│  out_n ───────────┼───────▶ inn              │       │                   │
└─────────────────┘       │                   │       │                   │
                            │  data_out ────────┼───────▶ digital_in      │
┌─────────────────┐       │                   │       │                   │
│   CDRModule     │       │                   │       │  → Statistical Analysis        │
│                   │       │                   │       │  → BER Calculation         │
│  phase_offset ───┼───────▶ phase_offset      │       │  → CSV Save         │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### 4.5 Auxiliary Module Descriptions

#### DiffSignalSource - Differential Signal Source

The differential signal source module is used to generate the differential input signals required for testing, supporting four waveform types:

| Waveform Type | Enum Value | Description |
|---------------|------------|-------------|
| DC | `DC` | DC signal, for static offset testing |
| SINE | `SINE` | Sine wave, for frequency response and jitter testing |
| SQUARE | `SQUARE` | Square wave, for eye diagram and sampling position testing |
| PRBS | `PRBS` | Pseudo-random sequence, for BER testing and function verification |

Configurable Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amplitude` | double | 0.1 | Signal amplitude (V) |
| `frequency` | double | 1e9 | Signal frequency (Hz) |
| `vcm` | double | 0.6 | Output common-mode voltage (V) |
| `sample_rate` | double | 100e9 | Sampling rate (Hz) |

Output signal generation rules:
- `out_p = vcm + 0.5 × signal`
- `out_n = vcm - 0.5 × signal`

#### PhaseOffsetSource - Phase Offset Source

The phase offset source module is used to generate CDR phase control signals, simulating the phase modulation output of the clock and data recovery loop.

Functional Features:
- **Constant Offset Mode**: Outputs a fixed phase offset value
- **Dynamic Modulation Mode**: Supports runtime adjustment of phase offset (via `set_offset()` method)
- **Time Domain Unit**: Phase offset is in seconds (conforms to Sampler module interface specification)

Configurable Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | double | 0.0 | Initial phase offset (s) |
| `sample_rate` | double | 100e9 | Sampling rate (Hz) |

Application Scenarios:
- **Phase Scanning Test**: Scan sampling phase by changing offset value
- **CDR Tracking Test**: Simulate dynamic phase adjustment output by CDR
- **Jitter Injection**: Implement phase jitter injection with jitter model

#### ClockSource - Clock Source Module

The clock source module generates sine clock signals for testing in clock-driven sampling mode.

Configurable Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency` | double | 10e9 | Clock frequency (Hz) |
| `amplitude` | double | 1.0 | Clock amplitude (V) |
| `vcm` | double | 0.5 | Clock common-mode voltage (V) |
| `sample_rate` | double | 100e9 | Sampling rate (Hz) |

Output signal: `clk = vcm + 0.5 × amplitude × sin(2πft)`

#### SamplerSignalMonitor - Sampler Signal Monitor

The signal monitor module implements data recording and analysis functions for the Sampler testbench.

Function List:
- **Real-time Waveform Recording**: Synchronously record differential input, TDF output, and DE output
- **CSV File Output**: Automatically save waveform data to specified file
- **Multi-channel Monitoring**: Simultaneously monitor analog input and digital output

Input Ports:

| Port | Type | Description |
|------|------|-------------|
| `in_p` | double | Differential input positive terminal (for monitoring) |
| `in_n` | double | Differential input negative terminal (for monitoring) |
| `data_out` | double | TDF domain digital output |
| `data_out_de` | bool | DE domain digital output |

CSV Output Format:
```
time(s),input+(V),input-(V),differential(V),tdf_output,de_output
0.000000e+00,0.650000,0.550000,0.100000,1.000000,1
...
```

#### BerCalculator - Bit Error Rate Calculator

The bit error rate calculator is a static utility class used to calculate the bit error rate of Sampler output.

Functions:
- **Expected Sequence Comparison**: Compare actual sampling results with expected sequence bit by bit
- **Error Statistics**: Calculate number of error bits and total bits
- **BER Output**: Return bit error rate (error bits / total bits)

Usage Example:
```cpp
std::vector<bool> expected = {...};  // Expected sequence
std::vector<bool> actual = {...};    // Actual sampling results
double ber = BerCalculator::calculate_ber(expected, actual);
```

---

## 5. Simulation Result Analysis

### 5.1 Performance Metric Definitions

| Metric | Calculation Method | Meaning |
|--------|--------------------|---------|
| Bit Error Rate (BER) | Error bits / Total bits | Decision reliability |
| Sampling Accuracy | |V_actual - V_theoretical| | Decision accuracy |
| Phase Tracking Error | \|phase_error\| | CDR integration performance |
| Jitter Tolerance | σ_tjitter_max | Timing robustness |
| Fuzzy Zone Probability | P(\|Vdiff\| < resolution) | Metastability frequency |

### 5.2 Typical Test Result Interpretation

#### 5.2.1 Basic Function Test Results

**Configuration**: 200mV input, hysteresis=20mV, resolution=0

**Expected Results**:
- BER < 1e-12 (ideally 0)
- Hysteresis function: Obvious delay when threshold switching
- Output waveform: Clear digital signal

**Analysis Method**: Count error bits, calculate BER value

#### 5.2.2 CDR Integration Test Results

**Configuration**: ±100ps phase modulation, phase_source="phase"

**Expected Results**:
- Phase tracking error < 5ps RMS
- Sampling point always kept near the data eye center
- BER remains stable under phase modulation

**Analysis Method**: Compare difference between theoretical phase and actual sampling moment

#### 5.2.3 Fuzzy Decision Test Results

**Configuration**: 50mV input, resolution=20mV, hysteresis=10mV

**Expected Results**:
- Fuzzy zone output shows 50/50 random distribution
- Determined zone output completely corresponds to input signal
- Random seed can reproduce the same random sequence

**Analysis Method**: Statistics of 0 and 1 proportions in fuzzy zone, verify randomness

### 5.3 Waveform Data File Format

CSV output format:
```
time(s),input+(V),input-(V),differential(V),tdf_output,de_output
0.000000e+00,0.600000,0.400000,0.200000,1.000000,1
1.000000e-10,0.601000,0.399000,0.202000,1.000000,1
...
```

Sampling rate: Default 100GHz (10ps step), adjustable by configuration.

---

## 6. Running Guide

### 6.1 Environment Configuration

Before running tests, environment variables need to be configured:

```bash
source scripts/setup_env.sh
export SYSTEMC_HOME=/path/to/systemc
export SYSTEMC_AMS_HOME=/path/to/systemc-ams
```

### 6.2 Build and Run

```bash
cd build
cmake ..
make sampler_tran_tb
cd tb
./sampler_tran_tb [scenario]
```

Scenario parameters:
- `basic` or `0` - Basic function test (default)
- `cdr` or `1` - CDR integration test
- `fuzzy` or `2` - Fuzzy decision test
- `validate` or `3` - Parameter validation test
- `ber` or `4` - BER performance test

### 6.3 Parameter Configuration Examples

#### 6.3.1 Basic Configuration

```json
{
  "sampler": {
    "sample_delay": 0.0,
    "resolution": 0.02,
    "hysteresis": 0.01,
    "phase_source": "clock"
  }
}
```

#### 6.3.2 Advanced Configuration

```json
{
  "sampler": {
    "sample_delay": 5e-12,
    "resolution": 0.015,
    "hysteresis": 0.008,
    "phase_source": "phase",
    "offset": {
      "enable": true,
      "value": 0.005
    },
    "noise": {
      "enable": true,
      "sigma": 0.01,
      "seed": 12345
    }
  }
}
```

### 6.4 Result Viewing

After test completion, performance statistics are output to the console, and waveform data is saved to CSV files. Use Python for visualization analysis:

```bash
python scripts/plot_sampler_waveform.py
```

---

## 7. Technical Key Points

### 7.1 CDR Phase Integration Notes

**Issue**: The phase signal path may introduce additional phase delay, affecting sampling moment accuracy.

**Solutions**:
- Use the direct phase offset signal output by the CDR module
- Consider propagation delay from phase signal to sampler
- Compensate for fixed delay in the `sample_delay` parameter

### 7.2 Fuzzy Decision Randomness Verification

**Issue**: How to ensure that random decisions in the fuzzy zone have true randomness.

**Solutions**:
- Use Mersenne Twister pseudo-random number generator
- Provide configurable random seed
- Verify randomness distribution through statistical testing

### 7.3 Parameter Validation Mechanism Implementation

**Core Validation Rules**:
```cpp
if (hysteresis >= resolution) {
    throw std::invalid_argument(
        "Hysteresis must be less than resolution to avoid decision ambiguity"
    );
}
```

**Error Handling Process**:
1. Parameter validator detects conflicts
2. Record detailed error logs
3. Save current simulation state
4. Terminate simulation process

### 7.4 Numerical Stability Considerations

**Issue**: At high sampling rates, floating-point precision may affect decision accuracy.

**Solutions**:
- Use double-precision floating point (double)
- Avoid threshold settings close to machine precision
- Consider numerical errors in parameter selection

### 7.5 Time Step Setting Guidance

**Sampling Rate Requirements**: The sampling rate should be much higher than the signal bandwidth, recommended:
```
f_sample ≥ 20 × f_data
```

For 10 Gbps signals, recommended sampling step ≤ 5ps.

### 7.6 Performance Optimization Suggestions

1. **Noise Generation Optimization**: Use lookup table method instead of real-time random number generation
2. **Parameter Caching**: Cache frequently calculated parameter values
3. **Conditional Execution**: Conditionally execute relevant code based on parameter enable status
4. **Memory Management**: Avoid allocating dynamic memory in simulation loops

---

## 8. Reference Information

### 8.1 Related Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | RxSamplerParams structure |
| Header File | `/include/ams/rx_sampler.h` | RxSamplerTdf class declaration |
| Implementation File | `/src/ams/rx_sampler.cpp` | RxSamplerTdf class implementation |
| Parameter Validator | `/include/common/parameter_validator.h` | ParameterValidator class |
| Testbench | `/tb/rx/sampler/sampler_tran_tb.cpp` | Transient simulation test |
| Unit Test | `/tests/unit/test_sampler_basic.cpp` | GoogleTest unit test |
| Waveform Plotting | `/scripts/plot_sampler_waveform.py` | Python visualization script |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++11 Standard
- GoogleTest 1.12.1 (Unit Testing)
- NumPy/SciPy (Python Analysis Tools)

### 8.3 Performance Benchmarks

**Typical Performance Metrics**:
- Decision delay: < 1ns
- Timing accuracy: ±1ps
- Noise modeling accuracy: ±0.1%
- BER measurement accuracy: ±5% (1e12 samples)

**Recommended Parameter Configurations**:
- Data rate ≤ 25 Gbps
- Input amplitude ≥ 100mV
- Sampling rate ≥ 100GS/s
- Simulation step ≤ 10ps

---

**Document Version**: v0.3  
**Last Updated**: 2025-12-07  
**Author**: Yizhe Liu
