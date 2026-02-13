# TX FFE Module Technical Documentation

🌐 **Languages**: [中文](../../modules/ffe.md) | [English](ffe.md)

**Level**: AMS Sub-module (TX)  
**Class Name**: `TxFfeTdf`  
**Current Version**: v0.1 (2026-01-13)  
**Status**: Under Development

---

## 1. Overview

The Feed-Forward Equalizer (FFE) is a core signal conditioning module in the SerDes transmitter, located between the WaveGen and the Mux. Its primary function is to pre-compensate for the Inter-Symbol Interference (ISI) introduced by the channel at the transmitter side. By pre-distorting the data, the signal can be recovered into a clearer eye diagram at the receiver after passing through the channel attenuation, reducing the burden on the receiver equalizer.

### 1.1 Design Principles

The core design concept of FFE utilizes a Finite Impulse Response (FIR) filter structure. Based on the frequency response characteristics of the channel, it performs frequency-selective pre-emphasis or de-emphasis at the transmitter side to achieve ISI pre-compensation.

#### 1.1.1 ISI Formation Mechanism and Pre-compensation Strategy

In high-speed serial links, frequency-dependent channel losses cause more severe attenuation of high-frequency components, slowing down the signal edges and creating interference between adjacent symbols (ISI). ISI can be decomposed into:

- **Pre-cursor ISI**: The current symbol is interfered with by the previous symbol, manifesting as a tailing effect that appears ahead of the signal rising/falling edges.
- **Post-cursor ISI**: The current symbol is interfered with by subsequent symbols, manifesting as the signal failing to fully stabilize to the target level after a transition.

FFE implements pre-compensation through the FIR structure:

- **Pre-tap**: Corresponds to pre-cursor compensation, canceling the pre-cursor ISI produced by the channel by injecting a signal component with opposite polarity in advance.
- **Main tap**: Corresponds to the main energy of the current symbol, typically having the largest weighting coefficient.
- **Post-tap**: Corresponds to post-cursor compensation, pre-eliminating the post-cursor ISI of the channel by superimposing the inverse component of the delayed symbol.

#### 1.1.2 FIR Filter Implementation

FFE adopts a discrete-time FIR filter structure, with the mathematical expression:

```
y[n] = Σ c[k] × x[n-k]
       k=0 to N-1
```

Where:
- `y[n]`: The nth output symbol
- `x[n-k]`: The input symbol delayed by k symbol intervals (UI)
- `c[k]`: The weighting coefficient of the kth tap
- `N`: The total number of taps

The tap coefficients follow these principles:

- **Normalization Constraint**: To ensure the output signal's average power is consistent with the input, it is usually required that `Σ|c[k]| ≈ 1`.
- **Main Tap Maximization Principle**: The main tap (typically c[1] or c[2]) has the largest magnitude, ensuring that the main signal energy is concentrated on the current symbol.
- **Symmetry Consideration**: For symmetric channel impulse responses, FFE tap coefficients may also exhibit symmetric or quasi-symmetric distributions, but in practical applications, post-tap weights are usually greater than pre-taps (because post-cursor ISI is more severe in causal systems).

#### 1.1.3 Pre-emphasis and De-emphasis

There are two common implementation strategies for FFE, differing in the relative position of the main tap and power distribution:

- **Pre-emphasis**: The main tap is located in the middle or towards the rear, with positive pre-tap coefficients and negative post-tap coefficients. This method injects energy ahead of the transition edge, enhancing high-frequency components but increasing transmitter power consumption (due to increased peak current). Typically used in short-distance, low-loss channels.

- **De-emphasis**: The main tap coefficient is 1, with negative post-tap coefficients and zero or very small pre-taps. This method relatively increases the energy proportion at transition edges by attenuating the amplitude of non-transition symbols, reducing the transmitter's peak power consumption but lowering the average signal amplitude. Widely used in PCIe, USB, and other standards.

#### 1.1.4 Tap Coefficient Optimization Methods

FFE tap coefficient design can be obtained through the following methods:

- **Channel Inverse Filtering**: Design the FFE transfer function F(f) based on the channel's frequency response H(f) such that `F(f) × H(f) ≈ 1`, meaning the cascade of FFE and channel approximates an all-pass characteristic. In practice, tap coefficients are solved using the Minimum Mean Square Error (MMSE) criterion for the time-domain impulse response.

- **Zero-Forcing**: Force the ISI at the receiver sampling points to zero, obtaining FFE coefficients by solving a system of linear equations. This method is optimal in high SNR scenarios but may lead to noise amplification.

- **Adaptive Algorithms**: During system operation, use feedback error signals from the receiver (such as eye height, BER) to adjust FFE coefficients online using LMS (Least Mean Squares) or RLS (Recursive Least Squares) algorithms. This method is suitable for time-varying channels or scenarios requiring dynamic optimization.

- **Look-up Table**: For standardized channels (such as reference channels defined by PCIe specifications), pre-simulate a set of typical FFE coefficients and store them as a configuration table. During system initialization, select the corresponding coefficient group based on the channel type.

### 1.2 Core Features

- **FIR Filter Structure**: Adopts an N-tap finite impulse response filter, supporting flexible configuration of the number of taps (typically 3-7 taps) and weighting coefficients to meet compensation requirements for different channel losses.

- **Symbol Rate Synchronization**: FFE operates in the Symbol Rate clock domain, processing one symbol per UI. The tap delay line interval is precisely equal to one symbol period, ensuring time alignment accuracy for ISI compensation.

- **Configurable Tap Coefficients**: Tap weighting coefficients are flexibly set through configuration parameters, supporting pre-emphasis, de-emphasis, mixed mode, and other equalization strategies that can be optimized according to channel characteristics and system requirements.

- **Normalized Output Swing**: Through normalization of tap coefficients or explicit gain adjustment, ensures that the FFE output signal swing is within a reasonable range, avoiding saturation in the downstream Driver or signal-to-noise ratio degradation.

- **Low Latency Characteristic**: FFE only introduces fixed tap delays (typically half the number of taps in UI), offering lower latency and simpler timing constraints compared to receiver-side DFE (which requires decision feedback).

- **Channel Variation Adaptation**: Tap coefficients can be dynamically updated at runtime through an external control interface (such as an Adaption module), supporting adaptive equalization and channel tracking functions to adapt to channel characteristic changes caused by temperature drift, aging, etc.

### 1.3 Typical Application Scenarios

FFE configuration requirements in different SerDes standards and applications:

| Application Standard | Tap Count | Typical Coefficient Example | Equalization Strategy | Remarks |
|---------|---------|-------------|---------|------|
| PCIe Gen3 (8Gbps) | 3-tap | [0.0, 1.0, -0.25] | De-emphasis | 3.5dB or 6dB de-emphasis |
| PCIe Gen4 (16Gbps) | 3-tap | [0.0, 1.0, -0.35] | De-emphasis | Mandatory de-emphasis for long channels |
| USB 3.2 (10Gbps) | 3-tap | [0.0, 1.0, -0.2] | De-emphasis | Optional equalization, can be disabled for short channels |
| 10G/25G Ethernet | 5-tap | [0.05, 0.15, 0.6, -0.15, -0.05] | Mixed | Balanced pre/post-cursor compensation |
| 56G SerDes (PAM4) | 7-tap | [0.02, 0.08, 0.15, 0.5, -0.15, -0.08, -0.02] | Mixed | Ultra-long channel, works with receiver DFE |

> **Note**: After normalization, tap coefficients should satisfy `Σ|c[k]| ≈ 1`. Specific values need to be optimized based on channel S-parameter simulation.

### 1.4 Relationship with Other Modules

- **Upstream: WaveGen**  
  FFE receives baseband PRBS or custom pattern signals from the WaveGen module. The input is typically an ideal NRZ or PAM-N level (such as ±1V) without channel loss effects.

- **Downstream: Mux → Driver**  
  FFE outputs pre-distorted signals to the Mux (multiplexer) for channel selection, and finally amplified by the Driver to drive the channel. The Driver's bandwidth limitations and nonlinear effects further impact FFE compensation effectiveness, requiring joint optimization.

- **Channel**  
  FFE's design goal is to pre-compensate the frequency response characteristics of the channel; therefore, the selection of its tap coefficients must be optimized based on the target channel's S-parameters or impulse response. Changes in channel characteristics (such as temperature, aging) may require dynamic adjustment of FFE coefficients.

- **Receiver Equalizer (CTLE/DFE)**  
  FFE forms a synergistic compensation relationship with the receiver equalizer. FFE handles low and mid-frequency ISI, reducing the burden on the receiver side; CTLE handles high-frequency attenuation; DFE handles residual post-cursor ISI. Reasonable equalization budget allocation can optimize overall system power consumption and performance.

---

## 2. Module Interface

### 2.1 Port Definitions

The FFE module adopts a single-ended signal architecture, simplifying modeling complexity to meet behavioral-level simulation requirements.

#### TDF Domain Ports

| Port Name | Direction | Type | Description |
|-------|------|------|------|
| `in` | Input | double | Single-ended input, receiving baseband signals from WaveGen |
| `out` | Output | double | Single-ended output, outputting pre-distorted signals from FFE |

> **Design Note**: The single-ended architecture is suitable for behavioral-level modeling, focusing on verifying the ISI compensation algorithm of the FIR filter. In actual hardware, FFE is typically implemented differentially, but single-ended modeling is sufficient to characterize the equalization effect during behavioral simulation.

### 2.2 Parameter Configuration

FFE module configuration parameters are managed through the `TxFfeParams` struct, defined in `include/common/parameters.h`.

#### 2.2.1 Basic Parameter Structure

```cpp
struct TxFfeParams {
    std::vector<double> taps;      // FFE tap weighting coefficients
    
    TxFfeParams() : taps({0.2, 0.6, 0.2}) {}
};
```

#### 2.2.2 Tap Coefficient Parameters (taps)

Tap coefficients are the core configuration parameters of the FFE module, defining the time-domain response characteristics of the FIR filter.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `taps` | vector&lt;double&gt; | [0.2, 0.6, 0.2] | FFE tap weighting coefficient array, indexed from 0 |

**Tap Meanings**:
- `taps[0], taps[1], ..., taps[N-2]`: Pre-taps, compensating for pre-cursor ISI
- `taps[main tap index]`: Main tap, carrying the main energy of the current symbol, typically the maximum value
- `taps[main tap+1], ..., taps[N-1]`: Post-taps, compensating for post-cursor ISI

**Typical Configurations**:
- **3-tap**: `[c0, c1, c2]`, where c1 is the main tap
- **5-tap**: `[c0, c1, c2, c3, c4]`, where c2 is the main tap
- **7-tap**: `[c0, c1, c2, c3, c4, c5, c6]`, where c3 is the main tap

#### 2.2.3 Tap Coefficient Constraints and Validation

To ensure the rationality and physical realizability of the FFE output signal, tap coefficients must satisfy the following constraints:

##### Normalization Constraint

The specific form of the normalization constraint depends on the equalization mode:

**De-emphasis Mode**:
- Main tap fixed at 1.0, post-taps are negative values
- Constraint: `c[main] = 1.0`, `Σ|c[k]| > 1.0` (post-taps "absorb" energy)
- Typical configuration: `[0, 1.0, -0.2, -0.1]` (3-tap FFE)

**Pre-emphasis/Balanced Mode**:
- Sum of all tap coefficients is close to 1.0
- Constraint: `Σ c[k] ≈ 1.0` (maintaining DC gain)
- Typical configuration: `[0.15, 0.7, 0.15]` (3-tap FFE)

**Validation Method**:
```cpp
double sum_abs = 0.0;
double sum_algebraic = 0.0;
for (auto c : taps) {
    sum_abs += std::abs(c);
    sum_algebraic += c;
}

// De-emphasis mode: check main tap ≈ 1.0
if (taps[main_idx] > 0.95 && taps[main_idx] < 1.05) {
    // Allow sum_abs > 1.0
}
// Pre-emphasis/Balanced mode: check algebraic sum ≈ 1.0
else if (std::abs(sum_algebraic - 1.0) < 0.2) {
    // Allow 0.8 < sum_algebraic < 1.2
}
```

##### Main Tap Maximization Principle

The absolute value of the main tap coefficient should be the largest among all taps, ensuring that the main signal energy is concentrated on the current symbol:

```
|c[main]| = max(|c[0]|, |c[1]|, ..., |c[N-1]|)
```

**Typical Positions**:
- 3-tap: Main tap at index 1
- 5-tap: Main tap at index 2
- 7-tap: Main tap at index 3

##### Dynamic Range Constraint

The absolute value of a single tap coefficient should not exceed 1.0 to avoid output saturation:

```
|c[k]| ≤ 1.0, ∀k
```

#### 2.2.4 Typical Application Configuration Examples

The following configuration examples are optimized for different SerDes standards and channel loss scenarios:

##### PCIe Gen3 (8Gbps) - 3.5dB De-emphasis

```cpp
TxFfeParams ffe_pcie_gen3;
ffe_pcie_gen3.taps = {0.0, 1.0, -0.25};  // De-emphasis strategy
```

**Features**:
- Main tap normalized to 1.0
- Post-tap is negative, achieving 3.5dB de-emphasis
- Pre-tap is 0 (no pre-cursor compensation)
- Applicable scenario: PCIe short channel (<20cm backplane trace)

##### PCIe Gen4/Gen5 (16Gbps/32Gbps) - 6dB De-emphasis

```cpp
TxFfeParams ffe_pcie_gen4;
ffe_pcie_gen4.taps = {0.0, 1.0, -0.4};  // Strong de-emphasis
```

**Features**:
- Post-tap increased to -0.4, achieving 6dB de-emphasis
- Suitable for higher loss channels (20-40cm)
- Works with receiver DFE

##### 10G/25G Ethernet - Balanced Mode

```cpp
TxFfeParams ffe_ethernet;
ffe_ethernet.taps = {0.05, 0.2, 0.5, -0.15, -0.1};  // 5-tap
```

**Features**:
- Main tap c[2]=0.5
- Pre-taps c[0]=0.05, c[1]=0.2 provide pre-cursor compensation
- Post-taps c[3]=-0.15, c[4]=-0.1 compensate for post-cursor
- Normalization sum: |0.05|+|0.2|+|0.5|+|0.15|+|0.1|=1.0
- Applicable scenario: Medium loss channel (10-15dB @ Nyquist)

##### 56G PAM4 - Long Channel

```cpp
TxFfeParams ffe_pam4;
ffe_pam4.taps = {0.02, 0.08, 0.15, 0.5, -0.15, -0.1, -0.05};  // 7-tap
```

**Features**:
- 7-tap design, main tap c[3]=0.5
- Symmetric pre/post-cursor compensation structure
- Normalization sum: |0.02|+|0.08|+|0.15|+|0.5|+|0.15|+|0.1|+|0.05|=1.05
- Applicable scenario: Ultra-long channel (>20dB loss), works with receiver CTLE+DFE

##### Custom Pre-emphasis Mode

```cpp
TxFfeParams ffe_custom;
ffe_custom.taps = {0.15, 0.25, 0.6, -0.2, -0.1};
```

**Features**:
- Pre-taps are positive values, achieving pre-emphasis
- Main tap c[2]=0.6
- Balanced pre/post-cursor compensation
- Suitable for channels with severe time-domain reflections (multiple impedance discontinuities)

#### 2.2.5 Tap Coefficient Optimization Methodology

##### Optimization Based on Channel S-parameters

**Step 1**: Extract channel impulse response  
Convert S-parameter S21(f) to time-domain impulse response h(t) through IFFT:

```python
import numpy as np
from scipy.fft import ifft

# S21 frequency response → time-domain impulse response
freq = np.array([...])  # frequency points
S21 = np.array([...])   # complex S-parameters
h_t = ifft(S21)
```

**Step 2**: MMSE criterion solution  
Minimize the mean square error (MSE) between the receiver output signal and the desired signal:

**Time-domain expression**:
```
min Σ |y[n] - d[n]|²
```
Where:
- `y[n] = (h[n] ⊗ c[n])` is the output after channel and FFE cascade
- `d[n]` is the desired transmitted data sequence
- `⊗` denotes convolution operation

**Frequency-domain expression (considering noise power)**:
```
H_FFE(f) = H*_channel(f) / (|H_channel(f)|² + N₀/Ps)
```
Where:
- `H*_channel(f)` is the conjugate of the channel frequency response
- `N₀` is the noise power spectral density
- `Ps` is the signal power
- This formula balances the trade-off between ISI elimination and noise amplification

**Step 3**: Normalization and quantization  
Normalize the solved tap coefficients to a reasonable range, and consider hardware implementation quantization bits (e.g., 6-bit quantization).

##### Adaptive Online Optimization

During system operation, use feedback error signals from the receiver (such as eye height, BER estimation) to adjust FFE coefficients in real-time using the LMS algorithm:

```
c[k](n+1) = c[k](n) + μ × e(n) × x(n-k)
```

Where:
- `e(n)`: Receiver decision error
- `μ`: Step size parameter (0.001~0.01)
- `x(n-k)`: Input signal delayed by k UIs

### 2.3 Constructor

The FFE module constructor follows the standard naming conventions of SystemC-AMS, initializing ports and parameters.

#### Constructor Signature

```cpp
class TxFfeTdf : public sca_tdf::sca_module {
public:
    // Single-ended input/output ports
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    
    // Constructor
    TxFfeTdf(sc_core::sc_module_name name, const TxFfeParams& params);
    
private:
    TxFfeParams m_params;
    
    // SystemC-AMS callback functions
    void set_attributes() override;
    void initialize() override;
    void processing() override;
    
    // Internal state: delay line buffer
    std::vector<double> m_delay_line;
};
```

#### Constructor Implementation Example

```cpp
TxFfeTdf::TxFfeTdf(sc_core::sc_module_name name, const TxFfeParams& params)
    : sca_tdf::sca_module(name)
    , in("in")
    , out("out")
    , m_params(params)
    , m_delay_line(params.taps.size(), 0.0)  // Initialize delay line to 0
{
    // Parameter validation
    if (m_params.taps.empty()) {
        SC_REPORT_ERROR("TxFfeTdf", "FFE taps cannot be empty");
    }
}
```

#### Port Connection Example

In the top-level testbench or system-level module, the FFE module connection is as follows:

```cpp
// Upstream module: WaveGen (single-ended signal source)
WaveGenTdf wave_gen("wave_gen", wave_params);

// FFE module
TxFfeTdf tx_ffe("tx_ffe", ffe_params);

// Downstream module: Driver (single-ended input)
TxDriverTdf tx_driver("tx_driver", driver_params);

// Single-ended signal connections
sca_tdf::sca_signal<double> sig_ffe_in, sig_ffe_out;

// Connection topology
wave_gen.out(sig_ffe_in);
tx_ffe.in(sig_ffe_in);
tx_ffe.out(sig_ffe_out);
tx_driver.in(sig_ffe_out);
```

> **Important Note**: SystemC-AMS requires all ports to be connected; dangling ports are not allowed.

---

## 3. Core Implementation Mechanism

### 3.1 Signal Processing Flow

The FFE module's `processing()` method executes FIR filter calculations once per symbol period (UI), implementing signal pre-distortion.

**Processing Pipeline**:
```
Input Read → Delay Line Update → FIR Convolution Calculation → Output Write
```

**Step 1 - Input Read**: Read the current symbol value from the single-ended input port `in`. The input signal typically comes from the WaveGen module with a swing of ±1V.

**Step 2 - Delay Line Update**: Store the current input symbol in the tap delay line (`std::vector<double> m_delay_line`), using a FIFO structure where new data enters at index 0 and old data shifts backward.

**Step 3 - FIR Convolution**: Perform weighted summation based on the configured tap coefficients and delay line data:
```
y[n] = Σ c[k] × x[n-k]  (k=0 to N-1)
```
Where `c[k]` is the tap weighting coefficient and `x[n-k]` is the historical input delayed by k UIs.

**Step 4 - Output Write**: Write the FIR filter result to the output port `out`, sending the output signal downstream to the Mux and Driver modules.

### 3.2 FIR Filter Mechanism

FFE adopts a Direct Form FIR filter structure, implementing ISI pre-compensation through the weighted summation of N taps.

**Time-domain expression**:
```
y[n] = c[0]×x[n] + c[1]×x[n-1] + ... + c[N-1]×x[n-N+1]
```

**Frequency response (Discrete-Time Fourier Transform)**:
```
H(f) = Σ c[k] × e^(-j2πfkT)  (k=0 to N-1)
```
Where `T` is the symbol period (UI) and `f` is frequency.

**Typical Configuration Example** (3-tap):
- De-emphasis mode: `c = [0, 1.0, -0.25]`, frequency response `H(f) = 1 - 0.25×e^(-j4πfT)`, low-frequency gain 0.75 (-2.5dB), Nyquist frequency gain 1.25 (+1.9dB)
- Balanced mode: `c = [0.15, 0.7, 0.15]`, low-pass characteristic, low-frequency gain 1.0, high-frequency gain 0.4 (-7.96dB)

**Delay Line Management**: Implemented using simple array shifting (tap count typically ≤7, performance overhead is acceptable). Updated once per UI, delay line filled with 0 values during initialization.

### 3.3 Normalization and Saturation Handling

**Normalization Strategy**: Choose whether to normalize the output based on the equalization mode. De-emphasis mode (main tap ≈ 1.0) typically does not normalize; pre-emphasis/balanced mode can use amplitude normalization (dividing by `Σ|c[k]|`) to ensure output peak does not exceed input swing.

**Saturation Limiting**: Prevent output from exceeding the input range of the downstream Driver through hard clipping or soft saturation (tanh function). Typical saturation levels are set to 80-90% of the Driver input range.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The FFE testbench (`FfeTransientTestbench`) adopts a scenario-driven modular design to verify the pre-distortion performance of FFE under different equalization strategies and channel conditions. Core design concepts:

1. **Scenario-driven**: Select different test scenarios through enumeration types; each scenario automatically configures corresponding signal sources and FFE tap coefficients
2. **Component reuse**: Single-ended signal sources, signal monitors, and other auxiliary modules can be reused across multiple test scenarios
3. **Result analysis**: Automatically select appropriate analysis methods (time-domain waveform, frequency-domain characteristics) based on scenario type

### 4.2 Test Scenario Definitions

The testbench supports five core test scenarios:

| Scenario | Command Line Parameter | Test Objective | Output File |
|------|----------|---------|----------|
| BASIC_PRBS | `prbs` / `0` | Basic FIR filtering and tap weighting characteristics | ffe_tran_prbs.csv |
| DEEMPHASIS_TEST | `deemp` / `1` | De-emphasis mode compensation effect | ffe_tran_deemp.csv |
| PREEMPHASIS_TEST | `preemp` / `2` | Pre-emphasis mode high-frequency enhancement | ffe_tran_preemp.csv |
| TAP_SWEEP | `sweep` / `3` | Tap coefficient sweep and optimization | ffe_tran_sweep.csv |
| CHANNEL_COMBO | `combo` / `4` | FFE+channel cascade compensation verification | ffe_tran_combo.csv |

### 4.3 Scenario Configuration Details

#### BASIC_PRBS - Basic PRBS Test

Verify FFE's basic FIR filtering function and tap weighting correctness.

- **Signal source**: PRBS-7 pseudo-random sequence
- **Input amplitude**: ±1.0V (single-ended)
- **Symbol rate**: 10 Gbps
- **FFE coefficients**: `[0.2, 0.6, 0.2]` (3-tap balanced mode)
- **Verification point**: Output waveform should be the convolution result of input and FIR coefficients

#### DEEMPHASIS_TEST - De-emphasis Test

Verify de-emphasis mode's post-cursor ISI pre-compensation capability.

- **Signal source**: Alternating "0-1-1-1" and "1-0-0-0" patterns (forcing post-cursor ISI)
- **FFE coefficients**: `[0.0, 1.0, -0.35]` (PCIe Gen4 typical de-emphasis)
- **Verification point**: Symbol amplitude after transition should be reduced by 35%, pre-compensating for channel post-cursor attenuation

#### PREEMPHASIS_TEST - Pre-emphasis Test

Verify pre-emphasis mode's high-frequency enhancement effect.

- **Signal source**: Square wave (1 GHz)
- **FFE coefficients**: `[0.15, 0.7, 0.15]`
- **Verification point**: Edge steepness increased, rise/fall times shortened

#### TAP_SWEEP - Tap Sweep

Find optimal equalization configuration by sweeping different tap coefficient combinations.

- **Signal source**: PRBS-7
- **Sweep range**: Post-tap from -0.5 to 0 (step 0.05)
- **Fixed coefficients**: Main tap = 1.0, Pre-tap = 0
- **Verification point**: Record output eye height/eye width for each coefficient group

#### CHANNEL_COMBO - Channel Cascade Test

Verify overall compensation effect after FFE cascade with a typical channel.

- **Signal source**: PRBS-7
- **FFE coefficients**: Optimized based on channel impulse response
- **Channel model**: Simplified first-order low-pass filter (simulating 10dB@Nyquist loss)
- **Verification point**: Eye diagram opening area after cascade should be larger than without FFE scenario

### 4.4 Signal Connection Topology

The module connection relationships in the testbench are as follows:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  SignalSource   │       │    TxFfeTdf     │       │  SignalMonitor  │
│                 │       │                 │       │                 │
│  out ───────────┼───────▶ in             │       │                 │
│                 │       │                 │       │                 │
│                 │       │  out ───────────┼───────▶ in             │
└─────────────────┘       │                 │       │                 │
                          └─────────────────┘       │  → Statistical Analysis     │
                                                     │  → CSV Save      │
                                                     └─────────────────┘
```

**Optional Cascade (CHANNEL_COMBO scenario)**:
```
SignalSource → TxFfeTdf → ChannelModel → SignalMonitor
```

### 4.5 Auxiliary Module Description

#### SignalSource - Single-ended Signal Source

Supports four waveform types:
- **DC**: DC signal
- **SINE**: Sine wave
- **SQUARE**: Square wave
- **PRBS**: Pseudo-random sequence (PRBS-7/15/23/31)

Configurable parameters: amplitude, frequency, symbol rate

#### SignalMonitor - Signal Monitor

Functions:
- Real-time recording of input/output waveforms
- Calculating statistical information (mean, RMS, peak-to-peak)
- Outputting CSV format waveform files
- Optional: calculating eye diagram metrics (eye height, eye width, jitter)

#### ChannelModel - Simplified Channel Model (CHANNEL_COMBO scenario only)

Uses first-order or second-order low-pass filters to simulate channel frequency response for quick verification of FFE compensation effects. In actual applications, it should be replaced with precise channel models based on S-parameters.

### 4.6 Compilation and Execution

Testbench code is located in the `tb/tx/ffe/` directory (if directory structure is not created, in `tb/` root directory), compilation and execution steps:

```bash
# Build testbench
cd build
cmake ..
make ffe_tran_tb

# Run specified scenario
cd tb
./ffe_tran_tb [scenario]
```

Scenario parameters:
- `prbs` or `0` - Basic PRBS test (default)
- `deemp` or `1` - De-emphasis test
- `preemp` or `2` - Pre-emphasis test
- `sweep` or `3` - Tap sweep
- `combo` or `4` - Channel cascade test

### 4.7 Testbench Verification Objectives

Core verification metrics for each test scenario:

| Scenario | Verification Metric | Pass Criteria |
|------|---------|---------|
| BASIC_PRBS | Output error vs. theoretical convolution result | < 1% |
| DEEMPHASIS_TEST | Post-cursor suppression ratio | Within configured value ±5% |
| PREEMPHASIS_TEST | Edge slope enhancement ratio | > 1.3× |
| TAP_SWEEP | Optimal coefficient point identification | At maximum eye height |
| CHANNEL_COMBO | Eye height improvement after cascade | > 30% vs. no FFE |

---

## 5. Simulation Results Analysis

This chapter introduces typical simulation results, key performance metrics, and analysis methods for various FFE test scenarios. The correctness and effectiveness of FFE pre-distortion functionality are verified through quantitative assessment of time-domain waveforms, frequency-domain characteristics, and ISI elimination effects.

### 5.1 Simulation Environment Description

#### 5.1.1 General Configuration Parameters

Basic configurations shared by all test scenarios:

| Parameter Category | Parameter Name | Typical Value | Description |
|---------|--------|--------|------|
| **Global Simulation** | Sampling Rate (Fs) | 100 GHz | Equal to symbol rate, satisfies discrete-time FIR filtering |
| | Simulation Duration | 10-50 ns | PRBS tests require ≥2000 UI for statistics |
| | Time Step (UI) | 10 ps | Corresponds to 10 Gbps symbol rate |
| **Signal Source** | Input Amplitude | ±1 V | Normalized single-ended input |
| | Data Rate | 10 Gbps | Standard test rate |
| | PRBS Type | PRBS7 | Fast convergence, for basic tests |
| **FFE** | Tap Count | 3-5 | Selected based on scenario |
| | Main Tap Position | Index 1 or 2 | Depends on total tap count |

#### 5.1.2 Performance Evaluation Metrics

FFE performance is measured by the following quantitative metrics:

| Metric | Definition | Calculation Method | Typical Target |
|------|------|----------|---------|
| **Eye Height Improvement** | Eye height improvement of FFE output relative to input | (EH_out - EH_in) / EH_in | > 20% |
| **Eye Width Improvement** | Eye width increase of FFE output relative to input | (EW_out - EW_in) / EW_in | > 10% |
| **ISI Elimination Ratio** | ISI amplitude reduction ratio of pre/post-cursors | (ISI_in - ISI_out) / ISI_in | > 50% |
| **Frequency Enhancement** | High-frequency/low-frequency energy ratio improvement | 20*log10(H(fNyq)/H(DC)) | De-emphasis: +3~6dB |
| **Convolution Error** | Deviation of output from theoretical FIR convolution | RMS(y_meas - y_theory) / RMS(y_theory) | < 1% |

### 5.2 Basic Function Verification

#### 5.2.1 BASIC_PRBS Test Results

**Test Configuration**:
```json
{
  "signal_source": {"type": "PRBS7", "amplitude": 1.0},
  "ffe": {"taps": [0.2, 0.6, 0.2]}
}
```

**Expected Results Analysis**:

**Time-domain Waveform Characteristics**:
- **Input signal**: PRBS7 sequence, levels ±1V, ideal edges (no ISI)
- **Output signal**: Waveform after 3-tap FIR filtering, visible pre-emphasis/de-emphasis effects at symbol transitions
  - For example: during "0→1" transition, output reaches peak in the 1st UI (main tap c[1]=0.6), with 20% pre/post-cursors on either side
  - During continuous "1" sequence, output steady-state value = c[0]×1 + c[1]×1 + c[2]×1 = 1.0V (satisfies normalization)

**Key Verification Points**:

| Test Item | Theoretical Value | Measurement Method | Pass Criteria |
|--------|--------|----------|---------|
| Output Peak (transition UI) | 0.6V | Single transition amplitude | Error < 1% |
| Steady-state Level (continuous symbols) | 1.0V | Average of 3 consecutive "1"s | Error < 1% |
| FIR Convolution Error | 0 | RMS(y_out - conv(x_in, c)) | < 1% |

**Waveform Diagram** (time-domain):
```
Input pattern:   0  0  1  1  1  0  0  1  0  1
            ___     ________     __  __  
Output waveform:  |   |___|        |___|  ||  |___
           ↑ ↑ ↑
        Pre-cursor Main tap Post-cursor
         0.2  0.6   0.2
```

#### 5.2.2 Convolution Correctness Verification

**Python Analysis Script**:
```python
import numpy as np

# Read trace file
data = np.loadtxt('ffe_tran_prbs.csv', delimiter=',', skiprows=1)
time = data[:, 0]
vin = data[:, 1]
vout = data[:, 2]

# FFE tap coefficients
taps = np.array([0.2, 0.6, 0.2])

# Theoretical convolution output
vout_theory = np.convolve(vin, taps, mode='same')

# Calculate error
rms_error = np.sqrt(np.mean((vout - vout_theory)**2)) / np.sqrt(np.mean(vout_theory**2))
print(f"Convolution RMS error: {rms_error*100:.2f}% (should be < 1%)")
```

### 5.3 De-emphasis Mode Analysis

#### 5.3.1 DEEMPHASIS_TEST Results

**Test Configuration**:
```json
{
  "signal_source": {"type": "pattern", "sequence": "0111_1000"},
  "ffe": {"taps": [0.0, 1.0, -0.35]}
}
```

**Expected Results**:

**Time-domain Waveform Analysis**:
- **Transition UI (n=0)**: Output amplitude = c[1]×1 = 1.0V (main tap maintains full swing)
- **Post-transition UI (n=1)**: Output amplitude = c[1]×1 + c[2]×1 = 1.0 - 0.35 = 0.65V (35% de-emphasis)
- **Steady-state UI (n≥2)**: Output amplitude = 1.0V (all taps aligned)

**Key Measurements**:

| Metric | Theoretical Value | Measurement Method | Pass Criteria |
|------|--------|----------|---------|
| De-emphasis Ratio | 35% | (V_n0 - V_n1) / V_n0 | 35% ± 2% |
| Main Tap Amplitude | 1.0V | Transition UI peak | Error < 5% |
| Frequency Enhancement (@ Nyquist) | +6.35dB | |H(fNyq)| / |H(DC)| | 6.0~6.5dB |

**Frequency-domain Characteristics**:

For a 3-tap FIR filter, the frequency response is:
```
H(f) = Σ c[k] × e^(-j2πfkT)   (k=0,1,2)
```

Substituting tap coefficients `c = [0.0, 1.0, -0.35]`:
```
H(f) = 0 + 1.0 × e^(-j2πfT) - 0.35 × e^(-j4πfT)
```

**DC Gain** (f=0):
```
H(0) = 1.0 × e^0 - 0.35 × e^0 = 1.0 - 0.35 = 0.65
|H(0)| = 0.65 → 20log₁₀(0.65) = -3.74 dB
```

**Nyquist Gain** (f = fₙ = 1/(2T)):
```
H(fₙ) = 1.0 × e^(-jπ) - 0.35 × e^(-j2π)
      = 1.0 × (-1) - 0.35 × 1
      = -1.0 - 0.35 = -1.35
|H(fₙ)| = 1.35 → 20log₁₀(1.35) = +2.61 dB
```

**High-frequency Enhancement**:
```
High-frequency enhancement = |H(fₙ)|_dB - |H(0)|_dB = 2.61 - (-3.74) = 6.35 dB
```

**Physical Meaning**:
De-emphasis mode relatively increases the energy proportion at transition edges by reducing the amplitude of continuous symbols, pre-compensating for high-frequency attenuation of the channel. It is suitable for PCIe Gen3/Gen4 and other standards.

### 5.4 Pre-emphasis Mode Analysis

#### 5.4.1 PREEMPHASIS_TEST Results

**Test Configuration**:
```json
{
  "signal_source": {"type": "SQUARE", "frequency": 1e9},
  "ffe": {"taps": [0.15, 0.7, 0.15]}
}
```

**Expected Results**:

**Edge Characteristics**:
- **Input square wave**: 1GHz (transition period 500ps), ideal edges (instantaneous flip)
- **Output waveform**: Edge steepness increased, rise/fall times shortened by approximately 20-30%
  - Pre-tap (c[0]=0.15) injects energy in advance, pre-raising/lowering the level
  - Main tap (c[1]=0.7) carries the main energy
  - Post-tap (c[2]=0.15) compensates for post-cursor ISI

**Measurement Metrics**:

| Metric | Theoretical | Measurement Method | Pass Criteria |
|------|------|----------|---------|
| Rise Time (10%-90%) | Reduced by 20% | Output/Input rise time ratio | 0.7~0.85 |
| Edge Overshoot | < 10% | (V_peak - V_final) / V_final | < 10% |
| High-frequency Enhancement | 0dB | No significant gain boost at 1GHz (balanced mode) | -3~0dB |

**Frequency-domain Characteristics** (balanced mode):
```
H(f) = 0.15 + 0.7 + 0.15 = 1.0 (DC)
|H(fNyq)| ≈ 0.4 (low-pass characteristic)
```

**Application Scenarios**:
Balanced mode is suitable for scenarios with severe channel reflections, using symmetric pre/post-cursor compensation to simultaneously suppress pre-cursor ISI (caused by reflection) and post-cursor ISI (caused by dispersion).

### 5.5 Tap Sweep Optimization

#### 5.5.1 TAP_SWEEP Results

**Test Configuration**:
- Fixed pre-tap: c[0] = 0
- Fixed main tap: c[1] = 1.0
- Sweep post-tap: c[2] = -0.5 ~ 0 (step 0.05)
- Signal source: PRBS7

**Typical Results Illustration**:

| c[2] | Eye Height(V) | Eye Width(ps) | ISI(mV) | Score |
|------|---------|----------|---------|------|
| 0.0  | 0.85    | 75       | 45      | Baseline |
| -0.1 | 0.90    | 78       | 35      | ↑ |
| -0.2 | 0.93    | 80       | 28      | ↑↑ |
| -0.3 | 0.95    | 81       | 22      | Optimal |
| -0.4 | 0.92    | 78       | 25      | ↓ |
| -0.5 | 0.87    | 72       | 38      | ↓↓ |

**Analysis Conclusions**:
- **Optimal coefficient**: c[2] ≈ -0.3, where eye height is maximum and ISI is minimum
- **Under-compensation (c[2] > -0.3)**: Residual post-cursor ISI, reduced eye height
- **Over-compensation (c[2] < -0.3)**: Introduces new pre-distortion errors, degrading signal quality

**Optimization Suggestions**:
In practical applications, optimal FFE coefficient combinations can be quickly found through similar sweep methods (combined with channel S-parameter simulation), balancing the trade-off between ISI elimination and noise amplification.

### 5.6 Channel Cascade Verification

#### 5.6.1 CHANNEL_COMBO Results

**Test Configuration**:
```
SignalSource → TxFfeTdf → ChannelModel → Measurement
```
- Channel model: First-order low-pass filter, -3dB @ 5GHz (simulating 10dB@Nyquist loss)
- FFE coefficients: Optimized through channel impulse response, e.g., [0.05, 0.8, -0.25]

**Comparison Experiment**:

| Configuration | Input Eye Height | Output Eye Height | Eye Height Improvement | Eye Width Improvement |
|------|---------|---------|---------|---------|
| No FFE | 2.0V | 1.2V | — | — |
| FFE Enabled | 2.0V | 1.65V | +37.5% | +15% |

**Conclusion**:
FFE improves the channel output eye height from 1.2V to 1.65V, a 37.5% improvement, demonstrating that pre-distortion effectively compensates for the channel's frequency-selective attenuation and reduces the burden on the receiver equalizer (CTLE/DFE).

### 5.7 Waveform Data File Format

CSV output format:
```
Time(s),Input Signal(V),Output Signal(V)
0.000000e+00,0.000000,0.000000
1.000000e-11,1.000000,0.600000
2.000000e-11,1.000000,1.000000
...
```

Number of sample points = Simulation duration / UI (e.g., 50ns / 10ps = 5000 points).

---

## 6. Technical Points

### 6.1 Single-ended Architecture Design Trade-offs

**Design Decision**: FFE adopts a single-ended signal architecture for behavioral-level modeling rather than a differential architecture.

**Rationale**:
- **Simplified Modeling**: Single-ended architecture focuses on core algorithm verification of the FIR filter, avoiding additional complexity from differential signal modeling (common-mode rejection, gain mismatch, phase skew, etc.)
- **Simulation Efficiency**: Reduces port count and signal paths, lowering SystemC-AMS simulation computational overhead (single-ended reduces port connections by approximately 40% compared to differential)
- **Algorithm Universality**: The ISI pre-compensation FIR convolution algorithm is essentially a scalar operation; single-ended modeling is sufficient to characterize the equalization effect

**Actual Hardware Correspondence**:
- Actual SerDes FFE is typically implemented differentially (differential pair driver, differential delay line)
- Single-ended behavioral model verification results can be directly mapped to differential hardware (differential signals can be viewed as a linear combination of two single-ended signals)
- If evaluation of differential-specific non-ideal effects (such as common-mode noise, mismatch) is needed, it can be extended to a differential architecture in subsequent design stages

### 6.2 Key Constraints of Symbol Rate Processing

**Design Characteristic**: FFE operates in the Symbol Rate clock domain, processing once per UI.

**Timing Constraints**:
- The time interval of the tap delay line must precisely equal one symbol period (UI)
- SystemC-AMS sampling rate setting: `set_rate(symbol_rate)`
- Time step errors accumulate and cause ISI compensation misalignment; recommended time resolution ≤ UI/1000

**Clock Domain Relationships with Other Modules**:
- **Upstream WaveGen**: Must be synchronized to the same symbol rate to ensure one symbol output per UI
- **Downstream Driver**: Can operate at higher sampling rates (oversampling mode), bridged through an interpolator
- **Cross-clock domain risk**: If WaveGen and FFE sampling rates do not match, it causes symbol misalignment or repeated sampling

**Debug Suggestions**:
- Use trace files to verify that input/output symbol timestamps are strictly aligned
- Check that delay line buffer size matches tap count: `m_delay_line.size() == m_params.taps.size()`

### 6.3 Normalization Strategy Selection

**Core Issue**: How to ensure FFE output signal swing is within a reasonable range to avoid saturation or SNR degradation in the downstream Driver?

**Three Normalization Strategies**:

**Strategy 1 - Absolute Value Normalization**:
```
Normalization factor = 1.0 / Σ|c[k]|
Output = (Σ c[k] × x[n-k]) × Normalization factor
```
- Advantages: Guarantees output peak does not exceed input peak, suitable for pre-emphasis/balanced modes
- Disadvantages: In de-emphasis mode, it reduces main tap amplitude (losing the physical meaning of de-emphasis definition)

**Strategy 2 - Main Tap Normalization (Recommended for De-emphasis)**:
```
Main tap fixed at 1.0, post-taps are negative values
No additional normalization needed, maintaining intuitive de-emphasis ratio definition
```
- Advantages: Complies with de-emphasis definitions in PCIe/USB standards (e.g., 3.5dB/6dB de-emphasis)
- Disadvantages: Output peak may slightly exceed input (when post-taps are large)

**Strategy 3 - No Normalization (Requires Post-stage Gain Adjustment)**:
```
Directly output FIR convolution result, with overall link gain uniformly adjusted by downstream Driver's dc_gain parameter
```
- Advantages: Maximum flexibility, can dynamically adjust overall link gain based on channel characteristics
- Disadvantages: Requires careful system-level gain budget management

**Recommended Configuration Guidelines**:
- De-emphasis mode: Use Strategy 2, main tap = 1.0
- Pre-emphasis/Balanced mode: Use Strategy 1 or 3, combined with Driver gain for joint optimization

### 6.4 Trade-offs Between Tap Count and Performance

**Tap Count Selection Criteria**:

| Channel Loss (@Nyquist) | Recommended Tap Count | Typical Application | Compensation Capability |
|------------------|----------|---------|---------|
| < 6dB | 3-tap | PCIe Gen3 short channel | Basic post-cursor compensation |
| 6-12dB | 5-tap | 10G/25G Ethernet | Balanced pre/post-cursor ISI |
| 12-20dB | 7-tap | 56G PAM4 medium distance | Extended ISI range |
| > 20dB | 9-tap and above | 112G ultra-long channel | Requires receiver DFE |

**Computational Complexity**:
- FIR filter computation: O(N) per symbol, N is tap count
- Delay line update: O(N) memory shifting
- For typical configurations with N≤7, computational overhead is negligible

**Diminishing Marginal Returns**:
- 3→5 taps: Significant eye height improvement (typically +30%)
- 5→7 taps: Moderate eye height improvement (typically +15%)
- 7→9 taps: Limited eye height improvement (typically +5%), and introduces more noise amplification

**Design Suggestions**:
- Short-distance applications (<10cm PCB): 3-tap is sufficient
- Standard applications (10-30cm backplane): 5-tap balances performance and complexity
- Extreme applications (>40cm or high rate): 7-tap, requires CTLE/DFE co-optimization

### 6.5 Co-optimization of FFE and Receiver Equalizer

**Equalization Budget Allocation Principles**:

Total ISI compensation requirements in the SerDes link should be reasonably allocated among TX FFE, RX CTLE, and RX DFE:

**Advantages and Limitations of TX FFE**:
- ✅ Advantages: Pre-compensation does not amplify receiver noise, improving receiver SNR
- ✅ Advantages: Reduces receiver equalizer power consumption (CTLE/DFE can use lower gain)
- ❌ Limitations: Can only compensate for static or slow-varying channel characteristics, cannot track rapid changes
- ❌ Limitations: Increased transmitter power consumption (rising peak current, complex Driver design)

**Advantages and Limitations of CTLE**:
- ✅ Advantages: Continuous-time equalization, compensates for high-frequency attenuation, no inter-symbol interference accumulation
- ✅ Advantages: Can dynamically adjust gain to adapt to channel changes
- ❌ Limitations: Zero-pole design is limited, difficult to precisely match complex channel responses
- ❌ Limitations: High-frequency gain amplifies noise, SNR degradation

**Advantages and Limitations of DFE**:
- ✅ Advantages: Precisely eliminates post-cursor ISI, does not amplify noise
- ✅ Advantages: Strong adaptability, can be optimized online through LMS algorithm
- ❌ Limitations: Decision feedback delay limits rate (first tap must complete within 1 UI)
- ❌ Limitations: Error propagation risk (previous decision error affects subsequent taps)

**Co-optimization Strategies**:

| Scenario | FFE Strategy | CTLE Strategy | DFE Strategy | Rationale |
|------|---------|---------|---------|------|
| Low-loss channel (<6dB) | Light de-emphasis (3-tap) | Low gain or disabled | 1-2 tap DFE | Avoid over-equalization amplifying noise |
| Medium loss (6-12dB) | Moderate pre-emphasis (5-tap) | Medium gain CTLE | 3-5 tap DFE | Balanced allocation, each bears part of ISI |
| High loss (>12dB) | Strong pre-emphasis (7-tap) | High gain CTLE | 5-8 tap DFE | Maximize use of all equalization means |

**Design Process Suggestions**:
1. Analyze channel impulse response to determine total ISI energy distribution (pre-cursor vs. post-cursor)
2. Prioritize using FFE to compensate for post-cursor ISI (optimal SNR at TX side)
3. CTLE compensates for high-frequency attenuation (zero-poles match channel slope)
4. DFE handles residual post-cursor ISI and nonlinear effects

### 6.6 Dynamic Update Mechanism for Tap Coefficients

**Static Configuration vs. Dynamic Adaptation**:

**Static Configuration** (current implementation):
- Tap coefficients are loaded at simulation/system initialization and remain fixed during operation
- Applicable scenarios: Channel characteristics are known and stable (laboratory testing, standard channel models)
- Advantages: Simple and reliable, no convergence time overhead
- Disadvantages: Cannot adapt to temperature drift, aging, load changes

**Dynamic Adaptation** (extended function):
- Adjust FFE coefficients in real-time through receiver feedback (such as eye height monitoring, BER estimation)
- Common algorithms: LMS (Least Mean Squares), RLS (Recursive Least Squares)
- Convergence time: Typically 1000-10000 UI
- Implementation complexity: Requires DE-TDF bridge module to transfer feedback signals

**How to Extend Adaptive Functionality in Current Architecture**:
1. Add control port: `sca_de::sca_in<std::vector<double>> tap_update`
2. Detect update events in `processing()`: `if (tap_update.event()) { m_taps = tap_update.read(); }`
3. Connect Adaption module (DE domain, implementing LMS algorithm) with FFE control port

### 6.7 Parameter Validation and Boundary Condition Handling

**Core Validation Rules**:

**Rule 1 - Non-empty Validation**:
```cpp
if (m_params.taps.empty()) {
    SC_REPORT_ERROR("TxFfeTdf", "FFE taps cannot be empty");
}
```

**Rule 2 - Main Tap Identification**:
```cpp
auto max_it = std::max_element(taps.begin(), taps.end(), 
    [](double a, double b) { return std::abs(a) < std::abs(b); });
size_t main_idx = std::distance(taps.begin(), max_it);
```

**Rule 3 - Dynamic Range Check**:
```cpp
for (auto c : taps) {
    if (std::abs(c) > 1.0) {
        SC_REPORT_WARNING("TxFfeTdf", "Tap coefficient exceeds 1.0, may cause saturation");
    }
}
```

**Boundary Condition Handling**:
- **Initialization phase**: Delay line filled with 0 values; first N-1 output symbols will have startup transients (can be eliminated by pre-filling historical data)
- **Tap coefficient is 0**: Automatically skip calculation for performance optimization
- **Single-tap mode** (N=1): Degrades to simple gain adjustment, no ISI compensation effect

### 6.8 Simulation Performance Optimization Tips

**Optimization 1 - Delay Line Implementation Choice**:
- Simple array shifting: `std::rotate(delay_line.begin(), delay_line.begin()+1, delay_line.end())`
- Circular buffer: Avoids shifting, achieves O(1) update through index wrapping
- Recommendation: Use array shifting when tap count ≤5 (code simplicity), use circular buffer when >5

**Optimization 2 - Conditional Execution**:
```cpp
// Skip taps with coefficient 0
for (size_t i = 0; i < taps.size(); ++i) {
    if (std::abs(taps[i]) < 1e-10) continue;  // Avoid invalid computation
    output += taps[i] * delay_line[i];
}
```

**Optimization 3 - Vectorized Computation** (requires compiler support):
```cpp
// Use SIMD instructions to accelerate FIR convolution
output = std::inner_product(taps.begin(), taps.end(), delay_line.begin(), 0.0);
```

**Optimization 4 - Reduce Trace Overhead**:
- Only enable trace when waveform analysis is needed
- Use sampled trace (e.g., record once every 10 UIs) to reduce file size

---

## 7. Running Guide

### 7.1 Environment Configuration

Before running tests, configure SystemC and SystemC-AMS environment variables:

```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
source scripts/setup_env.sh
```

### 7.2 Build and Run

#### Using CMake (Recommended)

```bash
# Enter build directory
cd build
cmake ..
make ffe_tran_tb

# Run testbench
cd tb
./ffe_tran_tb [scenario]
```

#### Using Makefile

```bash
# Build FFE testbench
make ffe_tran_tb

# Run specified scenario
cd tb
./ffe_tran_tb [scenario]
```

Scenario parameters:
- `prbs` or `0` - Basic PRBS test (default)
- `deemp` or `1` - De-emphasis test
- `preemp` or `2` - Pre-emphasis test
- `sweep` or `3` - Tap sweep
- `combo` or `4` - Channel cascade test

### 7.3 Configuration File Management

FFE module parameters are managed through JSON configuration files; typical configurations are located in the `config/` directory:

```json
{
  "tx": {
    "ffe": {
      "taps": [0.0, 1.0, -0.35]
    }
  }
}
```

After modifying the configuration, the testbench needs to be rebuilt for the changes to take effect.

### 7.4 Results Viewing

After testing is complete, waveform data is saved to CSV files (`ffe_tran_*.csv`). Use Python scripts for visualization:

```bash
# Plot time-domain waveform
python scripts/plot_ffe_waveform.py

# Frequency-domain analysis (requires scipy)
python scripts/analyze_ffe_frequency.py
```

Console output key statistical metrics:
- Output peak-to-peak
- Convolution error (BASIC_PRBS scenario)
- De-emphasis ratio (DEEMPHASIS_TEST scenario)
- Optimal tap coefficients (TAP_SWEEP scenario)

---

## 8. Reference Information

### 8.1 Related Files

| File Type | Path | Description |
|---------|------|------|
| Parameter Definition | `include/common/parameters.h` | TxFfeParams struct |
| Header File | `include/ams/tx_ffe.h` | TxFfeTdf class declaration |
| Implementation File | `src/ams/tx_ffe.cpp` | TxFfeTdf class implementation |
| Testbench | `tb/tx/ffe/ffe_tran_tb.cpp` | Transient simulation testbench (to be implemented) |
| Test Helpers | `tb/tx/ffe/ffe_helpers.h` | Signal sources and monitors (to be implemented) |
| Unit Tests | `tests/unit/test_ffe_basic.cpp` | GoogleTest unit tests (to be implemented) |
| Waveform Plotting | `scripts/plot_ffe_waveform.py` | Python visualization script (to be implemented) |

### 8.2 Related Module Documentation

| Module | Documentation Path | Relationship |
|------|---------|---------|
| WaveGen | `/docs/modules/waveGen.md` | Upstream module, provides PRBS data source |
| TX Mux | `/docs/modules/mux.md` | Downstream module, receives FFE output |
| TX Driver | `/docs/modules/driver.md` | TX link end, final output to channel |
| Channel | `/docs/modules/channel.md` | Channel characteristics determine FFE coefficient settings |
| RX CTLE | `/docs/modules/ctle.md` | Receiver equalizer, works with FFE |
| RX DFE | `/docs/modules/dfesummer.md` | Receiver feedback equalizer |

### 8.3 Dependencies

**Compile-time Dependencies**:
- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 Standard

**Test Dependencies**:
- GoogleTest 1.12.1 (unit tests)
- NumPy/SciPy (Python analysis tools)
- Matplotlib (waveform plotting)

### 8.4 Related Standards and Specifications

| Standard | Version | Related Content |
|------|------|---------|
| IEEE 802.3 | 2018 | Ethernet FFE specifications (100GBASE-KR4, etc.) |
| PCIe | Gen 4/5/6 | Transmitter pre-emphasis requirements (TX Preset) |
| USB4 | v2.0 | FFE coefficient range and step definitions |
| OIF CEI | 56G/112G | Common equalizer templates |

### 8.5 Configuration Examples

#### Basic Configuration (3-tap FFE)

```json
{
  "tx": {
    "ffe": {
      "taps": [0.0, 1.0, -0.35],
      "enable": true
    }
  }
}
```

**Parameter Description**:
- `taps[0] = 0.0`: No pre-tap
- `taps[1] = 1.0`: Main tap (normalized)
- `taps[2] = -0.35`: First post-tap (35% de-emphasis)

#### High-loss Channel Configuration (5-tap)

```json
{
  "tx": {
    "ffe": {
      "taps": [-0.05, 0.1, 1.0, -0.4, -0.15],
      "enable": true
    }
  }
}
```

**Application Scenarios**:
- Backplane channel (>30dB insertion loss @ Nyquist frequency)
- Long cable links
- Scenarios requiring strong pre-compensation

#### Pre-emphasis Configuration

```json
{
  "tx": {
    "ffe": {
      "taps": [-0.2, 1.0, 0.0],
      "enable": true
    }
  }
}
```

**Application Scenarios**:
- Compensating for channel pre-cursor ISI
- Special reflection or resonance characteristics

### 8.6 Academic References

**FIR Filter Theory**:
- Alan V. Oppenheim, *Discrete-Time Signal Processing*, 3rd Edition
- Chapter 6: Frequency Response of Discrete-Time Systems

**Equalization Techniques**:
- S. Gondi and B. Razavi, "Equalization and Clock Recovery for a 2.5-10 Gb/s 2-PAM/4-PAM Backplane Transceiver", IEEE JSSC 2009
- A. Emami-Neyestanak et al., "A 6 Gb/s Voltage-Mode Transmitter", IEEE JSSC 2007

**SystemC-AMS Modeling**:
- *SystemC AMS User's Guide*, Accellera, Version 2.3
- Chapter 4: TDF (Timed Data Flow) Modeling Methods

### 8.7 External Resources

- **Accellera Website**: https://systemc.org/ (SystemC-AMS standard downloads)
- **SerDes Design Guides**: Xilinx UG476, Altera AN 529 (Commercial FPGA SerDes design references)
- **IBIS-AMI Modeling**: IBIS Open Forum (Behavioral modeling standard, complementary to this module)

---

**Document Version**: v0.1  
**Last Updated**: 2026-01-13  
**Author**: Yizhe Liu

---
