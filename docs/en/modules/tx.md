# TX Transmitter Module Technical Documentation

🌐 **Languages**: [中文](../../modules/tx.md) | [English](tx.md)

**Level**: AMS Top-Level Module  
**Current Version**: v1.0 (2026-01-27)  
**Status**: Production Ready

---

## 1. Overview

The SerDes Transmitter (TX) is the starting module of a high-speed serial link, responsible for converting digital bit streams into high-swing analog differential signals with pre-equalization, driving them through the transmission line to the channel. The TX pre-compensates for channel losses through Feed-Forward Equalization (FFE) and provides sufficient driving capability and impedance matching through the Driver.

### 1.1 Design Principles

The core design philosophy of the TX transmitter adopts a cascaded architecture, proactively compensating for Inter-Symbol Interference (ISI) introduced by the channel at the transmit side, thereby reducing the burden on the receiver equalizer:

```
Digital Input → WaveGen → FFE → Mux → Driver → Differential Output → Channel
           (Digital→Analog) (FIR Eq) (Channel Sel) (Drive&Match)
```

**Signal Flow Processing Logic**:

1. **WaveGen (Waveform Generator)**: Converts digital bit streams (0/1) to analog NRZ waveforms (e.g., ±1V), supporting PRBS patterns and jitter injection
2. **FFE (Feed-Forward Equalizer)**: Pre-distorts the signal through an FIR filter to implement pre-emphasis or de-emphasis, compensating for high-frequency channel attenuation
3. **Mux (Multiplexer)**: Channel selection and time-division multiplexing functionality, supporting flexible configuration of multi-channel systems
4. **Driver (Output Driver)**: Final output buffer stage, providing sufficient driving capability, impedance matching, and swing control

**Pre-Equalization Strategy**:

- **Pre-emphasis**: Injects extra energy at transition edges to enhance high-frequency components
- **De-emphasis**: Attenuates the amplitude of non-transition symbols, relatively increasing the edge energy proportion
- **Hybrid Mode**: Uses both pre-cursor and post-cursor taps to balance pre- and post-cursor ISI compensation

### 1.2 Core Features

- **Four-Stage Cascaded Architecture**: WaveGen → FFE → Mux → Driver, covering the complete transmit-side signal chain
- **Pre-Equalization Capability**: FFE provides 3-7 tap FIR filters, supporting pre-emphasis and de-emphasis configurations
- **Differential Output**: Driver adopts a full differential architecture with configurable output impedance (typically 50Ω)
- **Configurable Swing**: Driver output swing adjustable (typically 800-1200mV peak-to-peak)
- **Bandwidth Limitation Modeling**: Driver multi-pole transfer function simulates parasitic effects and package impact
- **Nonlinear Effects**: Supports both soft saturation (tanh) and hard saturation (clamp) modes
- **PSRR Modeling**: Power supply ripple couples to output through configurable transfer function
- **Slew Rate Limiting**: Optional output edge rate constraints, simulating real device characteristics

### 1.3 Sub-Module Overview

| Module | Class Name | Function | Key Parameters | Standalone Documentation |
|--------|------------|----------|----------------|--------------------------|
| **WaveGen** | `WaveGenTdf` | Digital bit stream generator | pattern, jitter | waveGen.md |
| **FFE** | `TxFfeTdf` | Feed-Forward Equalizer | taps | ffe.md |
| **Mux** | `TxMuxTdf` | Multiplexer | lane_sel | mux.md |
| **Driver** | `TxDriverTdf` | Output Driver | dc_gain, poles, vswing | driver.md |

### 1.4 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v1.0 | 2026-01-27 | Initial version, integrated top-level documentation for four sub-modules |

---

## 2. Module Interface

### 2.1 Port Definitions (TDF Domain)

#### 2.1.1 Top-Level Input/Output Ports

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `data_in` | Input | int | Digital bit stream input (from encoder) |
| `out_p` | Output | double | Differential output positive terminal (drives channel) |
| `out_n` | Output | double | Differential output negative terminal |
| `vdd` | Input | double | Power supply voltage (for PSRR modeling) |

> **Important**: The `vdd` port must be connected even if PSRR functionality is not enabled (SystemC-AMS requires all ports to be connected).

#### 2.1.2 Internal Module Cascade Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TX Transmitter Top-Level Module                     │
│                                                                                  │
│  data_in                                                                         │
│     │                                                                            │
│     ↓                                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│  │ WaveGen │    │   FFE   │    │   Mux   │    │ Driver  │                       │
│  │         │    │         │    │         │    │         │                       │
│  │ data ←──┼────┼─        │    │         │    │         │                       │
│  │ _in     │    │         │    │         │    │         │     out_p             │
│  │         │    │ in ←────┼────┼─ out    │    │         │    ─────────→         │
│  │ out ────┼────┼→ in     │    │         │    │ in_p ←──┼── out_p              │
│  │         │    │ out ────┼────┼→ in     │    │ in_n ←──┼── out_n              │
│  │         │    │         │    │ out ────┼────┼→        │     out_n             │
│  │         │    │         │    │         │    │         │    ─────────→         │
│  └─────────┘    └─────────┘    │ lane ←──┼────┼── sel   │                       │
│                                │ _sel    │    │         │                       │
│                                └─────────┘    │ vdd ←───┼── vdd                 │
│                                               │         │                       │
│                                               │ out_p ──┼→ out_p               │
│                                               │ out_n ──┼→ out_n               │
│                                               └─────────┘                       │
│                                                    ↑                            │
│                                                   VDD                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Signal Flow**:

- **Main Signal Path**: `data_in` → WaveGen.out → FFE.in → FFE.out → Mux.in → Mux.out → Driver.in → `out_p/out_n`
- **Control Signals**: lane_sel (channel selection), vdd (power supply)

### 2.2 Parameter Configuration (TxParams Structure)

#### 2.2.1 Overall Parameter Structure

```cpp
struct TxParams {
    TxFfeParams ffe;            // FFE parameters
    int mux_lane;               // Mux channel selection
    TxDriverParams driver;      // Driver parameters
    
    TxParams() : mux_lane(0) {}
};

// WaveGen parameters defined independently
struct WaveGenParams {
    PRBSType type;              // PRBS type
    std::string poly;           // Polynomial expression
    std::string init;           // Initial state
    double single_pulse;        // Single pulse width
    JitterParams jitter;        // Jitter parameters
    ModulationParams modulation;// Modulation parameters
};
```

#### 2.2.2 Sub-Module Parameter Summary

| Sub-Module | Key Parameters | Default Configuration | Adjustment Purpose |
|------------|----------------|----------------------|--------------------|
| WaveGen | `type=PRBS31`, `jitter.RJ_sigma=0` | PRBS sequence | Data source generation |
| FFE | `taps=[0.2, 0.6, 0.2]` | 3-tap symmetric | Pre-compensate channel ISI |
| Mux | `mux_lane=0` | Bypass mode | Single-channel system |
| Driver | `dc_gain=1.0`, `poles=[50e9]`, `vswing=0.8` | Standard configuration | Drive & match |

#### 2.2.3 Configuration Example (JSON Format)

```json
{
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF",
    "single_pulse": 0.0,
    "jitter": {
      "RJ_sigma": 0.0,
      "SJ_freq": [],
      "SJ_pp": []
    }
  },
  "tx": {
    "ffe": {
      "taps": [0.0, 1.0, -0.25]
    },
    "mux_lane": 0,
    "driver": {
      "dc_gain": 1.0,
      "vswing": 0.8,
      "vcm_out": 0.6,
      "output_impedance": 50.0,
      "poles": [50e9],
      "sat_mode": "soft",
      "vlin": 1.0,
      "psrr": {"enable": false},
      "imbalance": {"gain_mismatch": 0.0, "skew": 0.0},
      "slew_rate": {"enable": false}
    }
  }
}
```

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The complete signal processing flow of the TX transmitter includes 5 key steps:

```
Step 1: Digital bit stream → WaveGen (Digital→Analog NRZ waveform)
Step 2: WaveGen output → FFE (FIR pre-equalization, pre-emphasis/de-emphasis)
Step 3: FFE output → Mux (Channel selection, optional delay/jitter)
Step 4: Mux output → Driver (Gain, bandwidth limiting, saturation)
Step 5: Driver output → Differential signal output to channel
```

**Timing Constraints**:

- WaveGen sampling rate = Data rate (e.g., 10Gbps corresponds to 10GHz)
- FFE delay line interval = 1 UI
- Mux operates at symbol rate
- Driver sampling rate consistent with upstream

### 3.2 WaveGen-FFE Cascade Design

#### 3.2.1 Waveform Generation

WaveGen maps digital bits (0/1) to NRZ analog waveforms:

- **NRZ Encoding**: `0 → -amplitude`, `1 → +amplitude`
- **Typical Amplitude**: ±1V (2V peak-to-peak)
- **Supported PRBS Types**: PRBS-7, PRBS-15, PRBS-23, PRBS-31

#### 3.2.2 FFE Equalization Strategy

FFE implements pre-compensation using an FIR filter, with the mathematical expression:

```
y[n] = Σ c[k] × x[n-k]
       k=0 to N-1
```

**De-emphasis Configuration Example** (PCIe Gen3):

| Configuration | Tap Coefficients | De-emphasis Amount | Application Scenario |
|---------------|------------------|--------------------|----------------------|
| 3.5dB | [0, 1.0, -0.25] | 3.5 dB | Short channel |
| 6dB | [0, 1.0, -0.35] | 6 dB | Medium channel |
| 9.5dB | [0, 1.0, -0.5] | 9.5 dB | Long channel |

**Pre-emphasis Configuration Example**:

| Configuration | Tap Coefficients | Description |
|---------------|------------------|-------------|
| Balanced Mode | [0.15, 0.7, 0.15] | Symmetric front and back |
| Hybrid Mode | [0.05, 0.85, -0.2] | Light pre-emphasis + de-emphasis |

#### 3.2.3 Tap Coefficient Constraints

- **Normalization Constraint**: `Σ|c[k]| ≈ 1` (maintain power)
- **Main Tap Maximum**: Main tap is typically the maximum value
- **Physical Implementability**: Tap coefficients should be within hardware implementation range

### 3.3 Mux Channel Selection and Delay Modeling

#### 3.3.1 Single-Channel Mode (Bypass)

- `lane_sel = 0`
- Pass-through mode, used for simple systems or debugging
- No additional delay

#### 3.3.2 Multi-Channel Mode (Extension)

- In complete N:1 serializer systems, Mux selects one of N parallel lanes
- This behavioral model simplifies to single-input single-output
- Channel index specified through `lane_sel` parameter

#### 3.3.3 Delay and Jitter Modeling

- **Propagation Delay**: 10-50ps (configurable)
- **Jitter Injection**: Supports DCD (Duty Cycle Distortion) and RJ (Random Jitter)
- **Application Scenario**: Testing CDR jitter tolerance capability

### 3.4 Driver Output Stage Design

#### 3.4.1 Gain and Impedance Matching

**Impedance Matching Voltage Divider Effect**:

```
Open-circuit voltage: Voc = Vin × dc_gain
Channel entry voltage: Vchannel = Voc × Z0/(Zout + Z0)

For ideal matching (Zout = Z0 = 50Ω):
Vchannel = Voc / 2
```

**Parameter Configuration Example**:

Assuming input is ±1V (2V peak-to-peak), desired channel entry is 800mV peak-to-peak, with ideal matching:

```
Driver internal open-circuit swing requirement: 800mV × 2 = 1600mV
Configuration: dc_gain = 1600mV / 2000mV = 0.8
```

#### 3.4.2 Bandwidth Limitation

Multi-pole transfer function simulates the frequency response of the driver:

```
H(s) = Gdc × ∏(1 + s/ωp_j)^(-1)
```

**Pole Frequency Selection**:

| Data Rate | Recommended Pole Frequency | Description |
|-----------|---------------------------|-------------|
| 10 Gbps | 10-15 GHz | 1.5-2× Nyquist |
| 25 Gbps | 25-35 GHz | 1.5-2× Nyquist |
| 56 Gbps | 40-50 GHz | 1.5-2× Nyquist |

#### 3.4.3 Nonlinear Saturation

**Soft Saturation (Recommended)**:

```
Vout = Vswing × tanh(Vin / Vlin)
```

- Continuous derivative, low harmonic distortion
- Vlin parameter determines linear region range
- Recommended: `Vlin = Vswing / 1.2`

**Hard Saturation**:

```
Vout = clamp(Vin, -Vswing/2, +Vswing/2)
```

- Simple calculation but discontinuous derivative
- Suitable for rapid functional verification

#### 3.4.4 PSRR Modeling

Power supply ripple impact on output:

```cpp
struct PsrrParams {
    bool enable;                     // Enable PSRR modeling
    double gain;                     // PSRR path gain (e.g., 0.01 = -40dB)
    std::vector<double> poles;       // Low-pass filter poles
    double vdd_nom;                  // Nominal power supply voltage
};
```

Working principle: `vdd_ripple = vdd - vdd_nom` → PSRR transfer function → coupled to differential output

#### 3.4.5 Differential Imbalance

```cpp
struct ImbalanceParams {
    double gain_mismatch;            // Gain mismatch (%)
    double skew;                     // Phase offset (s)
};
```

Gain mismatch effect:
- `gain_p = 1 + mismatch/200`
- `gain_n = 1 - mismatch/200`

### 3.5 Output Common-Mode Voltage Control

**Common-Mode Setting**:

- Driver's `vcm_out` parameter sets the differential output common-mode voltage (typically 0.6V)
- Ensures compliance with channel and receiver input range requirements

**AC-Coupled Link**:

- If the channel has AC coupling capacitors, common-mode is determined by the channel's DC blocking characteristics
- TX-side common-mode does not affect RX, but Driver output must avoid exceeding its linear range

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

TX testbenches are typically open-loop designs:

- **TX Side**: WaveGen + FFE + Mux + Driver cascade
- **Load**: Ideal 50Ω impedance or complete channel model
- **Performance Evaluation**: Eye diagram measurement, spectrum analysis, output swing statistics

Difference from RX testing:
- TX testing is open-loop (no feedback path)
- Primary focus on output signal quality rather than bit error rate

### 4.2 Test Scenario Definitions

| Scenario | Command Line Arguments | Test Objective | Output Files |
|----------|------------------------|----------------|--------------|
| BASIC_OUTPUT | `basic` / `0` | Basic output waveform and swing | tx_tran_basic.csv |
| FFE_SWEEP | `ffe_sweep` / `1` | Eye diagram under different FFE coefficients | tx_eye_ffe_*.csv |
| DRIVER_SATURATION | `sat` / `2` | Large signal saturation characteristics | tx_sat.csv |
| FREQUENCY_RESPONSE | `freq` / `3` | TX link frequency response | tx_freq_resp.csv |
| JITTER_INJECTION | `jitter` / `4` | WaveGen jitter injection effect | tx_jitter.csv |

### 4.3 Scenario Configuration Details

#### BASIC_OUTPUT - Basic Output Test

- **Signal Source**: PRBS-31, 10Gbps
- **FFE**: Default de-emphasis [0, 1.0, -0.25]
- **Driver**: Standard configuration, Vswing=800mV
- **Load**: 50Ω ideal impedance
- **Verification Points**:
  - Output swing ≈ 400mV (considering 50% voltage divider)
  - Eye height > 80% of swing
  - Eye width > 0.6 UI

#### FFE_SWEEP - FFE Parameter Sweep

- **Tap Coefficient Variations**:
  - Configuration 1: [0, 1.0, 0] (no equalization)
  - Configuration 2: [0, 1.0, -0.2] (3.5dB de-emphasis)
  - Configuration 3: [0, 1.0, -0.35] (6dB de-emphasis)
  - Configuration 4: [0.05, 0.9, -0.25] (hybrid mode)
- **Verification Points**: Compare eye diagram opening and spectrum under different configurations

#### DRIVER_SATURATION - Saturation Test

- **Input Amplitude Variation**: 0.5× nominal → 2× nominal
- **FFE**: No equalization (to avoid interference)
- **Verification Points**:
  - Linear region: output ∝ input
  - Saturation region: output clamped at sat_max/sat_min

#### FREQUENCY_RESPONSE - Frequency Response Test

- **Signal Source**: Sine sweep (100MHz ~ 50GHz)
- **Measurement**: Output/input amplitude ratio
- **Verification Points**:
  - -3dB bandwidth should be close to pole frequency
  - High-frequency roll-off slope ≈ -20dB/decade/pole

#### JITTER_INJECTION - Jitter Injection Test

- **WaveGen Jitter Configuration**: RJ_sigma=0.5ps, DCD=2%
- **Measurement**: Increase in output eye diagram jitter
- **Verification Points**: Jitter transfer consistent with configuration

### 4.4 Signal Connection Topology

```
┌──────────┐   ┌─────┐   ┌─────┐   ┌────────┐   ┌────────┐   ┌──────────┐
│ WaveGen  │→→→│ FFE │→→→│ Mux │→→→│ Driver │→→→│ Channel│→→→│ Eye Mon  │
│(PRBS-31) │   │(FIR)│   │(Sel)│   │(Amp&BW)│   │(50Ω/S4P│   │(Eye Capt)│
└──────────┘   └─────┘   └─────┘   └────────┘   └────────┘   └──────────┘
```

### 4.5 Auxiliary Module Descriptions

| Module | Function | Configuration Parameters |
|--------|----------|--------------------------|
| **Ideal Load** | 50Ω pure resistive load | impedance |
| **Channel Model** | S-parameter import, simulates real channel | touchstone |
| **Eye Monitor** | Captures eye diagram data, calculates eye height/width/jitter | ui_bins, amp_bins |
| **Spectrum Analyzer** | FFT analysis of output spectrum | fft_size |

---

## 5. Simulation Result Analysis

### 5.1 Statistical Metrics Description

| Metric | Calculation Method | Significance |
|--------|-------------------|--------------|
| **Output Swing** | max(out_p - out_n) - min(out_p - out_n) | Driving capability |
| **Eye Height** | min(high level) - max(low level) | Noise margin |
| **Eye Width** | Optimal sampling window (UI) | Timing margin |
| **Rise/Fall Time** | 10%-90% level crossing time | Edge speed |
| **Jitter (RMS)** | Edge position standard deviation | Timing accuracy |
| **Spectral Purity** | Harmonic distortion (dBc) | Nonlinear distortion |

### 5.2 Typical Test Result Interpretation

#### BASIC_OUTPUT Test Result Example

**Configuration**: 10Gbps, PRBS-31, FFE=[0, 1.0, -0.25], 50Ω load

**Expected Results**:
```
=== TX Performance Summary ===
Output Swing (Diff):   420 mV (target 400mV, considering voltage divider)
Eye Height:            360 mV (85% of swing)
Eye Width:             0.68 UI (68 ps)
Rise Time (20%-80%):   25 ps
Fall Time (20%-80%):   27 ps
Jitter (RMS):          1.2 ps (no WaveGen jitter)
FFE Main Tap Energy:   75%
FFE Post-Tap Energy:   -25%
```

**Waveform Characteristics**:
- WaveGen output: Standard NRZ square wave, ±1V
- FFE output: Pre-emphasis "spikes" at transitions, reduced level in flat regions
- Driver output: Slowed edges (bandwidth limitation), swing as expected

#### FFE_SWEEP Result Interpretation

**Eye Diagram Comparison (at channel entry)**:

| Configuration | Post-Tap Coefficient | Eye Height(mV) | Eye Width(UI) | Description |
|---------------|---------------------|----------------|---------------|-------------|
| No EQ | 0 | 280 | 0.55 | Baseline, worst ISI |
| 3.5dB De-emphasis | -0.2 | 340 | 0.65 | Standard configuration |
| 6dB De-emphasis | -0.35 | 370 | 0.70 | High-loss channel optimization |
| Hybrid Mode | Pre 0.05, Post -0.25 | 360 | 0.68 | Balanced pre/post cursor |

**Spectrum Comparison**:
- No equalization: High low-frequency energy, large high-frequency attenuation
- 6dB de-emphasis: +6dB high-frequency boost, low-frequency reduction, flatter spectrum

#### DRIVER_SATURATION Result Interpretation

**Input-Output Characteristic Curve**:
```
Output Swing(mV)
800 |         ┌─────────  Saturation Region (Vswing limit)
    |       ┌─┘
600 |     ┌─┘  Linear Region (gain=0.8)
    |   ┌─┘
400 | ┌─┘
    |─┘
0   └────────────────────▶ Input Swing(mV)
    0   500  1000  1500  2000
```

**Analysis Points**:
- Input < 1000mV: Linear amplification, output = input × 0.8
- Input > 1500mV: Enters saturation, output clamped at 800mV
- Soft saturation curve is smoother, harmonic distortion less than hard saturation

### 5.3 Waveform Data File Format

**tx_tran_basic.csv**:
```csv
Time(s),WaveGen_out(V),FFE_out(V),Driver_out_diff(V)
0.0e0,0.000,0.000,0.000
1.0e-11,1.000,1.000,0.380
2.0e-11,1.000,0.750,0.370
1.0e-10,1.000,0.750,0.375
1.1e-10,-1.000,-1.250,-0.485
```

---

## 6. Running Guide

### 6.1 Environment Configuration

Before running tests, configure environment variables:

```bash
source scripts/setup_env.sh
```

Ensure the following dependencies are correctly installed:
- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 compatible compiler

### 6.2 Build and Run

```bash
cd build
cmake ..
make tx_tran_tb         # Build TX top-level testbench
make ffe_tran_tb        # Build FFE single-module test
make tx_driver_tran_tb  # Build Driver single-module test
cd tb
./tx_tran_tb [scenario]
```

Scenario arguments:
- `basic` or `0` - Basic output test (default)
- `ffe_sweep` or `1` - FFE parameter sweep
- `sat` or `2` - Saturation test
- `freq` or `3` - Frequency response test
- `jitter` or `4` - Jitter injection test

### 6.3 Parameter Tuning Process

**Step 1: Determine Output Swing Requirements**

- Consult interface standards (PCIe/USB/Ethernet)
- Consider channel loss budget
- Set Driver's Vswing parameter

**Step 2: Configure FFE Tap Coefficients**

- Method A: According to standard recommended values (e.g., PCIe specification)
- Method B: Optimize based on channel S-parameter simulation
- Method C: Enable adaptive algorithm (extended feature)

**Step 3: Set Driver Gain and Bandwidth**

- `dc_gain`: Consider impedance matching voltage divider, set to 2× desired swing (for 50Ω matching)
- `poles`: Set according to data rate, fp ≈ 1.5-2×(Bitrate/2)

**Step 4: Run Simulation Verification**

```bash
./tx_tran_tb basic
# Check output swing, eye diagram, rise time
```

**Step 5: Iterative Optimization**

- If eye diagram closes: Optimize FFE coefficients or increase Driver bandwidth
- If saturation occurs: Reduce input amplitude or adjust Driver gain
- If jitter is too large: Check WaveGen configuration or reduce PSRR impact

### 6.4 Result Viewing

After testing completes, statistical results are output to the console, and waveform data is saved to CSV files. Use Python for visualization:

```bash
# Waveform visualization
python scripts/plot_tx_waveforms.py tx_tran_basic.csv

# Eye diagram plotting
python scripts/plot_eye_diagram.py tx_eye_ffe_*.csv

# Frequency response curve
python scripts/plot_freq_response.py tx_freq_resp.csv
```

---

## 7. Technical Key Points

### 7.1 FFE Coefficient Design Methods

#### 7.1.1 Standard Recommended Values

| Standard | Tap Coefficients | De-emphasis Amount |
|----------|------------------|--------------------|
| PCIe Gen3 | [0, 1.0, -0.25] | 3.5dB |
| PCIe Gen4 | [0, 1.0, -0.35] | 6dB |
| USB 3.2 | [0, 1.0, -0.2] | Optional |

Advantages: Quick configuration, good compatibility

#### 7.1.2 Channel Inverse Filtering

```
FFE transfer function F(f) designed as inverse of H_channel(f)
Objective: F(f) × H_channel(f) ≈ constant
```

- Solve in frequency domain, convert back to time domain for tap coefficients
- Advantages: Theoretically optimal, adapts to channel characteristics

#### 7.1.3 Time-Domain Optimization

- Measure channel pulse response h[n]
- Objective: Minimize Σ(h_eq[n])² for n≠0 (suppress ISI)
- Constraint: Σ|c[k]| ≤ 1 (power limit)

### 7.2 Driver Impedance Matching Principles

**Voltage Divider Effect**:
```
Open-circuit voltage: Voc = Vin × dc_gain
Channel entry voltage: Vchannel = Voc × Z0/(Zout + Z0)

For ideal matching (Zout = Z0 = 50Ω):
Vchannel = Voc / 2
```

**dc_gain Setting**:
```
If desired channel entry is 800mV peak-to-peak, input 2V:
Voc requirement = 800mV × 2 = 1600mV
dc_gain = 1600mV / 2000mV = 0.8
```

**Mismatch Impact**:
- `Zout > Z0`: Reflection coefficient > 0, signal bounces back to TX
- `Zout < Z0`: Reflection coefficient < 0, signal attenuates excessively
- Tolerance requirement: `|Zout - Z0| / Z0 < 10%`

### 7.3 Bandwidth Limitation Trade-offs

**Pole Frequency Selection**:
- **Too high**: Excessive bandwidth, high-frequency noise passes, EMI increases
- **Too low**: Insufficient bandwidth, ISI increases, eye diagram closes
- **Recommended**: fp = 1.5-2 × (Bitrate/2)

**Multi-Pole Modeling**:
- Real Drivers have multiple parasitic poles (transistor Cgs, load Cload, package Lpkg)
- Single-pole model is simplified but may underestimate high-frequency attenuation
- Critical links recommended to use 2-3 poles for accurate modeling

### 7.4 Soft Saturation vs Hard Saturation

**Soft Saturation (tanh)**:
- Advantages: Continuous derivative, low harmonic distortion, good convergence
- Disadvantages: Slightly more complex calculation
- Applicable: Production simulation, high precision requirements

**Hard Saturation (clamp)**:
- Advantages: Simple calculation, clear limit cases
- Disadvantages: Discontinuous derivative, may introduce high-order harmonics
- Applicable: Rapid verification, worst-case analysis

**Vlin Parameter Tuning**:
```
Vlin = Vswing / α
Larger α → Narrower linear region → Earlier saturation
Smaller α → Wider linear region → Larger signal margin
Recommended: α = 1.2-1.5
```

### 7.5 Power Management in FFE-Driver Cascade

**Problem**: FFE pre-emphasis increases peak swing

```
Example: FFE coefficients [0, 1.0, -0.3]
At transition: y[n] = 1.0×(+1) + (-0.3)×(-1) = 1.3 (30% increase)
```

**Solution 1: Normalize FFE Output**
```
scale_factor = 1 / max(|y[n]|)
FFE output × scale_factor
```

**Solution 2: Driver Margin Reservation**
```
Driver's sat_max set to 1.3-1.5× nominal swing
```

**Solution 3: Adaptive Power Control**
```
Monitor Driver output, if approaching saturation reduce FFE coefficient amplitude
```

### 7.6 Practicality of Jitter Modeling

**DCD (Duty Cycle Distortion)**:
- Physical source: Clock duty cycle ≠ 50%
- Impact: Odd/even UI widths unequal, produces deterministic jitter
- Modeling: Apply ±DCD/2 time offset to odd/even symbols in WaveGen

**RJ (Random Jitter)**:
- Physical source: PLL phase noise, thermal noise
- Impact: Random fluctuation of data edge timing
- Modeling: Add Gaussian-distributed time shift at each symbol moment

**Jitter Transfer**:
- TX-side injected jitter → Channel transmission → RX-side CDR needs to track
- Can be used to test CDR's JTOL capability

### 7.7 Time Step and Sampling Rate Settings

**Consistency Requirement**:
All TDF modules must use the same sampling rate:

```cpp
// Global configuration
double Fs = 100e9;  // 100 GHz
double Ts = 1.0 / Fs;

// Module set_attributes()
wavegen.set_timestep(Ts);
ffe.set_timestep(Ts);
mux.set_timestep(Ts);
driver.set_timestep(Ts);
```

**Sampling Rate Selection**:
- Minimum: 2× highest frequency (Nyquist)
- Recommended: 5-10× symbol rate
- Too high: Long simulation time, large files
- Too low: Waveform distortion, inaccurate bandwidth modeling

---

## 8. Reference Information

### 8.1 Related Files

| File Type | Path | Description |
|-----------|------|-------------|
| WaveGen Header | `/include/ams/wave_gen.h` | WaveGenTdf class declaration |
| WaveGen Implementation | `/src/ams/wave_gen.cpp` | WaveGenTdf class implementation |
| FFE Header | `/include/ams/tx_ffe.h` | TxFfeTdf class declaration |
| FFE Implementation | `/src/ams/tx_ffe.cpp` | TxFfeTdf class implementation |
| Mux Header | `/include/ams/tx_mux.h` | TxMuxTdf class declaration |
| Mux Implementation | `/src/ams/tx_mux.cpp` | TxMuxTdf class implementation |
| Driver Header | `/include/ams/tx_driver.h` | TxDriverTdf class declaration |
| Driver Implementation | `/src/ams/tx_driver.cpp` | TxDriverTdf class implementation |
| Parameter Definitions | `/include/common/parameters.h` | TxParams/WaveGenParams structures |
| WaveGen Documentation | `/docs/modules/waveGen.md` | WaveGen detailed technical documentation |
| FFE Documentation | `/docs/modules/ffe.md` | FFE detailed technical documentation |
| Mux Documentation | `/docs/modules/mux.md` | Mux detailed technical documentation |
| Driver Documentation | `/docs/modules/driver.md` | Driver detailed technical documentation |
| FFE Testbench | `/tb/tx/ffe_tran_tb.cpp` | FFE transient test |
| Driver Testbench | `/tb/tx/tx_driver_tran_tb.cpp` | Driver transient test |
| FFE Unit Tests | `/tests/unit/test_ffe_*.cpp` | FFE unit test suite |
| Driver Unit Tests | `/tests/unit/test_tx_driver_*.cpp` | Driver unit test suite |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 Standard
- GoogleTest 1.12.1 (Unit Testing)

### 8.3 Performance Metrics Summary

| Metric | Typical Value | Description |
|--------|---------------|-------------|
| Maximum Data Rate | 56 Gbps | Depends on process and package |
| Output Swing | 800-1200 mV | Compliant with PCIe/USB standards |
| Rise/Fall Time | 20-40 ps | @ 10Gbps |
| Output Jitter | < 2 ps RMS | Excluding channel impact |
| FFE Tap Count | 3-7 | Typical 3 taps (PCIe) |
| Output Impedance | 50 Ω | Differential 100Ω |
| PSRR | > 40 dB | @ 1MHz |

### 8.4 Interface Standard Reference

| Standard | Data Rate | Swing Requirement | FFE Configuration |
|----------|-----------|-------------------|-------------------|
| PCIe Gen3 | 8 Gbps | 800-1200mV | 3-tap, 3.5dB or 6dB de-emphasis |
| PCIe Gen4 | 16 Gbps | 800-1200mV | 3-tap, mandatory de-emphasis |
| USB 3.2 Gen2 | 10 Gbps | 800-1000mV | 3-tap, optional equalization |
| 10G Ethernet | 10.3125 Gbps | 500-800mV | 5-tap, hybrid mode |
| 25G Ethernet | 25.78125 Gbps | 400-800mV | 5-7 tap, strong equalization |

---

**Document Version**: v1.0  
**Last Updated**: 2026-01-27  
**Author**: Yizhe Liu
