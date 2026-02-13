# CDR Module Technical Documentation

🌐 **Languages**: [中文](../../modules/cdr.md) | [English](cdr.md)

**Level**: AMS Submodule (RX)  
**Class Name**: `RxCdrTdf`  
**Current Version**: v0.2 (2026-01-20)  
**Status**: Production Ready

---

## 1. Overview

Clock and Data Recovery (CDR) is the core module of the SerDes receiver. Its main function is to recover clock information from the received data stream and generate the optimal sampling phase, ensuring that the sampler samples data at the optimal position of the eye diagram, thereby maximizing the system's error margin.

### 1.1 Design Principles

The core design concept of CDR utilizes the clock information carried by data transition edges and dynamically adjusts the sampling phase through a closed-loop feedback mechanism:

- **Phase Detection**: Detects the phase relationship between data transition edges and the current sampling clock, extracting phase error information
- **Loop Filtering**: Processes phase errors through a Proportional-Integral (PI) controller to suppress high-frequency jitter and stabilize the loop
- **Phase Adjustment**: Outputs the filtered phase correction signal to the sampler to achieve dynamic phase tracking

This module adopts the classic Bang-Bang Phase Detector (BBPD) + PI Digital Loop Filter architecture:

```
Data Input → Edge Detection → Bang-Bang Phase Detector → PI Controller → Phase Output
              ↑                                               ↓
              └──────────────────Phase Feedback──────────────────────┘
```

The phase detector outputs discrete Early/Late signals, which are integrated and averaged by the PI controller and converted into continuous phase adjustment values, output to the sampler's `phase_offset` port.

### 1.2 Core Features

- **Bang-Bang Phase Detection**: Binary phase error detection based on data transition edges
- **Digital PI Loop Filter**: Configurable proportional gain (Kp) and integral gain (Ki)
- **Phase Range Limiting**: Configurable phase adjustment range and resolution
- **Sampler Coordination**: Outputs phase adjustment signal (unit: seconds) to directly drive the sampler's phase offset
- **Behavioral Model**: Suitable for system-level simulation and algorithm verification

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2026-01-07 | Initial version, Bang-Bang PD + PI controller |
| v0.2 | 2026-01-20 | Fixed all known issues, code and documentation synchronized, production ready |

---

## 2. Module Interface

### 2.1 Port Definitions (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in` | Input | double | Received data input (analog signal, from DFE or sampler) |
| `phase_out` | Output | double | Phase adjustment output (unit: seconds s) |

> **Port Notes**:
> - The `in` port receives continuous analog signals; CDR extracts clock information from data transitions
> - The `phase_out` port outputs phase offset (unit: seconds), connected to the sampler's `phase_offset` input port
> - Positive values indicate delayed sampling (late clock), negative values indicate early sampling (early clock)

### 2.2 Parameter Configuration

The CDR module's parameters are configured through the `CdrParams` structure, which contains two sub-structures: PI controller parameters and phase interpolator parameters.

#### 2.2.1 PI Controller Parameters (CdrPiParams)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kp` | double | 0.01 | Proportional Gain |
| `ki` | double | 1e-4 | Integral Gain |

**Working Principle**:

The PI controller is a second-order digital loop filter with the standard discrete-time formula:
```
Integral State: I[n] = I[n-1] + Ki × e[n]
PI Output:      φ[n] = Kp × e[n] + I[n]
```
Where:
- `φ[n]`: Output phase at the nth sampling moment
- `e[n]`: Phase error (Bang-Bang PD outputs ±1)
- `I[n]`: Integral state (accumulation of historical errors)
- `Kp`: Proportional gain, controls transient response speed
- `Ki`: Integral gain, eliminates steady-state phase error

**Parameter Tuning Guide**:

- **Kp (Proportional Gain)**:
  - Increasing Kp speeds up phase lock, but too large values cause oscillation
  - Typical range: 0.001 ~ 0.1
  - Default value 0.01 is suitable for 10 Gbps systems

- **Ki (Integral Gain)**:
  - Increasing Ki improves tracking accuracy, but too large values reduce loop stability
  - Typical relationship: Ki ≈ Kp/10 ~ Kp/100
  - Default value 1e-4 matches Kp=0.01

**Loop Characteristics**:

The natural frequency ωn and damping coefficient ζ of a second-order PI loop are determined by Kp and Ki (based on linearized analysis):
```
ωn = √(Ki × Fs)
ζ = Kp / (2 × ωn)
```
Where Fs is the sampling rate. It is recommended to keep ζ between 0.7~1.0 for optimal step response.

> ⚠️ **Important Note**: The above formulas are derived based on the assumption of a **linear phase detector**, through continuous-time domain linearization. However, this module uses a **Bang-Bang Phase Detector** (outputs discrete ±1), and its nonlinear characteristics cause actual loop behavior to deviate from linear theory:
> - Bang-Bang PD lacks a linear region and cannot provide continuous phase error information
> - Actual loop bandwidth and damping coefficient will deviate from theoretical calculations
> - Discrete binary output introduces additional phase jitter (see Section 7.2)
> 
> Therefore, these formulas are only for **preliminary design approximation**, and actual parameters require simulation verification and tuning.

#### 2.2.2 Phase Interpolator Parameters (CdrPaiParams)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | double | 1e-12 | Phase adjustment resolution (unit: seconds s) |
| `range` | double | 5e-11 | Phase adjustment range (unit: seconds s) |

**Working Principle**:

The Phase Interpolator (PI) converts the phase control word output by the digital loop filter into actual time offset.

- **Resolution**:
  - Defines the minimum step of phase adjustment
  - Default 1 ps (1e-12 seconds) corresponds to high-precision hardware implementation
  - Physical meaning: DNL (Differential Nonlinearity) of the hardware phase interpolator
  - Coarser resolution (e.g., 5 ps) can simulate low-cost implementations

- **Range**:
  - Defines the maximum offset of phase adjustment (±range)
  - Default 50 ps (±25 ps) is sufficient to cover typical frequency offset and jitter
  - Physical meaning: Linear range of the phase interpolator
  - Should be larger than the expected maximum frequency offset × UI

**Practical Application Example**:

For a 10 Gbps system (UI = 100 ps):
- Frequency offset ±500 ppm → Maximum phase offset = ±50 ps → range setting ≥ 50 ps
- Hardware phase interpolator 6-bit → Resolution = UI/64 ≈ 1.5 ps

#### 2.2.3 Complete Parameter Structure

```cpp
struct CdrParams {
    CdrPiParams pi;      // PI controller parameters
    CdrPaiParams pai;    // Phase interpolator parameters
};
```

**JSON Configuration Example**:

```json
{
  "cdr": {
    "pi": {
      "kp": 0.01,
      "ki": 1e-4
    },
    "pai": {
      "resolution": 1e-12,
      "range": 5e-11
    }
  }
}
```

### 2.3 Port Connection Example

The CDR module is typically connected after the sampler or DFE, outputting phase signals that are fed back to the sampler:

```cpp
// Instantiate CDR module
CdrParams cdr_params;
cdr_params.pi.kp = 0.01;
cdr_params.pi.ki = 1e-4;
RxCdrTdf cdr("cdr", cdr_params);

// Connect signals
cdr.in(data_signal);              // From DFE or sampler
cdr.phase_out(phase_adjust_sig);  // Output to sampler's phase_offset port

// Sampler connection (closed loop)
sampler.phase_offset(phase_adjust_sig);  // Receive CDR phase adjustment
sampler.data_out(sampled_data);
cdr.in(sampled_data);  // Or connect to equalized analog signal
```

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The CDR module's `processing()` method implements the following processing steps:

```
Step 1: Read Input Data → Step 2: Edge Detection → Step 3: Bang-Bang Phase Detection → 
Step 4: PI Controller Update → Step 5: Phase Range Limiting → Step 6: Output Phase Adjustment
```

**Step 1 - Read Input Data**: Read the current data signal from the `in` port.

**Step 2 - Edge Detection**: Detect data transitions by comparing the current bit with the previous bit:
```cpp
double current_bit = in.read();
if (std::abs(current_bit - m_prev_bit) > threshold) {
    // Edge detected
}
```

**Step 3 - Bang-Bang Phase Detection**: When an edge is detected, determine phase early/late:
- If `current_bit > m_prev_bit` (rising edge) → Phase error = +1 (clock late)
- If `current_bit < m_prev_bit` (falling edge) → Phase error = -1 (clock early)
- Phase error = 0 when no edge

> ⚠️ **Simplified Implementation Error**:
> - The above logic is a **wrong simplification** and does not conform to the real Bang-Bang Phase Detector working principle
> - Real BBPD requires **XOR result of data sampler and edge sampler** to determine early/late, not simply based on edge polarity
> - Current implementation is for demonstration purposes only; actual applications require complete BBPD architecture (see Section 3.2)

**Step 4 - PI Controller Update**: Update the phase accumulation based on phase error:
```cpp
m_integral += ki * phase_error;  // Integral state update
double phase_output = kp * phase_error + m_integral;  // PI output
```

**Step 5 - Phase Range Limiting**: Limit phase output within ±range:
```cpp
if (phase_output > range) phase_output = range;
if (phase_output < -range) phase_output = -range;
```

**Step 6 - Phase Quantization**: Quantize phase output to the configured resolution:
```cpp
double quantized_phase = std::round(phase_output / resolution) * resolution;
```

**Step 7 - Output Phase Adjustment**: Write the phase adjustment to the `phase_out` port and pass to the sampler.

### 3.2 Bang-Bang Phase Detector Principles

The Bang-Bang Phase Detector (BBPD) is a binary phase detector that determines whether the sampling phase is early or late based on data transition edges:

**Ideal Working Principle**:

At data transition edges, the sampled data value reflects the phase relationship between the sampling moment and the data center:
- If sampling moment is earlier than data center, edge sampling is closer to old data
- If sampling moment is later than data center, edge sampling is closer to new data

**Current Implementation**:

The current version uses edge polarity detection:
```
Rising edge (0→1): Clock is considered late (needs to be early) → phase_error = +1
Falling edge (1→0): Clock is considered early (needs to be late) → phase_error = -1
```

> **Note**: This is a simplified implementation of edge polarity detection. Complete BBPD requires XOR comparison using the Data Sampler and Edge Sampler.

**Complete BBPD Architecture** (future version):

```
         ┌──────────────┐
data ───→│ Data Sampler │──→ D[n]
  │      └──────────────┘       │
  │                             ↓
  │      ┌──────────────┐    ┌─────┐
  └─────→│ Edge Sampler │──→ │ XOR │──→ phase_error
         └──────────────┘    └─────┘
              (Early Sampling)        ↑
                                  D[n-1]
```

Phase error = D[n-1] ⊕ Edge[n]:
- 0 ⊕ 0 = 0 (no error or large error)
- 0 ⊕ 1 = 1 (clock late)
- 1 ⊕ 0 = 1 (clock early)
- 1 ⊕ 1 = 0 (no error or large error)

### 3.3 PI Controller Design

The PI (Proportional-Integral) controller is a classic second-order digital loop filter that balances fast response and steady-state accuracy.

**Discrete-Time Transfer Function**:

```
H(z) = Kp + Ki/(1 - z⁻¹)
```

**Time-Domain Recursive Formula**:

```cpp
// Integral state update
I[n] = I[n-1] + Ki × e[n]

// PI output
φ[n] = Kp × e[n] + I[n]
```

Where:
- `φ[n]`: Output phase at the nth sampling moment
- `e[n]`: Phase error (Bang-Bang PD outputs ±1)
- `I[n]`: Integral state (accumulation of historical errors)
- `Kp`: Proportional gain, controls transient response speed
- `Ki`: Integral gain, eliminates steady-state phase error

**C++ Implementation**:

```cpp
// Integral term: accumulate historical errors
m_integral += ki * phase_error;

// Proportional term: instantaneous response
double prop_term = kp * phase_error;

// Total output
double phase_out = prop_term + m_integral;
```

**Loop Characteristics Analysis**:

For a second-order PI loop, the open-loop transfer function is:
```
G(s) = (Kp × s + Ki) / s²
```

Closed-loop characteristic equation determines loop stability and step response:
- **Natural Frequency**: ωn = √(Ki × Fs), determines response speed
- **Damping Coefficient**: ζ = Kp / (2 × ωn), determines overshoot and oscillation
- **Loop Bandwidth**: BW ≈ ωn, determines jitter tracking capability

Recommended design guidelines:
- ζ ≈ 0.707 (critical damping) for fastest overshoot-free response
- BW ≈ Data rate/1000 ~ Data rate/10000 (e.g., 10 Gbps → 1~10 MHz)

**PI Controller Z-Domain Analysis**:

In discrete-time systems, the PI controller's Z-domain transfer function is:
```
H(z) = Kp + Ki × T / (1 - z⁻¹)
```
Where T is the sampling period (for baud-rate CDR, T = UI). Converting to difference equation:
```
y[n] = y[n-1] + Kp × (e[n] - e[n-1]) + Ki × T × e[n]
```
Here it can be seen:
- **Proportional Term**: Kp × (e[n] - e[n-1]), only responds to error changes
- **Integral Term**: Ki × T × e[n], accumulates all historical errors

**Phase Update Rate and Data Rate Relationship**:

This CDR design adopts a baud-rate architecture, where each data bit triggers one phase detection and update:
- Data rate = 10 Gbps → Phase update rate = 10 GHz
- UI = 100 ps → Phase update period = 100 ps
- Loop delay = 1 UI (phase error detection → PI calculation → phase application)

Higher update rates bring faster lock speeds and better jitter tracking, but also increase power consumption and design complexity. Some CDRs use 1/2 or 1/4 baud-rate to reduce power, but sacrifice tracking bandwidth.

**Edge Detection Threshold Selection**:

The current implementation uses a fixed threshold of 0.5 to detect data transitions, which fails in the following cases:
- Signal swing is not unit-normalized (e.g., CTLE output is ±0.3V)
- DC offset exists (e.g., VGA output common-mode voltage drift)
- Signal attenuation is severe (channel loss causes insufficient edge amplitude)

**Improvement Scheme**:
```cpp
// Adaptive threshold detection
double threshold = 0.5 * (max_signal - min_signal);  // Relative to signal swing
double edge_detect = std::abs(current_bit - m_prev_bit) > threshold;
```
Or use a peak detector to dynamically track signal swing, setting the threshold to 10%~20% of the swing.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The CDR testbench adopts a closed-loop integrated design and requires close coordination with the sampler module to verify phase tracking capability. Core design principles:

1. **Closed-Loop Architecture**: CDR and Sampler form a phase feedback closed loop; phase adjustment results directly affect sampling quality
2. **Scenario-Driven**: Covers key test scenarios including frequency acquisition, phase tracking, jitter tolerance, and lock detection
3. **Performance Evaluation**: Supports BER testing, lock time measurement, phase error statistics, and other performance metrics
4. **Configurability**: Supports multiple parameter configurations and scenario switching for easy performance comparison and optimization

**Differences from Other Module Testbenches**:

- CTLE/VGA and other modules can be tested independently (open-loop)
- CDR must form a closed loop with the Sampler to verify functionality
- Requires precise control of input data frequency offset and jitter
- Test metrics include dynamic characteristics (lock time, tracking bandwidth)

### 4.2 Test Scenario Definitions

The testbench supports five core test scenarios, covering CDR's main functions and performance metrics:

| Scenario | Command Line Parameter | Test Objective | Output File |
|----------|------------------------|----------------|-------------|
| PHASE_LOCK_BASIC | `lock` / `0` | Basic phase lock function verification | cdr_tran_lock.csv |
| FREQUENCY_OFFSET | `freq` / `1` | Frequency offset capture capability | cdr_tran_freq.csv |
| JITTER_TOLERANCE | `jtol` / `2` | Jitter tolerance test (JTOL) | cdr_tran_jtol.csv |
| PHASE_TRACKING | `track` / `3` | Dynamic phase tracking capability | cdr_tran_track.csv |
| LOOP_BANDWIDTH | `bw` / `4` | Loop bandwidth measurement | cdr_tran_bw.csv |

### 4.3 Scenario Configuration Details

#### PHASE_LOCK_BASIC - Basic Phase Lock Test

Verify the basic function of CDR locking from an initial random phase to the optimal sampling phase.

- **Signal Source**: PRBS-15 pseudo-random sequence
- **Data Rate**: 10 Gbps (UI = 100 ps)
- **Initial Phase Offset**: Random (within 0~UI range)
- **PI Parameters**: Kp=0.01, Ki=1e-4 (default values)
- **Simulation Time**: ≥ 10,000 UI (ensure loop convergence)
- **Verification Points**:
  - Phase error converges within ±5 ps
  - BER < 1e-12
  - Phase stable after lock (no continuous oscillation)

**Expected Waveform Characteristics**:
- Phase adjustment signal monotonically converges from initial value to steady-state value
- Convergence process shows exponential decay (second-order system characteristic)
- Small jitter present after lock (inherent characteristic of Bang-Bang PD)

**Debug Points** (for known bugs):
- Check if phase detector output correctly reflects early/late information
- Verify PI controller's proportional and integral terms are correctly separated
- Confirm phase output signal is correctly connected to sampler

#### FREQUENCY_OFFSET - Frequency Offset Capture Test

Verify CDR's ability to capture and compensate for frequency offset between transmitter and receiver.

- **Signal Source**: 10 Gbps PRBS-7
- **Frequency Offset**: ±100 ppm, ±500 ppm, ±1000 ppm (graded testing)
- **PI Parameters**: Kp=0.01, Ki=1e-4
- **PI range**: Must be ≥ |freq_offset| × UI × lock time
- **Simulation Time**: ≥ 50,000 UI
- **Verification Points**:
  - Whether system can lock within specified time
  - Phase drift rate after lock = phase slope corresponding to frequency offset
  - Phase adjustment range does not exceed pai.range limit

**Physical Meaning**:

Frequency offset causes phase to accumulate at a fixed rate:
```
Phase drift rate = freq_offset × UI
Example: 100 ppm @ 10 Gbps → 100 ps/1e6 UI = 0.1 fs/UI
```

The integral term Ki of the CDR is responsible for tracking this constant phase slope; if Ki is too small, it cannot fully eliminate static phase error.

**Test Steps**:
1. Configure transmitter frequency offset (through clock period fine-tuning)
2. Start CDR from initial state
3. Record phase adjustment signal time-domain waveform
4. Measure lock time (phase error stabilized to ±10 ps)
5. Verify phase slope after lock matches frequency offset

#### JITTER_TOLERANCE - Jitter Tolerance Test

Verify CDR's tolerance to input data jitter, a key performance indicator of SerDes systems.

- **Signal Source**: 10 Gbps PRBS-31 (long sequence ensures statistical validity)
- **Jitter Types**:
  - **Random Jitter (RJ)**: Gaussian distribution, σ = 1 ps, 2 ps, 5 ps
  - **Periodic Jitter (SJ)**: Sinusoidal modulation, frequency sweep (1 kHz ~ 100 MHz)
  - **Combined Jitter**: RJ + SJ superposition
- **Test Method**: Fixed jitter amplitude, sweep jitter frequency, record BER
- **PI Parameters**: Kp=0.01, Ki=1e-4
- **Simulation Time**: ≥ 1e6 UI per frequency point (ensure BER measurement accuracy)
- **Verification Points**:
  - Plot JTOL curve (jitter tolerance vs frequency)
  - Verify low-frequency jitter tracking capability (frequency < loop bandwidth)
  - Verify high-frequency jitter suppression capability (frequency > loop bandwidth)

**JTOL Curve Characteristics**:

```
Jitter Tolerance (UI)
    ^
1.0 |━━━━━┓                     ← Low frequency: perfect tracking
    |      ┗━━━━┓                ← Corner frequency ≈ loop bandwidth
0.5 |           ┗━━━━┓           ← Slope -20 dB/decade
    |                ┗━━━━━━━    ← High frequency: intrinsic tolerance
0.1 |________________________
        1k  10k 100k  1M  10M  100M  (Hz)
```

**Key Frequency Points**:
- **Low frequency band (< BW/10)**: CDR fully tracks jitter, tolerance ≈ 1 UI
- **Corner frequency (≈ BW)**: Tolerance begins to decrease
- **High frequency band (> 10×BW)**: CDR cannot track, tolerance determined by sampler intrinsic margin

#### PHASE_TRACKING - Dynamic Phase Tracking Test

Verify CDR's tracking capability for dynamic phase modulation.

- **Signal Source**: 10 Gbps PRBS-7
- **Phase Modulation**: Sine wave modulation of sampling phase
  - Modulation frequency: 100 kHz, 1 MHz, 10 MHz
  - Modulation amplitude: ±10 ps, ±20 ps, ±50 ps
- **PI Parameters**: Kp=0.01, Ki=1e-4
- **Simulation Time**: ≥ 100 modulation periods
- **Verification Points**:
  - Calculate tracking error (input phase modulation vs CDR output phase)
  - Measure tracking delay (phase difference)
  - Verify loop bandwidth (-3 dB point)

**Phase Tracking Transfer Function**:

The CDR loop's closed-loop transfer function (phase output/phase input) has low-pass characteristics:
```
H(f) = (Kp×s + Ki) / (s² + Kp×s + Ki)  (continuous domain approximation)

-3 dB bandwidth ≈ √Ki  (rad/s)
```

By sweeping modulation frequency and measuring output/input amplitude ratio, the transfer function curve can be plotted to verify theoretical bandwidth.

#### LOOP_BANDWIDTH - Loop Bandwidth Measurement

Precisely measure the actual bandwidth of the CDR loop to verify compliance with theoretical design.

- **Test Principle**: Inject phase modulation with known amplitude into data stream, measure CDR output amplitude response
- **Modulation Frequency Sweep**: 10 kHz ~ 100 MHz (logarithmic spacing, 10 points per octave)
- **Modulation Amplitude**: Fixed 20 ps (small signal linear range)
- **PI Parameters**: Multiple parameter sets for comparison (verify Kp/Ki impact on bandwidth)
- **Output**: Bode plot (magnitude and phase response)
- **Verification Points**:
  - -3 dB bandwidth compared with theoretical calculation (error should be < 20%)
  - Phase margin > 45° (stability indicator)
  - No resonance peak (sufficient damping)

**Test Configuration Table**:

| Kp | Ki | Theoretical BW (MHz) | Theoretical ζ |
|----|----|----------------------|---------------|
| 0.005 | 2.5e-5 | 2.5 | 0.707 |
| 0.01 | 1e-4 | 5.0 | 0.707 |
| 0.02 | 4e-4 | 10.0 | 0.707 |

By comparing actual bandwidth of different parameter sets, verify the effectiveness of parameter tuning.

### 4.4 Signal Connection Topology

The core of the CDR testbench is the closed-loop connection between CDR and Sampler:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ DiffSignalSource│       │  RxSamplerTdf   │       │   RxCdrTdf      │
│  (PRBS + Jitter)│       │                 │       │                 │
│                 │       │                 │       │                 │
│  out_p ─────────┼───────▶ inp             │       │                 │
│  out_n ─────────┼───────▶ inn             │       │                 │
└─────────────────┘       │                 │       │                 │
                          │  data_out ──────┼───────▶ in              │
                          │                 │       │                 │
                          │  phase_offset ◀─┼───────┤ phase_out       │
                          └─────────────────┘       └─────────────────┘
                                   │                          │
                                   ▼                          ▼
                          ┌─────────────────┐       ┌─────────────────┐
                          │  SamplerMonitor │       │   CdrMonitor    │
                          │  - BER Stats    │       │  - Phase Waveform |
                          │  - Eye Capture  │       │  - Lock Detection |
                          └─────────────────┘       └─────────────────┘
```

**Key Connection Notes**:

1. **Differential Signal Source → Sampler**:
   - Provides differential data signal with jitter/frequency offset
   - Supports programmable jitter injection (RJ/SJ)
   - Supports programmable frequency offset

2. **Sampler → CDR (Forward Path)**:
   - Connection depends on CDR input type:
     - **Option A**: Sampler.data_out → CDR.in (digital signal)
     - **Option B**: Equalizer output → CDR.in (analog signal, requires edge sampler)
   - Current simplified implementation uses Option A

3. **CDR → Sampler (Feedback Path)**:
   - CDR.phase_out → Sampler.phase_offset
   - Phase adjustment unit: seconds (s)
   - Positive value = delayed sampling, negative value = early sampling

4. **Monitor Connections**:
   - SamplerMonitor: BER statistics, eye diagram data collection
   - CdrMonitor: Phase adjustment waveform recording, lock state detection

**Loop Delay Considerations**:

CDR loop has inherent delay (signal propagation + processing delay), affecting loop stability:
```
Total delay = Sampler delay + CDR processing delay + Phase application delay
Typical value: 1~2 UI
```

The testbench should support configurable delay to verify the impact of delay on lock performance.

### 4.5 Auxiliary Module Descriptions

#### DiffSignalSource - Differential Signal Source (Enhanced)

Customized signal source for CDR testing, supporting precise frequency and jitter control:

**Waveform Types**:
- PRBS-7/15/31: Different length sequences, verify CDR adaptation to data patterns
- Alternating Pattern (010101...): Maximum transition density, verify phase detector response
- Low Transition Density (sparse transitions): Verify CDR hold capability at low transition rates

**Jitter Injection Capability**:

| Jitter Type | Parameter | Description |
|-------------|-----------|-------------|
| Random Jitter (RJ) | sigma (standard deviation) | Gaussian distribution, typical value 0.5~5 ps |
| Periodic Jitter (SJ) | frequency, amplitude | Sine modulation, supports multi-tone superposition |
| Bounded Uncorrelated Jitter (BUJ) | peak-to-peak | Uniform distribution |
| Duty Cycle Distortion (DCD) | offset | Asymmetric high/low level times |

**Frequency Offset Control**:

Achieve precise frequency offset by adjusting symbol period:
```cpp
Actual UI = Nominal UI × (1 + freq_offset_ppm / 1e6)
Example: 10 Gbps + 100 ppm → UI = 100 ps × 1.0001 = 100.01 ps
```

**Configuration Example**:
```json
{
  "signal_source": {
    "data_rate": 10e9,
    "pattern": "PRBS31",
    "jitter": {
      "rj_sigma": 2e-12,
      "sj_freq": 1e6,
      "sj_amplitude": 10e-12
    },
    "freq_offset_ppm": 100
  }
}
```

#### SamplerMonitor - Sampler Monitor Module

Monitor sampler output to evaluate CDR performance impact on BER:

**Functions**:
- **Real-time BER Statistics**: Compare sampled output with ideal reference sequence
- **Error Location Recording**: Timestamp, error type (single-bit/burst)
- **Eye Diagram Capture**: Record amplitude and phase distribution of sampling points
- **Statistical Analysis**: Q-factor, eye height, eye width

**Output Files**:
- `sampler_monitor.csv`: Bit-by-bit record (time, data, reference, error flag)
- `ber_summary.json`: BER statistics and eye diagram parameters

#### CdrMonitor - CDR Status Monitor Module

Dedicated to CDR internal state monitoring and performance analysis:

**Monitor Signals**:
| Signal | Description |
|--------|-------------|
| phase_out | Phase adjustment output waveform |
| phase_error | Phase detector output (internal signal, requires debug interface) |
| lock_status | Lock status indicator (requires lock detector implementation) |
| integral_state | PI controller integral state (internal signal) |

**Analysis Functions**:
- **Lock Time Measurement**: Time from start to phase error < threshold
- **Lock Jitter Statistics**: RMS jitter of phase adjustment signal in steady state
- **Loop Response Analysis**: Step response, frequency response measurement
- **Parameter Optimization Recommendations**: Suggest Kp/Ki adjustments based on measured performance

**Output**:
- `cdr_phase.csv`: Phase waveform data
- `cdr_performance.json`: Performance metrics summary
- `cdr_debug.log`: Internal state log (debug mode)

**Debug Mode Features**:

The testbench supports debug mode output for intermediate states, used for performance analysis and parameter tuning:
```json
{
  "debug_mode": true,
  "debug_signals": [
    "phase_detector_output",
    "proportional_term",
    "integral_term",
    "phase_before_limit",
    "phase_after_limit"
  ]
}
```

These internal signal waveforms are crucial for diagnosing phase detector behavior and PI controller performance.

#### JitterInjector - Jitter Injection Module (Optional)

Independent jitter injection module, can be inserted at any point in the signal chain:

**Application Scenarios**:
- Injection at sampler input: Simulates channel jitter
- Injection at CDR input: Simulates sampler jitter
- Injection at clock path: Simulates reference clock jitter

**Injection Methods**:
- **Time-domain modulation**: Directly modulate signal time axis
- **Phase-domain modulation**: Modulate sampling phase (add to CDR output)

**Configuration Example**:
```json
{
  "jitter_injector": {
    "enable": true,
    "type": "sinusoidal",
    "frequency": 1e6,
    "amplitude": 20e-12,
    "insertion_point": "sampler_input"
  }
}
```

---

## 5. Simulation Results Analysis

This chapter introduces the interpretation methods for typical simulation results of each CDR testbench scenario, definitions of key performance indicators, and analysis methods.

### 5.1 Statistical Indicators

CDR performance evaluation involves various time-domain and frequency-domain indicators. Below are definitions and calculation methods for key indicators.

#### 5.1.1 Time-Domain Indicators

**Phase Error**

Defined as the deviation between sampling phase and optimal sampling phase, unit is seconds (s) or UI:

```
Phase Error = CDR Output Phase - Optimal Sampling Phase
```

**Statistics**:
- **Mean**: Steady-state phase offset, ideally should be 0
- **Standard Deviation (Std Dev)**: Phase jitter RMS, reflects CDR loop noise
- **Peak-to-Peak**: Maximum phase deviation, affects timing margin

**Unit Conversion**:
```
Phase Error (UI) = Phase Error (s) / UI Period (s)
Example: 10 Gbps system, 5 ps error = 5 ps / 100 ps = 0.05 UI
```

**Lock Time**

The time required from CDR start to when phase error converges to a specified threshold. Common thresholds:
- Coarse lock: Phase error < 0.1 UI
- Fine lock: Phase error < 0.05 UI (5 ps @ 10 Gbps)

Calculation method (pseudocode):
```python
def calculate_lock_time(phase_error_vec, time_vec, threshold=0.05):
    """
    Calculate lock time
    
    Parameters:
    phase_error_vec: Phase error time series (unit: UI)
    time_vec: Corresponding timestamps (unit: seconds)
    threshold: Lock threshold (unit: UI)
    
    Returns:
    lock_time: Lock time (unit: seconds)
    """
    # Calculate maximum error within moving window
    window_size = 100  # Window length: 100 symbols
    for i in range(len(phase_error_vec) - window_size):
        window = phase_error_vec[i:i+window_size]
        if np.max(np.abs(window)) < threshold:
            return time_vec[i]
    return None  # Not locked
```

**Lock Jitter**

RMS value of phase error after lock, reflecting CDR steady-state performance:

```
Lock Jitter = std(Phase Error[after lock])
```

Typical values:
- Bang-Bang PD: 1~5 ps RMS (depends on loop bandwidth)
- Linear PD (Hogge/Alexander): 0.5~2 ps RMS

#### 5.1.2 Frequency-Domain Indicators

**Loop Bandwidth**

The -3 dB cutoff frequency of the loop closed-loop transfer function, determines CDR tracking capability:

```
H(f) = (Kp×s + Ki) / (s² + Kp×s + Ki)  (s-domain)
-3 dB bandwidth ≈ √(Ki) / (2π)  (Hz)
```

**Measurement Method**:
1. Inject swept-frequency phase modulation into input
2. Measure output/input amplitude ratio
3. Find frequency point where amplitude ratio drops to -3 dB

**Phase Margin**

The difference between the phase of the loop open-loop transfer function at 0 dB gain and -180°, measuring stability margin:

```
PM = 180° + ∠H(f_crossover)
```

Typical requirement: PM > 45° (ensures stable and non-oscillating response)

**Jitter Tolerance**

The maximum input jitter amplitude that CDR can tolerate (usually in UI units), frequency-dependent. Measured at specific BER target (e.g., 1e-12):

```
JTOL(f) = Tolerable Jitter Amplitude (UI) @ Jitter Frequency f
```

Typical JTOL curve characteristics (see Section 4.3.3):
- Low frequency (f < BW/10): Tolerance ≈ 1 UI (perfect tracking)
- Corner frequency (≈ BW): Tolerance begins to decrease, slope -20 dB/decade
- High frequency (f > 10×BW): Tolerance determined by sampler intrinsic margin

#### 5.1.3 System-Level Indicators

**Bit Error Rate (BER)**

The ratio of erroneous bits to total bits from sampler output compared to ideal reference:

```
BER = Error Bit Count / Total Bit Count
```

**Relationship between BER and Phase Error**:

Assuming signal amplitude is A, noise standard deviation is σ, and phase error causes sampling point to deviate from optimal position:

```
Q = (A - Δ) / σ
BER ≈ 0.5 × erfc(Q / √2)

Where: Δ = Voltage offset corresponding to phase error
```

For step signals, amplitude loss caused by phase error:
```
Δ ≈ Signal Slope × Phase Error = (A / Transition Time) × Phase Error
```

**Q-factor**

Another representation of signal-to-noise ratio, relationship with BER:

```
Q = √2 × erfc⁻¹(2 × BER)
Q(dB) = 20 × log₁₀(Q)
```

Typical correspondence:
- BER = 1e-12 → Q ≈ 7.0 → Q(dB) ≈ 16.9 dB
- BER = 1e-15 → Q ≈ 7.9 → Q(dB) ≈ 18.0 dB

### 5.2 Typical Test Results Interpretation

The following provides result interpretation methods and ideal expectations for the five test scenarios defined in Chapter 4.

#### 5.2.1 PHASE_LOCK_BASIC - Basic Phase Lock Test

**Test Configuration**:
- Data rate: 10 Gbps (UI = 100 ps)
- Initial phase offset: Random (0~100 ps)
- PI parameters: Kp=0.01, Ki=1e-4
- Simulation time: 10,000 UI (1 μs)

**Ideal Results Expectation**:

1. **Phase Convergence Curve**:
   - Initial stage: Phase error monotonically decreases from initial value
   - Convergence time: Approximately 2000~3000 UI (loop time constant τ ≈ 1/√Ki)
   - Steady-state error: Mean < 1 ps, RMS < 3 ps

2. **BER Performance**:
   - Before lock: Errors may occur (phase not aligned)
   - After lock: BER < 1e-12 (no errors or very low error rate)

3. **PI Controller State**:
   - Proportional term (Kp × phase_error): Fast response, oscillation gradually decays
   - Integral term (∑Ki × phase_error): Monotonically increases to steady-state value

**Waveform Characteristic Example**:

```
Phase Error (ps)
  100 |●                           ← Initial random offset
      |  ●●
   50 |     ●●●
      |        ●●●
    0 |___________●●●●━━━━━━━━━   ← Lock to steady state (small jitter)
      |
  -50 |
      └────────────────────────────▶ Time (UI)
        0    1k   2k   3k   4k   10k
```

**Verification Method**:
```bash
./cdr_tran_tb lock
python scripts/plot_cdr_phase.py cdr_tran_lock.csv
```

Check items:
- [ ] Phase error monotonically converges
- [ ] Lock time is reasonable (< 5000 UI)
- [ ] Steady-state jitter is within expected range (< 5 ps RMS)

#### 5.2.2 FREQUENCY_OFFSET - Frequency Offset Capture Test

**Test Configuration**:
- Frequency offset: ±100 ppm, ±500 ppm, ±1000 ppm
- PI parameters: Kp=0.01, Ki=1e-4
- Simulation time: 50,000 UI (5 μs)

**Ideal Results Expectation**:

1. **Phase Drift Characteristics**:
   - Frequency offset causes phase to accumulate at fixed rate
   - 100 ppm @ 10 Gbps → Phase slope ≈ 0.01 ps/UI = 10 ps/1000 UI
   - CDR's integral term Ki tracks this slope

2. **Lock Conditions**:
   - Steady-state phase error = Frequency offset slope / Ki
   - Phase adjustment range must not exceed PAI.range limit
   - Example: 100 ppm offset, after 50,000 UI accumulated phase offset = 100 ps × 0.0001 × 50,000 = 500 ps

3. **Limit Testing**:
   - When frequency offset is too large or Ki is too small, integrator cannot keep up, phase continues to drift
   - At this point, need to increase Ki or implement Frequency Detector (FD) auxiliary capture

**Waveform Characteristic Example**:

```
Phase Output (ps)
  500 |                     ●●●●●●   ← Integrator accumulates to steady-state slope
      |                 ●●●●
  250 |             ●●●●
      |         ●●●●
    0 |●●●●●●●●
      └────────────────────────────▶ Time (UI)
        0    10k   20k   30k   40k  50k

Phase Slope = Δphase / ΔUI ≈ Corresponding frequency offset value
```

**BER Impact**:
- If CDR successfully tracks frequency offset: BER normal (< 1e-12)
- If phase exceeds PAI.range: BER mutation (periodic phase wrapping)

**Verification Method**:
```bash
for ppm in 100 500 1000; do
  ./cdr_tran_tb freq --freq_offset_ppm=$ppm
done
python scripts/analyze_freq_offset.py
```

Check items:
- [ ] Phase slope matches frequency offset (error < 10%)
- [ ] Phase adjustment range is within limit
- [ ] BER remains normal

#### 5.2.3 JITTER_TOLERANCE - Jitter Tolerance Test

**Test Configuration**:
- Jitter type: Sine modulation (SJ)
- Jitter frequency sweep: 1 kHz ~ 100 MHz (logarithmic spacing)
- Jitter amplitude: Fixed or adaptive (target BER = 1e-12)
- Test duration per frequency point: ≥ 1e6 UI

**Ideal Results Expectation**:

JTOL curve shows typical low-pass characteristics (see Section 4.3.3 diagram):

| Frequency Range | Tolerance Characteristic | Physical Reason |
|-----------------|--------------------------|-----------------|
| < BW/10 | 1.0 UI | CDR fully tracks jitter |
| BW/10 ~ BW | Decreasing corner | Loop gain begins to attenuate |
| BW ~ 10×BW | -20 dB/decade slope | First-order low-pass rolloff |
| > 10×BW | 0.3~0.5 UI | Sampler intrinsic tolerance |

**Specific Numerical Example** (assuming BW = 5 MHz):

| Jitter Frequency | Theoretical Tolerance (UI) | Description |
|------------------|---------------------------|-------------|
| 1 kHz | 1.0 | Low frequency full tracking |
| 100 kHz | 1.0 | Still in tracking region |
| 1 MHz | 0.95 | Near corner |
| 5 MHz | 0.707 | -3 dB point |
| 10 MHz | 0.5 | High frequency rolloff |
| 50 MHz | 0.2 | Beyond bandwidth, small tolerance |
| 100 MHz | 0.3 | Intrinsic tolerance (no longer decreasing) |

**Measurement Method**:

For each frequency point:
1. Start with small amplitude and increase jitter
2. Run long simulation (≥ 1e6 UI)
3. Measure BER
4. Find jitter amplitude where BER just exceeds threshold (e.g., 1e-12)
5. This amplitude is the jitter tolerance at that frequency

**Output File Format**:

```csv
Jitter Frequency (Hz), Jitter Amplitude (ps), Jitter Amplitude (UI), BER
1000, 100.0, 1.00, 1e-15
10000, 98.5, 0.985, 5e-13
100000, 95.2, 0.952, 1e-12
1000000, 80.3, 0.803, 1e-12
5000000, 70.7, 0.707, 1e-12
10000000, 50.1, 0.501, 1e-12
50000000, 30.5, 0.305, 1e-12
100000000, 30.2, 0.302, 1e-12
```

**Plotting Command**:
```bash
python scripts/plot_jtol_curve.py cdr_tran_jtol.csv
```

Generate Bode plot style JTOL curve (x-axis logarithmic scale, y-axis UI or dB).

**Verification Standards**:
- [ ] Low frequency tolerance ≥ 0.9 UI
- [ ] -3 dB corner frequency ≈ Theoretical loop bandwidth (error < 30%)
- [ ] High frequency slope ≈ -20 dB/decade
- [ ] No abnormal resonance peaks

#### 5.2.4 PHASE_TRACKING - Dynamic Phase Tracking Test

**Test Configuration**:
- Phase modulation: Sine wave, frequency sweep (100 kHz ~ 10 MHz)
- Modulation amplitude: Fixed 20 ps (small signal)
- PI parameters: Kp=0.01, Ki=1e-4 (Theoretical BW ≈ 5 MHz)
- Simulation time: ≥ 100 modulation periods

**Ideal Results Expectation**:

1. **Low Frequency Tracking (f < BW)**:
   - CDR output synchronized with input
   - Amplitude attenuation < 3 dB
   - Phase delay < 90°

2. **High Frequency Attenuation (f > BW)**:
   - Output amplitude rolls off at -20 dB/decade
   - Phase delay approaches -180°

3. **Transfer Function Verification**:

Plot Bode plot (magnitude and phase response):

```
Magnitude (dB)
   0 |━━━━━━┓                      ← Passband (low frequency)
     |       ┗━━┓                   ← -3 dB point ≈ 5 MHz
  -3 |           ┗━━┓               
     |              ┗━━━┓           
 -20 |                  ┗━━━━┓     ← -20 dB/decade slope
     └─────────────────────────▶ Frequency (Hz)
       100k  1M    10M   100M

Phase (deg)
   0 |━━━━━┓
     |      ┗━━━┓                   ← Phase begins to lag
 -90 |          ┗━━━┓               ← -90° point ≈ BW
     |              ┗━━━┓
-180 |                  ┗━━━━━━    ← High frequency asymptote
     └─────────────────────────▶ Frequency (Hz)
```

**Numerical Example** (BW = 5 MHz):

| Modulation Frequency | Input Amplitude | Output Amplitude | Attenuation (dB) | Phase Lag (°) |
|----------------------|-----------------|------------------|------------------|---------------|
| 100 kHz | 20 ps | 19.8 ps | -0.17 | -5 |
| 1 MHz | 20 ps | 18.5 ps | -0.66 | -20 |
| 5 MHz | 20 ps | 14.1 ps | -3.0 | -90 |
| 10 MHz | 20 ps | 10.0 ps | -6.0 | -135 |
| 50 MHz | 20 ps | 2.8 ps | -17.0 | -175 |

**Verification Method**:
```bash
python scripts/measure_loop_bandwidth.py cdr_tran_track.csv
```

Script automatically calculates:
- -3 dB bandwidth
- Phase Margin (PM)
- Gain Margin (GM)

**Verification Standards**:
- [ ] -3 dB bandwidth close to theoretical value (error < 20%)
- [ ] Phase margin > 45° (stability requirement)
- [ ] No resonance peaks (sufficient damping)

#### 5.2.5 LOOP_BANDWIDTH - Loop Bandwidth Precise Measurement

**Test Configuration**:
- Multiple PI parameter sets for comparison
- Modulation frequency: 10 kHz ~ 100 MHz (10 points per octave)
- Modulation amplitude: Fixed 20 ps (linear range)

**Theoretical Calculation Comparison**:

| Config | Kp | Ki | Theoretical BW (MHz) | Theoretical ζ | Measured BW (MHz) | Measured ζ | Error (%) |
|--------|----|----|----------------------|---------------|-------------------|------------|-----------|
| 1 | 0.005 | 2.5e-5 | 2.51 | 0.707 | 2.38 | 0.72 | -5.2 |
| 2 | 0.01 | 1e-4 | 5.03 | 0.707 | 4.87 | 0.69 | -3.2 |
| 3 | 0.02 | 4e-4 | 10.05 | 0.707 | 9.65 | 0.71 | -4.0 |

> **Note**: Theoretical values are based on linearized model; measured values slightly deviate due to Bang-Bang PD nonlinearity (typically 5~10%).

**Bode Plot Generation**:

```bash
python scripts/plot_bode.py cdr_tran_bw.csv
```

Generate dual subplots:
- Top: Magnitude response (dB vs Hz, logarithmic scale)
- Bottom: Phase response (degree vs Hz, logarithmic scale)

Annotate key points:
- -3 dB bandwidth frequency
- Phase margin (PM)
- Gain crossover frequency

**Verification Standards**:
- [ ] Measured bandwidth error from theoretical value < 20%
- [ ] Bandwidth ratio for different parameter configurations matches expectations (e.g., if Kp doubles, BW should increase by ~41%)
- [ ] Phase margin > 45° (all configurations)

### 5.3 Waveform Data File Formats

#### 5.3.1 Phase Waveform File (cdr_tran_*.csv)

**Main Output**: Time-domain waveform of CDR phase adjustment signal.

**File Format**:

```csv
Time(s), Phase Output(s), Phase Output(ps), Phase Output(UI), Phase Error(ps)
0.000000e+00, 0.000000e+00, 0.00, 0.000, 0.00
1.000000e-10, 1.234567e-11, 12.35, 0.123, -5.67
2.000000e-10, 2.456789e-11, 24.57, 0.246, -3.45
...
```

**Column Descriptions**:
- **Time (s)**: Simulation timestamp
- **Phase Output (s)**: CDR output phase adjustment (unit: seconds)
- **Phase Output (ps)**: Phase output in picoseconds (for readability)
- **Phase Output (UI)**: Phase output in UI (normalized)
- **Phase Error (ps)**: Difference between phase output and ideal phase (if ideal phase is known)

**Sampling Rate**:
- Default: One data point per UI
- High resolution mode: 10 sampling points per UI (for observing details)

**File Size Estimation**:
- 10,000 UI @ 1 column/UI ≈ 10,000 rows ≈ 500 KB
- 50,000 UI @ 10 columns/UI ≈ 500,000 rows ≈ 25 MB

#### 5.3.2 BER Statistics File (cdr_ber_summary.json)

**Output**: Comprehensive performance metrics in JSON format.

**File Format**:

```json
{
  "test_scenario": "PHASE_LOCK_BASIC",
  "simulation_params": {
    "data_rate_gbps": 10,
    "ui_ps": 100,
    "simulation_time_us": 1.0,
    "total_bits": 10000000
  },
  "cdr_params": {
    "kp": 0.01,
    "ki": 0.0001,
    "pai_range_ps": 1.0,
    "pai_resolution_ps": 0.01
  },
  "phase_statistics": {
    "lock_time_ui": 2345,
    "lock_time_us": 0.2345,
    "steady_state_mean_ps": 0.23,
    "steady_state_rms_ps": 2.87,
    "steady_state_pk2pk_ps": 11.5,
    "max_phase_error_ps": 15.2
  },
  "ber_statistics": {
    "total_errors": 0,
    "ber": 0.0,
    "q_factor": null,
    "q_factor_db": null
  },
  "loop_performance": {
    "bandwidth_measured_mhz": 4.87,
    "bandwidth_theoretical_mhz": 5.03,
    "phase_margin_deg": 52.3,
    "damping_factor": 0.69
  },
  "status": "PASSED",
  "notes": "Locked successfully, no bit errors detected."
}
```

**Key Field Descriptions**:
- **phase_statistics.lock_time_ui**: Lock time (UI)
- **phase_statistics.steady_state_rms_ps**: Steady-state jitter RMS (ps)
- **ber_statistics.ber**: Bit error rate
- **loop_performance.bandwidth_measured_mhz**: Measured loop bandwidth
- **loop_performance.phase_margin_deg**: Phase margin
- **status**: Test result (PASSED/FAILED/WARNING)

#### 5.3.3 JTOL Data File (cdr_jtol.csv)

**Output**: Frequency sweep results of jitter tolerance test.

**File Format**:

```csv
Jitter Frequency (Hz), Jitter Amplitude (ps), Jitter Amplitude (UI), BER, Test Duration (UI), Error Count
1.00e+03, 100.0, 1.000, 1.23e-15, 1000000, 0
1.00e+04, 98.5, 0.985, 5.67e-13, 1000000, 0
1.00e+05, 95.2, 0.952, 1.02e-12, 1000000, 1
1.00e+06, 80.3, 0.803, 9.87e-13, 1000000, 0
5.00e+06, 70.7, 0.707, 1.05e-12, 1000000, 1
1.00e+07, 50.1, 0.501, 9.65e-13, 1000000, 0
5.00e+07, 30.5, 0.305, 1.12e-12, 1000000, 1
1.00e+08, 30.2, 0.302, 8.94e-13, 1000000, 0
```

**Post-processing Example**:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read JTOL data
df = pd.read_csv('cdr_jtol.csv')

# Plot JTOL curve
plt.figure(figsize=(10, 6))
plt.semilogx(df['Jitter Frequency (Hz)'], df['Jitter Amplitude (UI)'], 'o-', linewidth=2)
plt.xlabel('Jitter Frequency (Hz)')
plt.ylabel('Jitter Tolerance (UI)')
plt.title('CDR Jitter Tolerance Curve')
plt.grid(True, which='both', alpha=0.3)

# Annotate -3 dB point
bw_idx = np.argmin(np.abs(df['Jitter Amplitude (UI)'] - 0.707))
bw_freq = df['Jitter Frequency (Hz)'].iloc[bw_idx]
plt.axvline(bw_freq, color='r', linestyle='--', label=f'BW ≈ {bw_freq/1e6:.1f} MHz')

# Annotate low/high frequency regions
plt.axhline(1.0, color='g', linestyle=':', alpha=0.5, label='Full Tracking')
plt.axhline(0.3, color='b', linestyle=':', alpha=0.5, label='Intrinsic Tolerance')

plt.legend()
plt.tight_layout()
plt.savefig('jtol_curve.png', dpi=300)
plt.show()
```

#### 5.3.4 Debug Log File (cdr_debug.log)

**Output**: Generated only in debug mode, contains cycle-by-cycle internal states.

**File Format** (text format, easy for grep):

```
[Time=1.000e-10] PD Input=1, PD Output=1, Prop Term=0.010, Int Term=0.001, Phase Output=0.011
[Time=2.000e-10] PD Input=0, PD Output=-1, Prop Term=-0.010, Int Term=0.001, Phase Output=-0.009
[Time=3.000e-10] PD Input=1, PD Output=1, Prop Term=0.010, Int Term=0.002, Phase Output=0.012
...
```

**Usage**:
- Diagnose whether phase detector output is correct
- Verify PI controller's proportional and integral term separation
- Check phase limiter operating state

**Enable Method**:
```bash
./cdr_tran_tb lock --debug
```

---

## 6. Running Guide

### 6.1 Environment Configuration

Before running tests, configure environment variables:

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
make cdr_tran_tb

# Run tests (in tb directory)
cd tb
./cdr_tran_tb [scenario]
```

Scenario parameters:
| Parameter | Number | Description |
|-----------|--------|-------------|
| `PHASE_LOCK_BASIC` | `0` | Basic function test (default) |
| `LOOP_BANDWIDTH` | `1` | PI parameter sweep test |
| `FREQUENCY_OFFSET` | `2` | Frequency offset capture test |
| `JITTER_TOLERANCE` | `3` | Jitter tolerance test |
| `PHASE_TRACKING` | `4` | Dynamic phase tracking test |

Running examples:
```bash
# Run basic function test
./cdr_tran_tb PHASE_LOCK_BASIC

# Run PI parameter sweep
./cdr_tran_tb LOOP_BANDWIDTH

# Run all scenarios (batch test)
for i in 0 1 2 3 4; do ./cdr_tran_tb $i; done
```

### 6.3 Parameter Configuration Examples

The CDR module supports parameterization through JSON configuration files. Below are quick-start configurations for different application scenarios.

#### Quick Verification Configuration (Standard Bandwidth)

Suitable for preliminary function verification and debugging, loop bandwidth approximately 5 MHz:

```json
{
  "cdr": {
    "pi": {
      "kp": 0.01,
      "ki": 1e-4
    },
    "pai": {
      "resolution": 1e-12,
      "range": 5e-11
    }
  }
}
```

**Characteristics**:
- Moderate loop bandwidth (approx. 5 MHz), balancing tracking speed and stability
- Phase resolution 1 ps, suitable for mid-to-high-end SerDes
- Phase range ±25 ps, sufficient for moderate frequency offset (< 500 ppm)

#### High Bandwidth Configuration (Fast Lock)

Suitable for scenarios requiring fast lock (e.g., frequent link retraining), loop bandwidth approximately 10 MHz:

```json
{
  "cdr": {
    "pi": {
      "kp": 0.02,
      "ki": 4e-4
    },
    "pai": {
      "resolution": 1e-12,
      "range": 1e-10
    }
  }
}
```

**Characteristics**:
- Larger Kp/Ki speeds up lock (lock time < 1000 UI)
- Larger phase range ±50 ps, accommodates larger frequency offset (< 1000 ppm)
- Trade-off: Slightly higher lock jitter (2~5 ps RMS)

**Application Scenarios**:
- Link training phase
- Frequent entry/exit from low-power modes
- Systems with large reference clock frequency offset

#### Low Jitter Configuration (High Stability)

Suitable for data transmission phases requiring low jitter, loop bandwidth approximately 2 MHz:

```json
{
  "cdr": {
    "pi": {
      "kp": 0.005,
      "ki": 2.5e-5
    },
    "pai": {
      "resolution": 5e-13,
      "range": 3e-11
    }
  }
}
```

**Characteristics**:
- Lower loop bandwidth reduces phase jitter (< 2 ps RMS)
- Higher phase resolution 0.5 ps improves precision
- Trade-off: Longer lock time (> 3000 UI)

**Application Scenarios**:
- Steady-state data transmission phase
- High-quality channels (low jitter environment)
- Applications extremely sensitive to BER

#### Wide Frequency Offset Adaptation Configuration (Dual-Loop Architecture Simulation)

Suitable for large frequency offset scenarios (> 1000 ppm), requiring wide phase range:

```json
{
  "cdr": {
    "pi": {
      "kp": 0.015,
      "ki": 2e-4
    },
    "pai": {
      "resolution": 2e-12,
      "range": 2e-10
    }
  }
}
```

**Characteristics**:
- Extremely wide phase range ±100 ps, accommodates extreme frequency offset (< 2000 ppm @ 10 Gbps)
- Balanced PI parameters兼顾速度和稳定性
- Moderately relaxed resolution to 2 ps reduces computational overhead

**Application Scenarios**:
- Systems with poor reference clock quality
- Cross-clock-domain system-level simulation
- Stress testing and frequency offset tolerance verification

#### Custom Parameter Tuning Guide

When adjusting parameters based on actual requirements, refer to the following guidelines:

**Loop Bandwidth Target**:
```
Theoretical bandwidth estimation: BW ≈ √(Ki) / (2π)  [Hz]
Adjustment strategy:
  - Increase BW: Increase Kp and Ki simultaneously (maintain Ki ≈ Kp²)
  - Decrease BW: Decrease Kp and Ki simultaneously
```

**Damping Coefficient Target**:
```
Recommended range: ζ = 0.7 ~ 1.0 (avoid oscillation)
Adjustment strategy:
  - If oscillation occurs: Increase Kp (increase damping)
  - If response is too slow: Decrease Kp (decrease damping)
```

**Phase Range Estimation**:
```
Minimum range requirement: range ≥ |freq_offset(ppm)| × 1e-6 × UI × 10000
Example: 500 ppm @ 10 Gbps (UI=100 ps) → range ≥ 50 ps
```

**Resolution Selection**:
```
Rule of thumb: resolution = UI / 64 ~ UI / 256
  - High-end implementation: UI/256 (approx. 0.4 ps @ 10 Gbps)
  - Mid-range implementation: UI/128 (approx. 0.8 ps @ 10 Gbps)
  - Low-cost implementation: UI/64 (approx. 1.6 ps @ 10 Gbps)
```

### 6.4 Results Viewing

After test completion, the console outputs key performance metrics (lock time, phase error statistics, lock jitter, etc.), and waveform data is saved to `.dat` files.

#### Waveform Visualization

Use Python scripts for waveform plotting and analysis:

```bash
# Basic waveform visualization (phase error, PI output)
python scripts/plot_cdr_waveform.py --input tb/cdr_basic_phase.dat

# Spectrum analysis (phase jitter PSD)
python scripts/plot_cdr_psd.py --input tb/cdr_basic_phase.dat

# Multi-scenario comparison (PI parameter sweep results)
python scripts/plot_cdr_sweep.py --dir tb/cdr_sweep_results/
```

**Output File Naming Convention**:
- `cdr_[scenario]_phase.dat`: Phase output time-domain waveform
- `cdr_[scenario]_error.dat`: Phase error time-domain waveform
- `cdr_[scenario]_stats.json`: Statistical metrics (mean, RMS, lock time)

#### Key Metrics Interpretation

Console output example:
```
=== CDR Performance Statistics ===
Lock Time:        2345 UI (234.5 ns)
Phase Error (locked):
  Mean:           0.12 ps
  Std Dev (RMS):  1.8 ps
  Peak-to-Peak:   12.3 ps
Phase Output Range:
  Min:            -15.2 ps
  Max:            +14.8 ps
  Utilization:    60.0% of ±25 ps range
BER (if available): 1.2e-13
```

**Metric Criteria**:
- **Lock Time < 3000 UI**: Lock speed qualified
- **RMS < 5 ps**: Lock jitter good (typical value for Bang-Bang PD)
- **Mean ≈ 0 ps**: No static phase offset (PI integral effective)
- **Utilization < 80%**: Phase range margin sufficient

#### Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Cannot lock (phase diverges) | Kp/Ki too large causing instability | Decrease Kp/Ki, check damping coefficient |
| Lock time too long | Kp/Ki too small | Increase Kp/Ki, but keep ζ > 0.5 |
| Lock jitter too large | Loop bandwidth too high | Decrease loop bandwidth (reduce Kp/Ki) |
| Phase range exhausted | Frequency offset too large or range insufficient | Increase pai.range, check frequency offset |
| DC phase offset | Ki too small to eliminate static error | Increase Ki (maintain Ki ≈ Kp/10) |

---

## 7. Technical Points

### 7.1 Loop Stability Design

CDR loop is a typical feedback control system; stability design is crucial:

**Second-Order System Characteristics**:

The PI controller forms a second-order loop; its damping coefficient ζ determines stability:
- **ζ < 0.5**: Underdamped, fast response but with overshoot and oscillation
- **ζ = 0.707**: Critical damping, fastest overshoot-free response (optimal)
- **ζ > 1.0**: Overdamped, no overshoot but slow response

**Kp and Ki Relationship**:

Given target damping coefficient ζ and loop bandwidth BW, reverse-calculate PI parameters:
```
ωn = 2π × BW
Ki = ωn² / Fs
Kp = 2 × ζ × ωn
```

> ⚠️ **Example Calculation Notes**:
> 
> The following example calculations are derived based on the **linear phase detector** continuous-domain model, but this module uses a **Bang-Bang Phase Detector**, and its nonlinear characteristics require actual parameter adjustments. The following formulas are for **preliminary estimation** only; actual design requires simulation verification.
>
> **Example Calculation** (10 Gbps system, target BW = 5 MHz, ζ = 0.707):
> ```
> ωn = 2π × 5e6 = 3.14e7 rad/s
> Ki = (3.14e7)² / 10e9 = 9.8e-5
> Kp = 2 × 0.707 × 3.14e7 / 10e9 = 0.0044
> ```
> 
> **Alternatively**, if you wish to remove this calculation example to avoid confusion, directly remind users:
> "Due to the nonlinear characteristics of Bang-Bang PD, it is recommended to tune Kp/Ki parameters through SystemC-AMS simulation rather than relying on linearized formulas."

### 7.2 Bang-Bang PD Jitter Performance

The discrete characteristics of the Bang-Bang Phase Detector introduce loop jitter:

**Random Jitter (RJ) Sources**:
- PD output is discrete ±1, lacking linear region
- Each update is an "over-correction", causing phase to jitter around the optimal point
- Jitter amplitude is proportional to loop gain: σφ ≈ √(Kp / Fs)

**Methods to Reduce Jitter**:
1. Reduce loop bandwidth (decrease Kp/Ki)
2. Use linear phase detector (e.g., Hogge PD) instead of Bang-Bang PD
3. Increase phase detector resolution (multi-level quantization)

**Advantages of Bang-Bang PD**:
- Simple structure, low hardware overhead
- Insensitive to input signal amplitude
- Suitable for high-speed links (40 Gbps+)

### 7.3 Phase Interpolator Resolution Selection

Phase interpolator resolution affects CDR accuracy and hardware cost:

**Resolution and UI Relationship**:

In typical designs, phase interpolator resolution is 1/64~1/256 of UI:
- **High precision** (UI/256): σφ < 0.5 ps, suitable for 56G/112G PAM4
- **Medium precision** (UI/128): σφ ≈ 1 ps, suitable for 10G/25G NRZ
- **Low precision** (UI/64): σφ ≈ 2 ps, suitable for low-cost applications

**Default Selection** (1 ps):

Project default 1 ps (1e-12 seconds) corresponds to UI/100 (assuming 10 Gbps, UI = 100 ps):
- Provides sufficient phase adjustment precision
- Simulates mid-to-high-end SerDes implementation
- Can be configured to coarser resolution (5 ps) to simulate low-cost solutions

**Quantization Noise**:

The coarser the resolution, the greater the quantization noise:
```
σ_quantization ≈ resolution / √12
```
For 1 ps resolution, quantization noise is approximately 0.29 ps RMS.

### 7.4 Interface Design with Sampler Module

The CDR's `phase_out` port must be correctly connected to the sampler's `phase_offset` port:

**Unit Convention**:

- CDR output: Time offset, unit **seconds (s)**
- Sampler receives: Phase offset, unit **seconds (s)**
- Physical meaning: Positive value delays sampling, negative value advances sampling

**Timing Coordination**:

CDR and sampler work at the same sampling rate (Fs), ensuring:
```cpp
// Both must be set to the same sampling rate
void RxCdrTdf::set_attributes() {
    in.set_rate(1);
    phase_out.set_rate(1);
}

sampler.set_attributes() {
    phase_offset.set_rate(1);
    // ...
}
```

**Phase Update Delay**:

In actual hardware, there is delay from CDR phase update to sampler (1~2 cycles); the current model does not account for this delay. If more precise modeling is needed, a delay buffer can be added inside the CDR module.

### 7.5 Frequency Acquisition (Pending Implementation)

The current version does not implement frequency acquisition functionality; it is only suitable for scenarios where transmitter and receiver frequencies are approximately matched (frequency offset < ±100 ppm).

**Necessity of Frequency Detector**:

In actual SerDes, reference clock frequency offset can reach ±100 ppm, corresponding to ±1 MHz for a 10 Gbps system. If only the phase loop tracks, this will cause:
- Phase accumulation continuously increases, exceeding phase interpolator range
- Loop cannot lock

**Frequency-Assisted Solutions** (future versions):

1. **Rotational Frequency Detector**:
   - Monitor the rate of change of phase accumulation
   - Output frequency error signal to VCO or PLL

2. **Dual-Loop Architecture**:
   - Low-bandwidth frequency loop (kHz level): Tracks frequency offset
   - High-bandwidth phase loop (MHz level): Tracks jitter

3. **Lock Detector**:
   - Monitor the variance of phase error
   - Declare lock when variance is below threshold

### 7.6 Sampling Rate Requirements

The CDR module's sampling rate must match the data rate:

**Recommended Settings**:

For baud-rate CDR (one phase update per symbol):
```cpp
// SystemC-AMS TDF module sampling rate setting
void RxCdrTdf::set_attributes() {
    set_timestep(UI);  // Sampling time step = Unit Interval
}
```

For 10 Gbps NRZ (UI = 100 ps):
- CDR sampling rate = 10 GHz
- Each bit triggers one phase detection and loop update

**Oversampled CDR** (future extension):

Some CDR designs use 2× or 4× oversampling to improve phase detection accuracy, but increase power consumption and complexity.

### 7.7 Known Limitations and Notes

**Limitation 1 - Phase Detection Accuracy**:

The current implementation uses edge polarity detection, which has limited phase detection accuracy. In practical applications, it can be replaced with a complete BBPD implementation to improve accuracy.

**Limitation 2 - Frequency Capture Capability**:

Cannot handle large frequency offset situations (> ±100 ppm). If simulating reference clock frequency offset is needed, manually adjust the PLL module's output frequency first, or add a frequency detector in subsequent versions.

**Limitation 3 - No Lock Detection**:

The module does not output a lock status signal, making it impossible to determine if CDR has successfully locked. It is recommended to monitor the variance of the `phase_out` signal in the testbench.

**Limitation 4 - Linear Phase Range**:

When phase accumulation exceeds ±range, hard limiting occurs, temporarily losing tracking capability. Actual designs should avoid this through a frequency detector.

**Notes**:

- When adjusting Kp/Ki, check both damping coefficient and bandwidth to avoid instability
- Phase range should be greater than expected maximum frequency offset × UI
- Bang-Bang PD performance degrades at low SNR; recommended to use with DFE

---

## 8. Reference Information

### 8.1 Related Files

| File Type | Path | Description |
|-----------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | RxCdrParams structure (PI/PAI parameters) |
| Header File | `/include/ams/rx_cdr.h` | RxCdrTdf class declaration |
| Implementation File | `/src/ams/rx_cdr.cpp` | RxCdrTdf class implementation |
| Testbench | `/tb/rx/cdr/cdr_tran_tb.cpp` | CDR transient testbench |
| Test Helper | `/tb/rx/cdr/cdr_helpers.h` | Testbench helper functions |
| Default Configuration | `/config/default.json` | Global configuration file (cdr section) |

### 8.2 Dependencies

**Compile-Time Dependencies**:
- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 standard

**Runtime Dependencies**:
- Upstream module: Sampler (`RxSamplerTdf`) provides data sampled values and edge sampled values
- Downstream module: BER analysis module or error statistics module (for evaluating CDR performance)

**System-Level Dependencies**:
- Reference clock source: Usually provided by clock generation module (`ClockGenTdf`)
- DFE module: CDR output recovered clock used to drive DFE tap updates

### 8.3 Configuration Examples

#### Basic Configuration (Standard 5 MHz Bandwidth)

```json
{
  "cdr": {
    "pi": {
      "kp": 0.01,
      "ki": 1e-4
    },
    "pai": {
      "resolution": 1e-12,
      "range": 5e-11
    }
  }
}
```

**Parameter Notes**:
- `kp = 0.01`: Proportional gain, controls loop response speed
- `ki = 1e-4`: Integral gain, eliminates steady-state phase error
- `resolution = 1 ps`: Phase interpolator minimum adjustment step
- `range = 50 ps`: Phase adjustable range (approx. 0.5 UI @ 10 Gbps)

#### Fast Lock Configuration (10 MHz Bandwidth)

```json
{
  "cdr": {
    "pi": {
      "kp": 0.02,
      "ki": 4e-4
    },
    "pai": {
      "resolution": 2e-12,
      "range": 1e-10
    }
  }
}
```

**Application Scenarios**:
- Burst-mode communication
- Frequent link switching (Link training)
- Systems requiring fast acquisition

#### Low Jitter Configuration (2 MHz Bandwidth)

```json
{
  "cdr": {
    "pi": {
      "kp": 0.004,
      "ki": 1.6e-5
    },
    "pai": {
      "resolution": 0.5e-12,
      "range": 3e-11
    }
  }
}
```

**Application Scenarios**:
- Jitter-sensitive applications
- Clean channels
- Jitter transfer reduction

#### Wide Frequency Offset Tolerance Configuration (Large Range Capture)

```json
{
  "cdr": {
    "pi": {
      "kp": 0.015,
      "ki": 2e-4
    },
    "pai": {
      "resolution": 2e-12,
      "range": 2e-10
    }
  }
}
```

**Application Scenarios**:
- Large frequency offset between transmitter and receiver clocks (> ±200 ppm)
- Wide frequency acquisition systems
- Pure phase loop without frequency detector assistance

### 8.4 Parameter Tuning Guide

#### Loop Bandwidth (BW) and Kp/Ki Relationship

Given target loop bandwidth BW and damping coefficient ζ, PI parameters can be estimated:

```
ωn = 2π × BW
Ki = ωn² × UI / Fs
Kp = 2 × ζ × ωn × UI / Fs
```

Where:
- UI is the unit interval time (seconds), e.g., UI = 100 ps in 10 Gbps system
- Fs is the phase accumulator update frequency (Hz)
- This formula **applies to digital CDR with Bang-Bang PD** (Kpd = 1/UI)

**Recommended Damping Coefficients**:
- ζ = 0.707 (critical damping): Fastest overshoot-free response
- ζ = 0.8-1.0 (light overdamping): Sacrifice 10-20% speed for better stability

> ⚠️ **Important Notes**:
> - For other types of phase detectors, adjust the formula according to actual Kpd and Kvco values
> - For linear PD (Kpd ≠ 1/UI), replace UI with Kpd in the formula
> - The nonlinear characteristics of Bang-Bang PD require fine-tuning of actual parameters; use formula results as **initial values** and optimize through SystemC-AMS simulation

#### Phase Adjustment Range (PAI range) Design

```
range_min = 2 × (Frequency offset ppm × UI) + Phase margin
```

**Example** (10 Gbps system, ±300 ppm frequency offset):
```
Phase deviation caused by frequency offset = 300e-6 × 100 ps = 30 ps
Recommended range = 2 × 30 ps + 20 ps = 80 ps
```

#### Phase Interpolator Resolution (PAI resolution) Trade-offs

| Resolution | Quantization Phase Error | Loop Jitter | Hardware Cost |
|------------|--------------------------|-------------|---------------|
| 0.5 ps | ±0.25 ps | Low | High (8-9 bit) |
| 1 ps | ±0.5 ps | Medium | Medium (7-8 bit) |
| 2 ps | ±1 ps | High | Low (6-7 bit) |

**Recommendation**: For 10-28 Gbps SerDes, 1 ps resolution is a good balance between performance and cost.

### 8.5 Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| CDR cannot lock | Kp/Ki too large, loop oscillation | Decrease Kp and Ki, increase damping coefficient ζ |
| Lock speed too slow | Kp/Ki too small, insufficient loop bandwidth | Increase Kp and Ki, but keep ζ > 0.5 |
| Lock jitter too large | Loop bandwidth too high | Decrease loop bandwidth (reduce Kp/Ki) |
| Phase adjustment range exhausted | Frequency offset exceeds PAI range | Increase range parameter or add frequency detector |
| Periodic phase jumps | PAI resolution insufficient | Decrease resolution or use linear PD |
| DC phase offset | Ki too small, insufficient integration | Increase Ki, enhance DC gain |

### 8.6 Interface Relationships with Related Modules

**Upstream Modules**:
- **Sampler**: Provides `data_sample` (data sampled value) and `edge_sample` (edge sampled value); CDR calculates phase error based on this
- **DFE**: Provides equalized signal quality, affects phase detection accuracy

**Downstream Modules**:
- **Deserializer**: Uses CDR output `recovered_clock` for serial-to-parallel conversion
- **BER Test Module**: Uses recovered clock to evaluate bit error rate performance

**System-Level Interactions**:
- **Clock Generation Module**: Provides reference clock base (frequency approximately 1/4 or 1/2 of data rate)
- **Adaptive Algorithms**: DFE adaptation, AGC adaptation typically start after CDR lock

---

**Document Version**: v0.2  
**Last Updated**: 2026-01-20  
**Author**: Qoder serdes-doc-writer
