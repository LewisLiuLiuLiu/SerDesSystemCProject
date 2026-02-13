# Clock Generator Module Technical Documentation

🌐 **Languages**: [中文](../../modules/clkGen.md) | [English](clkGen.md)

**Level**: AMS Top-Level Module  
**Class Name**: `ClockGenerationTdf`  
**Current Version**: v0.1 (2026-01-20)  
**Status**: In Development

---

## 1. Overview

The Clock Generator is the core clock source module of the SerDes system, responsible for providing stable clock phase signals to the Transmitter (TX), Receiver (RX), and Clock Data Recovery (CDR) circuits. The module supports multiple clock generation modes, including Ideal Clock, Analog Phase-Locked Loop (PLL), and All-Digital Phase-Locked Loop (ADPLL).

### 1.1 Design Principles

The core design philosophy of the Clock Generator is to provide accurate phase information for driving timing control circuits in the SerDes link:

- **Phase Continuity**: Phase values vary continuously in the 0 to 2π range, achieving periodic cycling through modulo 2π operation
- **Adaptive Time Step**: Sampling time steps are dynamically adjusted based on clock frequency, ensuring sufficient sampling points per cycle
- **Phase Accumulation Mechanism**: Uses a phase accumulator to generate clock phase, with phase increment of `Δφ = 2π × f × Δt` per time step

The mathematical form of the phase output is:
```
φ(t) = 2π × f × t (mod 2π)
```
where `f` is the clock frequency, `t` is the simulation time, and `mod` represents the modulo operation.

**Current Implementation**: The module currently implements the ideal clock generation mode, directly outputting linearly increasing phase values. Future plans include extending support for PLL and ADPLL modes to enable more realistic clock noise and jitter modeling.

### 1.2 Core Features

- **Multiple Clock Types**: Supports Ideal Clock (IDEAL), Analog PLL (PLL), and All-Digital PLL (ADPLL)
- **Phase Output Interface**: Outputs continuous phase values (0~2π) for use by downstream modules
- **Adaptive Sampling Rate**: Time step is automatically adjusted based on clock frequency (default is 100× the frequency)
- **PLL Parameterized Configuration**: Supports complete configuration of phase detector type, charge pump current, loop filter, VCO parameters, and divider ratio
- **Flexible Frequency Settings**: Supports arbitrary frequency configuration (e.g., 40GHz for high-speed SerDes)

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2026-01-20 | Initial version, implemented ideal clock generation and PLL parameter structure definitions |

---

## 2. Module Interface

### 2.1 Port Definitions (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `clk_phase` | Output | double | Clock phase output (radians, range 0~2π) |

The phase output is a continuous-time signal, with the instantaneous phase value of the current clock output at each time step. Downstream modules (such as Sampler, CDR) can calculate sampling moments or perform phase adjustments based on the phase information.

### 2.2 Parameter Configuration (ClockParams)

#### Basic Parameters

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `type` | ClockType | PLL | Clock generation type (IDEAL/PLL/ADPLL) |
| `frequency` | double | 40e9 | Clock frequency (Hz) |

**Clock Type Descriptions**:
- `IDEAL`: Ideal clock, no jitter, no noise, phase grows strictly linearly
- `PLL`: Analog phase-locked loop, supports complete chain modeling of PD/CP/LF/VCO/Divider
- `ADPLL`: All-digital phase-locked loop, digital phase detector, TDC, DCO and other digital domain implementations

#### PLL Substructure (ClockPllParams)

Analog phase-locked loop parameters, used for implementing real clock generation circuit modeling.

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `pd` | string | "tri-state" | Phase detector type |
| `cp.I` | double | 5e-5 | Charge pump current (A) |
| `lf.R` | double | 10000 | Loop filter resistance (Ω) |
| `lf.C` | double | 1e-10 | Loop filter capacitance (F) |
| `vco.Kvco` | double | 1e8 | VCO gain (Hz/V) |
| `vco.f0` | double | 1e10 | VCO center frequency (Hz) |
| `divider` | int | 4 | Feedback divider ratio |

**Working Principle**:
The PLL adopts a typical second-order loop structure, consisting of the following sub-modules:

1. **Phase Detector (PD)**: Compares the phase difference between the reference clock and the feedback clock, outputs an error signal
   - `tri-state`: Tri-state phase detector, outputs UP/DOWN pulses
   - Other types (future extensions): `bang-bang`, `linear`, `hogge`

2. **Charge Pump (CP)**: Charges/discharges the loop filter based on the phase detector output
   - Current is controlled by the `cp_current` parameter
   - Charge/discharge current magnitude determines the loop bandwidth

3. **Loop Filter (LF)**: Converts the charge pump output current to a control voltage
   - Uses a passive RC low-pass filter structure
   - Time constant τ = R × C, determines loop bandwidth and stability
   - Transfer function: `Z(s) = R / (1 + s × R × C)`

4. **Voltage-Controlled Oscillator (VCO)**: Generates output frequency based on the control voltage
   - Output frequency: `f_out = f0 + Kvco × Vctrl`
   - `Kvco` is the VCO gain, representing the frequency change per volt of voltage change

5. **Divider**: Divides the VCO output and feeds it back to the phase detector
   - Divider ratio `divider` determines the relationship between output frequency and reference frequency
   - Output frequency = Reference frequency × divider

**PLL Closed-Loop Characteristics**:
- Loop bandwidth: `ωn = sqrt(Kvco × Icp / (N × C))`, where N is the divider ratio
- Damping coefficient: `ζ = (R/2) × sqrt(Icp × Kvco × C / N)`
- Lock time: approximately 4/(ζ × ωn)

**Current Implementation Status**:
- PLL parameter structure is fully defined
- Current `processing()` method only implements ideal clock generation
- Future versions will implement complete PLL dynamic behavior modeling

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The `processing()` method of the Clock Generator module adopts a concise phase accumulator architecture, executing the following operations at each time step:

```
Output current phase → Calculate phase increment → Update accumulator → Phase normalization (mod 2π)
```

**Step 1 - Output Current Phase**: Write the currently accumulated phase value `m_phase` to the output port `clk_phase`. The phase value ranges from [0, 2π), representing the instantaneous phase position in the current clock cycle.

**Step 2 - Calculate Phase Increment**: Calculate the phase increment based on the configured clock frequency and current time step:
```
Δφ = 2π × m_frequency × Δt
```
where `Δt` is the time step, obtained through `clk_phase.get_timestep().to_seconds()`.

**Step 3 - Update Accumulator**: Accumulate the phase increment to the internal phase accumulator:
```
m_phase += Δφ
```
This accumulation mechanism ensures phase continuity and accuracy.

**Step 4 - Phase Normalization**: When the accumulated phase value reaches or exceeds 2π, perform modulo operation to normalize the phase to the [0, 2π) range:
```cpp
if (m_phase >= 2.0 * M_PI) {
    m_phase -= 2.0 * M_PI;
}
```
This simulates the periodic characteristic of the clock, with each 2π corresponding to a complete clock cycle.

### 3.2 Adaptive Time Step Setting

The module implements adaptive time step setting in the `set_attributes()` method, ensuring the sampling rate matches the clock frequency:

```cpp
clk_phase.set_timestep(1.0 / (m_frequency * 100.0), sc_core::SC_SEC);
```

**Design Principles**:
- Sampling rate = Clock frequency × 100
- 100 sampling points per clock cycle
- Time step = 1 / Sampling rate

**Examples**:
- For 40GHz clock: Time step = 1 / (40e9 × 100) = 0.25ps
- For 10GHz clock: Time step = 1 / (10e9 × 100) = 1ps

This adaptive mechanism ensures:
1. **Sufficient Time Domain Resolution**: Enough sampling points per cycle to accurately represent phase changes
2. **Reasonable Simulation Efficiency**: Sampling rate proportional to frequency, avoiding unnecessary computational overhead for low-frequency clocks
3. **Nyquist Criterion Satisfaction**: Sampling rate is much higher than signal frequency (100×), avoiding aliasing

### 3.3 Phase Accumulator Design Philosophy

Reasons for using a phase accumulator instead of directly calculating `φ = 2π × f × t`:

**Accuracy Advantages**:
- Direct calculation relies on simulation time `t`, which may have floating-point cumulative errors
- Phase accumulator uses incremental updates, with independent error at each step, avoiding cumulative errors
- Modulo 2π operation keeps values within a reasonable range, improving numerical stability

**Flexibility Advantages**:
- Phase accumulator is easily extensible to support Frequency Modulation (FM)
- Can conveniently inject phase noise and jitter
- When implementing PLL in the future, the phase increment can be directly modified without reconstructing the entire calculation chain

**Implementation Simplicity**:
- Clear code logic, easy to understand and maintain
- Low computational overhead, requiring only one addition and one conditional judgment per time step
- Suitable for SystemC-AMS TDF domain semantics

**Future Extensibility**:
The current implementation is an ideal clock with strictly linear phase growth. When extending to PLL mode in the future, the phase increment will be dynamically controlled by the loop filter output:
```
Δφ = 2π × (f0 + Kvco × Vctrl) × Δt
```
where `Vctrl` is the loop filter output, driven by the phase detector and charge pump.

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The Clock Generator module currently does not have an independent testbench and is primarily verified as a clock source in system-level tests. Core design concepts:

1. **System Integration Verification**: The Clock Generator serves as the clock source for the SerDes link, with its functional correctness verified through complete link simulation
2. **Phase Continuity Verification**: The continuity and accuracy of phase output are indirectly verified through the behavior of downstream modules (such as Sampler, CDR)
3. **Configuration Flexibility Verification**: The correctness of the time step adaptive mechanism is tested through different frequency configurations

**Future Expansion Plans**:
- Design an independent Clock Generator testbench, supporting multiple test scenarios
- Implement specialized tests for PLL and ADPLL modes
- Add phase noise and jitter injection tests

### 4.2 Test Scenario Definitions

The Clock Generator functionality is currently verified through system-level testbenches:

| Scenario | Command Line Parameters | Test Objectives | Output File | Verification Method |
|----------|------------------------|-----------------|-------------|---------------------|
| System-level Integration Test | None | Verify Clock Generator functionality in complete SerDes link | simple_link.dat | Run simple_link_tb, check that the link operates normally |
| Frequency Configuration Test | None | Verify time step adaptation under different frequency configurations | simple_link.dat | Modify clock frequency in configuration file, observe time step changes |
| Phase Continuity Test | None | Verify continuity and periodicity of phase output | simple_link.dat | Analyze sampling moments of downstream modules, verify phase continuity |

**Planned Future Scenarios**:
- Ideal clock basic test
- PLL lock time test
- PLL loop bandwidth test
- Phase noise injection test
- Jitter tolerance test

### 4.3 Scenario Configuration Details

#### System-level Integration Test

Verify the Clock Generator functionality in the complete SerDes link.

- **Testbench**: `simple_link_tb.cpp`
- **Configuration File**: `config/default.yaml`
- **Clock Configuration**: 40GHz ideal clock
- **Verification Points**:
  - Link can start and run normally
  - Downstream modules (Sampler, CDR) can receive clock phase normally
  - Simulation time step matches clock frequency (0.25ps)

#### Frequency Configuration Test

Verify the correctness of the time step adaptive mechanism.

- **Test Method**: Modify `clock.frequency` in `config/default.yaml`
- **Test Frequencies**: 10GHz, 20GHz, 40GHz, 80GHz
- **Verification Points**:
  - 10GHz: Time step should be 1ps
  - 20GHz: Time step should be 0.5ps
  - 40GHz: Time step should be 0.25ps
  - 80GHz: Time step should be 0.125ps

#### Phase Continuity Test

Verify the continuity and periodicity of phase output.

- **Test Method**: Analyze sampling moments of downstream modules
- **Verification Points**:
  - Phase values vary continuously within [0, 2π) range
  - Each complete cycle phase value grows from 0 to 2π
  - Phase increment is constant (ideal clock)

### 4.4 Signal Connection Topology

Typical connection relationships of the Clock Generator in system-level testing:

```
┌─────────────────┐
│ ClockGenerator  │
│                 │
│  clk_phase ─────┼───────▶ Sampler (sampling moments)
│                 │
└─────────────────┘
```

**Description**:
- Clock Generator outputs phase signal to the Sampler module
- Sampler determines sampling moments based on phase information
- In the current implementation, the CDR module uses internal phase generation logic and does not use ClockGenerator

### 4.5 Auxiliary Module Descriptions

#### Related Modules in Current System-level Testing

**WaveGenerationTdf (Waveform Generator)**:
- Function: Generates PRBS test signals
- Relationship with Clock Generator: Both together constitute the signal source for the SerDes link

**RxSamplerTdf (Receiver Sampler)**:
- Function: Samples input signals based on clock phase
- Relationship with Clock Generator: Receives clock phase, used to determine sampling moments

**RxCdrTdf (Clock Data Recovery)**:
- Function: Recovers data clock
- Current Implementation: Uses internal phase generation logic, does not use ClockGenerator

**Future Independent Testbench Plans**:

**PhaseMonitor (Phase Monitor)**:
- Function: Real-time recording of clock phase output
- Output: Phase vs time curve
- Purpose: Verify phase continuity and periodicity

**JitterAnalyzer (Jitter Analyzer)**:
- Function: Analyze clock jitter characteristics
- Output: RJ, DJ, TJ and other jitter metrics
- Purpose: Performance evaluation under PLL/ADPLL modes

**FrequencyCounter (Frequency Counter)**:
- Function: Measure output clock frequency
- Purpose: Verify correctness of frequency configuration

---

## 5. Simulation Results Analysis

### 5.1 Statistical Metrics Description

The Clock Generator outputs phase signals, which require specific statistical metrics to evaluate performance:

| Metric | Calculation Method | Significance |
|--------|-------------------|--------------|
| Phase Mean (phase_mean) | Arithmetic mean of all sampling points | Reflects DC offset of phase (ideally should be π) |
| Phase Range (phase_range) | Max value - Min value | Reflects dynamic range of phase (ideally should be close to 2π) |
| Phase RMS (phase_rms) | Root mean square | Reflects effective value of phase fluctuation |
| Cycle Count (cycle_count) | Number of times phase crosses 2π | Verifies number of clock cycles |
| Phase Increment (phase_increment) | Phase difference between adjacent sampling points | Verifies constancy of phase increment (ideal clock) |
| Time Step (timestep) | 1 / (frequency × 100) | Verifies time step adaptive mechanism |

**Phase Signal Characteristics**:
- Ideal clock phase should grow linearly within [0, 2π) range
- Phase mean should be close to π (uniformly distributed within [0, 2π) range)
- Phase increment should be constant (ideal clock, no jitter)
- Cycle count should match simulation duration and clock frequency

### 5.2 Typical Test Result Interpretation

#### System-level Integration Test Result Example

Configuration: 40GHz ideal clock, simulation duration 1μs

Expected Results:
- **Time Step**: 0.25ps (1 / (40e9 × 100))
- **Total Sampling Points**: 4,000,000 (1μs / 0.25ps)
- **Cycle Count**: 40,000 (40GHz × 1μs)
- **Phase Range**: Close to 2π (approximately 6.283)
- **Phase Increment**: 0.06283 radians (2π / 100, 100 sampling points per cycle)
- **Phase Mean**: Approximately 3.14159 (π, uniform distribution)

Analysis Methods:
1. Check if time step matches configured frequency
2. Verify if phase range is within [0, 2π)
3. Calculate phase increment, verify its constancy
4. Count cycles, verify clock frequency accuracy

#### Frequency Configuration Test Result Interpretation

Test time step adaptation under different frequency configurations:

| Configured Frequency | Expected Time Step | Actual Time Step | Verification Result |
|---------------------|-------------------|------------------|---------------------|
| 10GHz | 1.0ps | 1.0ps | ✓ Pass |
| 20GHz | 0.5ps | 0.5ps | ✓ Pass |
| 40GHz | 0.25ps | 0.25ps | ✓ Pass |
| 80GHz | 0.125ps | 0.125ps | ✓ Pass |

Analysis Methods:
- Read the time step output by the module
- Compare with theoretical value: `timestep = 1 / (frequency × 100)`
- Verify error is within acceptable range (floating-point precision limits)

#### Phase Continuity Test Result Interpretation

Verify continuity and periodicity of phase output:

**Ideal Clock Characteristics**:
- Phase values grow strictly linearly
- Each complete cycle phase value grows from 0 to 2π
- Phase increment is constant, no jitter

**Analysis Methods**:
1. Plot phase vs time curve
2. Check if curve is a linear sawtooth wave
3. Calculate standard deviation of phase increment:
   - Ideal clock: Standard deviation = 0 (phase increment constant)
   - Actual measurement: Standard deviation should be close to machine precision (approximately 1e-15)
4. Verify phase normalization:
   - Phase values should always be within [0, 2π) range
   - When exceeding 2π, should correctly normalize to [0, 2π)

**Future PLL/ADPLL Mode Results**:
- Phase increment will no longer be constant, reflecting dynamic adjustment of loop filter
- Phase noise will manifest as random fluctuations in phase increment
- Jitter metrics can be obtained through phase increment statistical analysis

### 5.3 Waveform Data File Format

The Clock Generator module's phase output can be recorded to a file through SystemC-AMS's trace function.

#### Tabular Format (.dat file)

SystemC-AMS default tabular format output:

```
Time(s)    clk_phase(rad)
0.000000e+00  0.000000e+00
2.500000e-13  6.283185e-02
5.000000e-13  1.256637e-01
7.500000e-13  1.884956e-01
...
```

**Format Description**:
- First column: Simulation time (seconds)
- Second column: Phase value (radians)
- Separator: Space or tab
- Number of sampling points: Determined by simulation duration and time step

#### CSV Format (.csv file)

Can be converted to CSV format through Python post-processing:

```python
import numpy as np

# Read .dat file
t, phase = np.loadtxt('clock_phase.dat', unpack=True)

# Save as CSV
np.savetxt('clock_phase.csv', np.column_stack([t, phase]), 
           delimiter=',', header='time(s),phase(rad)', comments='')
```

**CSV Format Example**:
```
time(s),phase(rad)
0.000000e+00,0.000000e+00
2.500000e-13,6.283185e-02
5.000000e-13,1.256637e-01
...
```

#### Python Post-processing Examples

**Phase Waveform Plotting**:
```python
import matplotlib.pyplot as plt
import numpy as np

# Read data
t, phase = np.loadtxt('clock_phase.dat', unpack=True)

# Plot phase vs time
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, phase)  # Convert to ns units
plt.xlabel('Time (ns)')
plt.ylabel('Phase (rad)')
plt.title('Clock Phase Output')
plt.grid(True)
plt.show()
```

**Phase Increment Analysis**:
```python
# Calculate phase increment
phase_increment = np.diff(phase)

# Plot phase increment
plt.figure(figsize=(12, 4))
plt.plot(phase_increment)
plt.xlabel('Sample Index')
plt.ylabel('Phase Increment (rad)')
plt.title('Phase Increment (Ideal Clock)')
plt.grid(True)
plt.show()

# Statistical analysis
print(f'Phase increment mean: {np.mean(phase_increment):.6f} rad')
print(f'Phase increment std: {np.std(phase_increment):.2e} rad')
```

**Phase Histogram**:
```python
# Plot phase distribution histogram
plt.figure(figsize=(8, 4))
plt.hist(phase, bins=100, density=True, edgecolor='black')
plt.xlabel('Phase (rad)')
plt.ylabel('Probability Density')
plt.title('Phase Distribution (Ideal Clock)')
plt.grid(True)
plt.show()
```

The phase distribution of an ideal clock should be approximately uniformly distributed within the [0, 2π) range.

---

## 6. Running Guide

### 6.1 Environment Configuration

Before running simulation, configure SystemC and SystemC-AMS environment variables:

```bash
# Method 1: Use project-provided script
source scripts/setup_env.sh

# Method 2: Manually set environment variables
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$SYSTEMC_AMS_HOME/lib-linux64:$LD_LIBRARY_PATH
```

**Environment Variable Descriptions**:
- `SYSTEMC_HOME`: Installation path of SystemC library
- `SYSTEMC_AMS_HOME`: Installation path of SystemC-AMS library
- `LD_LIBRARY_PATH`: Dynamic link library search path

**Verify Environment Configuration**:
```bash
# Check SystemC version
ls $SYSTEMC_HOME/lib-linux64/libsystemc-*

# Check SystemC-AMS version
ls $SYSTEMC_AMS_HOME/lib-linux64/libsystemc-ams-*
```

### 6.2 Build and Run

#### Build with CMake

```bash
# Create build directory
mkdir -p build
cd build

# Configure (Debug or Release)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j4

# Run system-level test
cd bin
./simple_link_tb
```

#### Build with Makefile

```bash
# Build all modules and testbenches
make all

# Run system-level test
make run

# Clean build artifacts
make clean
```

#### Modify Clock Configuration

Clock Generator configuration is controlled through configuration files, supporting JSON and YAML formats:

**Modify YAML Configuration File** (`config/default.yaml`):
```yaml
clock:
  type: PLL          # ⚠️ Note: PLL mode not currently implemented, only IDEAL mode supported
  frequency: 40e9    # Clock frequency (Hz)
  pd: "tri-state"    # Phase detector type
  cp:
    I: 5e-5          # Charge pump current (A)
  lf:
    R: 10000         # Loop filter resistance (Ω)
    C: 1e-10         # Loop filter capacitance (F)
  vco:
    Kvco: 1e8        # VCO gain (Hz/V)
    f0: 1e10         # VCO center frequency (Hz)
  divider: 4         # Divider ratio
```

**Modify JSON Configuration File** (`config/default.json`):
```json
{
  "clock": {
    "type": "PLL",
    "frequency": 40000000000,
    "pd": "tri-state",
    "cp": {
      "I": 5e-5
    },
    "lf": {
      "R": 10000,
      "C": 1e-10
    },
    "vco": {
      "Kvco": 1e8,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

**Test Different Clock Frequencies**:
```yaml
# 10GHz clock
clock:
  frequency: 10e9

# 20GHz clock
clock:
  frequency: 20e9

# 40GHz clock (default)
clock:
  frequency: 40e9

# 80GHz clock
clock:
  frequency: 80e9
```

After modifying configuration, recompile and run tests:
```bash
cd build
make simple_link_tb
cd bin
./simple_link_tb
```

#### Run Parameter Descriptions

The system-level testbench (`simple_link_tb`) currently does not support command line parameters; all configuration is controlled through configuration files.

**Future Independent Testbench Plans**:
```bash
# Command line parameters planned for future support
./clock_gen_tb [scenario]

# Scenario parameters:
# `ideal` or `0` - Ideal clock test (default)
# `pll_lock` or `1` - PLL lock time test
# `pll_bw` or `2` - PLL loop bandwidth test
# `jitter` or `3` - Phase noise and jitter test
```

### 6.3 Result Viewing

#### View Simulation Output

After running the system-level testbench, console outputs configuration information and simulation progress:

```
=== SerDes SystemC-AMS Simple Link Testbench ===
Configuration loaded:
  Sampling rate: 80 GHz
  Data rate: 40 Gbps
  Simulation time: 1 us

Creating TX modules...
Creating Channel module...
Creating RX modules...
Connecting TX chain...
Connecting Channel...
Connecting RX chain...

Creating trace file...

Starting simulation...

=== Simulation completed successfully! ===
Trace file: simple_link.dat
```

#### View Waveform Data

The simulation output file is `simple_link.dat`, containing waveform data of all traced signals.

**View File Contents**:
```bash
# View first 20 lines
head -n 20 simple_link.dat

# Count sampling points
wc -l simple_link.dat

# View file size
ls -lh simple_link.dat
```

**File Format**:
```
Time(s)    wave_out(V)  ffe_out(V)  driver_out(V)  ...
0.000000e+00  1.000000e+00  1.000000e+00  8.000000e-01  ...
1.250000e-11  1.000000e+00  1.000000e+00  8.000000e-01  ...
2.500000e-11  1.000000e+00  1.000000e+00  8.000000e-01  ...
...
```

#### Python Post-processing Analysis

**Basic Waveform Plotting**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Read waveform data
data = np.loadtxt('simple_link.dat')
t = data[:, 0]  # Time column
wave_out = data[:, 1]  # Waveform output

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, wave_out)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.title('Waveform Output')
plt.grid(True)
plt.show()
```

**Clock Phase Analysis** (if clock phase trace is added):
```python
# Assuming clock phase trace is added
# Add to simple_link_tb.cpp:
# sca_util::sca_trace(tf, clk_phase_signal, "clk_phase");

# Read clock phase data
t, clk_phase = np.loadtxt('simple_link.dat', usecols=(0, -1), unpack=True)

# Plot phase vs time
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, clk_phase)
plt.xlabel('Time (ns)')
plt.ylabel('Phase (rad)')
plt.title('Clock Phase Output')
plt.grid(True)
plt.show()

# Calculate phase increment
phase_increment = np.diff(clk_phase)

# Plot phase increment
plt.figure(figsize=(12, 4))
plt.plot(phase_increment)
plt.xlabel('Sample Index')
plt.ylabel('Phase Increment (rad)')
plt.title('Phase Increment (Ideal Clock)')
plt.grid(True)
plt.show()

# Statistical analysis
print(f'Phase increment mean: {np.mean(phase_increment):.6f} rad')
print(f'Phase increment std: {np.std(phase_increment):.2e} rad')
```

**Frequency Verification**:
```python
# Verify clock frequency
total_time = t[-1] - t[0]
expected_cycles = 40e9 * total_time  # 40GHz clock

# Count actual cycle number (number of times phase crosses 2π)
phase_wraps = np.sum(clk_phase[1:] < clk_phase[:-1])
actual_cycles = phase_wraps

print(f'Total simulation time: {total_time * 1e6:.2f} us')
print(f'Expected cycles: {expected_cycles:.0f}')
print(f'Actual cycles: {actual_cycles}')
print(f'Frequency error: {(actual_cycles/expected_cycles - 1) * 100:.4f}%')
```

#### Advanced Analysis Scripts

**Create Clock Analysis Script** (`scripts/analyze_clock.py`):
```python
#!/usr/bin/env python3
"""
Clock Generator Analysis Script
Analyzes output characteristics of the Clock Generator
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def analyze_clock_phase(filename='simple_link.dat'):
    """Analyze clock phase output"""
    print(f"Loading data from {filename}...")
    
    # Read data
    data = np.loadtxt(filename)
    t = data[:, 0]
    
    # Check if phase column exists
    if data.shape[1] < 2:
        print("Error: No clock phase data found in file")
        return
    
    clk_phase = data[:, -1]  # Assume last column is phase
    
    # Calculate statistical metrics
    phase_mean = np.mean(clk_phase)
    phase_std = np.std(clk_phase)
    phase_min = np.min(clk_phase)
    phase_max = np.max(clk_phase)
    phase_range = phase_max - phase_min
    
    # Calculate phase increment
    phase_increment = np.diff(clk_phase)
    increment_mean = np.mean(phase_increment)
    increment_std = np.std(phase_increment)
    
    # Count cycles
    phase_wraps = np.sum(clk_phase[1:] < clk_phase[:-1])
    
    # Calculate time step
    timestep = np.mean(np.diff(t))
    expected_timestep = 1 / (40e9 * 100)  # 40GHz clock
    
    # Print statistical results
    print("\n=== Clock Phase Statistics ===")
    print(f"Phase mean: {phase_mean:.6f} rad (expected: π ≈ 3.141593)")
    print(f"Phase std: {phase_std:.6f} rad")
    print(f"Phase range: {phase_range:.6f} rad (expected: 2π ≈ 6.283185)")
    print(f"Phase min: {phase_min:.6f} rad")
    print(f"Phase max: {phase_max:.6f} rad")
    print(f"\nPhase increment mean: {increment_mean:.6f} rad")
    print(f"Phase increment std: {increment_std:.2e} rad")
    print(f"\nCycle count: {phase_wraps}")
    print(f"Timestep: {timestep * 1e12:.2f} ps (expected: {expected_timestep * 1e12:.2f} ps)")
    print(f"Timestep error: {(timestep/expected_timestep - 1) * 100:.4f}%")
    
    # Plot graphs
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Phase vs time
    axes[0].plot(t * 1e9, clk_phase)
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Phase (rad)')
    axes[0].set_title('Clock Phase vs Time')
    axes[0].grid(True)
    
    # Phase increment
    axes[1].plot(phase_increment)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Phase Increment (rad)')
    axes[1].set_title('Phase Increment')
    axes[1].grid(True)
    
    # Phase histogram
    axes[2].hist(clk_phase, bins=100, density=True, edgecolor='black')
    axes[2].set_xlabel('Phase (rad)')
    axes[2].set_ylabel('Probability Density')
    axes[2].set_title('Phase Distribution')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('clock_analysis.png', dpi=150)
    print(f"\nPlot saved to: clock_analysis.png")
    plt.show()

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'simple_link.dat'
    analyze_clock_phase(filename)
```

**Run Analysis Script**:
```bash
cd build/bin
python3 ../../scripts/analyze_clock.py simple_link.dat
```

#### Docker Environment Running

If using Docker container, follow these steps:

```bash
# Build Docker image
docker build -t serdes-systemc .

# Run container
docker run -it serdes-systemc /bin/bash

# Build and run inside container
mkdir build && cd build
cmake ..
make -j4
cd bin
./simple_link_tb

# Copy result files to host
docker cp <container_id>:/app/build/bin/simple_link.dat ./
```

---

## 7. Technical Highlights

### 7.1 Phase Accumulator Design Advantages

**Problem**: Why use a phase accumulator instead of directly calculating `φ = 2π × f × t`?

**Solution**:
- Phase accumulator uses **incremental updates** (`m_phase += Δφ`), with independent error at each step
- Modulo 2π operation keeps values within a reasonable range, avoiding floating-point cumulative errors
- Improves numerical stability, especially suitable for long-duration simulations

**Advantages Summary**:
1. **Accuracy**: Avoids floating-point cumulative errors, more stable for long-term simulations
2. **Flexibility**: Easily extensible to support frequency modulation, phase noise injection
3. **Simplicity**: Clear code logic, low computational overhead
4. **Extensibility**: When implementing PLL in the future, the phase increment can be directly modified

### 7.2 Adaptive Time Step Mechanism

The module adopts adaptive time step setting, ensuring the sampling rate matches the clock frequency:

```
Sampling rate = Clock frequency × 100
Time step = 1 / Sampling rate
```

**Design Considerations**:
- **Nyquist Criterion**: Sampling rate (100× frequency) is much higher than signal frequency, avoiding aliasing
- **Simulation Efficiency**: Sampling rate proportional to frequency, reducing computational overhead for low-frequency clocks
- **Time Domain Resolution**: 100 sampling points per cycle, sufficient to represent phase changes

**Limitations**:
- SystemC-AMS requires all modules in the same time domain to use the same time step
- The Clock Generator's time step affects the entire system's sampling rate
- High-frequency clocks (e.g., 80GHz) cause the entire system's time step to be very small (0.125ps)

### 7.3 Phase Normalization Processing

**Problem**: The phase accumulator grows indefinitely, how to keep phase within [0, 2π) range?

**Solution**:
```cpp
if (m_phase >= 2.0 * M_PI) {
    m_phase -= 2.0 * M_PI;
}
```

**Design Considerations**:
- Uses simple subtraction instead of modulo operation (`fmod`) for higher efficiency
- Only performs normalization when phase exceeds 2π, reducing unnecessary computation
- Keeps phase values within a reasonable range, improving numerical stability

**Notes**:
- Phase normalization may introduce tiny errors under certain boundary conditions
- For PLL mode, need to ensure loop filter output considers the effect of phase normalization

### 7.4 PLL Parameter Structure Design

Although the current implementation only supports ideal clocks, the PLL parameter structure is fully defined:

**Design Advantages**:
1. **Parameter Completeness**: Includes all necessary parameters for PD/CP/LF/VCO/Divider
2. **Extensibility**: Easy to add new phase detector types and loop filter structures
3. **Configuration-Driven**: Flexibly adjust PLL parameters through configuration files
4. **Future-Ready**: Reserves complete interfaces for PLL implementation

**Current Status**:
- Parameter structure is fully defined
- `processing()` method only implements ideal clocks
- PLL parameter configuration is not used in current implementation

### 7.5 Frequency Configuration Limitations

**Problem**: What are the limitations of clock frequency configuration?

**Limitation Descriptions**:
1. **Frequency must be greater than zero**: `frequency > 0`, otherwise time step calculation will error
2. **Frequency upper limit**: Limited by SystemC-AMS's minimum time step (typically about 1fs)
3. **Frequency and simulation duration**: High-frequency clocks with long simulation times generate large amounts of data

**Calculation Examples**:
- 40GHz clock, 1μs simulation: 4,000,000 sampling points, approximately 32MB data
- 80GHz clock, 1μs simulation: 8,000,000 sampling points, approximately 64MB data

**Recommendations**:
- High-frequency clocks (>40GHz) recommend shortening simulation duration
- Pay attention to memory management when using Python post-processing
- Consider using sparse sampling or downsampling techniques

### 7.6 Floating-Point Precision Issues

**Problem**: How does floating-point precision of the phase accumulator affect long-term simulation?

**Precision Analysis**:
- `double` type provides approximately 15-17 significant digits
- Phase increment per time step: `Δφ = 2π × f × Δt`
- For 40GHz clock, `Δφ ≈ 0.06283` radians
- Floating-point relative error: approximately 1e-15 (machine precision)

**Long-Term Simulation Impact**:
- 1μs simulation (40,000 cycles): Cumulative error negligible
- 1ms simulation (40,000,000 cycles): Observable phase drift may occur
- Recommendation: Use higher precision types (e.g., `long double`) for ultra-long simulations

### 7.7 Interface Differences with CDR Module

**Problem**: Why doesn't the current CDR module use the Clock Generator's phase output?

**Status Description**:
- Clock Generator outputs phase signal: `clk_phase`
- Current CDR module uses internal phase generation logic
- Interface mismatch, CDR not connected to Clock Generator

**Future Improvements**:
1. **Unified Clock Architecture**: CDR should use Clock Generator's phase output
2. **Phase Alignment**: Ensure CDR and Sampler use the same clock phase
3. **Clock Tree Modeling**: Implement complete clock distribution network

**Current Implementation**:
- Sampler module uses Clock Generator's phase (if connected)
- CDR module independently generates phase for clock data recovery

### 7.8 Limitations of Ideal Clock Model

**Current Implementation**: Ideal clock model, phase grows strictly linearly

**Unmodeled Non-Ideal Effects**:
1. **Phase Noise**: Real clocks have random phase fluctuations
2. **Jitter**: Including Random Jitter (RJ) and Deterministic Jitter (DJ)
3. **Frequency Drift**: Frequency offset caused by temperature and voltage variations
4. **Duty Cycle Distortion**: Real clock duty cycle may deviate from 50%

**Applicable Scenarios**:
- Functional verification: Verify basic functionality of SerDes link
- Performance baseline: Provide performance reference under ideal conditions
- Algorithm development: For developing and testing equalization, CDR algorithms

**Inapplicable Scenarios**:
- Jitter tolerance testing: Requires real clock jitter model
- Phase noise analysis: Requires phase noise injection
- System-level performance evaluation: Requires complete non-ideal effect modeling

### 7.9 Key Challenges for Future PLL Implementation

**Challenge 1: Reference Clock Source**
- Problem: Current system lacks an independent reference clock source
- Solution: Need to add reference clock module or generate in testbench

**Challenge 2: Loop Stability**
- Problem: Improper PLL parameter configuration may cause loop instability or oscillation
- Solution: Need to implement loop stability analysis and parameter optimization algorithms

**Challenge 3: Lock Time Modeling**
- Problem: PLL lock process requires multiple clock cycles, long simulation time
- Solution: Use fast simulation techniques or simplified lock process models

**Challenge 4: Numerical Stability**
- Problem: Integrators in PLL loop may cause numerical cumulative errors
- Solution: Use appropriate numerical integration methods and error control

**Challenge 5: Integration with Existing Modules**
- Problem: PLL output needs to correctly integrate with Sampler, CDR and other modules
- Solution: Design unified clock distribution interfaces and synchronization mechanisms

### 7.10 Known Limitations and Special Requirements

**Known Limitations**:
1. **Only supports ideal clock**: PLL/ADPLL modes not currently implemented
2. **No noise modeling**: Does not support phase noise and jitter injection
3. **No reference clock**: PLL mode requires additional reference clock source
4. **Global unified time step**: Clock Generator's time step affects entire system
5. **No independent testbench**: Only verified in system-level tests

**Special Requirements**:
1. **Frequency configuration**: Must be greater than zero, recommended within 1GHz-100GHz range
2. **Simulation duration**: High-frequency clocks recommend shortening simulation duration to avoid excessive data
3. **Environment configuration**: Must correctly set `SYSTEMC_HOME` and `SYSTEMC_AMS_HOME`
4. **Floating-point precision**: Ultra-long simulations may require higher precision types
5. **Phase trace**: To analyze phase output, must add trace in testbench

### 7.11 Comparison with Other Clock Modules

**Comparison with WaveGenerationTdf**:
- WaveGenerationTdf: Generates data signals (PRBS), does not involve clock phase
- ClockGenerationTdf: Generates clock phase, used for timing control
- Complementary: WaveGeneration provides data, ClockGeneration provides clock

**Comparison with RxCdrTdf**:
- RxCdrTdf: Recovers clock from data, implements clock data recovery functionality
- ClockGenerationTdf: Generates source clock, serves as system clock reference
- Relationship: ClockGeneration provides reference clock, CDR recovers synchronized clock from data

**Comparison of Ideal vs Real Clocks**:
- Ideal clock: No jitter, no noise, phase strictly linear
- Real clock: Has various non-ideal effects, requires complex modeling
- Current implementation: Ideal clock, future plans support real clock modeling

---

## 8. Reference Information

### 8.1 Related Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definitions | `/include/common/parameters.h` | ClockParams, ClockPllParams structures |
| Type Definitions | `/include/common/types.h` | ClockType enumeration and conversion functions |
| Header File | `/include/ams/clock_generation.h` | ClockGenerationTdf class declaration |
| Implementation File | `/src/ams/clock_generation.cpp` | ClockGenerationTdf class implementation |
| System-level Test | `/tb/simple_link_tb.cpp` | SerDes link integration test |
| Configuration File | `/config/default.yaml` | YAML format default configuration |
| Configuration File | `/config/default.json` | JSON format default configuration |
| Environment Script | `/scripts/setup_env.sh` | SystemC/SystemC-AMS environment configuration |

**File Descriptions**:
- **Parameter Definitions**: Contains all parameter structures for the Clock Generator, including basic parameters and PLL sub-parameters
- **Type Definitions**: Defines clock type enumeration (IDEAL/PLL/ADPLL) and string conversion functions
- **Header File**: Declares ClockGenerationTdf class, inheriting from sca_tdf::sca_module
- **Implementation File**: Implements phase accumulator, adaptive time step setting and other core functions
- **System-level Test**: Current Clock Generator is primarily verified through system-level tests
- **Configuration Files**: Supports YAML and JSON formats, loaded through ConfigLoader

### 8.2 Dependencies

**Core Dependencies**:
- **SystemC 2.3.4**: System-level modeling framework, provides event-driven and TDF domain support
- **SystemC-AMS 2.3.4**: Analog/Mixed-Signal extension, provides TDF domain continuous-time modeling

**Compilation Dependencies**:
- **C++14 Standard**: Module uses C++14 features (e.g., constexpr, auto, etc.)
- **CMake 3.15+** or **GNU Make**: Build system
- **Standard Libraries**: `<cmath>` (M_PI constant), `<iostream>`, etc.

**Optional Dependencies**:
- **Python 3.x**: For post-processing analysis and visualization
  - `numpy`: Numerical computation and array processing
  - `matplotlib`: Waveform plotting and data visualization
  - `scipy`: Signal processing and statistical analysis

**Version Compatibility**:
| Dependency | Minimum Version | Recommended Version | Tested Version |
|------------|-----------------|---------------------|----------------|
| SystemC | 2.3.3 | 2.3.4 | 2.3.4 |
| SystemC-AMS | 2.3.3 | 2.3.4 | 2.3.4 |
| C++ Standard | C++11 | C++14 | C++14 |
| CMake | 3.10 | 3.15+ | 3.20+ |
| Python | 3.6 | 3.8+ | 3.10.12 |

### 8.3 Configuration Examples

#### Basic Configuration (Ideal Clock)

**YAML Format**:
```yaml
clock:
  type: IDEAL         # Clock type: IDEAL/PLL/ADPLL
  frequency: 40e9     # Clock frequency: 40GHz
```

**JSON Format**:
```json
{
  "clock": {
    "type": "IDEAL",
    "frequency": 40000000000
  }
}
```

#### PLL Configuration (Analog Phase-Locked Loop)

**YAML Format**:
```yaml
clock:
  type: PLL
  frequency: 40e9
  pd: "tri-state"    # Phase detector type: tri-state/bang-bang/linear/hogge
  cp:
    I: 5e-5          # Charge pump current: 50μA
  lf:
    R: 10000         # Loop filter resistance: 10kΩ
    C: 1e-10         # Loop filter capacitance: 100pF
  vco:
    Kvco: 1e8        # VCO gain: 100MHz/V
    f0: 1e10         # VCO center frequency: 10GHz
  divider: 4         # Feedback divider ratio: 4
```

**JSON Format**:
```json
{
  "clock": {
    "type": "PLL",
    "frequency": 40000000000,
    "pd": "tri-state",
    "cp": {
      "I": 5e-5
    },
    "lf": {
      "R": 10000,
      "C": 1e-10
    },
    "vco": {
      "Kvco": 1e8,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

**PLL Parameter Descriptions**:
- **Loop bandwidth**: `ωn = sqrt(Kvco × Icp / (N × C))` ≈ 2.24×10^6 rad/s (356kHz)
- **Damping coefficient**: `ζ = (R/2) × sqrt(Icp × Kvco × C / N)` ≈ 0.707
- **Lock time**: `T_lock ≈ 4/(ζ × ωn)` ≈ 2.5μs

#### ADPLL Configuration (All-Digital Phase-Locked Loop)

**YAML Format** (Future implementation):
```yaml
clock:
  type: ADPLL
  frequency: 40e9
  pd: "digital"      # Digital phase detector
  tdc:
    resolution: 1e-12 # Time-to-Digital Converter resolution: 1ps
  dco:
    resolution: 1e6   # Digitally-Controlled Oscillator resolution: 1MHz
    f0: 1e10          # DCO center frequency: 10GHz
  divider: 4
```

**JSON Format** (Future implementation):
```json
{
  "clock": {
    "type": "ADPLL",
    "frequency": 40000000000,
    "pd": "digital",
    "tdc": {
      "resolution": 1e-12
    },
    "dco": {
      "resolution": 1e6,
      "f0": 1e10
    },
    "divider": 4
  }
}
```

#### Different Frequency Configuration Examples

**10GHz Clock (Low-Speed SerDes)**:
```yaml
clock:
  type: IDEAL
  frequency: 10e9
```

**20GHz Clock (Medium-Speed SerDes)**:
```yaml
clock:
  type: IDEAL
  frequency: 20e9
```

**40GHz Clock (High-Speed SerDes, default)**:
```yaml
clock:
  type: IDEAL
  frequency: 40e9
```

**80GHz Clock (Ultra-High-Speed SerDes)**:
```yaml
clock:
  type: IDEAL
  frequency: 80e9
```

**Note**: High-frequency clocks (>40GHz) cause very small time steps, recommend shortening simulation duration to avoid excessive data.

#### Complete System Configuration Example

**YAML Format** (`config/default.yaml`):
```yaml
global:
  Fs: 80e9           # Sampling rate: 80GHz
  UI: 2.5e-11        # Unit Interval: 25ps (40Gbps)
  duration: 1e-6     # Simulation duration: 1μs
  seed: 12345        # Random seed

wave:
  type: PRBS31
  poly: "x^31 + x^28 + 1"
  init: "0x7FFFFFFF"

tx:
  ffe_taps: [0.2, 0.6, 0.2]
  mux_lane: 0
  driver:
    swing: 0.8
    bw: 20e9

channel:
  attenuation_db: 10.0
  bandwidth_hz: 20e9

rx:
  ctle:
    zeros: [2e9]
    poles: [30e9]
    dc_gain: 1.5
    vcm_out: 0.6
  vga:
    gain: 4.0
  sampler:
    threshold: 0.0
    hysteresis: 0.02
  dfe:
    taps: [-0.05, -0.02, 0.01]
    update: "sign-lms"
    mu: 1e-4

cdr:
  pi:
    kp: 0.01
    ki: 1e-4
  pai:
    resolution: 1e-12
    range: 5e-11

clock:
  type: IDEAL
  frequency: 40e9

eye:
  ui_bins: 128
  amp_bins: 128
  measure_length: 1e-4
```

**JSON Format** (`config/default.json`):
```json
{
  "global": {
    "Fs": 80000000000,
    "UI": 2.5e-11,
    "duration": 1e-6,
    "seed": 12345
  },
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF"
  },
  "tx": {
    "ffe_taps": [0.2, 0.6, 0.2],
    "mux_lane": 0,
    "driver": {
      "swing": 0.8,
      "bw": 20000000000
    }
  },
  "channel": {
    "attenuation_db": 10.0,
    "bandwidth_hz": 20000000000
  },
  "rx": {
    "ctle": {
      "zeros": [2000000000],
      "poles": [30000000000],
      "dc_gain": 1.5,
      "vcm_out": 0.6
    },
    "vga": {
      "gain": 4.0
    },
    "sampler": {
      "threshold": 0.0,
      "hysteresis": 0.02
    },
    "dfe": {
      "taps": [-0.05, -0.02, 0.01],
      "update": "sign-lms",
      "mu": 0.0001
    }
  },
  "cdr": {
    "pi": {
      "kp": 0.01,
      "ki": 0.0001
    },
    "pai": {
      "resolution": 1e-12,
      "range": 5e-11
    }
  },
  "clock": {
    "type": "IDEAL",
    "frequency": 40000000000
  },
  "eye": {
    "ui_bins": 128,
    "amp_bins": 128,
    "measure_length": 0.0001
  }
}
```

---

**Document Version**: v0.1  
**Last Updated**: 2026-01-20  
**Author**: Yizhe Liu
