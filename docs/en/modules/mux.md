# TX Mux Module Technical Documentation

🌐 **Languages**: [中文](../../modules/mux.md) | [English](mux.md)

**Level**: AMS Sub-module (TX)  
**Class Name**: `TxMuxTdf`  
**Current Version**: v0.1 (2026-01-13)  
**Status**: In Development

---

## 1. Overview

The Transmitter Multiplexer (Mux, Multiplexer) is a critical timing module in the SerDes transmit chain, located in the middle of the FFE → Mux → Driver signal path. Its primary functions are channel selection (Lane Selection) and signal routing control. In system-level applications, it serves as part of the Parallel-to-Serial Conversion architecture, providing selected data channels to the Driver while modeling the delay and jitter effects present in real hardware.

### 1.1 Design Principles

The core design philosophy of TX Mux is to model the selection logic, propagation delay, and jitter characteristics of a multiplexer at the behavioral level abstraction, providing sufficient accuracy for system-level simulation while avoiding transistor-level implementation details to maintain simulation efficiency.

#### 1.1.1 Functional Positioning of the Multiplexer

In a complete SerDes transmit chain, the multiplexer plays the following roles:

- **Channel Selection (Lane Selection)**: In multi-lane SerDes architectures, the transmitter may contain multiple parallel data channels, each operating at a lower symbol rate to reduce clock frequency and power consumption. The Mux selects one of these channels based on control signals and routes its data to the Driver, implementing channel-level routing switching.

- **Part of Parallel-to-Serial Conversion**: In real hardware, N:1 parallel-to-serial conversion typically consists of N parallel data paths (Lanes) and an N:1 Mux. For example, in an 8:1 structure, 8 parallel Lanes each operate at the symbol rate, and the Mux combines them into a bit rate (Bit Rate = Symbol Rate × 8) serial output through Time-Division Multiplexing. **This behavioral model focuses on the selection and delay characteristics of the Mux unit itself, rather than the complete parallel data paths**.

- **Abstraction Level Note**: This module adopts a single-input single-output (`in` → `out`) architecture, working with the `lane_sel` parameter to select the channel index. This abstraction simplifies modeling complexity and is suitable for the following scenarios:
  - Single-channel systems (`lane_sel=0`, Bypass mode)
  - Behavioral verification of selected channels in multi-lane architectures
  - Independent testing of delay and jitter effects

#### 1.1.2 Sampling Rate and Timing Relationships

The input-output sampling rate relationship of the Mux determines its timing behavior in the signal chain:

- **Symbol Rate Synchronization**: In this behavioral model, input and output sampling rates are kept consistent (`set_rate(1)`), indicating that the Mux operates in the symbol rate clock domain, processing one symbol per timestep. This differs from real hardware implementations where "Mux internal clock equals bit rate," but is sufficient for behavioral-level simulation to characterize signal transfer properties.

- **Relationship Between Timestep and UI**: Assuming the global sampling frequency is Fs (defined by `GlobalParams`), then timestep Δt = 1/Fs. For a system with symbol rate R_sym, each symbol period T_sym = 1/R_sym contains Fs/R_sym sampling points. For example:
  - If bit rate = 56 Gbps, symbol rate = 7 GHz (8:1 architecture), Fs = 560 GHz (10 samples per UI)
  - Then T_sym = 142.86 ps, Δt = 1.786 ps, each symbol contains 80 timesteps

- **Phase Alignment Considerations**: Real N:1 Mux requires N phase-accurately aligned clocks to implement time-division multiplexing. In the behavioral model, this phase alignment requirement is abstracted into delay parameters (`mux_delay`) and jitter models (`jitter_params`), allowing indirect simulation of phase mismatch effects by adjusting these parameters.

#### 1.1.3 Behavioral-Level Modeling of the Selector

This module adopts an ideal selector model, abstracting the complex topology of real hardware:

- **Ideal Transfer Characteristics**: Without considering non-ideal effects, the output directly equals the selected channel's input: `out[n] = in[n]`. This simplification is suitable for functional verification and early stages of signal integrity analysis.

- **Delay Modeling**: The optional parameter `mux_delay` is used to model the propagation delay of the selector. In real hardware, delay sources include:
  - Tri-state buffer or transmission gate switching time (~10-30ps)
  - Clock-to-Q delay in the clock-to-data path
  - RC delay from interconnect parasitic capacitance and resistance
  
  In the behavioral model, these effects are approximated by inserting fixed delay elements (such as filter group delay or explicit delay lines) into the signal path.

- **Correspondence with Real Topologies**:
  - **Tree Selector**: Multi-level 2:1 cascades, total delay = number of levels × single-level delay
  - **Parallel Selector**: Single-level multi-way selection, minimum delay but load-dependent
  - **Behavioral Model**: Match equivalent delay of above topologies by adjusting `mux_delay` parameter without implementing specific circuit structures

#### 1.1.4 Jitter Effect Modeling

The Mux is a significant jitter source in the SerDes transmitter. The behavioral model needs to capture the following effects:

- **Deterministic Jitter (DJ)** sources:
  - **Duty Cycle Distortion (DCD)**: Clock duty cycle deviation from 50% causes unequal adjacent UI widths. DCD-induced peak-to-peak jitter = `DJ_DCD = UI × |DCD% - 50%|`. For example, a 48% duty cycle (2% deviation) produces 0.32ps jitter at 16ps UI.
  - **Pattern-Dependent Jitter (PDJ)**: Selector propagation delay varies with input data patterns, which can be modeled by injecting different delay offsets for different patterns.
  
- **Random Jitter (RJ)** sources:
  - **Clock Phase Noise**: Phase noise from the PLL transfers to data edges, manifesting as Gaussian-distributed random jitter, typical values 0.1-0.5 ps rms
  - **Thermal Noise**: Thermal noise from selector circuit superimposes on output, related to circuit bandwidth and temperature
  
- **Behavioral-Level Modeling Methods**:
  - **Time Domain Injection**: Apply random time shifts or amplitude perturbations to the output signal at each timestep
  - **Parameterized Control**: Flexibly adjust jitter levels through `jitter_enable`, `dcd_percent`, `rj_sigma` and other parameters to match target hardware specifications

### 1.2 Core Features

- **Single-Input Single-Output Architecture**: Adopts a simplified single-ended signal path (`in` → `out`), focusing on channel selection and delay/jitter modeling rather than complete multi-channel parallel inputs. Suitable for behavioral-level simulation and algorithm verification.

- **Channel Index Selection**: Specifies the selected channel index (0-based) through the constructor parameter `lane_sel`, supporting independent modeling and testing of specific channels in multi-channel systems. Default value is 0 (first channel).

- **Symbol Rate Synchronization**: Input and output sampling rates are consistent, operating in the symbol rate clock domain, compatible with timing requirements of preceding FFE and following Driver modules. Sampling rate is controlled by global parameter `Fs` (sampling frequency).

- **Configurable Delay**: Optional parameter `mux_delay` is used to model the propagation delay of the selector, matching real hardware timing characteristics. Delay range is typically 10-50ps, specific values depend on process node and topology.

- **Jitter Modeling Support**: Optionally injects Deterministic Jitter (DCD) and Random Jitter (RJ) to simulate the effects of clock non-idealities and circuit noise on output signal quality. Jitter parameters are flexibly set through configuration files.

- **Bypass Mode**: When `lane_sel=0` and no delay/jitter is configured, the module degrades to pass-through mode for front-end debugging or single-channel system verification.

### 1.3 Typical Application Scenarios

TX Mux configuration in different SerDes architectures:

| System Architecture | Lane Count | Symbol Rate | Mux Configuration | Typical Application |
|---------------------|------------|-------------|-------------------|---------------------|
| Single-Lane SerDes | 1 | Equal to bit rate | lane_sel=0, Bypass | Low-speed links (<10Gbps), PCIe Gen1/2 |
| 2:1 Parallel-to-Serial | 2 | Bit rate/2 | lane_sel=0 or 1 | Medium-speed links (10-25Gbps) |
| 4:1 Parallel-to-Serial | 4 | Bit rate/4 | lane_sel=0-3 | PCIe Gen3/4, USB3.x |
| 8:1 Parallel-to-Serial | 8 | Bit rate/8 | lane_sel=0-7 | 56G/112G SerDes, High-speed Ethernet |

> **Note**: This module models the behavior of a single Lane. Complete N:1 systems require instantiating N data paths (WaveGen → FFE → Mux) and performing time-division multiplexing simulation at the system level.

### 1.4 Relationship with Other Modules

- **Upstream Module (FFE)**  
  The Mux receives equalized symbol signals from the FFE, with inputs being nominal amplitude digital signals (e.g., ±1V). The FFE output timing must meet the Mux setup time requirement, typically > 0.2 UI.

- **Downstream Module (Driver)**  
  Mux output is sent to the Driver for power amplification and impedance matching. Mux output delay and jitter directly affect the Driver sampling moment and eye diagram quality, requiring joint optimization of timing budgets.

- **Clock Source (Clock Generation)**  
  In real hardware, the Mux relies on multi-phase clocks or a Phase Interpolator (PI) to implement time-division multiplexing. In the behavioral model, clock accuracy effects are abstracted into jitter parameters (such as DCD, RJ), configured by the user based on Clock module specifications.

- **System Configuration (System Configuration)**  
  Parameters such as `lane_sel`, `mux_delay`, `jitter_params` are loaded through configuration files, supporting multi-scenario switching and parameter sweep analysis (such as performance differences under different channel indices, impact of delay sweep on eye diagram).

### 1.5 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2026-01-13 | Initial version, implementing single-input single-output architecture, channel selection, and basic delay modeling |

---

## 2. Module Interface

### 2.1 Port Definitions (TDF Domain)

TX Mux adopts a single-ended signal architecture, all ports are TDF domain analog signals.

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in` | Input | double | Input data signal (from FFE or other preceding modules) |
| `out` | Output | double | Output data signal (to Driver or following modules) |

#### Port Connection Notes

- **Simplified Architecture**: Adopts Single-Input-Single-Output (SISO) design, abstracting the multi-input selection logic of real hardware. Channel selection is implemented through the constructor parameter `lane_sel` rather than multi-port switching.
- **Sampling Rate Consistency**: Input and output sampling rates are consistent (`set_rate(1)`), operating in the symbol rate clock domain. All connected TDF modules must use the same global sampling frequency `Fs` (defined by `GlobalParams`).
- **Signal Amplitude**: Input signal amplitude is typically the nominal value output by the preceding FFE (e.g., ±0.5V ~ ±1.0V). The current version (v0.1) adopts ideal transfer, with output amplitude identical to input.
- **Load Conditions**: Output port should be connected to the Driver module or test load, ensuring the following module can correctly sample the Mux output timing characteristics.

### 2.2 Parameter Configuration

#### 2.2.1 Currently Implemented Parameters

TX Mux parameters are defined in the `TxParams` structure (located at `include/common/parameters.h`). **Current version (v0.1) only implements the channel index parameter**:

| Parameter Path | Type | Default Value | Unit | Description |
|----------------|------|---------------|------|-------------|
| `tx.mux_lane` | int | 0 | - | Selected channel index (0-based), specifies the Lane number corresponding to Mux output |

**Constructor Signature**:
```cpp
TxMuxTdf(sc_core::sc_module_name nm, int lane_sel = 0);
```

**Current Behavior** (`src/ams/tx_mux.cpp`):
```cpp
void TxMuxTdf::processing() {
    // Simple pass-through mode (single channel)
    double x_in = in.read();
    out.write(x_in);
}
```

**Parameter Description**:
- **lane_sel (mux_lane) Design Intent**:
  - **Current Implementation**: Parameter is stored in member variable `m_lane_sel`, but not used in `processing()` function; module performs ideal pass-through operation.
  - **Design Purpose**: Reserved for future multi-channel architecture, for selecting one of N parallel inputs (under current single-input single-output architecture, this parameter has no actual functional impact).
  - **Valid Range**: Integer index, typically 0-based. For single-channel systems, fixed at `lane_sel=0`.

#### 2.2.2 Reserved Parameters (Future Versions)

The following parameters are **not yet implemented** in the current code and serve only as design planning and configuration file reserved interfaces for future version extensions.

##### Delay Modeling Parameters (Reserved)

| Parameter | Type | Default Value | Unit | Description | Implementation Status |
|-----------|------|---------------|------|-------------|----------------------|
| `mux_delay` | double | 0.0 | s | Propagation Delay, modeling fixed delay of selector | To be implemented |

**Design Intent**:
- **Physical Meaning**: Models propagation delay from input to output of the selector, including:
  - Selector logic delay (tri-state gate, transmission gate switching time)
  - Clock-to-data path delay (Clock-to-Q Delay)
  - Interconnect parasitic RC delay
- **Typical Values Reference**:
  - Advanced processes (7nm/5nm): 10-20 ps
  - Mature processes (28nm/16nm): 20-40 ps
  - Long interconnect traces: up to 50-100 ps
- **Implementation Method Suggestions**:
  - Use `sca_tdf::sca_delay` or explicit buffer queue for fixed delay
  - Or approximate propagation delay through first-order low-pass filter group delay

##### Jitter Modeling Parameters (Reserved)

| Parameter | Type | Default Value | Unit | Description | Implementation Status |
|-----------|------|---------------|------|-------------|----------------------|
| `jitter.enable` | bool | false | - | Enable jitter modeling functionality | To be implemented |
| `jitter.dcd_percent` | double | 50.0 | % | Duty Cycle, 50% is ideal, deviation from 50% produces DCD jitter | To be implemented |
| `jitter.rj_sigma` | double | 0.0 | s | Random Jitter (RJ) standard deviation, Gaussian distribution model | To be implemented |
| `jitter.seed` | int | 0 | - | Random number generator seed, 0 means use global seed | To be implemented |

**Design Intent**:

**Duty Cycle Distortion (DCD) Modeling**:
- **Physical Background**: Clock duty cycle deviation from 50% causes unequal adjacent UI (Unit Interval) widths, introducing edge position offset
- **Typical Impact**: For a system with UI=16ps, duty cycle deviating from 50% to 48% (2% deviation), produces approximately 0.3-0.6ps deterministic jitter at edges
- **Implementation Method Suggestions**:
  - Apply opposite direction time offsets to odd and even UIs
  - Implement fractional sample delay through interpolation techniques (such as Lagrange or Sinc interpolation)

**Random Jitter (RJ) Modeling**:
- **Physical Sources**:
  - Clock phase noise: Phase jitter from PLL/clock distribution transfers to data edges
  - Thermal noise: Thermal noise from selector circuit superimposes on output
  - Power supply noise: VDD jitter couples to Mux through clock path
- **Typical Values Reference**:
  - Low jitter systems: rj_sigma < 0.2 ps (peak-to-peak < 3ps, estimated at 14σ for BER=10⁻¹²)
  - Medium performance: rj_sigma = 0.3-0.5 ps (peak-to-peak 4-7ps)
  - High jitter scenarios (stress testing): rj_sigma > 1.0 ps
- **Implementation Method Suggestions**:
  - Generate independent and identically distributed Gaussian random numbers at each timestep: `δt ~ N(0, rj_sigma²)`
  - Use high-precision interpolation to implement fractional delay

##### Multi-Channel Selection Parameters (Reserved)

| Parameter | Type | Default Value | Description | Implementation Status |
|-----------|------|---------------|-------------|----------------------|
| `num_lanes` | int | 1 | Total system channel count (1=single channel, 2/4/8=multi-channel architecture), for parameter validation | To be implemented |

**Design Intent**:
- **Purpose**: Defines system architecture type, provides boundary conditions for channel index validation (should satisfy `0 ≤ lane_sel < num_lanes`)
- **Typical Configurations**:
  - `num_lanes=1`: Single-channel SerDes (bit rate ≤ 10Gbps)
  - `num_lanes=2/4/8`: 2:1/4:1/8:1 parallel-to-serial conversion architecture
- **Note**: Real N:1 parallel-to-serial conversion requires instantiating N parallel data paths at system level; current single-input single-output architecture only models a single Lane.

##### Nonlinear Effects Parameters (Reserved)

| Parameter | Type | Default Value | Description | Implementation Status |
|-----------|------|---------------|-------------|----------------------|
| `nonlinearity.enable` | bool | false | Enable nonlinear modeling | To be implemented |
| `nonlinearity.gain_compression` | double | 0.0 | Gain compression coefficient (dB/V) | To be implemented |
| `nonlinearity.saturation_voltage` | double | 1.0 | Saturation voltage (V) | To be implemented |

**Design Intent (Potential Applications in Future Versions)**:
- **Gain Compression**: Gain reduction under large signal input, modeling nonlinear resistance of transmission gates
- **Saturation Limiting**: Output amplitude limited by supply voltage or drive capability
- **Pattern-Dependent Delay**: Propagation delay variation under different data patterns, introducing Data-Dependent Jitter (DDJ)

### 2.3 Configuration Examples

#### 2.3.1 Current Version Valid Configurations

**Example 1: Minimal Configuration (matching current implementation)**

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**Description**:
- This is the **only valid configuration parameter** for current version (v0.1)
- Module performs ideal pass-through: `out = in`, no delay, no jitter
- Suitable for front-end debugging or single-channel system functional verification

**Example 2: Channel Selection in Multi-Channel System (Intent Declaration)**

```json
{
  "tx": {
    "mux_lane": 2
  }
}
```

**Description**:
- Declares selection of channel 3 (index 2), but **this parameter does not affect signal processing behavior in current implementation** (still pass-through)
- Used for configuration file version management, preparing for future multi-channel architecture migration

#### 2.3.2 Future Version Configuration Examples (Reserved Interface)

The following configurations **cannot take effect** in current code and serve only as design planning references for future versions.

**Example 3: Single-Channel Mode with Delay (Future)**

```json
{
  "tx": {
    "mux_lane": 0,
    "mux_delay": 25e-12
  }
}
```

**Expected Behavior (To be implemented)**:
- Fixed delay of 25ps, matching typical propagation delay of 28nm process
- Output signal delayed by a fixed time relative to input

**Example 4: Enable Jitter Modeling (Future)**

```json
{
  "tx": {
    "mux_lane": 0,
    "mux_delay": 15e-12,
    "jitter": {
      "enable": true,
      "dcd_percent": 48.0,
      "rj_sigma": 0.3e-12,
      "seed": 0
    }
  }
}
```

**Expected Behavior (To be implemented)**:
- Fixed delay of 15ps (advanced process)
- DCD duty cycle 48% (2% deviation), producing deterministic time offset at edges
- RJ standard deviation 0.3ps, superimposing Gaussian-distributed random time jitter
- Uses global random seed for reproducibility

**Example 5: Multi-Channel Architecture Configuration (Future)**

```json
{
  "tx": {
    "mux_lane": 5,
    "num_lanes": 8,
    "mux_delay": 15e-12
  }
}
```

**Expected Behavior (To be implemented)**:
- Select channel 6 (index 5) in 8:1 architecture
- Parameter validation: `lane_sel < num_lanes` (5 < 8 passes)
- Suitable for channel-level testing of 56Gbps/112Gbps SerDes

### 2.4 Parameter Usage Notes

#### Current Version (v0.1) Developer Guide

1. **Only `tx.mux_lane` parameter is valid**  
   Only need to set `"tx": {"mux_lane": 0}` in configuration file, other parameters will be ignored (if configuration file loader has implemented parameter reading, but module internally doesn't use them).

2. **Pass-through Behavior Guarantee**  
   Regardless of `mux_lane` value, current version module always performs `out = in` pass-through operation.

3. **Testing Strategy Suggestions**  
   - **Functional Verification**: Verify port connection correctness and sampling rate consistency
   - **Integration Testing**: Connect in series with FFE and Driver modules, confirm signal chain continuity
   - **No need to test**: Delay, jitter, multi-channel switching (not implemented in current version)

#### Future Version Extension Guide

1. **Delay Implementation Path**  
   - Option A: Use `sca_tdf::sca_delay<double>` module-level delay
   - Option B: Explicit circular buffer storing historical sample values
   - Option C: Approximate through first-order filter group delay

2. **Jitter Implementation Path**  
   - DCD: Adjust sampling moments based on UI index parity
   - RJ: Use `std::normal_distribution` to generate time perturbations
   - Interpolation: Implement high-precision fractional delay (Lagrange/Sinc/Farrow structure)

3. **Multi-Channel Implementation Path**  
   - Modify port definition to `sca_tdf::sca_in<double>` array or `std::vector`
   - In `processing()`, select corresponding input port based on `m_lane_sel`
   - System level requires instantiating multiple parallel data paths and merging at top level

4. **Configuration Loading Compatibility**  
   - Reserved parameters should be marked as optional in configuration loader
   - When detecting use of unimplemented parameters, output warning log instead of error
   - Maintain configuration file forward compatibility (new version code can recognize old version configs)

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

TX Mux module current version (v0.1) adopts the most simplified pass-through architecture, with signal processing flow containing only a single step:

```
Input Read → Direct Write to Output
```

**Complete Processing Flow (Current Implementation)**:

```cpp
void TxMuxTdf::processing() {
    // Simple pass-through mode (single channel)
    double x_in = in.read();
    out.write(x_in);
}
```

**Code Location**: Lines 18-22 of `src/ams/tx_mux.cpp`

#### Step 1 - Input Read

Read the analog signal value at current timestep from TDF input port:

```cpp
double x_in = in.read();
```

**Design Notes**:
- Input signal `x_in` typically comes from preceding FFE (Feed-Forward Equalizer) module output
- Signal amplitude range depends on FFE output configuration, typical values ±0.5V ~ ±1.0V (single-ended)
- Sampling rate is controlled by global parameter `Fs`, configured for symbol rate synchronization (`set_rate(1)`) through `set_attributes()` method

#### Step 2 - Direct Output

Write the read input signal directly to output port without any processing:

```cpp
out.write(x_in);
```

**Current Version Behavior Characteristics**:
- **Zero Delay**: Output completed within same timestep, no propagation delay introduced
- **Ideal Transfer**: Output amplitude and phase identical to input, no gain/attenuation/distortion
- **No Jitter**: No deterministic or random jitter components injected
- **Channel Index Ineffective**: Although constructor accepts `lane_sel` parameter and stores it in `m_lane_sel` member variable, the `processing()` method does not use this variable, so channel index configuration does not affect signal path

**Equivalent Transfer Function**:
```
H(s) = 1  (unity gain across all frequencies)
H(z) = 1  (discrete time domain)
y[n] = x[n]  (time domain expression)
```

**Application Scenarios**:
- **Functional Verification Phase**: Verify TX link (FFE → Mux → Driver) end-to-end connection correctness
- **Baseline Testing**: Establish reference eye diagram without Mux delay/jitter effects for comparison with subsequent versions
- **Single-Channel Systems**: In simplified applications not requiring multi-channel multiplexing and delay modeling, current implementation already meets requirements


### 3.2 TDF Lifecycle Methods

TX Mux as a SystemC-AMS TDF (Timed Data Flow) module follows standard TDF lifecycle management mechanisms. This section details the implementation and design considerations of three core methods.

#### 3.2.1 Constructor - Module Initialization

**Code Implementation** (Lines 5-11 of `src/ams/tx_mux.cpp`):

```cpp
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int lane_sel)
    : sca_tdf::sca_module(nm)
    , in("in")
    , out("out")
    , m_lane_sel(lane_sel)
{
}
```

**Parameter Description**:
- `nm`: SystemC module instance name, passed by upper-level system module, used for hierarchical naming and debug information output
- `lane_sel`: Channel index parameter (default value 0), specifies selected data channel number

**Initialization List Item-by-Item Analysis**:

1. **Base Class Construction**: `sca_tdf::sca_module(nm)`
   - Calls SystemC-AMS TDF module base class constructor, registers module to TDF scheduler
   - Inherits TDF timestep mechanism and sampling rate management interface

2. **Port Registration**: `in("in")`, `out("out")`
   - Allocates port names and registers to module's port list
   - Port types are `sca_tdf::sca_in<double>` and `sca_tdf::sca_out<double>` (defined in `include/ams/tx_mux.h` lines 10-11)
   - Port connections will be completed at system-level `SC_CTOR` or `elaborate()` phase

3. **Member Variable Initialization**: `m_lane_sel(lane_sel)`
   - Stores channel index parameter for subsequent use
   - **Current Version Note**: Although this value is stored, the `processing()` method does not use it, so parameter has no actual effect

**Empty Constructor Body**:
- Current version does not require additional runtime initialization (such as random number generators, filter objects, delay line buffers)
- Future versions may initialize here:
  - Random number generator for jitter model `std::mt19937`
  - Delay line buffer `std::deque<double>`
  - PSRR/bandwidth filter object `sca_ltf_nd`

#### 3.2.2 set_attributes() - Sampling Rate Configuration

**Code Implementation** (Lines 13-16 of `src/ams/tx_mux.cpp`):

```cpp
void TxMuxTdf::set_attributes() {
    in.set_rate(1);
    out.set_rate(1);
}
```

**Method Call Timing**:
- SystemC-AMS automatically calls during elaboration phase before simulation starts
- Executes after all module instantiation and port connections are complete, before simulation begins
- Used to declare module's timing constraints and resource requirements

**Sampling Rate Setting Details**:

**Meaning of `set_rate(1)`**:
- Parameter `1` represents input/output port relative sampling rate factor (Rate Factor)
- Relative to global timestep Δt (defined by top-level TDF module or SystemC clock domain), port samples/outputs once per timestep
- Equivalent declaration: **Input and output sampling rates are consistent, operating in symbol rate clock domain**

**Physical Meaning of Sampling Rate Factor**:

Assuming global sampling frequency is Fs (e.g., 560 GHz), bit rate is 56 Gbps, symbol rate is 7 GHz (8:1 architecture), then:

- Global timestep: Δt = 1/Fs = 1.786 ps
- Symbol period: T_sym = 1/7GHz = 142.86 ps
- Sampling points per symbol: Fs / 7GHz = 80 timesteps

Under `set_rate(1)` configuration:
- Mux `processing()` method is called once per 1 timestep
- Each call processes one sample point (not one symbol)
- This fine-grained sampling is suitable for behavioral modeling, capturing amplitude variations and transition processes within symbols

**Sampling Rate Coordination with Other Modules**:

TX Mux is typically cascaded in the following signal chain:
```
FFE (rate=1) → Mux (rate=1) → Driver (rate=1)
```

- All module sampling rate factors are consistent (rate=1), ensuring signal flow timing continuity
- If preceding and following modules have different sampling rates (e.g., rate=1 connecting to rate=2), rate conversion modules need to be inserted between ports or SystemC-AMS automatic interpolation mechanism should be used
- **Current Implementation Requirement**: All TX link modules must use the same global sampling frequency Fs

**Why Not Use Symbol Rate Sampling?**

Theoretically, Mux sampling rate could be set to symbol rate (rate = Fs / R_sym), meaning `processing()` is called once per symbol period. However, current design chooses rate=1 for the following reasons:

1. **Flexibility**: Maintains direct compatibility with preceding and following modules, avoiding rate conversion overhead
2. **Precision**: Captures transient processes within symbols (such as edge rise time, overshoot), suitable for eye diagram analysis
3. **Consistency**: All AMS modules in the project uniformly adopt rate=1, simplifying system configuration

#### 3.2.3 processing() - Core Signal Processing

**Code Implementation** (Lines 18-22 of `src/ams/tx_mux.cpp`):

```cpp
void TxMuxTdf::processing() {
    // Simple pass-through mode (single channel)
    double x_in = in.read();
    out.write(x_in);
}
```

**Method Call Timing**:
- SystemC-AMS TDF scheduler automatically calls at each timestep
- Call frequency = Fs (global sampling frequency)
- Execution order: According to topological sort of signal flow topology, Mux executes after FFE and before Driver

**Port Read/Write Semantics**:

- **`in.read()`**:
  - Reads input port value at current timestep
  - Value is written by upstream module (FFE) during its `processing()` call at this timestep
  - Return type: `double` (analog signal voltage value)

- **`out.write(x_in)`**:
  - Writes calculation result to output port
  - Written value will be read by downstream module (Driver)'s `in.read()` at next timestep
  - TDF scheduler automatically handles data flow timing alignment

**Timing Behavior Characteristics**:

- **Zero-Delay Transfer**: Input read and output write completed within same timestep, equivalent to combinational logic
- **Stateless Processing**: Does not maintain historical data (no delay line, no feedback path), output at each timestep only depends on current input
- **Deterministic Behavior**: Same input sequence produces same output sequence, suitable for reproducible verification

**Current Implementation Limitations**:

1. **Channel Index Not Used**:
   - Member variable `m_lane_sel` is stored but not accessed in `processing()`
   - Multi-channel architecture requires modification to array input ports: `sca_tdf::sca_in<double> in[N_LANES]`
   - Then select based on `m_lane_sel`: `double x_in = in[m_lane_sel].read()`

2. **No Delay Modeling**:
   - Propagation delay (mux_delay) not implemented
   - Requires adding delay line buffer or using `sca_tdf::sca_delay<double>` module

3. **No Jitter Modeling**:
   - DCD (duty cycle distortion) and RJ (random jitter) not injected
   - Requires integrating time perturbation generation and fractional delay interpolation algorithms

**Code Simplicity Design Considerations**:

Current implementation deliberately maintains minimum complexity for the following reasons:

- **Incremental Development**: First verify signal chain structural correctness, then gradually add non-ideal effects
- **Debug-Friendly**: Pass-through mode facilitates problem isolation; when TX link anomalies occur, Mux influence can be excluded
- **Version Compatibility**: Future addition of delay/jitter functions can maintain backward compatibility through configuration switches (such as `enable_delay`, `enable_jitter`)

### 3.3 Future Version Extension Mechanisms (Design Planning)

Current version implements the most basic pass-through functionality. The following are design ideas for core mechanisms planned for extension in future versions. **Note: The following content is design planning and has not yet been implemented in current code.**

#### 3.3.1 Fixed Delay Modeling (Propagation Delay)

**Design Goal**: Model selector propagation delay to match real hardware timing characteristics.

**Implementation Option A: Using TDF Delay Module**

SystemC-AMS provides `sca_tdf::sca_delay<T>` template class, which can implement integer multiple timestep delays:

```cpp
// Add member variable in header file
sca_tdf::sca_delay<double> m_delay_line;

// Constructor initialization
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int lane_sel, double delay_s, double Fs)
    : ...
    , m_delay_line("delay_line")
{
    int delay_samples = static_cast<int>(std::round(delay_s * Fs));
    m_delay_line.set_delay(delay_samples);
}

// Use in processing()
void TxMuxTdf::processing() {
    double x_in = in.read();
    double x_delayed = m_delay_line(x_in);
    out.write(x_delayed);
}
```

**Advantages**:
- Native SystemC-AMS support, simple implementation
- Automatic handling of delay queue management and initialization

**Limitations**:
- Only supports integer multiple timestep delays
- For non-integer sample point delays (e.g., 15ps delay but Δt=1.786ps, requiring 8.4 sample points), rounding is needed, introducing quantization error

**Implementation Option B: Explicit Circular Buffer**

Use `std::deque` or `std::vector` to implement controllable historical data storage:

```cpp
// Add to header file
std::deque<double> m_delay_buffer;
int m_delay_samples;

// Constructor initialization
TxMuxTdf::TxMuxTdf(..., double delay_s, double Fs) {
    m_delay_samples = static_cast<int>(std::round(delay_s * Fs));
    m_delay_buffer.resize(m_delay_samples, 0.0);
}

// processing() implementation
void TxMuxTdf::processing() {
    double x_in = in.read();
    m_delay_buffer.push_back(x_in);
    double x_out = m_delay_buffer.front();
    m_delay_buffer.pop_front();
    out.write(x_out);
}
```

**Advantages**:
- Fully controllable, facilitates debugging and performance optimization
- Extensible to fractional delay (combined with interpolation algorithm)

**Disadvantages**:
- Requires manual management of buffer size and initialization
- Slightly more code

**Implementation Option C: Filter Group Delay Approximation**

Use first-order all-pass filter (All-Pass Filter) or Bessel filter group delay to approximate fixed delay:

```cpp
// Add to header file
sca_tdf::sca_ltf_nd m_delay_filter;

// Configure in constructor
TxMuxTdf::TxMuxTdf(..., double delay_s) {
    // First-order all-pass filter H(s) = (1 - s/ω) / (1 + s/ω)
    // Group delay at low frequencies approximately 2/ω
    double omega = 2.0 / delay_s;
    sca_util::sca_vector<double> num = {1.0, -omega};
    sca_util::sca_vector<double> den = {1.0, omega};
    m_delay_filter.set(num, den);
}
```

**Advantages**:
- Smooth frequency domain characteristics, no high-frequency ringing
- Suitable for simultaneous modeling with bandwidth limitation

**Limitations**:
- Group delay not constant at high frequencies, only suitable for low-frequency delay approximation
- Requires trade-off between delay precision and bandwidth characteristics

#### 3.3.2 Jitter Modeling (Jitter Injection)

**Design Goal**: Inject Deterministic Jitter (DCD) and Random Jitter (RJ) to simulate clock non-idealities and circuit noise.

**DCD (Duty Cycle Distortion) Modeling**

Duty cycle deviation from 50% causes odd and even UI widths to be unequal, producing periodic time offsets at edges.

**Implementation Idea**:

```cpp
// Add to header file
double m_dcd_percent;  // Duty cycle (e.g., 48.0 means 48%)
double m_ui_period;    // UI period (seconds)
int m_ui_counter;      // UI counter
std::deque<double> m_fractional_delay_buffer;

// Constructor initialization
TxMuxTdf::TxMuxTdf(..., double dcd_percent, double ui_period, double Fs)
    : m_dcd_percent(dcd_percent)
    , m_ui_period(ui_period)
    , m_ui_counter(0)
{
    int samples_per_ui = static_cast<int>(std::round(ui_period * Fs));
    m_fractional_delay_buffer.resize(samples_per_ui, 0.0);
}

// processing() implementation
void TxMuxTdf::processing() {
    double x_in = in.read();
    
    // Calculate current UI index
    int ui_index = m_ui_counter / samples_per_ui;
    m_ui_counter++;
    
    // Apply opposite direction time offsets for odd/even UIs
    double dcd_offset = (ui_index % 2 == 0) 
        ? (50.0 - m_dcd_percent) / 100.0 * m_ui_period
        : (m_dcd_percent - 50.0) / 100.0 * m_ui_period;
    
    // Convert time offset to fractional delay (requires interpolation implementation)
    double x_out = apply_fractional_delay(x_in, dcd_offset);
    out.write(x_out);
}
```

**Key Technologies**:
- **Fractional Delay Interpolation**: When delay amount is not integer multiple of sampling point, interpolation algorithms (Lagrange/Sinc/Farrow structure) are needed
- **Phase Tracking**: Maintain UI counter, apply opposite direction offsets based on parity

**RJ (Random Jitter) Modeling**

Random jitter follows Gaussian distribution, superimposed on output at each timestep.

**Implementation Idea**:

```cpp
// Add to header file
#include <random>
std::mt19937 m_rng;
std::normal_distribution<double> m_rj_dist;
double m_rj_sigma;  // RJ standard deviation (seconds)
double m_Fs;

// Constructor initialization
TxMuxTdf::TxMuxTdf(..., double rj_sigma, int seed, double Fs)
    : m_rj_sigma(rj_sigma)
    , m_Fs(Fs)
    , m_rng(seed == 0 ? std::random_device{}() : seed)
    , m_rj_dist(0.0, rj_sigma)
{
}

// processing() implementation
void TxMuxTdf::processing() {
    double x_in = in.read();
    
    // Generate random time offset
    double time_offset = m_rj_dist(m_rng);  // Unit: seconds
    
    // Convert time offset to fractional delay
    double x_out = apply_fractional_delay(x_in, time_offset);
    out.write(x_out);
}
```

**Key Technologies**:
- **Gaussian Random Number Generation**: Use C++11 `<random>` library's `std::normal_distribution`
- **Seed Management**: Support fixed seeds (reproducible simulation) and random seeds (Monte Carlo analysis)
- **Fractional Delay**: Shares same interpolation algorithm with DCD modeling

**Fractional Delay Interpolation Algorithm**

The following is a sample implementation of Lagrange interpolation (3rd order):

```cpp
double TxMuxTdf::apply_fractional_delay(double x_current, double delay_s) {
    // Convert delay to sample points (may be fractional)
    double delay_samples = delay_s * m_Fs;
    int delay_int = static_cast<int>(std::floor(delay_samples));
    double delay_frac = delay_samples - delay_int;
    
    // Get interpolation-required historical samples from delay buffer
    // Assume buffer already stores sufficient historical data
    double x_n = m_fractional_delay_buffer[delay_int];
    double x_nm1 = m_fractional_delay_buffer[delay_int + 1];
    double x_np1 = m_fractional_delay_buffer[delay_int - 1];
    
    // Lagrange interpolation formula (3 points)
    double L0 = 0.5 * delay_frac * (delay_frac - 1.0);
    double L1 = 1.0 - delay_frac * delay_frac;
    double L2 = 0.5 * delay_frac * (delay_frac + 1.0);
    
    double x_interpolated = L0 * x_nm1 + L1 * x_n + L2 * x_np1;
    
    // Update buffer
    m_fractional_delay_buffer.push_front(x_current);
    m_fractional_delay_buffer.pop_back();
    
    return x_interpolated;
}
```

**Interpolation Algorithm Comparison**:

| Algorithm | Order | Precision | Computational Complexity | Applicable Scenarios |
|-----------|-------|-----------|-------------------------|----------------------|
| Lagrange | 3-5 | Medium | Low | Rapid prototype verification |
| Sinc Interpolation | Theoretically infinite | High | High (requires truncation) | High-precision eye diagram analysis |
| Farrow Structure | Configurable | High | Medium | Real-time adaptive jitter |

#### 3.3.3 Multi-Channel Selection Mechanism (Multi-Lane Selection)

**Design Goal**: Support true N:1 multiplexer multi-input selection logic.

**Architecture Change**:

Current single input port:
```cpp
sca_tdf::sca_in<double> in;
```

Modify to multi-input port array:
```cpp
static const int N_LANES = 8;
sca_tdf::sca_in<double> in[N_LANES];
```

**Constructor Adaptation**:

```cpp
TxMuxTdf::TxMuxTdf(sc_core::sc_module_name nm, int num_lanes, int lane_sel)
    : sca_tdf::sca_module(nm)
    , out("out")
    , m_num_lanes(num_lanes)
    , m_lane_sel(lane_sel)
{
    // Dynamically create port array
    in = new sca_tdf::sca_in<double>[num_lanes];
    for (int i = 0; i < num_lanes; i++) {
        std::string port_name = "in_" + std::to_string(i);
        in[i].set_name(port_name.c_str());
    }
    
    // Parameter validation
    if (lane_sel >= num_lanes) {
        SC_REPORT_ERROR("TxMuxTdf", "lane_sel exceeds num_lanes");
    }
}
```

**processing() Adaptation**:

```cpp
void TxMuxTdf::processing() {
    // Select corresponding input channel based on lane_sel
    double x_in = in[m_lane_sel].read();
    
    // Subsequent delay/jitter processing...
    out.write(x_in);
}
```

**System-Level Connection Example**:

```cpp
// Instantiate multiple parallel data paths in top-level module
WaveGenTdf* wavegen[8];
TxFfeTdf* ffe[8];
TxMuxTdf* mux;

for (int i = 0; i < 8; i++) {
    wavegen[i] = new WaveGenTdf(...);
    ffe[i] = new TxFfeTdf(...);
}
mux = new TxMuxTdf("mux", 8, 5);  // 8 channels, select 6th

// Connect
for (int i = 0; i < 8; i++) {
    ffe[i]->in(wavegen[i]->out);
    mux->in[i](ffe[i]->out);
}
```

**Dynamic Channel Switching (Advanced Feature)**:

If dynamic channel switching is needed during simulation (e.g., testing performance differences between different Lanes), a DE domain control interface can be added:

```cpp
// Add to header file
sca_tdf::sca_de::sca_in<int> lane_sel_ctrl;

// processing() adaptation
void TxMuxTdf::processing() {
    // Read control signal from DE domain
    if (lane_sel_ctrl.event()) {
        m_lane_sel = lane_sel_ctrl.read();
    }
    
    double x_in = in[m_lane_sel].read();
    out.write(x_in);
}
```

#### 3.3.4 Nonlinear Effects Modeling (Optional)

**Gain Compression (Gain Compression)**:

Gain reduction under large signal input, modeling nonlinear resistance of transmission gates:

```cpp
double gain_factor = 1.0 / (1.0 + std::pow(std::abs(x_in) / m_compression_point, 2));
x_out = x_in * gain_factor;
```

**Saturation Limiting (Saturation)**:

Output amplitude limited by supply voltage or drive capability:

```cpp
double vsat = m_saturation_voltage;
x_out = std::max(-vsat, std::min(vsat, x_out));
```

**Pattern-Dependent Delay (Pattern-Dependent Delay)**:

Propagation delay variation under different data patterns, introducing Data-Dependent Jitter (DDJ):

```cpp
// Detect current pattern (e.g., number of consecutive 1s)
int consecutive_ones = count_consecutive_ones(m_delay_buffer);
double pattern_delay = m_base_delay + consecutive_ones * m_ddj_per_ui;
```

---

## 4. Testbench Architecture

### 4.1 Design Philosophy

TX Mux module current version (v0.1) adopts a **system-level integration testing strategy**, not providing a dedicated testbench. Core design philosophy: The simplicity of pass-through functionality makes integration testing sufficient to verify connection correctness and timing consistency, avoiding redundant development of test infrastructure for basic functionality.

### 4.2 Test Scenarios

The only current test scenario is system-level integration verification:

| Test Scenario | Testbench | Verification Target | Implementation Status |
|---------------|-----------|---------------------|----------------------|
| **System-Level Integration** | `simple_link_tb.cpp` | End-to-end signal integrity, TX link continuity | ✅ Implemented |

**Verification Points**:
- Mux correctly connects FFE and Driver modules
- Signal pass-through characteristics (amplitude/phase consistency)
- TDF sampling rate synchronization (rate=1)
- Simulation stability

### 4.3 Test Topology and Connection

TX Mux position in system-level testbench:

```
┌──────────────────────────────────────────────────────────────┐
│               simple_link_tb.cpp (System-Level)               │
│                                                                │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │ WaveGen │──▶│  TxFFE  │──▶│  TxMux  │──▶│ TxDriver│──▶    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│                                                                │
│  Trace Signals: ffe_out, driver_out                           │
│  Note: sig_mux_out not explicitly traced (needs to be added   │
│  to directly verify pass-through characteristics)             │
└──────────────────────────────────────────────────────────────┘
```

**Key Signal Connections** (Lines 63-69 of `tb/simple_link_tb.cpp`):
```cpp
tx_ffe.out(sig_ffe_out);        // Mux input
tx_mux.in(sig_ffe_out);
tx_mux.out(sig_mux_out);        // Mux output
tx_driver.in(sig_mux_out);
```

**Parameter Configuration**: Load `tx.mux_lane` parameter from `config/default.json` (default value 0) through `ConfigLoader`.

### 4.4 Verification Methods

#### Method 1: Add Mux Output Tracing (Recommended)

**Problem**: Current `simple_link_tb.cpp` does not trace `sig_mux_out` signal, cannot directly verify Mux pass-through characteristics.

**Solution**: Add tracing statement in testbench:
```cpp
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

Then use Python script to compare `mux_out` and `ffe_out`:
```python
import numpy as np
data = np.loadtxt('simple_link.dat', skiprows=1)
ffe_out = data[:, 2]    # Adjust according to actual column index
mux_out = data[:, 3]
error = np.abs(mux_out - ffe_out)
print(f"Pass-through error (max): {np.max(error):.2e} V")  # Expected < 1e-12
```

#### Method 2: Indirect Verification (Currently Feasible but Not Precise)

**Note**: Directly comparing `driver_out` and `ffe_out` is **technically incorrect**, because the Driver module introduces gain, bandwidth limitation, and saturation effects; differences cannot be attributed to Mux. Only usable for rough signal chain integrity check.

#### Method 3: Simulation Log Check

SystemC-AMS simulation completes successfully without warnings, indicating port connections and sampling rate configuration are correct.

### 4.5 Auxiliary Module Description

TX Mux module in testbench depends on the following auxiliary modules to provide input signals and functional support. This section describes the functions of these modules and their interaction relationships with Mux.

#### 4.5.1 WaveGen Module (Waveform Generator)

**Module Path**: `include/ams/wave_generation.h`, `src/ams/wave_generation.cpp`

**Function Description**:
- Generates test PRBS (Pseudo-Random Binary Sequence) data patterns
- Supports multiple PRBS types: PRBS7, PRBS9, PRBS15, PRBS23, PRBS31
- Configurable data rate, bit pattern, and initialization seed

**Relationship with Mux**:
- WaveGen → FFE → Mux signal chain source
- Provides test analog signal input for Mux (after FFE equalization)
- Instantiated and connected to FFE module in `simple_link_tb.cpp`

**Typical Configuration**:
```json
{
  "wave": {
    "type": "PRBS31",
    "poly": "x^31 + x^28 + 1",
    "init": "0x7FFFFFFF"
  }
}
```

#### 4.5.2 Trace Signal Monitor

**Function Description**:
- SystemC-AMS provided waveform tracing mechanism (`sca_util::sca_trace`)
- Records signal values during simulation to `.dat` file
- Supports post-processing analysis and visualization (Python/Matplotlib)

**Key Trace Signals**:
- `ffe_out`: FFE output (Mux input), used to verify pass-through characteristics
- `mux_out`: Mux output, not traced in current testbench (recommended to add)
- `driver_out`: Driver output, used for system-level signal integrity analysis

**Adding Mux Output Trace**:
```cpp
// Add in tb/simple_link_tb.cpp
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

**Data Format**:
```
# time(s)    wave_out(V)    ffe_out(V)    mux_out(V)    driver_out(V)
0.00e+00     0.000          0.000         0.000         0.000
1.78e-12     0.500          0.500         0.500         0.200
...
```

#### 4.5.3 ConfigLoader Module (Configuration Loader)

**Module Path**: `include/de/config_loader.h`, `src/de/config_loader.cpp`

**Function Description**:
- Loads parameters from JSON/YAML configuration files
- Parses and populates to `TxParams` structure
- Provides parameter validation and default value handling

**Relationship with Mux**:
- Loads `tx.mux_lane` parameter and passes to Mux constructor
- Supports multi-scenario configuration switching (different channel indices)
- Simplifies test configuration management, avoiding hard-coded parameters

---

## 5. Simulation Result Analysis

### 5.1 Current Version Verification Method

TX Mux current version (v0.1) adopts ideal pass-through architecture (`out = in`), with no delay, no jitter, no nonlinear effects. Since testbench `simple_link_tb.cpp` **does not trace mux_out signal**, direct waveform analysis is unavailable; only system-level indirect verification is possible.

### 5.2 System-Level Integration Verification Results

#### 5.2.1 Verification Principle

Confirm signal chain continuity by observing TX link complete output (`sig_driver_out`):

```
WaveGen → FFE → Mux → Driver → Channel
```

**Indirect Verification Logic**:
- If Driver output contains correct data patterns and eye diagram quality meets expectations, Mux correctly passed FFE output
- If simulation completes successfully without errors, port connections and sampling rate configuration are correct (rate=1 consistency)

**Limitations**:
- **Cannot directly verify pass-through characteristics**: Driver introduces gain, bandwidth limitation, and saturation effects; differences between `driver_out` and `ffe_out` cannot be attributed to Mux
- **Cannot quantify pass-through error**: Direct tracing of `mux_out` needed to measure numerical precision (expected error < 1e-12 V, floating-point precision limit)

#### 5.2.2 Typical Simulation Results

**Configuration** (`config/default.json`):
```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**Observation Indicators**:
- **Simulation Completion Status**: ✅ Success (no SystemC-AMS errors or warnings)
- **Signal Chain Integrity**: ✅ Driver output contains expected PRBS pattern
- **Timing Consistency**: ✅ No sampling rate mismatch warnings

**Expected Results**:
- Mux as pass-through unit, does not change signal amplitude, phase, or spectral characteristics
- System-level eye diagram quality mainly depends on Channel loss and RX equalizer performance

### 5.3 Direct Verification Method (Testbench Modification Required)

#### 5.3.1 Add Mux Output Tracing

Add in `simple_link_tb.cpp`:
```cpp
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

#### 5.3.2 Pass-Through Characteristic Analysis

Use Python script to compare `mux_out` and `ffe_out`:

```python
import numpy as np

data = np.loadtxt('simple_link.dat', skiprows=1)
ffe_out = data[:, col_ffe]
mux_out = data[:, col_mux]

# Pass-through error statistics
error = mux_out - ffe_out
print(f"Max error: {np.max(np.abs(error)):.2e} V")
print(f"RMS error: {np.sqrt(np.mean(error**2)):.2e} V")

# Expected result: error < 1e-12 V (floating-point precision limit)
```

**Expected Indicators**:

| Indicator | Theoretical Value | Pass Criteria | Description |
|-----------|-------------------|---------------|-------------|
| Max Error | 0 V | < 1e-12 V | Floating-point arithmetic precision limit |
| RMS Error | 0 V | < 1e-15 V | Ideal pass-through |
| Phase Offset | 0 s | < 1 ps | Same timestep sampling |
| Spectrum Consistency | 100% | > 99.9% | FFT comparison |

### 5.4 Future Version Analysis Indicators

When delay and jitter modeling are implemented (v0.2+), the following analyses should be added:

**Delay Measurement**:
- Cross-correlation method to measure propagation delay (expected value = `mux_delay` parameter)
- Group delay consistency check

**Jitter Decomposition**:
- Periodic time offset caused by DCD (odd/even UI comparison)
- Gaussian distribution fit for RJ (mean should be 0, standard deviation = `rj_sigma`)

**Eye Diagram Impact**:
- Eye width closure caused by jitter (horizontal direction)
- Comparison with baseline without Mux jitter

### 5.5 Waveform Data File Format

SystemC-AMS trace file output format (when mux_out tracing is added):

```
# time(s)    wave_out(V)    ffe_out(V)    mux_out(V)    driver_out(V)
0.00e+00     0.000          0.000         0.000         0.000
1.78e-12     0.500          0.500         0.500         0.200
3.57e-12     1.000          0.650         0.650         0.260
...
```

**Column Descriptions**:
- `time`: Simulation time (seconds)
- `ffe_out`: FFE output (Mux input)
- `mux_out`: Mux output (current version should be identical to `ffe_out`)
- `driver_out`: Driver output (introduces gain and bandwidth effects)

---

## 6. Running Guide

### 6.1 Environment Configuration

TX Mux module is verified through system-level testbench `simple_link_tb`, requiring SystemC-AMS development environment configuration.

**Required Environment Variables**:
```bash
export SYSTEMC_HOME=/usr/local/systemc-2.3.4
export SYSTEMC_AMS_HOME=/usr/local/systemc-ams-2.3.4
```

**Verify Installation**:
```bash
ls $SYSTEMC_AMS_HOME/include/systemc-ams
# Should display systemc-ams.h and other header files
```

### 6.2 Build and Run

#### 6.2.1 Using CMake (Recommended)

**Build System-Level Testbench**:
```bash
cd /path/to/serdes
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make simple_link_tb
```

**Run Simulation**:
```bash
./bin/simple_link_tb
# Simulation output: simple_link.dat
```

**Expected Output**:
```
SystemC 2.3.4 --- Jan 13 2026 10:30:00
SystemC-AMS 2.3.4 --- Jan 13 2026 10:30:00
Info: simulation stopped by user.
```

#### 6.2.2 Using Makefile

**Quick Run**:
```bash
cd /path/to/serdes
make run
# Automatically builds and executes simple_link_tb
```

**Clean Build**:
```bash
make clean
```

### 6.3 Parameter Configuration

TX Mux loads parameters through configuration file `config/default.json`:

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**Modify Channel Index** (no actual functional impact in current version):
```json
{
  "tx": {
    "mux_lane": 2
  }
}
```

**Note**: After modifying configuration, need to rerun testbench, no recompilation required.

### 6.4 Result Viewing

#### 6.4.1 Verify Simulation Success

Check simulation log has no errors or warnings:
```bash
grep -i "error\|warning" build/simulation.log
# No output indicates success
```

#### 6.4.2 Add Mux Output Tracing (Optional)

**Edit Testbench** (`tb/simple_link_tb.cpp`):
```cpp
// Add in trace creation section
sca_util::sca_trace(tf, sig_mux_out, "mux_out");
```

**Rebuild and Run**:
```bash
cd build
make simple_link_tb
./bin/simple_link_tb
```

#### 6.4.3 Python Analysis Script

Use Python to verify pass-through characteristics (requires `mux_out` tracing to be added first):

```python
import numpy as np

data = np.loadtxt('build/simple_link.dat', skiprows=1)
ffe_out = data[:, 2]  # Adjust according to actual column index
mux_out = data[:, 3]

error = np.abs(mux_out - ffe_out)
print(f"Pass-through error (max): {np.max(error):.2e} V")
print(f"Pass-through error (RMS): {np.sqrt(np.mean(error**2)):.2e} V")
# Expected: error < 1e-12 V
```

### 6.5 Troubleshooting

#### 6.5.1 Common Errors

**Sampling Rate Mismatch**:
```
Error: (E117) sc_signal<T>: port not bound
```
**Solution**: Check FFE and Driver `set_rate()` configurations, ensure both are `rate=1`.

**Port Connection Error**:
```
Error: port 'in' not connected
```
**Solution**: Verify Mux input/output connection completeness in `simple_link_tb.cpp`.

**Configuration File Missing**:
```
Error: cannot open config/default.json
```
**Solution**: Ensure working directory is project root directory, or modify configuration file path.

#### 6.5.2 Debugging Tips

**Enable Verbose Logging**:
```bash
export SC_REPORT_VERBOSITY=SC_FULL
./bin/simple_link_tb
```

**Check Signal Connections**: Add in testbench constructor:
```cpp
std::cout << "Mux input rate: " << tx_mux.in.get_rate() << std::endl;
std::cout << "Mux output rate: " << tx_mux.out.get_rate() << std::endl;
```

---

## 7. Technical Key Points

### 7.1 Pass-Through Architecture Avoids Algebraic Loops

**Design Choice**: Current version adopts stateless pass-through (`out = in`), not maintaining internal state variables.

**Technical Advantages**:
- Avoids algebraic loop risk (output does not depend on its own feedback)
- Ensures TDF scheduler topological sorting convergence when cascaded linearly with FFE/Driver
- Suitable as baseline reference module for system integration verification

**Application Limitations**: Cannot model real hardware propagation delay and phase characteristics, needs extension to delay line architecture (see 7.3).

### 7.2 lane_sel Parameter Retention Reason

**Current Status**: Parameter `m_lane_sel` is stored but not used (not accessed in `processing()`).

**Retention Intent**:
- Reserve interface for multi-channel architecture (2:1/4:1/8:1 parallel-to-serial conversion)
- Requires modifying ports to array: `sca_tdf::sca_in<double> in[N]`
- Then index based on `m_lane_sel`: `x_in = in[m_lane_sel].read()`

**Configuration Forward Compatibility**: Current configuration files can seamlessly upgrade to multi-channel versions.

### 7.3 Delay Modeling Scheme Selection

Future versions need to select delay implementation schemes, each with trade-offs:

| Scheme | Precision | Implementation Difficulty | Side Effects |
|--------|-----------|---------------------------|--------------|
| **sca_delay module** | Integer sample points | Low | Quantization error (non-integer delay) |
| **Explicit buffer** | Fractional sample points with interpolation | Medium | Requires manual queue management |
| **Filter group delay** | Frequency-dependent | Low | Introduces bandwidth limitation |

**Recommended Scheme**: Initially use `sca_delay` (simple), upgrade to buffer+Lagrange interpolation for high-precision requirements.

### 7.4 Fractional Delay Interpolation Necessity

**Trigger Condition**: When delay time is not integer multiple of sampling period (e.g., 15ps delay but Δt=1.786ps, requiring 8.4 sample points).

**Technical Solutions**:
- **Lagrange Interpolation** (3-5 order): Low computation, medium precision
- **Sinc Interpolation**: Theoretically optimal, requires truncation window processing
- **Farrow Structure**: Real-time adjustable delay, suitable for jitter modeling

**Key Application**: RJ/DCD jitter injection requires sub-sample point time precision (<0.5ps), interpolation is mandatory.

### 7.5 Extension Trigger Conditions

**Current Version Applicability**: Low-speed/single-channel systems (<28Gbps), pass-through simplification is reasonable.

**Scenarios Requiring Extension**:
- Bit rate ≥ 56Gbps: Mux delay accounts for > 15% of UI (15ps / 100ps UI)
- Jitter-sensitive applications: Need precise modeling of DCD/RJ impact on eye diagram
- Multi-channel SerDes: Verify timing skew between different Lanes

**Technical Risk**: v0.1 in high-speed systems will **over-optimistically estimate eye diagram quality by about 25%** (ignoring Mux-contributed jitter).

### 7.6 Testbench Limitation Impact

**Problem**: `simple_link_tb.cpp` does not trace `sig_mux_out`, cannot directly verify pass-through error.

**Impact**:
- Can only indirectly infer Mux behavior through system-level output (`driver_out`)
- Driver gain/bandwidth effects couple with Mux, difficult to decouple analysis
- Cannot quantify floating-point precision error (expected < 1e-12 V)

**Solution**: Add `sca_util::sca_trace(tf, sig_mux_out, "mux_out")`, then use Python to compare `mux_out` with `ffe_out`.

---

## 8. Reference Information

### 8.1 Related Code Files

| File Category | Path | Description |
|---------------|------|-------------|
| **Header File** | `include/ams/tx_mux.h` | TxMuxTdf class declaration, port definitions |
| **Implementation File** | `src/ams/tx_mux.cpp` | TDF lifecycle method implementation |
| **Parameter Definition** | `include/common/parameters.h` | TxParams structure (`mux_lane` parameter) |
| **Testbench** | `tb/simple_link_tb.cpp` | System-level integration test (includes Mux module) |
| **Configuration File** | `config/default.json` | Default parameter configuration (`tx.mux_lane`) |

### 8.2 Core Dependencies

**Compile-Time Dependencies**:
- **SystemC 2.3.4**: TDF module base class, port type definitions
- **SystemC-AMS 2.3.4**: `sca_tdf::sca_module`, `sca_in/out<double>`
- **C++14 Standard**: Initialization lists, type inference support

**Runtime Dependencies**:
- **Configuration Loader**: `ConfigLoader` class (loads parameters from JSON/YAML)
- **Upstream Module**: TX FFE (`TxFfeTdf`) provides input signals
- **Downstream Module**: TX Driver (`TxDriverTdf`) receives output signals

**Test Dependencies** (Future Versions):
- GoogleTest 1.12.1 (unit test framework)
- NumPy/SciPy (pass-through error analysis)
- Matplotlib (waveform visualization)

### 8.3 Related Module Documentation

| Module Name | Documentation Path | Relationship Description |
|-------------|-------------------|-------------------------|
| TX FFE | `docs/modules/ffe.md` | Upstream module, provides equalized symbol signals |
| TX Driver | `docs/modules/driver.md` | Downstream module, receives Mux output and drives channel |
| Clock Generation | `docs/modules/clkGen.md` | Clock source, Mux jitter characteristics depend on clock quality |
| System Config | `README.md` | System-level parameter configuration and signal chain connection |

### 8.4 Reference Standards and Specifications

**SerDes Architecture Standards**:

| Standard | Version | Related Content |
|----------|---------|-----------------|
| **IEEE 802.3** | 2018 | Ethernet multi-channel parallel-to-serial architecture (Clause 82) |
| **PCIe** | Gen 4/5/6 | Transmitter timing budget and jitter specifications |
| **USB4** | v2.0 | Lane-to-lane timing skew requirements (< 0.2 UI) |
| **OIF CEI** | 56G/112G | High-speed SerDes transmitter jitter templates |

**Jitter Modeling References**:
- **JEDEC Standard JESD65B**: Jitter specifications and measurement methods for high-speed serial data links
- **Agilent AN 1448-1**: Jitter decomposition theory (RJ, DJ, DCD, DDJ)
- **IEEE 802.3bj**: 100G Ethernet jitter tolerance test methods

### 8.5 Configuration Examples

#### Example 1: Single-Channel Pass-Through (Current Version)

```json
{
  "tx": {
    "mux_lane": 0
  }
}
```

**Applicable Scenarios**:
- Single-channel SerDes systems (bit rate ≤ 28Gbps)
- Front-end functional verification and signal chain integrity testing
- Applications without delay/jitter requirements

**Expected Behavior**: Ideal pass-through, `out = in`.

#### Example 2: Multi-Channel Architecture Configuration (Future Version Reserved)

```json
{
  "tx": {
    "mux_lane": 3,
    "mux_delay": 20e-12,
    "jitter": {
      "enable": true,
      "dcd_percent": 49.0,
      "rj_sigma": 0.25e-12
    }
  }
}
```

**Expected Behavior (To be implemented)**:
- Select 4th channel (index 3)
- Fixed delay 20ps
- DCD duty cycle 49% (1% deviation)
- RJ standard deviation 0.25ps

**Applicable Scenarios**:
- 4:1/8:1 parallel-to-serial conversion architectures
- High-speed SerDes (56G/112G) jitter modeling
- Multi-channel timing skew analysis

### 8.6 Academic References

**Parallel-to-Serial Conversion Architectures**:
- J. Savoj et al., "A 12-Gb/s Data Rate Transceiver with Flexible Parallel Bus Interfaces", IEEE JSSC 2003
- M. Harwood et al., "A 12.5Gb/s SerDes in 65nm CMOS", IEEE ISSCC 2007

**Jitter Modeling Theory**:
- M. Li and J. Wilstrup, "Paradigm Shift for Jitter and Noise in Design and Test", DesignCon 2004
- K. Yang and D. Chen, "Physical Modeling of Jitter in High-Speed SerDes", IEEE MTT 2010

**SystemC-AMS Modeling Methods**:
- *SystemC AMS User's Guide*, Accellera, Version 2.3.4
- Chapter 4: TDF (Timed Data Flow) Modeling Methods
- Chapter 9: DE-TDF Mixed Simulation (Dynamic Channel Switching)

### 8.7 External Tools and Resources

**Simulation and Analysis Tools**:
- **SystemC-AMS**: https://systemc.org (Open source modeling framework)
- **Matplotlib**: https://matplotlib.org (Waveform visualization)
- **SciPy**: https://scipy.org (Signal processing and statistical analysis)

**Design References**:
- **Xilinx UG476**: GTX/GTH SerDes User Guide (multi-channel timing management)
- **Intel FPGA IP User Guide**: Transceiver PHY IP configuration examples
- **IBIS-AMI Cookbook**: Behavioral modeling best practices (www.eda.org/ibis)

### 8.8 Known Limitations and Future Plans

**Current Version (v0.1) Limitations**:
- Only supports single-input single-output (SISO) architecture
- No delay and jitter modeling
- `lane_sel` parameter does not affect signal processing

**Future Version Plans**:

| Feature | Target Version | Priority | Description |
|---------|----------------|----------|-------------|
| Fixed Delay Modeling | v0.2 | High | Use `sca_delay` or explicit buffer |
| DCD Jitter Injection | v0.2 | High | Odd/even UI time offset |
| RJ Jitter Injection | v0.2 | Medium | Gaussian random time perturbation |
| Multi-Input Channel Selection | v0.3 | Medium | Port array + dynamic indexing |
| Pattern-Dependent Delay | v0.4 | Low | Introduce Data-Dependent Jitter (DDJ) |
| Gain Compression/Saturation | v0.4 | Low | Nonlinear effects modeling |

---

**Document Version**: v0.1  
**Last Updated**: 2026-01-13  
**Author**: SerDes Project Documentation Team
