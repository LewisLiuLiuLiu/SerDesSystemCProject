# Channel Module Technical Documentation

🌐 **Languages**: [中文](../../zh/channel.md) | [English](channel.md)

**Level**: AMS Top-Level Module  
**Class Name**: `ChannelSParamTdf`  
**Current Version**: v1.1 (2026-10-18)  
**Status**: Production Ready (Full S-Parameter Modeling)

> **2026-10 implementation note**: `full_model.delay_s` is now **applied at runtime**. Each active output gets a ring-buffer delay of `round(tau/dt)` samples, using the `(active_output, active_inputs[0])` entry of `port_pairs` as the main-path lookup (previously the delay was parsed and ignored, so the simulated channel had ~0 ns while fixtures carry ~4 ns). Per-input skew inside a shared MIMO realization is not separable from one state vector and is not modeled; the channel testbench now drives input[0] and records output[1] (S21 path) with unused MIMO inputs tied to 0 V.

---

## v1.0 Major Changes Summary

**Breaking Changes in v1.0:**

1. **Unified Modeling Methods**: Removed RATIONAL and IMPULSE methods, keeping only:
   - **SIMPLE**: First-order low-pass filter for fast validation/fallback
   - **STATE_SPACE**: MIMO state-space representation as the primary VF modeling entry

2. **MIMO Support**: Added `sc_vector` ports for multi-input multi-output modeling

3. **Timestep Inheritance**: Channel module no longer sets its own timestep. It inherits from upstream modules (e.g., WaveGen) to ensure consistent sampling rate across the entire link.

4. **Removed `fs` from ChannelExtendedParams**: Sampling rate is no longer configured per-module. Use global configuration or ensure all modules use the same sampling rate.

5. **Active Port Configuration**: Added `port_config` in JSON to select active input/output ports, supporting submatrix extraction from full N×N models.



---

## 1. Overview

The Channel Module is a critical transmission path connecting the transmitter and receiver in a SerDes link. Its primary function is to model real-world channel characteristics based on measured S-parameter data, including frequency-dependent attenuation, phase distortion, crosstalk coupling, and reflection effects. The module provides two high-precision time-domain modeling methods, supporting multi-port differential transmission and complex topology scenarios.

### 1.1 Design Principles

The core design philosophy of the Channel Module is to convert frequency-domain S-parameters (Scattering Parameters) into time-domain state-space models using the Vector Fitting algorithm:

- **Frequency to Time Domain Conversion**: S-parameter frequency-domain data is fitted to rational function form using Vector Fitting, then converted to state-space implementation
- **State-Space Representation**: MIMO state-space model `dx/dt = A·x + B·u`, `y = C·x + D·u + E·du/dt`
- **Causality Guarantee**: VF algorithm ensures poles are in the left-half plane, satisfying physical causality
- **Stability Constraints**: All poles have negative real parts, ensuring system stability
- **Passivity Preservation**: Optional passivity enforcement ensures energy conservation

The module provides two implementation methods:

**Method 1: SIMPLE Method (Fast Validation/Fallback)**
- Core Concept: Using first-order low-pass filter for fast channel approximation
- Mathematical Form: `H(s) = A / (1 + s/ω₀)`, where `A = 10^(-attenuation_db/20)`, `ω₀ = 2π × bandwidth_hz`
- Time-Domain Implementation: Simple first-order IIR filter
- Advantages: No preprocessing required, minimal computational overhead, suitable for quick validation

**Method 2: STATE_SPACE Method (VF Modeling Entry)**
- Core Concept: Converting S-parameters to MIMO state-space model using Vector Fitting algorithm
- Algorithm Source: Based on [SINTEF Vector Fitting](https://www.sintef.no/en/software/vector-fitting/downloads/vfit3/) `vectfit3.m` (Bjørn Gustavsen, SINTEF Energy Research)
- Mathematical Form:
  ```
  dx/dt = A·x + B·u
  y = C·x + D·u + E·du/dt
  ```
- Time-Domain Implementation: Utilizing SystemC-AMS `sca_ss` state-space module
- Advantages: Full MIMO support, good numerical stability, supports differential conversion and delay extraction

### 1.2 Core Features

- **Dual Method Support**: SIMPLE method for fast validation, STATE_SPACE method for full S-parameter modeling
- **MIMO Modeling Support**: Multi-input multi-output support via `sc_vector` ports, configurable active port subsets
- **Differential Signal Support**: Python toolchain supports converting single-ended S-parameters to differential mode
- **Delay Extraction and Compensation**: Automatic extraction of transmission delay, improving high-frequency fitting accuracy
- **Shared Pole Vector Fitting**: Multi-port S-parameters use shared poles, ensuring system consistency
- **Flexible Port Configuration**: Select active input/output ports via JSON configuration, supporting submatrix extraction
- **Numerical Stability**: LU decomposition for DC gain calculation, avoiding numerical issues with matrix inversion
- **Configuration-Driven Design**: State-space matrices managed through JSON configuration files, decoupling offline processing from online simulation

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-09 | Initial version, basic Vector Fitting framework placeholder |
| v0.2 | 2025-10-16 | Dual method refactor: Added complete documentation for Rational Fitting and Impulse Response Convolution methods |
| v0.3 | 2025-10-16 | GPU acceleration support: Added Metal GPU acceleration (Apple Silicon exclusive), supporting direct convolution and FFT convolution |
| v0.4 | 2025-12-07 | Improved Chapter 1 overview: Rewrote design principles and core features, aligned with CTLE/VGA documentation style standards |
| v1.0 | 2026-03-22 | **Major Refactor**: Unified modeling to State Space, removed RATIONAL/IMPULSE methods, full MIMO support |

---

## 2. Module Interface

### 2.1 Class Declaration and Inheritance

```cpp
namespace serdes {
class ChannelSParamTdf : public sca_tdf::sca_module {
public:
    // TDF ports (MIMO support)
    sc_core::sc_vector<sca_tdf::sca_in<double>> in;
    sc_core::sc_vector<sca_tdf::sca_out<double>> out;
    
    // Constructors
    ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
    ChannelSParamTdf(sc_core::sc_module_name nm, 
                     const ChannelParams& params,
                     const ChannelExtendedParams& ext_params);
    
    // SystemC-AMS lifecycle methods
    void set_attributes();
    void initialize();
    void processing();
    
    // Configuration loading
    bool load_config(const std::string& config_path);
    
    // Query interfaces
    ChannelMethod get_method() const;
    double get_dc_gain() const;
    int get_n_active_inputs() const;
    int get_n_active_outputs() const;
    
private:
    ChannelParams m_params;
    ChannelExtendedParams m_ext_params;
    // ... see implementation
};
}
```

**Inheritance Hierarchy**:
- Base Class: `sca_tdf::sca_module` (SystemC-AMS TDF domain module)
- Domain Type: TDF (Timed Data Flow)

**v1.0 Implementation Notes**:
- Module operates in TDF domain, timestep is inherited from upstream modules (e.g., WaveGen) to ensure consistent sampling rate across the link
- Supports dynamic port numbers via `sc_vector` for MIMO
- Active ports can be selected via JSON configuration, supporting submatrix extraction from full N×N matrix

### 2.2 Port Definitions (TDF Domain)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in[N]` | Input | `sc_core::sc_vector<sca_tdf::sca_in<double>>` | N input ports (supporting differential pair configuration) |
| `out[M]` | Output | `sc_core::sc_vector<sca_tdf::sca_out<double>>` | M output ports (corresponding to S-parameter outputs) |

**Port Pairing Example (4-port Differential Channel)**:
- Input differential pair 1: `in[0]` (positive) + `in[1]` (negative)
- Input differential pair 2: `in[2]` (positive) + `in[3]` (negative)
- Output differential pair 1: `out[0]` (positive) + `out[1]` (negative)
- Output differential pair 2: `out[2]` (positive) + `out[3]` (negative)

**Active Port Configuration**:
Select active ports via JSON `port_config` to extract subsystems from the full matrix:
```json
{
  "port_config": {
    "active_inputs": [0, 1],
    "active_outputs": [0, 1]
  }
}
```

### 2.3 Constructor and Initialization

```cpp
// Basic constructor (SIMPLE method, backward compatible)
ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);

// Extended constructor (supports STATE_SPACE method)
ChannelSParamTdf(sc_core::sc_module_name nm, 
                 const ChannelParams& params,
                 const ChannelExtendedParams& ext_params);
```

**Parameter Description**:
- `nm`: SystemC module name, used for simulation hierarchy identification and waveform tracing
- `params`: Basic channel parameter structure (`ChannelParams`), containing simplified model parameters
- `ext_params`: Extended channel parameters (`ChannelExtendedParams`), containing method and configuration file

**Initialization Flow**:
1. Call base class constructor to register module name
2. Store parameters to member variables
3. Select modeling method based on `ext_params.method`
4. If using STATE_SPACE method, load JSON configuration file
5. Extract state-space submatrix corresponding to active ports
6. Initialize `sca_ss` state-space filter

### 2.4 Parameter Configuration

#### ChannelParams (Basic Parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `touchstone` | string | "" | S-parameter file path (.sNp format) - for Python toolchain use |
| `ports` | int | 2 | Number of ports (N≥2) - for Python toolchain use |
| `attenuation_db` | double | 10.0 | SIMPLE method attenuation (dB) |
| `bandwidth_hz` | double | 20e9 | SIMPLE method bandwidth (Hz) |

#### ChannelExtendedParams (Extended Parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | ChannelMethod | SIMPLE | Modeling method: SIMPLE or STATE_SPACE |
| `config_file` | string | "" | JSON configuration file path (required for STATE_SPACE method) |

**Note**: Channel module inherits timestep from upstream modules (e.g., WaveGen) to ensure consistent sampling rate across the link.


### 2.5 Public API Methods

#### set_attributes()

Sets TDF module timing attributes and port rates.

```cpp
void set_attributes();
```

**Responsibilities**:
- Initialize `in` and `out` port vectors based on active port count
- Declare port rates: `in[i].set_rate(1)`, `out[i].set_rate(1)`
- Inherit timestep from upstream modules (e.g., WaveGen) to ensure sampling rate consistency

#### processing()

Signal processing function for each time step, implements channel transfer function.

```cpp
void processing();
```

**Responsibilities**:
- **SIMPLE Method**: Apply first-order low-pass filter (configured via `attenuation_db` and `bandwidth_hz`)
- **STATE_SPACE Method**: Call `sca_ss` to calculate output, automatically handle MIMO

---

## 3. Core Implementation Mechanisms

### 3.1 SIMPLE Method

SIMPLE method provides fast first-order low-pass filter modeling, suitable for quick validation and fallback scenarios.

#### 3.1.1 Transfer Function

```
H(s) = A / (1 + s/ω₀)
```

Where:
- `A = 10^(-attenuation_db/20)`: Linear amplitude attenuation factor
- `ω₀ = 2π × bandwidth_hz`: Angular frequency corresponding to -3dB bandwidth

#### 3.1.2 Discrete Implementation

Using first-order IIR filter:

```cpp
y[n] = alpha * y[n-1] + (1 - alpha) * A * x[n]
```

Where `alpha = exp(-ω₀ * Ts)`, `Ts` is the sampling period (timestep inherited from upstream modules).

#### 3.1.3 Usage Scenarios

- Quick validation of link connectivity
- Simple tests without S-parameter files
- Fallback when STATE_SPACE method fails

### 3.2 STATE_SPACE Method

STATE_SPACE method is the complete S-parameter modeling implementation, based on Vector Fitting and state-space representation.

#### 3.2.1 State-Space Representation

MIMO system state-space representation:

```
dx/dt = A·x(t) + B·u(t)
y(t) = C·x(t) + D·u(t) + E·du/dt
```

Matrix dimensions:
- `A`: (n_states × n_states) - State matrix
- `B`: (n_states × n_inputs) - Input matrix
- `C`: (n_outputs × n_states) - Output matrix
- `D`: (n_outputs × n_inputs) - Direct transmission matrix
- `E`: (n_outputs × n_inputs) - Differential matrix (optional)

For N-port differential system:
- `n_inputs = 2 × n_diff_ports` (positive and negative terminals of differential pairs)
- `n_outputs = n_diff_ports²` (each input-output combination)

#### 3.2.2 Active Matrix Extraction

Support extracting submatrices corresponding to active ports from the full model:

```cpp
void extract_active_matrices();
```

Extraction logic:
1. Select columns of B matrix based on `port_config.active_inputs`
2. Select rows of C matrix, rows/columns of D/E matrices based on `port_config.active_outputs`
3. A matrix remains unchanged (shared pole characteristic)

#### 3.2.3 SystemC-AMS Implementation

Using `sca_tdf::sca_ss` module for state-space:

```cpp
sca_tdf::sca_ss m_ss_filter;
sca_util::sca_vector<double> m_ss_state;
```

Initialization:
```cpp
// Timestep inherited from upstream modules
m_ss_filter.set_timestep(get_timestep());
m_ss_filter.set_model(A, B, C, D, E);
m_ss_state.init(n_states);
```

Processing:
```cpp
// Read inputs
sca_util::sca_vector<double> u(n_inputs);
for (int i = 0; i < n_inputs; ++i) {
    u[i] = in[i].read();
}

// Calculate output
sca_util::sca_vector<double> y = m_ss_filter.calculate(u, m_ss_state);

// Write outputs
for (int i = 0; i < n_outputs; ++i) {
    out[i].write(y[i]);
}
```

#### 3.2.4 DC Gain Calculation

Using LU decomposition to solve `A·X = B`, avoiding direct matrix inversion:

```cpp
double get_dc_gain() const {
    // Solve A * X = B
    auto LU = A;
    lu_decompose(LU, pivot);
    auto X = B;
    lu_solve(LU, pivot, X);
    
    // DC gain = D - C * X
    // ...
}
```

---

---

## 4. Testbench Architecture

### 4.1 Standalone Testbench

Located at `tb/channel/channel_sparam_tb.cpp`

**Test Content**:
- SIMPLE method functional verification
- STATE_SPACE method functional verification
- Configuration loading test
- DC gain calculation verification

**Run**:
```bash
cd build
make channel_sparam_tb
./bin/channel_sparam_tb
```

### 4.2 Integration Test

Located at `tb/simple_link_tb.cpp`

**Test Content**:
- Complete TX→Channel→RX link
- Eye diagram quality verification
- Co-working with other modules

---

## 5. Running Guide

### 5.1 Using SIMPLE Method

```cpp
// Base parameters
ChannelParams params;
params.attenuation_db = 10.0;
params.bandwidth_hz = 20e9;

// Create module (automatically uses SIMPLE method)
auto channel = std::make_unique<ChannelSParamTdf>("channel", params);
```

### 5.2 Using STATE_SPACE Method

```cpp
// 1. Use Python toolchain to generate JSON config
// python scripts/vector_fitting.py --input channel.s4p --output channel.json

// 2. Use STATE_SPACE method in C++ code
ChannelParams params;
ChannelExtendedParams ext_params;
ext_params.method = ChannelMethod::STATE_SPACE;
ext_params.config_file = "config/channel.json";
// Note: Channel module timestep is inherited from upstream modules

auto channel = std::make_unique<ChannelSParamTdf>("channel", params, ext_params);
```

### 5.3 Configuration Examples

**SIMPLE method configuration**:
```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20000000000.0
    }
  }
}
```

**STATE_SPACE method configuration**:
```json
{
  "channel": {
    "method": "state_space",
    "config_file": "config/channel_ss.json"
  }
}
```

---

## 6. Technical Points

### 6.1 Vector Fitting Algorithm Key Points

**Shared Poles**:
- All S-parameter elements use the same pole set
- Ensures MIMO system consistency and stability
- State matrix A is shared across all transmission paths

**Delay Extraction**:
- Linear phase component extracted as delay
- Improves high-frequency fitting accuracy
- Delay stored separately in JSON as part of group delay characteristics

**Passivity Enforcement**:
- Ensures scattering matrix eigenvalues ≤ 1
- Prevents energy growth in simulation

### 6.2 Numerical Stability

**LU Decomposition**:
- Uses partial pivot LU decomposition to solve linear systems
- Avoids numerical issues from direct matrix inversion
- Used for DC gain calculation and state-space analysis

**Matrix Condition Number**:
- High-order VF may produce ill-conditioned matrices
- Recommended order: 6-16 (depending on channel complexity)

### 6.3 Performance Considerations

**Computational Complexity**:
- SIMPLE method: O(1) per timestep
- STATE_SPACE method: O(n_states²) per timestep

**State Dimension**:
- `n_states = order × n_outputs`
- Example: 14th order × 4 outputs = 56-dimensional state vector

---

## 7. Reference Information

### 7.1 Related Files

#### Source Files

| File | Path | Description |
|------|------|-------------|
| Parameter Definition | `/include/common/parameters.h` | ChannelParams structure |
| Header File | `/include/ams/channel_sparam.h` | ChannelSParamTdf class declaration |
| Implementation File | `/src/ams/channel_sparam.cpp` | ChannelSParamTdf class implementation |
| Python Tool | `/scripts/vector_fitting.py` | Vector Fitting and preprocessing tool |

#### Test Files

| File | Path | Description |
|------|------|-------------|
| Standalone Test | `/tb/channel/channel_sparam_tb.cpp` | Channel module standalone test |
| Integration Test | `/tb/simple_link_tb.cpp` | Complete link integration test |

### 7.2 Dependencies

| Dependency | Version | Purpose | Required |
|-----------|---------|---------|----------|
| SystemC | 2.3.4 | SystemC core library | Yes |
| SystemC-AMS | 2.3.4 | AMS extension library | Yes |
| C++ Standard | C++14 | Compiler standard | Yes |
| nlohmann/json | 3.x | JSON parsing | Yes |
| Python | 3.7+ | Preprocessing toolchain | Recommended |
| numpy | 1.19+ | Numerical computation | Recommended |
| scipy | 1.5+ | Signal processing | Recommended |

### 7.3 Configuration Examples

#### SIMPLE Method Configuration

```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20000000000.0
    }
  }
}
```

#### STATE_SPACE Method Configuration

```json
{
  "channel": {
    "method": "state_space",
    "config_file": "config/channel_state_space.json"
  }
}
```

#### State Space JSON Format (Generated by Python Tool)

```json
{
  "version": "3.0",
  "method": "state_space",
  "fs": 80000000000.0,
  "full_model": {
    "n_diff_ports": 2,
    "n_outputs": 4,
    "n_states": 28,
    "port_pairs": [[0,0], [0,1], [1,0], [1,1]],
    "delay_s": [4.0e-09, 4.0e-09, 4.0e-09, 4.0e-09],
    "state_space": {
      "A": [[...], ...],
      "B": [[...], ...],
      "C": [[...], ...],
      "D": [0.0, 0.0, 0.0, 0.0],
      "E": [0.0, 0.0, 0.0, 0.0]
    }
  },
  "port_config": {
    "active_inputs": [0],
    "active_outputs": [0]
  }
}
```

### 7.4 Parameter Reference Table

#### ChannelParams Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `touchstone` | string | "" | S-parameter file path (for Python tool) |
| `ports` | int | 2 | Number of ports (for Python tool) |
| `attenuation_db` | double | 10.0 | SIMPLE method attenuation (dB) |
| `bandwidth_hz` | double | 20e9 | SIMPLE method bandwidth (Hz) |

#### ChannelExtendedParams Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `method` | ChannelMethod | SIMPLE | Modeling method |
| `config_file` | string | "" | JSON configuration file path |

**Note**: Channel module timestep is inherited from upstream modules, not set independently.

### 7.5 FAQ

**Q1: How to choose between SIMPLE and STATE_SPACE methods?**

A:
- **SIMPLE**: Quick validation, no S-parameter file needed, low computation overhead
- **STATE_SPACE**: Requires precise S-parameter modeling, supports MIMO, requires preprocessing

**Q2: How to use Python toolchain?**

A:
```python
from scripts.vector_fitting import SParamModel

model = SParamModel()
model.load_snp('channel.s4p')
model.to_differential()
model.fit(order=14, extract_delay=True)
model.export_json('channel.json', fs=80e9)
```

**Q3: How to verify channel module is working correctly?**

A:
1. Run `channel_sparam_tb` to verify basic functionality
2. Run `simple_link_tb` to verify link integration
3. Check output waveform eye diagram quality

**Q4: What is the computational overhead of STATE_SPACE method?**

A: Computational complexity is O(n_states²) per timestep. For example, a 14th order 4-output system (56-dimensional state) can easily support 80GSa/s sampling rate on modern CPUs.

**Q5: How to handle multi-port S-parameter files?**

A: Python toolchain automatically handles multi-port files, converts to differential mode via `to_differential()`, uses shared pole VF fitting, and exports complete MIMO state-space model.

**Q6: Can partial ports be extracted from full model?**

A: Yes. Select active ports via `port_config.active_inputs` and `port_config.active_outputs` in JSON, and the module will automatically extract corresponding submatrices.

---

## References

[1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency domain responses by Vector Fitting," *IEEE Transactions on Power Delivery*, vol. 14, no. 3, pp. 1052-1061, July 1999.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," *IEEE Transactions on Power Delivery*, vol. 21, no. 3, pp. 1587-1592, July 2006.

[3] B. Gustavsen, "Fast Relaxed Vector Fitting," *SINTEF Energy Research*, version 1.0, August 2008. [Online]. Available: https://www.sintef.no/en/software/vector-fitting/downloads/vfit3/

---

**Document Version**: v1.0  
**Last Updated**: 2026-03-22  
**Author**: Yizhe Liu



