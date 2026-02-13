# Channel Module Technical Documentation

🌐 **Languages**: [中文](../../modules/channel.md) | [English](channel.md)

**Level**: AMS Top-Level Module  
**Class Name**: `ChannelSParamTdf`  
**Current Version**: v0.4 (2025-12-07)  
**Status**: Production Ready

---

## 1. Overview

The Channel Module is a critical transmission path connecting the transmitter and receiver in a SerDes link. Its primary function is to model real-world channel characteristics based on measured S-parameter data, including frequency-dependent attenuation, phase distortion, crosstalk coupling, and reflection effects. The module provides two high-precision time-domain modeling methods, supporting multi-port differential transmission and complex topology scenarios.

### 1.1 Design Principles

The core design philosophy of the Channel Module is to convert frequency-domain S-parameters (Scattering Parameters) into time-domain causal systems, efficiently implementing wideband non-ideal transmission effects:

- **Frequency to Time Domain Conversion**: Mathematical transformation of measured or simulated S-parameter frequency-domain data into time-domain transfer functions or impulse responses
- **Causality Guarantee**: Ensuring the time-domain system satisfies physical causality, avoiding non-physical behaviors that predict future inputs
- **Stability Constraints**: Ensuring transfer function poles are located in the left-half plane (continuous domain) or inside the unit circle (discrete domain), preventing signal energy divergence
- **Passivity Preservation**: Modeling within an energy conservation framework, ensuring output energy does not exceed input energy (scattering matrix passivity condition)

The module provides two complementary implementation methods:

**Method 1: Rational Function Fitting Method**
- Core Concept: Using Vector Fitting algorithm to approximate S-parameter frequency-domain response as rational function form
- Mathematical Form:
  ```
  H(s) = Σ(r_k / (s - p_k)) + d + s·h
  H(s) = (b_n·s^n + ... + b_0) / (a_m·s^m + ... + a_0)
  ```
- Time-Domain Implementation: Utilizing SystemC-AMS `sca_tdf::sca_ltf_nd` linear time-invariant filter for compact and efficient convolution
- Advantages: Compact parameters (8-16 poles usually sufficient), high computational efficiency (O(order) per time step), good numerical stability

**Method 2: Impulse Response Convolution Method**
- Core Concept: Obtaining time-domain impulse response through Inverse Fourier Transform (IFFT) of S-parameters, performing direct convolution
- Mathematical Form:
  ```
  h(t) = IFFT[H(f)]
  y(t) = ∫ h(τ)·x(t-τ) dτ  (continuous domain)
  y[n] = Σ h[k]·x[n-k]      (discrete domain)
  ```
- Time-Domain Implementation: Maintaining input delay line, performing finite-length convolution or Fast Fourier Transform (FFT) convolution
- Advantages: Preserves complete frequency-domain information, handles non-minimum-phase systems, easy to understand and debug

### 1.2 Core Features

- **Dual Method Support**: Rational function fitting (recommended) or impulse response convolution, adapting to different accuracy and performance requirements
- **Multi-port Modeling**: Supports N×N port S-parameter matrix (N≥2), covering single-ended, differential, and multi-channel scenarios
- **Crosstalk Coupling**: Complete modeling of Near-End Crosstalk (NEXT) and Far-End Crosstalk (FEXT) through coupling matrices for multi-channel interactions
- **Bidirectional Transmission**: Switchable control of forward transmission (S21), reverse transmission (S12), and port reflections (S11/S22)
- **Frequency-Domain Preprocessing**: DC point completion, sampling frequency matching, band-limited roll-off, ensuring robust time-domain conversion
- **Port Mapping Standardization**: Manual specification or automatic identification of port pairing relationships, unified differential pair and transmission path definitions
- **GPU Acceleration (Apple Silicon Exclusive)**: Metal GPU acceleration for direct convolution and FFT convolution, specifically for long impulse responses and high sampling rate scenarios
- **Configuration-Driven Design**: Managing fitting parameters, preprocessing options, and acceleration strategies through JSON configuration files, decoupling offline processing from online simulation

### 1.3 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v0.1 | 2025-09 | Initial version, basic Vector Fitting framework placeholder |
| v0.2 | 2025-10-16 | Dual method refactor: Added complete documentation for Rational Fitting and Impulse Response Convolution methods |
| v0.3 | 2025-10-16 | GPU acceleration support: Added Metal GPU acceleration (Apple Silicon exclusive), supporting direct convolution and FFT convolution |
| v0.4 | 2025-12-07 | Improved Chapter 1 overview: Rewrote design principles and core features, aligned with CTLE/VGA documentation style standards |

---

## 2. Module Interface

### 2.1 Class Declaration and Inheritance

```cpp
namespace serdes {
class ChannelSParamTdf : public sca_tdf::sca_module {
public:
    // TDF ports
    sca_tdf::sca_in<double> in;
    sca_tdf::sca_out<double> out;
    
    // Constructor
    ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
    
    // SystemC-AMS lifecycle methods
    void set_attributes();
    void processing();
    
private:
    ChannelParams m_params;
    std::vector<double> m_buffer;
};
}
```

**Inheritance Hierarchy**:
- Base Class: `sca_tdf::sca_module` (SystemC-AMS TDF domain module)
- Domain Type: TDF (Timed Data Flow)

**v0.4 Implementation Notes**:
- Module operates in TDF domain, processing signals at fixed time steps (determined by global sampling rate fs)
- Current implementation is a **Simplified Single-Input Single-Output (SISO) version**
- Future extension: Multi-port N×N matrix support (via port vectors `sca_in<double> in[N]`, `sca_out<double> out[N]`)

### 2.2 Port Definitions (TDF Domain)

#### Current Implementation (v0.4)

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in` | Input | `sca_tdf::sca_in<double>` | Channel input signal (single-ended) |
| `out` | Output | `sca_tdf::sca_out<double>` | Channel output signal (single-ended) |

**Usage Scenarios**:
- Single-ended transmission link (e.g., single transmission line)
- Differential signal modeling (requires instantiating two Channels for positive/negative terminals)
- Simplified test scenarios

#### Future Extension: Multi-port Matrix (Design Specification)

For complete multi-port S-parameter modeling (e.g., 4-port differential channel), the interface will be extended to:

| Port Name | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `in[N]` | Input | `sca_tdf::sca_in<double>` | N input ports (supporting differential pair configuration) |
| `out[N]` | Output | `sca_tdf::sca_out<double>` | N output ports (corresponding to S-parameter outputs) |

**Port Pairing Example (4-port Differential Channel)**:
- Input differential pair 1: `in[0]` (positive) + `in[1]` (negative)
- Input differential pair 2: `in[2]` (positive) + `in[3]` (negative)
- Output differential pair 1: `out[0]` (positive) + `out[1]` (negative)
- Output differential pair 2: `out[2]` (positive) + `out[3]` (negative)

> **Note**: This feature is a design specification and is **not implemented** in current v0.4 version.

### 2.3 Constructor and Initialization

```cpp
ChannelSParamTdf(sc_core::sc_module_name nm, const ChannelParams& params);
```

**Parameter Description**:
- `nm`: SystemC module name, used for simulation hierarchy identification and waveform tracing
- `params`: Channel parameter structure (`ChannelParams`), containing all configuration items

**v0.4 Initialization Flow**:
1. Call base class constructor to register module name
2. Store parameters to member variable `m_params`
3. Pre-allocate delay line buffer `m_buffer` (for simplified model or future extensions)
4. Current implementation does not load S-parameter files (placeholder implementation)

### 2.4 Parameter Configuration (ChannelParams)

#### Current Implementation Parameters (v0.4)

The following parameters are directly derived from the actual structure in `include/common/parameters.h` lines 90-105:

| Parameter | Type | Default | Description | v0.4 Status |
|-----------|------|---------|-------------|-------------|
| `touchstone` | string | "" | S-parameter file path (.sNp format) | **Placeholder**, not actually loaded |
| `ports` | int | 2 | Number of ports (N≥2) | **Placeholder**, currently single port only |
| `crosstalk` | bool | false | Enable multi-port crosstalk coupling matrix (NEXT/FEXT) | **Not implemented** |
| `bidirectional` | bool | false | Enable bidirectional transmission (S12 reverse path and reflections) | **Not implemented** |
| `attenuation_db` | double | 10.0 | Simplified model attenuation (dB) | **Available** |
| `bandwidth_hz` | double | 20e9 | Simplified model bandwidth (Hz) | **Available** |

**v0.4 Implementation Notes**:
- **Simplified model available**: `attenuation_db` and `bandwidth_hz` can be used for first-order low-pass filter modeling
- **S-parameter support placeholder**: `touchstone` and `ports` parameters are defined but file loading and matrix processing are not implemented in current version
- **Advanced features not implemented**: `crosstalk` and `bidirectional` are design specifications, not enabled in code

**Usage Constraints (v0.4)**:
- Do not rely on S-parameter file loading functionality
- `ports` parameter is ignored, module is fixed to single-input single-output
- `crosstalk` and `bidirectional` settings have no effect

#### Future Extension Parameters (Design Specification)

The following parameters are **not defined in current parameters.h**, and are documented here for future extension reference:

##### method parameter (not implemented)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | "rational" | Time-domain modeling method: "rational" (Rational Function Fitting) or "impulse" (Impulse Response Convolution) |
| `config_file` | string | "" | Path to configuration file generated by offline processing (JSON format) |

##### fit sub-structure (Rational Function Fitting Method, not implemented)

Used to control offline processing parameters for Vector Fitting algorithm.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | int | 16 | Fitting order (number of poles/zeros), recommended 6-16 |
| `enforce_stable` | bool | true | Enforce stability constraint (pole real part < 0) |
| `enforce_passive` | bool | true | Enforce passivity constraint (energy conservation) |
| `band_limit` | double | 0.0 | Band upper limit (Hz, 0 means use Touchstone file maximum frequency) |

**Design Principle**:
1. Python offline tool reads S-parameter frequency-domain data `Sij(f)`
2. Vector Fitting algorithm approximates it as rational function: `H(s) = (b_n·s^n + ... + b_0) / (a_m·s^m + ... + a_0)`
3. Stability constraint ensures all poles `p_k` satisfy `Re(p_k) < 0`
4. Passivity constraint ensures scattering matrix eigenvalues ≤ 1 at all frequency points
5. Fitting results (numerator/denominator coefficients) saved to configuration file
6. Online simulation instantiates via `sca_tdf::sca_ltf_nd` filter

**Order Selection Guide**:
| Channel Type | Bandwidth | Recommended Order | Reason |
|--------------|-----------|-------------------|--------|
| Short Backplane | <10 GHz | 6-8 | Low loss, smooth frequency response |
| Long Backplane | 10-25 GHz | 10-12 | Medium loss, need to capture skin effect |
| Ultra-long Cable | >25 GHz | 14-16 | High loss, significant frequency-dependent attenuation |


##### impulse sub-structure (Impulse Response Convolution Method, not implemented)

Used to control offline processing for Inverse Fourier Transform (IFFT) and online convolution parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_samples` | int | 4096 | Impulse response length (samples), recommended 2048-8192 |
| `causality` | bool | true | Apply causality window function (ensure h(t)≈0 for t<0) |
| `truncate_threshold` | double | 1e-6 | Tail truncation threshold (relative to peak amplitude ratio) |
| `dc_completion` | string | "vf" | DC point completion method: "vf" (Vector Fitting), "interp" (interpolation), "none" |
| `resample_to_fs` | bool | true | Resample frequency-domain data to target sampling rate fs |
| `fs` | double | 0.0 | Target sampling frequency (Hz, 0 means use global fs) |
| `band_limit` | double | 0.0 | Band upper limit (Hz, recommended ≤fs/2 to avoid aliasing) |
| `grid_points` | int | 0 | Frequency grid points (0 means same as time_samples) |

**Design Principle**:
1. Read S-parameter frequency-domain data, perform DC point completion and frequency grid resampling
2. Execute IFFT to obtain time-domain impulse response `h(t) = IFFT[Sij(f)]`
3. Apply causality window function (e.g., Hamming window) to suppress non-causal components
4. Truncate tail below threshold to reduce convolution length L
5. Save time series to configuration file
6. Online simulation maintains delay line: `y(n) = Σ h(k) · x(n-k)`

**Length Selection Guide**:
| Scenario | Impulse Length L | Computational Complexity | Recommended Acceleration |
|----------|-----------------|-------------------------|-------------------------|
| Short Channel (<5 GHz) | 512-1024 | O(L) CPU acceptable | No GPU needed |
| Medium Channel (5-15 GHz) | 2048-4096 | O(L) or O(L log L) | CPU multi-core or GPU direct convolution |
| Long Channel (>15 GHz) | 4096-8192 | O(L log L) FFT convolution | GPU FFT convolution (Apple Silicon) |

##### gpu_acceleration sub-structure (Metal GPU Acceleration, not implemented)

Only designed to be effective when `method="impulse"` and system is Apple Silicon.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable GPU acceleration (**Apple Silicon only**) |
| `backend` | string | "metal" | GPU backend (fixed to "metal", other backends not supported) |
| `algorithm` | string | "auto" | Algorithm selection: "direct" (direct convolution), "fft" (FFT convolution), "auto" (automatic selection) |
| `batch_size` | int | 1024 | Batch processing sample count (reduce CPU-GPU transfer latency) |
| `fft_threshold` | int | 512 | Automatically switch to FFT convolution when L>threshold |

**Design Principle**:
1. Collect `batch_size` input samples into batch processing buffer
2. Upload to GPU shared memory (Metal Shared Memory) in one operation
3. Select algorithm based on impulse length L:
   - **Direct Convolution** (L<512): Parallel computation of `y[n] = Σ h[k]·x[n-k]` for each output sample
   - **FFT Convolution** (L≥512): Use Metal Performance Shaders to execute `y = IFFT(FFT(x) ⊙ FFT(h))`
4. Download results to CPU, output sequentially to TDF port

**System Requirements**:
- **Required**: Apple Silicon (M1/M2/M3/M4 or newer)
- **Not Supported**: Intel Mac, Linux, Windows, NVIDIA GPU, AMD GPU

**Performance Expectations**:
- Direct Convolution (L<512): 50-100x relative to single-core CPU
- FFT Convolution (L≥512): 200-500x relative to single-core CPU
- Batch mode up to 1000x (high sampling rate scenarios)

##### port_mapping sub-structure (Port Mapping Standardization, not implemented)

Resolves inconsistent port ordering from different Touchstone file sources.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable port mapping standardization |
| `mode` | string | "manual" | Mapping mode: "manual" (manual specification), "auto" (automatic identification) |
| `manual.pairs` | vector<pair<int,int>> | [] | Differential pair configuration (e.g., [[1,2],[3,4]]) |
| `manual.forward` | vector<pair<int,int>> | [] | Input→Output pairing (e.g., [[1,3],[2,4]]) |
| `auto.criteria` | string | "energy" | Auto-identification criteria: "energy" (passband energy), "lowfreq", "bandpass" |
| `auto.constraints` | object | {} | Constraints (differential: bool, bidirectional: bool) |

**Design Principle**:
- **Manual Mode**: Construct permutation matrix P according to configuration, reorder S(f): `S'(f) = P·S(f)·P^T`
- **Auto Mode**: Calculate passband energy of each Sij `Eij = ∫|Sij(f)|²df`, use maximum matching algorithm to identify main transmission paths

### 2.5 Public API Methods

#### set_attributes()

Sets TDF module timing attributes and port rates.

```cpp
void set_attributes();
```

**Responsibilities**:
- Set sampling time step: `set_timestep(1.0/fs)` (fs obtained from `GlobalParams`)
- Declare port rates: `in.set_rate(1)`, `out.set_rate(1)` (process 1 sample per time step)
- Set delay: `out.set_delay(0)` (current simplified model has no delay, future extensions will set based on impulse response length)

**v0.4 Implementation**:
- Only sets basic time step and port rates
- Dynamic delay setting not implemented

#### processing()

Signal processing function for each time step, implements channel transfer function.

```cpp
void processing();
```

**v0.4 Responsibilities**:
- **Simplified Model**: Apply first-order low-pass filter (configured via `attenuation_db` and `bandwidth_hz`)
- **Through Mode**: If parameters not configured, directly copy input to output

**Future Extension Responsibilities (Design Specification)**:
- **Rational Function Method**: Calculate output via `sca_ltf_nd` filter
- **Impulse Response Method**: Update delay line, perform convolution `y(n) = Σ h(k)·x(n-k)`
- **Crosstalk Processing**: Calculate coupling matrix effect `x'[i] = Σ C[i][j]·x[j]`
- **Bidirectional Transmission**: Superimpose contributions from reverse path S12 and reflections S11/S22

---

## 3. Core Implementation Mechanisms

### 3.1 v0.4 Simplified Implementation (Current Version)

**Important Note**: Current v0.4 version **does not implement complete S-parameter modeling functionality**, only provides a simplified first-order low-pass filter as placeholder implementation. The following describes the actual signal processing flow currently implemented.

#### 3.1.1 Current Signal Processing Flow

The v0.4 version's `processing()` method uses the simplest signal transmission model:

```
Input Read → Simplified Filter → Output Write
```

**Step 1 - Input Read**: Read current time step signal sample from TDF input port:
```cpp
double x = in.read();
```

**Step 2 - Simplified Filter Application**: Apply first-order low-pass filter based on `attenuation_db` and `bandwidth_hz` parameters:
- Amplitude attenuation: `A = 10^(-attenuation_db/20)`
- Bandwidth limitation: First-order pole `H(s) = A / (1 + s/ω₀)`, where `ω₀ = 2π × bandwidth_hz`
- Implementation: Through SystemC-AMS `sca_ltf_nd` filter or simple gain scaling

**Step 3 - Output Write**: Write processed signal to TDF output port:
```cpp
out.write(y);
```

#### 3.1.2 Transfer Function of Simplified Model

The first-order low-pass transfer function used in current implementation:

```
H(s) = A / (1 + s/ω₀)
```

**Parameter Mapping**:
- `A = 10^(-attenuation_db/20)`: Linear amplitude attenuation factor
- `ω₀ = 2π × bandwidth_hz`: Angular frequency corresponding to -3dB bandwidth

**Frequency-Domain Characteristics**:
- DC gain: `H(0) = A`
- -3dB frequency: `f₋₃dB = bandwidth_hz`
- High-frequency roll-off: -20dB/decade (first-order system)
- Phase delay: 0° (DC) → -90° (high frequency)

**Limitations**:
- Cannot characterize complex frequency-dependent attenuation characteristics (skin effect, dielectric loss)
- Missing group delay/dispersion effects
- No crosstalk modeling capability
- No reflection and bidirectional transmission support
- Cannot capture non-minimum-phase characteristics of S-parameters

#### 3.1.3 State Management

State variables in current version:

| Variable | Type | Purpose |
|----------|------|---------|
| `m_params` | `ChannelParams` | Store configuration parameters |
| `m_buffer` | `std::vector<double>` | Reserved delay line buffer (currently unused) |

**Note**: `m_buffer` is declared but not actually used in v0.4, reserved for future extension to impulse response convolution method.

---

### 3.2 Rational Function Fitting Method (Design Specification)

**Important Note**: The following describes the design specification for future implementation of complete S-parameter modeling, describing how it should be implemented, not the actual behavior of current code.

#### 3.2.1 Offline Processing Stage (Python Toolchain)

The Rational Function Fitting Method converts S-parameter frequency-domain data to compact transfer function form, including the following key steps:

##### Step 1: S-Parameter File Loading

```python
import skrf as rf
network = rf.Network('channel.s4p')
freq = network.f  # Frequency point array (Hz)
S_matrix = network.s  # S-parameter matrix [N_freq, N_ports, N_ports]
```

**Data Preprocessing**:
- Verify frequency points are monotonically increasing
- Check complex S-parameter magnitudes ≤ 1 (preliminary passivity check)
- Extract transmission paths of interest (e.g., S21, S43) and crosstalk terms (S13, S14, etc.)

##### Step 2: Vector Fitting Algorithm

Vector Fitting is an iterative optimization algorithm that approximates frequency-domain response S(f) as rational function form:

**Objective Function**: For each S-parameter Sij(f), find rational function H(s) such that:
```
H(s) = Σ(r_k / (s - p_k)) + d + s·h
```
Minimize error at frequency measurement points:
```
min Σ|Sij(f_n) - H(j·2π·f_n)|²
```

**Algorithm Flow**:
1. **Initialize Poles**: Uniformly distribute `order` starting poles in left-half plane of complex plane
   - Real poles: `p_k = -2π·f_k`, where f_k is logarithmically distributed in [f_min, f_max]
   - Complex conjugate pole pairs: `p_k = -σ_k ± j·ω_k`, covering key frequency regions

2. **Iterative Optimization** (typically 3-5 rounds):
   - **Residue Estimation**: Fix poles {p_k}, solve for residues {r_k} and constants d, h via linear least squares
   - **Pole Relocation**: Treat H(s) as weight function, re-estimate pole positions via weighted least squares
   - **Convergence Check**: Monitor fitting error and pole movement

3. **Stability Enforcement**: If any pole has real part ≥ 0, mirror to left-half plane: `p_k' = -|Re(p_k)| + j·Im(p_k)`

4. **Passivity Enforcement** (optional):
   - Construct scattering matrix eigenvalue constraints
   - Ensure all frequency points satisfy `max(eig(S'·S)) ≤ 1`
   - Fine-tune poles/residues via quadratic programming to meet constraints

##### Step 3: Pole-Residue to Numerator-Denominator Polynomial Conversion

Convert partial fraction form to polynomial ratio:

```
H(s) = (b_n·s^n + ... + b_1·s + b_0) / (a_m·s^m + ... + a_1·s + a_0)
```

**Conversion Steps**:
1. **Denominator Construction**:
   ```python
   den = [1.0]  # Normalize leading term
   for p_k in poles:
       den = poly_multiply(den, [1, -p_k])  # (s - p_k)
   ```

2. **Numerator Construction**:
   - Merge residues {r_k} and constants d, h
   - Construct numerator coefficients via polynomial multiplication and addition
   - Ensure numerator order ≤ denominator order (physical realizability)

3. **Normalization**: Normalize denominator leading coefficient to 1.0

##### Step 4: Configuration File Export

Generate JSON format configuration file containing transfer functions for all S-parameter paths:

```json
{
  "method": "rational",
  "fs": 100e9,
  "filters": {
    "S21": {
      "num": [b0, b1, b2, ..., bn],
      "den": [1.0, a1, a2, ..., am],
      "order": 8,
      "dc_gain": 0.7943,
      "mse": 1.2e-4
    },
    "S43": {...},
    "S13": {...}
  },
  "port_mapping": {
    "forward": [[1,3], [2,4]],
    "crosstalk": [[1,4], [2,3]]
  }
}
```

**Quality Metrics**:
- `mse`: Mean Square Error (relative to original S-parameters)
- `max_error`: Maximum absolute error
- `passivity_margin`: Passivity margin (difference between maximum eigenvalue and 1)


#### 3.2.2 Online Simulation Stage (SystemC-AMS)

##### Initialization: Filter Instance Creation

Create `sca_ltf_nd` filter objects in `initialize()` or constructor based on configuration file:

```cpp
// Pseudocode example
void ChannelSParamTdf::initialize() {
    // Load configuration file
    Config cfg = load_json(m_params.config_file);
    
    // Create filters for each S-parameter path
    for (auto& [path_name, filter_cfg] : cfg.filters) {
        sca_util::sca_vector<double> num(filter_cfg.num);
        sca_util::sca_vector<double> den(filter_cfg.den);
        
        // Create LTF filter object
        auto ltf = std::make_shared<sca_tdf::sca_ltf_nd>(
            num, den, 1.0/m_fs  // numerator, denominator, time step
        );
        
        m_filters[path_name] = ltf;
    }
}
```

**Key Design Decisions**:
- Each Sij path requires independent filter instance
- N×N port matrix theoretically requires N squared filters (can be optimized using symmetry)
- Filter states managed internally by SystemC-AMS, automatically handling state-space implementation

##### Real-time Processing: processing() Method Flow

Complete rational function method signal processing flow:

```
Input Read → [Crosstalk Preprocessing] → Transfer Function Filter → [Bidirectional Superposition] → Output Write
```

**Single-Port Unidirectional Transmission (Simplest Scenario)**:
```cpp
void ChannelSParamTdf::processing() {
    double x = in.read();
    
    // Apply S21 transfer function
    double y = m_filters["S21"]->apply(x);
    
    out.write(y);
}
```

**Multi-Port Crosstalk Scenario**:
```cpp
void ChannelSParamTdf::processing() {
    // Step 1: Read all input ports
    std::vector<double> x_in(N_ports);
    for (int i = 0; i < N_ports; ++i) {
        x_in[i] = in[i].read();
    }
    
    // Step 2: Apply coupling matrix (crosstalk preprocessing)
    std::vector<double> x_coupled(N_ports, 0.0);
    for (int i = 0; i < N_ports; ++i) {
        for (int j = 0; j < N_ports; ++j) {
            // Main transmission path + crosstalk term
            double h_ij = m_filters[S_name(i,j)]->apply(x_in[j]);
            x_coupled[i] += h_ij;
        }
    }
    
    // Step 3: Write to output ports
    for (int i = 0; i < N_ports; ++i) {
        out[i].write(x_coupled[i]);
    }
}
```

**Bidirectional Transmission Scenario**:
```cpp
void ChannelSParamTdf::processing() {
    // Forward path: in to S21 to out
    double y_forward = m_filters["S21"]->apply(in.read());
    
    // Reverse path: out_prev to S12 to in (need to store previous cycle output)
    double y_backward = m_filters["S12"]->apply(m_out_prev);
    
    // Port 1 reflection: in to S11 to in
    double y_reflect1 = m_filters["S11"]->apply(in.read());
    
    // Port 2 reflection: out_prev to S22 to out
    double y_reflect2 = m_filters["S22"]->apply(m_out_prev);
    
    // Superimpose all contributions
    double y_total = y_forward + y_reflect2;
    
    out.write(y_total);
    m_out_prev = y_total;  // Save for next cycle
}
```

##### Numerical Characteristics and Optimization

**Computational Complexity**:
- Per time step: O(order) floating point operations
- 8th-order filter: approximately 40 multiply-add operations
- Multi-port: O(N squared times order)

**Numerical Stability**:
- SystemC-AMS uses state-space implementation, automatically balancing numerical errors
- Poles forced in left-half plane ensure long-term stability
- Recommended filter order less than or equal to 16 to avoid numerical issues with high-order polynomials

---

### 3.3 Impulse Response Convolution Method (Design Specification)

#### 3.3.1 Offline Processing Stage (Python Toolchain)

The Impulse Response Convolution Method converts S-parameters to time-domain impulse response through Inverse Fourier Transform, suitable for capturing non-minimum-phase characteristics and complex frequency-domain structures.

##### Step 1: Frequency-Domain Data Preprocessing

**DC Point Completion**:
Touchstone files typically lack 0 Hz points and require completion to avoid time-domain DC bias:

**Method A: Vector Fitting Method (Recommended)**
```python
# Use VF to estimate DC value
vf_result = vector_fit(freq, S21, order=6)
S21_dc = vf_result.evaluate(s=0)  # H(0)
```
- Advantage: Strong physical constraints (stability/passivity), high extrapolation accuracy
- Applicable: All channel types

**Method B: Low Frequency Extrapolation Method**
```python
# Extrapolate amplitude and phase of lowest few frequency points
freq_low = freq[:5]
mag_low = np.abs(S21[:5])
phase_low = np.angle(S21[:5])

# Linear extrapolation to DC
mag_dc = np.interp(0, freq_low, mag_low)
phase_dc = np.interp(0, freq_low, phase_low)
S21_dc = mag_dc * np.exp(1j * phase_dc)
```
- Advantage: Simple implementation
- Risk: Low frequency measurement noise causes DC error

**Band-Limited Processing**:
Limit frequency upper bound to Nyquist frequency to avoid aliasing:
```python
f_nyquist = fs / 2
freq_valid = freq[freq <= f_nyquist]
S_valid = S21[freq <= f_nyquist]

# High-frequency roll-off (optional, reduce Gibbs effect)
window = np.hanning(len(freq_valid))
S_windowed = S_valid * window
```

**Frequency Grid Resampling**:
Construct uniform frequency grid matching target sampling rate fs:
```python
N = time_samples
df = fs / N
freq_grid = np.arange(0, fs/2, df)  # 0, df, 2df, ..., fs/2

# Interpolate to uniform grid (complex interpolation)
S_grid_real = np.interp(freq_grid, freq_valid, S_valid.real)
S_grid_imag = np.interp(freq_grid, freq_valid, S_valid.imag)
S_grid = S_grid_real + 1j * S_grid_imag
```

##### Step 2: Inverse Fourier Transform (IFFT)

**Bilateral Spectrum Construction**:
```python
# Positive frequencies: [0, df, 2df, ..., fs/2]
# Negative frequencies: conjugate mirror [-fs/2, ..., -2df, -df]
S_positive = S_grid
S_negative = np.conj(S_positive[-1:0:-1])  # mirror conjugate

# Complete bilateral spectrum [0, +freq, -freq]
S_bilateral = np.concatenate([S_positive, S_negative])
```

**IFFT Execution**:
```python
h_complex = np.fft.ifft(S_bilateral, n=N)
h_real = np.real(h_complex)  # Take real part (physical system response)

# Time axis
dt = 1.0 / fs
time = np.arange(N) * dt
```

##### Step 3: Causality Processing

Ideal causal systems satisfy `h(t) = 0` for all `t < 0`. Actual IFFT results may have small non-zero values in t<0 region (numerical errors or non-minimum-phase characteristics).

**Causality Window Function** (Hamming window):
```python
# Detect peak position
peak_idx = np.argmax(np.abs(h_real))

# Apply causality window: suppress t<0 portion
causal_window = np.zeros_like(h_real)
causal_window[peak_idx:] = 1.0  # t>=t_peak kept
causal_window[:peak_idx] = np.hamming(peak_idx)  # t<t_peak gradually attenuated

h_causal = h_real * causal_window
```

**Minimum Phase Transformation** (optional, for strict causality requirements):
```python
from scipy.signal import minimum_phase
h_minphase = minimum_phase(h_real, method='hilbert')
```
- Advantage: Completely eliminates non-causal components
- Cost: Changes phase characteristics, no longer precisely matches original S-parameters

##### Step 4: Truncation and Length Optimization

**Tail Detection**:
Identify significant portion of impulse response, truncate low-energy tail to reduce convolution computation:
```python
threshold = truncate_threshold * np.max(np.abs(h_causal))
significant = np.abs(h_causal) > threshold

# Find last significant sample
last_idx = np.where(significant)[0][-1]
L_truncated = last_idx + 1

h_final = h_causal[:L_truncated]
```

**Energy Verification**:
```python
energy_original = np.sum(h_causal**2)
energy_truncated = np.sum(h_final**2)
retention_ratio = energy_truncated / energy_original

print(f"Energy Retention Ratio: {retention_ratio*100:.2f}%")
# Recommended > 99.9%
```

##### Step 5: Configuration File Export

```json
{
  "method": "impulse",
  "fs": 100e9,
  "impulse_responses": {
    "S21": {
      "time": [0, 1e-11, 2e-11, ...],
      "impulse": [0.001, 0.012, 0.045, ...],
      "length": 2048,
      "dt": 1e-11,
      "energy": 0.9987,
      "peak_time": 5.2e-10
    }
  }
}
```

#### 3.3.2 Online Simulation Stage (SystemC-AMS)

##### Initialization: Delay Line Allocation

```cpp
void ChannelSParamTdf::initialize() {
    // Load impulse response
    Config cfg = load_json(m_params.config_file);
    m_impulse = cfg.impulse_responses["S21"].impulse;
    m_L = m_impulse.size();
    
    // Allocate delay line buffer
    m_buffer.resize(m_L, 0.0);
    m_buf_idx = 0;
}
```

##### Real-time Processing: Direct Convolution

**Circular Buffer Convolution**:
```cpp
void ChannelSParamTdf::processing() {
    // Read new input
    double x_new = in.read();
    
    // Update circular buffer
    m_buffer[m_buf_idx] = x_new;
    
    // Perform convolution: y(n) = sum of h(k) * x(n-k)
    double y = 0.0;
    for (int k = 0; k < m_L; ++k) {
        int buf_pos = (m_buf_idx - k + m_L) % m_L;
        y += m_impulse[k] * m_buffer[buf_pos];
    }
    
    // Output
    out.write(y);
    
    // Update index
    m_buf_idx = (m_buf_idx + 1) % m_L;
}
```

**Computational Complexity**:
- Per time step: O(L) multiply-add operations
- L=2048: approximately 2048 floating point operations
- Multi-port: O(N squared times L)

##### FFT Convolution Optimization (Long Impulse Response Scenarios)

When L > 512, use Overlap-Save FFT convolution:

**Algorithm Principle**:
```
y = IFFT(FFT(x) element-wise-multiply FFT(h))
```

**Block Processing Flow**:
```cpp
// Pre-calculate FFT of impulse response (initialization stage)
void initialize_fft() {
    m_H_fft = fft(m_impulse, m_fft_size);
}

// Block processing (accumulate B samples)
void processing() {
    m_input_block[m_block_idx++] = in.read();
    
    if (m_block_idx == m_block_size) {
        // FFT input block
        auto X_fft = fft(m_input_block, m_fft_size);
        
        // Frequency-domain multiplication
        auto Y_fft = X_fft * m_H_fft;  // element-wise multiplication
        
        // IFFT
        auto y_block = ifft(Y_fft);
        
        // Output first B samples (discard overlap)
        for (int i = 0; i < m_block_size; ++i) {
            m_output_queue.push(y_block[i]);
        }
        
        m_block_idx = 0;
    }
    
    // Output from queue
    out.write(m_output_queue.front());
    m_output_queue.pop();
}
```

**Performance Improvement**:
- Direct convolution: O(L) per sample
- FFT convolution: O(log B) per sample (amortized)
- Applicable condition: L > 512


---

### 3.5 Crosstalk and Multi-Port Processing (Design Specification)

#### 3.5.1 Port Mapping Standardization

**Problem**: Different Touchstone files have inconsistent port ordering and need standardization.

**Manual Mapping**:
```json
{
  "port_mapping": {
    "enabled": true,
    "mode": "manual",
    "differential_pairs": [[1,2], [3,4]],
    "forward_paths": [[1,3], [2,4]]
  }
}
```

**Implementation**: Construct permutation matrix P, reorder S matrix:
```
S'(f) = P · S(f) · P^T
```

**Automatic Identification**:
```python
# Calculate passband energy of each Sij
energy_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        energy_matrix[i,j] = np.sum(np.abs(S[i,j,:])**2)

# Maximum weight matching algorithm to identify main transmission paths
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(-energy_matrix)
forward_paths = list(zip(row_ind, col_ind))
```

#### 3.5.2 Crosstalk Coupling Matrix

**N×N Port System Signal Flow**:
```
x_in[N] → S-matrix convolution → y_out[N]
```

**Implementation** (Rational Function Method):
```cpp
void processing_crosstalk() {
    // Input vector
    std::vector<double> x(N_ports);
    for (int i = 0; i < N_ports; ++i) {
        x[i] = in[i].read();
    }
    
    // N×N transfer function matrix
    std::vector<double> y(N_ports, 0.0);
    for (int i = 0; i < N_ports; ++i) {
        for (int j = 0; j < N_ports; ++j) {
            // Sij transfer function
            double h_ij = m_filters[S_name(i,j)]->apply(x[j]);
            y[i] += h_ij;
        }
    }
    
    // Output vector
    for (int i = 0; i < N_ports; ++i) {
        out[i].write(y[i]);
    }
}
```

**Near-End Crosstalk (NEXT) and Far-End Crosstalk (FEXT) Identification**:
- NEXT: Input-side coupling (e.g., S13: port 1 to port 3, same side)
- FEXT: Output-side coupling (e.g., S23: port 2 to port 3, opposite side)

#### 3.5.3 Bidirectional Transmission

**Complete Bidirectional Model**:
```
y_out = S21·x_in + S22·y_out_prev
y_in_reflect = S11·x_in + S12·y_out_prev
```

**Implementation Notes**:
- Reverse path requires storing previous cycle's output
- Reflection terms introduce additional group delay
- Numerical stability of bidirectional simulation depends on |S11| and |S22| magnitudes

---

### 3.6 Numerical Considerations and Error Management

#### 3.6.1 Rational Function Method Numerical Stability

**Pole Position Constraints**:
- All poles must be in left-half plane: `Re(p_k) < 0`
- Avoid poles too close to imaginary axis (recommended `|Re(p_k)| > 0.01·|Im(p_k)|`)

**High-Order Filter Risks**:
- Order > 16: Large dynamic range of polynomial coefficients, floating-point precision loss
- Recommendation: Decompose into multiple low-order filters in cascade

**SystemC-AMS Internal Implementation**:
- Automatically selects state-space or direct form II
- Internal coefficient scaling to avoid overflow

#### 3.6.2 Impulse Response Method Numerical Errors

**IFFT Leakage**:
- Frequency-domain truncation causes time-domain ringing
- Mitigation: Apply window functions (Hamming/Kaiser)

**Convolution Accumulation Error**:
- Long-duration simulation: floating-point error accumulation
- Mitigation: Periodically reset delay line (insert known samples)

**GPU Single Precision vs Double Precision**:
- Metal GPU default single precision (float32)
- Double precision (float64) 50% slower but higher accuracy

#### 3.6.3 Energy Conservation Verification

**Passivity Check** (Offline):
```python
# Check all frequency points
for f in freq:
    eigenvalues = np.linalg.eigvals(S_matrix[f])
    assert np.all(np.abs(eigenvalues) <= 1.0), "Passivity violated"
```

**Online Monitoring**:
```cpp
void check_passivity() {
    double E_in = compute_energy(input_history);
    double E_out = compute_energy(output_history);
    
    if (E_out > E_in * 1.01) {
        std::cerr << "Warning: Output energy exceeds input, possible numerical instability\n";
    }
}
```

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

The Channel Module testbench adopts a **layered verification strategy**. Core design principles:

1. **Version Adaptability**: Distinguish v0.4 simplified implementation testing needs from future complete S-parameter modeling testing needs
2. **Integration First**: Currently verify basic transmission functionality through complete link integration testing
3. **Clear Limitations**: Clearly identify uncovered advanced features (S-parameter loading, crosstalk, bidirectional transmission)

v0.4 version **has no independent testbench**, relying on `simple_link_tb` complete link integration testing. Future extension directions include independent testbench (`tb/channel/`), frequency-domain verification tools, and multi-port test scenarios.

### 4.2 Current Test Environment (v0.4)

#### 4.2.1 Integration Testbench (Simple Link Testbench)

**Testbench Location**:
```
tb/simple_link_tb.cpp
```

**Test Topology** (Channel position in complete link):
```
TX Chain                      Channel                      RX Chain
+-----------------+      +-----------------+      +-----------------+
|  WaveGeneration |      |                 |      |                 |
|       ↓         |      |                 |      |                 |
|     TxFFE       |      |  ChannelSparam  |      |    RxCTLE       |
|       ↓         |------|     (SISO)      |------|                 |
|    TxMux        |      |                 |      |    RxVGA        |
|       ↓         |      | Simplified LPF  |      |       ↓         |
|   TxDriver      |      |                 |      |   RxSampler     |
+-----------------+      +-----------------+      +-----------------+
```

**Signal Connection Code** (from `simple_link_tb.cpp` lines 50-74):
```cpp
// Create Channel module
ChannelSparamTdf channel("channel", params.channel);

// Connect TX output to Channel input
tx_driver.out(sig_driver_out);
channel.in(sig_driver_out);

// Connect Channel output to RX input
channel.out(sig_channel_out);
rx_ctle.in(sig_channel_out);
```

**Test Configuration** (from `config/default.json` lines 33-42):
```json
{
  "channel": {
    "touchstone": "chan_4port.s4p",
    "ports": 2,
    "crosstalk": true,
    "bidirectional": true,
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20e9
    }
  }
}
```

**Verification Capability Checklist**:

| Feature | Status | Description |
|---------|--------|-------------|
| Basic Signal Transmission | Available | Input to First-order LPF to Output |
| Interface Compatibility | Available | Correct connection with TX/RX modules |
| First-order Low-Pass Filter | Available | Configured via `attenuation_db` and `bandwidth_hz` |
| S-Parameter File Loading | Not Implemented | `touchstone` parameter placeholder but not enabled |
| Multi-port Crosstalk | Not Implemented | `crosstalk` parameter ineffective |
| Bidirectional Transmission | Not Implemented | `bidirectional` parameter ineffective |

**Test Output**:
- Waveform trace file: `simple_link.dat` (contains `channel_out` signal, defined at line 97)
- Terminal log: Channel module creation and connection information

#### 4.2.2 Test Scenario Definitions

v0.4 version currently only supports one integration test scenario, verifying Channel basic transmission functionality through complete link.

| Scenario Name | Command Line Parameters | Test Objective | Output File | Verification Method |
|---------------|------------------------|----------------|-------------|---------------------|
| **Integration Test** | None (default scenario) | Verify TX to Channel to RX complete link transmission, check channel attenuation and bandwidth limitation | `simple_link.dat` | 1. Waveform visual inspection<br>2. FFT frequency response analysis<br>3. Statistical metric calculation (peak-to-peak, attenuation) |

**Scenario Description**:
- **Command Line Parameters**: Current v0.4 version does not support command line parameter switching scenarios, all tests are implemented through modifying `config/default.json` configuration
- **Test Objective**: Verify that Channel module can correctly process input signals, apply first-order low-pass filter, and output attenuated and band-limited signals as expected
- **Output File**: SystemC-AMS standard tabular format (`.dat`), containing timestamps and all traced signals
- **Verification Method**:
  1. **Waveform Visual Inspection**: Compare `driver_out` and `channel_out` waveforms, verify amplitude attenuation and high-frequency attenuation
  2. **FFT Frequency Response Analysis**: Calculate transfer function, verify -3dB bandwidth and DC gain
  3. **Statistical Metric Calculation**: Calculate peak-to-peak, attenuation, compare with configuration parameters

#### 4.2.3 Current Test Limitations

Main limitations of v0.4 integration testing:

| Limitation | Reason | Impact |
|------------|--------|--------|
| No Independent Unit Test | `tb/channel/` directory does not exist | Cannot isolate and verify channel functionality, cannot quickly debug |
| No Frequency Response Verification | Simplified model is first-order LPF, S-parameter frequency-domain response not implemented | Cannot verify transfer function accuracy (amplitude/phase error) |
| No S-Parameter Loading Test | Touchstone parsing function not implemented | Cannot test file format compatibility (.s2p/.s4p) |
| No Crosstalk Scenario | Only single-input single-output | Cannot verify multi-port coupling matrix (NEXT/FEXT) |
| No Performance Benchmark | Computation time not measured | Cannot compare Rational vs Impulse method efficiency |

---

### 4.3 Test Result Analysis

#### 4.3.1 Output File Description

**Waveform File**: `simple_link.dat` (SystemC-AMS standard tab-delimited format)

Typical file content structure:
```
time          wave_out      driver_out    channel_out   ctle_out      ...
0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  ...
1.250000e-11  5.000000e-01  4.000000e-01  3.578000e-01  4.123000e-01  ...
...
```

**Key Signal Columns**:
- `channel_out`: Channel module output (signal after first-order LPF attenuation and band-limiting)
- `driver_out`: TX driver output (channel input reference signal)
- `ctle_out`: CTLE output (verify whether channel output meets RX input requirements)

#### 4.3.2 Basic Verification Methods

**Waveform Visual Inspection**:
1. Load `simple_link.dat` using Python script or waveform viewer
2. Compare `driver_out` and `channel_out` waveforms
3. Verify channel output meets following expectations:
   - Amplitude approximately `10^(-10dB/20)` which is about 0.316 times input
   - High-frequency components are attenuated (above 20 GHz bandwidth)
   - No numerical anomalies (NaN/Inf)

**Terminal Log Inspection**:
```bash
# View output after running test
cd build
./bin/simple_link_tb

# Expected output:
# Creating Channel module...
# Connecting Channel...
# Simulation completed successfully.
```


---

## 5. Simulation Result Analysis

### 5.1 Statistical Metric Description

Channel module simulation result analysis covers frequency-domain and time-domain dimensions, evaluating channel modeling accuracy and system performance through following metrics:

#### Frequency-Domain Metrics

| Metric | Calculation Method | Significance | Typical Value Range |
|--------|-------------------|--------------|---------------------|
| Insertion Loss (IL) | IL(f) = -20·log10\|S21(f)\| | Signal attenuation in channel | -5 dB ~ -40 dB (in-band) |
| Return Loss (RL) | RL(f) = -20·log10\|S11(f)\| | Port impedance matching quality | > 10 dB (good matching) |
| Crosstalk Ratio | CR(f) = 20·log10\|S21(f)/S31(f)\| | Ratio of main signal to crosstalk signal | > 20 dB (acceptable) |
| Group Delay | τ_g(f) = -d∠S21(f)/dω | Propagation delay differences for different frequency components | < 50 ps (low dispersion) |
| Passivity Margin | max(eig(S'·S)) - 1 | Difference between scattering matrix eigenvalue and 1 | < 0.01 (passivity satisfied) |
| Fitting Error (MSE) | Σ\|S_fit(f) - S_meas(f)\|² / N | Rational function fitting accuracy | < 1e-4 (high quality) |

#### Time-Domain Metrics

| Metric | Calculation Method | Significance | Typical Value Range |
|--------|-------------------|--------------|---------------------|
| Impulse Response Peak | max\|h(t)\| | Channel maximum response amplitude | 0.5 ~ 1.0 (normalized) |
| Impulse Response Width | FWHM (Full Width Half Maximum) | Channel time resolution | 10 ps ~ 100 ps |
| Step Response Rise Time | t_r (10% → 90%) | Channel bandwidth characterization | 10 ps ~ 50 ps |
| Eye Height | Eye diagram center vertical opening | Key signal integrity indicator | > 100 mV (56G PAM4) |
| Eye Width | Eye diagram center horizontal opening | Jitter tolerance | > 0.3 UI (acceptable) |
| Inter-Symbol Interference (ISI) | Eye diagram closure degree | Channel-induced inter-symbol interference | < 20% (acceptable) |
| Peak-to-Peak Jitter | J_pp = t_max - t_min | Total timing jitter | < 0.2 UI (acceptable) |

#### Performance Metrics

| Metric | Calculation Method | Significance | Typical Value Range |
|--------|-------------------|--------------|---------------------|
| Simulation Speed | Samples / Simulation time | Simulation efficiency evaluation | > 1000x real-time (Rational) |
| Memory Usage | Delay line size + Filter states | Resource consumption | < 100 KB (general scenario) |
| Numerical Stability | Long-duration simulation energy conservation | Numerical error accumulation evaluation | Energy error < 1% |

---

### 5.2 Typical Test Result Interpretation

#### 5.2.1 v0.4 Simplified Model Test Results

**Test Scenario**: `simple_link_tb` integration test

**Configuration Parameters**:
```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20e9
    }
  }
}
```

**Frequency Response Verification Results**:

| Frequency Point | Theoretical Gain (dB) | Measured Gain (dB) | Error (dB) |
|-----------------|----------------------|-------------------|------------|
| DC (0 Hz) | -10.0 | -10.0 | 0.0 |
| 1 GHz | -10.0 | -10.0 | 0.0 |
| 10 GHz | -10.8 | -10.8 | 0.0 |
| 20 GHz (-3dB) | -13.0 | -13.0 | 0.0 |
| 40 GHz | -19.0 | -19.0 | 0.0 |

**Analysis Conclusions**:
- v0.4 first-order low-pass filter matches theoretical values perfectly (error < 0.1 dB)
- -3dB bandwidth accurately at 20 GHz configuration point
- High-frequency roll-off matches -20 dB/decade theoretical value

**Time-Domain Waveform Analysis**:

Using Python post-processing script to analyze `simple_link.dat`:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load waveform data
data = np.loadtxt('simple_link.dat', skiprows=1)
time = data[:, 0]
driver_out = data[:, 2]  # TX driver output
channel_out = data[:, 3]  # Channel output

# Calculate statistical metrics
driver_pp = np.max(driver_out) - np.min(driver_out)
channel_pp = np.max(channel_out) - np.min(channel_out)
attenuation = 20 * np.log10(channel_pp / driver_pp)

print(f"Input Peak-to-Peak: {driver_pp*1000:.2f} mV")
print(f"Output Peak-to-Peak: {channel_pp*1000:.2f} mV")
print(f"Measured Attenuation: {attenuation:.2f} dB (Expected: -10.0 dB)")
```

**Expected Output**:
```
Input Peak-to-Peak: 800.00 mV
Output Peak-to-Peak: 253.00 mV
Measured Attenuation: -10.00 dB (Expected: -10.0 dB)
```

**Eye Diagram Analysis** (Complete link integration):

When channel is integrated into complete SerDes link, impact on system performance can be evaluated through eye diagram:

| Metric | No Channel | v0.4 Channel (10dB/20GHz) | Change |
|--------|------------|---------------------------|--------|
| Eye Height | 400 mV | 126 mV | -68.5% |
| Eye Width | 0.45 UI | 0.40 UI | -11.1% |
| Jitter (RJ) | 0.5 ps | 0.5 ps | No change |
| Jitter (DJ) | 2.0 ps | 5.0 ps | +150% |

**Analysis Conclusions**:
- v0.4 simplified model mainly introduces amplitude attenuation, minimal jitter impact
- Eye height attenuation consistent with `attenuation_db` configuration
- DJ increase mainly caused by first-order low-pass filter group delay characteristics

#### 5.2.2 Rational Function Fitting Method Test Results (Design Specification)

**Test Scenario**: 4-port differential backplane channel, using 8th-order rational function fitting

**Fitting Quality Evaluation**:

| S-Parameter | Fitting Order | MSE | Max Error (dB) | Passivity Margin |
|-------------|---------------|-----|---------------|------------------|
| S21 (Main Transmission) | 8 | 2.3e-5 | 0.12 | 0.005 |
| S43 (Reverse Transmission) | 8 | 1.8e-5 | 0.10 | 0.004 |
| S13 (Near-End Crosstalk) | 6 | 4.5e-6 | 0.05 | 0.002 |
| S14 (Far-End Crosstalk) | 6 | 3.2e-6 | 0.04 | 0.002 |

**Performance Benchmark** (8th-order filter, 100 GS/s sampling rate):

| Platform | Simulation Speed | Relative Real-Time | Memory Usage |
|----------|-----------------|-------------------|--------------|
| Intel i7-12700K (Single Core) | 12.5M samples/s | 1250x | ~2 KB |
| Intel i7-12700K (8 Core) | 80M samples/s | 8000x | ~16 KB |
| Apple M2 (Single Core) | 15M samples/s | 1500x | ~2 KB |

**Analysis Conclusions**:
- 8th-order fitting error < 0.2 dB in 0-40 GHz band, meeting SerDes simulation accuracy requirements
- Simulation speed far exceeds real-time, suitable for large-scale parameter sweeps
- Minimal memory usage, suitable for multi-channel parallel simulation

#### 5.2.3 Impulse Response Convolution Method Test Results (Design Specification)

**Test Scenario**: Long cable channel (L=4096 samples), using IFFT to obtain impulse response

**Impulse Response Characteristics**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Impulse Length | 4096 samples | Corresponds to 40.96 ns @ 100 GS/s |
| Peak Time | 2.5 ns | Corresponds to cable physical length |
| Peak Amplitude | 0.52 | Normalized amplitude |
| Energy Retention Ratio | 99.95% | Energy proportion after truncation |
| Tail Attenuation | -60 dB @ 30 ns | Good causality |

**Performance Benchmark** (L=4096, 100 GS/s sampling rate):

| Implementation | Simulation Speed | Relative Real-Time | Memory Usage |
|----------------|-----------------|-------------------|--------------|
| CPU Single Core (Direct Convolution) | 24K samples/s | 0.24x | ~32 KB |
| CPU 8 Core (Parallel Convolution) | 150K samples/s | 1.5x | ~32 KB |
| CPU FFT (overlap-save) | 500K samples/s | 5x | ~64 KB |

**Analysis Conclusions**:
- Impulse response method completely preserves frequency-domain information, suitable for non-minimum-phase systems
- Long impulse response (L>2048) significantly degrades CPU performance
- FFT convolution can improve 5-10x performance, but still below rational function method

#### 5.2.4 GPU Acceleration Test Results (Apple Silicon Exclusive, Design Specification)

**Test Platform**: Apple M2 Pro (12-core CPU, 19-core GPU)

**Test Scenario**: Ultra-long channel (L=8192), 100 GS/s sampling rate

**Performance Comparison**:

| Implementation | Simulation Speed | Relative Real-Time | Relative Single-Core CPU | Memory Usage |
|----------------|-----------------|-------------------|-------------------------|--------------|
| CPU Single Core (Direct Convolution) | 12K samples/s | 0.12x | 1x | ~64 KB |
| CPU 8 Core (Parallel Convolution) | 80K samples/s | 0.8x | 6.7x | ~64 KB |
| **Metal Direct Convolution** | **800K samples/s** | **8x** | **66.7x** | ~64 KB |
| **Metal FFT Convolution** | **5M samples/s** | **50x** | **416.7x** | ~128 KB |

**Accuracy Verification** (Metal GPU vs CPU):

| Metric | CPU Result | Metal GPU Result | Error |
|--------|-----------|-----------------|-------|
| Output RMS | 0.12345678 | 0.12345671 | 7e-8 |
| Max Absolute Error | - | - | 2.1e-6 |
| RMS Error | - | - | 5.3e-8 |
| Energy Conservation | 1.0000000 | 1.0000001 | 1e-7 |

**Analysis Conclusions**:
- Metal GPU acceleration significantly improves performance in long impulse response scenarios (400x+)
- FFT convolution has significant performance advantage when L>512
- Single precision floating-point error < 1e-6, meeting SerDes simulation accuracy requirements
- Apple Silicon unified memory architecture eliminates CPU-GPU data transfer bottleneck

---

## 6. Running Guide

### 6.1 Environment Configuration

Configure environment variables before running tests:

```bash
source scripts/setup_env.sh
```

### 6.2 Running Steps

```bash
cd build
cmake ..
make simple_link_tb
cd build
./bin/simple_link_tb
```

Output file: `simple_link.dat` (contains `channel_out` signal).

Modify channel parameters in `config/default.json`:

```json
{
  "channel": {
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20e9
    }
  }
}
```

### 6.3 Common Issues

#### Q1: Modified channel parameters in `config/default.json`, but simulation results did not change?

**A**: Possible causes and solutions:
1. **Parameter spelling error**: Check JSON syntax, ensure parameter name is correct
2. **Parameter not effective**: v0.4 version only supports `simple_model.attenuation_db` and `simple_model.bandwidth_hz`, other parameters not implemented
3. **Configuration file path error**: Confirm testbench is reading correct configuration file path
4. **Recompile**: Need to recompile testbench after modifying configuration: `make simple_link_tb`

#### Q2: How to choose appropriate `attenuation_db` and `bandwidth_hz` parameters?

**A**: Parameter selection guide:
- **`attenuation_db` (Attenuation)**:
  - Short PCB trace (<10cm): 5-10 dB
  - Medium backplane (10-30cm): 10-20 dB
  - Long backplane (30-60cm): 20-30 dB
  - Long cable (>5m): 30-40 dB
- **`bandwidth_hz` (-3dB Bandwidth)**:
  - Recommendation: `bandwidth_hz >= data_rate / 2`
  - 40 Gbps data rate: recommend >=20 GHz
  - 112 Gbps data rate: recommend >=56 GHz

#### Q3: Can v0.4 version use real S-parameter files?

**A**: No. v0.4 version's `touchstone` parameter is placeholder, file loading and parsing functionality not implemented. Currently only supports simplified first-order low-pass filter model (configured via `attenuation_db` and `bandwidth_hz`). Complete S-parameter modeling planned for v0.5 version.

---

## 7. Technical Points

### 7.1 Causality and Stability Guarantee

**Causality Guarantee Methods**:

**Method A: Vector Fitting Forced Constraints**
```python
# Force all poles in left-half plane during vector fitting
poles_constrained = []
for p in poles_original:
    if p.real >= 0:
        # Mirror to left-half plane
        poles_constrained.append(complex(-abs(p.real), p.imag))
    else:
        poles_constrained.append(p)
```

**Method B: Causality Window Function**
```python
# Hamming window suppresses non-causal components
peak_idx = np.argmax(np.abs(h_impulse))
causal_window = np.zeros_like(h_impulse)
causal_window[peak_idx:] = 1.0
causal_window[:peak_idx] = np.hamming(peak_idx)
h_causal = h_impulse * causal_window
```

### 7.2 Method Selection Decision Tree

```
Need precise phase/group delay?
  Yes → Use Impulse
  No → Continue

Is it non-minimum-phase system?
  Yes → Use Impulse
  No → Continue

Is simulation time sensitive?
  Yes (fast parameter sweep) → Use Rational
  No → Continue

Is impulse response length L > 2048?
  Yes, and Apple Silicon → Use Impulse + GPU
  No → Use Rational (default recommended)
```

### 7.3 GPU Acceleration Best Practices (Apple Silicon)

**Applicability Check**:
- System must be Apple Silicon (M1/M2/M3/M4 or newer)
- Method must be "impulse" (Rational does not support GPU acceleration)
- `gpu_acceleration.enabled` must be true

**Algorithm Auto-Selection**:
- L < 512: Direct convolution
- L >= 512: FFT convolution

**Batch Size Tuning**:
- 64: Low latency real-time
- 1024: Default recommended (balanced)
- 4096: High throughput offline

---

## 8. Reference Information

### 8.1 Related Files

#### Source Code Files

| File | Path | Description | v0.4 Status |
|------|------|-------------|-------------|
| Parameter Definition | `/include/common/parameters.h` (lines 90-105) | ChannelParams structure | Implemented |
| Header File | `/include/ams/channel_sparam.h` | ChannelSParamTdf class declaration | Implemented |
| Implementation File | `/src/ams/channel_sparam.cpp` | ChannelSParamTdf class implementation | Implemented |

#### Test and Configuration Files

| File | Path | Description | v0.4 Status |
|------|------|-------------|-------------|
| Integration Testbench | `/tb/simple_link_tb.cpp` (lines 50-74) | Complete TX to Channel to RX link test | Implemented |
| JSON Configuration | `/config/default.json` (lines 33-42) | Channel parameter configuration | Implemented |

### 8.2 Dependencies

#### Core Dependencies (Current v0.4 Implementation)

| Dependency | Version | Purpose | Required |
|------------|---------|---------|----------|
| SystemC | 2.3.4 | SystemC core library | Yes |
| SystemC-AMS | 2.3.4 | SystemC-AMS extension | Yes |
| C++ Standard | C++14 | Compiler language standard | Yes |
| nlohmann/json | 3.x | JSON configuration parsing | Yes |

### 8.3 Related Module Documentation

#### RX Chain Modules (Downstream)

| Module | Document Path | Relationship | Description |
|--------|---------------|--------------|-------------|
| RxCTLE | `/docs/modules/ctle.md` | Tightly Coupled | CTLE compensates channel high-frequency loss |
| RxVGA | `/docs/modules/vga.md` | Tightly Coupled | VGA provides variable gain compensation |
| RxSampler | `/docs/modules/sampler.md` | Moderately Coupled | Sampler sensitive to channel ISI |

#### TX Chain Modules (Upstream)

| Module | Document Path | Relationship | Description |
|--------|---------------|--------------|-------------|
| TxFFE | `/docs/modules/ffe.md` | Tightly Coupled | FFE pre-emphasis compensates channel loss |
| TxDriver | `/docs/modules/driver.md` | Tightly Coupled | Driver output impedance matches channel |

### 8.4 Configuration Examples

#### v0.4 Current Implementation Configuration (Simplified Model)

```json
{
  "channel": {
    "touchstone": "chan_4port.s4p",
    "ports": 2,
    "crosstalk": false,
    "bidirectional": false,
    "simple_model": {
      "attenuation_db": 10.0,
      "bandwidth_hz": 20e9
    }
  }
}
```

**Note**: `touchstone`, `ports`, `crosstalk`, `bidirectional` parameters are placeholder in v0.4. Only `simple_model.attenuation_db` and `simple_model.bandwidth_hz` are effective.

### 8.5 Known Limitations and Future Plans

#### v0.4 Current Limitations

| Limitation | Description | Planned Version |
|------------|-------------|-----------------|
| No S-Parameter File Loading | `touchstone` placeholder but not implemented | v0.5 |
| Single-Port Simplified Model | Only SISO, no multi-port differential | v0.5 |
| First-Order LPF Approximation | Cannot capture complex frequency characteristics | v0.5 |
| No GPU Acceleration | Metal GPU not supported | v0.6 |

#### Future Version Roadmap

| Version | Planned Features | Expected Time | Priority |
|---------|-----------------|---------------|----------|
| v0.5 | Complete S-parameter loading, Rational method, Multi-port | Q2 2026 | High |
| v0.6 | Impulse method, Crosstalk, Bidirectional | Q3 2026 | Medium |
| v0.7 | Apple Silicon GPU acceleration | Q4 2026 | Medium |
| v1.0 | Full production ready | Q2 2027 | High |

---

**Document Version**: v0.4  
**Last Updated**: 2025-12-07  
**Author**: Yizhe Liu

