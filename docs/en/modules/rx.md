# RX Receiver Module Technical Documentation

🌐 **Languages**: [中文](../../modules/rx.md) | [English](rx.md)

**Level**: AMS Top-Level Module  
**Current Version**: v1.0 (2026-01-27)  
**Status**: Production Ready

---

## 1. Overview

The SerDes Receiver (RX) is a core component of the high-speed serial link, responsible for recovering the original digital bitstream from analog differential signals that have been attenuated and distorted by the channel. RX achieves comprehensive compensation of channel impairments through multi-stage equalization, automatic gain control, decision feedback, and clock data recovery techniques.

### 1.1 Design Principles

The core design philosophy of the RX receiver adopts a hierarchical cascaded architecture, where each sub-module focuses on specific signal processing tasks:

```
╔════════════════════════════════════════════════════════════╗
║                     DE Domain Control Layer (Adaption)                      ║
║  AGC Control | DFE Tap Update | Threshold Adaptation | CDR Control Interface | Freeze/Rollback  ║
╚════════════════════════════════════════════════════════════╝
            │ vga_gain      │ dfe_taps      │ threshold     │
            ▼               ▼               ▼               ▼
Channel Output → CTLE → VGA → DFE Summer → Sampler → Digital Output
                                ↑           ↓
                            Historical Decision    Sampled Data
                                ↑           ↓
                            data_out  ←  CDR  ← phase_offset
                                            ↑
                                      phase_error
                                            ↑
                              ╔═══════════════╗
                              ║  DE-TDF Bridge Layer  ║
                              ╚═══════════════╝
```

**Signal Flow Processing Logic**:

1. **CTLE (Continuous-Time Linear Equalizer)**: Frequency-domain equalization, boosting high-frequency gain through zero-pole transfer functions to compensate for frequency-dependent channel loss
2. **VGA (Variable Gain Amplifier)**: Amplitude adjustment, dynamically controlling signal swing to the optimal range in conjunction with AGC algorithm
3. **DFE Summer (Decision Feedback Equalizer)**: Time-domain equalization, using feedback from already-decided symbols to cancel post-cursor inter-symbol interference (ISI)
4. **Sampler**: Threshold decision, performing binary decision at the optimal sampling moment specified by CDR
5. **CDR (Clock Data Recovery)**: Phase tracking, extracting clock information from data transitions, dynamically adjusting sampling phase

**Hierarchical Equalization Strategy**:

- **CTLE handles linear ISI**: Through continuous-time frequency-domain equalization, compensates for high-frequency channel attenuation, but amplifies high-frequency noise
- **DFE handles non-linear ISI**: Through time-domain decision feedback equalization, eliminates post-cursor ISI without amplifying noise but with error propagation risk
- **Gain allocation balance**: Total gain of CTLE and VGA must bring Sampler input signal swing to the optimal decision range (typically 200-600mV)

### 1.2 Core Features

- **Five-stage cascaded architecture**: CTLE → VGA → DFE → Sampler → CDR, covering the complete receiver signal chain
- **Differential signal path**: Full differential transmission, CTLE/VGA/Sampler all use differential input/output, common-mode rejection ratio >40dB
- **Closed-loop clock recovery**: CDR provides sampling phase feedback, Sampler receives phase adjustment signal, forming a phase-locked loop
- **Multi-domain collaboration**: TDF domain analog modules (CTLE/VGA/DFE/Sampler/CDR) work in coordination with DE domain adaptive modules (Adaption)
- **Configurable equalization depth**:
  - CTLE: Multi-zero-pole transfer function, frequency-domain equalization
  - VGA: Variable gain amplification, dynamic range control
  - DFE: 1-8 taps configurable, post-cursor ISI cancellation
- **Adaptive optimization**: Supports LMS/Sign-LMS/NLMS and other adaptive algorithms for dynamic equalization parameter optimization
- **Non-ideal effect modeling**: Integrates offset, noise, PSRR, CMFB, CMRR, saturation and other practical device characteristics
- **DE domain adaptive control** (Adaption module):
  - AGC automatic gain control: PI controller dynamically adjusts VGA gain
  - DFE tap online update: LMS/Sign-LMS/NLMS algorithms optimize tap coefficients
  - Threshold adaptation: Dynamically tracks level drift, optimizes decision threshold
  - CDR control interface: Provides parameterized configuration and monitoring for TDF domain CDR
  - Safety mechanisms: Freeze/rollback strategies to prevent algorithm divergence
- **Multi-rate scheduling architecture**: Fast path (threshold adjustment, every 10-100 UI) and slow path (AGC/DFE, every 1000-10000 UI) running in parallel

### 1.3 Sub-module Overview

| Module | Class Name | Function | Key Parameters | Independent Documentation |
|------|------|------|---------|---------|
| **CTLE** | `RxCtleTdf` | Continuous-Time Linear Equalizer | zeros, poles, dc_gain | ctle.md |
| **VGA** | `RxVgaTdf` | Variable Gain Amplifier | dc_gain(adjustable), zeros, poles | vga.md |
| **DFE Summer** | `RxDfeSummerTdf` | Decision Feedback Equalizer Summer | tap_coeffs, vtap, map_mode | dfesummer.md |
| **Sampler** | `RxSamplerTdf` | Sampler/Decision Circuit | resolution, hysteresis, phase_source | sampler.md |
| **CDR** | `RxCdrTdf` | Clock Data Recovery | kp, ki, resolution, range | cdr.md |
| **Adaption** | `AdaptionDe` | DE Domain Adaptive Control Hub | agc, dfe, threshold, cdr_pi, safety | adaption.md |

### 1.4 Version History

| Version | Date | Major Changes |
|------|------|----------|
| v1.0 | 2026-01-27 | Initial version, integrating top-level documentation of five sub-modules |
| v1.1 | 2026-01-28 | Added adaption module |

---

## 2. Module Interfaces

### 2.1 Port Definitions (TDF Domain)

#### 2.1.1 Top-Level Input/Output Ports

| Port Name | Direction | Type | Description |
|-------|------|------|------|
| `in_p` | Input | double | Differential input positive terminal from channel |
| `in_n` | Input | double | Differential input negative terminal from channel |
| `vdd` | Input | double | Supply voltage (for PSRR modeling) |
| `data_out` | Output | int | Recovered digital bitstream (0/1) |

> **Important**: Even if PSRR functionality is not enabled, the `vdd` port must be connected (SystemC-AMS requires all ports to be connected).

#### 2.1.2 Internal Module Cascade Relationships

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                   RX Receiver Top-Level Module                                    ║
║                                                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              DE Domain (Adaption)                                    │ ║
║  │  ┌─────────┐  ┌────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐                  │ ║
║  │  │   AGC   │  │  DFE   │  │Threshold│  │ CDR PI  │  │Safety  │                  │ ║
║  │  │   PI    │  │  LMS   │  │ Adapt   │  │ Ctrl    │  │Mechan  │                  │ ║
║  │  └─────────┘  └────────┘  └─────────┘  └─────────┘  └────────┘                  │ ║
║  └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                 │ vga_gain     │ dfe_taps    │ threshold    │ phase_cmd              ║
║                 ▼              ▼             ▼              ▼                        ║
║  ┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐        ║
║  │  CTLE   │───▶│   VGA   │───▶│ DFE Summer  │───▶│ Sampler  │───▶│  CDR    │        ║
║  │         │    │         │    │             │    │          │    │         │        ║
║  │ in_p ←──┼────┼─ out_p  │    │             │    │          │    │         │        ║
║  │ in_n ←──┼────┼─ out_n  │    │             │    │          │    │         │        ║
║  │         │    │         │    │             │    │          │    │         │        ║
║  │ out_p ──┼───▶│ in_p    │    │             │    │          │    │         │        ║
║  │ out_n ──┼───▶│ in_n    │    │ in_p ←──────┼───▶│ out_p    │    │ in      │        ║
║  │         │    │         │    │ in_n ←──────┼───▶│ out_n    │    │         │        ║
║  │ vdd ←───┼───▶│ vdd     │    │             │    │          │    │         │        ║
║  └─────────┘    └─────────┘    │             │    │          │    │         │        ║
║       │              │         │             │    │          │    │         │        ║
║       │              │         │ data_in     │    │          │    │         │        ║
║       │              │         │             │    │          │    │         │        ║
║       │              │         └─────────────┘    │          │    │         │        ║
║       │              │                            │ data_out │    │         │        ║
║       │              │                            │          │    │         │        ║
║       │              │                            └──────────┘    │ phase   │        ║
║       │              │                                            │ _out    │        ║
║       │              │                                            └─────────┘        ║
║       │              │                                                               ║
║       │              └──────────────────────────────────────────────────────────────-║
║       │                                                                              ║
║       └─────────────────────────────────────────────────────────────────────────────-║
║                                                                                      ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                              TDF Domain (RX Processing Chain)                                   │  ║
║  │                                                                                │  ║
║  │  ┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐  │  ║
║  │  │  CTLE   │───▶│   VGA   │───▶│ DFE Summer  │───▶│ Sampler  │───▶│  CDR    │  │  ║
║  │  │         │    │         │    │             │    │          │    │         │  │  ║
║  │  │ in_p ←──┼────┼─ out_p  │    │             │    │          │    │         │  │  ║
║  │  │ in_n ←──┼────┼─ out_n  │    │             │    │          │    │         │  │  ║
║  │  │         │    │         │    │             │    │          │    │         │  │  ║
║  │  │ out_p ──┼────┼→ in_p   │    │             │    │          │    │         │  │  ║
║  │  │ out_n ──┼────┼→ in_n   │    │ in_p ←──────┼────┼─ out_p   │    │ in      │  │  ║
║  │  │         │    │         │    │ in_n ←──────┼────┼─ out_n   │    │         │  │  ║
║  │  │ vdd ←───┼────┼─ vdd    │    │             │    │          │    │         │  │  ║
║  │  └─────────┘    │ out_p ──┼────┼→ in_p       │    │ inp ←────┼────|───out_p │  │  ║
║  │       │         │ out_n ──┼────┼→ in_n       │    │ inn ←────┼────|───out_n │  │  ║
║  │       │         │         │    │             │    │          │    |         │  │  ║
║  │      VDD        │ vdd ←───┼────┼─────────────┼────┼──────────┼────| vdd     │  │  ║
║  │                 └─────────┘    │             │    │          │    |         │  │  ║
║  │                                │ data_in ←───┼────┼──────────┼────|─data_out│  │  ║
║  │                                │ (Historical Decision)   |    │          │    |         │  │  ║
║  │                                │             │    │          │    |         │  │  ║
║  │                                └─────────────┘    │ phase ←─┼─────|phase_o  │  │  ║
║  │                                                   │ _offset │     │ offset  │  │  ║
║  │                                                   └─────────┘     └─────────┘  │  ║
║  │                                                                                │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                      ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                           DE-TDF Bridge Layer (Signal Flow)                                  │  ║
║  │                                                                                │  ║
║  │  phase_error ← CDR.phase_out    │    phase_cmd → CDR.phase_cmd                 │  ║
║  │  amplitude_rms ← VGA.output     │    vga_gain → VGA.gain_setting               │  ║
║  │  error_count ← Sampler.errors   │    threshold → Sampler.threshold             │  ║
║  │  isi_metric ← DFE.isi           │    dfe_taps → DFE.tap_coeffs                 │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

**Key Signal Flow**:

- **Forward path**: `in_p/in_n` → CTLE → VGA → DFE Summer → Sampler → `data_out`
- **DFE feedback path**: Sampler.data_out (historical decision) → DFE.data_in (tap input)
- **CDR closed-loop path**: Sampler.data_out → CDR.in → CDR.phase_out → Sampler.phase_offset
- **DE-TDF bridge path**: Adaption ↔ VGA/DFE/Sampler/CDR (parameter control and feedback)

### 2.2 Port Definitions (DE Domain - Adaption Module)

The Adaption module serves as the DE domain adaptive control hub, interacting with TDF domain modules through the DE-TDF bridging mechanism.

#### 2.2.1 Adaption Input Ports

| Port Name | Type | Description |
|-------|------|------|
| `phase_error` | double | Phase error from CDR (s or normalized UI) |
| `amplitude_rms` | double | RMS value from RX amplitude statistics |
| `error_count` | int | Decision error count from Sampler |
| `isi_metric` | double | ISI metric (optional, for DFE strategy) |
| `mode` | int | Operating mode: 0=init, 1=training, 2=data, 3=freeze |
| `reset` | bool | Global reset signal |
| `scenario_switch` | double | Scenario switching event (optional) |

#### 2.2.2 Adaption Output Ports

| Port Name | Type | Description | Target Module |
|-------|------|------|----------|
| `vga_gain` | double | VGA gain setting (linear) | VGA |
| `ctle_zero` | double | CTLE zero frequency (Hz, optional) | CTLE |
| `ctle_pole` | double | CTLE pole frequency (Hz, optional) | CTLE |
| `ctle_dc_gain` | double | CTLE DC gain (linear, optional) | CTLE |
| `dfe_tap1`~`dfe_tap8` | double | DFE tap coefficients (fixed 8 independent ports) | DFE Summer |
| `sampler_threshold` | double | Sampler threshold (V) | Sampler |
| `sampler_hysteresis` | double | Sampler hysteresis window (V) | Sampler |
| `phase_cmd` | double | Phase interpolator command (s) | CDR |
| `update_count` | int | Update counter (for diagnostics) | External monitoring |
| `freeze_flag` | bool | Freeze/rollback status flag | External monitoring |

#### 2.2.3 DE-TDF Bridge Relationship Diagram

```
┌──────────────────────────────────────────────────────┐
│                    DE Domain (Adaption)                    │
│    ┌─────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐    │
│    │   AGC   │ │  DFE   │ │Threshold│ │ CDR PI  │    │
│    │   PI    │ │  LMS   │ │ Adapt   │ │ Ctrl*   │    │
│    └────┬────┘ └────┬───┘ └────┬────┘ └────┬────┘    │
│         │         │          │          │            │
└─────────┼─────────┼──────────┼──────────┼────────────┘
          │         │          │          │  DE-TDF Bridge
          ▼         ▼          ▼          ▼
┌─────────┼─────────┼──────────┼──────────┼────────────┐
│       vga_gain dfe_taps  threshold  phase_cmd        │
│         │         │          │          │            │
│         ▼         ▼          ▼          ▼            │
│      ┌─────┐ ┌───────┐ ┌─────────┐ ┌───────┐         │
│      │ VGA │ │  DFE  │ │ Sampler │ │  CDR  │         │
│      └─────┘ └───────┘ └─────────┘ └───────┘         │
│                    TDF Domain                             │
└──────────────────────────────────────────────────────┘

* CDR PI Ctrl: Provides parameterized configuration and monitoring for TDF domain CDR, detailed principles see cdr.md
```

### 2.3 Parameter Configuration (RxParams Structure)

#### 2.3.1 Overall Parameter Structure

```cpp
struct RxParams {
    RxCtleParams ctle;          // CTLE parameters
    RxVgaParams vga;            // VGA parameters
    RxSamplerParams sampler;    // Sampler parameters
    RxDfeParams dfe;            // DFE parameters
};

// CDR parameters defined independently
struct CdrParams {
    CdrPiParams pi;             // PI controller parameters
    CdrPaiParams pai;           // Phase interpolator parameters
    double ui;                  // Unit interval (s)
    bool debug_enable;          // Debug output enable
};
```

#### 2.3.2 Sub-module Parameter Summary

| Sub-module | Key Parameters | Default Configuration | Adjustment Purpose |
|--------|---------|---------|----------|
| CTLE | `zeros=[2e9]`, `poles=[30e9]`, `dc_gain=1.5` | Single zero single pole | High-frequency boost, bandwidth limiting |
| VGA | `zeros=[1e9]`, `poles=[20e9]`, `dc_gain=2.0` | Variable gain | AGC dynamic adjustment |
| DFE | `taps=[-0.05,-0.02,0.01]`, `mu=1e-4` | 3 taps | Post-cursor ISI cancellation |
| Sampler | `resolution=0.02`, `hysteresis=0.02` | Fuzzy decision | Metastability modeling |
| CDR | `kp=0.01`, `ki=1e-4`, `resolution=1e-12` | PI controller | Phase tracking |
| Adaption | `agc.target=0.4`, `dfe.mu=1e-4`, `safety.freeze_threshold=1000` | Multi-rate scheduling | Adaptive optimization |

#### 2.3.4 Adaption Parameter Details

| Sub-structure | Parameter | Default Value | Description |
|----------|------|--------|------|
| **AGC** | `target_amplitude` | 0.4 | Target output amplitude (V) |
| | `kp`, `ki` | 0.01, 1e-4 | PI controller coefficients |
| | `gain_min`, `gain_max` | 0.5, 10.0 | Gain range limits |
| **DFE** | `algorithm` | "sign-lms" | Update algorithm: lms/sign-lms/nlms |
| | `mu` | 1e-4 | Step size parameter |
| | `tap_min`, `tap_max` | -0.5, 0.5 | Tap coefficient range |
| **Threshold** | `adaptation_rate` | 1e-3 | Threshold adjustment rate |
| | `threshold_min`, `threshold_max` | -0.2, 0.2 | Threshold range (V) |
| **CDR PI*** | `enabled` | false | Whether to enable DE domain CDR control interface |
| | `kp`, `ki` | 0.01, 1e-4 | PI controller coefficients |
| **Safety** | `freeze_threshold` | 1000 | Error count threshold triggering freeze |
| | `snapshot_interval` | 10000 | Snapshot save interval (UI) |
| **Scheduling** | `fast_update_period` | 10-100 UI | Fast path update period |
| | `slow_update_period` | 1000-10000 UI | Slow path update period |

> \* CDR PI Control Interface: Only provides parameterized configuration and monitoring for TDF domain CDR, for complete CDR technical documentation see [cdr.md](cdr.md)

#### 2.3.5 Configuration Example (JSON Format)

```json
{
  "rx": {
    "ctle": {
      "zeros": [2e9],
      "poles": [30e9],
      "dc_gain": 1.5,
      "vcm_out": 0.6,
      "psrr": {"enable": false},
      "cmfb": {"enable": true, "bandwidth": 1e6},
      "cmrr": {"enable": false}
    },
    "vga": {
      "zeros": [1e9],
      "poles": [20e9],
      "dc_gain": 2.0,
      "vcm_out": 0.6
    },
    "dfe": {
      "taps": [-0.05, -0.02, 0.01],
      "update": "sign-lms",
      "mu": 1e-4
    },
    "sampler": {
      "threshold": 0.0,
      "resolution": 0.02,
      "hysteresis": 0.02,
      "phase_source": "phase"
    }
  },
  "cdr": {
    "pi": {"kp": 0.01, "ki": 1e-4, "edge_threshold": 0.5},
    "pai": {"resolution": 1e-12, "range": 5e-11},
    "ui": 1e-10
  },
  "adaption": {
    "agc": {
      "target_amplitude": 0.4,
      "kp": 0.01,
      "ki": 1e-4,
      "gain_min": 0.5,
      "gain_max": 10.0
    },
    "dfe": {
      "algorithm": "sign-lms",
      "mu": 1e-4,
      "tap_min": -0.5,
      "tap_max": 0.5
    },
    "threshold": {
      "adaptation_rate": 1e-3,
      "threshold_min": -0.2,
      "threshold_max": 0.2
    },
    "cdr_pi": {
      "enabled": false,
      "kp": 0.01,
      "ki": 1e-4
    },
    "safety": {
      "freeze_threshold": 1000,
      "snapshot_interval": 10000
    },
    "scheduling": {
      "fast_update_period_ui": 50,
      "slow_update_period_ui": 5000
    }
  }
}
```

---

## 3. Core Implementation Mechanisms

### 3.1 Signal Processing Flow

The complete signal processing flow of the RX receiver includes 7 key steps:

```
Step 1: Channel output reading → Differential signal in_p/in_n
Step 2: CTLE equalization    → Frequency-domain compensation, high-frequency boost
Step 3: VGA amplification     → Amplitude adjustment, dynamic gain control
Step 4: DFE feedback cancellation → Subtract ISI component from historical decision feedback
Step 5: Sampler decision → Threshold decision at optimal phase moment
Step 6: Update DFE history → Push current decision into history buffer
Step 7: CDR phase update → Detect edge, adjust next sampling phase
```

**Timing Constraints**:

- All TDF modules must run at the same sampling rate (typically 10-20x symbol rate)
- DFE feedback delay ≥ 1 UI (avoid algebraic loop)
- CDR phase update delay typically 1-2 UI

### 3.2 CTLE-VGA Cascade Design

#### 3.2.1 Gain Allocation Strategy

Total gain requirement calculation:

```
G_total = V_sampler_min / V_channel_out
```

Where:
- `V_sampler_min`: Sampler minimum input swing (typically 100-200mV)
- `V_channel_out`: Channel output swing (depends on channel loss)

**Recommended Gain Allocation**:

| Channel Loss | CTLE Gain | VGA Gain | Total Gain |
|---------|---------|---------|-------|
| Mild (5dB) | 1.2 | 1.5 | 1.8 |
| Moderate (10dB) | 1.5 | 2.0 | 3.0 |
| Severe (15dB) | 2.0 | 3.0 | 6.0 |
| Extreme (20dB) | 2.5 | 4.0 | 10.0 |

#### 3.2.2 Common-Mode Voltage Management

- CTLE output common-mode = VGA input common-mode = 0.6V (configurable)
- Both stages use independent CMFB loops to stabilize common-mode voltage
- Avoid inter-stage common-mode mismatch causing non-linear distortion

#### 3.2.3 Bandwidth Matching

- CTLE pole frequency (30GHz) >> VGA pole frequency (20GHz)
- VGA serves as secondary filtering, further suppressing high-frequency noise
- System total bandwidth determined by lowest pole frequency

### 3.3 DFE Feedback Loop Design

#### 3.3.1 Historical Decision Maintenance

```cpp
// Pseudocode example
std::vector<int> history_bits(N_taps);  // N_taps = DFE tap count

void update_history(int new_bit) {
    // Shift operation: new decision enters position 0, old decisions shift back
    for (int i = N_taps - 1; i > 0; i--) {
        history_bits[i] = history_bits[i - 1];
    }
    history_bits[0] = new_bit;
}

// DFE Summer reads history_bits each UI to calculate feedback voltage
double compute_feedback() {
    double feedback = 0.0;
    for (int i = 0; i < N_taps; i++) {
        // Map 0/1 to -1/+1
        int symbol = (history_bits[i] == 1) ? +1 : -1;
        feedback += taps[i] * symbol * vtap;
    }
    return feedback;
}
```

#### 3.3.2 Tap Coefficient Calculation

- **Method 1**: Channel impulse response measurement + post-cursor sampling + proportional scaling
- **Method 2**: LMS adaptive algorithm online optimization
- **Normalization constraint**: `Σ|tap_coeffs[k]| < 0.5` (avoid over-compensation)

#### 3.3.3 Stability Considerations

- Excessive DFE tap coefficients lead to error propagation
- Requires CDR phase alignment to ensure optimal decision moment
- Adaptive algorithm requires convergence verification

### 3.4 Sampler-CDR Closed-Loop Mechanism

#### 3.4.1 Phase Detection and Adjustment

1. **Sampler**: Dynamically adjusts sampling moment based on `phase_offset` signal
2. **CDR**: Detects relationship between data edge and current phase, calculates phase error
3. **PI Controller**: Updates phase accumulation based on error
4. **Phase Quantization**: Quantizes according to PAI resolution and outputs to Sampler

#### 3.4.2 Bang-Bang Phase Detection Algorithm

```cpp
// Phase detection logic in CDR
double bit_diff = current_bit - prev_bit;
double phase_error = 0.0;

if (std::abs(bit_diff) > edge_threshold) {
    if (bit_diff > 0)
        phase_error = +1.0;   // Rising edge: clock late, needs advance
    else
        phase_error = -1.0;   // Falling edge: clock early, needs delay
}

// PI controller update
integral += ki * phase_error;
double prop_term = kp * phase_error;
double pi_output = prop_term + integral;
phase = pi_output * ui;  // Scale to seconds
```

#### 3.4.3 Locking Process

- **Initial stage**: Large phase error, PI controller adjusts rapidly
- **Convergence stage**: Error gradually decreases, phase jitter converges to Bang-Bang PD inherent level (1-5ps RMS)
- **Steady-state stage**: Phase locked, tracking frequency offset and low-frequency jitter

#### 3.4.4 Closed-Loop Bandwidth

- Typical value: 1-10MHz (far below data rate)
- Function: Track low-frequency jitter, suppress high-frequency noise
- Adjustment method: Modify CDR Kp/Ki parameters

### 3.5 Adaptive Optimization Mechanism (Adaption Module)

The Adaption module serves as the DE domain adaptive control hub, performing online parameter updates to TDF domain modules through the DE-TDF bridging mechanism. For detailed technical documentation, see [adaption.md](adaption.md).

#### 3.5.1 Adaptive Algorithm Overview

| Algorithm | Target Module | Adaptive Parameters | Update Period | Implementation Method |
|------|----------|-----------|---------|----------|
| AGC | VGA | dc_gain | 1000-10000 UI (slow path) | PI controller |
| DFE tap update | DFE Summer | tap_coeffs | 1000-10000 UI (slow path) | LMS/Sign-LMS/NLMS |
| Threshold adaptation | Sampler | threshold | 10-100 UI (fast path) | Statistical tracking |
| CDR control interface* | CDR | phase_cmd | 10-100 UI (fast path) | PI controller |

> \* CDR Control Interface: Only provides parameterized configuration and monitoring for TDF domain CDR, CDR core functionality implemented by TDF domain `RxCdrTdf`, see [cdr.md](cdr.md) for details

#### 3.5.2 AGC PI Controller

```cpp
// AGC update algorithm
double agc_pi_update(double amplitude_rms) {
    double error = target_amplitude - amplitude_rms;
    
    // PI controller
    m_agc_integral += ki * error;
    m_agc_integral = clamp(m_agc_integral, integral_min, integral_max);
    
    double gain_adjust = kp * error + m_agc_integral;
    m_current_gain = clamp(m_current_gain + gain_adjust, gain_min, gain_max);
    
    return m_current_gain;
}
```

#### 3.5.3 DFE Tap Update Algorithms

**LMS Algorithm**:
```cpp
for (int i = 0; i < N_taps; i++) {
    taps[i] += mu * error * history_bits[i];
    taps[i] = clamp(taps[i], tap_min, tap_max);
}
```

**Sign-LMS Algorithm** (hardware-friendly):
```cpp
for (int i = 0; i < N_taps; i++) {
    taps[i] += mu * sign(error) * sign(history_bits[i]);
    taps[i] = clamp(taps[i], tap_min, tap_max);
}
```

**NLMS Algorithm** (normalized step size):
```cpp
double norm = epsilon;
for (int i = 0; i < N_taps; i++) {
    norm += history_bits[i] * history_bits[i];
}
for (int i = 0; i < N_taps; i++) {
    taps[i] += (mu / norm) * error * history_bits[i];
    taps[i] = clamp(taps[i], tap_min, tap_max);
}
```

#### 3.5.4 Threshold Adaptation

```cpp
// Threshold tracking algorithm
double threshold_adapt(int error_count) {
    double error_rate = (double)error_count / symbol_count;
    
    // Adjust threshold based on error rate trend
    if (error_rate > prev_error_rate) {
        // Error rate rising, reverse adjustment
        m_current_threshold -= adaptation_rate * sign(m_current_threshold);
    } else {
        // Error rate falling, continue current direction
        m_current_threshold += adaptation_rate * threshold_direction;
    }
    
    return clamp(m_current_threshold, threshold_min, threshold_max);
}
```

#### 3.5.5 Multi-Rate Scheduling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaption Module Scheduling Architecture                        │
│                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │      Fast Path      │   │      Slow Path     │  │
│  │      Every 10-100 UI          │   │      Every 1000-10000 UI     │  │
│  ├─────────────────────────┤   ├─────────────────────────┤  │
│  │ • Threshold Adaptation   │   │ • AGC PI Controller           │  │
│  │ • CDR Control Interface* (phase_cmd)│   │ • DFE Tap Update (LMS)       │  │
│  └─────────────────────────┘   └─────────────────────────┘  │
│           │                           │                       │
│           ▼                           ▼                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │               Safety and Rollback Mechanism (Safety)                  │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │ • Freeze condition: error_count > freeze_threshold              │  │
│  │ • Snapshot save: Save parameter snapshot every snapshot_interval UI     │  │
│  │ • Rollback strategy: Roll back to last valid snapshot when algorithm divergence detected       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.5.6 DE-TDF Bridging Mechanism

The DE domain Adaption module and TDF domain modules bridge through SystemC signal ports:

| Bridge Direction | Signal Type | Description |
|----------|----------|------|
| DE→TDF | `vga_gain` | Control VGA gain |
| DE→TDF | `dfe_tap1`~`dfe_tap8` | Control DFE tap coefficients |
| DE→TDF | `sampler_threshold` | Control sampling threshold |
| DE→TDF | `phase_cmd` | Phase control command |
| TDF→DE | `amplitude_rms` | Amplitude statistics feedback |
| TDF→DE | `error_count` | Error count feedback |
| TDF→DE | `phase_error` | Phase error feedback |

### 3.6 CDR and Adaption Coordination Mechanism

The CDR module (TDF domain) and Adaption CDR PI control interface (DE domain) have different responsibility positioning, supporting three usage modes.

#### 3.6.1 Responsibility Division

| Dimension | CDR Module (`RxCdrTdf`) | Adaption CDR PI |
|------|---------------------|------------------|
| Domain | TDF domain (analog signal processing) | DE domain (digital event control) |
| Function | Complete closed-loop CDR: Bang-Bang PD + PI controller + phase output | Parameterized configuration, monitoring, enhanced control of CDR |
| Detailed Documentation | [cdr.md](cdr.md) (complete technical documentation) | [adaption.md](adaption.md) (control interface description) |

#### 3.6.2 Three Usage Modes

**Mode A: Standard Mode** (recommended)
```json
{
  "cdr": {"pi": {"kp": 0.01, "ki": 1e-4}},
  "adaption": {"cdr_pi": {"enabled": false}}
}
```
- Only uses TDF domain CDR complete closed-loop
- Adaption CDR PI control interface disabled
- Applicable scenario: Most standard applications

**Mode B: Enhanced Mode**
```json
{
  "cdr": {"pi": {"kp": 0.005, "ki": 5e-5}},
  "adaption": {"cdr_pi": {"enabled": true, "kp": 0.005, "ki": 5e-5}}
}
```
- TDF domain CDR + DE domain Adaption CDR PI dual-loop coordination
- TDF domain CDR provides fast phase tracking
- DE domain provides slow stability enhancement
- Applicable scenario: High jitter tolerance requirement applications

**Mode C: Flexible Mode**
```json
{
  "cdr": {"pd_only": true},
  "adaption": {"cdr_pi": {"enabled": true, "kp": 0.01, "ki": 1e-4}}
}
```
- TDF domain CDR only performs phase detection, outputs `phase_error`
- DE domain Adaption completes PI control, outputs `phase_cmd`
- Applicable scenario: Requires deep integration with other DE domain algorithms

#### 3.6.3 Mode Selection Guide

| Application Scenario | Recommended Mode | Rationale |
|----------|----------|------|
| Standard link establishment | Mode A | Simple and reliable, TDF domain closed-loop is sufficient |
| High jitter tolerance requirement | Mode B | Dual-loop provides additional stability |
| Full DE domain scheduling | Mode C | Facilitates unified management |
| Debug/verification | Mode A/C | Select based on test objectives |

> **Important**: For complete CDR technical documentation (including Bang-Bang PD principles, PI controller design, locking process analysis, test scenarios, etc.) see [cdr.md](cdr.md)

---

## 4. Testbench Architecture

### 4.1 Testbench Design Philosophy

RX testbench requires closed-loop integrated design:

- **TX Side**: Signal source (PRBS) + Channel model
- **RX Side**: CTLE + VGA + DFE + Sampler + CDR cascade
- **Performance Evaluation**: BER statistics + Eye diagram capture + Phase error measurement

Differences from sub-module testbenches:
- Sub-module testing (e.g., ctle_tran_tb): Single-module open-loop testing
- RX top-level testing: Full-link closed-loop testing, includes feedback paths

### 4.2 Test Scenario Definitions

| Scenario | Command Line Parameter | Test Objective | Output File |
|------|----------|---------|----------|
| BASIC_PRBS | `prbs` / `0` | Basic link establishment and locking | rx_tran_prbs.csv |
| CHANNEL_SWEEP | `ch_sweep` / `1` | BER under different channel losses | rx_ber_sweep.csv |
| ADAPTION_TEST | `adapt` / `2` | Adaptive algorithm convergence | rx_adaption.csv |
| JITTER_TOLERANCE | `jtol` / `3` | System-level JTOL test | rx_jtol.csv |
| EYE_SCAN | `eye` / `4` | 2D eye diagram scan | rx_eye_2d.csv |

### 4.3 Scenario Configuration Details

#### BASIC_PRBS - Basic Link Test

- **Signal Source**: PRBS-31, 10Gbps
- **Channel**: Moderate loss (10dB @ Nyquist)
- **RX Configuration**: Default parameters
- **Simulation Time**: ≥100,000 UI
- **Verification Points**:
  - CDR lock time < 5000 UI
  - BER after lock < 1e-12
  - Phase stability < 5ps RMS

#### CHANNEL_SWEEP - Channel Loss Sweep

- **Channel Variation**: 5dB, 10dB, 15dB, 20dB @ Nyquist
- **RX Configuration**: Fixed parameters or adaptive enabled
- **Verification Points**: Plot BER vs loss curve, determine link margin

#### ADAPTION_TEST - Adaptive Convergence Test

- **Initial State**: DFE tap coefficients zero
- **Adaptive Algorithm**: LMS, step size μ=0.001
- **Monitor Signals**: Tap coefficient time-domain evolution + BER convergence curve
- **Verification Points**:
  - Convergence time < 50,000 UI
  - Steady-state BER reaches optimal value

#### ADAPTION_TEST Detailed Test Scenarios

The Adaption module supports multiple test scenarios (detailed testbench see [adaption.md](adaption.md)):

| Scenario Name | Test Objective | Key Metrics |
|--------|----------|----------|
| `BASIC_AGC` | AGC basic convergence test | Convergence time, steady-state error |
| `AGC_STEP_RESPONSE` | AGC step response test | Settling time, overshoot |
| `DFE_CONVERGENCE` | DFE tap convergence test | Convergence time, tap stability |
| `DFE_ALGORITHM_COMPARE` | LMS/Sign-LMS/NLMS comparison | Algorithm performance comparison |
| `THRESHOLD_TRACKING` | Threshold adaptation test | Tracking accuracy, response speed |
| `FREEZE_ROLLBACK` | Freeze/rollback mechanism test | Safety trigger, recovery capability |
| `MULTI_RATE_SCHEDULING` | Multi-rate scheduling test | Fast/slow path coordination |
| `FULL_ADAPTION` | Full algorithm joint test | Comprehensive convergence performance |

> **Note**: CDR-related test scenarios see [cdr.md](cdr.md), including lock test, jitter tolerance test, frequency offset tracking test, etc.

#### JITTER_TOLERANCE - System-Level Jitter Tolerance

- Difference from CDR standalone test: Includes CTLE/VGA/DFE effects
- Jitter injection position: Channel output
- Test method: Sweep jitter frequency (1kHz-100MHz), record BER

#### EYE_SCAN - Two-Dimensional Eye Diagram Scan

- **X-axis**: Phase scan (-0.5UI ~ +0.5UI)
- **Y-axis**: Threshold scan (-Vswing ~ +Vswing)
- **Per-point statistics**: BER measurement of ≥10,000 UI
- **Output**: Eye diagram heatmap, annotated with eye height/width

### 4.4 Signal Connection Topology

```
┌──────────┐   ┌────────┐   ┌──────┐   ┌─────┐   ┌─────────┐   ┌────────┐   ┌─────────┐
│ PRBS Gen │→→→│ Channel│→→→│ CTLE │→→→│ VGA │→→→│DFE Summ │→→→│Sampler │→→→│ BER Mon │
└──────────┘   └────────┘   └──────┘   └─────┘   └─────────┘   └────────┘   └─────────┘
                                                       ↑            ↓    ↓
                                                       │            │    └──→┌─────┐
                                                       │            │        │ CDR │
                                                       └────────────┴────────└─────┘
                                                      DFE Feedback     CDR Phase Feedback
```

### 4.5 Auxiliary Module Descriptions

| Module | Function | Configuration Parameters |
|------|------|---------|
| **Channel Model** | S-parameter import or analytical loss model | touchstone, attenuation_db |
| **PRBS Generator** | Supports PRBS-7/15/31, configurable jitter injection | type, jitter |
| **BER Monitor** | Real-time BER statistics, supports eye diagram capture | measure_length, eye_params |
| **Adaption Controller** | DE domain adaptive algorithm controller | agc, dfe, threshold |
| **Performance Analyzer** | Eye height/width/Q-factor analysis | ui_bins, amp_bins |

---

## 5. Simulation Results Analysis

### 5.1 Statistical Metrics Description

| Metric | Calculation Method | Significance |
|------|----------|------|
| **BER** | Error count / Total bit count | Core system reliability indicator |
| **Eye Height** | min(signal high level) - max(signal low level) | Noise margin |
| **Eye Width** | Optimal sampling phase range (UI) | Timing margin |
| **Q-factor** | √2 × erfc⁻¹(2×BER) | SNR equivalent indicator |
| **Lock Time** | Moment when CDR phase error < 5ps | Link establishment speed |

#### 5.1.2 Adaption-Specific Metrics

| Metric Category | Metric Name | Calculation Method | Significance |
|----------|--------|----------|------|
| **Convergence** | `convergence_time` | UI count when steady-state error < threshold | Algorithm convergence speed |
| | `steady_state_error` | Target - actual value at steady-state | Convergence accuracy |
| | `convergence_stability` | Steady-state error variance | Convergence stability |
| **AGC** | `agc_gain` | Current VGA gain value | Gain adjustment effect |
| | `amplitude_error` | Target - actual amplitude | AGC accuracy |
| **DFE** | `tap_coeffs[]` | Current tap coefficients | DFE equalization effect |
| | `isi_residual` | Residual ISI energy | Equalization quality |
| **Threshold** | `threshold_value` | Current decision threshold | Threshold tracking effect |
| | `threshold_drift` | Threshold drift amount | DC drift compensation |
| **Safety** | `freeze_count` | Freeze trigger count | Safety mechanism activation |
| | `rollback_count` | Rollback execution count | Recovery mechanism usage |

### 5.2 Typical Test Results Interpretation

#### BASIC_PRBS Test Results Example

**Configuration**: 10Gbps, moderate channel (10dB @ Nyquist)

**Expected Results**:
```
=== RX Performance Summary ===
CDR Lock Time:        2345 UI (234.5 ns)
BER (after lock):     0.0 (no errors in 1e7 bits)
Eye Height:           450 mV (corresponding to Q=7.2, BER=1e-12 theoretical value)
Eye Width:            0.65 UI (65 ps)
Phase Jitter (RMS):   2.1 ps
CTLE Output Swing:    300 mV (gain 1.5× input 200mV)
VGA Output Swing:     600 mV (gain 2.0× CTLE output)
DFE Tap Coeffs:       [-0.08, -0.03, 0.01] (adaptive converged values)
```

**Waveform Characteristics**:
- CTLE output: High-frequency boost obvious, edges become steep
- VGA output: Amplitude amplification, maintains differential characteristics
- DFE output: ISI significantly reduced, eye opening increased
- Sampler output: Clear digital transitions, very few errors

#### CHANNEL_SWEEP Results Interpretation

**BER vs Channel Loss Curve**:

| Loss (dB@Nyq) | BER (no DFE) | BER (with DFE) | Improvement (dB) |
|-------------|-----------|-----------|---------|
| 5 | 1e-15 | 1e-15 | 0 (sufficient margin) |
| 10 | 1e-9 | 1e-13 | 4dB |
| 15 | 1e-5 | 1e-11 | 6dB |
| 20 | 1e-3 | 1e-9 | 6dB |
| 25 | >1e-1 | 1e-6 | >5dB |

**Analysis Points**:
- DFE has significant effect in high-loss channels (>15dB)
- 20dB loss approaches system limit, requires stronger CTLE/VGA
- 25dB loss may require enabling more DFE taps (5-7 taps)

#### ADAPTION_TEST Results Interpretation

**Configuration**: 10Gbps, moderate channel, AGC+DFE+threshold adaptation all enabled

**Expected Results**:
```
=== Adaption Performance Summary ===
--- AGC Convergence ---
Convergence Time:     12,500 UI
Target Amplitude:     0.400 V
Actual Amplitude:     0.398 V
Steady-State Error:   0.002 V (0.5%)
VGA Gain (final):     3.25

--- DFE Tap Convergence ---
Algorithm:            Sign-LMS
Step Size (mu):       1e-4
Convergence Time:     35,000 UI
Tap Coeffs (final):   [-0.082, -0.031, 0.012, 0.002]
ISI Residual:         0.015 V (3.8%)

--- Threshold Adaptation ---
Initial Threshold:    0.000 V
Final Threshold:      0.008 V
Drift Compensation:   OK

--- Safety Mechanism ---
Freeze Events:        0
Rollback Events:      0
Update Count:         1,250,000
```

**Convergence Curve Characteristics**:
- **AGC convergence**: Fast adjustment phase (0-5000 UI) + steady-state fine-tuning phase (5000-12500 UI)
- **DFE convergence**: Tap coefficients gradually approach optimal values from zero, no obvious oscillation
- **Threshold tracking**: Smoothly tracks DC drift, no abrupt changes

**Abnormal Result Diagnosis**:

| Phenomenon | Possible Cause | Suggested Adjustment |
|------|----------|----------|
| AGC not converging | PI parameters inappropriate or gain range insufficient | Adjust kp/ki or extend gain_max |
| DFE tap oscillation | Step size μ too large | Reduce step size to 1e-5 |
| Frequent freezing | freeze_threshold too low | Increase threshold or check link quality |

### 5.3 Waveform Data File Format

**rx_tran_prbs.csv**:
```csv
Time(s),CTLE_out_diff(V),VGA_out_diff(V),DFE_out_diff(V),Sampler_out,CDR_phase(ps),BER
0.0e0,0.000,0.000,0.000,0,0.0,N/A
1.0e-10,0.150,0.300,0.280,1,2.5,N/A
2.0e-10,-0.145,-0.290,-0.275,0,2.3,N/A
...
1.0e-6,0.148,0.296,0.283,1,1.8,1.2e-13
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
make rx_tran_tb
cd tb
./rx_tran_tb [scenario]
```

Scenario parameters:
- `prbs` or `0` - Basic PRBS test (default)
- `ch_sweep` or `1` - Channel loss sweep
- `adapt` or `2` - Adaptive convergence test
- `jtol` or `3` - Jitter tolerance test
- `eye` or `4` - Eye diagram scan

### 6.3 Parameter Tuning Flow

**Step 1: Channel Characterization**
```bash
python scripts/analyze_channel.py channel.s4p
# Output: Loss@Nyquist, group delay, suggested CTLE zero-pole
```

**Step 2: CTLE/VGA Basic Configuration**
- Set CTLE zero-pole based on channel analysis results
- Set VGA gain preliminary to 1.5-2.0

**Step 3: DFE Initialization**
- Method A: Set tap coefficients to zero, enable adaptation
- Method B: Preset initial values based on channel impulse response

**Step 4: CDR Parameter Selection**
- Estimate Kp/Ki according to Section 8.4 formula in cdr.md
- Target bandwidth: data_rate/1000 ~ data_rate/10000

**Step 5: Run Simulation Verification**
```bash
./rx_tran_tb prbs
# Check BER, eye diagram, lock time
```

**Step 6: Iterative Optimization**
- If BER not meeting target: Increase DFE tap count or optimize CTLE parameters
- If CDR not locking: Adjust Kp/Ki or increase PAI range
- If eye diagram closed: Check saturation limits or noise configuration

### 6.4 Result Viewing

After test completion, console outputs statistical results, waveform data saved to CSV files. Use Python for visualization:

```bash
# Waveform visualization
python scripts/plot_rx_waveforms.py rx_tran_prbs.csv

# Eye diagram plotting
python scripts/plot_eye_diagram.py rx_eye_2d.csv

# BER curve
python scripts/plot_ber_sweep.py rx_ber_sweep.csv
```

---

## 7. Technical Essentials

### 7.1 Cascade Gain Allocation Principles

**Total Gain Requirement**:
```
G_total = V_sampler_min / V_channel_out
```

**Allocation Strategy**:
- **CTLE**: Provides 1.5-2.0× gain + frequency-domain shaping
- **VGA**: Provides 1.5-5.0× adjustable gain + secondary filtering
- **DFE**: Does not change average gain, only cancels ISI

**Saturation Management**:
- Each stage output limit range should match next stage input range
- Avoid intermediate stage premature saturation causing non-linear distortion
- Soft saturation (tanh) preferred over hard saturation (clamp)

### 7.2 DFE Feedback Delay Algebraic Loop Issue

**Problem Description**:
If DFE feedback delay is 0 UI, an algebraic loop forms:
```
Current output → Sampler decision → DFE feedback → Current output (circular dependency)
```

**Solution**:
- DFE `data_in` port reads **previous UI decision**
- Maintain decision history buffer in RX top-level module
- Ensure causality of signal flow

**Implementation Example**:
```cpp
// In RX top-level processing() function
int current_bit = sampler.data_out.read();
dfe.data_in.write(history_buffer);  // Use historical decision
history_buffer.push_front(current_bit);
history_buffer.pop_back();
```

### 7.3 Sampler-CDR Timing Coordination

**Phase Update Timing**:
- CDR detects phase error at nth UI
- PI controller calculates new phase offset
- Sampler applies new phase at (n+1)th UI

**Delay Impact**:
- 1 UI delay does not affect stability (loop bandwidth far below data rate)
- But increases lock time (approximately 10-20%)

### 7.4 Common-Mode Voltage Cascade Management

**Common-Mode Requirements by Stage**:
- CTLE output: 0.6V (typical)
- VGA output: 0.6V (matches CTLE)
- DFE output: 0.6V or 0.0V (depending on Sampler requirements)
- Sampler input: Must be within common-mode input range of differential pair

**Mismatch Handling**:
- If inter-stage common-mode mismatch, can insert AC coupling capacitor
- AC coupling introduces low-frequency roll-off, trade-off required

### 7.5 Adaptive Algorithm Stability

**LMS Algorithm Convergence Condition**:
```
0 < μ < 2 / λ_max
```

Where:
- μ: Step size
- λ_max: Maximum eigenvalue of input signal autocorrelation matrix

**Practical Recommendations**:
- Conservative value: μ = 0.001 ~ 0.01
- Monitor whether tap coefficients oscillate or diverge
- Use normalized LMS (NLMS) to improve stability when necessary

### 7.6 Time Step and Sampling Rate Setting

**Consistency Requirement**:
All TDF modules must be set to the same sampling rate:

```cpp
// Global configuration
double Fs = 100e9;  // 100 GHz (symbol rate 10Gbps × 10x oversampling)
double Ts = 1.0 / Fs;

// Each module set_attributes()
ctle.set_timestep(Ts);
vga.set_timestep(Ts);
dfe.set_timestep(Ts);
sampler.set_timestep(Ts);
cdr.set_timestep(Ts);
```

**Oversampling Considerations**:
- Minimum sampling rate = 2 × highest frequency component (Nyquist criterion)
- Recommended sampling rate = 5-10 × symbol rate (ensures waveform fidelity)

### 7.7 DE-TDF Bridge Timing Alignment and Delay Handling

Timing differences exist between DE domain Adaption module and TDF domain modules, attention needed:

**Timing Alignment Mechanism**:
- DE domain uses event-driven `SC_METHOD` or period-waiting `SC_THREAD`
- TDF domain uses fixed time step `processing()` function
- Parameter passing through `sc_signal` ports with interpolation synchronization

**Delay Impact**:
- DE domain parameter update to TDF domain effect has 1 time step delay
- For slow path algorithms (AGC/DFE) impact is negligible
- For fast path algorithms (threshold/CDR PI) delay compensation needed

### 7.8 Multi-Rate Scheduling Architecture Implementation Details

**Fast Path and Slow Path Separation**:

```cpp
// Fast path process - triggered every 10-100 UI
void AdaptionDe::fast_path_process() {
    // Threshold adaptation
    double new_threshold = threshold_adapt(error_count.read());
    sampler_threshold.write(new_threshold);
    
    // CDR PI control interface (if enabled)
    if (m_params.cdr_pi.enabled) {
        double cmd = cdr_pi_update(phase_error.read());
        phase_cmd.write(cmd);
    }
}

// Slow path process - triggered every 1000-10000 UI
void AdaptionDe::slow_path_process() {
    // AGC update
    double gain = agc_pi_update(amplitude_rms.read());
    vga_gain.write(gain);
    
    // DFE tap update
    dfe_lms_update(isi_metric.read());
    write_dfe_outputs();
}
```

### 7.9 AGC PI Controller Convergence and Stability

**Convergence Conditions**:
- PI parameters must satisfy stability condition: `kp + ki*T < 2` (T is update period)
- Gain range limits prevent overshoot: `gain_min ≤ gain ≤ gain_max`
- Integral term limiting prevents integral saturation

**Parameter Adjustment Guide**:
| Performance Goal | kp Adjustment | ki Adjustment |
|----------|--------|--------|
| Speed up convergence | Increase | Increase |
| Reduce overshoot | Decrease | - |
| Improve accuracy | - | Increase |
| Improve stability | Decrease | Decrease |

### 7.10 DFE Sign-LMS Algorithm Convergence and Stability

**Sign-LMS Advantages**:
- Only requires sign operation, simple hardware implementation
- Insensitive to outliers, good robustness

**Convergence Speed Comparison**:
- Standard LMS > NLMS > Sign-LMS
- Sign-LMS convergence time approximately 1.5-2× that of LMS

**Step Size Selection Guide**:
- Initial training stage: μ = 1e-3 ~ 1e-4
- Steady-state tracking stage: μ = 1e-4 ~ 1e-5

### 7.11 Threshold Adaptation Algorithm Robustness Design

**Strategies to Prevent Oscillation**:
- Dead zone design: No adjustment when error is within set range
- Low-pass filtering: Smooth error count
- Rate limiting: Single adjustment amount does not exceed set maximum

### 7.12 Safety Mechanism Trigger Conditions and Recovery Strategy

**Freeze Trigger Conditions**:
- `error_count > freeze_threshold`: Error count exceeds threshold
- `|gain - prev_gain| > delta_threshold`: Parameter change too large
- `mode == 3`: External forced freeze signal

**Rollback Strategy**:
1. Stop parameter update when anomaly detected
2. Load last valid snapshot
3. Reset integrator state
4. Wait for external recovery signal or auto-recover after timeout

### 7.13 CDR and Adaption CDR PI Coordination Mechanism

**Comparison of Three Usage Modes**:

| Mode | TDF Domain CDR | DE Domain Adaption CDR PI | Applicable Scenario |
|------|----------|---------------------|----------|
| A (Standard) | Complete closed-loop | Disabled | Most applications |
| B (Enhanced) | Fast loop | Slow loop | High stability requirement |
| C (Flexible) | PD only | PI control | DE domain integration |

**Parameter Configuration Principles**:
- Mode B requires bandwidth separation between two loops: TDF loop > 10× DE loop
- Mode C TDF domain CDR only outputs `phase_error`, does not maintain PI state

> **Important**: For complete CDR technical documentation (including Bang-Bang PD principles, PI controller design, locking process analysis, test scenarios, etc.) see [cdr.md](cdr.md)

---

## 8. Reference Information

### 8.1 Related Files

| File Type | Path | Description |
|---------|------|------|
| CTLE Header | `/include/ams/rx_ctle.h` | RxCtleTdf class declaration |
| CTLE Implementation | `/src/ams/rx_ctle.cpp` | RxCtleTdf class implementation |
| VGA Header | `/include/ams/rx_vga.h` | RxVgaTdf class declaration |
| VGA Implementation | `/src/ams/rx_vga.cpp` | RxVgaTdf class implementation |
| DFE Header | `/include/ams/rx_dfe.h` | RxDfeTdf class declaration |
| DFE Implementation | `/src/ams/rx_dfe.cpp` | RxDfeTdf class implementation |
| Sampler Header | `/include/ams/rx_sampler.h` | RxSamplerTdf class declaration |
| Sampler Implementation | `/src/ams/rx_sampler.cpp` | RxSamplerTdf class implementation |
| CDR Header | `/include/ams/rx_cdr.h` | RxCdrTdf class declaration |
| CDR Implementation | `/src/ams/rx_cdr.cpp` | RxCdrTdf class implementation |
| **Adaption Header** | `/include/ams/adaption.h` | AdaptionDe class declaration |
| **Adaption Implementation** | `/src/ams/adaption.cpp` | AdaptionDe class implementation |
| **Adaption Testbench** | `/tb/rx/adaption/adaption_tran_tb.cpp` | Adaption simulation testbench |
| **Adaption Unit Tests** | `/tests/unit/test_adaption_*.cpp` | Adaption unit test suite |
| Parameter Definitions | `/include/common/parameters.h` | RxParams/CdrParams/AdaptionParams structures |
| CTLE Documentation | `/docs/modules/ctle.md` | CTLE detailed technical documentation |
| VGA Documentation | `/docs/modules/vga.md` | VGA detailed technical documentation |
| DFE Documentation | `/docs/modules/dfesummer.md` | DFE Summer detailed technical documentation |
| Sampler Documentation | `/docs/modules/sampler.md` | Sampler detailed technical documentation |
| **CDR Documentation** | `/docs/modules/cdr.md` | CDR complete technical documentation (CDR principles and testing) |
| **Adaption Documentation** | `/docs/modules/adaption.md` | Adaption detailed technical documentation (adaptive algorithms and control interfaces) |

### 8.2 Dependencies

- SystemC 2.3.4
- SystemC-AMS 2.3.4
- C++14 standard
- GoogleTest 1.12.1 (unit testing)

### 8.3 Performance Metrics Summary

| Metric | Typical Value | Description |
|------|-------|------|
| Maximum Data Rate | 56 Gbps | Depends on channel and process |
| BER Target | < 1e-12 | With FEC can reach 1e-15 |
| Lock Time | 1-5 μs | CDR convergence time |
| Phase Jitter | < 5 ps RMS | CDR jitter after locking |
| CTLE Gain Range | 1.0-3.0 | Configurable |
| VGA Gain Range | 1.0-10.0 | Dynamically adjusted with AGC |
| DFE Tap Count | 1-8 | Typically 3-5 taps |
| Eye Height Target | > 200 mV | At Sampler input |
| Eye Width Target | > 0.5 UI | Timing margin requirement |

---

**Document Version**: v1.0  
**Last Updated**: 2026-01-27  
**Author**: Yizhe Liu
