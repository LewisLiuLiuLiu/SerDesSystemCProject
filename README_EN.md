# SerDes SystemC-AMS Behavioral Modeling Platform

[![C++](https://img.shields.io/badge/C++-11-blue.svg)](https://isocpp.org/)
[![SystemC-AMS](https://img.shields.io/badge/SystemC--AMS-2.3.4-orange.svg)](https://accellera.org/community/systemc-ams)
[![CMake](https://img.shields.io/badge/CMake-3.15+-green.svg)](https://cmake.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)

🌐 **Languages**: [中文](README.md) | [English](README_EN.md)

A high-speed serial link (SerDes) behavioral modeling and simulation platform based on **SystemC-AMS**, supporting complete signal chain simulation from TX → Channel → RX, including PRBS generation, jitter injection, equalization, clock recovery, and Python eye diagram analysis.

---

## 📋 Features

### TX Transmitter
- **FFE (Feed-Forward Equalization)**: FIR filter with configurable tap coefficients
- **Mux (Multiplexer)**: Lane selection and channel multiplexing
- **Driver**: Supports nonlinear saturation, bandwidth limiting, differential output

### Channel
- **S-Parameter Model**: Based on Touchstone (.sNp) files
- **Vector Fitting**: Offline rational function fitting ensuring causal stability
- **Crosstalk & Bidirectional Transmission**: Supports multi-port coupling and reflection

### RX Receiver
- **CTLE (Continuous-Time Linear Equalizer)**: Configurable zero-pole locations, supports noise/offset/saturation modeling
- **VGA (Variable Gain Amplifier)**: Programmable gain with AGC support
- **Sampler**: Phase-configurable, supports threshold/hysteresis
- **DFE (Decision Feedback Equalization)**: FIR structure with LMS/Sign-LMS adaptation
- **CDR (Clock and Data Recovery)**: PI control loop with Bang-Bang/linear phase detection

### Clock & Waveform
- **Clock Generation**: Ideal clock / PLL / ADPLL options
- **Wave Generation**: PRBS7/9/15/23/31 and custom polynomials with RJ/SJ/DJ jitter injection

### Python EyeAnalyzer
- Eye diagram generation and metrics (eye height, eye width, opening area)
- Jitter decomposition (RJ/DJ/TJ)
- PSD/PDF analysis and visualization

---

## 🏗️ Project Structure

```
serdes/
├── include/                    # Header files
│   ├── ams/                    # AMS modules (TDF domain)
│   │   ├── tx_*.h              # TX: FFE, Mux, Driver
│   │   ├── channel_sparam.h    # Channel S-parameter model
│   │   ├── rx_ctle.h           # RX: CTLE, VGA, Sampler
│   │   ├── rx_dfe*.h           # DFE Summer, DAC
│   │   ├── rx_cdr.h            # CDR (PI controller)
│   │   ├── wave_generation.h   # PRBS/waveform generation
│   │   └── clock_generation.h  # Clock generation
│   ├── common/                 # Common types, parameters, constants
│   └── de/                     # DE domain modules
│       └── config_loader.h     # JSON/YAML config loader
├── src/                        # Implementation files
│   ├── ams/                    # AMS module implementations
│   └── de/                     # DE module implementations
├── tb/                         # Testbenches
│   ├── top/                    # Full-link simulation
│   ├── rx/, tx/, periphery/    # Subsystem tests
├── tests/                      # Unit tests (GoogleTest)
│   └── unit/                   # 139+ test cases
├── eye_analyzer/               # Python eye analysis package
│   ├── core.py                 # Core analysis engine
│   ├── jitter.py               # Jitter decomposition
│   └── visualization.py        # Visualization
├── scripts/                    # Script tools
│   ├── run_*.sh                # Test run scripts
│   ├── analyze_serdes_link.py  # Link result analysis
│   └── vector_fitting.py       # S-parameter vector fitting
├── config/                     # Configuration templates
│   ├── default.json            # Default configuration
│   └── default.yaml
└── docs/modules/               # Module documentation
```

---

## 🚀 Quick Start

### Requirements

| Component | Version |
|-----------|---------|
| C++ Standard | C++14 |
| SystemC | 2.3.4 |
| SystemC-AMS | 2.3.4 |
| CMake | ≥3.15 |
| Python | ≥3.8 |

Dependencies: `numpy`, `scipy`, `matplotlib`

### Prerequisites

#### 1. Install SystemC and SystemC-AMS

```bash
# Download and compile SystemC 2.3.4
wget https://www.accellera.org/images/downloads/standards/systemc/systemc-2.3.4.tar.gz
tar xzf systemc-2.3.4.tar.gz
cd systemc-2.3.4
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j4
make install

# Download and compile SystemC-AMS 2.3.4
cd ../..
wget https://www.accellera.org/images/downloads/standards/systemc/systemc-ams-2.3.4.tar.gz
tar xzf systemc-ams-2.3.4.tar.gz
cd systemc-ams-2.3.4
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DCMAKE_PREFIX_PATH=/path/to/install
make -j4
make install
```

#### 2. Set Environment Variables (Recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export SYSTEMC_HOME=/path/to/systemc-2.3.4
export SYSTEMC_AMS_HOME=/path/to/systemc-ams-2.3.4

# Or temporarily
export SYSTEMC_HOME=~/systemc-2.3.4
export SYSTEMC_AMS_HOME=~/systemc-ams-2.3.4
```

> **Note**: The project supports the following methods for specifying SystemC paths (in order of priority):
> 1. CMake options: `-DSYSTEMC_HOME=path -DSYSTEMC_AMS_HOME=path`
> 2. Environment variables: `SYSTEMC_HOME`, `SYSTEMC_AMS_HOME`
> 3. Auto-detection of standard installation paths

### Build Project

```bash
# 1. Clone repository
git clone https://github.com/LewisLiuLiuLiu/SerDesSystemCProject.git
cd serdes

# 2. Create build directory
mkdir build && cd build

# 3. Configure (auto-detect SystemC paths)
cmake ..

# Or manually specify paths (if not using environment variables)
# cmake -DSYSTEMC_HOME=/path/to/systemc -DSYSTEMC_AMS_HOME=/path/to/systemc-ams ..

# 4. Compile
make -j4

# 5. Run tests (optional)
ctest
```

### Run Full-Link Simulation

```bash
# Run using script
./scripts/run_serdes_link.sh basic yes

# Or run manually
cd build
./tb/serdes_link_tb basic

# Python post-processing analysis
cd ..
python3 scripts/analyze_serdes_link.py basic
python3 scripts/plot_dfe_taps.py build/serdes_link_basic_dfe_taps.csv
```

### Run Unit Tests

```bash
# Run all tests
./scripts/run_unit_tests.sh

# Or run specific module tests
./scripts/run_cdr_tests.sh
./scripts/run_adaption_tests.sh
```

---

## 📊 Usage Examples

### Configure Simulation Parameters

Edit `config/default.json`:

```json
{
  "global": {
    "Fs": 80e9,
    "UI": 2.5e-11,
    "duration": 1e-6,
    "seed": 12345
  },
  "wave": {
    "type": "PRBS31",
    "jitter": {
      "RJ_sigma": 5e-13,
      "SJ_freq": [5e6],
      "SJ_pp": [2e-12]
    }
  },
  "tx": {
    "ffe_taps": [0.2, 0.6, 0.2],
    "driver": { "swing": 0.8, "bw": 20e9 }
  },
  "rx": {
    "ctle": {
      "zeros": [2e9],
      "poles": [30e9],
      "dc_gain": 1.5
    },
    "dfe": { "taps": [-0.05, -0.02, 0.01] }
  },
  "cdr": {
    "pi": { "kp": 0.01, "ki": 1e-4 }
  }
}
```

### Python Eye Diagram Analysis

```python
from eye_analyzer import EyeAnalyzer
import numpy as np

# Initialize analyzer
analyzer = EyeAnalyzer(
    ui=2.5e-11,      # 10Gbps
    ui_bins=128,
    amp_bins=128,
    jitter_method='dual-dirac'
)

# Load waveform and analyze
time, voltage = analyzer.load_waveform('waveform.csv')
metrics = analyzer.analyze(time, voltage)

# Output results
print(f"Eye Height: {metrics['eye_height']:.3f} V")
print(f"Eye Width: {metrics['eye_width']:.3f} UI")
print(f"TJ @ 1e-12: {metrics['tj_at_ber']:.3e} s")
```

---

## 📚 Documentation Index

### AMS Module Documentation

| Module | Document |
|--------|----------|
| **TX** | [TX System](docs/en/modules/tx.md) |
| └ FFE | [FFE](docs/en/modules/ffe.md) |
| └ Mux | [Mux](docs/en/modules/mux.md) |
| └ Driver | [Driver](docs/en/modules/driver.md) |
| **Channel** | [Channel S-Parameter](docs/en/modules/channel.md) |
| **RX** | [RX System](docs/en/modules/rx.md) |
| └ CTLE | [CTLE](docs/en/modules/ctle.md) |
| └ VGA | [VGA](docs/en/modules/vga.md) |
| └ Sampler | [Sampler](docs/en/modules/sampler.md) |
| └ DFE Summer | [DFE Summer](docs/en/modules/dfesummer.md) |
| └ CDR | [CDR](docs/en/modules/cdr.md) |
| **Periphery** | WaveGen / [ClockGen](docs/en/modules/clkGen.md) |
| **Adaption** | [Adaption](docs/en/modules/adaption.md) |

### Python Components

| Component | Document |
|-----------|----------|
| EyeAnalyzer | [EyeAnalyzer](docs/en/modules/EyeAnalyzer.md) |

---

## 🧪 Test Coverage

The project includes **139+** unit tests covering:

| Module | Test Count | Test Content |
|--------|------------|--------------|
| Adaption | 18 | AGC, DFE LMS, CDR PI, threshold adaptation |
| CDR | 20 | PI controller, PAI, edge detection, pattern recognition |
| ClockGen | 18 | Ideal/PLL/ADPLL clock, frequency/phase tests |
| FFE | 10 | Tap coefficients, convolution, pre/de-emphasis |
| Sampler | 16 | Decision, hysteresis, noise, offset |
| TX Driver | 8 | DC gain, saturation, bandwidth, PSRR |
| WaveGen | 21 | PRBS patterns, jitter, pulses, stability |
| DFE | 3 | Tap feedback, history update |
| Channel | 3 | S-parameter, VF/IR consistency |
| Top Level | 13 | TX/RX integration tests |

---

## 🔧 Technical Details

### Modeling Domains

- **TDF (Timed Data Flow)**: Primary modeling domain for analog/mixed-signal modules
- **DE (Discrete Event)**: Control/algorithm modules, bridged to AMS via `sca_de::sca_in/out`

### Key Design Patterns

```cpp
// Standard TDF module structure
class RxCtleTdf : public sca_tdf::sca_module {
public:
    sca_tdf::sca_in<double> in_p, in_n;
    sca_tdf::sca_out<double> out_p, out_n;
    
    void set_attributes() override;
    void initialize() override;
    void processing() override;
};
```

### Transfer Function Implementation

CTLE/VGA uses zero-pole configuration, implemented via `sca_tdf::sca_ltf_nd`:

```cpp
// H(s) = dc_gain * prod(1 + s/wz_i) / prod(1 + s/wp_j)
sca_util::sca_vector<double> num, den;
build_transfer_function(zeros, poles, dc_gain, num, den);
double output = m_ltf(m_num, m_den, input);
```

---

## 📄 License

[LICENSE](LICENSE)

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

## 📧 Contact

For questions or suggestions, please use GitHub Issues.
