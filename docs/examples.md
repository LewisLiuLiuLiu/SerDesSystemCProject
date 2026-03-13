# Eye Analyzer Examples

This document provides comprehensive usage examples for the Eye Analyzer toolkit.

## Table of Contents

- [Basic Usage](#basic-usage)
  - [Statistical Analysis (PAM4)](#statistical-analysis-pam4)
  - [Empirical Analysis (NRZ)](#empirical-analysis-nrz)
- [Advanced Features](#advanced-features)
  - [Jitter Tolerance Testing](#jitter-tolerance-testing)
  - [BER Contour Analysis](#ber-contour-analysis)
  - [Multi-Eye Jitter Analysis (PAM4)](#multi-eye-jitter-analysis-pam4)
- [Visualization](#visualization)
  - [Custom Eye Diagram Plots](#custom-eye-diagram-plots)
  - [Bathtub Curves](#bathtub-curves)
  - [JTOL Curves](#jtol-curves)
- [Working with Modulation Formats](#working-with-modulation-formats)
- [Report Generation](#report-generation)

---

## Basic Usage

### Statistical Analysis (PAM4)

Statistical eye analysis is used for pre-simulation channel characterization.

```python
import numpy as np
from eye_analyzer import EyeAnalyzer

# Load or create pulse response
# Pulse response represents the channel's impulse response
pulse_response = np.loadtxt('channel_pulse_response.csv')

# Create PAM4 analyzer
analyzer = EyeAnalyzer(
    ui=2.5e-11,          # 40 Gbps (25 ps UI)
    modulation='pam4',
    mode='statistical',
    target_ber=1e-12,    # Target BER for TJ calculation
    samples_per_symbol=16,
    ui_bins=128,
    amp_bins=256
)

# Analyze with noise and jitter injection
result = analyzer.analyze(
    pulse_response,
    noise_sigma=0.01,    # 10 mV RMS noise
    jitter_dj=0.05,      # 0.05 UI DJ
    jitter_rj=0.02       # 0.02 UI RJ (1 sigma)
)

# Access eye metrics
metrics = result['eye_metrics']
print(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
print(f"Eye Width: {metrics['eye_width']:.3f} UI")
print(f"Eye Area: {metrics['eye_area']*1000:.2f} mV·UI")

# PAM4-specific: per-eye metrics
if 'eye_heights_per_eye' in metrics:
    for i, (h, w) in enumerate(zip(
        metrics['eye_heights_per_eye'],
        metrics['eye_widths_per_eye']
    )):
        eye_names = ['Lower', 'Middle', 'Upper']
        print(f"{eye_names[i]} Eye - Height: {h*1000:.2f} mV, Width: {w:.3f} UI")

# Access jitter results
jitter = result['jitter']
print(f"RJ: {jitter['rj']*1000:.2f} ps")
print(f"DJ: {jitter['dj']*1000:.2f} ps")
```

### Empirical Analysis (NRZ)

Empirical eye analysis is used for post-simulation or measured waveforms.

```python
import numpy as np
from eye_analyzer import EyeAnalyzer

# Load waveform data
# Format: time (seconds), voltage (volts)
data = np.loadtxt('waveform.csv', delimiter=',')
time_array = data[:, 0]
value_array = data[:, 1]

# Create NRZ analyzer
analyzer = EyeAnalyzer(
    ui=2.5e-11,          # 10 Gbps (100 ps UI)
    modulation='nrz',
    mode='empirical',
    ui_bins=128,
    amp_bins=128
)

# Analyze waveform
result = analyzer.analyze((time_array, value_array))

# Access results
metrics = result['eye_metrics']
print(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
print(f"Eye Width: {metrics['eye_width']:.3f} UI")

# Visualize
fig = analyzer.plot_eye(title='NRZ Eye Diagram')
fig.savefig('eye_diagram.png', dpi=300)
```

---

## Advanced Features

### Jitter Tolerance Testing

Test receiver jitter tolerance against industry standards.

```python
import numpy as np
from eye_analyzer import EyeAnalyzer

# Load multiple pulse responses with different SJ
# Each pulse response corresponds to a different SJ frequency/amplitude
sj_frequencies = [1e6, 5e6, 10e6, 50e6, 100e6]  # Hz
pulse_responses = []

for freq in sj_frequencies:
    # Load pulse response with SJ at this frequency
    pr = np.loadtxt(f'pulse_response_sj_{freq/1e6:.0f}mhz.csv')
    pulse_responses.append(pr)

# Create analyzer
analyzer = EyeAnalyzer(
    ui=2.5e-11,
    modulation='pam4',
    mode='statistical'
)

# Run JTOL test with IEEE 802.3ck template
jtol_result = analyzer.analyze_jtol(
    pulse_responses=pulse_responses,
    sj_frequencies=sj_frequencies,
    template='ieee_802_3ck'  # Options: 'ieee_802_3ck', 'oif_cei_56g', 'jedec_ddr5', 'pcie_gen6'
)

# Check results
print(f"Overall Pass: {jtol_result['overall_pass']}")
print(f"Margins: {jtol_result['margins']}")

# Print per-frequency results
for freq, pf in zip(
    jtol_result['sj_frequencies'],
    jtol_result['pass_fail']
):
    status = "PASS" if pf else "FAIL"
    print(f"  {freq/1e6:.1f} MHz: {status}")

# Plot JTOL curve
fig = analyzer.plot_jtol(jtol_result, title='Jitter Tolerance Test')
fig.savefig('jtol_curve.png', dpi=300)
```

### BER Contour Analysis

Generate and analyze BER contours for eye mask compliance.

```python
import numpy as np
from eye_analyzer import EyeAnalyzer, BERContour

# Statistical analysis
analyzer = EyeAnalyzer(
    ui=2.5e-11,
    modulation='pam4',
    mode='statistical'
)

pulse_response = np.loadtxt('pulse_response.csv')
result = analyzer.analyze(pulse_response, noise_sigma=0.005)

# Access BER contour
ber_contour = result['ber_contour']

# Create BER contour analyzer
contour_analyzer = BERContour(
    levels=[1e-6, 1e-9, 1e-12, 1e-15]  # BER levels to plot
)

# Analyze eye opening at specific BER
opening = contour_analyzer.eye_opening(ber_contour, target_ber=1e-12)
print(f"Eye opening at 1e-12: {opening['width']:.3f} UI x {opening['height']*1000:.2f} mV")

# Plot with BER contours
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
analyzer.plot_eye(ax=ax, title='Eye Diagram with BER Contours')

# Overlay BER contours
X, Y = np.meshgrid(result['xedges'][:-1], result['yedges'][:-1])
ax.contour(X, Y, ber_contour.T, levels=[1e-12], colors='white', linewidths=2)

fig.savefig('eye_with_contours.png', dpi=300)
```

### Multi-Eye Jitter Analysis (PAM4)

Analyze jitter separately for each eye in PAM4 signals.

```python
import numpy as np
from eye_analyzer import JitterAnalyzer

# Load waveform
data = np.loadtxt('pam4_waveform.csv', delimiter=',')
time = data[:, 0]
signal = data[:, 1]

# Create PAM4 jitter analyzer
analyzer = JitterAnalyzer(
    modulation='pam4',
    signal_amplitude=1.0  # Normalized amplitude
)

# Analyze jitter
results = analyzer.analyze(signal, time, ber=1e-12)

# Results is a list of per-eye results
eye_names = ['Lower Eye', 'Middle Eye', 'Upper Eye']
for eye_result in results:
    eye_id = eye_result['eye_id']
    name = eye_result['eye_name']
    rj = eye_result['rj'] * 1e12  # Convert to ps
    dj = eye_result['dj'] * 1e12
    tj = eye_result['tj'] * 1e12
    
    print(f"{name} (ID: {eye_id}):")
    print(f"  RJ: {rj:.2f} ps")
    print(f"  DJ: {dj:.2f} ps")
    print(f"  TJ@1e-12: {tj:.2f} ps")
```

---

## Visualization

### Custom Eye Diagram Plots

Create customized eye diagram visualizations.

```python
import numpy as np
import matplotlib.pyplot as plt
from eye_analyzer import EyeAnalyzer, plot_eye_diagram

# Analyze
analyzer = EyeAnalyzer(ui=2.5e-11, modulation='nrz', mode='empirical')
time = np.linspace(0, 1e-6, 10000)
waveform = np.random.choice([-0.4, 0.4], size=10000)
result = analyzer.analyze((time, waveform))

# Method 1: Using analyzer method
fig1 = analyzer.plot_eye(
    cmap='hot',
    title='NRZ Eye (Hot Colormap)'
)
fig1.savefig('eye_hot.png', dpi=300)

# Method 2: Using standalone function with custom styling
fig2, ax = plt.subplots(figsize=(12, 8))
plot_eye_diagram(
    eye_matrix=result['eye_matrix'],
    xedges=result['xedges'],
    yedges=result['yedges'],
    ax=ax,
    cmap='viridis',
    title='NRZ Eye (Viridis)'
)

# Add custom annotations
ax.axvline(x=0.5, color='red', linestyle='--', label='Center')
ax.legend()
fig2.savefig('eye_custom.png', dpi=300)

# Method 3: Multi-plot figure
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Eye diagram
analyzer.plot_eye(ax=axes[0], title='Eye Diagram')

# Cross-section at center
x_centers = (result['xedges'][:-1] + result['xedges'][1:]) / 2
center_idx = len(x_centers) // 2
cross_section = result['eye_matrix'][center_idx, :]
y_centers = (result['yedges'][:-1] + result['yedges'][1:]) / 2

axes[1].plot(cross_section, y_centers)
axes[1].set_xlabel('Probability Density')
axes[1].set_ylabel('Voltage [V]')
axes[1].set_title('Cross-section at Center')
axes[1].grid(True)

fig3.tight_layout()
fig3.savefig('eye_analysis.png', dpi=300)
```

### Bathtub Curves

Generate and plot bathtub curves.

```python
import numpy as np
import matplotlib.pyplot as plt
from eye_analyzer import EyeAnalyzer, BathtubCurve

# Analyze
analyzer = EyeAnalyzer(ui=2.5e-11, modulation='nrz', mode='statistical')
pulse_response = np.loadtxt('pulse_response.csv')
result = analyzer.analyze(pulse_response, noise_sigma=0.01)

# Time bathtub (horizontal slice)
bathtub_time = BathtubCurve(direction='time')
t_phases, t_ber = bathtub_time.generate(
    result['eye_matrix'],
    threshold=0.0  # At zero crossing
)

# Voltage bathtub (vertical slice at center phase)
bathtub_voltage = BathtubCurve(direction='voltage')
v_voltages, v_ber = bathtub_voltage.generate(
    result['eye_matrix'],
    phase=0.5  # At center UI
)

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time bathtub
axes[0].semilogy(t_phases, t_ber, 'b-', linewidth=2)
axes[0].axhline(y=1e-12, color='r', linestyle='--', label='Target BER')
axes[0].set_xlabel('Phase [UI]')
axes[0].set_ylabel('BER')
axes[0].set_title('Time Bathtub Curve')
axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].legend()

# Voltage bathtub
axes[1].semilogy(v_voltages, v_ber, 'g-', linewidth=2)
axes[1].axhline(y=1e-12, color='r', linestyle='--', label='Target BER')
axes[1].set_xlabel('Voltage [V]')
axes[1].set_ylabel('BER')
axes[1].set_title('Voltage Bathtub Curve')
axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
axes[1].legend()

fig.tight_layout()
fig.savefig('bathtub_curves.png', dpi=300)
```

### JTOL Curves

Plot jitter tolerance curves with industry standard templates.

```python
import numpy as np
import matplotlib.pyplot as plt
from eye_analyzer import EyeAnalyzer, JTolTemplate

# Compare multiple standards
standards = ['ieee_802_3ck', 'oif_cei_56g', 'jedec_ddr5']
colors = ['blue', 'green', 'red']

fig, ax = plt.subplots(figsize=(10, 6))

for std, color in zip(standards, colors):
    template = JTolTemplate.from_standard(std)
    ax.loglog(
        template.frequencies,
        template.amplitudes,
        color=color,
        linewidth=2,
        label=std.upper().replace('_', '-')
    )

ax.set_xlabel('SJ Frequency [Hz]')
ax.set_ylabel('SJ Amplitude [UI]')
ax.set_title('JTOL Standards Comparison')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

fig.savefig('jtol_standards.png', dpi=300)
```

---

## Working with Modulation Formats

Direct usage of modulation format classes.

```python
from eye_analyzer import NRZ, PAM4, create_modulation

# Create modulation instances
nrz = NRZ()
pam4 = PAM4()

# Query properties
print(f"NRZ levels: {nrz.get_levels()}")           # [-1, 1]
print(f"PAM4 thresholds: {pam4.get_thresholds()}")  # [-2, 0, 2]
print(f"PAM4 eye centers: {pam4.get_eye_centers()}")  # [-2, 0, 2]
print(f"PAM4 number of eyes: {pam4.num_eyes}")      # 3

# Factory function
mod = create_modulation('pam4')
print(f"Created: {mod.name}")

# Iterate over PAM4 levels and thresholds
levels = pam4.get_levels()
thresholds = pam4.get_thresholds()
print("PAM4 level transitions:")
for i in range(len(levels) - 1):
    print(f"  {levels[i]:+.0f} -> {levels[i+1]:+.0f} (threshold: {thresholds[i]:+.0f})")
```

---

## Report Generation

Generate comprehensive analysis reports.

```python
from eye_analyzer import EyeAnalyzer

# Analyze
analyzer = EyeAnalyzer(ui=2.5e-11, modulation='pam4', mode='statistical')
pulse_response = np.loadtxt('pulse_response.csv')
result = analyzer.analyze(pulse_response, noise_sigma=0.01)

# Text report
text_report = analyzer.create_report(format='text')
print(text_report)

# Markdown report
md_report = analyzer.create_report(format='markdown')
with open('report.md', 'w') as f:
    f.write(md_report)

# HTML report
html_report = analyzer.create_report(format='html')
with open('report.html', 'w') as f:
    f.write(html_report)

# Save to file directly
analyzer.create_report(
    output_file='analysis_report.txt',
    format='text'
)

# Custom report with plots
analyzer.create_report(
    output_file='full_report.pdf',
    format='pdf',
    include_plots=True
)
```

### Sample Text Report Output

```
============================================================
Eye Diagram Analysis Report
============================================================

Mode: statistical
Modulation: pam4
UI: 25.00 ps
Target BER: 1e-12

Eye Metrics:
----------------------------------------
  Eye Height: 120.50 mV
  Eye Width: 0.750 UI
  Eye Area: 90.38 mV*UI

Jitter Analysis:
----------------------------------------
  DJ: 0.050 UI
  RJ: 0.020 UI

============================================================
```

---

## Complete Workflow Example

A complete workflow from channel characterization to compliance testing.

```python
import numpy as np
from eye_analyzer import (
    EyeAnalyzer, JitterAnalyzer,
    BERContour, JTolTemplate
)

# Step 1: Channel Characterization
print("Step 1: Channel Characterization")
pulse_response = np.loadtxt('channel_pulse_response.csv')

analyzer = EyeAnalyzer(
    ui=2.5e-11,  # 40 Gbps
    modulation='pam4',
    mode='statistical',
    target_ber=1e-12
)

result = analyzer.analyze(
    pulse_response,
    noise_sigma=0.008,
    jitter_dj=0.04,
    jitter_rj=0.015
)

print(f"Eye Height: {result['eye_metrics']['eye_height']*1000:.2f} mV")
print(f"Eye Width: {result['eye_metrics']['eye_width']:.3f} UI")

# Step 2: BER Contour Analysis
print("\nStep 2: BER Contour Analysis")
contour = BERContour(levels=[1e-6, 1e-9, 1e-12])
ber_data = contour.generate(result['eye_matrix'])

# Step 3: Jitter Tolerance Test
print("\nStep 3: Jitter Tolerance Test")
sj_freqs = np.logspace(6, 8, 20)  # 1 MHz to 100 MHz
sj_pulse_responses = [
    np.loadtxt(f'pulse_sj_{f/1e6:.0f}mhz.csv')
    for f in sj_freqs
]

jtol_result = analyzer.analyze_jtol(
    pulse_responses=sj_pulse_responses,
    sj_frequencies=sj_freqs,
    template='ieee_802_3ck'
)

print(f"JTOL Pass: {jtol_result['overall_pass']}")

# Step 4: Generate Report
print("\nStep 4: Generate Report")
analyzer.create_report(
    output_file='channel_compliance_report.txt',
    format='text'
)

# Step 5: Visualization
print("\nStep 5: Generate Plots")
fig = analyzer.plot_eye(title='PAM4 Eye Diagram')
fig.savefig('eye_diagram.png', dpi=300)

fig = analyzer.plot_jtol(jtol_result)
fig.savefig('jtol_curve.png', dpi=300)

print("\nAnalysis complete!")
```
