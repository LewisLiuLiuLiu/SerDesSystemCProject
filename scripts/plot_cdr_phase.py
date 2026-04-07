#!/usr/bin/env python3
"""
CDR Phase Convergence and Tracking Plot

Plots CDR phase evolution over time to show:
1. Lock-in process (initial convergence)
2. Phase tracking during steady state
3. Phase error statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os

# ============================================================================
# Load data
# ============================================================================
if len(sys.argv) < 2:
    print("Usage: python3 plot_cdr_phase.py <cdr_phase.csv>")
    print("Example: python3 plot_cdr_phase.py build/nrz_10g_cdr_phase.csv")
    sys.exit(1)

csv_file = sys.argv[1]

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found!")
    sys.exit(1)

print("=" * 60)
print("Loading CDR phase data...")
df = pd.read_csv(csv_file)

time_ns = df['time_s'].values * 1e9  # Convert to nanoseconds
phase_rad = df['phase'].values

# Convert phase to picoseconds for better visualization
phase_ps = phase_rad * 1e12

print(f"Loaded {len(df)} samples")
print(f"Time range: {time_ns[0]:.2f} ns to {time_ns[-1]:.2f} ns")
print(f"Phase range: {phase_ps.min():.3f} ps to {phase_ps.max():.3f} ps")

# ============================================================================
# Calculate statistics
# ============================================================================
# Split into transient and steady state
total_time_ns = time_ns[-1] - time_ns[0]
transient_end_ns = total_time_ns * 0.1  # First 10% is transient

transient_mask = time_ns <= transient_end_ns
steady_mask = ~transient_mask

if steady_mask.any():
    phase_steady = phase_ps[steady_mask]
    phase_mean = np.mean(phase_steady)
    phase_std = np.std(phase_steady)
    phase_pp = np.max(phase_steady) - np.min(phase_steady)
    
    print(f"\n{'='*60}")
    print("CDR Phase Statistics:")
    print(f"{'='*60}")
    print(f"Transient period: 0 - {transient_end_ns:.2f} ns")
    print(f"Steady state period: {transient_end_ns:.2f} - {total_time_ns:.2f} ns")
    print(f"\nSteady State Phase:")
    print(f"  Mean: {phase_mean:.3f} ps")
    print(f"  Std:  {phase_std:.3f} ps")
    print(f"  P-P:  {phase_pp:.3f} ps")
else:
    phase_mean = np.mean(phase_ps)
    phase_std = np.std(phase_ps)
    phase_pp = np.max(phase_ps) - np.min(phase_ps)

# ============================================================================
# Create figure
# ============================================================================
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Full phase evolution (picoseconds)
# ============================================================================
ax1 = plt.subplot(3, 2, 1)
ax1.plot(time_ns, phase_ps, 'b-', linewidth=0.5, alpha=0.7)
ax1.axvline(x=transient_end_ns, color='r', linestyle='--', alpha=0.5, label='Transient end')
ax1.axhline(y=phase_mean, color='g', linestyle='--', alpha=0.5, label=f'Mean: {phase_mean:.3f} ps')
ax1.set_xlabel('Time [ns]', fontsize=10)
ax1.set_ylabel('Phase [ps]', fontsize=10)
ax1.set_title('CDR Phase Evolution (Full Simulation)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Full phase evolution (raw)
# ============================================================================
ax2 = plt.subplot(3, 2, 2)
ax2.plot(time_ns, phase_ps, 'b-', linewidth=0.5, alpha=0.7)
ax2.axvline(x=transient_end_ns, color='r', linestyle='--', alpha=0.5, label='Transient end')
ax2.set_xlabel('Time [ns]', fontsize=10)
ax2.set_ylabel('Phase [ps]', fontsize=10)
ax2.set_title('CDR Phase Evolution (Raw)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Transient phase (first 10%)
# ============================================================================
ax3 = plt.subplot(3, 2, 3)
transient_time = time_ns[transient_mask]
transient_phase = phase_ps[transient_mask]
ax3.plot(transient_time, transient_phase, 'b-', linewidth=1.0)
ax3.set_xlabel('Time [ns]', fontsize=10)
ax3.set_ylabel('Phase [ps]', fontsize=10)
ax3.set_title(f'Transient Phase (0 - {transient_end_ns:.2f} ns)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ============================================================================
# Plot 4: Steady state phase
# ============================================================================
ax4 = plt.subplot(3, 2, 4)
steady_time = time_ns[steady_mask]
steady_phase = phase_ps[steady_mask]
ax4.plot(steady_time, steady_phase, 'b-', linewidth=0.5, alpha=0.7)
ax4.axhline(y=phase_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {phase_mean:.3f} ps')
ax4.axhline(y=phase_mean + phase_std, color='g', linestyle=':', alpha=0.5, label=f'±1σ')
ax4.axhline(y=phase_mean - phase_std, color='g', linestyle=':', alpha=0.5)
ax4.set_xlabel('Time [ns]', fontsize=10)
ax4.set_ylabel('Phase [ps]', fontsize=10)
ax4.set_title(f'Steady State Phase ({transient_end_ns:.2f} - {total_time_ns:.2f} ns)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ============================================================================
# Plot 5: Phase histogram
# ============================================================================
ax5 = plt.subplot(3, 2, 5)
ax5.hist(phase_ps[steady_mask], bins=100, alpha=0.7, edgecolor='black')
ax5.axvline(x=phase_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {phase_mean:.3f} ps')
ax5.axvline(x=phase_mean + phase_std, color='g', linestyle=':', linewidth=1.5, label=f'σ={phase_std:.3f} ps')
ax5.axvline(x=phase_mean - phase_std, color='g', linestyle=':', linewidth=1.5)
ax5.set_xlabel('Phase [ps]', fontsize=10)
ax5.set_ylabel('Count', fontsize=10)
ax5.set_title('Steady State Phase Distribution', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 6: Phase spectrum (FFT)
# ============================================================================
ax6 = plt.subplot(3, 2, 6)
if steady_mask.any():
    phase_steady_arr = phase_ps[steady_mask]
    dt = np.mean(np.diff(time_ns[steady_mask])) * 1e-9  # Convert to seconds
    fs = 1.0 / dt  # Sampling frequency
    
    # Remove DC component
    phase_steady_detrend = phase_steady_arr - np.mean(phase_steady_arr)
    
    # Calculate FFT
    N = len(phase_steady_detrend)
    fft_vals = np.fft.rfft(phase_steady_detrend)
    fft_freq = np.fft.rfftfreq(N, dt)
    fft_magnitude = np.abs(fft_vals) / N
    
    # Plot only up to 10 GHz (or half the sampling rate)
    max_freq = min(10e9, fs / 2)
    freq_mask = fft_freq <= max_freq
    
    ax6.semilogy(fft_freq[freq_mask] / 1e9, fft_magnitude[freq_mask], 'b-', linewidth=1.0)
    ax6.set_xlabel('Frequency [GHz]', fontsize=10)
    ax6.set_ylabel('Magnitude', fontsize=10)
    ax6.set_title('Phase Noise Spectrum', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')

plt.suptitle('CDR Phase Analysis (10 Gbps NRZ)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# ============================================================================
# Save and display
# ============================================================================
output_file = 'output_eye/cdr_phase_analysis.png'
os.makedirs('output_eye', exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nCDR phase plot saved to: {output_file}")

plt.show()
