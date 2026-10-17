#!/usr/bin/env python3
"""
Compare frequency response of C++ output vs original S-parameters.
"""

import numpy as np
import sys

def compare_response(s4p_file, waveform_file):
    """Compare C++ output waveform with original S21."""
    import skrf
    
    # Load original S21
    nw = skrf.Network(s4p_file)
    freq_orig = nw.frequency.f
    s21_orig = nw.s[:, 1, 0]
    
    # Load C++ output (CSV with header; prefer differential/output column)
    with open(waveform_file, 'r') as f:
        header = [h.strip() for h in f.readline().strip().split(',')]
    data = np.loadtxt(waveform_file, delimiter=',', skiprows=1, ndmin=2)
    time = data[:, 0]
    if 'voltage_diff' in header:
        voltage = data[:, header.index('voltage_diff')]
    elif 'output_V' in header:
        voltage = data[:, header.index('output_V')]
    elif 'voltage_v' in header:
        voltage = data[:, header.index('voltage_v')]
    else:
        voltage = data[:, 1]
    dt = time[1] - time[0]
    
    # Compute FFT
    n_fft = len(voltage)
    V_fft = np.fft.fft(voltage)
    freq_sim = np.fft.fftfreq(n_fft, dt)
    
    pos_idx = freq_sim > 0
    freq_sim = freq_sim[pos_idx]
    V_mag = 20 * np.log10(np.abs(V_fft[pos_idx]) + 1e-20)
    
    # Interpolate original
    s21_mag_orig = 20 * np.log10(np.abs(s21_orig) + 1e-20)
    s21_interp = np.interp(freq_sim, freq_orig, s21_mag_orig)
    
    valid_idx = freq_sim <= freq_orig[-1]
    corr = np.corrcoef(s21_interp[valid_idx], V_mag[valid_idx])[0, 1]
    rmse = np.sqrt(np.mean((s21_interp[valid_idx] - V_mag[valid_idx])**2))
    
    print(f"Correlation: {corr:.4f} (target > 0.95)")
    print(f"RMSE: {rmse:.2f} dB (target < 3 dB)")
    
    return corr, rmse

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('s4p', help='Original S4P file')
    parser.add_argument('waveform', help='C++ output waveform')
    args = parser.parse_args()
    
    corr, rmse = compare_response(args.s4p, args.waveform)
    sys.exit(0 if (corr > 0.95 and rmse < 3.0) else 1)
