#!/usr/bin/env python3
"""
S-parameter preprocessing script for SerDes channel modeling.
This script reads Touchstone S-parameter files and generates configuration files
for SystemC-AMS channel simulation using two methods:
1. Rational Fitting Method (Vector Fitting)
2. Impulse Response Convolution Method (IFFT)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

# Import local modules
from vector_fitting import VectorFitting

logger = logging.getLogger(__name__)


class TouchstoneReader:
    """
    Reader for Touchstone S-parameter files (.sNp format).
    Supports .s1p, .s2p, .s4p, and higher port count files.
    """
    
    def __init__(self, filepath):
        """
        Initialize reader with file path.
        
        Args:
            filepath: Path to Touchstone file
        """
        self.filepath = Path(filepath)
        self.freq = None
        self.s_matrix = None
        self.n_ports = None
        self.freq_unit = 1.0  # Hz multiplier
        self.format = 'MA'    # MA, DB, RI
        self.z0 = 50.0        # Reference impedance
        
    def read(self):
        """
        Read and parse the Touchstone file.
        
        Returns:
            freq: Frequency array (Hz)
            s_matrix: S-parameter matrix [N_freq, N_ports, N_ports]
        """
        # Determine number of ports from file extension
        ext = self.filepath.suffix.lower()
        if ext.startswith('.s') and ext.endswith('p'):
            self.n_ports = int(ext[2:-1])
        else:
            raise ValueError(f"Invalid Touchstone file extension: {ext}")
        
        logger.info(f"Reading {self.n_ports}-port Touchstone file: {self.filepath}")
        
        freq_list = []
        s_data = []
        current_freq_data = []  # Accumulate S-param data for current frequency
        current_freq = None
        
        with open(self.filepath, 'r') as f:
            for line in f:
                # Skip comments
                if line.strip().startswith('!'):
                    continue
                
                # Parse option line
                if line.strip().startswith('#'):
                    self._parse_options(line.strip())
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse data line
                values = line.split()
                if not values:
                    continue
                
                try:
                    data = [float(v) for v in values]
                except ValueError:
                    continue
                
                # Determine if this is a new frequency point or continuation
                # Continuation lines typically start with whitespace (indented)
                is_continuation = line[0].isspace() if line else False
                
                if is_continuation and current_freq is not None:
                    # This is a continuation line - append all values to current freq data
                    current_freq_data.extend(data)
                else:
                    # This is a new frequency point
                    # Save previous frequency data if exists
                    if current_freq is not None:
                        freq_list.append(current_freq)
                        s_data.append(current_freq_data)
                    
                    # Start new frequency point
                    current_freq = data[0] * self.freq_unit
                    current_freq_data = data[1:]
        
        # Don't forget the last frequency point
        if current_freq is not None:
            freq_list.append(current_freq)
            s_data.append(current_freq_data)
        
        # Convert to numpy arrays
        self.freq = np.array(freq_list)
        
        # Ensure all rows have the same length (pad if necessary)
        max_len = max(len(row) for row in s_data) if s_data else 0
        s_data_padded = []
        for row in s_data:
            if len(row) < max_len:
                row = list(row) + [0.0] * (max_len - len(row))
            s_data_padded.append(row)
        s_raw = np.array(s_data_padded)
        
        # Reshape and convert S-parameters
        self.s_matrix = self._convert_s_params(s_raw)
        
        # Validate data
        self._validate()
        
        logger.info(f"Loaded {len(self.freq)} frequency points, "
                   f"range: {self.freq[0]/1e9:.3f} - {self.freq[-1]/1e9:.3f} GHz")
        
        return self.freq, self.s_matrix
    
    def _parse_options(self, line):
        """Parse the option line starting with #."""
        tokens = line[1:].upper().split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Frequency unit
            if token in ['HZ', 'KHZ', 'MHZ', 'GHZ']:
                self.freq_unit = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}[token]
            
            # Data format
            elif token in ['MA', 'DB', 'RI']:
                self.format = token
            
            # Reference impedance
            elif token == 'R':
                if i + 1 < len(tokens):
                    self.z0 = float(tokens[i + 1])
                    i += 1
            
            i += 1
    
    def _convert_s_params(self, s_raw):
        """
        Convert raw S-parameter data to complex matrix format.
        
        Args:
            s_raw: Raw data array [N_freq, N_ports*N_ports*2]
            
        Returns:
            Complex S-matrix [N_freq, N_ports, N_ports]
        """
        N_freq = len(self.freq)
        N = self.n_ports
        
        s_matrix = np.zeros((N_freq, N, N), dtype=complex)
        
        for f_idx in range(N_freq):
            data = s_raw[f_idx]
            param_idx = 0
            
            for i in range(N):
                for j in range(N):
                    if param_idx * 2 + 1 < len(data):
                        v1 = data[param_idx * 2]
                        v2 = data[param_idx * 2 + 1]
                        
                        if self.format == 'MA':
                            # Magnitude, Angle (degrees)
                            s_matrix[f_idx, i, j] = v1 * np.exp(1j * np.radians(v2))
                        elif self.format == 'DB':
                            # dB, Angle (degrees)
                            mag = 10 ** (v1 / 20.0)
                            s_matrix[f_idx, i, j] = mag * np.exp(1j * np.radians(v2))
                        elif self.format == 'RI':
                            # Real, Imaginary
                            s_matrix[f_idx, i, j] = v1 + 1j * v2
                        
                        param_idx += 1
        
        return s_matrix
    
    def _validate(self):
        """Validate the loaded S-parameter data."""
        # Check frequency monotonicity
        if not np.all(np.diff(self.freq) > 0):
            logger.warning("Frequency points are not monotonically increasing")
        
        # Check passivity (preliminary)
        max_mag = np.max(np.abs(self.s_matrix))
        if max_mag > 1.0 + 1e-3:
            logger.warning(f"S-parameter magnitude exceeds 1.0: max={max_mag:.4f}")
    
    def get_sparam(self, i, j):
        """
        Get a specific S-parameter.
        
        Args:
            i, j: Port indices (1-based, as in S21)
            
        Returns:
            Complex S-parameter array over frequency
        """
        return self.s_matrix[:, i-1, j-1]


class DCCompletion:
    """
    DC point completion for S-parameters missing the 0 Hz data point.
    """
    
    @staticmethod
    def magnitude_phase_extrapolation(freq, S_data, n_points=10):
        """
        Extrapolate DC value using magnitude and phase separately.
        This is more robust for transmission lines with delay.
        
        For S21 (transmission): DC magnitude should approach low-freq magnitude,
        and DC phase should be 0 (or extrapolated from delay-removed phase).
        
        Args:
            freq: Frequency array (must not include 0)
            S_data: S-parameter data
            n_points: Number of low-frequency points to use
            
        Returns:
            DC value (complex)
        """
        n_points = min(n_points, len(freq) // 2)
        freq_low = freq[:n_points]
        S_low = S_data[:n_points]
        
        # Extrapolate magnitude (use quadratic for smoother fit)
        mag_low = np.abs(S_low)
        if n_points >= 3:
            coef_mag = np.polyfit(freq_low, mag_low, 2)
            dc_mag = coef_mag[2]  # Constant term
        else:
            dc_mag = mag_low[0]
        
        # Ensure magnitude is positive and reasonable
        dc_mag = max(0.0, min(dc_mag, 1.0))
        
        # For phase: estimate delay and extrapolate residual phase
        phase_low = np.unwrap(np.angle(S_low))
        omega_low = 2 * np.pi * freq_low
        
        # Linear fit: phase = -delay * omega + phase_offset
        if n_points >= 2:
            coef_phase = np.polyfit(omega_low, phase_low, 1)
            dc_phase = coef_phase[1]  # Phase at omega=0
        else:
            dc_phase = phase_low[0]
        
        # DC phase should typically be near 0 for transmission
        # Wrap to [-pi, pi]
        dc_phase = np.angle(np.exp(1j * dc_phase))
        
        dc_value = dc_mag * np.exp(1j * dc_phase)
        
        return dc_value
    
    @staticmethod
    def linear_extrapolation(freq, S_data, n_points=5):
        """
        Linear extrapolation from lowest frequency points.
        
        Args:
            freq: Frequency array
            S_data: S-parameter data
            n_points: Number of low-frequency points to use
            
        Returns:
            DC value (complex)
        """
        n_points = min(n_points, len(freq) // 2)
        freq_low = freq[:n_points]
        S_low = S_data[:n_points]
        
        # Fit linear model separately for real and imaginary parts
        coef_real = np.polyfit(freq_low, S_low.real, 1)
        coef_imag = np.polyfit(freq_low, S_low.imag, 1)
        
        dc_real = coef_real[1]  # Intercept
        dc_imag = coef_imag[1]
        
        return dc_real + 1j * dc_imag
    
    @staticmethod
    def complete(freq, S_data, method='magphase'):
        """
        Complete S-parameter data with DC point.
        
        Args:
            freq: Frequency array
            S_data: S-parameter data
            method: 'magphase' (magnitude/phase), 'interp' (linear interpolation)
            
        Returns:
            freq_new: Frequency array with DC point
            S_new: S-parameter data with DC point
        """
        if freq[0] == 0:
            return freq, S_data
        
        if method == 'magphase':
            S_dc = DCCompletion.magnitude_phase_extrapolation(freq, S_data)
        else:
            S_dc = DCCompletion.linear_extrapolation(freq, S_data)
        
        freq_new = np.concatenate([[0.0], freq])
        S_new = np.concatenate([[S_dc], S_data])
        
        logger.info(f"DC completion ({method}): S(0) = {S_dc:.4f}, |S(0)| = {np.abs(S_dc):.4f}")
        
        return freq_new, S_new


class ImpulseResponseGenerator:
    """
    Generate time-domain impulse response from S-parameter data using IFFT.
    """
    
    def __init__(self, fs, n_samples=4096):
        """
        Initialize generator.
        
        Args:
            fs: Target sampling frequency (Hz)
            n_samples: Number of time samples
        """
        self.fs = fs
        self.n_samples = n_samples
        self.dt = 1.0 / fs
        self.time = np.arange(n_samples) * self.dt
    
    def resample_to_grid(self, freq, S_data):
        """
        Resample S-parameter data to uniform frequency grid.
        
        Args:
            freq: Original frequency array
            S_data: Original S-parameter data
            
        Returns:
            freq_grid: Uniform frequency grid
            S_grid: Resampled S-parameter data
        """
        # Create uniform grid up to Nyquist
        f_nyquist = self.fs / 2
        df = self.fs / self.n_samples
        freq_grid = np.arange(0, f_nyquist + df, df)
        
        # Limit to available data range
        freq_grid = freq_grid[freq_grid <= freq.max()]
        
        # Interpolate (separate real and imaginary)
        interp_real = interp1d(freq, S_data.real, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        interp_imag = interp1d(freq, S_data.imag, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        
        S_grid = interp_real(freq_grid) + 1j * interp_imag(freq_grid)
        
        return freq_grid, S_grid
    
    def apply_bandlimit(self, S_data, rolloff_ratio=0.02):
        """
        Apply bandlimiting window to reduce Gibbs phenomenon.
        Uses a cosine rolloff at the high-frequency edge.
        
        Args:
            S_data: S-parameter data on uniform grid
            rolloff_ratio: Fraction of bandwidth for rolloff (default 0.02 = 2%)
            
        Returns:
            Windowed S-parameter data
        """
        N = len(S_data)
        n_rolloff = max(1, int(N * rolloff_ratio))
        
        window = np.ones(N)
        if n_rolloff > 0:
            # Cosine rolloff (half of raised cosine window)
            rolloff = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n_rolloff)))
            window[-n_rolloff:] = rolloff
        
        return S_data * window
    
    def compute_ifft(self, freq_grid, S_grid):
        """
        Compute inverse FFT to get time-domain impulse response.
        
        Args:
            freq_grid: Uniform frequency grid (positive frequencies)
            S_grid: S-parameter data on grid
            
        Returns:
            time: Time array
            impulse: Time-domain impulse response
        """
        N = self.n_samples
        
        # Build bilateral spectrum (Hermitian symmetric)
        # The freq_grid may not extend to Nyquist, so we need to handle this
        # properly by zero-padding the spectrum beyond available data
        
        N_pos = len(S_grid)
        N_half = N // 2 + 1  # Number of positive frequency bins including DC and Nyquist
        
        S_bilateral = np.zeros(N, dtype=complex)
        
        # Positive frequencies: fill available data, zero-pad the rest
        n_fill = min(N_pos, N_half)
        S_bilateral[:n_fill] = S_grid[:n_fill]
        
        # Negative frequencies (conjugate mirror)
        # For real signal: S[-k] = conj(S[k]) for k = 1, 2, ..., N/2-1
        # S_bilateral[N-k] = conj(S_bilateral[k])
        if n_fill > 1:
            for k in range(1, n_fill):
                if N - k < N:  # Ensure we don't go out of bounds
                    S_bilateral[N - k] = np.conj(S_bilateral[k])
        
        # IFFT
        h_complex = np.fft.ifft(S_bilateral)
        h_real = np.real(h_complex)
        
        return self.time[:len(h_real)], h_real
    
    def apply_causality(self, impulse, method='window'):
        """
        Apply causality constraint to impulse response.
        
        Args:
            impulse: Raw impulse response
            method: 'window' (Hamming) or 'minphase' (minimum phase)
            
        Returns:
            Causal impulse response
        """
        if method == 'window':
            # Find peak
            peak_idx = np.argmax(np.abs(impulse))
            
            # Apply windowed transition for t < t_peak
            h_causal = impulse.copy()
            if peak_idx > 0:
                window = np.hamming(2 * peak_idx)[:peak_idx]
                h_causal[:peak_idx] = impulse[:peak_idx] * window
            
            return h_causal
        
        elif method == 'minphase':
            # Minimum phase transformation
            return signal.minimum_phase(impulse, method='hilbert')
        
        else:
            return impulse
    
    def truncate(self, impulse, threshold=1e-6):
        """
        Truncate impulse response tail below threshold.
        
        Args:
            impulse: Impulse response
            threshold: Relative threshold for truncation
            
        Returns:
            Truncated impulse response
            Truncation length
        """
        abs_impulse = np.abs(impulse)
        peak = np.max(abs_impulse)
        cutoff = threshold * peak
        
        # Find last significant sample
        significant = abs_impulse > cutoff
        if np.any(significant):
            last_idx = np.where(significant)[0][-1]
            L = last_idx + 1
        else:
            L = len(impulse)
        
        # Ensure minimum length
        L = max(L, 64)
        
        # Calculate energy retention
        energy_full = np.sum(impulse ** 2)
        energy_trunc = np.sum(impulse[:L] ** 2)
        retention = energy_trunc / (energy_full + 1e-12)
        
        logger.info(f"Truncated impulse: {len(impulse)} -> {L} samples, "
                   f"energy retention: {retention*100:.2f}%")
        
        return impulse[:L], L
    
    def generate(self, freq, S_data, dc_method='magphase', causality='window', 
                 truncate_threshold=1e-6):
        """
        Generate complete impulse response from S-parameter data.
        
        Args:
            freq: Frequency array
            S_data: S-parameter data
            dc_method: DC completion method
            causality: Causality enforcement method
            truncate_threshold: Truncation threshold
            
        Returns:
            dict: Impulse response data
        """
        # DC completion
        freq_dc, S_dc = DCCompletion.complete(freq, S_data, method=dc_method)
        
        # Resample to uniform grid
        freq_grid, S_grid = self.resample_to_grid(freq_dc, S_dc)
        
        # Apply bandlimit
        S_windowed = self.apply_bandlimit(S_grid)
        
        # IFFT
        time, impulse = self.compute_ifft(freq_grid, S_windowed)
        
        # Causality
        impulse_causal = self.apply_causality(impulse, method=causality)
        
        # Truncation
        impulse_trunc, length = self.truncate(impulse_causal, truncate_threshold)
        time_trunc = time[:length]
        
        # Compute energy and peak time
        # For discrete impulse response, energy = sum(h^2)
        # (Parseval's theorem: energy in time domain = energy in freq domain / N)
        energy = np.sum(impulse_trunc ** 2)
        peak_idx = np.argmax(np.abs(impulse_trunc))
        peak_time = time_trunc[peak_idx]
        
        # DC gain = sum(h) for discrete impulse response
        dc_gain = np.sum(impulse_trunc)
        
        return {
            'time': time_trunc.tolist(),
            'impulse': impulse_trunc.tolist(),
            'length': length,
            'dt': self.dt,
            'energy': float(energy),
            'peak_time': float(peak_time),
            'dc_gain': float(dc_gain)
        }


class ConfigGenerator:
    """
    Generate JSON configuration files for SystemC-AMS channel module.
    """
    
    def __init__(self, fs):
        """
        Initialize generator.
        
        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.config = {
            'version': '1.0',
            'fs': fs,
            'method': None,
            'filters': {},
            'impulse_responses': {},
            'port_mapping': {
                'forward': [],
                'crosstalk': []
            },
            'metadata': {}
        }
    
    def add_rational_filter(self, name, vf_result):
        """
        Add a rational function filter from VF result.
        
        Args:
            name: Filter name (e.g., 'S21')
            vf_result: Vector fitting result dictionary
        """
        self.config['method'] = 'rational'
        num = np.asarray(vf_result['num'], dtype=float)
        den = np.asarray(vf_result['den'], dtype=float)
        self.config['filters'][name] = {
            # Normalized variable u = s / s_scale (2026-10):
            #   H_rational(s) = polyval(num, s/s_scale) / polyval(den, s/s_scale)
            #   full transfer = H_rational(s) * exp(-s * delay_s)
            'num': num.tolist(),
            'den': den.tolist(),
            's_scale': float(vf_result.get('s_scale', 1.0)),
            'delay_s': float(vf_result.get('delay', 0.0)),
            'order': int(vf_result['order']),
            'dc_gain': float(num[-1] / den[-1]) if den[-1] != 0 else 0.0,
            'mse': float(vf_result['mse']),
            'max_error': float(vf_result['max_error']),
            'rel_rms': float(vf_result.get('rel_rms', float('nan'))),
            'median_error_db': float(vf_result.get('median_error_db', float('nan'))),
        }
    
    def add_impulse_response(self, name, ir_result):
        """
        Add an impulse response from IR generator result.
        
        Args:
            name: Response name (e.g., 'S21')
            ir_result: Impulse response result dictionary
        """
        self.config['method'] = 'impulse'
        # Ensure all values are JSON serializable
        ir_serializable = {
            'time': ir_result['time'] if isinstance(ir_result['time'], list) else list(ir_result['time']),
            'impulse': ir_result['impulse'] if isinstance(ir_result['impulse'], list) else list(ir_result['impulse']),
            'length': int(ir_result['length']),
            'dt': float(ir_result['dt']),
            'energy': float(ir_result['energy']),
            'peak_time': float(ir_result['peak_time'])
        }
        self.config['impulse_responses'][name] = ir_serializable
    
    def set_port_mapping(self, forward_pairs, crosstalk_pairs=None):
        """
        Set port mapping configuration.
        
        Args:
            forward_pairs: List of [in_port, out_port] pairs
            crosstalk_pairs: List of [in_port, out_port] crosstalk pairs
        """
        self.config['port_mapping']['forward'] = forward_pairs
        if crosstalk_pairs:
            self.config['port_mapping']['crosstalk'] = crosstalk_pairs
    
    def set_metadata(self, source_file, n_ports, freq_range):
        """
        Set metadata about the source file.
        
        Args:
            source_file: Source Touchstone file path
            n_ports: Number of ports
            freq_range: (f_min, f_max) tuple
        """
        self.config['metadata'] = {
            'source_file': str(source_file),
            'n_ports': n_ports,
            'freq_min_hz': float(freq_range[0]),
            'freq_max_hz': float(freq_range[1])
        }
    
    def save(self, filepath):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")


def process_sparam(input_file, output_file, method='both', fs=100e9, 
                   order=8, n_samples=4096, sparams=None):
    """
    Main processing function for S-parameter files.
    
    Args:
        input_file: Input Touchstone file path
        output_file: Output configuration file path
        method: Processing method ('rational', 'impulse', or 'both')
        fs: Sampling frequency (Hz)
        order: Vector fitting order
        n_samples: Number of impulse response samples
        sparams: List of S-parameters to process (e.g., ['S21', 'S11'])
    """
    # Read Touchstone file
    reader = TouchstoneReader(input_file)
    freq, s_matrix = reader.read()
    
    # Check passivity (max singular value of S over frequency, <= 1 for passive)
    singular_values = np.linalg.svd(s_matrix, compute_uv=False)
    max_eig = float(np.max(singular_values))
    violations = np.where(singular_values.max(axis=1) > 1.0 + 1e-3)[0]
    is_passive = len(violations) == 0
    if not is_passive:
        logger.warning(f"S-parameters violate passivity at {len(violations)} frequencies")
        logger.warning(f"Maximum eigenvalue: {max_eig:.4f}")
    
    # Initialize config generator
    config_gen = ConfigGenerator(fs)
    config_gen.set_metadata(input_file, reader.n_ports, (freq[0], freq[-1]))
    
    # Determine which S-parameters to process
    if sparams is None:
        # Default: process main transmission path (S21 for 2-port)
        if reader.n_ports == 2:
            sparams = ['S21']
        elif reader.n_ports == 4:
            sparams = ['S21', 'S43']  # Differential pair
        else:
            sparams = ['S21']
    
    # Process each S-parameter
    for sparam in sparams:
        # Parse S-parameter name (e.g., 'S21' -> i=2, j=1)
        i = int(sparam[1])
        j = int(sparam[2])
        S_data = reader.get_sparam(i, j)
        
        logger.info(f"Processing {sparam}")
        
        # Rational fitting method (delay-aware VF, 2026-10)
        if method in ['rational', 'both']:
            vf = VectorFitting(order=order)
            vf_result = vf.fit(2j * np.pi * freq, S_data)  # VF expects rad/s
            logger.info("%s: VF order=%d delay=%.3f ns rel_rms=%.2e "
                        "median dB err=%.2f",
                        sparam, order, vf_result.get('delay', 0.0)*1e9,
                        vf_result.get('rel_rms', float('nan')),
                        vf_result.get('median_error_db', float('nan')))
            config_gen.add_rational_filter(sparam, vf_result)
        
        # Impulse response method
        if method in ['impulse', 'both']:
            ir_gen = ImpulseResponseGenerator(fs, n_samples)
            ir_result = ir_gen.generate(freq, S_data)
            config_gen.add_impulse_response(sparam, ir_result)
    
    # Set port mapping (simplified for now)
    forward_pairs = [[int(s[2]), int(s[1])] for s in sparams]
    config_gen.set_port_mapping(forward_pairs)
    
    # Save configuration
    config_gen.save(output_file)
    
    return config_gen.config


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Process S-parameter files for SystemC-AMS channel simulation'
    )
    parser.add_argument('input', help='Input Touchstone file (.sNp)')
    parser.add_argument('-o', '--output', default='channel_config.json',
                       help='Output configuration file (default: channel_config.json)')
    parser.add_argument('-m', '--method', choices=['rational', 'impulse', 'both'],
                       default='impulse',
                       help='Processing method (default: impulse; rational exports '
                            'a delay-aware VF model, impulse drives the simulator)')
    parser.add_argument('-f', '--fs', type=float, default=100e9,
                       help='Sampling frequency in Hz (default: 100e9)')
    parser.add_argument('--order', type=int, default=8,
                       help='Vector fitting order (default: 8)')
    parser.add_argument('--samples', type=int, default=4096,
                       help='Impulse response samples (default: 4096)')
    parser.add_argument('-s', '--sparams', nargs='+',
                       help='S-parameters to process (e.g., S21 S11)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Process
    try:
        config = process_sparam(
            args.input, args.output, 
            method=args.method,
            fs=args.fs,
            order=args.order,
            n_samples=args.samples,
            sparams=args.sparams
        )
        print(f"Successfully processed {args.input}")
        print(f"Configuration saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
