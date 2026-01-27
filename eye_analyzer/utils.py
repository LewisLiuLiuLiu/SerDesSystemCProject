"""
Utility Functions for EyeAnalyzer

This module provides helper functions for validation, file I/O, and common operations.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
from scipy.special import erfcinv


def validate_ui(ui: float) -> None:
    """
    Validate the UI (Unit Interval) parameter.

    Args:
        ui: Unit interval in seconds

    Raises:
        ValueError: If UI is invalid (non-positive or too small)
    """
    if ui <= 0:
        raise ValueError(f"UI must be positive, got {ui}")
    if ui < 1e-15:
        raise ValueError(f"UI is too small (< 1e-15s), got {ui}")


def create_output_directory(output_dir: str) -> None:
    """
    Create output directory if it does not exist.

    Args:
        output_dir: Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save analysis metrics to a JSON file.

    Args:
        metrics: Dictionary containing analysis metrics
        filepath: Path to the output JSON file
    """
    # Add metadata
    output_data = {
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    # Write JSON file
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)


def validate_bins(bins: int, name: str = "bins") -> None:
    """
    Validate histogram bin count.

    Args:
        bins: Number of bins
        name: Parameter name for error message

    Raises:
        ValueError: If bins is invalid
    """
    if bins <= 0:
        raise ValueError(f"{name} must be positive, got {bins}")
    if bins > 10000:
        raise ValueError(f"{name} is too large (> 10000), got {bins}")
    if bins % 2 != 0:
        raise ValueError(f"{name} must be even, got {bins}")


def validate_input_arrays(time_array, value_array) -> None:
    """
    Validate input time and value arrays.

    Args:
        time_array: Time array
        value_array: Value array

    Raises:
        ValueError: If arrays are invalid
    """
    if len(time_array) != len(value_array):
        raise ValueError(
            f"Time array length ({len(time_array)}) "
            f"does not match value array length ({len(value_array)})"
        )

    if len(time_array) == 0:
        raise ValueError("Input arrays are empty")

    if len(time_array) < 100:
        print(f"Warning: Only {len(time_array)} samples, results may be unreliable")


def q_function(ber: float) -> float:
    """
    Compute Q function value for given BER.

    Q function relates BER to the number of standard deviations in a Gaussian distribution.
    It is defined as the inverse of the complementary error function.

    Q(x) = 0.5 * erfc(x / sqrt(2))

    For BER = 1e-12, Q ≈ 7.03

    Args:
        ber: Bit error rate (e.g., 1e-12)

    Returns:
        Q function value (e.g., 7.03 for BER=1e-12)

    Raises:
        ValueError: If BER is not in valid range (0 < BER < 0.5)

    Examples:
        >>> q_function(1e-12)
        7.034...
        >>> q_function(1e-9)
        5.997...
    """
    if ber <= 0:
        raise ValueError(f"BER must be positive, got {ber}")
    if ber >= 0.5:
        raise ValueError(f"BER must be less than 0.5, got {ber}")

    # Q(ber) = sqrt(2) * erfcinv(2 * ber)
    return np.sqrt(2) * erfcinv(2 * ber)


def calculate_r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    R-squared measures the proportion of variance in the dependent variable
    that is predictable from the independent variable(s). It ranges from 0 to 1,
    with higher values indicating better fit.

    Formula: R² = 1 - (SS_res / SS_tot)
    Where:
    - SS_res = sum of squared residuals (sum((y_actual - y_predicted)²))
    - SS_tot = total sum of squares (sum((y_actual - mean(y_actual))²))

    Args:
        y_actual: Actual observed values
        y_predicted: Predicted values from model

    Returns:
        R-squared value in range [0, 1], where:
        - 1.0: Perfect fit
        - 0.0: Model performs no better than predicting the mean
        - Negative values: Model performs worse than predicting the mean

    Examples:
        >>> y_actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_predicted = np.array([1.1, 1.9, 3.1, 4.0, 5.0])
        >>> r2 = calculate_r_squared(y_actual, y_predicted)
        >>> print(f"R-squared: {r2:.3f}")
        R-squared: 0.997
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot == 0:
        # All actual values are the same, R-squared is undefined
        return 0.0

    return 1.0 - (ss_res / ss_tot)


# ============================================================================
# CSV Output Functions (per EyeAnalyzer.md specification)
# ============================================================================

def save_hist2d_csv(hist2d: np.ndarray, xedges: np.ndarray, yedges: np.ndarray,
                    filepath: str) -> None:
    """
    Save 2D histogram data to CSV file.

    Output format (per EyeAnalyzer.md):
    phase_bin,amplitude_bin,density

    Args:
        hist2d: 2D histogram matrix (shape: ui_bins x amp_bins)
        xedges: Phase bin edges
        yedges: Amplitude bin edges
        filepath: Output CSV file path
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phase_bin', 'amplitude_bin', 'density'])
        for i in range(hist2d.shape[0]):
            for j in range(hist2d.shape[1]):
                writer.writerow([i, j, hist2d[i, j]])


def save_psd_csv(frequencies: np.ndarray, psd_values: np.ndarray,
                 filepath: str) -> None:
    """
    Save Power Spectral Density data to CSV file.

    Output format (per EyeAnalyzer.md):
    frequency_hz,psd_v2_per_hz

    Args:
        frequencies: Frequency array in Hz
        psd_values: PSD values in V^2/Hz
        filepath: Output CSV file path
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frequency_hz', 'psd_v2_per_hz'])
        for freq, psd in zip(frequencies, psd_values):
            writer.writerow([f'{freq:.6e}', f'{psd:.6e}'])


def save_pdf_csv(amplitudes: np.ndarray, pdf_values: np.ndarray,
                 filepath: str) -> None:
    """
    Save Probability Density Function data to CSV file.

    Output format (per EyeAnalyzer.md):
    amplitude_v,probability_density

    Args:
        amplitudes: Amplitude values in Volts
        pdf_values: Probability density values
        filepath: Output CSV file path
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['amplitude_v', 'probability_density'])
        for amp, pdf in zip(amplitudes, pdf_values):
            writer.writerow([f'{amp:.6f}', f'{pdf:.6f}'])


def save_jitter_distribution_csv(time_offsets: np.ndarray, probabilities: np.ndarray,
                                  filepath: str) -> None:
    """
    Save jitter distribution data to CSV file.

    Output format (per EyeAnalyzer.md):
    time_offset_s,probability

    Args:
        time_offsets: Time offset values in seconds
        probabilities: Probability values
        filepath: Output CSV file path
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_offset_s', 'probability'])
        for offset, prob in zip(time_offsets, probabilities):
            writer.writerow([f'{offset:.6e}', f'{prob:.6f}'])


# ============================================================================
# JSON Output Functions (per EyeAnalyzer.md specification)
# ============================================================================

def format_metrics_to_spec(metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics dictionary to match EyeAnalyzer.md specification.

    The output JSON structure follows the document specification:
    - metadata: version, timestamp, dat_path, ui, ui_bins, amp_bins, measure_length
    - eye_geometry: eye_height, eye_width, eye_area, linearity_error, optimal_sampling_phase, optimal_threshold
    - jitter_decomposition: rj_sigma, dj_pp, tj_at_ber, target_ber, q_factor, method
    - signal_quality: mean, rms, peak_to_peak, psd_peak_freq, psd_peak_value
    - data_provenance: total_samples, analyzed_samples, sampling_rate, duration

    Args:
        metrics: Raw metrics dictionary from EyeAnalyzer.analyze()
        metadata: Configuration metadata (ui, ui_bins, amp_bins, etc.)

    Returns:
        Formatted dictionary matching the specification
    """
    return {
        "metadata": {
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dat_path": metadata.get('dat_path', ''),
            "ui": metadata.get('ui', 0.0),
            "ui_bins": metadata.get('ui_bins', 128),
            "amp_bins": metadata.get('amp_bins', 128),
            "measure_length": metadata.get('measure_length', None)
        },
        "eye_geometry": {
            "eye_height": metrics.get('eye_height', 0.0),
            "eye_width": metrics.get('eye_width', 0.0),
            "eye_area": metrics.get('eye_area', 0.0),
            "linearity_error": metrics.get('linearity_error', 0.0),
            "optimal_sampling_phase": metrics.get('optimal_sampling_phase', 0.5),
            "optimal_threshold": metrics.get('optimal_threshold', 0.0)
        },
        "jitter_decomposition": {
            "rj_sigma": metrics.get('rj_sigma', 0.0),
            "dj_pp": metrics.get('dj_pp', 0.0),
            "tj_at_ber": metrics.get('tj_at_ber', 0.0),
            "target_ber": metrics.get('target_ber', 1e-12),
            "q_factor": metrics.get('q_factor', 7.034),
            "method": metrics.get('fit_method', 'dual-dirac')
        },
        "signal_quality": {
            "mean": metrics.get('signal_mean', 0.0),
            "rms": metrics.get('signal_rms', 0.0),
            "peak_to_peak": metrics.get('signal_peak_to_peak', 0.0),
            "psd_peak_freq": metrics.get('psd_peak_freq', 0.0),
            "psd_peak_value": metrics.get('psd_peak_value', 0.0)
        },
        "data_provenance": {
            "total_samples": metrics.get('total_samples', 0),
            "analyzed_samples": metrics.get('analyzed_samples', 0),
            "sampling_rate": metrics.get('sampling_rate', 0.0),
            "duration": metrics.get('duration', 0.0)
        }
    }


def save_metrics_json_spec(metrics: Dict[str, Any], metadata: Dict[str, Any],
                           filepath: str) -> None:
    """
    Save analysis metrics to JSON file using EyeAnalyzer.md specification format.

    Args:
        metrics: Raw metrics dictionary from EyeAnalyzer.analyze()
        metadata: Configuration metadata
        filepath: Path to the output JSON file
    """
    output_data = format_metrics_to_spec(metrics, metadata)

    output_dir = os.path.dirname(filepath)
    if output_dir:
        create_output_directory(output_dir)

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)