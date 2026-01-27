"""
Utility Functions for EyeAnalyzer

This module provides helper functions for validation, file I/O, and common operations.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

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