"""
Data Loading Module for EyeAnalyzer

This module provides functions to load waveform data from SystemC-AMS output files.
Supports both Tabular format (.dat) and CSV format (.csv).
"""

import os
from typing import Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_waveform_from_dat(dat_path: str, signal_column: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load waveform data from SystemC-AMS Tabular format file.

    The .dat file format is a space-separated text file with:
    - Column 0: Time (seconds)
    - Column 1..N: Signal values (volts)
    - Lines starting with '#' are comments

    Args:
        dat_path: Path to the .dat file
        signal_column: Index of the signal column to load (default: 1 for second column)

    Returns:
        Tuple of (time_array, value_array) as numpy arrays

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"File not found: {dat_path}")

    try:
        # Read all data from .dat file
        data = np.loadtxt(dat_path, comments='#')

        # Validate data shape
        if data.ndim == 1:
            # Single column file
            if signal_column == 0:
                time_array = data
                value_array = np.zeros_like(time_array)
            else:
                raise ValueError(f"File has only 1 column, but signal_column={signal_column}")
        elif data.ndim == 2:
            # Multi-column file
            if signal_column >= data.shape[1]:
                raise ValueError(
                    f"signal_column={signal_column} out of range "
                    f"(file has {data.shape[1]} columns)"
                )
            time_array = data[:, 0]
            value_array = data[:, signal_column]
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        # Validate time monotonicity
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("Time stamps are not strictly increasing")

        return time_array, value_array

    except Exception as e:
        raise ValueError(f"Failed to load .dat file: {e}")


def load_waveform_from_csv(csv_path: str, time_col: str = 'time',
                           signal_col: str = 'diff') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load waveform data from CSV format file.

    The CSV file format is a standard comma-separated file with headers.
    Common columns include: 'time', 'diff', 'cm', etc.

    Args:
        csv_path: Path to the .csv file
        time_col: Name of the time column (default: 'time')
        signal_col: Name of the signal column (default: 'diff')

    Returns:
        Tuple of (time_array, value_array) as numpy arrays

    Raises:
        FileNotFoundError: If the file does not exist
        ImportError: If pandas is not installed
        ValueError: If the file format is invalid or columns are missing
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required to load CSV files. "
            "Install it with: pip install pandas"
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        # Validate required columns
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in CSV file")
        if signal_col not in df.columns:
            raise ValueError(f"Column '{signal_col}' not found in CSV file")

        # Extract arrays
        time_array = df[time_col].values
        value_array = df[signal_col].values

        # Validate time monotonicity
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("Time stamps are not strictly increasing")

        return time_array, value_array

    except Exception as e:
        raise ValueError(f"Failed to load .csv file: {e}")


def auto_load_waveform(filepath: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect file format and load waveform data.

    Supports .dat (Tabular) and .csv formats.

    Args:
        filepath: Path to the waveform file
        **kwargs: Additional arguments passed to the specific loader function
                 - For .dat: signal_column (default: 1)
                 - For .csv: time_col (default: 'time'), signal_col (default: 'diff')

    Returns:
        Tuple of (time_array, value_array) as numpy arrays

    Raises:
        ValueError: If the file format is not supported

    Examples:
        >>> time, value = auto_load_waveform('results.dat')
        >>> time, value = auto_load_waveform('results.csv', signal_col='v_out')
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect file format based on extension
    if filepath.endswith('.dat'):
        return load_waveform_from_dat(filepath, **kwargs)
    elif filepath.endswith('.csv'):
        return load_waveform_from_csv(filepath, **kwargs)
    else:
        # Try to detect format by reading first line
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if ',' in first_line:
                    # Likely CSV format
                    return load_waveform_from_csv(filepath, **kwargs)
                else:
                    # Likely .dat format
                    return load_waveform_from_dat(filepath, **kwargs)
        except Exception:
            raise ValueError(
                f"Unsupported file format: {filepath}. "
                f"Supported formats: .dat, .csv"
            )