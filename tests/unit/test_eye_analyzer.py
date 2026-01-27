"""
Unit Tests for EyeAnalyzer

This module contains unit tests for the EyeAnalyzer package,
covering data loading, core analysis functions, and utilities.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eye_analyzer import EyeAnalyzer, auto_load_waveform
from eye_analyzer.io import load_waveform_from_dat, load_waveform_from_csv
from eye_analyzer.utils import validate_ui, validate_bins, save_metrics_json


class TestIOFunctions:
    """Test data loading functions."""

    def test_load_waveform_from_dat_with_mock_data(self, tmp_path):
        """Test loading .dat file with mock data."""
        # Create mock .dat file
        dat_file = tmp_path / "test.dat"
        with open(dat_file, 'w') as f:
            f.write("# Mock data\n")
            f.write("0.0 0.0\n")
            f.write("1.0e-11 0.4\n")
            f.write("2.0e-11 -0.4\n")
            f.write("3.0e-11 0.4\n")

        # Load data
        time_array, value_array = load_waveform_from_dat(str(dat_file))

        # Verify
        assert len(time_array) == 4
        assert len(value_array) == 4
        assert time_array[0] == 0.0
        assert value_array[1] == 0.4

    def test_load_waveform_from_csv_with_mock_data(self, tmp_path):
        """Test loading .csv file with mock data."""
        # Create mock .csv file
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("time,diff\n")
            f.write("0.0,0.0\n")
            f.write("1.0e-11,0.4\n")
            f.write("2.0e-11,-0.4\n")

        # Load data
        time_array, value_array = load_waveform_from_csv(str(csv_file))

        # Verify
        assert len(time_array) == 3
        assert len(value_array) == 3
        assert time_array[0] == 0.0
        assert value_array[1] == 0.4

    def test_load_waveform_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_waveform_from_dat("nonexistent.dat")

    def test_load_waveform_invalid_format(self, tmp_path):
        """Test error handling for invalid format."""
        invalid_file = tmp_path / "invalid.dat"
        with open(invalid_file, 'w') as f:
            f.write("invalid data format")

        with pytest.raises(ValueError):
            load_waveform_from_dat(str(invalid_file))


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_ui_valid(self):
        """Test UI validation with valid values."""
        validate_ui(2.5e-11)  # Should not raise

    def test_validate_ui_negative(self):
        """Test UI validation with negative value."""
        with pytest.raises(ValueError):
            validate_ui(-1.0)

    def test_validate_ui_zero(self):
        """Test UI validation with zero."""
        with pytest.raises(ValueError):
            validate_ui(0.0)

    def test_validate_ui_too_small(self):
        """Test UI validation with too small value."""
        with pytest.raises(ValueError):
            validate_ui(1e-16)

    def test_validate_bins_valid(self):
        """Test bins validation with valid values."""
        validate_bins(128)  # Should not raise

    def test_validate_bins_negative(self):
        """Test bins validation with negative value."""
        with pytest.raises(ValueError):
            validate_bins(-10)

    def test_validate_bins_too_large(self):
        """Test bins validation with too large value."""
        with pytest.raises(ValueError):
            validate_bins(20000)

    def test_validate_bins_odd(self):
        """Test bins validation with odd value."""
        with pytest.raises(ValueError):
            validate_bins(127)

    def test_save_metrics_json(self, tmp_path):
        """Test saving metrics to JSON file."""
        metrics = {
            'eye_height': 0.756,
            'eye_width': 0.942
        }

        output_path = tmp_path / "metrics.json"
        save_metrics_json(metrics, str(output_path))

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify content
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)

        assert 'version' in data
        assert 'timestamp' in data
        assert 'metrics' in data
        assert data['metrics']['eye_height'] == 0.756


class TestEyeAnalyzer:
    """Test EyeAnalyzer class."""

    def test_initialization_default(self):
        """Test analyzer initialization with default parameters."""
        analyzer = EyeAnalyzer(ui=2.5e-11)

        assert analyzer.ui == 2.5e-11
        assert analyzer.ui_bins == 128
        assert analyzer.amp_bins == 128

    def test_initialization_custom_bins(self):
        """Test analyzer initialization with custom bins."""
        analyzer = EyeAnalyzer(ui=2.5e-11, ui_bins=256, amp_bins=256)

        assert analyzer.ui_bins == 256
        assert analyzer.amp_bins == 256

    def test_initialization_invalid_ui(self):
        """Test analyzer initialization with invalid UI."""
        with pytest.raises(ValueError):
            EyeAnalyzer(ui=-1.0)

    def test_phase_normalization(self):
        """Test phase normalization."""
        # Create time array spanning multiple UIs
        ui = 2.5e-11
        time_array = np.array([0, 1e-11, 2.5e-11, 3e-11, 5e-11, 7.5e-11])

        analyzer = EyeAnalyzer(ui=ui)
        phase = analyzer._normalize_phase(time_array)

        # Verify phase is in [0, 1)
        assert np.all(phase >= 0)
        assert np.all(phase < 1)

        # Verify boundary conditions
        assert abs(phase[0] - 0.0) < 1e-10  # t=0 -> phi=0
        assert abs(phase[2] - 0.0) < 1e-10  # t=UI -> phi=0
        assert abs(phase[4] - 0.0) < 1e-10  # t=2*UI -> phi=0
        assert abs(phase[1] - 0.4) < 1e-10  # t=1e-11 -> phi=0.4
        assert abs(phase[3] - 0.2) < 1e-10  # t=3e-11 -> phi=0.2

    def test_eye_diagram_construction(self):
        """Test eye diagram construction."""
        # Generate test data: 1000 UI of binary signal
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui, ui_bins=128, amp_bins=128)
        hist2d, xedges, yedges = analyzer._build_eye_diagram(
            analyzer._normalize_phase(time_array),
            value_array
        )

        # Verify histogram shape
        assert hist2d.shape == (128, 128)
        assert len(xedges) == 129
        assert len(yedges) == 129

        # Verify total count
        assert np.sum(hist2d) == num_ui

        # Verify edge ranges
        assert xedges[0] == 0.0
        assert abs(xedges[-1] - 1.0) < 1e-10  # Tolerate floating point precision

    def test_eye_height_calculation(self):
        """Test eye height calculation."""
        # Create artificial eye diagram: two horizontal bands
        hist2d = np.zeros((128, 128))
        # Upper eye edge
        hist2d[50:78, 30:50] = 100
        # Lower eye edge
        hist2d[50:78, 78:98] = 100

        yedges = np.linspace(-1.0, 1.0, 129)

        analyzer = EyeAnalyzer(ui=2.5e-11, amp_bins=128)
        eye_height = analyzer._compute_eye_height(hist2d, yedges)

        # Verify eye height is positive
        assert eye_height > 0

        # Verify eye height is reasonable (should be less than total range)
        assert eye_height < 2.0

    def test_eye_width_calculation(self):
        """Test eye width calculation."""
        # Create artificial eye diagram: two vertical bands
        hist2d = np.zeros((128, 128))
        # Left eye edge
        hist2d[30:50, 50:78] = 100
        # Right eye edge
        hist2d[78:98, 50:78] = 100

        xedges = np.linspace(0.0, 1.0, 129)

        analyzer = EyeAnalyzer(ui=2.5e-11, ui_bins=128)
        eye_width = analyzer._compute_eye_width(hist2d, xedges)

        # Verify eye width is positive
        assert eye_width > 0

        # Verify eye width is in [0, 1] UI
        assert eye_width <= 1.0

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate test data: 10000 UI of binary signal with some noise
        ui = 2.5e-11
        num_ui = 10000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        # Add small noise
        value_array += np.random.normal(0, 0.01, size=num_ui)

        analyzer = EyeAnalyzer(ui=ui)
        metrics = analyzer.analyze(time_array, value_array)

        # Verify metrics exist
        assert 'eye_height' in metrics
        assert 'eye_width' in metrics

        # Verify metrics are positive
        assert metrics['eye_height'] > 0
        assert metrics['eye_width'] > 0

        # Verify metrics are reasonable
        assert metrics['eye_height'] < 1.0  # Less than signal swing
        assert metrics['eye_width'] <= 1.0  # At most 1 UI

    def test_save_results(self, tmp_path):
        """Test saving analysis results."""
        # Generate test data
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui)
        metrics = analyzer.analyze(time_array, value_array)

        # Save results
        output_dir = str(tmp_path)
        analyzer.save_results(metrics, output_dir)

        # Verify files were created
        assert os.path.exists(os.path.join(output_dir, 'eye_metrics.json'))
        assert os.path.exists(os.path.join(output_dir, 'eye_diagram.png'))

    def test_analysis_with_empty_arrays(self):
        """Test error handling for empty arrays."""
        analyzer = EyeAnalyzer(ui=2.5e-11)

        with pytest.raises(ValueError):
            analyzer.analyze(np.array([]), np.array([]))

    def test_analysis_with_mismatched_arrays(self):
        """Test error handling for mismatched array lengths."""
        analyzer = EyeAnalyzer(ui=2.5e-11)

        with pytest.raises(ValueError):
            analyzer.analyze(np.array([0, 1, 2]), np.array([0, 1]))


class TestAutoLoadWaveform:
    """Test auto_load_waveform function."""

    def test_auto_load_dat_file(self, tmp_path):
        """Test auto-loading .dat file."""
        dat_file = tmp_path / "test.dat"
        with open(dat_file, 'w') as f:
            f.write("# Mock data\n")
            f.write("0.0 0.0\n")
            f.write("1.0e-11 0.4\n")

        time_array, value_array = auto_load_waveform(str(dat_file))

        assert len(time_array) == 2
        assert len(value_array) == 2

    def test_auto_load_csv_file(self, tmp_path):
        """Test auto-loading .csv file."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("time,diff\n")
            f.write("0.0,0.0\n")
            f.write("1.0e-11,0.4\n")

        time_array, value_array = auto_load_waveform(str(csv_file))

        assert len(time_array) == 2
        assert len(value_array) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])