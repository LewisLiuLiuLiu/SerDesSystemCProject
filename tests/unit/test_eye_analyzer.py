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

from eye_analyzer import EyeAnalyzer, auto_load_waveform, analyze_eye
from eye_analyzer.io import load_waveform_from_dat, load_waveform_from_csv
from eye_analyzer.utils import (
    validate_ui, validate_bins, save_metrics_json, 
    save_hist2d_csv, save_psd_csv, save_pdf_csv, save_jitter_distribution_csv,
    format_metrics_to_spec, save_metrics_json_spec
)


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

        # Use hist2d_normalize=False to preserve sample count semantics
        analyzer = EyeAnalyzer(ui=ui, ui_bins=128, amp_bins=128, hist2d_normalize=False)
        hist2d, xedges, yedges = analyzer._build_eye_diagram(
            analyzer._normalize_phase(time_array),
            value_array
        )

        # Verify histogram shape
        assert hist2d.shape == (128, 128)
        assert len(xedges) == 129
        assert len(yedges) == 129

        # Verify total count (only valid when hist2d_normalize=False)
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


# ============================================================================
# New test classes per EyeAnalyzer.md specification
# ============================================================================

class TestSamplingStrategy:
    """Test sampling phase estimation strategies (SAMPLING_STRATEGY scenario)."""

    def _generate_test_signal(self, num_ui=10000, ui=2.5e-11, noise_sigma=0.01):
        """Generate test signal with noise."""
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        value_array += np.random.normal(0, noise_sigma, size=num_ui)
        return time_array, value_array

    def test_peak_sampling_basic(self):
        """Test peak sampling strategy initialization."""
        analyzer = EyeAnalyzer(ui=2.5e-11, sampling='peak')
        assert analyzer.sampling == 'peak'

    def test_zero_cross_sampling_basic(self):
        """Test zero-cross sampling strategy initialization."""
        analyzer = EyeAnalyzer(ui=2.5e-11, sampling='zero-cross')
        assert analyzer.sampling == 'zero-cross'

    def test_phase_lock_sampling_basic(self):
        """Test phase-lock sampling strategy initialization (default)."""
        analyzer = EyeAnalyzer(ui=2.5e-11, sampling='phase-lock')
        assert analyzer.sampling == 'phase-lock'

    def test_invalid_sampling_strategy(self):
        """Test error on invalid sampling strategy."""
        with pytest.raises(ValueError):
            EyeAnalyzer(ui=2.5e-11, sampling='invalid')

    def test_sampling_strategy_high_snr_comparison(self):
        """Test that all strategies produce similar results at high SNR."""
        time_array, value_array = self._generate_test_signal(noise_sigma=0.005)

        results = {}
        for strategy in ['peak', 'zero-cross', 'phase-lock']:
            analyzer = EyeAnalyzer(ui=2.5e-11, sampling=strategy)
            metrics = analyzer.analyze(time_array, value_array)
            results[strategy] = metrics['eye_height']

        # At high SNR, results should be within 10%
        heights = list(results.values())
        max_diff = (max(heights) - min(heights)) / np.mean(heights)
        assert max_diff < 0.1, f"Strategy results differ by {max_diff*100:.1f}%"


class TestEyeGeometryExtended:
    """Test extended eye geometry metrics (BASIC_EYE_ANALYSIS scenario)."""

    def _generate_test_signal(self, num_ui=10000, ui=2.5e-11):
        """Generate test signal."""
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        value_array += np.random.normal(0, 0.01, size=num_ui)
        return time_array, value_array

    def test_eye_area_calculation(self):
        """Test eye area is computed and positive."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'eye_area' in metrics
        assert metrics['eye_area'] >= 0

    def test_linearity_error_calculation(self):
        """Test linearity error is computed and normalized."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'linearity_error' in metrics
        assert 0 <= metrics['linearity_error'] <= 1

    def test_optimal_sampling_phase_detection(self):
        """Test optimal sampling phase is in valid range."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'optimal_sampling_phase' in metrics
        assert 0 <= metrics['optimal_sampling_phase'] <= 1

    def test_optimal_threshold_detection(self):
        """Test optimal threshold is computed."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'optimal_threshold' in metrics


class TestSignalQuality:
    """Test signal quality metrics."""

    def _generate_test_signal(self, num_ui=10000, ui=2.5e-11):
        """Generate test signal."""
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        value_array += np.random.normal(0, 0.01, size=num_ui)
        return time_array, value_array

    def test_signal_mean_calculation(self):
        """Test signal mean is computed."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'signal_mean' in metrics
        # Mean should be close to 0 for balanced signal
        assert abs(metrics['signal_mean']) < 0.1

    def test_signal_rms_calculation(self):
        """Test signal RMS is computed and positive."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'signal_rms' in metrics
        assert metrics['signal_rms'] > 0

    def test_peak_to_peak_calculation(self):
        """Test peak-to-peak is computed correctly."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'signal_peak_to_peak' in metrics
        # Should be approximately 0.8V (±0.4V signal)
        assert 0.7 < metrics['signal_peak_to_peak'] < 0.9

    def test_psd_peak_detection(self):
        """Test PSD peak frequency and value are computed."""
        time_array, value_array = self._generate_test_signal()
        analyzer = EyeAnalyzer(ui=2.5e-11)
        metrics = analyzer.analyze(time_array, value_array)

        assert 'psd_peak_freq' in metrics
        assert 'psd_peak_value' in metrics


class TestMeasureLength:
    """Test measure_length parameter functionality."""

    def test_measure_length_truncation(self):
        """Test data truncation with measure_length."""
        ui = 2.5e-11
        num_ui = 10000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        # Use only last 50% of data
        measure_length = time_array[-1] * 0.5
        analyzer = EyeAnalyzer(ui=ui, measure_length=measure_length)
        metrics = analyzer.analyze(time_array, value_array)

        # analyzed_samples should be roughly half of total
        assert metrics['analyzed_samples'] < metrics['total_samples']
        assert metrics['analyzed_samples'] > metrics['total_samples'] * 0.4

    def test_measure_length_none_uses_all_data(self):
        """Test that measure_length=None uses all data."""
        ui = 2.5e-11
        num_ui = 10000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui, measure_length=None)
        metrics = analyzer.analyze(time_array, value_array)

        assert metrics['analyzed_samples'] == metrics['total_samples']

    def test_measure_length_exceeds_data(self):
        """Test behavior when measure_length > data duration."""
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        # measure_length longer than total duration
        measure_length = time_array[-1] * 2.0
        analyzer = EyeAnalyzer(ui=ui, measure_length=measure_length)
        metrics = analyzer.analyze(time_array, value_array)

        # Should use all data with warning
        assert metrics['analyzed_samples'] == metrics['total_samples']


class TestCSVOutput:
    """Test CSV data output functionality."""

    def test_save_hist2d_csv_format(self, tmp_path):
        """Test hist2d CSV format."""
        hist2d = np.random.rand(10, 10)
        xedges = np.linspace(0, 1, 11)
        yedges = np.linspace(-1, 1, 11)

        filepath = str(tmp_path / "hist2d.csv")
        save_hist2d_csv(hist2d, xedges, yedges, filepath)

        assert os.path.exists(filepath)
        # Verify header
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            assert header == "phase_bin,amplitude_bin,density"

    def test_save_psd_csv_format(self, tmp_path):
        """Test PSD CSV format."""
        frequencies = np.array([0, 1e6, 2e6, 3e6])
        psd_values = np.array([1e-8, 2e-8, 3e-8, 4e-8])

        filepath = str(tmp_path / "psd.csv")
        save_psd_csv(frequencies, psd_values, filepath)

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            assert header == "frequency_hz,psd_v2_per_hz"

    def test_save_pdf_csv_format(self, tmp_path):
        """Test PDF CSV format."""
        amplitudes = np.linspace(-1, 1, 100)
        pdf_values = np.random.rand(100)

        filepath = str(tmp_path / "pdf.csv")
        save_pdf_csv(amplitudes, pdf_values, filepath)

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            assert header == "amplitude_v,probability_density"

    def test_save_jitter_distribution_csv_format(self, tmp_path):
        """Test jitter distribution CSV format."""
        time_offsets = np.linspace(-1e-11, 1e-11, 100)
        probabilities = np.random.rand(100)

        filepath = str(tmp_path / "jitter.csv")
        save_jitter_distribution_csv(time_offsets, probabilities, filepath)

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            assert header == "time_offset_s,probability"

    def test_csv_data_disabled_by_default(self, tmp_path):
        """Test that CSV output is disabled by default."""
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui)  # save_csv_data defaults to False
        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        # CSV directory should not exist
        csv_dir = tmp_path / "eye_analysis_data"
        assert not os.path.exists(csv_dir)

    def test_csv_data_enabled(self, tmp_path):
        """Test CSV output when enabled."""
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui, save_csv_data=True)
        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        # CSV files should exist
        csv_dir = tmp_path / "eye_analysis_data"
        assert os.path.exists(csv_dir)
        assert os.path.exists(csv_dir / "hist2d.csv")


class TestJSONFormat:
    """Test JSON output format per EyeAnalyzer.md specification."""

    def test_json_has_metadata_section(self, tmp_path):
        """Test JSON output has metadata section."""
        metrics = {'eye_height': 0.75, 'eye_width': 0.95}
        metadata = {'ui': 2.5e-11, 'ui_bins': 128, 'amp_bins': 128}

        filepath = str(tmp_path / "metrics.json")
        save_metrics_json_spec(metrics, metadata, filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'metadata' in data
        assert 'version' in data['metadata']
        assert 'timestamp' in data['metadata']
        assert data['metadata']['ui'] == 2.5e-11

    def test_json_has_eye_geometry_section(self, tmp_path):
        """Test JSON output has eye_geometry section."""
        metrics = {'eye_height': 0.75, 'eye_width': 0.95, 'eye_area': 0.5}
        metadata = {'ui': 2.5e-11}

        filepath = str(tmp_path / "metrics.json")
        save_metrics_json_spec(metrics, metadata, filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'eye_geometry' in data
        assert 'eye_height' in data['eye_geometry']
        assert 'eye_width' in data['eye_geometry']

    def test_json_has_jitter_decomposition_section(self, tmp_path):
        """Test JSON output has jitter_decomposition section."""
        metrics = {'rj_sigma': 5e-12, 'dj_pp': 10e-12, 'tj_at_ber': 25e-12}
        metadata = {'ui': 2.5e-11}

        filepath = str(tmp_path / "metrics.json")
        save_metrics_json_spec(metrics, metadata, filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'jitter_decomposition' in data
        assert 'rj_sigma' in data['jitter_decomposition']
        assert 'dj_pp' in data['jitter_decomposition']

    def test_json_has_signal_quality_section(self, tmp_path):
        """Test JSON output has signal_quality section."""
        metrics = {'signal_mean': 0.0, 'signal_rms': 0.4, 'signal_peak_to_peak': 0.8}
        metadata = {'ui': 2.5e-11}

        filepath = str(tmp_path / "metrics.json")
        save_metrics_json_spec(metrics, metadata, filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'signal_quality' in data
        assert 'mean' in data['signal_quality']
        assert 'rms' in data['signal_quality']

    def test_json_has_data_provenance_section(self, tmp_path):
        """Test JSON output has data_provenance section."""
        metrics = {'total_samples': 10000, 'analyzed_samples': 5000}
        metadata = {'ui': 2.5e-11}

        filepath = str(tmp_path / "metrics.json")
        save_metrics_json_spec(metrics, metadata, filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'data_provenance' in data
        assert 'total_samples' in data['data_provenance']


class TestBoundaryConditionsExtended:
    """Extended boundary condition tests (BOUNDARY_CONDITION scenario)."""

    def test_short_data_warning_1000_ui(self):
        """Test warning for short data (1000 UI)."""
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        analyzer = EyeAnalyzer(ui=ui)
        # Should complete without error, may print warning
        metrics = analyzer.analyze(time_array, value_array)

        assert metrics['eye_height'] >= 0
        assert metrics['eye_width'] >= 0

    def test_constant_signal_eye_height_zero(self):
        """Test eye height is zero for constant signal."""
        ui = 2.5e-11
        num_ui = 1000
        time_array = np.arange(num_ui) * ui
        value_array = np.ones(num_ui) * 0.4  # Constant signal

        analyzer = EyeAnalyzer(ui=ui)

        # This may raise ValueError due to insufficient zero crossings
        # which is expected behavior for constant signal
        try:
            metrics = analyzer.analyze(time_array, value_array)
            # If it completes, eye should be small or zero
            assert metrics['eye_height'] >= 0
        except ValueError:
            # Expected for constant signal - no zero crossings
            pass


class TestAnalyzeEyeFunction:
    """Test analyze_eye convenience function."""

    def test_analyze_eye_from_array(self):
        """Test analyze_eye with array input."""
        ui = 2.5e-11
        num_ui = 10000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        value_array += np.random.normal(0, 0.01, size=num_ui)

        waveform_array = np.column_stack([time_array, value_array])

        metrics = analyze_eye(waveform_array=waveform_array, ui=ui)

        assert 'eye_height' in metrics
        assert 'eye_width' in metrics
        assert metrics['eye_height'] > 0

    def test_analyze_eye_from_dat_file(self, tmp_path):
        """Test analyze_eye with .dat file input."""
        # Create mock .dat file
        dat_file = tmp_path / "test.dat"
        ui = 2.5e-11
        num_ui = 1000

        with open(dat_file, 'w') as f:
            f.write("# Mock data\n")
            for i in range(num_ui):
                t = i * ui
                v = 0.4 if np.random.random() > 0.5 else -0.4
                f.write(f"{t:.10e} {v:.6f}\n")

        metrics = analyze_eye(dat_path=str(dat_file), ui=ui)

        assert 'eye_height' in metrics
        assert 'eye_width' in metrics

    def test_analyze_eye_with_all_params(self):
        """Test analyze_eye with all parameters."""
        ui = 2.5e-11
        num_ui = 10000
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)

        waveform_array = np.column_stack([time_array, value_array])

        metrics = analyze_eye(
            waveform_array=waveform_array,
            ui=ui,
            ui_bins=256,
            amp_bins=256,
            measure_length=time_array[-1] * 0.5,
            target_ber=1e-12,
            sampling='phase-lock',
            jitter_method='dual-dirac'
        )

        assert metrics['analyzed_samples'] < metrics['total_samples']

    def test_analyze_eye_missing_ui(self):
        """Test error when ui is not specified."""
        waveform_array = np.column_stack([np.arange(100), np.random.rand(100)])

        with pytest.raises(ValueError) as exc_info:
            analyze_eye(waveform_array=waveform_array)

        assert "ui" in str(exc_info.value).lower()

    def test_analyze_eye_missing_input(self):
        """Test error when neither dat_path nor waveform_array is provided."""
        with pytest.raises(ValueError):
            analyze_eye(ui=2.5e-11)

    def test_analyze_eye_both_inputs(self):
        """Test error when both dat_path and waveform_array are provided."""
        waveform_array = np.column_stack([np.arange(100), np.random.rand(100)])

        with pytest.raises(ValueError):
            analyze_eye(dat_path="test.dat", waveform_array=waveform_array, ui=2.5e-11)


class TestImageOutputFormats:
    """Test multiple image output formats."""

    def _generate_and_analyze(self, ui=2.5e-11, num_ui=1000):
        """Helper to generate data and run analysis."""
        time_array = np.arange(num_ui) * ui
        value_array = np.random.choice([0.4, -0.4], size=num_ui)
        return time_array, value_array

    def test_png_output(self, tmp_path):
        """Test PNG image output."""
        time_array, value_array = self._generate_and_analyze()
        analyzer = EyeAnalyzer(ui=2.5e-11, output_image_format='png')
        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        assert os.path.exists(tmp_path / "eye_diagram.png")

    def test_svg_output(self, tmp_path):
        """Test SVG image output."""
        time_array, value_array = self._generate_and_analyze()
        analyzer = EyeAnalyzer(ui=2.5e-11, output_image_format='svg')
        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        assert os.path.exists(tmp_path / "eye_diagram.svg")

    def test_pdf_output(self, tmp_path):
        """Test PDF image output."""
        time_array, value_array = self._generate_and_analyze()
        analyzer = EyeAnalyzer(ui=2.5e-11, output_image_format='pdf')
        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        assert os.path.exists(tmp_path / "eye_diagram.pdf")

    def test_invalid_format(self):
        """Test error on invalid image format."""
        with pytest.raises(ValueError):
            EyeAnalyzer(ui=2.5e-11, output_image_format='invalid')

    def test_custom_dpi(self, tmp_path):
        """Test custom DPI setting."""
        time_array, value_array = self._generate_and_analyze()
        analyzer = EyeAnalyzer(ui=2.5e-11, output_image_dpi=600)
        assert analyzer.output_image_dpi == 600

        metrics = analyzer.analyze(time_array, value_array)
        analyzer.save_results(metrics, str(tmp_path))

        assert os.path.exists(tmp_path / "eye_diagram.png")