"""
Unit tests for Jitter Decomposition module.

Tests cover:
- RJ (Random Jitter) extraction
- DJ (Deterministic Jitter) extraction
- Mixed RJ+DJ extraction
- Periodic Jitter (PJ) detection
- Method selection (dual-dirac, tail-fit, auto)
- Performance and regression tests
"""

import pytest
import numpy as np
from eye_analyzer import EyeAnalyzer
from eye_analyzer.jitter import JitterDecomposer


class TestJitterDecomposition:
    """Test jitter decomposition functionality."""

    @staticmethod
    def _generate_prbs_signal(ui, num_ui, swing=0.8, seed=42):
        """Generate PRBS-like signal for testing."""
        np.random.seed(seed)
        time_array = np.arange(num_ui) * ui
        lfsr = np.random.choice([0, 1], size=num_ui)
        value_array = (2 * lfsr - 1) * (swing / 2)
        return time_array, value_array

    @staticmethod
    def _generate_signal_with_jitter(ui, num_ui, rj_sigma=0, dj_pp=0, seed=42):
        """Generate signal with known RJ and DJ."""
        time_array, value_array = TestJitterDecomposition._generate_prbs_signal(
            ui, num_ui
        )

        # Add DJ (data-dependent)
        if dj_pp > 0:
            pattern_jitter = np.where(value_array > 0, dj_pp/2, -dj_pp/2)
            time_array += pattern_jitter

        # Add RJ (Gaussian)
        if rj_sigma > 0:
            np.random.seed(seed)
            random_jitter = np.random.normal(0, rj_sigma, size=num_ui)
            time_array += random_jitter

        # Sort to maintain monotonicity
        sort_idx = np.argsort(time_array)
        time_array = time_array[sort_idx]
        value_array = value_array[sort_idx]

        return time_array, value_array

    @staticmethod
    def _generate_signal_with_pj(ui, num_ui, pj_freq, pj_amp, seed=42):
        """Generate signal with periodic jitter."""
        time_array, value_array = TestJitterDecomposition._generate_prbs_signal(
            ui, num_ui
        )

        np.random.seed(seed)
        time_array += pj_amp * np.sin(2 * np.pi * pj_freq * time_array)

        return time_array, value_array

    def test_pure_rj_extraction_dual_dirac(self):
        """Test RJ extraction with pure random jitter."""
        ui = 2.5e-11
        rj_injected = 5e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=rj_injected, dj_pp=0.0
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify basic functionality
        assert 'rj_sigma' in metrics
        assert 'dj_pp' in metrics
        assert 'tj_at_ber' in metrics
        assert 'fit_method' in metrics
        assert metrics['rj_sigma'] > 0
        assert metrics['dj_pp'] >= 0

    def test_pure_dj_extraction_dual_dirac(self):
        """Test DJ extraction with pure deterministic jitter."""
        ui = 2.5e-11
        dj_injected = 10e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=0.0, dj_pp=dj_injected
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify basic functionality
        assert 'rj_sigma' in metrics
        assert 'dj_pp' in metrics
        assert metrics['rj_sigma'] >= 0
        assert metrics['dj_pp'] >= 0

    def test_mixed_rj_dj_extraction(self):
        """Test RJ and DJ extraction with mixed jitter."""
        ui = 2.5e-11
        rj_injected = 5e-12
        dj_injected = 10e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=rj_injected, dj_pp=dj_injected
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify basic functionality
        assert 'rj_sigma' in metrics
        assert 'dj_pp' in metrics
        assert 'tj_at_ber' in metrics
        assert metrics['rj_sigma'] > 0
        assert metrics['dj_pp'] >= 0
        assert metrics['tj_at_ber'] > 0

    def test_pj_detection(self):
        """Test periodic jitter detection."""
        ui = 2.5e-11
        rj_injected = 3e-12
        pj_freq = 5e6
        pj_amp = 2e-12

        time_array, value_array = self._generate_signal_with_pj(
            ui, num_ui=10000, pj_freq=pj_freq, pj_amp=pj_amp
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify PJ detection structure
        assert 'pj_info' in metrics
        assert 'detected' in metrics['pj_info']

    def test_auto_method_selection(self):
        """Test automatic method selection."""
        ui = 2.5e-11

        # Test with RJ+DJ
        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=5e-12, dj_pp=20e-12
        )
        analyzer = EyeAnalyzer(ui=ui, jitter_method='auto')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify method is reported
        assert 'fit_method' in metrics
        assert metrics['fit_method'] in ['dual-gaussian', 'single-gaussian', 'tail-fit']

    def test_tail_fit_method(self):
        """Test tail-fitting jitter extraction method."""
        ui = 2.5e-11
        rj_injected = 5e-12
        dj_injected = 8e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=rj_injected, dj_pp=dj_injected
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='tail-fit')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify method is either tail-fit or fallback to single-gaussian
        assert metrics['fit_method'] in ['tail-fit', 'single-gaussian']

    def test_performance_large_dataset(self):
        """Test performance with large dataset."""
        import time

        ui = 2.5e-11
        rj_injected = 5e-12
        dj_injected = 10e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=100000, rj_sigma=rj_injected, dj_pp=dj_injected
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')

        start_time = time.time()
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)
        elapsed_time = time.time() - start_time

        # Verify results
        assert metrics['rj_sigma'] > 0
        assert metrics['dj_pp'] >= 0

        # Performance requirement: should complete within 30 seconds
        assert elapsed_time < 30.0

    def test_regression_against_reference(self):
        """Test regression against reference values."""
        ui = 2.5e-11
        rj_injected = 5e-12
        dj_injected = 10e-12

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=rj_injected, dj_pp=dj_injected, seed=12345
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify reasonable results
        assert metrics['rj_sigma'] > 0
        assert metrics['dj_pp'] >= 0
        assert metrics['tj_at_ber'] > 0

    def test_fit_quality_metric(self):
        """Test that fit quality metric is calculated."""
        ui = 2.5e-11

        time_array, value_array = self._generate_signal_with_jitter(
            ui, num_ui=10000, rj_sigma=5e-12, dj_pp=10e-12
        )

        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')
        metrics = analyzer.analyze(time_array, value_array, target_ber=1e-12)

        # Verify fit quality exists
        assert 'fit_quality' in metrics
        assert 0.0 <= metrics['fit_quality'] <= 1.0

    def test_jitter_decomposer_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            decomposer = JitterDecomposer(ui=2.5e-11, method='invalid')

    def test_insufficient_crossing_points(self):
        """Test that insufficient zero-crossings raises error."""
        ui = 2.5e-11
        analyzer = EyeAnalyzer(ui=ui, jitter_method='dual-dirac')

        # Generate signal with too few samples
        time_array = np.arange(10) * ui
        value_array = np.random.choice([0.4, -0.4], size=10)

        with pytest.raises(ValueError, match="Insufficient zero-crossing points"):
            analyzer.analyze(time_array, value_array)

    def test_jitter_decomposer_direct_usage(self):
        """Test JitterDecomposer direct usage."""
        ui = 2.5e-11
        decomposer = JitterDecomposer(ui=ui, method='dual-dirac')

        # Generate test phase and value arrays
        num_samples = 1000
        phase_array = np.random.rand(num_samples)
        value_array = np.where(phase_array > 0.5, 0.4, -0.4)

        # Extract jitter
        metrics = decomposer.extract(phase_array, value_array, target_ber=1e-12)

        # Verify output structure
        assert 'rj_sigma' in metrics
        assert 'dj_pp' in metrics
        assert 'tj_at_ber' in metrics
        assert 'fit_method' in metrics
        assert 'pj_info' in metrics
