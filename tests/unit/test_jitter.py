"""
Unit tests for JitterAnalyzer - Multi-eye jitter extraction for PAM4/NRZ.

TDD Tests for:
- JitterAnalyzer class initialization
- NRZ single eye jitter extraction
- PAM4 multi-eye jitter extraction
- Crossing points extraction
- Dual-Dirac model fitting
- TJ calculation with Q function
"""

import unittest
import numpy as np
import sys
import os

# Add eye_analyzer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from eye_analyzer.jitter import JitterAnalyzer


class TestJitterAnalyzerInit(unittest.TestCase):
    """Test JitterAnalyzer initialization."""
    
    def test_init_nrz_default(self):
        """Test initialization with default NRZ modulation."""
        analyzer = JitterAnalyzer()
        self.assertEqual(analyzer.modulation, 'nrz')
        self.assertEqual(analyzer.signal_amplitude, 1.0)
        
    def test_init_pam4(self):
        """Test initialization with PAM4 modulation."""
        analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=0.8)
        self.assertEqual(analyzer.modulation, 'pam4')
        self.assertEqual(analyzer.signal_amplitude, 0.8)
        
    def test_init_invalid_modulation(self):
        """Test initialization with invalid modulation raises error."""
        with self.assertRaises(ValueError):
            JitterAnalyzer(modulation='invalid')


class TestJitterAnalyzerNRZ(unittest.TestCase):
    """Test JitterAnalyzer with NRZ modulation."""
    
    def setUp(self):
        self.analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
        
    def test_analyze_nrz_returns_dict(self):
        """Test NRZ analyze returns single dict with jitter metrics."""
        # Generate synthetic eye data for NRZ
        np.random.seed(42)
        ui = 1e-9  # 1 ns UI
        num_bits = 1000
        
        # Create synthetic waveform with known jitter
        time = np.linspace(0, num_bits * ui, num_bits * 100)
        # Square wave with transitions and jitter
        signal = np.sign(np.sin(2 * np.pi * time / (2 * ui)) + 
                        0.1 * np.random.randn(len(time)))
        
        result = self.analyzer.analyze(signal, time)
        
        # Verify return type is dict (not list)
        self.assertIsInstance(result, dict)
        
        # Verify required keys
        self.assertIn('rj', result)
        self.assertIn('dj', result)
        self.assertIn('tj', result)
        
    def test_analyze_nrz_jitter_values_positive(self):
        """Test NRZ jitter values are positive."""
        np.random.seed(42)
        ui = 1e-9
        num_bits = 1000
        
        time = np.linspace(0, num_bits * ui, num_bits * 100)
        signal = np.sign(np.sin(2 * np.pi * time / (2 * ui)) + 
                        0.05 * np.random.randn(len(time)))
        
        result = self.analyzer.analyze(signal, time)
        
        self.assertGreaterEqual(result['rj'], 0)
        self.assertGreaterEqual(result['dj'], 0)
        self.assertGreater(result['tj'], 0)


class TestJitterAnalyzerPAM4(unittest.TestCase):
    """Test JitterAnalyzer with PAM4 modulation."""
    
    def setUp(self):
        self.analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=1.0)
        
    def test_analyze_pam4_returns_list(self):
        """Test PAM4 analyze returns list of eye results."""
        np.random.seed(42)
        ui = 1e-9
        num_bits = 1000
        
        time = np.linspace(0, num_bits * ui, num_bits * 100)
        # PAM4-like multi-level signal
        levels = np.random.choice([-3, -1, 1, 3], num_bits)
        signal = np.repeat(levels, 100) + 0.05 * np.random.randn(num_bits * 100)
        
        result = self.analyzer.analyze(signal, time)
        
        # Verify return type is list
        self.assertIsInstance(result, list)
        # PAM4 has 3 eyes
        self.assertEqual(len(result), 3)
        
    def test_analyze_pam4_eye_structure(self):
        """Test PAM4 eye results have correct structure."""
        np.random.seed(42)
        ui = 1e-9
        num_bits = 1000
        
        time = np.linspace(0, num_bits * ui, num_bits * 100)
        levels = np.random.choice([-3, -1, 1, 3], num_bits)
        signal = np.repeat(levels, 100) + 0.05 * np.random.randn(num_bits * 100)
        
        result = self.analyzer.analyze(signal, time)
        
        expected_names = ['lower', 'middle', 'upper']
        for i, eye in enumerate(result):
            self.assertIn('eye_id', eye)
            self.assertIn('eye_name', eye)
            self.assertIn('rj', eye)
            self.assertIn('dj', eye)
            self.assertIn('tj', eye)
            self.assertEqual(eye['eye_id'], i)
            self.assertEqual(eye['eye_name'], expected_names[i])


class TestExtractCrossingPoints(unittest.TestCase):
    """Test crossing points extraction."""
    
    def setUp(self):
        self.analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
        
    def test_extract_crossing_points_returns_array(self):
        """Test extract_crossing_points returns numpy array."""
        # Generate signal with known crossings
        time = np.linspace(0, 10e-9, 1000)  # 10 ns
        signal = np.sin(2 * np.pi * time / 1e-9)  # 1 GHz sine wave
        
        crossings = self.analyzer.extract_crossing_points(signal, threshold=0.0)
        
        self.assertIsInstance(crossings, np.ndarray)
        self.assertGreater(len(crossings), 0)
        
    def test_extract_crossing_points_rising_edge(self):
        """Test extract crossing points for rising edge."""
        time = np.linspace(0, 10e-9, 1000)
        signal = np.sin(2 * np.pi * time / 1e-9)
        
        rising = self.analyzer.extract_crossing_points(signal, threshold=0.0, edge='rising')
        falling = self.analyzer.extract_crossing_points(signal, threshold=0.0, edge='falling')
        
        # Rising and falling should have similar count
        self.assertGreater(len(rising), 0)
        self.assertGreater(len(falling), 0)


class TestFitDualDirac(unittest.TestCase):
    """Test Dual-Dirac model fitting."""
    
    def setUp(self):
        self.analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
        
    def test_fit_dual_dirac_returns_tuple(self):
        """Test fit_dual_dirac returns (rj, dj) tuple."""
        # Generate synthetic crossing points with known RJ
        np.random.seed(42)
        crossing_points = np.random.normal(0, 0.01, 1000)  # 10 ps sigma
        
        rj, dj = self.analyzer.fit_dual_dirac(crossing_points)
        
        self.assertIsInstance(rj, (int, float))
        self.assertIsInstance(dj, (int, float))
        self.assertGreaterEqual(rj, 0)
        self.assertGreaterEqual(dj, 0)
        
    def test_fit_dual_dirac_pure_rj(self):
        """Test fit on pure random jitter (no DJ)."""
        np.random.seed(42)
        # Pure Gaussian, no deterministic component
        crossing_points = np.random.normal(0, 0.01, 1000)
        
        rj, dj = self.analyzer.fit_dual_dirac(crossing_points)
        
        # RJ should be reasonable (positive), DJ should be small for pure RJ
        self.assertGreater(rj, 0)
        self.assertLess(dj, 0.05)  # DJ should be small for pure Gaussian


class TestCalculateTJ(unittest.TestCase):
    """Test TJ calculation with Q function."""
    
    def setUp(self):
        self.analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
        
    def test_calculate_tj_basic(self):
        """Test TJ calculation with known values."""
        rj = 1e-12  # 1 ps
        dj = 2e-12  # 2 ps
        ber = 1e-12
        
        tj = self.analyzer.calculate_tj(rj, dj, ber)
        
        # TJ = DJ + 2 * Q(BER) * RJ
        # Q(1e-12) ≈ 7.03
        expected_tj = dj + 2 * 7.034 * rj
        self.assertAlmostEqual(tj, expected_tj, delta=0.5e-12)
        
    def test_calculate_tj_zero_rj(self):
        """Test TJ calculation with zero RJ."""
        rj = 0
        dj = 2e-12
        ber = 1e-12
        
        tj = self.analyzer.calculate_tj(rj, dj, ber)
        
        # With RJ=0, TJ should equal DJ
        self.assertAlmostEqual(tj, dj)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""
    
    def test_nrz_simple_api(self):
        """Test simple API for NRZ (backward compatible)."""
        analyzer = JitterAnalyzer(modulation='nrz')
        
        # Simple usage pattern
        np.random.seed(42)
        time = np.linspace(0, 1e-6, 10000)
        signal = np.sign(np.sin(2 * np.pi * time / 1e-9)) + 0.01 * np.random.randn(10000)
        
        result = analyzer.analyze(signal, time)
        
        # Should return dict with rj, dj, tj keys
        self.assertIn('rj', result)
        self.assertIn('dj', result)
        self.assertIn('tj', result)


if __name__ == '__main__':
    unittest.main()
