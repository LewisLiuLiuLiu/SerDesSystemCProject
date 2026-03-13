#!/usr/bin/env python3
"""
Tests for JitterTolerance class.

Task 3.5: Jitter Tolerance (JTol) 测试
- SJ (Sinusoidal Jitter) 频率扫描
- SJ 幅度扫描  
- 与标准模板对比
- Pass/Fail 判定
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from eye_analyzer.ber.jtol import JitterTolerance
from eye_analyzer.ber.template import JTolTemplate
from eye_analyzer.schemes.statistical import StatisticalScheme


class TestJitterToleranceInitialization:
    """Tests for JitterTolerance initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with NRZ modulation."""
        jtol = JitterTolerance()
        assert jtol.modulation == 'nrz'
        assert jtol.target_ber == 1e-12
    
    def test_pam4_initialization(self):
        """Test initialization with PAM4 modulation."""
        jtol = JitterTolerance(modulation='pam4', target_ber=1e-15)
        assert jtol.modulation == 'pam4'
        assert jtol.target_ber == 1e-15
    
    def test_invalid_modulation_raises_error(self):
        """Test invalid modulation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid modulation"):
            JitterTolerance(modulation='invalid')


class TestJitterToleranceTemplateComparison:
    """Tests for template comparison functionality."""
    
    def setup_method(self):
        self.jtol = JitterTolerance()
    
    def test_compare_with_template_basic(self):
        """Test basic template comparison."""
        # Create mock measured data
        frequencies = np.array([1e5, 1e6, 1e7])
        measured_sj = np.array([0.08, 0.07, 0.05])
        
        result = self.jtol.compare_with_template(
            frequencies, measured_sj, template_name='ieee_802_3ck'
        )
        
        assert 'frequencies' in result
        assert 'measured_sj' in result
        assert 'template_limits' in result
        assert 'margins' in result
        assert 'pass_fail' in result
        assert 'overall_pass' in result
    
    def test_compare_with_template_margins(self):
        """Test margin calculation."""
        frequencies = np.array([1e5, 1e6])
        # Measured higher than template = pass
        measured_sj = np.array([0.15, 0.15])  # Above 0.1 UI template limit
        
        result = self.jtol.compare_with_template(
            frequencies, measured_sj, template_name='ieee_802_3ck'
        )
        
        assert len(result['margins']) == len(frequencies)
        # Margins should be positive when measured > template
        assert all(m > 0 for m in result['margins'])
        assert result['overall_pass'] is True
    
    def test_compare_with_template_fail_case(self):
        """Test fail case when measured < template."""
        frequencies = np.array([1e5, 1e6])
        # Measured lower than template = fail
        measured_sj = np.array([0.01, 0.01])  # Below 0.1 UI template limit
        
        result = self.jtol.compare_with_template(
            frequencies, measured_sj, template_name='ieee_802_3ck'
        )
        
        # Should have negative margins
        assert any(m < 0 for m in result['margins'])
        assert result['overall_pass'] is False
    
    def test_compare_invalid_template(self):
        """Test comparison with invalid template raises error."""
        with pytest.raises(ValueError, match="Unknown template"):
            self.jtol.compare_with_template(
                np.array([1e5]), np.array([0.1]), template_name='invalid'
            )
    
    def test_compare_mismatched_array_lengths(self):
        """Test comparison with mismatched array lengths raises error."""
        with pytest.raises(ValueError, match="same length"):
            self.jtol.compare_with_template(
                np.array([1e5, 1e6]), np.array([0.1]), template_name='ieee_802_3ck'
            )


class TestJitterToleranceSJSearch:
    """Tests for SJ amplitude search functionality."""
    
    def setup_method(self):
        self.jtol = JitterTolerance()
    
    def test_find_sj_limit_binary_search(self):
        """Test binary search for SJ limit."""
        # Create a simple mock eye analyzer that returns eye_height based on SJ
        class MockEyeAnalyzer:
            def __init__(self, max_sj=0.15):
                self.max_sj = max_sj
            
            def analyze(self, pulse_response, noise_sigma=0.0, dj=0.0, rj=0.0, target_ber=1e-12):
                # Eye closes when SJ (dj) exceeds max_sj
                eye_open = 1.0 if dj < self.max_sj else 0.0
                return {'eye_height': eye_open}
        
        mock_analyzer = MockEyeAnalyzer(max_sj=0.1)
        
        sj_limit = self.jtol._find_sj_limit_at_frequency(
            eye_analyzer=mock_analyzer,
            frequency=1e6,
            pulse_response=np.array([1.0, 0.5, 0.2]),
            min_sj=0.001,
            max_sj=0.2,
            tolerance=0.01
        )
        
        # Should be close to 0.1 UI (the max_sj threshold)
        assert 0.08 <= sj_limit <= 0.12


class TestJitterToleranceMeasureJTol:
    """Tests for full JTol measurement."""
    
    def setup_method(self):
        self.jtol = JitterTolerance()
        # Create simple pulse response
        self.pulse_response = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    
    def test_measure_jtol_structure(self):
        """Test measure_jtol returns correct structure."""
        # Use statistical scheme for testing
        scheme = StatisticalScheme(ui=1e-9, modulation='nrz')
        
        sj_frequencies = np.array([1e5, 1e6])
        
        result = self.jtol.measure_jtol(
            eye_analyzer=scheme,
            sj_frequencies=sj_frequencies,
            template='ieee_802_3ck',
            pulse_response=self.pulse_response
        )
        
        assert 'frequencies' in result
        assert 'sj_limits' in result
        assert 'template_limits' in result
        assert 'margins' in result
        assert 'pass_fail' in result
        assert 'overall_pass' in result
    
    def test_measure_jtol_frequency_count(self):
        """Test measure_jtol returns correct number of frequency points."""
        scheme = StatisticalScheme(ui=1e-9, modulation='nrz')
        sj_frequencies = np.array([1e5, 1e6, 1e7])
        
        result = self.jtol.measure_jtol(
            eye_analyzer=scheme,
            sj_frequencies=sj_frequencies,
            template='ieee_802_3ck',
            pulse_response=self.pulse_response
        )
        
        assert len(result['frequencies']) == len(sj_frequencies)
        assert len(result['sj_limits']) == len(sj_frequencies)
        assert len(result['template_limits']) == len(sj_frequencies)
        assert len(result['margins']) == len(sj_frequencies)
        assert len(result['pass_fail']) == len(sj_frequencies)


class TestJitterTolerancePlot:
    """Tests for JTol plotting functionality."""
    
    def setup_method(self):
        self.jtol = JitterTolerance()
    
    def test_plot_jtol_structure(self):
        """Test plot_jtol returns correct structure."""
        # Create mock results
        results = {
            'frequencies': np.array([1e5, 1e6, 1e7]),
            'sj_limits': np.array([0.1, 0.08, 0.05]),
            'template_limits': np.array([0.1, 0.1, 0.1]),
            'margins': np.array([0.0, -0.02, -0.05]),
            'pass_fail': [True, False, False],
            'overall_pass': False
        }
        
        fig, ax = self.jtol.plot_jtol(results)
        
        assert fig is not None
        assert ax is not None
    
    def test_plot_jtol_with_output_file(self, tmp_path):
        """Test plot_jtol saves to file."""
        results = {
            'frequencies': np.array([1e5, 1e6]),
            'sj_limits': np.array([0.1, 0.08]),
            'template_limits': np.array([0.1, 0.1]),
            'margins': np.array([0.0, -0.02]),
            'pass_fail': [True, False],
            'overall_pass': False
        }
        
        output_file = tmp_path / "jtol_plot.png"
        fig, ax = self.jtol.plot_jtol(results, output_file=str(output_file))
        
        assert output_file.exists()


class TestJitterToleranceIntegration:
    """Integration tests for JitterTolerance."""
    
    def test_full_jtol_workflow(self):
        """Test complete JTol workflow."""
        jtol = JitterTolerance(modulation='nrz', target_ber=1e-12)
        
        # Create simple pulse response
        pulse_response = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
        
        # Use statistical scheme
        scheme = StatisticalScheme(ui=1e-9, modulation='nrz')
        
        # Define test frequencies
        sj_frequencies = np.array([1e5, 5e5, 1e6])
        
        # Measure JTol
        result = jtol.measure_jtol(
            eye_analyzer=scheme,
            sj_frequencies=sj_frequencies,
            template='ieee_802_3ck',
            pulse_response=pulse_response
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert len(result['frequencies']) == len(sj_frequencies)
        assert isinstance(result['overall_pass'], bool)
        
        # All SJ limits should be positive
        assert all(sj > 0 for sj in result['sj_limits'])
        
        # Plot should work
        fig, ax = jtol.plot_jtol(result)
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
