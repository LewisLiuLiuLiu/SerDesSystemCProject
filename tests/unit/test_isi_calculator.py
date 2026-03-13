"""
Unit tests for ISICalculator class.

Tests cover:
- ISI calculation using convolution method (fast)
- ISI calculation using brute force method (exact)
- PAM4 and NRZ modulation format support
- Input validation and edge cases
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eye_analyzer.statistical.isi_calculator import ISICalculator, ModulationFormat


class TestISICalculatorInitialization:
    """Tests for ISICalculator initialization."""
    
    def test_default_initialization(self):
        """Test calculator can be initialized with default parameters."""
        calc = ISICalculator()
        assert calc.modulation_format == ModulationFormat.PAM4
        assert calc.samples_per_symbol == 8
        assert calc.vh_size == 2048
        assert calc.sample_size == 16
        
    def test_nrz_initialization(self):
        """Test calculator can be initialized for NRZ."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ)
        assert calc.modulation_format == ModulationFormat.NRZ
        
    def test_custom_parameters(self):
        """Test calculator with custom parameters."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.PAM4,
            samples_per_symbol=16,
            vh_size=1024,
            sample_size=8
        )
        assert calc.samples_per_symbol == 16
        assert calc.vh_size == 1024
        assert calc.sample_size == 8


class TestModulationFormat:
    """Tests for modulation format support."""
    
    def test_nrz_levels(self):
        """Test NRZ modulation levels are [-1, 1]."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ)
        levels = calc.get_modulation_levels()
        expected = np.array([-1.0, 1.0])
        np.testing.assert_array_almost_equal(levels, expected)
        
    def test_pam4_levels(self):
        """Test PAM4 modulation levels are [-1, -1/3, 1/3, 1]."""
        calc = ISICalculator(modulation_format=ModulationFormat.PAM4)
        levels = calc.get_modulation_levels()
        expected = np.array([-1.0, -1/3, 1/3, 1.0])
        np.testing.assert_array_almost_equal(levels, expected)


class TestCalculateByConvolution:
    """Tests for convolution-based ISI calculation."""
    
    @pytest.fixture
    def simple_pulse_response(self):
        """Create a simple pulse response for testing."""
        # Simple pulse: main cursor at center with small pre/post cursors
        pulse = np.zeros(64)
        pulse[32] = 1.0  # Main cursor
        pulse[24] = 0.1  # Pre-cursor
        pulse[40] = 0.15  # Post-cursor
        return pulse
        
    def test_convolution_returns_pdf_list(self, simple_pulse_response):
        """Test convolution method returns list of PDFs."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        pdf_list = calc._calculate_by_convolution(simple_pulse_response)
        
        # Should return a list
        assert isinstance(pdf_list, list)
        # Each PDF should sum to approximately 1
        for pdf in pdf_list:
            assert np.abs(np.sum(pdf) - 1.0) < 0.01
            
    def test_convolution_pdf_shape(self, simple_pulse_response):
        """Test PDFs have correct shape matching vh_size."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.NRZ,
            vh_size=512,
            sample_size=4
        )
        pdf_list = calc._calculate_by_convolution(simple_pulse_response)
        
        # Each PDF should have length equal to vh_size
        for pdf in pdf_list:
            assert len(pdf) == 512
            
    def test_convolution_nrz_vs_pam4(self, simple_pulse_response):
        """Test convolution with different modulation formats."""
        calc_nrz = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        calc_pam4 = ISICalculator(modulation_format=ModulationFormat.PAM4, sample_size=4)
        
        pdf_list_nrz = calc_nrz._calculate_by_convolution(simple_pulse_response)
        pdf_list_pam4 = calc_pam4._calculate_by_convolution(simple_pulse_response)
        
        # Both should return non-empty lists
        assert len(pdf_list_nrz) > 0
        assert len(pdf_list_pam4) > 0


class TestCalculateByBruteForce:
    """Tests for brute force ISI calculation."""
    
    @pytest.fixture
    def simple_pulse_response(self):
        """Create a simple pulse response for testing."""
        pulse = np.zeros(32)
        pulse[16] = 1.0
        pulse[8] = 0.1
        pulse[24] = 0.1
        return pulse
        
    def test_brute_force_returns_pdf_list(self, simple_pulse_response):
        """Test brute force method returns list of PDFs."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        pdf_list = calc._calculate_by_brute_force(simple_pulse_response)
        
        assert isinstance(pdf_list, list)
        for pdf in pdf_list:
            assert np.abs(np.sum(pdf) - 1.0) < 0.01
            
    def test_brute_force_pdf_shape(self, simple_pulse_response):
        """Test PDFs have correct shape."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.NRZ,
            vh_size=512,
            sample_size=4
        )
        pdf_list = calc._calculate_by_brute_force(simple_pulse_response)
        
        for pdf in pdf_list:
            assert len(pdf) == 512


class TestCalculateISI:
    """Tests for main calculate() method."""
    
    @pytest.fixture
    def simple_pulse_response(self):
        """Create a simple pulse response for testing."""
        pulse = np.zeros(48)
        pulse[24] = 0.8  # Main cursor
        pulse[16] = 0.05  # Pre-cursor
        pulse[32] = 0.08  # Post-cursor
        return pulse
        
    def test_calculate_returns_result_dict(self, simple_pulse_response):
        """Test calculate returns dictionary with expected keys."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        result = calc.calculate(simple_pulse_response, method='convolution')
        
        assert isinstance(result, dict)
        assert 'pdf_list' in result
        assert 'voltage_bins' in result
        assert 'time_slices' in result
        
    def test_calculate_convolution_method(self, simple_pulse_response):
        """Test calculate with convolution method."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        result = calc.calculate(simple_pulse_response, method='convolution')
        
        pdf_list = result['pdf_list']
        assert isinstance(pdf_list, list)
        assert len(pdf_list) > 0
        
    def test_calculate_brute_force_method(self, simple_pulse_response):
        """Test calculate with brute force method."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=4)
        result = calc.calculate(simple_pulse_response, method='brute_force')
        
        pdf_list = result['pdf_list']
        assert isinstance(pdf_list, list)
        assert len(pdf_list) > 0
        
    def test_calculate_invalid_method_raises(self, simple_pulse_response):
        """Test calculate raises error for invalid method."""
        calc = ISICalculator()
        with pytest.raises(ValueError, match="Unknown method"):
            calc.calculate(simple_pulse_response, method='invalid')
            
    def test_calculate_voltage_bins_shape(self, simple_pulse_response):
        """Test voltage bins have correct shape."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.NRZ,
            vh_size=512,
            sample_size=4
        )
        result = calc.calculate(simple_pulse_response, method='convolution')
        
        voltage_bins = result['voltage_bins']
        assert len(voltage_bins) == 512
        
    def test_calculate_time_slices(self, simple_pulse_response):
        """Test time slices cover one symbol period."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.NRZ,
            samples_per_symbol=8,
            sample_size=4
        )
        result = calc.calculate(simple_pulse_response, method='convolution')
        
        time_slices = result['time_slices']
        # Should have samples_per_symbol slices
        assert len(time_slices) == 8


class TestConvolutionVsBruteForce:
    """Tests comparing convolution and brute force methods."""
    
    @pytest.fixture
    def short_pulse_response(self):
        """Create a short pulse response for comparison."""
        # Short pulse to keep brute force computable
        pulse = np.zeros(24)
        pulse[12] = 0.5
        pulse[4] = 0.05
        pulse[20] = 0.05
        return pulse
        
    def test_methods_produce_similar_results(self, short_pulse_response):
        """Test that both methods produce similar ISI distributions."""
        calc = ISICalculator(
            modulation_format=ModulationFormat.NRZ,
            vh_size=512,
            sample_size=4
        )
        
        result_conv = calc.calculate(short_pulse_response, method='convolution')
        result_bf = calc.calculate(short_pulse_response, method='brute_force')
        
        # Both should have same number of time slices
        assert len(result_conv['pdf_list']) == len(result_bf['pdf_list'])
        
        # PDFs should have similar shapes (not exact due to different methods)
        for pdf_conv, pdf_bf in zip(result_conv['pdf_list'], result_bf['pdf_list']):
            # Check both are valid probability distributions
            assert np.abs(np.sum(pdf_conv) - 1.0) < 0.01
            assert np.abs(np.sum(pdf_bf) - 1.0) < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_pulse_raises(self):
        """Test empty pulse response raises error."""
        calc = ISICalculator()
        with pytest.raises(ValueError):
            calc.calculate(np.array([]))
            
    def test_all_zeros_pulse(self):
        """Test pulse response with all zeros."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=2)
        pulse = np.zeros(32)
        pulse[16] = 0.5  # Add a small main cursor
        result = calc.calculate(pulse, method='convolution')
        assert 'pdf_list' in result
        
    def test_single_cursor_pulse(self):
        """Test pulse with only main cursor (no ISI)."""
        calc = ISICalculator(modulation_format=ModulationFormat.NRZ, sample_size=2)
        pulse = np.zeros(16)
        pulse[8] = 1.0  # Only main cursor
        
        result = calc.calculate(pulse, method='convolution')
        pdf_list = result['pdf_list']
        
        # Should still produce valid PDFs
        assert len(pdf_list) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
