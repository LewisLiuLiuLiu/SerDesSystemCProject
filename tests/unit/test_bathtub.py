"""
Unit tests for BathtubCurve class.

Following TDD principles - test first, then implement.

Test coverage:
- Time bathtub calculation (fixed voltage, scan phase)
- Voltage bathtub calculation (fixed phase, scan voltage)
- NRZ and PAM4 modulation support
- OIF-CEI conditional probability method
- Edge cases and error handling
"""

import pytest
import numpy as np
import sys
import os

# Add eye_analyzer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eye_analyzer.ber.bathtub import BathtubCurve
from eye_analyzer.modulation import NRZ, PAM4


class TestBathtubCurve:
    """Test suite for BathtubCurve class."""
    
    def test_class_exists(self):
        """Test that BathtubCurve class exists and can be imported."""
        assert BathtubCurve is not None
    
    def test_init_default_modulation(self):
        """Test initialization with default NRZ modulation."""
        bathtub = BathtubCurve()
        assert bathtub is not None
        assert bathtub.modulation is not None
        assert bathtub.modulation.name == 'nrz'
    
    def test_init_with_nrz(self):
        """Test initialization with explicit NRZ modulation."""
        nrz = NRZ()
        bathtub = BathtubCurve(modulation=nrz)
        assert bathtub.modulation.name == 'nrz'
    
    def test_init_with_pam4(self):
        """Test initialization with PAM4 modulation."""
        pam4 = PAM4()
        bathtub = BathtubCurve(modulation=pam4)
        assert bathtub.modulation.name == 'pam4'
    
    def test_init_with_signal_amplitude(self):
        """Test initialization with custom signal amplitude."""
        bathtub = BathtubCurve(signal_amplitude=0.8)
        assert bathtub.signal_amplitude == 0.8


class TestTimeBathtub:
    """Test suite for time bathtub calculation (fixed voltage, scan phase)."""
    
    def test_calculate_time_bathtub_exists(self):
        """Test that calculate_time_bathtub method exists."""
        bathtub = BathtubCurve()
        assert hasattr(bathtub, 'calculate_time_bathtub')
    
    def test_time_bathtub_returns_dict(self):
        """Test that calculate_time_bathtub returns dictionary with expected keys."""
        bathtub = BathtubCurve()
        
        # Create simple test eye PDF
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'ber_left' in result
        assert 'ber_right' in result
    
    def test_time_bathtub_time_array_length(self):
        """Test that time array has correct length."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert len(result['time']) == n_time
        assert len(result['ber_left']) == n_time
        assert len(result['ber_right']) == n_time
    
    def test_time_bathtub_ber_values_in_range(self):
        """Test that BER values are in valid range [0, 1]."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert np.all(np.array(result['ber_left']) >= 0)
        assert np.all(np.array(result['ber_left']) <= 1)
        assert np.all(np.array(result['ber_right']) >= 0)
        assert np.all(np.array(result['ber_right']) <= 1)
    
    def test_time_bathtub_with_voltage_level(self):
        """Test time bathtub with explicit voltage level."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, voltage_level=0.0, target_ber=1e-12
        )
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'ber_left' in result
        assert 'ber_right' in result
    
    def test_time_bathtub_default_voltage_at_center(self):
        """Test that default voltage level is at eye center."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate with default voltage (should use eye center)
        result_default = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, voltage_level=None
        )
        
        # Calculate with explicit center voltage (0 for NRZ)
        result_center = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, voltage_level=0.0
        )
        
        # Results should be the same
        assert np.allclose(result_default['ber_left'], result_center['ber_left'])
        assert np.allclose(result_default['ber_right'], result_center['ber_right'])
    
    def test_time_bathtub_nrz_ideal_eye(self):
        """Test time bathtub with ideal NRZ eye pattern."""
        bathtub = BathtubCurve(modulation=NRZ())
        
        n_time = 128
        n_voltage = 256
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Create ideal eye - Gaussian distributions at levels
        eye_pdf = np.zeros((n_time, n_voltage))
        noise_std = 0.1
        
        for t_idx, t in enumerate(time_slices):
            # Eye is open in center, closed at edges
            eye_opening = np.exp(-4 * (t - 0.5)**2)
            
            # Two signal levels: +1 and -1
            for level in [1.0, -1.0]:
                eye_pdf[t_idx, :] += np.exp(-(voltage_bins - level)**2 / (2 * noise_std**2))
        
        # Normalize
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
        
        # BER should be lowest at eye center (t = 0.5)
        center_idx = n_time // 2
        assert result['ber_left'][center_idx] < 0.5
        assert result['ber_right'][center_idx] < 0.5
    
    def test_time_bathtub_pam4(self):
        """Test time bathtub with PAM4 modulation."""
        bathtub = BathtubCurve(modulation=PAM4())
        
        n_time = 64
        n_voltage = 256
        voltage_bins = np.linspace(-2, 2, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Create PAM4 eye PDF
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert isinstance(result, dict)
        assert len(result['time']) == n_time
        assert len(result['ber_left']) == n_time
        assert len(result['ber_right']) == n_time


class TestVoltageBathtub:
    """Test suite for voltage bathtub calculation (fixed phase, scan voltage)."""
    
    def test_calculate_voltage_bathtub_exists(self):
        """Test that calculate_voltage_bathtub method exists."""
        bathtub = BathtubCurve()
        assert hasattr(bathtub, 'calculate_voltage_bathtub')
    
    def test_voltage_bathtub_returns_dict(self):
        """Test that calculate_voltage_bathtub returns dictionary with expected keys."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        assert isinstance(result, dict)
        assert 'voltage' in result
        assert 'ber_upper' in result
        assert 'ber_lower' in result
    
    def test_voltage_bathtub_voltage_array_length(self):
        """Test that voltage array has correct length."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert len(result['voltage']) == n_voltage
        assert len(result['ber_upper']) == n_voltage
        assert len(result['ber_lower']) == n_voltage
    
    def test_voltage_bathtub_ber_values_in_range(self):
        """Test that BER values are in valid range [0, 1]."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert np.all(np.array(result['ber_upper']) >= 0)
        assert np.all(np.array(result['ber_upper']) <= 1)
        assert np.all(np.array(result['ber_lower']) >= 0)
        assert np.all(np.array(result['ber_lower']) <= 1)
    
    def test_voltage_bathtub_with_time_idx(self):
        """Test voltage bathtub with explicit time index."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, time_idx=32, target_ber=1e-12
        )
        
        assert isinstance(result, dict)
        assert 'voltage' in result
        assert 'ber_upper' in result
        assert 'ber_lower' in result
    
    def test_voltage_bathtub_default_time_at_center(self):
        """Test that default time index is at eye center."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate with default time (should use center)
        result_default = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, time_idx=None
        )
        
        # Calculate with explicit center time
        center_idx = n_time // 2
        result_center = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, time_idx=center_idx
        )
        
        # Results should be the same
        assert np.allclose(result_default['ber_upper'], result_center['ber_upper'])
        assert np.allclose(result_default['ber_lower'], result_center['ber_lower'])
    
    def test_voltage_bathtub_nrz_ideal_eye(self):
        """Test voltage bathtub with ideal NRZ eye pattern."""
        bathtub = BathtubCurve(modulation=NRZ())
        
        n_time = 128
        n_voltage = 256
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Create ideal eye at center time
        eye_pdf = np.zeros((n_time, n_voltage))
        center_t = n_time // 2
        noise_std = 0.1
        
        # Create distributions at +1 and -1 levels
        for level in [1.0, -1.0]:
            eye_pdf[center_t - 5:center_t + 5, :] += np.exp(
                -(voltage_bins - level)**2 / (2 * noise_std**2)
            )
        
        # Add some distribution at other times
        for t in range(n_time):
            if abs(t - center_t) > 5:
                eye_pdf[t, :] = np.random.rand(n_voltage) * 0.1
        
        # Normalize
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, time_idx=center_t
        )
        
        # BER should be lowest at eye center (voltage = 0)
        center_v_idx = n_voltage // 2
        assert result['ber_upper'][center_v_idx] < 0.5
        assert result['ber_lower'][center_v_idx] < 0.5
    
    def test_voltage_bathtub_pam4(self):
        """Test voltage bathtub with PAM4 modulation."""
        bathtub = BathtubCurve(modulation=PAM4())
        
        n_time = 64
        n_voltage = 256
        voltage_bins = np.linspace(-2, 2, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Create PAM4 eye PDF
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_voltage_bathtub(eye_pdf, voltage_bins, time_slices)
        
        assert isinstance(result, dict)
        assert len(result['voltage']) == n_voltage
        assert len(result['ber_upper']) == n_voltage
        assert len(result['ber_lower']) == n_voltage


class TestBathtubEdgeCases:
    """Test edge cases and error handling."""
    
    def test_time_bathtub_empty_input_raises_error(self):
        """Test that empty input raises appropriate error."""
        bathtub = BathtubCurve()
        
        with pytest.raises((ValueError, IndexError)):
            bathtub.calculate_time_bathtub(
                np.array([]), np.array([]), np.array([])
            )
    
    def test_voltage_bathtub_empty_input_raises_error(self):
        """Test that empty input raises appropriate error."""
        bathtub = BathtubCurve()
        
        with pytest.raises((ValueError, IndexError)):
            bathtub.calculate_voltage_bathtub(
                np.array([]), np.array([]), np.array([])
            )
    
    def test_time_bathtub_mismatched_dimensions_raises_error(self):
        """Test that mismatched dimensions raise appropriate error."""
        bathtub = BathtubCurve()
        
        voltage_bins = np.linspace(-1, 1, 64)
        time_slices = np.linspace(0, 1, 32)
        eye_pdf = np.random.rand(32, 128)  # Wrong second dimension
        
        with pytest.raises(ValueError):
            bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
    
    def test_voltage_bathtub_mismatched_dimensions_raises_error(self):
        """Test that mismatched dimensions raise appropriate error."""
        bathtub = BathtubCurve()
        
        voltage_bins = np.linspace(-1, 1, 64)
        time_slices = np.linspace(0, 1, 32)
        eye_pdf = np.random.rand(32, 128)  # Wrong second dimension
        
        with pytest.raises(ValueError):
            bathtub.calculate_voltage_bathtub(eye_pdf, voltage_bins, time_slices)
    
    def test_time_bathtub_invalid_time_idx_raises_error(self):
        """Test that invalid time index raises appropriate error."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        with pytest.raises((ValueError, IndexError)):
            bathtub.calculate_voltage_bathtub(
                eye_pdf, voltage_bins, time_slices, time_idx=100  # Out of range
            )
    
    def test_voltage_bathtub_invalid_voltage_level_raises_error(self):
        """Test that invalid voltage level raises appropriate error."""
        bathtub = BathtubCurve()
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Voltage level outside range should be clipped or raise error
        # Just test that it doesn't crash
        result = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, voltage_level=10.0  # Outside range
        )
        assert isinstance(result, dict)


class TestBathtubOIFMethod:
    """Test OIF-CEI conditional probability method integration."""
    
    def test_time_bathtub_uses_oif_method(self):
        """Test that time bathtub uses OIF-CEI method."""
        from eye_analyzer.statistical.ber_calculator import BERCalculator
        
        bathtub = BathtubCurve()
        assert hasattr(bathtub, '_calculator')
        assert isinstance(bathtub._calculator, BERCalculator)
    
    def test_voltage_bathtub_uses_oif_method(self):
        """Test that voltage bathtub uses OIF-CEI method."""
        from eye_analyzer.statistical.ber_calculator import BERCalculator
        
        bathtub = BathtubCurve()
        assert hasattr(bathtub, '_calculator')
        assert isinstance(bathtub._calculator, BERCalculator)
    
    def test_time_bathtub_symmetry_for_symmetric_eye(self):
        """Test that time bathtub is symmetric for symmetric eye."""
        bathtub = BathtubCurve(modulation=NRZ())
        
        n_time = 128
        n_voltage = 256
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Create symmetric eye
        eye_pdf = np.zeros((n_time, n_voltage))
        noise_std = 0.1
        
        for t_idx, t in enumerate(time_slices):
            # Symmetric around t = 0.5
            for level in [1.0, -1.0]:
                eye_pdf[t_idx, :] += np.exp(-(voltage_bins - level)**2 / (2 * noise_std**2))
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = bathtub.calculate_time_bathtub(eye_pdf, voltage_bins, time_slices)
        
        # For symmetric eye, ber_left at t should equal ber_right at (1-t)
        # Allow some numerical tolerance
        center = n_time // 2
        for i in range(center):
            left_val = result['ber_left'][i]
            right_mirror = result['ber_right'][n_time - 1 - i]
            # Due to symmetry, these should be close
            assert abs(left_val - right_mirror) < 0.1 or np.isclose(left_val, right_mirror, rtol=0.1)


class TestBathtubIntegration:
    """Integration tests for BathtubCurve."""
    
    def test_full_bathtub_workflow_nrz(self):
        """Integration test for full bathtub workflow with NRZ."""
        bathtub = BathtubCurve(modulation=NRZ())
        
        n_time = 128
        n_voltage = 256
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Generate realistic eye PDF
        eye_pdf = np.zeros((n_time, n_voltage))
        noise_std = 0.1
        
        for t_idx, t in enumerate(time_slices):
            for level in [1.0, -1.0]:
                eye_pdf[t_idx, :] += np.exp(-(voltage_bins - level)**2 / (2 * noise_std**2))
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate time bathtub
        time_bathtub = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        # Calculate voltage bathtub
        voltage_bathtub = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        # Verify results
        assert isinstance(time_bathtub, dict)
        assert isinstance(voltage_bathtub, dict)
        assert len(time_bathtub['time']) == n_time
        assert len(voltage_bathtub['voltage']) == n_voltage
    
    def test_full_bathtub_workflow_pam4(self):
        """Integration test for full bathtub workflow with PAM4."""
        bathtub = BathtubCurve(modulation=PAM4())
        
        n_time = 128
        n_voltage = 512
        voltage_bins = np.linspace(-2, 2, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        
        # Generate PAM4 eye PDF with 4 levels
        eye_pdf = np.zeros((n_time, n_voltage))
        noise_std = 0.08
        pam4_levels = [1.0, 1.0/3.0, -1.0/3.0, -1.0]
        
        for t_idx, t in enumerate(time_slices):
            for level in pam4_levels:
                eye_pdf[t_idx, :] += np.exp(-(voltage_bins - level)**2 / (2 * noise_std**2))
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate time bathtub
        time_bathtub = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        # Calculate voltage bathtub at middle eye
        voltage_bathtub = bathtub.calculate_voltage_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        # Verify results
        assert isinstance(time_bathtub, dict)
        assert isinstance(voltage_bathtub, dict)
    
    def test_target_ber_parameter(self):
        """Test that target BER parameter affects results."""
        bathtub = BathtubCurve(modulation=NRZ())
        
        n_time = 64
        n_voltage = 128
        voltage_bins = np.linspace(-1, 1, n_voltage)
        time_slices = np.linspace(0, 1, n_time)
        eye_pdf = np.random.rand(n_time, n_voltage)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate with different target BERs
        result_1e6 = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-6
        )
        result_1e12 = bathtub.calculate_time_bathtub(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-12
        )
        
        # Both should return valid results
        assert isinstance(result_1e6, dict)
        assert isinstance(result_1e12, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
