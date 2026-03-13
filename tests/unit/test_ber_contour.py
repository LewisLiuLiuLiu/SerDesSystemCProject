"""
Unit tests for BERContour class.

Following TDD principles - test first, then implement.
"""

import pytest
import numpy as np
import sys
import os

# Add eye_analyzer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eye_analyzer.ber.contour import BERContour
from eye_analyzer.modulation import NRZ, PAM4


class TestBERContour:
    """Test suite for BERContour class."""
    
    def test_class_exists(self):
        """Test that BERContour class exists and can be imported."""
        assert BERContour is not None
    
    def test_init_default_modulation(self):
        """Test initialization with default NRZ modulation."""
        contour = BERContour()
        assert contour is not None
        assert contour.modulation is not None
        assert contour.modulation.name == 'nrz'
    
    def test_init_with_nrz(self):
        """Test initialization with explicit NRZ modulation."""
        nrz = NRZ()
        contour = BERContour(modulation=nrz)
        assert contour.modulation.name == 'nrz'
    
    def test_init_with_pam4(self):
        """Test initialization with PAM4 modulation."""
        pam4 = PAM4()
        contour = BERContour(modulation=pam4)
        assert contour.modulation.name == 'pam4'
    
    def test_calculate_returns_array(self):
        """Test that calculate method returns numpy array."""
        contour = BERContour()
        
        # Create simple test eye PDF
        voltage_bins = np.linspace(-1, 1, 64)
        eye_pdf = np.random.rand(32, 64)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = contour.calculate(eye_pdf, voltage_bins)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == eye_pdf.shape
    
    def test_calculate_nrz_single_target_ber(self):
        """Test calculate with NRZ and single target BER."""
        contour = BERContour(modulation=NRZ())
        
        voltage_bins = np.linspace(-1, 1, 128)
        eye_pdf = np.random.rand(64, 128)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = contour.calculate(eye_pdf, voltage_bins, target_bers=[1e-12])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 128)
        # BER values should be between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_calculate_multiple_target_bers(self):
        """Test calculate with multiple target BER values."""
        contour = BERContour(modulation=NRZ())
        
        voltage_bins = np.linspace(-1, 1, 64)
        eye_pdf = np.random.rand(32, 64)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        target_bers = [1e-12, 1e-9, 1e-6]
        result = contour.calculate(eye_pdf, voltage_bins, target_bers=target_bers)
        
        # Result should be a list/array of contours for each BER level
        assert isinstance(result, (list, np.ndarray))
        assert len(result) == len(target_bers)
    
    def test_calculate_pam4(self):
        """Test calculate with PAM4 modulation."""
        contour = BERContour(modulation=PAM4())
        
        voltage_bins = np.linspace(-2, 2, 256)
        eye_pdf = np.random.rand(128, 256)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = contour.calculate(eye_pdf, voltage_bins, target_bers=[1e-12])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 256)
    
    def test_get_eye_dimensions_returns_dict(self):
        """Test that get_eye_dimensions returns dictionary with eye measurements."""
        contour = BERContour()
        
        # Create sample BER contour
        voltage_bins = np.linspace(-1, 1, 64)
        time_slices = np.linspace(0, 1, 32)
        ber_contour = np.random.rand(32, 64) * 1e-15
        
        result = contour.get_eye_dimensions(ber_contour, voltage_bins, time_slices, target_ber=1e-12)
        
        assert isinstance(result, dict)
        assert 'eye_width_ui' in result or 'eye_width' in result
        assert 'eye_height_v' in result or 'eye_height' in result
    
    def test_get_eye_dimensions_nrz(self):
        """Test eye dimension extraction for NRZ."""
        contour = BERContour(modulation=NRZ())
        
        voltage_bins = np.linspace(-1, 1, 128)
        time_slices = np.linspace(0, 1, 64)
        
        # Create ideal eye pattern - low BER in center, high at edges
        ber_contour = np.ones((64, 128)) * 0.5
        # Create eye opening in center
        ber_contour[20:44, 40:88] = 1e-15
        
        result = contour.get_eye_dimensions(ber_contour, voltage_bins, time_slices, target_ber=1e-12)
        
        assert isinstance(result, dict)
        if 'eye_width_ui' in result:
            assert result['eye_width_ui'] > 0
        if 'eye_height_v' in result:
            assert result['eye_height_v'] > 0
    
    def test_get_eye_dimensions_pam4(self):
        """Test eye dimension extraction for PAM4 (3 eyes)."""
        contour = BERContour(modulation=PAM4())
        
        voltage_bins = np.linspace(-2, 2, 256)
        time_slices = np.linspace(0, 1, 64)
        
        # Create PAM4 eye pattern with 3 eyes
        ber_contour = np.ones((64, 256)) * 0.5
        # Upper eye
        ber_contour[20:44, 160:200] = 1e-15
        # Middle eye
        ber_contour[20:44, 110:145] = 1e-15
        # Lower eye
        ber_contour[20:44, 55:95] = 1e-15
        
        result = contour.get_eye_dimensions(ber_contour, voltage_bins, time_slices, target_ber=1e-12)
        
        assert isinstance(result, dict)
    
    def test_calculate_empty_eye_pdf_raises_error(self):
        """Test that empty eye PDF raises appropriate error."""
        contour = BERContour()
        
        with pytest.raises((ValueError, IndexError)):
            contour.calculate(np.array([]), np.array([]))
    
    def test_calculate_mismatched_dimensions_raises_error(self):
        """Test that mismatched dimensions raise appropriate error."""
        contour = BERContour()
        
        voltage_bins = np.linspace(-1, 1, 64)
        eye_pdf = np.random.rand(32, 128)  # Wrong second dimension
        
        with pytest.raises(ValueError):
            contour.calculate(eye_pdf, voltage_bins)
    
    def test_integration_full_workflow(self):
        """Integration test for full BER contour workflow."""
        # Create analyzer with NRZ
        contour = BERContour(modulation=NRZ())
        
        # Generate realistic eye PDF
        voltage_bins = np.linspace(-1.5, 1.5, 256)
        time_slices = np.linspace(0, 1, 128)
        
        # Simulate eye opening
        eye_pdf = np.zeros((128, 256))
        for t_idx, t in enumerate(time_slices):
            for v_idx, v in enumerate(voltage_bins):
                # Create eye shape - tighter at edges, open in center
                eye_opening = 0.8 * np.exp(-4 * (t - 0.5)**2)
                noise_std = 0.1 + 0.2 * abs(t - 0.5)
                eye_pdf[t_idx, v_idx] = np.exp(-(v)**2 / (2 * noise_std**2))
        
        # Normalize
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        # Calculate BER contour
        ber_contour = contour.calculate(eye_pdf, voltage_bins, target_bers=[1e-12])
        
        # Extract eye dimensions
        dims = contour.get_eye_dimensions(ber_contour, voltage_bins, time_slices, target_ber=1e-12)
        
        assert isinstance(dims, dict)


class TestBERContourEncapsulation:
    """Test that BERContour properly encapsulates BERCalculator."""
    
    def test_uses_ber_calculator_internally(self):
        """Test that BERContour uses BERCalculator internally."""
        from eye_analyzer.statistical.ber_calculator import BERCalculator
        
        contour = BERContour()
        # Should have a calculator instance
        assert hasattr(contour, '_calculator')
        assert isinstance(contour._calculator, BERCalculator)
    
    def test_calculate_delegates_to_calculator(self):
        """Test that calculate delegates to BERCalculator.calculate_ber_contour."""
        contour = BERContour()
        
        voltage_bins = np.linspace(-1, 1, 64)
        eye_pdf = np.random.rand(32, 64)
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        result = contour.calculate(eye_pdf, voltage_bins)
        
        # Result should be valid BER contour
        assert result.shape == eye_pdf.shape
        assert np.all(result >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
