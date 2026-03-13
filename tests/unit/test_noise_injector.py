"""
Tests for NoiseInjector - Statistical Eye Analysis

TDD Workflow:
1. Write failing test (RED)
2. Verify it fails correctly
3. Write minimal code to pass (GREEN)
4. Verify it passes
5. Refactor if needed
"""

import pytest
import numpy as np
from eye_analyzer.statistical.noise_injector import NoiseInjector


class TestNoiseInjector:
    """Test suite for NoiseInjector class."""
    
    def test_class_exists(self):
        """RED: Verify NoiseInjector class exists."""
        assert NoiseInjector is not None
    
    def test_initialization_default_sigma(self):
        """RED: Test injector initializes with default sigma."""
        injector = NoiseInjector()
        assert injector is not None
        assert injector.sigma > 0  # Default sigma should be positive
    
    def test_initialization_custom_sigma(self):
        """RED: Test injector initializes with custom sigma."""
        injector = NoiseInjector(sigma=0.01)
        assert injector.sigma == 0.01
    
    def test_initialization_zero_sigma(self):
        """RED: Test injector accepts zero sigma (no noise)."""
        injector = NoiseInjector(sigma=0.0)
        assert injector.sigma == 0.0
    
    def test_initialization_negative_sigma_raises(self):
        """RED: Test injector rejects negative sigma."""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            NoiseInjector(sigma=-0.01)


class TestGaussianPDFGeneration:
    """Tests for Gaussian PDF generation."""
    
    def test_gaussian_pdf_shape(self):
        """RED: Test generated Gaussian PDF has correct shape."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        assert len(noise_pdf) == len(voltage_bins)
    
    def test_gaussian_pdf_peak_at_zero(self):
        """RED: Test Gaussian PDF peaks at zero (center)."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.1, 0.1, 101)
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        # Peak should be at the center (index 50 for 101 points)
        peak_idx = np.argmax(noise_pdf)
        assert peak_idx == 50  # Center of symmetric range
    
    def test_gaussian_pdf_normalization(self):
        """RED: Test Gaussian PDF is approximately normalized."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 1001)
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        # Integral should be approximately 1 (using trapezoidal rule)
        bin_width = voltage_bins[1] - voltage_bins[0]
        integral = np.trapz(noise_pdf, voltage_bins)
        assert np.isclose(integral, 1.0, atol=0.01)
    
    def test_gaussian_pdf_zero_sigma(self):
        """RED: Test Gaussian PDF with zero sigma is a delta function."""
        injector = NoiseInjector(sigma=0.0)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        # Should be zero everywhere (or at least very small)
        # With sigma=0, scipy returns inf at 0, 0 elsewhere
        # We handle this as a special case
        assert np.sum(noise_pdf) > 0  # At least has some mass


class TestNoiseInjection:
    """Tests for noise injection via convolution."""
    
    def test_inject_noise_basic(self):
        """RED: Test basic noise injection on ISI PDF."""
        injector = NoiseInjector(sigma=0.01)
        # Create a simple ISI PDF (delta-like)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0  # Delta at center
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        # Result should have same shape
        assert len(result) == len(isi_pdf)
        # Result should be non-negative
        assert np.all(result >= 0)
        # Result should still sum to approximately 1
        assert np.isclose(np.sum(result), 1.0, atol=0.01)
    
    def test_inject_noise_two_peaks(self):
        """RED: Test noise injection on bimodal ISI PDF."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[30] = 0.5  # Peak 1
        isi_pdf[70] = 0.5  # Peak 2
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        # Should have same length
        assert len(result) == len(isi_pdf)
        # Should be non-negative
        assert np.all(result >= 0)
        # Should still be normalized
        assert np.isclose(np.sum(result), 1.0, atol=0.01)
        # After convolution, peaks should be smoothed (wider)
        nonzero_count_original = np.count_nonzero(isi_pdf)
        nonzero_count_result = np.count_nonzero(result > 1e-6)
        assert nonzero_count_result >= nonzero_count_original
    
    def test_inject_noise_zero_sigma(self):
        """RED: Test noise injection with zero sigma (no change)."""
        injector = NoiseInjector(sigma=0.0)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        # With zero sigma, result should be very close to original
        # (may have small numerical differences)
        assert np.allclose(result, isi_pdf, atol=1e-3)
    
    def test_inject_noise_larger_sigma_more_smoothing(self):
        """RED: Test that larger sigma produces more smoothing."""
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0  # Delta function
        
        injector_small = NoiseInjector(sigma=0.01)
        result_small = injector_small.inject_noise(isi_pdf, voltage_bins)
        
        injector_large = NoiseInjector(sigma=0.05)
        result_large = injector_large.inject_noise(isi_pdf, voltage_bins)
        
        # Larger sigma should produce wider spread
        # Measure by standard deviation
        mean_small = np.sum(voltage_bins * result_small)
        var_small = np.sum((voltage_bins - mean_small)**2 * result_small)
        std_small = np.sqrt(var_small)
        
        mean_large = np.sum(voltage_bins * result_large)
        var_large = np.sum((voltage_bins - mean_large)**2 * result_large)
        std_large = np.sqrt(var_large)
        
        assert std_large > std_small


class TestConvolution:
    """Tests for the convolution operation."""
    
    def test_convolve_preserves_shape(self):
        """RED: Test that convolution preserves array shape with mode='same'."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        
        # Simple ISI: single peak
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0
        
        # Gaussian noise PDF
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        
        # Manual convolution
        result = np.convolve(isi_pdf, noise_pdf, mode='same')
        
        assert len(result) == len(isi_pdf)
    
    def test_convolve_normalization(self):
        """RED: Test that convolving two normalized PDFs produces normalized result."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        bin_width = voltage_bins[1] - voltage_bins[0]
        
        # Normalized ISI PDF
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[40] = 0.5
        isi_pdf[60] = 0.5
        
        # Gaussian noise PDF (normalized)
        noise_pdf = injector.generate_gaussian_pdf(voltage_bins)
        
        # Convolution
        result = np.convolve(isi_pdf, noise_pdf, mode='same') * bin_width
        
        # Should preserve total probability (approximately)
        assert np.isclose(np.sum(result), 1.0, atol=0.1)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_inject_noise_mismatched_lengths(self):
        """RED: Test error when ISI PDF and voltage bins have different lengths."""
        injector = NoiseInjector(sigma=0.01)
        isi_pdf = np.array([0.3, 0.4, 0.3])
        voltage_bins = np.linspace(-0.5, 0.5, 10)  # Different length
        
        with pytest.raises(ValueError, match="length"):
            injector.inject_noise(isi_pdf, voltage_bins)
    
    def test_inject_noise_empty_array(self):
        """RED: Test error when given empty arrays."""
        injector = NoiseInjector(sigma=0.01)
        
        with pytest.raises(ValueError, match="empty"):
            injector.inject_noise(np.array([]), np.array([]))
    
    def test_inject_noise_negative_pdf(self):
        """RED: Test error when ISI PDF contains negative values."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0
        isi_pdf[60] = -0.1  # Negative value
        
        with pytest.raises(ValueError, match="non-negative"):
            injector.inject_noise(isi_pdf, voltage_bins)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_inject_noise_uniform_pdf(self):
        """RED: Test noise injection on uniform PDF."""
        injector = NoiseInjector(sigma=0.01)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.ones_like(voltage_bins) / len(voltage_bins)  # Uniform
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        assert len(result) == len(isi_pdf)
        assert np.all(result >= 0)
        # Convolution of uniform with Gaussian rounds the edges
    
    def test_inject_noise_very_small_sigma(self):
        """RED: Test noise injection with very small sigma."""
        injector = NoiseInjector(sigma=1e-6)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[50] = 1.0
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        # Very small sigma should produce minimal smoothing
        assert np.allclose(result, isi_pdf, atol=0.01)
    
    def test_inject_noise_very_large_sigma(self):
        """RED: Test noise injection with very large sigma."""
        injector = NoiseInjector(sigma=0.5)
        voltage_bins = np.linspace(-1.0, 1.0, 201)
        isi_pdf = np.zeros_like(voltage_bins)
        isi_pdf[100] = 1.0  # Delta at center
        
        result = injector.inject_noise(isi_pdf, voltage_bins)
        
        # Large sigma should spread the PDF significantly
        nonzero_count = np.count_nonzero(result > 1e-6)
        assert nonzero_count > 10  # Should spread across many bins


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
