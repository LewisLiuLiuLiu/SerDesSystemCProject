"""
Unit tests for JitterInjector.

Tests Dual-Dirac jitter model (DJ + RJ) as specified in OIF-CEI standard.

Core Algorithm:
    - Two Gaussian distributions: norm(x, -DJ, RJ) and norm(x, +DJ, RJ)
    - Mixed: jitter_pdf = (gauss1 + gauss2) / 2
    - Applied to each time slice
"""

import numpy as np
import pytest
from scipy import stats

# Import the module to test
from eye_analyzer.statistical.jitter_injector import JitterInjector


class TestJitterInjectorInitialization:
    """Tests for JitterInjector initialization and parameter validation."""
    
    def test_init_with_valid_params(self):
        """Test initialization with valid DJ and RJ parameters."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        assert injector.dj == 0.05
        assert injector.rj == 0.01
    
    def test_init_with_zero_dj(self):
        """Test initialization with zero DJ (RJ only)."""
        injector = JitterInjector(dj=0.0, rj=0.01)
        assert injector.dj == 0.0
        assert injector.rj == 0.01
    
    def test_init_with_zero_rj(self):
        """Test initialization with zero RJ (DJ only)."""
        injector = JitterInjector(dj=0.05, rj=0.0)
        assert injector.dj == 0.05
        assert injector.rj == 0.0
    
    def test_init_negative_dj_raises(self):
        """Test that negative DJ raises ValueError."""
        with pytest.raises(ValueError, match="DJ must be non-negative"):
            JitterInjector(dj=-0.01, rj=0.01)
    
    def test_init_negative_rj_raises(self):
        """Test that negative RJ raises ValueError."""
        with pytest.raises(ValueError, match="RJ must be non-negative"):
            JitterInjector(dj=0.05, rj=-0.01)


class TestDualDiracPDF:
    """Tests for Dual-Dirac PDF generation."""
    
    def test_generate_pdf_shape(self):
        """Test that generated PDF has correct shape."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        time_bins = np.linspace(-0.2, 0.2, 201)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        assert pdf.shape == time_bins.shape
    
    def test_generate_pdf_normalized(self):
        """Test that generated PDF integrates to 1.0."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        time_bins = np.linspace(-0.5, 0.5, 1001)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        # Numerical integration using trapezoidal rule
        integral = np.trapezoid(pdf, time_bins)
        assert np.isclose(integral, 1.0, rtol=1e-3)
    
    def test_generate_pdf_bimodal(self):
        """Test that PDF has two peaks at +/- DJ."""
        dj = 0.05
        injector = JitterInjector(dj=dj, rj=0.005)
        time_bins = np.linspace(-0.2, 0.2, 1001)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        
        # Find peaks near +/- DJ
        idx_neg = np.argmin(np.abs(time_bins - (-dj)))
        idx_pos = np.argmin(np.abs(time_bins - dj))
        
        # Check peaks are at expected locations
        assert pdf[idx_neg] > pdf[len(pdf)//2]  # Higher than center
        assert pdf[idx_pos] > pdf[len(pdf)//2]
    
    def test_generate_pdf_zero_dj_single_peak(self):
        """Test that DJ=0 produces single Gaussian peak."""
        injector = JitterInjector(dj=0.0, rj=0.01)
        time_bins = np.linspace(-0.1, 0.1, 501)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        
        # Peak should be at center
        center_idx = len(pdf) // 2
        assert pdf[center_idx] == np.max(pdf)
    
    def test_generate_pdf_zero_rj_dirac_like(self):
        """Test that RJ=0 produces delta-like peaks."""
        dj = 0.05
        injector = JitterInjector(dj=dj, rj=0.0)
        time_bins = np.linspace(-0.2, 0.2, 401)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        
        # Should have sharp peaks at +/- DJ
        idx_neg = np.argmin(np.abs(time_bins - (-dj)))
        idx_pos = np.argmin(np.abs(time_bins - dj))
        
        assert pdf[idx_neg] > 0
        assert pdf[idx_pos] > 0


class TestJitterInjection:
    """Tests for jitter injection into time-domain data."""
    
    def test_inject_jitter_shape_preservation(self):
        """Test that output shape matches input shape."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        
        # Create a simple 2D eye PDF: time x voltage
        time_bins = np.linspace(-0.5, 0.5, 101)
        voltage_bins = np.linspace(-0.5, 0.5, 51)
        eye_pdf = np.zeros((len(time_bins), len(voltage_bins)))
        eye_pdf[50, 25] = 1.0  # Delta at center
        
        result = injector.inject_jitter(eye_pdf, time_bins)
        assert result.shape == eye_pdf.shape
    
    def test_inject_jitter_normalization(self):
        """Test that output PDF remains normalized."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        
        time_bins = np.linspace(-0.5, 0.5, 201)
        voltage_bins = np.linspace(-0.5, 0.5, 101)
        eye_pdf = np.zeros((len(time_bins), len(voltage_bins)))
        eye_pdf[100, 50] = 1.0
        
        result = injector.inject_jitter(eye_pdf, time_bins)
        
        # Sum over voltage for each time slice
        total_prob = np.sum(result)
        assert np.isclose(total_prob, 1.0, rtol=1e-3)
    
    def test_inject_jitter_increases_spread(self):
        """Test that jitter injection increases time spread."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        
        time_bins = np.linspace(-0.2, 0.2, 401)
        voltage_bins = np.linspace(-0.5, 0.5, 51)
        
        # Create delta function at time=0
        eye_pdf = np.zeros((len(time_bins), len(voltage_bins)))
        center_idx = len(time_bins) // 2
        eye_pdf[center_idx, :] = 1.0 / len(voltage_bins)  # Uniform in voltage
        
        result = injector.inject_jitter(eye_pdf, time_bins)
        
        # Calculate standard deviation of time distribution
        time_marginal_before = np.sum(eye_pdf, axis=1)
        time_marginal_after = np.sum(result, axis=1)
        
        std_before = np.sqrt(np.sum(time_bins**2 * time_marginal_before))
        std_after = np.sqrt(np.sum(time_bins**2 * time_marginal_after))
        
        assert std_after > std_before
    
    def test_inject_zero_jitter_no_change(self):
        """Test that DJ=0, RJ=0 leaves input unchanged."""
        injector = JitterInjector(dj=0.0, rj=0.0)
        
        time_bins = np.linspace(-0.5, 0.5, 101)
        voltage_bins = np.linspace(-0.5, 0.5, 51)
        eye_pdf = np.zeros((len(time_bins), len(voltage_bins)))
        eye_pdf[50, 25] = 1.0
        
        result = injector.inject_jitter(eye_pdf, time_bins)
        
        # Should be essentially unchanged (allowing for numerical precision)
        np.testing.assert_array_almost_equal(result, eye_pdf, decimal=10)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_time_bins_raises(self):
        """Test that empty time bins raises error."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        with pytest.raises(ValueError, match="empty"):
            injector.generate_dual_dirac_pdf(np.array([]))
    
    def test_empty_eye_pdf_raises(self):
        """Test that empty eye PDF raises error."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        time_bins = np.linspace(-0.5, 0.5, 101)
        with pytest.raises(ValueError, match="empty"):
            injector.inject_jitter(np.array([]), time_bins)
    
    def test_mismatched_dimensions_raises(self):
        """Test that mismatched time bins and PDF dimensions raise error."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        time_bins = np.linspace(-0.5, 0.5, 100)
        eye_pdf = np.zeros((50, 51))  # Wrong first dimension
        with pytest.raises(ValueError, match="dimension"):
            injector.inject_jitter(eye_pdf, time_bins)
    
    def test_negative_pdf_values_raises(self):
        """Test that negative PDF values raise error."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        time_bins = np.linspace(-0.5, 0.5, 101)
        voltage_bins = np.linspace(-0.5, 0.5, 51)
        eye_pdf = np.zeros((len(time_bins), len(voltage_bins)))
        eye_pdf[50, 25] = -1.0  # Invalid negative value
        with pytest.raises(ValueError, match="negative"):
            injector.inject_jitter(eye_pdf, time_bins)


class TestOIFCompliance:
    """Tests for OIF-CEI standard compliance."""
    
    def test_dual_dirac_model_matches_spec(self):
        """Test that dual-Dirac PDF matches OIF-CEI mathematical model."""
        dj = 0.05  # UI units
        rj = 0.01
        injector = JitterInjector(dj=dj, rj=rj)
        
        time_bins = np.linspace(-0.2, 0.2, 801)
        pdf = injector.generate_dual_dirac_pdf(time_bins)
        
        # Expected: (N(x; -DJ, RJ) + N(x; +DJ, RJ)) / 2
        gauss1 = stats.norm.pdf(time_bins, loc=-dj, scale=rj)
        gauss2 = stats.norm.pdf(time_bins, loc=dj, scale=rj)
        expected_pdf = (gauss1 + gauss2) / 2.0
        
        # Normalize both to compare shape
        pdf_norm = pdf / np.sum(pdf)
        expected_norm = expected_pdf / np.sum(expected_pdf)
        
        np.testing.assert_array_almost_equal(pdf_norm, expected_norm, decimal=5)
    
    def test_tj_at_ber(self):
        """Test TJ calculation at specific BER levels."""
        injector = JitterInjector(dj=0.05, rj=0.01)
        
        # For dual-Dirac: TJ(DJ, RJ, BER) = DJ + 2 * Q(BER) * RJ
        # where Q is the Q-factor for given BER
        ber = 1e-12
        q_factor = stats.norm.ppf(1 - ber)  # ~7.03 for 1e-12
        expected_tj = 0.05 + 2 * q_factor * 0.01
        
        tj = injector.calculate_tj(ber)
        assert np.isclose(tj, expected_tj, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
