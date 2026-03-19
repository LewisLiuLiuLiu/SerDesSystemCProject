"""
Unit tests for Vector Fitting Python implementation.
"""

import numpy as np
import pytest
import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from vector_fitting_py import VectorFitting, fit_vector_fitting


class TestPoleIdentification:
    """Test pole identification core functionality."""
    
    def test_build_Dk_real_poles(self):
        """Test Dk matrix construction with real poles."""
        vf = VectorFitting(order=2, asymp=1)  # asymp=1: no constant term
        s = np.array([1j, 2j, 3j])
        poles = np.array([-1.0, -2.0])
        cindex = np.array([0, 0])
        
        Dk = vf._build_Dk(s, poles, cindex)
        
        # Check shape: asymp=1 gives N columns (no extra terms)
        assert Dk.shape == (3, 2)
        
        # Check values for real pole
        expected_0 = 1.0 / (s - poles[0])
        np.testing.assert_allclose(Dk[:, 0], expected_0)
    
    def test_build_Dk_complex_poles(self):
        """Test Dk matrix construction with complex conjugate pairs."""
        vf = VectorFitting(order=2, asymp=1)
        s = np.array([1j, 2j, 3j])
        poles = np.array([-1.0 + 2j, -1.0 - 2j])
        cindex = np.array([1, 2])
        
        Dk = vf._build_Dk(s, poles, cindex)
        
        # First column: 1/(s-p) + 1/(s-p*)
        expected_0 = 1.0/(s - poles[0]) + 1.0/(s - poles[1])
        np.testing.assert_allclose(Dk[:, 0], expected_0)
        
        # Second column: i/(s-p) - i/(s-p*)
        expected_1 = 1j/(s - poles[0]) - 1j/(s - poles[1])
        np.testing.assert_allclose(Dk[:, 1], expected_1)
    
    def test_build_cindex(self):
        """Test cindex construction."""
        vf = VectorFitting(order=4)
        
        # Real, complex pair, real
        poles = np.array([-1.0, -2.0 + 3j, -2.0 - 3j, -4.0])
        cindex = vf._build_cindex(poles)
        
        expected = np.array([0, 1, 2, 0])
        np.testing.assert_array_equal(cindex, expected)
    
    def test_pole_identification_simple(self):
        """Test pole identification with simple 2-pole system."""
        # Create synthetic 2-pole system
        true_poles = np.array([-1e9, -5e9])
        true_residues = np.array([1e9, -0.5e9])
        true_d = 0.1
        
        # Generate frequency response
        freq = np.logspace(8, 10, 100)  # 100 MHz to 10 GHz
        s = 2j * np.pi * freq
        H = np.zeros_like(s)
        for r, p in zip(true_residues, true_poles):
            H += r / (s - p)
        H += true_d
        
        # Initialize with perturbed poles
        vf = VectorFitting(order=2, max_iterations=5)
        vf.poles = np.array([-0.5e9, -3e9])  # Initial guess
        
        weight = np.ones(len(s))
        opts = {'relax': True, 'stable': True, 'asymp': 2}
        
        # Run pole identification
        new_poles = vf._pole_identification(s, H, vf.poles, weight, opts)
        
        # Check that poles moved closer to true values
        # Sort poles for comparison
        new_poles_sorted = np.sort(np.real(new_poles))
        true_poles_sorted = np.sort(true_poles)
        
        # Check convergence toward true values
        initial_error = np.abs(np.sort(vf.poles) - true_poles_sorted)
        final_error = np.abs(new_poles_sorted - true_poles_sorted)
        
        assert np.all(final_error < initial_error), \
            f"Poles should converge: initial_err={initial_error}, final_err={final_error}"
    
    def test_pole_identification_complex_poles(self):
        """Test pole identification with complex conjugate poles."""
        # Create synthetic system with complex poles
        true_poles = np.array([-1e9 + 2e9j, -1e9 - 2e9j])
        true_residues = np.array([1e9, 1e9])
        true_d = 0.1
        
        # Generate frequency response
        freq = np.logspace(8, 10, 100)
        s = 2j * np.pi * freq
        H = np.zeros_like(s)
        for r, p in zip(true_residues, true_poles):
            H += r / (s - p)
        H += true_d
        
        # Initialize with perturbed poles
        vf = VectorFitting(order=2, max_iterations=5)
        vf.poles = np.array([-0.5e9 + 1.5e9j, -0.5e9 - 1.5e9j])
        
        weight = np.ones(len(s))
        opts = {'relax': True, 'stable': True, 'asymp': 2}
        
        # Run pole identification
        new_poles = vf._pole_identification(s, H, vf.poles, weight, opts)
        
        # Check that complex conjugate structure is preserved
        assert len(new_poles) == 2
        # Check conjugate relationship
        np.testing.assert_allclose(new_poles[1], np.conj(new_poles[0]), rtol=1e-10)
    
    def test_stabilization(self):
        """Test that unstable poles are moved to left half-plane."""
        vf = VectorFitting(order=2)
        s = np.array([1j, 2j, 3j], dtype=complex)
        
        # Create H with unstable pole (positive real part)
        unstable_poles = np.array([1e9, -2e9])  # First is unstable
        H = np.zeros(len(s), dtype=complex)
        for p in unstable_poles:
            H += 1e9 / (s - p)
        
        # Start with unstable pole
        initial_poles = np.array([0.5e9, -1.5e9])
        
        weight = np.ones(len(s))
        opts = {'relax': True, 'stable': True, 'asymp': 2}
        
        # Run pole identification
        new_poles = vf._pole_identification(s, H, initial_poles, weight, opts)
        
        # All poles should be stable (negative real part)
        assert np.all(np.real(new_poles) <= 0), \
            f"All poles should be stable, got: {new_poles}"
    
    def test_pole_sorting(self):
        """Test pole sorting functionality."""
        vf = VectorFitting(order=4)
        
        # Unsorted poles
        poles = np.array([-1.0 + 2j, -3.0, -1.0 - 2j, -0.5])
        sorted_poles = vf._sort_poles(poles)
        
        # Real poles should come first, sorted by value
        assert np.isreal(sorted_poles[0])
        assert np.isreal(sorted_poles[1])
        assert np.real(sorted_poles[0]) <= np.real(sorted_poles[1])
        
        # Complex poles come after
        assert not np.isreal(sorted_poles[2])
        assert not np.isreal(sorted_poles[3])
    
    def test_relaxation_vs_non_relaxation(self):
        """Test that relaxation gives reasonable results."""
        # Simple system
        true_poles = np.array([-1e9, -5e9])
        freq = np.logspace(8, 10, 50)
        s = 2j * np.pi * freq
        H = 1e9 / (s - true_poles[0]) + 0.5e9 / (s - true_poles[1]) + 0.1
        
        vf = VectorFitting(order=2, max_iterations=3)
        initial_poles = np.array([-0.8e9, -4e9])
        
        weight = np.ones(len(s))
        
        # With relaxation
        opts_relax = {'relax': True, 'stable': True, 'asymp': 2}
        poles_relax = vf._pole_identification(s, H, initial_poles.copy(), weight, opts_relax)
        
        # Without relaxation
        opts_no_relax = {'relax': False, 'stable': True, 'asymp': 2}
        vf2 = VectorFitting(order=2, max_iterations=3)
        poles_no_relax = vf2._pole_identification(s, H, initial_poles.copy(), weight, opts_no_relax)
        
        # Both should produce stable poles
        assert np.all(np.real(poles_relax) <= 0)
        assert np.all(np.real(poles_no_relax) <= 0)
    
    def test_asymptotic_options(self):
        """Test different asymptotic options."""
        vf1 = VectorFitting(order=2, asymp=1)
        vf2 = VectorFitting(order=2, asymp=2)
        vf3 = VectorFitting(order=2, asymp=3)
        
        s = np.array([1j, 2j])
        poles = np.array([-1.0, -2.0])
        cindex = np.array([0, 0])
        
        # Different asymp options should give different Dk shapes
        Dk1 = vf1._build_Dk(s, poles, cindex)
        Dk2 = vf2._build_Dk(s, poles, cindex)
        Dk3 = vf3._build_Dk(s, poles, cindex)
        
        assert Dk1.shape == (2, 2)  # No asymptotic terms
        assert Dk2.shape == (2, 3)  # +1 for D term
        assert Dk3.shape == (2, 4)  # +2 for D + E*s terms


class TestVectorFittingIntegration:
    """Integration tests for full Vector Fitting flow."""
    
    def test_fit_simple_system(self):
        """Test full fitting on simple 2-pole system."""
        # Create synthetic system
        true_poles = np.array([-1e9, -5e9])
        true_residues = np.array([1e9, -0.5e9])
        true_d = 0.1
        
        freq = np.logspace(8, 10, 100)
        s = 2j * np.pi * freq
        H = np.zeros_like(s)
        for r, p in zip(true_residues, true_poles):
            H += r / (s - p)
        H += true_d
        
        # Fit
        result = fit_vector_fitting(s, H, order=2, max_iterations=10)
        
        assert 'poles' in result
        assert 'residues' in result
        assert len(result['poles']) == 2
    
    def test_fit_complex_system(self):
        """Test fitting with complex conjugate poles."""
        true_poles = np.array([-1e9 + 2e9j, -1e9 - 2e9j, -5e9])
        true_residues = np.array([0.5e9, 0.5e9, -0.3e9])
        true_d = 0.05
        
        freq = np.logspace(8, 10, 200)
        s = 2j * np.pi * freq
        H = np.zeros_like(s)
        for r, p in zip(true_residues, true_poles):
            H += r / (s - p)
        H += true_d
        
        result = fit_vector_fitting(s, H, order=3, max_iterations=10)
        
        assert len(result['poles']) == 3
        # Check complex conjugate structure is preserved
        complex_poles = result['poles'][np.abs(np.imag(result['poles'])) > 1e-6]
        assert len(complex_poles) == 2
        np.testing.assert_allclose(complex_poles[1], np.conj(complex_poles[0]), rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
