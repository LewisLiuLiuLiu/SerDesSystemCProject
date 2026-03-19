#!/usr/bin/env python3
"""
Unit tests for VectorFittingPY module.

Tests follow the plan in docs/plans/2024-03-19-vectorfitting-python-plan.md
"""

import numpy as np
import sys
import os

# Add scripts directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from vector_fitting_py import VectorFittingPY


def test_import():
    """Test that the module can be imported and class is available."""
    assert VectorFittingPY is not None


def test_init_1d():
    """Test class initialization with 1D frequency response (SISO)."""
    freq = np.linspace(1e6, 10e9, 100)
    h = np.random.randn(100) + 1j * np.random.randn(100)
    vf = VectorFittingPY(freq, h)
    
    # Check dimensions
    assert vf.Nc == 1
    assert vf.Ns == 100
    
    # Check arrays
    assert np.array_equal(vf.freq, freq)
    assert vf.h.shape == (1, 100)
    
    # Check complex frequency
    expected_s = 1j * 2 * np.pi * freq
    assert np.allclose(vf.s, expected_s)
    
    # Check initial state
    assert vf.poles is None
    assert vf.residues is None
    assert vf.SER is None


def test_init_2d():
    """Test class initialization with 2D frequency response (MIMO)."""
    freq = np.linspace(1e6, 10e9, 50)
    h = np.random.randn(4, 50) + 1j * np.random.randn(4, 50)
    vf = VectorFittingPY(freq, h)
    
    # Check dimensions
    assert vf.Nc == 4
    assert vf.Ns == 50
    
    # Check arrays
    assert vf.h.shape == (4, 50)


def test_init_preserves_dtype():
    """Test that input arrays are properly converted to numpy arrays."""
    freq = [1e6, 1e7, 1e8, 1e9]
    h = [1+2j, 2+3j, 3+4j, 4+5j]
    
    vf = VectorFittingPY(freq, h)
    
    assert isinstance(vf.freq, np.ndarray)
    assert isinstance(vf.h, np.ndarray)
    assert vf.freq.dtype == float
    assert vf.h.dtype == complex


if __name__ == '__main__':
    # Run tests manually if pytest is not available
    print("Running tests...")
    
    test_import()
    print("✓ test_import passed")
    
    test_init_1d()
    print("✓ test_init_1d passed")
    
    test_init_2d()
    print("✓ test_init_2d passed")
    
    test_init_preserves_dtype()
    print("✓ test_init_preserves_dtype passed")
    
    print("\nAll tests passed!")
