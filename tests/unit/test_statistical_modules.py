"""Basic tests for statistical eye analysis modules."""

import pytest
import numpy as np
from eye_analyzer.statistical import (
    PulseResponseProcessor,
    ISICalculator,
    NoiseInjector,
    JitterInjector,
    BERContourCalculator
)
from eye_analyzer.modulation import PAM4, NRZ


def test_isi_calculator_convolution():
    """Test ISI calculator with convolution method."""
    calc = ISICalculator(method='convolution')
    
    pulse = np.zeros(64)
    pulse[32] = 1.0
    
    pdf = calc.calculate(pulse, PAM4(), samples_per_symbol=8, sample_size=4, vh_size=512)
    
    assert pdf is not None
    assert pdf.shape[0] == 512
    assert np.all(pdf >= 0)


def test_isi_calculator_brute_force():
    """Test ISI calculator with brute force method."""
    calc = ISICalculator(method='brute_force')
    
    pulse = np.zeros(64)
    pulse[32] = 1.0
    
    pdf = calc.calculate(pulse, NRZ(), samples_per_symbol=8, sample_size=3, vh_size=512)
    
    assert pdf is not None
    assert pdf.shape[0] == 512


def test_noise_injector():
    """Test noise injection."""
    injector = NoiseInjector()
    
    pdf = np.random.rand(100, 16)
    pdf = pdf / np.sum(pdf, axis=0, keepdims=True)
    
    noisy = injector.inject(pdf, sigma=0.1, mu=0.0)
    
    assert noisy.shape == pdf.shape
    assert np.all(noisy >= 0)


def test_jitter_injector():
    """Test jitter injection."""
    injector = JitterInjector()
    
    pdf = np.random.rand(100, 16)
    pdf = pdf / np.sum(pdf, axis=0, keepdims=True)
    
    jittered = injector.inject(pdf, dj=0.01, rj=0.01, samples_per_symbol=8)
    
    assert jittered.shape == pdf.shape
    assert np.all(jittered >= 0)


def test_ber_calculator():
    """Test BER contour calculator."""
    calc = BERContourCalculator()
    
    pdf = np.random.rand(100, 16)
    
    contours = calc.calculate_contours(pdf, PAM4(), ber_levels=[1e-6, 1e-9])
    
    assert len(contours) == 2
    assert 1e-6 in contours
    assert 1e-9 in contours
