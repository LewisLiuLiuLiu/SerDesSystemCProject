"""Unit tests for GoldenCdrScheme PAM4 support."""

import numpy as np
import pytest
from eye_analyzer.schemes import GoldenCdrScheme


def generate_pam4_waveform(ui, num_bits=1000, samples_per_ui=64):
    """Generate a simple PAM4 test waveform."""
    t = np.linspace(0, num_bits * ui, num_bits * samples_per_ui)
    
    # Simple PAM4 pattern: cycling through levels
    levels = [-3, -1, 1, 3]
    pattern = np.array([levels[i % 4] for i in range(num_bits)])
    
    # Create waveform with simple pulse shaping (hold value per UI)
    waveform = np.repeat(pattern, samples_per_ui)
    
    # Truncate to match time array length
    waveform = waveform[:len(t)]
    
    # Add some noise for realistic appearance
    waveform = waveform + np.random.normal(0, 0.05, len(waveform))
    
    return t, waveform


def test_golden_cdr_pam4_analysis():
    """Test GoldenCdrScheme with PAM4 modulation."""
    ui = 2.5e-11  # 40 Gbps
    scheme = GoldenCdrScheme(ui=ui, modulation='pam4', ui_bins=128, amp_bins=256)
    
    time_array, voltage_array = generate_pam4_waveform(ui)
    
    metrics = scheme.analyze(time_array, voltage_array)
    
    # Check basic structure
    assert 'eye_heights_per_eye' in metrics
    assert 'eye_widths_per_eye' in metrics
    
    # PAM4 should have 3 eyes
    assert len(metrics['eye_heights_per_eye']) == 3
    assert len(metrics['eye_widths_per_eye']) == 3
    
    # All eye heights should be positive
    for h in metrics['eye_heights_per_eye']:
        assert h > 0, f"Eye height should be positive, got {h}"
    
    # Check modulation info
    assert metrics.get('modulation') == 'pam4'
    assert metrics.get('num_eyes') == 3


def test_golden_cdr_nrz_still_works():
    """Test GoldenCdrScheme with NRZ modulation (backward compatibility)."""
    ui = 2.5e-11
    scheme = GoldenCdrScheme(ui=ui, modulation='nrz')
    
    # Generate simple NRZ waveform
    samples_per_ui = 64
    num_bits = 500
    t = np.linspace(0, num_bits * ui, num_bits * samples_per_ui)
    
    # Simple square wave pattern
    pattern = np.array([1, -1] * (num_bits // 2))
    v = np.repeat(pattern, samples_per_ui)
    v = v[:len(t)]
    
    metrics = scheme.analyze(t, v)
    
    # Basic metrics should exist
    assert 'eye_height' in metrics
    assert 'eye_width' in metrics
    assert metrics['eye_height'] > 0


def test_golden_cdr_modulation_attribute():
    """Test that modulation attribute is set correctly."""
    pam4_scheme = GoldenCdrScheme(ui=1e-12, modulation='pam4')
    assert pam4_scheme.modulation.name == 'pam4'
    
    nrz_scheme = GoldenCdrScheme(ui=1e-12, modulation='nrz')
    assert nrz_scheme.modulation.name == 'nrz'


def test_golden_cdr_pam4_eye_metrics_structure():
    """Test PAM4 eye metrics have correct structure."""
    ui = 2.5e-11
    scheme = GoldenCdrScheme(ui=ui, modulation='pam4')
    
    time_array, voltage_array = generate_pam4_waveform(ui, num_bits=500)
    metrics = scheme.analyze(time_array, voltage_array)
    
    # Check aggregate metrics
    assert 'eye_height_min' in metrics
    assert 'eye_height_avg' in metrics
    assert 'eye_width_min' in metrics
    assert 'eye_width_avg' in metrics
    
    # Min should be <= avg
    assert metrics['eye_height_min'] <= metrics['eye_height_avg']
    assert metrics['eye_width_min'] <= metrics['eye_width_avg']


def test_golden_cdr_default_modulation_is_nrz():
    """Test that default modulation is NRZ for backward compatibility."""
    scheme = GoldenCdrScheme(ui=1e-12)
    assert scheme.modulation.name == 'nrz'
