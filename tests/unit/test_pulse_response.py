"""Unit tests for pulse response processor."""

import pytest
import numpy as np
from eye_analyzer.statistical import PulseResponseProcessor
from eye_analyzer.modulation import PAM4, NRZ


def test_pulse_response_basic():
    """Test basic pulse response processing."""
    processor = PulseResponseProcessor()
    
    # Create simple pulse response
    pulse = np.zeros(100)
    pulse[40:60] = 1.0  # Simple rectangular pulse
    
    result = processor.process(pulse)
    
    assert len(result) > 0
    assert not np.all(result == 0)


def test_pulse_response_dc_removal():
    """Test DC offset removal."""
    processor = PulseResponseProcessor()
    
    # Create pulse with DC offset
    pulse = np.ones(100) * 0.5
    pulse[40:60] = 1.5
    
    result = processor.process(pulse)
    
    # First sample should be 0 after DC removal
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_pulse_response_diff_signal():
    """Test differential signaling factor."""
    processor = PulseResponseProcessor()
    
    pulse = np.zeros(100)
    pulse[40:60] = 2.0
    
    result_diff = processor.process(pulse, diff_signal=True)
    result_single = processor.process(pulse, diff_signal=False)
    
    # Differential should be half
    assert np.max(result_diff) == pytest.approx(np.max(result_single) * 0.5)


def test_pulse_response_upsampling():
    """Test upsampling functionality."""
    processor = PulseResponseProcessor()
    
    pulse = np.zeros(100)
    pulse[40:60] = 1.0
    
    result_1x = processor.process(pulse, upsampling=1)
    result_4x = processor.process(pulse, upsampling=4)
    
    # 4x upsampling should give ~4x length
    assert len(result_4x) > len(result_1x) * 3


def test_pulse_response_all_zeros_raises():
    """Test that all-zero pulse raises error."""
    processor = PulseResponseProcessor()
    
    with pytest.raises(ValueError, match="all zeros"):
        processor.process(np.zeros(100))


def test_find_main_cursor():
    """Test main cursor detection."""
    processor = PulseResponseProcessor()
    
    pulse = np.zeros(100)
    pulse[50] = 1.0  # Peak at index 50
    
    idx = processor.find_main_cursor(pulse)
    assert idx == 50


def test_estimate_voltage_range_pam4():
    """Test voltage range estimation for PAM4."""
    processor = PulseResponseProcessor()
    
    # Create pulse with amplitude 1.0
    pulse = np.zeros(100)
    pulse[45:55] = 1.0
    
    v_min, v_max = processor.estimate_voltage_range(
        pulse, PAM4(), multiplier=2.0
    )
    
    # For PAM4, max level is 3, so display should be ±(1.0 * 3 * 2 / 3) = ±2.0
    assert v_max > 0
    assert v_min < 0
    assert abs(v_max) == pytest.approx(abs(v_min))
