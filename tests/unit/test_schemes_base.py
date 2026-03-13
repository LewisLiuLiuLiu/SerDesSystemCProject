"""Unit tests for BaseScheme modulation support."""

import pytest
import numpy as np
from eye_analyzer.schemes.base import BaseScheme
from eye_analyzer.modulation import PAM4, NRZ, create_modulation


class MockScheme(BaseScheme):
    """Mock implementation for testing BaseScheme."""
    
    def analyze(self, time_array, voltage_array, **kwargs):
        return {
            'eye_height': 0.5,
            'eye_width': 0.8,
            'modulation': self.modulation.name
        }
    
    def get_xedges(self):
        return np.linspace(0, 1, self.ui_bins + 1)
    
    def get_yedges(self):
        return np.linspace(-1, 1, self.amp_bins + 1)


def test_basescheme_with_string_modulation():
    """Test BaseScheme accepts modulation as string."""
    scheme = MockScheme(ui=1e-12, modulation='pam4', ui_bins=64, amp_bins=128)
    
    assert scheme.ui == 1e-12
    assert scheme.modulation.name == 'pam4'
    assert scheme.modulation.num_levels == 4
    assert scheme.ui_bins == 64
    assert scheme.amp_bins == 128


def test_basescheme_with_object_modulation():
    """Test BaseScheme accepts modulation as ModulationFormat object."""
    pam4 = PAM4()
    scheme = MockScheme(ui=1e-12, modulation=pam4)
    
    assert scheme.modulation.name == 'pam4'
    assert scheme.modulation is pam4  # Same object


def test_basescheme_default_is_nrz():
    """Test BaseScheme defaults to NRZ modulation."""
    scheme = MockScheme(ui=1e-12)
    
    assert scheme.modulation.name == 'nrz'
    assert scheme.modulation.num_levels == 2


def test_basescheme_nrz_string():
    """Test BaseScheme with 'nrz' string."""
    scheme = MockScheme(ui=1e-12, modulation='nrz')
    
    assert scheme.modulation.name == 'nrz'
    assert scheme.modulation.num_eyes == 1


def test_basescheme_invalid_modulation():
    """Test BaseScheme raises error for invalid modulation."""
    with pytest.raises(ValueError, match="Unknown modulation"):
        MockScheme(ui=1e-12, modulation='invalid')


def test_basescheme_validation_still_works():
    """Test that existing validation still works."""
    # Invalid UI
    with pytest.raises(ValueError, match="UI must be positive"):
        MockScheme(ui=-1e-12, modulation='pam4')
    
    # Invalid ui_bins
    with pytest.raises(ValueError, match="ui_bins must be at least"):
        MockScheme(ui=1e-12, modulation='pam4', ui_bins=1)
    
    # Invalid amp_bins
    with pytest.raises(ValueError, match="amp_bins must be at least"):
        MockScheme(ui=1e-12, modulation='pam4', amp_bins=1)


def test_basescheme_num_eyes_property():
    """Test that num_eyes is correctly derived from modulation."""
    pam4_scheme = MockScheme(ui=1e-12, modulation='pam4')
    assert pam4_scheme.modulation.num_eyes == 3
    
    nrz_scheme = MockScheme(ui=1e-12, modulation='nrz')
    assert nrz_scheme.modulation.num_eyes == 1
