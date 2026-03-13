"""Unit tests for modulation format abstraction layer."""

import pytest
import numpy as np
from eye_analyzer.modulation import (
    ModulationFormat, PAM4, NRZ, 
    create_modulation, MODULATION_REGISTRY
)


def test_pam4_basic():
    """Test PAM4 modulation format basic properties."""
    pam4 = PAM4()
    assert pam4.name == 'pam4'
    assert pam4.num_levels == 4
    assert pam4.num_eyes == 3
    np.testing.assert_array_equal(pam4.get_levels(), [-3, -1, 1, 3])
    np.testing.assert_array_equal(pam4.get_thresholds(), [-2, 0, 2])
    np.testing.assert_array_equal(pam4.get_eye_centers(), [-2, 0, 2])


def test_nrz_basic():
    """Test NRZ modulation format basic properties."""
    nrz = NRZ()
    assert nrz.name == 'nrz'
    assert nrz.num_levels == 2
    assert nrz.num_eyes == 1
    np.testing.assert_array_equal(nrz.get_levels(), [-1, 1])
    np.testing.assert_array_equal(nrz.get_thresholds(), [0])
    np.testing.assert_array_equal(nrz.get_eye_centers(), [0])


def test_factory_function():
    """Test modulation factory function."""
    pam4 = create_modulation('pam4')
    assert isinstance(pam4, PAM4)
    
    nrz = create_modulation('nrz')
    assert isinstance(nrz, NRZ)
    
    with pytest.raises(ValueError):
        create_modulation('invalid')


def test_pam4_level_names():
    """Test PAM4 level names."""
    pam4 = PAM4()
    names = pam4.get_level_names()
    assert names == ['LV3', 'LV2', 'LV1', 'LV0']


def test_nrz_level_names():
    """Test NRZ level names."""
    nrz = NRZ()
    names = nrz.get_level_names()
    assert names == ['LV0', 'LV1']


def test_modulation_registry():
    """Test modulation registry contents."""
    assert 'nrz' in MODULATION_REGISTRY
    assert 'pam4' in MODULATION_REGISTRY
    assert MODULATION_REGISTRY['nrz'] == NRZ
    assert MODULATION_REGISTRY['pam4'] == PAM4
