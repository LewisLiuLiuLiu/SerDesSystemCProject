"""Quick test for SamplerCentricScheme modulation support."""

import numpy as np
from eye_analyzer.schemes import SamplerCentricScheme


def test_sampler_centric_accepts_modulation():
    """Test that SamplerCentricScheme accepts modulation parameter."""
    scheme = SamplerCentricScheme(ui=1e-12, modulation='pam4')
    assert scheme.modulation.name == 'pam4'
    
    scheme_nrz = SamplerCentricScheme(ui=1e-12, modulation='nrz')
    assert scheme_nrz.modulation.name == 'nrz'


def test_sampler_centric_default_modulation():
    """Test that default modulation is NRZ."""
    scheme = SamplerCentricScheme(ui=1e-12)
    assert scheme.modulation.name == 'nrz'
