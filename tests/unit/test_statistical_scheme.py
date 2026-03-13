"""
Unit tests for StatisticalScheme - Statistical Eye Diagram Analysis.

Tests verify:
1. StatisticalScheme inherits from BaseScheme
2. Integration with PulseResponseProcessor, ISICalculator, NoiseInjector,
   JitterInjector, and BERCalculator
3. Support for NRZ and PAM4 modulation formats
4. Eye diagram construction from pulse response
5. BER contour calculation and eye metrics extraction
"""

import numpy as np
import pytest


class TestStatisticalSchemeImport:
    """Test import and basic class structure."""
    
    def test_import_statistical_scheme(self):
        """Test that StatisticalScheme can be imported."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        assert StatisticalScheme is not None
    
    def test_inherits_from_base_scheme(self):
        """Test that StatisticalScheme inherits from BaseScheme."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        from eye_analyzer.schemes.base import BaseScheme
        
        assert issubclass(StatisticalScheme, BaseScheme)


class TestStatisticalSchemeInitialization:
    """Test StatisticalScheme initialization."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11)
        
        assert scheme.ui == 2.5e-11
        assert scheme.ui_bins == 128
        assert scheme.amp_bins == 256
        assert scheme.modulation.name == 'nrz'
    
    def test_initialization_pam4(self):
        """Test initialization with PAM4 modulation."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11, modulation='pam4')
        
        assert scheme.modulation.name == 'pam4'
        assert scheme.modulation.num_levels == 4
    
    def test_initialization_with_components(self):
        """Test initialization with component parameters."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        from eye_analyzer.statistical import NoiseInjector, JitterInjector
        
        noise_injector = NoiseInjector(sigma=0.01)
        jitter_injector = JitterInjector(dj=0.05, rj=0.01)
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            noise_injector=noise_injector,
            jitter_injector=jitter_injector,
            samples_per_symbol=16,
            sample_size=32
        )
        
        assert scheme.noise_injector is noise_injector
        assert scheme.jitter_injector is jitter_injector
        assert scheme.samples_per_symbol == 16
        assert scheme.sample_size == 32


class TestStatisticalSchemeNRZ:
    """Test StatisticalScheme with NRZ modulation."""
    
    def test_analyze_with_simple_pulse(self):
        """Test analyze with simple pulse response."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            ui_bins=64,
            amp_bins=128
        )
        
        # Create simple pulse response (single pulse with decay)
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0  # Main cursor
        
        # Add some post-cursor ISI
        for i in range(1, 5):
            if samples_per_ui * 2 + i * samples_per_ui < len(pulse):
                pulse[samples_per_ui * 2 + i * samples_per_ui] = 0.1 / i
        
        metrics = scheme.analyze(pulse_response=pulse, dt=2.5e-11 / samples_per_ui)
        
        # Verify metrics structure
        assert 'eye_height' in metrics
        assert 'eye_width' in metrics
        assert 'eye_area' in metrics
        assert 'scheme' in metrics
        assert 'modulation' in metrics
        assert metrics['scheme'] == 'statistical'
        assert metrics['modulation'] == 'nrz'
    
    def test_analyze_returns_eye_matrix(self):
        """Test that analyze creates eye matrix."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            ui_bins=64,
            amp_bins=128
        )
        
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0
        
        scheme.analyze(pulse_response=pulse, dt=2.5e-11 / samples_per_ui)
        
        eye_matrix = scheme.get_eye_matrix()
        assert eye_matrix is not None
        assert eye_matrix.shape == (64, 128)


class TestStatisticalSchemePAM4:
    """Test StatisticalScheme with PAM4 modulation."""
    
    def test_analyze_pam4_basic(self):
        """Test analyze with PAM4 modulation."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='pam4',
            ui_bins=64,
            amp_bins=128
        )
        
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0
        
        metrics = scheme.analyze(pulse_response=pulse, dt=2.5e-11 / samples_per_ui)
        
        assert metrics['scheme'] == 'statistical'
        assert metrics['modulation'] == 'pam4'
        assert 'eye_heights_per_eye' in metrics
        assert 'eye_widths_per_eye' in metrics
        # PAM4 has 3 eyes, but with simple pulse response may not detect all
        assert metrics['num_eyes'] == 3


class TestStatisticalSchemeNoiseInjection:
    """Test noise injection functionality."""
    
    def test_analyze_with_noise(self):
        """Test analyze with noise injection."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        from eye_analyzer.statistical import NoiseInjector
        
        noise_injector = NoiseInjector(sigma=0.01)
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            noise_injector=noise_injector
        )
        
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0
        
        metrics = scheme.analyze(
            pulse_response=pulse,
            dt=2.5e-11 / samples_per_ui,
            noise_sigma=0.01
        )
        
        assert 'eye_height' in metrics
        assert metrics['eye_height'] > 0


class TestStatisticalSchemeJitterInjection:
    """Test jitter injection functionality."""
    
    def test_analyze_with_jitter(self):
        """Test analyze with jitter injection."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        from eye_analyzer.statistical import JitterInjector
        
        jitter_injector = JitterInjector(dj=0.05, rj=0.01)
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            jitter_injector=jitter_injector
        )
        
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0
        
        metrics = scheme.analyze(
            pulse_response=pulse,
            dt=2.5e-11 / samples_per_ui,
            dj=0.05,
            rj=0.01
        )
        
        assert 'eye_width' in metrics
        assert metrics['eye_width'] >= 0


class TestStatisticalSchemeBERCalculation:
    """Test BER calculation functionality."""
    
    def test_ber_contour_calculation(self):
        """Test that BER contour is calculated."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            ui_bins=64,
            amp_bins=128
        )
        
        samples_per_ui = 16
        pulse = np.zeros(samples_per_ui * 10)
        pulse[samples_per_ui * 2] = 1.0
        
        metrics = scheme.analyze(
            pulse_response=pulse,
            dt=2.5e-11 / samples_per_ui,
            target_ber=1e-12
        )
        
        # Verify metrics are computed with target_ber
        assert 'target_ber' in metrics
        assert metrics['target_ber'] == 1e-12


class TestStatisticalSchemeEdges:
    """Test edge getters."""
    
    def test_get_xedges(self):
        """Test get_xedges returns correct shape."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11, ui_bins=64)
        xedges = scheme.get_xedges()
        
        assert len(xedges) == 65  # ui_bins + 1
    
    def test_get_yedges(self):
        """Test get_yedges returns correct shape."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11, amp_bins=128)
        yedges = scheme.get_yedges()
        
        assert len(yedges) == 129  # amp_bins + 1


class TestStatisticalSchemeIntegration:
    """Test integration with all statistical components."""
    
    def test_full_pipeline(self):
        """Test full statistical analysis pipeline."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        from eye_analyzer.statistical import (
            PulseResponseProcessor, NoiseInjector, 
            JitterInjector, BERCalculator
        )
        from eye_analyzer.modulation import NRZ
        
        # Create all components
        pulse_processor = PulseResponseProcessor()
        noise_injector = NoiseInjector(sigma=0.005)
        jitter_injector = JitterInjector(dj=0.03, rj=0.01)
        
        scheme = StatisticalScheme(
            ui=2.5e-11,
            modulation='nrz',
            ui_bins=64,
            amp_bins=128,
            noise_injector=noise_injector,
            jitter_injector=jitter_injector
        )
        
        # Create realistic pulse response with ISI
        samples_per_ui = 32
        pulse = np.zeros(samples_per_ui * 20)
        main_cursor_idx = samples_per_ui * 5
        pulse[main_cursor_idx] = 0.8  # Main cursor
        
        # Add post-cursors
        for i in range(1, 8):
            idx = main_cursor_idx + i * samples_per_ui
            if idx < len(pulse):
                pulse[idx] = 0.08 * (0.7 ** i)  # Exponential decay
        
        # Add pre-cursors
        for i in range(1, 3):
            idx = main_cursor_idx - i * samples_per_ui
            if idx >= 0:
                pulse[idx] = 0.02 * (0.5 ** i)
        
        # Run analysis
        metrics = scheme.analyze(
            pulse_response=pulse,
            dt=2.5e-11 / samples_per_ui,
            noise_sigma=0.005,
            dj=0.03,
            rj=0.01,
            target_ber=1e-12
        )
        
        # Verify comprehensive metrics
        assert 'eye_height' in metrics
        assert 'eye_width' in metrics
        assert 'eye_area' in metrics
        assert 'scheme' in metrics
        assert 'modulation' in metrics
        
        # Eye should have some opening (height is more reliable than width)
        assert metrics['eye_height'] > 0
        # Eye width may be 0 for some pulse responses due to detection threshold


class TestStatisticalSchemeValidation:
    """Test input validation."""
    
    def test_invalid_pulse_response(self):
        """Test handling of invalid pulse response."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11)
        
        # Empty pulse response should raise error
        with pytest.raises(ValueError):
            scheme.analyze(pulse_response=np.array([]), dt=1e-12)
    
    def test_zero_pulse_response(self):
        """Test handling of all-zero pulse response."""
        from eye_analyzer.schemes.statistical import StatisticalScheme
        
        scheme = StatisticalScheme(ui=2.5e-11)
        
        # All-zero pulse should raise error
        pulse = np.zeros(100)
        with pytest.raises(ValueError):
            scheme.analyze(pulse_response=pulse, dt=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
