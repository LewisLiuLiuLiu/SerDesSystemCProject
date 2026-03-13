"""Tests for BERAnalyzer unified interface.

This module tests the BERAnalyzer class which provides a unified interface
for all BER analysis components:
- BERContour - BER contour generation
- BathtubCurve - Bathtub curve analysis
- QFactor - BER/Q conversion
- JTolTemplate - Standard templates
- JitterTolerance - JTOL testing
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch


class TestBERAnalyzerInitialization:
    """Test BERAnalyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """BERAnalyzer should initialize with default values."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        assert analyzer.modulation == 'nrz'
        assert analyzer.target_ber == 1e-12
        assert analyzer.signal_amplitude == 1.0
    
    def test_pam4_initialization(self):
        """BERAnalyzer should support PAM4 modulation."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer(modulation='pam4', target_ber=1e-15)
        
        assert analyzer.modulation == 'pam4'
        assert analyzer.target_ber == 1e-15
    
    def test_custom_parameters(self):
        """BERAnalyzer should accept custom parameters."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer(
            modulation='nrz',
            target_ber=1e-9,
            signal_amplitude=0.8
        )
        
        assert analyzer.target_ber == 1e-9
        assert analyzer.signal_amplitude == 0.8
    
    def test_invalid_modulation_raises(self):
        """BERAnalyzer should raise for invalid modulation."""
        from eye_analyzer.ber import BERAnalyzer
        
        with pytest.raises(ValueError, match="Invalid modulation"):
            BERAnalyzer(modulation='invalid')
    
    def test_components_initialized(self):
        """BERAnalyzer should initialize all component instances."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        assert analyzer._contour is not None
        assert analyzer._bathtub is not None
        assert analyzer._qfactor is not None


class TestBERAnalyzerAnalyzeEye:
    """Test BERAnalyzer.analyze_eye() method."""
    
    @pytest.fixture
    def sample_eye_data(self):
        """Create sample eye data for testing."""
        np.random.seed(42)
        n_time = 64
        n_voltage = 128
        
        # Create a simple eye pattern
        time_slices = np.linspace(0, 1, n_time)
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        
        eye_pdf = np.zeros((n_time, n_voltage))
        for i, t in enumerate(time_slices):
            # Two eye openings for NRZ
            center1 = -0.5 + 0.3 * np.sin(2 * np.pi * t)
            center2 = 0.5 - 0.3 * np.sin(2 * np.pi * t)
            eye_pdf[i] = 0.5 * np.exp(-0.5 * ((voltage_bins - center1) / 0.2) ** 2)
            eye_pdf[i] += 0.5 * np.exp(-0.5 * ((voltage_bins - center2) / 0.2) ** 2)
        
        # Normalize
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        return eye_pdf, voltage_bins, time_slices
    
    def test_analyze_eye_returns_dict(self, sample_eye_data):
        """analyze_eye should return a dictionary with expected keys."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        
        assert isinstance(result, dict)
        assert 'ber_contour' in result
        assert 'eye_height' in result
        assert 'eye_width' in result
        assert 'eye_area' in result
    
    def test_analyze_eye_returns_arrays(self, sample_eye_data):
        """analyze_eye should return numpy arrays for contour."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        
        assert isinstance(result['ber_contour'], np.ndarray)
        assert result['ber_contour'].shape == eye_pdf.shape
    
    def test_analyze_eye_with_default_target_ber(self, sample_eye_data):
        """analyze_eye should use target_ber from initialization."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer(target_ber=1e-9)
        
        result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        
        assert 'ber_contour' in result
        assert result['ber_contour'].shape == eye_pdf.shape
    
    def test_analyze_eye_with_custom_target_ber(self, sample_eye_data):
        """analyze_eye should accept custom target_ber parameter."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_eye(
            eye_pdf, voltage_bins, time_slices, target_ber=1e-6
        )
        
        assert 'ber_contour' in result
    
    def test_analyze_eye_empty_input_raises(self):
        """analyze_eye should raise for empty input."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.analyze_eye(
                np.array([]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3])
            )
    
    def test_analyze_eye_dimension_mismatch_raises(self, sample_eye_data):
        """analyze_eye should raise for dimension mismatch."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, _ = sample_eye_data
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.analyze_eye(
                eye_pdf,
                voltage_bins[:50],  # Wrong length
                np.linspace(0, 1, eye_pdf.shape[0])
            )


class TestBERAnalyzerAnalyzeBathtub:
    """Test BERAnalyzer.analyze_bathtub() method."""
    
    @pytest.fixture
    def sample_eye_data(self):
        """Create sample eye data for testing."""
        np.random.seed(42)
        n_time = 64
        n_voltage = 128
        
        time_slices = np.linspace(0, 1, n_time)
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        
        eye_pdf = np.zeros((n_time, n_voltage))
        for i, t in enumerate(time_slices):
            center1 = -0.5 + 0.3 * np.sin(2 * np.pi * t)
            center2 = 0.5 - 0.3 * np.sin(2 * np.pi * t)
            eye_pdf[i] = 0.5 * np.exp(-0.5 * ((voltage_bins - center1) / 0.2) ** 2)
            eye_pdf[i] += 0.5 * np.exp(-0.5 * ((voltage_bins - center2) / 0.2) ** 2)
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        return eye_pdf, voltage_bins, time_slices
    
    def test_analyze_bathtub_time_direction(self, sample_eye_data):
        """analyze_bathtub should support time direction."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices, direction='time'
        )
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'ber_left' in result
        assert 'ber_right' in result
    
    def test_analyze_bathtub_voltage_direction(self, sample_eye_data):
        """analyze_bathtub should support voltage direction."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices, direction='voltage'
        )
        
        assert isinstance(result, dict)
        assert 'voltage' in result
        assert 'ber_upper' in result
        assert 'ber_lower' in result
    
    def test_analyze_bathtub_default_direction(self, sample_eye_data):
        """analyze_bathtub should default to time direction."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices
        )
        
        # Should default to time direction
        assert 'time' in result
        assert 'ber_left' in result
    
    def test_analyze_bathtub_invalid_direction_raises(self, sample_eye_data):
        """analyze_bathtub should raise for invalid direction."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError, match="Invalid direction"):
            analyzer.analyze_bathtub(
                eye_pdf, voltage_bins, time_slices, direction='invalid'
            )
    
    def test_analyze_bathtub_returns_arrays(self, sample_eye_data):
        """analyze_bathtub should return numpy arrays."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        result = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices, direction='time'
        )
        
        assert isinstance(result['time'], np.ndarray)
        assert isinstance(result['ber_left'], np.ndarray)
        assert isinstance(result['ber_right'], np.ndarray)
        assert len(result['time']) == len(time_slices)


class TestBERAnalyzerConvertBerQ:
    """Test BERAnalyzer.convert_ber_q() method."""
    
    def test_ber_to_q_conversion(self):
        """convert_ber_q should convert BER to Q-factor."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
        
        assert isinstance(q, float)
        assert q > 0
        # Q for BER 1e-12 should be approximately 7.03
        assert 7.0 < q < 7.1
    
    def test_q_to_ber_conversion(self):
        """convert_ber_q should convert Q-factor to BER."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        ber = analyzer.convert_ber_q(7.03, direction='q_to_ber')
        
        assert isinstance(ber, float)
        assert ber > 0
        # BER for Q=7.03 should be approximately 1e-12
        assert 1e-13 < ber < 1e-11
    
    def test_ber_q_roundtrip(self):
        """BER to Q to BER should be approximately consistent."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        original_ber = 1e-9
        q = analyzer.convert_ber_q(original_ber, direction='ber_to_q')
        recovered_ber = analyzer.convert_ber_q(q, direction='q_to_ber')
        
        # Should be close (within 1%)
        assert np.isclose(original_ber, recovered_ber, rtol=0.01)
    
    def test_invalid_direction_raises(self):
        """convert_ber_q should raise for invalid direction."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError, match="Invalid direction"):
            analyzer.convert_ber_q(1e-12, direction='invalid')
    
    def test_invalid_ber_raises(self):
        """convert_ber_q should raise for invalid BER values."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.convert_ber_q(0, direction='ber_to_q')
        
        with pytest.raises(ValueError):
            analyzer.convert_ber_q(1, direction='ber_to_q')
    
    def test_invalid_q_raises(self):
        """convert_ber_q should raise for invalid Q values."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.convert_ber_q(0, direction='q_to_ber')
        
        with pytest.raises(ValueError):
            analyzer.convert_ber_q(-1, direction='q_to_ber')


class TestBERAnalyzerAnalyzeJtol:
    """Test BERAnalyzer.analyze_jtol() method."""
    
    @pytest.fixture
    def sample_pulse_response(self):
        """Create sample pulse response for testing."""
        # Simple exponential decay pulse response
        t = np.linspace(0, 1e-9, 100)
        pulse = np.exp(-t / 100e-12)
        return pulse
    
    def test_analyze_jtol_returns_dict(self, sample_pulse_response):
        """analyze_jtol should return a dictionary."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        sj_frequencies = np.array([1e5, 1e6, 1e7])
        
        with patch.object(analyzer._jtol, 'measure_jtol') as mock_measure:
            mock_measure.return_value = {
                'frequencies': sj_frequencies,
                'sj_limits': np.array([0.1, 0.05, 0.01]),
                'template_limits': np.array([0.1, 0.05, 0.01]),
                'margins': np.array([0.0, 0.0, 0.0]),
                'pass_fail': [True, True, True],
                'overall_pass': True,
                'modulation': 'nrz',
                'target_ber': 1e-12
            }
            
            result = analyzer.analyze_jtol(
                pulse_response=sample_pulse_response,
                sj_frequencies=sj_frequencies,
                template='ieee_802_3ck'
            )
        
        assert isinstance(result, dict)
        assert 'frequencies' in result
        assert 'sj_limits' in result
        assert 'overall_pass' in result
    
    def test_analyze_jtol_with_different_templates(self, sample_pulse_response):
        """analyze_jtol should support different templates."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        sj_frequencies = np.array([1e5, 1e6])
        
        templates = ['ieee_802_3ck', 'oif_cei_112g', 'jedec_ddr5', 'pcie_gen6']
        
        for template in templates:
            with patch.object(analyzer._jtol, 'measure_jtol') as mock_measure:
                mock_measure.return_value = {
                    'frequencies': sj_frequencies,
                    'sj_limits': np.array([0.1, 0.05]),
                    'template_limits': np.array([0.1, 0.05]),
                    'margins': np.array([0.0, 0.0]),
                    'pass_fail': [True, True],
                    'overall_pass': True,
                    'modulation': 'nrz',
                    'target_ber': 1e-12
                }
                
                result = analyzer.analyze_jtol(
                    pulse_response=sample_pulse_response,
                    sj_frequencies=sj_frequencies,
                    template=template
                )
            
            assert result is not None
    
    def test_analyze_jtol_invalid_template_raises(self, sample_pulse_response):
        """analyze_jtol should raise for invalid template."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        sj_frequencies = np.array([1e5, 1e6])
        
        with pytest.raises(ValueError, match="Invalid template"):
            analyzer.analyze_jtol(
                pulse_response=sample_pulse_response,
                sj_frequencies=sj_frequencies,
                template='invalid_template'
            )
    
    def test_analyze_jtol_missing_pulse_response_raises(self):
        """analyze_jtol should raise if pulse_response is missing."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        sj_frequencies = np.array([1e5, 1e6])
        
        with pytest.raises(ValueError, match="pulse_response"):
            analyzer.analyze_jtol(
                pulse_response=None,
                sj_frequencies=sj_frequencies,
                template='ieee_802_3ck'
            )


class TestBERAnalyzerGetTemplate:
    """Test BERAnalyzer.get_template() method."""
    
    def test_get_template_ieee_802_3ck(self):
        """get_template should return IEEE 802.3ck template."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        template = analyzer.get_template('ieee_802_3ck')
        
        assert template is not None
        assert template.template_name == 'ieee_802_3ck'
    
    def test_get_template_oif_cei_112g(self):
        """get_template should return OIF-CEI-112G template."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        template = analyzer.get_template('oif_cei_112g')
        
        assert template is not None
        assert template.template_name == 'oif_cei_112g'
    
    def test_get_template_invalid_raises(self):
        """get_template should raise for invalid template name."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.get_template('invalid_template')
    
    def test_get_template_returns_jtol_template(self):
        """get_template should return a JTolTemplate instance."""
        from eye_analyzer.ber import BERAnalyzer, JTolTemplate
        
        analyzer = BERAnalyzer()
        
        template = analyzer.get_template('ieee_802_3ck')
        
        assert isinstance(template, JTolTemplate)


class TestBERAnalyzerEyeDimensions:
    """Test BERAnalyzer.get_eye_dimensions() method."""
    
    @pytest.fixture
    def sample_eye_data(self):
        """Create sample eye data for testing."""
        np.random.seed(42)
        n_time = 64
        n_voltage = 128
        
        time_slices = np.linspace(0, 1, n_time)
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        
        eye_pdf = np.zeros((n_time, n_voltage))
        for i, t in enumerate(time_slices):
            center1 = -0.5 + 0.3 * np.sin(2 * np.pi * t)
            center2 = 0.5 - 0.3 * np.sin(2 * np.pi * t)
            eye_pdf[i] = 0.5 * np.exp(-0.5 * ((voltage_bins - center1) / 0.2) ** 2)
            eye_pdf[i] += 0.5 * np.exp(-0.5 * ((voltage_bins - center2) / 0.2) ** 2)
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        return eye_pdf, voltage_bins, time_slices
    
    def test_get_eye_dimensions_returns_dict(self, sample_eye_data):
        """get_eye_dimensions should return a dictionary."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        # First get BER contour
        eye_result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        ber_contour = eye_result['ber_contour']
        
        dims = analyzer.get_eye_dimensions(
            ber_contour, voltage_bins, time_slices
        )
        
        assert isinstance(dims, dict)
        assert 'eye_width_ui' in dims
        assert 'eye_height_v' in dims
        assert 'num_eyes' in dims
    
    def test_get_eye_dimensions_with_target_ber(self, sample_eye_data):
        """get_eye_dimensions should respect target_ber parameter."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer()
        
        eye_result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        ber_contour = eye_result['ber_contour']
        
        dims = analyzer.get_eye_dimensions(
            ber_contour, voltage_bins, time_slices, target_ber=1e-6
        )
        
        assert isinstance(dims, dict)
        assert dims['eye_width_ui'] >= 0
        assert dims['eye_height_v'] >= 0


class TestBERAnalyzerPAM4Support:
    """Test BERAnalyzer with PAM4 modulation."""
    
    @pytest.fixture
    def sample_pam4_eye_data(self):
        """Create sample PAM4 eye data for testing."""
        np.random.seed(42)
        n_time = 64
        n_voltage = 128
        
        time_slices = np.linspace(0, 1, n_time)
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        
        # PAM4 has 4 levels: -1, -1/3, 1/3, 1
        eye_pdf = np.zeros((n_time, n_voltage))
        for i, t in enumerate(time_slices):
            # Four levels for PAM4
            levels = [-1.0, -1/3, 1/3, 1.0]
            for level in levels:
                center = level + 0.1 * np.sin(2 * np.pi * t)
                eye_pdf[i] += 0.25 * np.exp(-0.5 * ((voltage_bins - center) / 0.15) ** 2)
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        return eye_pdf, voltage_bins, time_slices
    
    def test_pam4_analyze_eye(self, sample_pam4_eye_data):
        """analyze_eye should work with PAM4 modulation."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_pam4_eye_data
        analyzer = BERAnalyzer(modulation='pam4')
        
        result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        
        assert isinstance(result, dict)
        assert 'ber_contour' in result
        assert 'eye_dimensions' in result
        # num_eyes is inside eye_dimensions dict
        assert 'num_eyes' in result['eye_dimensions']
    
    def test_pam4_analyze_bathtub(self, sample_pam4_eye_data):
        """analyze_bathtub should work with PAM4 modulation."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_pam4_eye_data
        analyzer = BERAnalyzer(modulation='pam4')
        
        result = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices, direction='time'
        )
        
        assert isinstance(result, dict)
        assert 'time' in result


class TestBERAnalyzerIntegration:
    """Integration tests for BERAnalyzer."""
    
    @pytest.fixture
    def sample_eye_data(self):
        """Create sample eye data for testing."""
        np.random.seed(42)
        n_time = 64
        n_voltage = 128
        
        time_slices = np.linspace(0, 1, n_time)
        voltage_bins = np.linspace(-1.5, 1.5, n_voltage)
        
        eye_pdf = np.zeros((n_time, n_voltage))
        for i, t in enumerate(time_slices):
            center1 = -0.5 + 0.3 * np.sin(2 * np.pi * t)
            center2 = 0.5 - 0.3 * np.sin(2 * np.pi * t)
            eye_pdf[i] = 0.5 * np.exp(-0.5 * ((voltage_bins - center1) / 0.2) ** 2)
            eye_pdf[i] += 0.5 * np.exp(-0.5 * ((voltage_bins - center2) / 0.2) ** 2)
        
        eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
        
        return eye_pdf, voltage_bins, time_slices
    
    def test_full_analysis_workflow(self, sample_eye_data):
        """Complete analysis workflow should work end-to-end."""
        from eye_analyzer.ber import BERAnalyzer
        
        eye_pdf, voltage_bins, time_slices = sample_eye_data
        analyzer = BERAnalyzer(modulation='nrz', target_ber=1e-12)
        
        # Analyze eye
        eye_result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        assert 'ber_contour' in eye_result
        
        # Get eye dimensions
        dims = analyzer.get_eye_dimensions(
            eye_result['ber_contour'], voltage_bins, time_slices
        )
        assert 'eye_width_ui' in dims
        
        # Analyze bathtub
        bathtub = analyzer.analyze_bathtub(
            eye_pdf, voltage_bins, time_slices, direction='time'
        )
        assert 'ber_left' in bathtub
        
        # Convert BER to Q
        q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
        assert q > 0
        
        # Get template
        template = analyzer.get_template('ieee_802_3ck')
        assert template.template_name == 'ieee_802_3ck'
    
    def test_ber_q_consistency(self):
        """BER-Q conversion should be consistent."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        # Test common BER values
        test_bers = [1e-3, 1e-6, 1e-9, 1e-12]
        
        for ber in test_bers:
            q = analyzer.convert_ber_q(ber, direction='ber_to_q')
            recovered_ber = analyzer.convert_ber_q(q, direction='q_to_ber')
            
            # Should be within 5%
            assert np.isclose(ber, recovered_ber, rtol=0.05), \
                f"BER {ber} -> Q {q} -> BER {recovered_ber}"


class TestBERAnalyzerStringRepresentation:
    """Test string representation of BERAnalyzer."""
    
    def test_repr(self):
        """BERAnalyzer should have a useful string representation."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer(modulation='pam4', target_ber=1e-15)
        
        repr_str = repr(analyzer)
        
        assert 'BERAnalyzer' in repr_str
        assert 'pam4' in repr_str
        assert '1e-15' in repr_str or '1e-15' in repr_str.lower()
    
    def test_str(self):
        """BERAnalyzer should have a readable string representation."""
        from eye_analyzer.ber import BERAnalyzer
        
        analyzer = BERAnalyzer()
        
        str_str = str(analyzer)
        
        assert 'BERAnalyzer' in str_str
