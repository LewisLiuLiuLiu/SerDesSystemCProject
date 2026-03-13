"""
Unit tests for EyeAnalyzer - unified entry point.

Tests cover:
- Statistical mode with pulse response input
- Empirical mode with time-domain waveform input
- NRZ and PAM4 modulation support
- JTOL analysis
- Visualization methods
"""

import pytest
import numpy as np


class TestEyeAnalyzerInitialization:
    """Test EyeAnalyzer initialization and configuration."""
    
    def test_import_eye_analyzer(self):
        """Test that EyeAnalyzer can be imported."""
        from eye_analyzer import EyeAnalyzer
        assert EyeAnalyzer is not None
    
    def test_init_statistical_mode_nrz(self):
        """Test initialization in statistical mode with NRZ."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='nrz',
            mode='statistical'
        )
        assert analyzer.ui == 2.5e-11
        assert analyzer.modulation == 'nrz'
        assert analyzer.mode == 'statistical'
    
    def test_init_statistical_mode_pam4(self):
        """Test initialization in statistical mode with PAM4."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='pam4',
            mode='statistical'
        )
        assert analyzer.modulation == 'pam4'
    
    def test_init_empirical_mode(self):
        """Test initialization in empirical mode."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='nrz',
            mode='empirical'
        )
        assert analyzer.mode == 'empirical'
    
    def test_init_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        from eye_analyzer import EyeAnalyzer
        with pytest.raises(ValueError, match="Invalid mode"):
            EyeAnalyzer(ui=2.5e-11, mode='invalid')
    
    def test_init_invalid_modulation(self):
        """Test that invalid modulation raises ValueError."""
        from eye_analyzer import EyeAnalyzer
        with pytest.raises(ValueError, match="Invalid modulation"):
            EyeAnalyzer(ui=2.5e-11, modulation='invalid')
    
    def test_default_parameters(self):
        """Test default parameter values."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert analyzer.target_ber == 1e-12
        assert analyzer.samples_per_symbol == 16
        assert analyzer.ui_bins == 128
        assert analyzer.amp_bins == 256


class TestEyeAnalyzerStatisticalMode:
    """Test EyeAnalyzer in statistical mode (pulse response input)."""
    
    @pytest.fixture
    def sample_pulse_response(self):
        """Create a sample pulse response for testing."""
        dt = 1e-12
        t = np.arange(0, 500e-12, dt)
        # Simple exponential decay pulse
        pulse = np.exp(-t / 50e-12) * np.sin(2 * np.pi * t / 100e-12)
        return pulse
    
    def test_analyze_statistical_nrz(self, sample_pulse_response):
        """Test statistical analysis with NRZ."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='nrz',
            mode='statistical'
        )
        
        result = analyzer.analyze(
            input_data=sample_pulse_response,
            noise_sigma=0.01,
            jitter_dj=0.05,
            jitter_rj=0.01
        )
        
        assert 'eye_matrix' in result
        assert 'xedges' in result
        assert 'yedges' in result
        assert 'eye_metrics' in result
        assert 'eye_height' in result['eye_metrics']
        assert 'eye_width' in result['eye_metrics']
    
    def test_analyze_statistical_pam4(self, sample_pulse_response):
        """Test statistical analysis with PAM4."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='pam4',
            mode='statistical'
        )
        
        result = analyzer.analyze(
            input_data=sample_pulse_response,
            noise_sigma=0.01
        )
        
        assert 'eye_matrix' in result
        assert 'eye_metrics' in result
        # PAM4 should have multiple eye heights
        assert 'eye_heights_per_eye' in result['eye_metrics']
    
    def test_analyze_returns_ber_contour_statistical(self, sample_pulse_response):
        """Test that statistical mode returns BER contour."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='statistical')
        
        result = analyzer.analyze(input_data=sample_pulse_response)
        
        assert 'ber_contour' in result
        assert result['ber_contour'] is not None


class TestEyeAnalyzerEmpiricalMode:
    """Test EyeAnalyzer in empirical mode (time-domain waveform input)."""
    
    @pytest.fixture
    def sample_waveform(self):
        """Create a sample time-domain waveform for testing."""
        ui = 2.5e-11
        fs = 64 / ui  # 64 samples per UI
        t = np.arange(0, 1000 * ui, 1/fs)
        # Simple NRZ-like signal with some noise
        np.random.seed(42)
        bits = np.random.randint(0, 2, 1000)
        signal = np.repeat(bits, 64).astype(float)
        # Add transitions
        for i in range(1, len(signal)):
            if signal[i] != signal[i-1]:
                # Add some transition smoothing
                pass
        signal = signal * 0.8 - 0.4  # Scale to +/- 0.4V
        signal += np.random.randn(len(signal)) * 0.01  # Add noise
        return t, signal
    
    def test_analyze_empirical_nrz(self, sample_waveform):
        """Test empirical analysis with NRZ."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(
            ui=2.5e-11,
            modulation='nrz',
            mode='empirical'
        )
        
        time_array, value_array = sample_waveform
        result = analyzer.analyze(
            input_data=(time_array, value_array)
        )
        
        assert 'eye_matrix' in result
        assert 'xedges' in result
        assert 'yedges' in result
        assert 'eye_metrics' in result
    
    def test_analyze_empirical_tuple_input(self, sample_waveform):
        """Test that empirical mode accepts (time, value) tuple."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='empirical')
        
        time_array, value_array = sample_waveform
        result = analyzer.analyze(input_data=(time_array, value_array))
        
        assert result is not None
        assert 'eye_matrix' in result


class TestEyeAnalyzerJitterAnalysis:
    """Test jitter analysis capabilities."""
    
    def test_jitter_results_in_metrics(self):
        """Test that jitter analysis results are included."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        
        # Create simple pulse response
        pulse = np.exp(-np.arange(100) * 1e-12 / 50e-12)
        result = analyzer.analyze(
            input_data=pulse,
            jitter_dj=0.05,
            jitter_rj=0.01
        )
        
        assert 'jitter' in result
        assert 'dj' in result['jitter']
        assert 'rj' in result['jitter']
    
    def test_bathtub_curves_in_result(self):
        """Test that bathtub curves are included in results."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='statistical')
        
        pulse = np.exp(-np.arange(100) * 1e-12 / 50e-12)
        result = analyzer.analyze(input_data=pulse)
        
        assert 'bathtub_time' in result
        assert 'bathtub_voltage' in result


class TestEyeAnalyzerJtol:
    """Test JTOL (Jitter Tolerance) analysis."""
    
    def test_analyze_jtol_exists(self):
        """Test that analyze_jtol method exists."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert hasattr(analyzer, 'analyze_jtol')
    
    def test_analyze_jtol_returns_results(self):
        """Test JTOL analysis returns expected results."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='statistical')
        
        # Create list of pulse responses with different SJ
        pulse_responses = [
            np.exp(-np.arange(100) * 1e-12 / 50e-12) for _ in range(3)
        ]
        sj_frequencies = [1e6, 10e6, 100e6]
        
        result = analyzer.analyze_jtol(
            pulse_responses=pulse_responses,
            sj_frequencies=sj_frequencies,
            template='ieee_802_3ck'
        )
        
        assert result is not None
        assert 'sj_frequencies' in result
        assert 'sj_amplitudes' in result


class TestEyeAnalyzerVisualization:
    """Test visualization methods."""
    
    def test_plot_eye_exists(self):
        """Test that plot_eye method exists."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert hasattr(analyzer, 'plot_eye')
    
    def test_plot_jtol_exists(self):
        """Test that plot_jtol method exists."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert hasattr(analyzer, 'plot_jtol')
    
    def test_plot_bathtub_exists(self):
        """Test that plot_bathtub method exists."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert hasattr(analyzer, 'plot_bathtub')
    
    def test_create_report_exists(self):
        """Test that create_report method exists."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11)
        assert hasattr(analyzer, 'create_report')


class TestEyeAnalyzerModesConsistency:
    """Test consistency between statistical and empirical modes."""
    
    def test_both_modes_produce_eye_matrix(self):
        """Test that both modes produce eye_matrix in output."""
        from eye_analyzer import EyeAnalyzer
        
        # Statistical mode
        analyzer_stat = EyeAnalyzer(ui=2.5e-11, mode='statistical')
        pulse = np.exp(-np.arange(100) * 1e-12 / 50e-12)
        result_stat = analyzer_stat.analyze(input_data=pulse)
        
        # Empirical mode
        analyzer_emp = EyeAnalyzer(ui=2.5e-11, mode='empirical')
        t = np.arange(0, 1000 * 2.5e-11, 2.5e-11/64)
        v = np.random.randn(len(t)) * 0.1
        result_emp = analyzer_emp.analyze(input_data=(t, v))
        
        assert 'eye_matrix' in result_stat
        assert 'eye_matrix' in result_emp
        assert isinstance(result_stat['eye_matrix'], np.ndarray)
        assert isinstance(result_emp['eye_matrix'], np.ndarray)


class TestEyeAnalyzerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_pulse_response_raises_error(self):
        """Test that empty pulse response raises error."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='statistical')
        
        with pytest.raises(ValueError):
            analyzer.analyze(input_data=np.array([]))
    
    def test_mismatched_time_value_arrays_raises_error(self):
        """Test that mismatched time/value arrays raise error."""
        from eye_analyzer import EyeAnalyzer
        analyzer = EyeAnalyzer(ui=2.5e-11, mode='empirical')
        
        with pytest.raises(ValueError):
            analyzer.analyze(input_data=(np.array([1, 2]), np.array([1, 2, 3])))
    
    def test_negative_ui_raises_error(self):
        """Test that negative UI raises error."""
        from eye_analyzer import EyeAnalyzer
        
        with pytest.raises(ValueError):
            EyeAnalyzer(ui=-1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
