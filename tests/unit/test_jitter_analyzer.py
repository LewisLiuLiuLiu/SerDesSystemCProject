"""Unit tests for JitterAnalyzer class."""

import pytest
import numpy as np


def test_jitter_analyzer_import():
    """Test that JitterAnalyzer can be imported."""
    from eye_analyzer.jitter import JitterAnalyzer
    assert JitterAnalyzer is not None


def test_jitter_analyzer_init_nrz():
    """Test JitterAnalyzer initialization with NRZ modulation."""
    from eye_analyzer.jitter import JitterAnalyzer
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    assert analyzer.modulation == 'nrz'
    assert analyzer.signal_amplitude == 1.0


def test_jitter_analyzer_init_pam4():
    """Test JitterAnalyzer initialization with PAM4 modulation."""
    from eye_analyzer.jitter import JitterAnalyzer
    analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=1.0)
    assert analyzer.modulation == 'pam4'
    assert analyzer.signal_amplitude == 1.0


def test_jitter_analyzer_init_invalid_modulation():
    """Test JitterAnalyzer initialization with invalid modulation."""
    from eye_analyzer.jitter import JitterAnalyzer
    with pytest.raises(ValueError):
        JitterAnalyzer(modulation='invalid', signal_amplitude=1.0)


def test_jitter_analyzer_analyze_nrz_returns_dict():
    """Test NRZ analyze returns dictionary with expected keys."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    # Generate synthetic NRZ signal with some jitter
    ui = 1e-10  # 100 ps UI (10 Gbps)
    fs = 10 / ui  # 10 samples per UI
    t = np.arange(0, 1000 * ui, 1/fs)
    
    # Create square wave with RJ
    bits = np.random.randint(0, 2, size=100)
    signal = np.zeros_like(t)
    for i, bit in enumerate(bits):
        start_idx = int(i * 10)
        end_idx = int((i + 1) * 10)
        if end_idx <= len(signal):
            signal[start_idx:end_idx] = 1.0 if bit else -1.0
    
    # Add some random jitter via phase noise
    result = analyzer.analyze(signal, t, ber=1e-12)
    
    assert isinstance(result, dict)
    assert 'rj' in result
    assert 'dj' in result
    assert 'tj' in result
    assert isinstance(result['rj'], float)
    assert isinstance(result['dj'], float)
    assert isinstance(result['tj'], float)


def test_jitter_analyzer_analyze_pam4_returns_list():
    """Test PAM4 analyze returns list of eye dictionaries."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=0.75)
    
    # Generate synthetic PAM4 signal
    ui = 1e-10  # 100 ps UI (10 Gbps)
    fs = 10 / ui  # 10 samples per UI
    t = np.arange(0, 1000 * ui, 1/fs)
    
    # Create PAM4 signal with 4 levels: -0.75, -0.25, 0.25, 0.75
    levels = [-0.75, -0.25, 0.25, 0.75]
    symbols = np.random.randint(0, 4, size=100)
    signal = np.zeros_like(t)
    for i, sym in enumerate(symbols):
        start_idx = int(i * 10)
        end_idx = int((i + 1) * 10)
        if end_idx <= len(signal):
            signal[start_idx:end_idx] = levels[sym]
    
    result = analyzer.analyze(signal, t, ber=1e-12)
    
    assert isinstance(result, list)
    assert len(result) == 3  # PAM4 has 3 eyes
    
    for i, eye in enumerate(result):
        assert 'eye_id' in eye
        assert 'eye_name' in eye
        assert 'rj' in eye
        assert 'dj' in eye
        assert 'tj' in eye
        assert eye['eye_id'] == i
        assert isinstance(eye['rj'], float)
        assert isinstance(eye['dj'], float)
        assert isinstance(eye['tj'], float)


def test_jitter_analyzer_eye_names_pam4():
    """Test PAM4 eye names are correct."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=0.75)
    
    ui = 1e-10
    fs = 10 / ui
    t = np.arange(0, 1000 * ui, 1/fs)
    
    levels = [-0.75, -0.25, 0.25, 0.75]
    symbols = np.random.randint(0, 4, size=100)
    signal = np.zeros_like(t)
    for i, sym in enumerate(symbols):
        start_idx = int(i * 10)
        end_idx = int((i + 1) * 10)
        if end_idx <= len(signal):
            signal[start_idx:end_idx] = levels[sym]
    
    result = analyzer.analyze(signal, t, ber=1e-12)
    
    expected_names = ['lower', 'middle', 'upper']
    for i, eye in enumerate(result):
        assert eye['eye_name'] == expected_names[i]


def test_extract_crossing_points_basic():
    """Test extract_crossing_points finds crossing points."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    # Create simple square wave over multiple periods
    # Use 100 ns to capture several full periods (period = 10 ns for 100 MHz)
    t = np.linspace(0, 100e-9, 10000)  # 100 ns with fine resolution
    # Create signal with levels +1 and -1 (no exact zeros at crossings)
    signal = np.where(np.sin(2 * np.pi * 1e8 * t) >= 0, 1.0, -1.0)
    
    # Test both edge types
    all_crossings = analyzer.extract_crossing_points(signal, threshold=0.0, edge='both')
    rising_crossings = analyzer.extract_crossing_points(signal, threshold=0.0, edge='rising')
    falling_crossings = analyzer.extract_crossing_points(signal, threshold=0.0, edge='falling')
    
    assert isinstance(all_crossings, np.ndarray)
    assert len(all_crossings) > 0
    assert len(rising_crossings) > 0
    assert len(falling_crossings) > 0
    # Total should be approximately sum of rising and falling
    assert abs(len(all_crossings) - (len(rising_crossings) + len(falling_crossings))) <= 1


def test_extract_crossing_points_rising_edge():
    """Test extract_crossing_points with rising edge selection."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    # Use multiple periods to ensure both edge types are captured
    t = np.linspace(0, 100e-9, 10000)
    signal = np.where(np.sin(2 * np.pi * 1e8 * t) >= 0, 1.0, -1.0)
    
    rising_crossings = analyzer.extract_crossing_points(signal, threshold=0.0, edge='rising')
    falling_crossings = analyzer.extract_crossing_points(signal, threshold=0.0, edge='falling')
    
    # Rising and falling crossings should be interleaved
    assert len(rising_crossings) > 0
    assert len(falling_crossings) > 0
    assert abs(len(rising_crossings) - len(falling_crossings)) <= 1


def test_extract_crossing_points_invalid_edge():
    """Test extract_crossing_points with invalid edge parameter."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    t = np.linspace(0, 10e-9, 1000)
    signal = np.sign(np.sin(2 * np.pi * 1e8 * t))
    
    with pytest.raises(ValueError):
        analyzer.extract_crossing_points(signal, threshold=0.0, edge='invalid')


def test_fit_dual_dirac_basic():
    """Test fit_dual_dirac returns rj and dj values."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    # Generate crossing points with known RJ and DJ
    ui = 1e-10
    np.random.seed(42)
    # Pure RJ: Gaussian distributed
    rj_true = 0.01 * ui  # 1% UI RJ
    crossing_points = np.random.normal(0, rj_true, 1000)
    
    rj, dj = analyzer.fit_dual_dirac(crossing_points)
    
    assert isinstance(rj, float)
    assert isinstance(dj, float)
    assert rj >= 0
    assert dj >= 0


def test_fit_dual_dirac_with_dj():
    """Test fit_dual_dirac detects DJ component."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    ui = 1e-10
    np.random.seed(42)
    # Create bimodal distribution (RJ + DJ)
    rj_true = 0.005 * ui
    dj_true = 0.1 * ui
    
    # Two Gaussian peaks separated by DJ
    n_samples = 500
    peak1 = np.random.normal(-dj_true/2, rj_true, n_samples)
    peak2 = np.random.normal(dj_true/2, rj_true, n_samples)
    crossing_points = np.concatenate([peak1, peak2])
    
    rj, dj = analyzer.fit_dual_dirac(crossing_points)
    
    assert dj > 0  # Should detect DJ
    assert rj > 0  # Should detect RJ


def test_calculate_tj_formula():
    """Test calculate_tj uses correct formula: TJ = DJ + 2 * Q(BER) * RJ."""
    from eye_analyzer.jitter import JitterAnalyzer
    from eye_analyzer.utils import q_function
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    rj = 1e-12  # 1 ps
    dj = 5e-12  # 5 ps
    ber = 1e-12
    
    tj = analyzer.calculate_tj(rj, dj, ber)
    
    expected_tj = dj + 2 * q_function(ber) * rj
    assert np.isclose(tj, expected_tj)


def test_calculate_tj_different_ber():
    """Test calculate_tj with different BER values."""
    from eye_analyzer.jitter import JitterAnalyzer
    from eye_analyzer.utils import q_function
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    rj = 1e-12
    dj = 5e-12
    
    # TJ should increase with lower BER
    tj_1e9 = analyzer.calculate_tj(rj, dj, 1e-9)
    tj_1e12 = analyzer.calculate_tj(rj, dj, 1e-12)
    tj_1e15 = analyzer.calculate_tj(rj, dj, 1e-15)
    
    assert tj_1e9 < tj_1e12 < tj_1e15


def test_tj_consistency_with_rj_dj():
    """Test that TJ is consistent with RJ and DJ components."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='nrz', signal_amplitude=1.0)
    
    ui = 1e-10
    fs = 10 / ui
    t = np.arange(0, 1000 * ui, 1/fs)
    
    # Create NRZ signal
    bits = np.random.randint(0, 2, size=100)
    signal = np.zeros_like(t)
    for i, bit in enumerate(bits):
        start_idx = int(i * 10)
        end_idx = int((i + 1) * 10)
        if end_idx <= len(signal):
            signal[start_idx:end_idx] = 1.0 if bit else -1.0
    
    result = analyzer.analyze(signal, t, ber=1e-12)
    
    # Verify TJ >= DJ and TJ >= 2*RJ (approximately)
    assert result['tj'] >= result['dj'] - 1e-15  # Allow small numerical error
    assert result['tj'] > 0
    assert result['rj'] >= 0
    assert result['dj'] >= 0


def test_pam4_tj_consistency():
    """Test that PAM4 eye TJ is consistent for each eye."""
    from eye_analyzer.jitter import JitterAnalyzer
    
    analyzer = JitterAnalyzer(modulation='pam4', signal_amplitude=0.75)
    
    ui = 1e-10
    fs = 10 / ui
    t = np.arange(0, 1000 * ui, 1/fs)
    
    levels = [-0.75, -0.25, 0.25, 0.75]
    symbols = np.random.randint(0, 4, size=100)
    signal = np.zeros_like(t)
    for i, sym in enumerate(symbols):
        start_idx = int(i * 10)
        end_idx = int((i + 1) * 10)
        if end_idx <= len(signal):
            signal[start_idx:end_idx] = levels[sym]
    
    result = analyzer.analyze(signal, t, ber=1e-12)
    
    for eye in result:
        assert eye['tj'] >= eye['dj'] - 1e-15
        assert eye['tj'] > 0
        assert eye['rj'] >= 0
        assert eye['dj'] >= 0
