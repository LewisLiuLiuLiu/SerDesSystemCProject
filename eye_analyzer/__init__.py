"""Eye Analyzer - SerDes Signal Integrity Analysis Platform.

A comprehensive Python toolkit for eye diagram analysis, BER computation,
and jitter tolerance testing. Supports NRZ and PAM4 modulation formats.

Example:
    >>> from eye_analyzer import EyeAnalyzer
    >>> analyzer = EyeAnalyzer(ui=2.5e-11, modulation='pam4')
    >>> result = analyzer.analyze(pulse_response)
    >>> analyzer.plot_eye()
"""

__version__ = '2.0.0'

# ============================================================================
# Core Analyzer (Unified Entry Point)
# ============================================================================
from .analyzer import EyeAnalyzer

# ============================================================================
# Modulation Formats
# ============================================================================
from .modulation import (
    ModulationFormat,
    NRZ,
    PAM4,
    create_modulation,
)

# ============================================================================
# Scheme Classes
# ============================================================================
from .schemes import (
    BaseScheme,
    GoldenCdrScheme,
    SamplerCentricScheme,
    StatisticalScheme,
)

# ============================================================================
# Statistical Analysis (Pre-simulation)
# ============================================================================
from .statistical import (
    PulseResponseProcessor,
    ISICalculator,
    NoiseInjector,
    JitterInjector,
    BERCalculator,
)

# ============================================================================
# BER Analysis
# ============================================================================
from .ber import (
    BERAnalyzer,
    BERContour,
    BathtubCurve,
    QFactor,
    JTolTemplate,
    JitterTolerance,
)

# ============================================================================
# Jitter Analysis
# ============================================================================
from .jitter import (
    JitterAnalyzer,
    JitterDecomposer,
)

# ============================================================================
# Visualization
# ============================================================================
from .visualization import (
    plot_eye_diagram,
    plot_jtol_curve,
    plot_bathtub_curve,
    create_analysis_report,
)

# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
UnifiedEyeAnalyzer = EyeAnalyzer  # Legacy name compatibility

__all__ = [
    # Version
    '__version__',
    
    # Core
    'EyeAnalyzer',
    'UnifiedEyeAnalyzer',
    
    # Modulation
    'ModulationFormat',
    'NRZ',
    'PAM4',
    'create_modulation',
    
    # Schemes
    'BaseScheme',
    'GoldenCdrScheme',
    'SamplerCentricScheme',
    'StatisticalScheme',
    
    # Statistical
    'PulseResponseProcessor',
    'ISICalculator',
    'NoiseInjector',
    'JitterInjector',
    'BERCalculator',
    
    # BER
    'BERAnalyzer',
    'BERContour',
    'BathtubCurve',
    'QFactor',
    'JTolTemplate',
    'JitterTolerance',
    
    # Jitter
    'JitterAnalyzer',
    'JitterDecomposer',
    
    # Visualization
    'plot_eye_diagram',
    'plot_jtol_curve',
    'plot_bathtub_curve',
    'create_analysis_report',
]
