"""BER (Bit Error Rate) analysis modules.

This package provides tools for BER contour generation, bathtub curve analysis,
jitter tolerance testing, and eye dimension analysis.

Main Classes:
    BERAnalyzer: Unified interface for all BER analysis components
    BERContour: BER contour calculation from eye diagram PDFs
    BathtubCurve: Bathtub curve generation (BER vs. time/voltage)
    QFactor: BER to Q-factor conversion utilities
    JTolTemplate: Industry standard jitter tolerance templates
    JitterTolerance: SJ tolerance testing and measurement

Example:
    >>> from eye_analyzer.ber import BERAnalyzer
    >>> 
    >>> # Create unified analyzer
    >>> analyzer = BERAnalyzer(modulation='pam4', target_ber=1e-12)
    >>> 
    >>> # Analyze eye diagram
    >>> result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
    >>> print(f"Eye height: {result['eye_height']:.3f} V")
    >>> 
    >>> # Bathtub curve analysis
    >>> bathtub = analyzer.analyze_bathtub(
    ...     eye_pdf, voltage_bins, time_slices, direction='time'
    ... )
    >>> 
    >>> # BER-Q conversion
    >>> q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
    
Submodules:
    contour: BERContour class for BER contour generation
    bathtub: BathtubCurve class for bathtub curve analysis
    qfactor: QFactor class for BER/Q conversion
    template: JTolTemplate class for standard templates
    jtol: JitterTolerance class for JTOL testing
    analyzer: BERAnalyzer class for unified interface
"""

from .analyzer import BERAnalyzer
from .contour import BERContour
from .bathtub import BathtubCurve
from .qfactor import QFactor
from .template import JTolTemplate
from .jtol import JitterTolerance

__all__ = [
    'BERAnalyzer',
    'BERContour',
    'BathtubCurve',
    'QFactor',
    'JTolTemplate',
    'JitterTolerance',
]
