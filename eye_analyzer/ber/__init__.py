"""BER (Bit Error Rate) analysis modules.

This package provides tools for BER contour generation and eye dimension analysis.
"""

from .contour import BERContour
from .bathtub import BathtubCurve
from .template import JTolTemplate
from .jtol import JitterTolerance

__all__ = [
    'BERContour',
    'BathtubCurve',
    'JTolTemplate',
    'JitterTolerance',
]
