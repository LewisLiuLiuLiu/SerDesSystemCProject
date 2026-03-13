"""Statistical eye diagram analysis modules.

This package provides tools for statistical eye diagram analysis based on
channel pulse response, supporting both NRZ and PAM4 modulation formats.
"""

from .pulse_response import PulseResponseProcessor
from .isi_calculator import ISICalculator
from .noise_injector import NoiseInjector
from .jitter_injector import JitterInjector
from .ber_calculator import BERCalculator

# Backward compatibility alias
BERContourCalculator = BERCalculator

__all__ = [
    'PulseResponseProcessor',
    'ISICalculator',
    'NoiseInjector',
    'JitterInjector',
    'BERCalculator',
    'BERContourCalculator',  # Backward compatibility
]
