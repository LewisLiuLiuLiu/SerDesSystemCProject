"""Statistical eye diagram analysis modules.

This package provides tools for statistical eye diagram analysis based on
channel pulse response, supporting both NRZ and PAM4 modulation formats.
"""

from .pulse_response import PulseResponseProcessor

__all__ = [
    'PulseResponseProcessor',
]
