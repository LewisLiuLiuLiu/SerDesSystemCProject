"""
EyeAnalyzer - SerDes Link Eye Diagram Analysis Tool

A Python tool for analyzing eye diagrams from SystemC-AMS simulation output.
Supports eye diagram construction, eye height/width calculation, jitter decomposition,
and visualization.

Version: 1.2.0
Author: SerDes SystemC Project Team
"""

from .core import EyeAnalyzer, analyze_eye
from .io import auto_load_waveform
from .jitter import JitterDecomposer

__version__ = "1.2.0"
__all__ = ["EyeAnalyzer", "analyze_eye", "auto_load_waveform", "JitterDecomposer"]