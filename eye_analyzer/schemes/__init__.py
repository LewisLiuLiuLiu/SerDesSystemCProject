"""
Eye Diagram Analysis Schemes

This module provides complementary eye diagram analysis schemes:
- SamplerCentricScheme: Streaming scheme based on actual CDR sampling timestamps
- GoldenCdrScheme: Standard scheme based on ideal clock (Golden CDR)
- StatisticalScheme: Statistical analysis based on channel pulse response
"""

from .base import BaseScheme
from .sampler_centric import SamplerCentricScheme
from .golden_cdr import GoldenCdrScheme
from .statistical import StatisticalScheme

__all__ = ['BaseScheme', 'SamplerCentricScheme', 'GoldenCdrScheme', 'StatisticalScheme']
