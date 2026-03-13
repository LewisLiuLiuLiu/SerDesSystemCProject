"""
Noise Injector for Statistical Eye Analysis.

This module provides Gaussian noise injection into ISI (Inter-Symbol Interference)
PDFs through PDF convolution for statistical eye diagram analysis.

Features:
- Gaussian noise PDF generation using scipy.stats.norm
- PDF convolution with configurable sigma (standard deviation)
- Support for zero noise (sigma=0) for testing purposes
- Input validation for PDF arrays

Core Algorithm:
    1. Generate Gaussian noise PDF: scipy.stats.norm.pdf(voltage_bins, 0, sigma)
    2. For each time slice: np.convolve(isi_pdf, noise_pdf, mode='same')
    3. Normalize to preserve total probability

Reference: Statistical Eye Methodology for Channel Operating Margin (COM)
"""

import numpy as np
from scipy import stats
from typing import Optional


class NoiseInjector:
    """
    Inject Gaussian noise into ISI PDFs via convolution.
    
    This class handles the statistical combination of ISI (Inter-Symbol Interference)
    probability density functions with Gaussian noise. The noise is injected by
    convolving the ISI PDF with a Gaussian PDF representing the noise distribution.
    
    Attributes:
        sigma: Standard deviation of the Gaussian noise (volts)
    
    Example:
        >>> injector = NoiseInjector(sigma=0.01)
        >>> voltage_bins = np.linspace(-0.5, 0.5, 101)
        >>> isi_pdf = np.zeros_like(voltage_bins)
        >>> isi_pdf[50] = 1.0  # Delta at center
        >>> result = injector.inject_noise(isi_pdf, voltage_bins)
        >>> print(f"Result shape: {result.shape}, sum: {np.sum(result):.4f}")
    """
    
    def __init__(self, sigma: float = 0.01):
        """
        Initialize the NoiseInjector.
        
        Args:
            sigma: Standard deviation of Gaussian noise in volts (must be non-negative)
        
        Raises:
            ValueError: If sigma is negative
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma = sigma
    
    def generate_gaussian_pdf(self, voltage_bins: np.ndarray) -> np.ndarray:
        """
        Generate a Gaussian noise PDF over the given voltage bins.
        
        The Gaussian is centered at 0V with standard deviation sigma.
        
        Args:
            voltage_bins: Array of voltage values defining the PDF bins
        
        Returns:
            Gaussian PDF array with same shape as voltage_bins
        
        Note:
            When sigma=0, returns a delta-like function (all zeros except at center)
        """
        if self.sigma == 0:
            # Special case: delta function at center
            pdf = np.zeros_like(voltage_bins)
            center_idx = len(voltage_bins) // 2
            pdf[center_idx] = 1.0
            return pdf
        
        # Generate Gaussian PDF using scipy.stats.norm
        pdf = stats.norm.pdf(voltage_bins, loc=0.0, scale=self.sigma)
        
        return pdf
    
    def inject_noise(
        self,
        isi_pdf: np.ndarray,
        voltage_bins: np.ndarray
    ) -> np.ndarray:
        """
        Inject Gaussian noise into an ISI PDF via convolution.
        
        This method convolves the input ISI PDF with a Gaussian noise PDF to
        produce the combined PDF representing ISI + noise.
        
        Args:
            isi_pdf: Input ISI probability density function (must be non-negative)
            voltage_bins: Array of voltage values defining the bins
        
        Returns:
            Convolved PDF with same shape as input (normalized to sum=1)
        
        Raises:
            ValueError: If arrays are empty, have mismatched lengths, or if ISI PDF
                       contains negative values
        
        Algorithm:
            1. Validate inputs (non-empty, matching lengths, non-negative PDF)
            2. Generate Gaussian noise PDF
            3. Convolve: np.convolve(isi_pdf, noise_pdf, mode='same')
            4. Normalize result to preserve total probability
        """
        # Input validation
        if len(isi_pdf) == 0 or len(voltage_bins) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(isi_pdf) != len(voltage_bins):
            raise ValueError(
                f"ISI PDF and voltage bins must have same length, "
                f"got {len(isi_pdf)} and {len(voltage_bins)}"
            )
        
        if np.any(isi_pdf < 0):
            raise ValueError("ISI PDF values must be non-negative")
        
        # Special case: zero sigma - return normalized ISI PDF as-is
        if self.sigma == 0:
            # Normalize to ensure sum=1
            pdf_sum = np.sum(isi_pdf)
            if pdf_sum > 0:
                return isi_pdf / pdf_sum
            else:
                # All zeros case - return as-is
                return isi_pdf.copy()
        
        # Generate Gaussian noise PDF
        noise_pdf = self.generate_gaussian_pdf(voltage_bins)
        
        # Convolve ISI PDF with noise PDF
        # mode='same' preserves the array shape
        result = np.convolve(isi_pdf, noise_pdf, mode='same')
        
        # Normalize to preserve total probability
        result_sum = np.sum(result)
        if result_sum > 0:
            # Account for bin width in continuous PDF conversion
            bin_width = voltage_bins[1] - voltage_bins[0] if len(voltage_bins) > 1 else 1.0
            # Normalize such that sum(result) * bin_width ≈ 1 for PDF
            # But since we're returning discrete probabilities, normalize to sum=1
            result = result / result_sum
        
        return result
