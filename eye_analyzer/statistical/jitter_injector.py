"""
Jitter Injector for Statistical Eye Analysis.

This module provides Dual-Dirac jitter injection for statistical eye diagram analysis.
Implements the OIF-CEI standard jitter model combining Deterministic Jitter (DJ)
and Random Jitter (RJ).

Features:
- Dual-Dirac PDF generation (two Gaussian distributions at +/- DJ)
- Gaussian RJ with configurable sigma
- Time-domain jitter injection via convolution
- TJ calculation at specified BER levels

Core Algorithm (Dual-Dirac Model):
    jitter_pdf = (N(x; -DJ, RJ) + N(x; +DJ, RJ)) / 2
    
    where:
        N(x; μ, σ) = Gaussian PDF with mean μ and standard deviation σ
        DJ = Deterministic Jitter (peak-to-peak / 2)
        RJ = Random Jitter (standard deviation)

Reference: OIF-CEI (Common Electrical Interface) Standard
"""

import numpy as np
from scipy import stats
from typing import Optional


class JitterInjector:
    """
    Inject Dual-Dirac jitter into eye diagram PDFs.
    
    This class implements the OIF-CEI standard Dual-Dirac jitter model,
    which combines Deterministic Jitter (DJ) and Random Jitter (RJ).
    
    The Dual-Dirac model represents jitter as the sum of:
    - DJ (Deterministic Jitter): Bimodal distribution with peaks at +/- DJ
    - RJ (Random Jitter): Gaussian distribution with standard deviation RJ
    
    The combined jitter PDF is:
        p(t) = [N(t; -DJ, RJ) + N(t; +DJ, RJ)] / 2
    
    Attributes:
        dj: Deterministic Jitter amplitude (UI units, half of peak-to-peak)
        rj: Random Jitter standard deviation (UI units)
    
    Example:
        >>> injector = JitterInjector(dj=0.05, rj=0.01)
        >>> time_bins = np.linspace(-0.2, 0.2, 401)
        >>> jitter_pdf = injector.generate_dual_dirac_pdf(time_bins)
        >>> tj_at_1e12 = injector.calculate_tj(ber=1e-12)
    """
    
    def __init__(self, dj: float, rj: float):
        """
        Initialize the JitterInjector with DJ and RJ parameters.
        
        Args:
            dj: Deterministic Jitter amplitude in UI units (half of peak-to-peak)
            rj: Random Jitter standard deviation in UI units
        
        Raises:
            ValueError: If dj or rj is negative
        
        Note:
            DJ represents half of the peak-to-peak deterministic jitter.
            For example, if DJ(peak-to-peak) = 0.1 UI, use dj=0.05.
        """
        if dj < 0:
            raise ValueError(f"DJ must be non-negative, got {dj}")
        if rj < 0:
            raise ValueError(f"RJ must be non-negative, got {rj}")
        
        self.dj = dj
        self.rj = rj
    
    def generate_dual_dirac_pdf(self, time_bins: np.ndarray) -> np.ndarray:
        """
        Generate the Dual-Dirac jitter PDF.
        
        Creates a bimodal distribution with two Gaussian peaks at +/- DJ
        locations, each with standard deviation RJ.
        
        Args:
            time_bins: Array of time values in UI units
        
        Returns:
            Dual-Dirac PDF array with same shape as time_bins (normalized)
        
        Raises:
            ValueError: If time_bins is empty
        
        Algorithm:
            1. Generate Gaussian centered at -DJ: N(x; -DJ, RJ)
            2. Generate Gaussian centered at +DJ: N(x; +DJ, RJ)
            3. Mix: pdf = (gauss1 + gauss2) / 2
            4. Normalize to preserve total probability
        
        Example:
            >>> injector = JitterInjector(dj=0.05, rj=0.01)
            >>> time_bins = np.linspace(-0.2, 0.2, 401)
            >>> pdf = injector.generate_dual_dirac_pdf(time_bins)
            >>> print(f"PDF sum: {np.sum(pdf):.4f}")  # Should be ~1.0
        """
        if len(time_bins) == 0:
            raise ValueError("Time bins array cannot be empty")
        
        # Handle special case: both DJ and RJ are zero
        if self.dj == 0 and self.rj == 0:
            pdf = np.zeros_like(time_bins)
            center_idx = len(time_bins) // 2
            pdf[center_idx] = 1.0
            return pdf
        
        # Handle special case: RJ=0 (delta functions at +/- DJ)
        if self.rj == 0:
            pdf = np.zeros_like(time_bins)
            bin_spacing = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 1.0
            # Find closest bins to +/- DJ
            idx_neg = np.argmin(np.abs(time_bins - (-self.dj)))
            idx_pos = np.argmin(np.abs(time_bins - self.dj))
            # Scale by bin spacing so integral = 1
            if self.dj == 0:
                # Both at center - single delta
                pdf[idx_neg] = 1.0 / bin_spacing
            else:
                pdf[idx_neg] = 0.5 / bin_spacing
                pdf[idx_pos] = 0.5 / bin_spacing
            return pdf
        
        # Standard Dual-Dirac: two Gaussians at +/- DJ
        # N(x; -DJ, RJ)
        gauss_neg = stats.norm.pdf(time_bins, loc=-self.dj, scale=self.rj)
        # N(x; +DJ, RJ)
        gauss_pos = stats.norm.pdf(time_bins, loc=self.dj, scale=self.rj)
        
        # Mix: (N(x; -DJ, RJ) + N(x; +DJ, RJ)) / 2
        pdf = (gauss_neg + gauss_pos) / 2.0
        
        # The scipy.stats.norm.pdf already returns a proper PDF where
        # the integral over the entire range equals 1.0.
        # No additional normalization needed for continuous PDF.
        
        return pdf
    
    def inject_jitter(
        self,
        eye_pdf: np.ndarray,
        time_bins: np.ndarray
    ) -> np.ndarray:
        """
        Inject Dual-Dirac jitter into an eye diagram PDF.
        
        This method applies jitter to the time dimension of an eye diagram PDF
        by convolving each time slice with the Dual-Dirac jitter PDF.
        
        Args:
            eye_pdf: 2D eye diagram PDF with shape (time_bins, voltage_bins)
            time_bins: Array of time values in UI units
        
        Returns:
            Jitter-injected eye PDF with same shape as input
        
        Raises:
            ValueError: If eye_pdf is empty, time_bins is empty, dimensions
                       don't match, or eye_pdf contains negative values
        
        Algorithm:
            1. Validate inputs (non-empty, matching dimensions, non-negative)
            2. Generate Dual-Dirac jitter PDF
            3. For each voltage slice: convolve(time_pdf, jitter_pdf)
            4. Normalize result to preserve total probability
        
        Example:
            >>> injector = JitterInjector(dj=0.05, rj=0.01)
            >>> time_bins = np.linspace(-0.5, 0.5, 101)
            >>> voltage_bins = np.linspace(-0.5, 0.5, 51)
            >>> eye_pdf = np.zeros((101, 51))
            >>> eye_pdf[50, 25] = 1.0  # Delta at center
            >>> result = injector.inject_jitter(eye_pdf, time_bins)
        """
        # Input validation
        if eye_pdf.size == 0:
            raise ValueError("Eye PDF array cannot be empty")
        if len(time_bins) == 0:
            raise ValueError("Time bins array cannot be empty")
        
        # Check dimensions match
        if eye_pdf.shape[0] != len(time_bins):
            raise ValueError(
                f"Eye PDF time dimension ({eye_pdf.shape[0]}) must match "
                f"time bins length ({len(time_bins)})"
            )
        
        # Check for negative PDF values
        if np.any(eye_pdf < 0):
            raise ValueError("Eye PDF values must be non-negative")
        
        # Special case: no jitter - return normalized copy
        if self.dj == 0 and self.rj == 0:
            result = eye_pdf.copy()
            total = np.sum(result)
            if total > 0:
                result = result / total
            return result
        
        # Generate jitter PDF
        jitter_pdf = self.generate_dual_dirac_pdf(time_bins)
        
        # Initialize result array
        result = np.zeros_like(eye_pdf)
        
        # Apply convolution for each voltage slice
        for v_idx in range(eye_pdf.shape[1]):
            time_slice = eye_pdf[:, v_idx]
            
            # Convolve with jitter PDF
            convolved = np.convolve(time_slice, jitter_pdf, mode='same')
            result[:, v_idx] = convolved
        
        # Normalize to preserve total probability
        result_sum = np.sum(result)
        if result_sum > 0:
            result = result / result_sum
        
        return result
    
    def calculate_tj(self, ber: float = 1e-12) -> float:
        """
        Calculate Total Jitter (TJ) at a specified BER level.
        
        For the Dual-Dirac model, total jitter at a given BER is:
            TJ = DJ + 2 * Q(BER) * RJ
        
        where Q(BER) is the Q-factor corresponding to the bit error rate.
        
        Args:
            ber: Bit Error Rate (default: 1e-12 for 1e-12 BER)
        
        Returns:
            Total Jitter in UI units at the specified BER
        
        Raises:
            ValueError: If ber is not in (0, 1)
        
        Reference:
            OIF-CEI-02.0 "Common Electrical I/O (CEI) - Electrical and 
            Jitter Interoperability agreements for 6G+ bps and 11G+ bps 
            I/O"
        
        Example:
            >>> injector = JitterInjector(dj=0.05, rj=0.01)
            >>> tj_1e12 = injector.calculate_tj(ber=1e-12)  # ~0.19 UI
            >>> tj_1e15 = injector.calculate_tj(ber=1e-15)  # Higher TJ
        """
        if not (0 < ber < 1):
            raise ValueError(f"BER must be in (0, 1), got {ber}")
        
        # Calculate Q-factor for given BER
        # Q = Φ^(-1)(1 - BER), where Φ is the standard normal CDF
        q_factor = stats.norm.ppf(1 - ber)
        
        # TJ = DJ + 2 * Q * RJ
        # Note: DJ is already half of peak-to-peak
        tj = self.dj + 2 * q_factor * self.rj
        
        return tj
