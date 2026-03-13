#!/usr/bin/env python3
"""
JitterTolerance - SJ (Sinusoidal Jitter) Tolerance Testing.

This module implements Jitter Tolerance (JTol) testing for high-speed serial links.
It performs SJ frequency and amplitude sweeps to measure the maximum tolerable
sinusoidal jitter at different modulation frequencies, and compares results
against industry standard templates.

Key Features:
- SJ frequency sweep across specified range
- SJ amplitude sweep using binary search for efficiency
- Support for multiple industry standard templates (IEEE 802.3ck, OIF-CEI-112G, etc.)
- Pass/Fail determination with margin calculation
- Visualization of JTol curves

Algorithm:
1. For each frequency point in the SJ frequency sweep:
   - Use binary search to find maximum tolerable SJ amplitude
   - Eye closure at target BER indicates SJ limit
2. Compare measured SJ limits against template requirements
3. Calculate margin at each frequency point
4. Determine overall Pass/Fail

Example:
    >>> from eye_analyzer.ber.jtol import JitterTolerance
    >>> from eye_analyzer.schemes.statistical import StatisticalScheme
    >>> 
    >>> jtol = JitterTolerance(modulation='nrz', target_ber=1e-12)
    >>> scheme = StatisticalScheme(ui=2.5e-11, modulation='nrz')
    >>> 
    >>> sj_frequencies = np.logspace(5, 9, 20)  # 100 kHz to 1 GHz
    >>> pulse_response = load_pulse_response('channel.s4p')
    >>> 
    >>> results = jtol.measure_jtol(
    ...     eye_analyzer=scheme,
    ...     sj_frequencies=sj_frequencies,
    ...     template='ieee_802_3ck',
    ...     pulse_response=pulse_response
    ... )
    >>> 
    >>> print(f"Overall Pass: {results['overall_pass']}")
    >>> print(f"Margins: {results['margins']}")
    >>> 
    >>> jtol.plot_jtol(results, output_file='jtol_curve.png')
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from .template import JTolTemplate
from ..schemes.statistical import StatisticalScheme

# Setup logging
logger = logging.getLogger(__name__)


class JitterTolerance:
    """
    Jitter Tolerance (JTol) tester for high-speed serial links.
    
    Measures the maximum tolerable Sinusoidal Jitter (SJ) at various modulation
    frequencies and compares against industry standard templates.
    
    Attributes:
        modulation: Modulation format ('nrz' or 'pam4')
        target_ber: Target bit error rate for eye opening measurement
        
    Example:
        >>> jtol = JitterTolerance(modulation='nrz', target_ber=1e-12)
        >>> 
        >>> # Measure JTol curve
        >>> sj_freqs = np.logspace(5, 9, 10)  # 100 kHz to 1 GHz
        >>> results = jtol.measure_jtol(
        ...     eye_analyzer=scheme,
        ...     sj_frequencies=sj_freqs,
        ...     template='ieee_802_3ck',
        ...     pulse_response=pulse
        ... )
        >>> 
        >>> # Check compliance
        >>> if results['overall_pass']:
        ...     print("JTol compliance: PASS")
        ... else:
        ...     print(f"JTol compliance: FAIL - min margin: {min(results['margins']):.3f} UI")
    """
    
    # Valid modulation formats
    VALID_MODULATIONS = ['nrz', 'pam4']
    
    def __init__(self, modulation: str = 'nrz', target_ber: float = 1e-12):
        """
        Initialize JitterTolerance tester.
        
        Args:
            modulation: Modulation format. Options: 'nrz', 'pam4'
            target_ber: Target bit error rate for eye opening measurement.
                       Default is 1e-12.
        
        Raises:
            ValueError: If modulation is not a valid format.
        
        Example:
            >>> jtol = JitterTolerance(modulation='nrz', target_ber=1e-12)
            >>> jtol = JitterTolerance(modulation='pam4', target_ber=1e-15)
        """
        if modulation.lower() not in self.VALID_MODULATIONS:
            raise ValueError(
                f"Invalid modulation '{modulation}'. "
                f"Valid options: {self.VALID_MODULATIONS}"
            )
        
        self.modulation = modulation.lower()
        self.target_ber = target_ber
        
        logger.debug(
            f"JitterTolerance initialized: modulation={self.modulation}, "
            f"target_ber={self.target_ber}"
        )
    
    def measure_jtol(
        self,
        eye_analyzer: StatisticalScheme,
        sj_frequencies: np.ndarray,
        template: str = 'ieee_802_3ck',
        pulse_response: Optional[np.ndarray] = None,
        min_sj: float = 0.001,
        max_sj: float = 0.5,
        sj_tolerance: float = 0.005,
        noise_sigma: float = 0.0,
        rj: float = 0.0
    ) -> Dict[str, Any]:
        """
        Measure JTol curve by sweeping SJ frequency and amplitude.
        
        For each frequency point, performs binary search to find the maximum
        SJ amplitude that still maintains the target BER eye opening.
        
        Args:
            eye_analyzer: StatisticalScheme instance for eye analysis
            sj_frequencies: Array of SJ modulation frequencies in Hz
            template: Template name for comparison. Options:
                     - 'ieee_802_3ck': IEEE 802.3ck (200G/400G Ethernet)
                     - 'oif_cei_112g': OIF-CEI-112G
                     - 'jedec_ddr5': JEDEC DDR5
                     - 'pcie_gen6': PCIe Gen6
            pulse_response: Channel pulse response array (required for statistical scheme)
            min_sj: Minimum SJ amplitude to test in UI (default: 0.001)
            max_sj: Maximum SJ amplitude to test in UI (default: 0.5)
            sj_tolerance: Convergence tolerance for SJ search in UI (default: 0.005)
            noise_sigma: Gaussian noise sigma to apply during measurement
            rj: Random jitter to apply during measurement (in UI)
        
        Returns:
            Dictionary containing:
            - frequencies: Array of test frequencies in Hz
            - sj_limits: Array of measured SJ limits in UI
            - template_limits: Array of template SJ limits in UI
            - margins: Array of margins (measured - template) in UI
            - pass_fail: List of Pass/Fail per frequency point
            - overall_pass: True if all points pass, False otherwise
            - modulation: Modulation format used
            - target_ber: Target BER used
        
        Raises:
            ValueError: If pulse_response is not provided for statistical scheme
            ValueError: If template name is invalid
        
        Example:
            >>> scheme = StatisticalScheme(ui=2.5e-11, modulation='nrz')
            >>> sj_freqs = np.array([1e5, 1e6, 1e7, 1e8])
            >>> pulse = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
            >>> 
            >>> results = jtol.measure_jtol(
            ...     eye_analyzer=scheme,
            ...     sj_frequencies=sj_freqs,
            ...     template='ieee_802_3ck',
            ...     pulse_response=pulse
            ... )
        """
        if pulse_response is None:
            raise ValueError("pulse_response is required for statistical scheme")
        
        # Validate template
        if template not in JTolTemplate.TEMPLATES:
            valid_templates = list(JTolTemplate.TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template}'. "
                f"Valid templates: {valid_templates}"
            )
        
        sj_frequencies = np.asarray(sj_frequencies)
        sj_limits = []
        
        logger.info(f"Starting JTol measurement for {len(sj_frequencies)} frequency points")
        
        # Measure SJ limit at each frequency
        for i, freq in enumerate(sj_frequencies):
            logger.debug(f"Measuring SJ limit at frequency {freq/1e6:.2f} MHz")
            
            sj_limit = self._find_sj_limit_at_frequency(
                eye_analyzer=eye_analyzer,
                frequency=freq,
                pulse_response=pulse_response,
                min_sj=min_sj,
                max_sj=max_sj,
                tolerance=sj_tolerance,
                noise_sigma=noise_sigma,
                rj=rj
            )
            
            sj_limits.append(sj_limit)
            logger.debug(f"  SJ limit: {sj_limit:.4f} UI")
        
        sj_limits = np.array(sj_limits)
        
        # Compare with template
        comparison = self.compare_with_template(
            measured_frequencies=sj_frequencies,
            measured_sj=sj_limits,
            template_name=template
        )
        
        # Build result dictionary
        result = {
            'frequencies': sj_frequencies,
            'sj_limits': sj_limits,
            'template_limits': comparison['template_limits'],
            'margins': comparison['margins'],
            'pass_fail': comparison['pass_fail'],
            'overall_pass': comparison['overall_pass'],
            'modulation': self.modulation,
            'target_ber': self.target_ber
        }
        
        logger.info(
            f"JTol measurement complete: overall_pass={result['overall_pass']}"
        )
        
        return result
    
    def _find_sj_limit_at_frequency(
        self,
        eye_analyzer: StatisticalScheme,
        frequency: float,
        pulse_response: np.ndarray,
        min_sj: float = 0.001,
        max_sj: float = 0.5,
        tolerance: float = 0.005,
        noise_sigma: float = 0.0,
        rj: float = 0.0
    ) -> float:
        """
        Find maximum SJ amplitude at a specific frequency using binary search.
        
        Performs binary search to find the SJ amplitude at which the eye
        closes to the target BER level.
        
        Args:
            eye_analyzer: StatisticalScheme instance for eye analysis
            frequency: SJ modulation frequency in Hz
            pulse_response: Channel pulse response array
            min_sj: Minimum SJ amplitude in UI
            max_sj: Maximum SJ amplitude in UI
            tolerance: Convergence tolerance in UI
            noise_sigma: Gaussian noise sigma
            rj: Random jitter in UI
        
        Returns:
            Maximum SJ amplitude in UI that maintains target BER eye opening
        
        Note:
            The algorithm assumes eye closure increases monotonically with SJ.
            Uses eye_height as the metric for eye closure detection.
        """
        low = min_sj
        high = max_sj
        
        # Binary search for SJ limit
        while (high - low) > tolerance:
            mid = (low + high) / 2.0
            
            # Analyze eye with current SJ
            try:
                metrics = eye_analyzer.analyze(
                    pulse_response=pulse_response,
                    noise_sigma=noise_sigma,
                    dj=mid,  # Use DJ as SJ for this test
                    rj=rj,
                    target_ber=self.target_ber
                )
                
                # Check if eye is still open
                eye_height = metrics.get('eye_height', 0.0)
                eye_open = eye_height > 0.01  # Threshold for eye opening
                
                if eye_open:
                    # Eye still open, can increase SJ
                    low = mid
                else:
                    # Eye closed, need to decrease SJ
                    high = mid
                    
            except Exception as e:
                logger.warning(f"Eye analysis failed at SJ={mid}: {e}")
                # Assume eye closed on error
                high = mid
        
        # Return the converged SJ limit
        return (low + high) / 2.0
    
    def compare_with_template(
        self,
        measured_frequencies: np.ndarray,
        measured_sj: np.ndarray,
        template_name: str = 'ieee_802_3ck'
    ) -> Dict[str, Any]:
        """
        Compare measured SJ limits with template requirements.
        
        Calculates margin at each frequency point and determines Pass/Fail
        for each point and overall.
        
        Args:
            measured_frequencies: Array of measured frequency points in Hz
            measured_sj: Array of measured SJ limits in UI
            template_name: Name of template to compare against
        
        Returns:
            Dictionary containing:
            - frequencies: Array of frequency points in Hz
            - measured_sj: Array of measured SJ limits in UI
            - template_limits: Array of template SJ limits in UI
            - margins: Array of margins (measured - template) in UI
            - pass_fail: List of bool, True if point passes
            - overall_pass: True if all points pass
        
        Raises:
            ValueError: If template name is unknown
            ValueError: If frequency and SJ arrays have different lengths
        
        Example:
            >>> freqs = np.array([1e5, 1e6, 1e7])
            >>> measured = np.array([0.12, 0.11, 0.08])
            >>> result = jtol.compare_with_template(freqs, measured, 'ieee_802_3ck')
            >>> print(f"Pass: {result['overall_pass']}")
            >>> print(f"Min margin: {min(result['margins']):.3f} UI")
        """
        # Validate inputs
        if template_name not in JTolTemplate.TEMPLATES:
            valid_templates = list(JTolTemplate.TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Valid templates: {valid_templates}"
            )
        
        measured_frequencies = np.asarray(measured_frequencies)
        measured_sj = np.asarray(measured_sj)
        
        if len(measured_frequencies) != len(measured_sj):
            raise ValueError(
                f"Frequencies and measured_sj must have same length. "
                f"Got {len(measured_frequencies)} and {len(measured_sj)}"
            )
        
        # Load template
        template = JTolTemplate(template_name)
        
        # Get template limits at measured frequencies
        template_limits = template.evaluate(measured_frequencies)
        
        # Calculate margins (measured - template)
        margins = measured_sj - template_limits
        
        # Determine Pass/Fail for each point
        # Pass if measured >= template (margin >= 0)
        pass_fail = margins >= 0
        pass_fail_list = pass_fail.tolist()
        
        # Overall pass if all points pass
        overall_pass = bool(np.all(pass_fail))
        
        return {
            'frequencies': measured_frequencies,
            'measured_sj': measured_sj,
            'template_limits': template_limits,
            'margins': margins,
            'pass_fail': pass_fail_list,
            'overall_pass': overall_pass
        }
    
    def plot_jtol(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """
        Plot JTol curve with template comparison.
        
        Creates a plot showing measured SJ limits vs. template requirements,
        with pass/fail indication and margin information.
        
        Args:
            results: Results dictionary from measure_jtol() or compare_with_template()
            output_file: Optional file path to save plot (e.g., 'jtol.png')
            figsize: Figure size as (width, height) in inches
            title: Optional plot title (auto-generated if None)
        
        Returns:
            Tuple of (figure, axes) objects
        
        Example:
            >>> results = jtol.measure_jtol(...)
            >>> fig, ax = jtol.plot_jtol(results, output_file='jtol_curve.png')
            >>> 
            >>> # Or plot existing comparison
            >>> comparison = jtol.compare_with_template(freqs, measured, 'ieee_802_3ck')
            >>> fig, ax = jtol.plot_jtol(comparison)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        frequencies = results['frequencies']
        sj_limits = results.get('sj_limits', results.get('measured_sj'))
        template_limits = results['template_limits']
        margins = results['margins']
        pass_fail = results['pass_fail']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot template limit
        ax.semilogx(
            frequencies, template_limits,
            'b-', linewidth=2, label='Template Limit', marker='s', markersize=6
        )
        
        # Plot measured SJ limits
        ax.semilogx(
            frequencies, sj_limits,
            'r-o', linewidth=2, label='Measured SJ Limit', markersize=6
        )
        
        # Mark pass/fail points
        pass_freqs = [f for f, pf in zip(frequencies, pass_fail) if pf]
        pass_sj = [sj for sj, pf in zip(sj_limits, pass_fail) if pf]
        fail_freqs = [f for f, pf in zip(frequencies, pass_fail) if not pf]
        fail_sj = [sj for sj, pf in zip(sj_limits, pass_fail) if not pf]
        
        if pass_freqs:
            ax.scatter(
                pass_freqs, pass_sj,
                c='green', s=100, marker='o', label='Pass', zorder=5, edgecolors='black'
            )
        
        if fail_freqs:
            ax.scatter(
                fail_freqs, fail_sj,
                c='red', s=100, marker='x', label='Fail', zorder=5, linewidths=2
            )
        
        # Labels and grid
        ax.set_xlabel('SJ Frequency (Hz)', fontsize=12)
        ax.set_ylabel('SJ Amplitude (UI)', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            overall_pass = results.get('overall_pass', False)
            status = "PASS" if overall_pass else "FAIL"
            ax.set_title(
                f'Jitter Tolerance Curve - {status} (Target BER: {self.target_ber})',
                fontsize=14
            )
        
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=10)
        
        # Add margin annotation
        min_margin = np.min(margins)
        min_margin_freq = frequencies[np.argmin(margins)]
        ax.annotate(
            f'Min Margin: {min_margin:.4f} UI\\n@ {min_margin_freq/1e6:.2f} MHz',
            xy=(min_margin_freq, sj_limits[np.argmin(margins)]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
            fontsize=9
        )
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"JTol plot saved to {output_file}")
        
        return fig, ax
    
    def generate_sj_frequency_sweep(
        self,
        f_start: float = 1e4,
        f_stop: float = 1e9,
        num_points: int = 20,
        scale: str = 'log'
    ) -> np.ndarray:
        """
        Generate SJ frequency sweep points.
        
        Convenience method to generate logarithmically or linearly spaced
        frequency points for JTol testing.
        
        Args:
            f_start: Start frequency in Hz (default: 10 kHz)
            f_stop: Stop frequency in Hz (default: 1 GHz)
            num_points: Number of frequency points (default: 20)
            scale: Spacing scale ('log' or 'linear')
        
        Returns:
            Array of frequency points in Hz
        
        Example:
            >>> jtol = JitterTolerance()
            >>> freqs = jtol.generate_sj_frequency_sweep(
            ...     f_start=1e5, f_stop=1e9, num_points=15
            ... )
        """
        if scale == 'log':
            return np.logspace(np.log10(f_start), np.log10(f_stop), num_points)
        else:
            return np.linspace(f_start, f_stop, num_points)
    
    def __repr__(self) -> str:
        """String representation of JitterTolerance."""
        return (
            f"JitterTolerance("
            f"modulation='{self.modulation}', "
            f"target_ber={self.target_ber})"
        )
