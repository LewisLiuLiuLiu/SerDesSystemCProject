"""
Jitter Decomposition Module for EyeAnalyzer

This module provides the JitterDecomposer class for extracting jitter components
from eye diagram data using various methods (dual-dirac, tail-fit, auto).

Supports:
- Random Jitter (RJ) extraction via Gaussian fitting
- Deterministic Jitter (DJ) extraction via dual-Dirac model
- Periodic Jitter (PJ) detection via frequency domain analysis
- Multiple extraction methods: 'dual-dirac', 'tail-fit', 'auto'
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.optimize import curve_fit
from .utils import q_function, calculate_r_squared


class JitterDecomposer:
    """
    Jitter decomposition engine for EyeAnalyzer.

    Supports multiple extraction methods:
    - 'dual-dirac': Dual-Dirac model with Gaussian fitting
    - 'tail-fit': Tail-fitting method for non-bimodal DJ
    - 'auto': Auto-detect best method based on distribution

    Attributes:
        ui: Unit interval in seconds
        method: Extraction method ('dual-dirac', 'tail-fit', 'auto')

    Example:
        >>> decomposer = JitterDecomposer(ui=2.5e-11, method='dual-dirac')
        >>> metrics = decomposer.extract(phase_array, value_array, target_ber=1e-12)
        >>> print(f"RJ: {metrics['rj_sigma']*1e12:.2f} ps")
        >>> print(f"DJ: {metrics['dj_pp']*1e12:.2f} ps")
    """

    def __init__(self, ui: float, method: str = 'dual-dirac', psd_nperseg: int = 16384):
        """
        Initialize jitter decomposer.

        Args:
            ui: Unit interval in seconds
            method: Jitter extraction method
                   ('dual-dirac', 'tail-fit', 'auto')
            psd_nperseg: Number of samples per segment for PSD calculation
                        (affects frequency resolution, default: 16384)

        Raises:
            ValueError: If method is invalid
        """
        valid_methods = ['dual-dirac', 'tail-fit', 'auto']
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. "
                f"Valid methods: {valid_methods}"
            )

        self.ui = ui
        self.method = method
        self.psd_nperseg = psd_nperseg
        
        # Store last analysis data for CSV export
        self._last_timing_offsets = None
        self._last_histogram = None
        self._last_bin_edges = None

    def extract(self, phase_array: np.ndarray, value_array: np.ndarray,
                target_ber: float = 1e-12) -> Dict[str, Any]:
        """
        Extract jitter components from waveform data.

        This method performs the following steps:
        1. Extract zero-crossing timing offsets
        2. Select extraction method based on configuration
        3. Extract RJ and DJ using selected method
        4. Detect periodic jitter (PJ) components
        5. Calculate total jitter at target BER (TJ@BER)

        Args:
            phase_array: Phase array in [0, 1)
            value_array: Amplitude array in volts
            target_ber: Target bit error rate for TJ calculation

        Returns:
            Dictionary containing:
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds)
            - tj_at_ber: Total jitter at target BER (seconds)
            - target_ber: Target BER used for calculation
            - q_factor: Q function value at target BER
            - fit_method: Method used for extraction
            - fit_quality: R-squared value of fit (0-1)
            - pj_info: Dictionary with periodic jitter info
                - detected: Boolean indicating if PJ was detected
                - frequencies: List of detected frequencies (Hz)
                - amplitudes: List of detected amplitudes (seconds)
                - count: Number of detected PJ components

        Raises:
            ValueError: If insufficient zero-crossing points for analysis
        """
        # Step 1: Extract zero-crossing timing offsets
        timing_offsets = self._extract_timing_offsets(phase_array, value_array)
        
        # Store for CSV export
        self._last_timing_offsets = timing_offsets
        hist, bin_edges = np.histogram(timing_offsets, bins=200, density=True)
        self._last_histogram = hist
        self._last_bin_edges = bin_edges

        # Step 2: Select extraction method
        if self.method == 'auto':
            is_bimodal = self._detect_bimodal(timing_offsets)
            if is_bimodal:
                jitter_result = self._extract_dual_dirac(timing_offsets)
            else:
                jitter_result = self._extract_single_gaussian(timing_offsets)
        elif self.method == 'dual-dirac':
            jitter_result = self._extract_dual_dirac(timing_offsets)
        elif self.method == 'tail-fit':
            jitter_result = self._extract_tail_fit(timing_offsets, target_ber)
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown method: {self.method}")

        # Step 3: Detect periodic jitter
        pj_info = self._detect_periodic_jitter(timing_offsets)

        # Step 4: Calculate TJ@BER
        q_value = q_function(target_ber)
        tj_at_ber = jitter_result['dj_pp'] + 2 * q_value * jitter_result['rj_sigma']

        return {
            'rj_sigma': float(jitter_result['rj_sigma']),
            'dj_pp': float(jitter_result['dj_pp']),
            'tj_at_ber': float(tj_at_ber),
            'target_ber': target_ber,
            'q_factor': float(q_value),
            'fit_method': jitter_result['fit_method'],
            'fit_quality': jitter_result.get('fit_quality', 0.0),
            'pj_info': pj_info
        }

    def _extract_timing_offsets(self, phase_array: np.ndarray,
                                value_array: np.ndarray) -> np.ndarray:
        """
        Extract zero-crossing timing offsets.

        This method finds all zero-crossing points in the waveform and
        calculates the timing offset from the ideal crossing point (0.5 UI).

        Args:
            phase_array: Phase array in [0, 1)
            value_array: Amplitude array in volts

        Returns:
            Timing offsets wrapped to [-0.5, 0.5) range

        Raises:
            ValueError: If insufficient zero-crossing points (< 100)
        """
        threshold = 0.0
        crossing_indices = np.where(np.diff(np.signbit(value_array - threshold)))[0]

        if len(crossing_indices) < 100:
            raise ValueError(
                f"Insufficient zero-crossing points for jitter analysis: "
                f"found {len(crossing_indices)}, need at least 100"
            )

        crossing_phases = phase_array[crossing_indices]
        ideal_crossing = 0.5  # Mid-UI crossing
        timing_offsets = crossing_phases - ideal_crossing
        timing_offsets = ((timing_offsets + 0.5) % 1.0) - 0.5  # Wrap to [-0.5, 0.5)

        return timing_offsets

    def _detect_bimodal(self, timing_offsets: np.ndarray,
                       min_separation: float = 0.05) -> bool:
        """
        Detect if timing offset distribution is bimodal.

        Bimodality indicates the presence of deterministic jitter (DJ).
        This method builds a histogram and detects peaks using scipy.signal.find_peaks.

        Args:
            timing_offsets: Timing offsets wrapped to [-0.5, 0.5)
            min_separation: Minimum peak separation in UI (default: 0.05)

        Returns:
            True if distribution is bimodal, False otherwise
        """
        # Build histogram
        hist, bin_edges = np.histogram(timing_offsets, bins=200, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Detect peaks
        height_threshold = np.max(hist) * 0.1
        peaks, _ = find_peaks(hist, height=height_threshold)

        if len(peaks) < 2:
            return False

        # Check peak separation
        peak_separation = abs(bin_centers[peaks[0]] - bin_centers[peaks[-1]])
        return peak_separation > min_separation

    def _fit_dual_gaussian(self, x: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, float]:
        """
        Fit dual-Gaussian distribution to data.

        The dual-Gaussian model represents the dual-Dirac model:
        PDF(x) = a1 * Gaussian(x; mu1, sigma1) + a2 * Gaussian(x; mu2, sigma2)

        Args:
            x: X coordinates (timing offsets)
            y: Y coordinates (probability density)

        Returns:
            Tuple of (fitted_parameters, r_squared):
            - fitted_parameters: [a1, mu1, sigma1, a2, mu2, sigma2]
            - r_squared: R-squared value of fit (0-1)
        """
        def dual_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
            """Dual-Gaussian probability density function."""
            return (a1 / (sigma1 * np.sqrt(2 * np.pi)) *
                    np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) +
                    a2 / (sigma2 * np.sqrt(2 * np.pi)) *
                    np.exp(-0.5 * ((x - mu2) / sigma2) ** 2))

        # Initial guess from histogram peaks
        hist, _ = np.histogram(x, bins=len(y), density=True)
        peak_indices, _ = find_peaks(hist, height=np.max(hist) * 0.1)

        if len(peak_indices) >= 2:
            mu1_init = x[peak_indices[0]]
            mu2_init = x[peak_indices[-1]]
        else:
            # Fallback to symmetric peaks
            mu1_init, mu2_init = -0.1, 0.1

        initial_guess = [0.5, mu1_init, 0.02, 0.5, mu2_init, 0.02]

        try:
            popt, _ = curve_fit(dual_gaussian, x, y, p0=initial_guess, maxfev=10000)
            y_fit = dual_gaussian(x, *popt)
            r_squared = calculate_r_squared(y, y_fit)
            return popt, r_squared
        except RuntimeError:
            # Fallback to single Gaussian if dual fit fails
            return self._fit_single_gaussian(x, y)

    def _fit_single_gaussian(self, x: np.ndarray, y: np.ndarray
                            ) -> Tuple[np.ndarray, float]:
        """
        Fit single-Gaussian distribution to data.

        Used for pure random jitter (RJ) cases where DJ is negligible.

        Args:
            x: X coordinates (timing offsets)
            y: Y coordinates (probability density)

        Returns:
            Tuple of (fitted_parameters, r_squared):
            - fitted_parameters: [mu, sigma]
            - r_squared: R-squared value of fit (0-1)
        """
        def single_gaussian(x, mu, sigma):
            """Single-Gaussian probability density function."""
            return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        initial_guess = [0.0, 0.02]

        try:
            popt, _ = curve_fit(single_gaussian, x, y, p0=initial_guess, maxfev=10000)
            y_fit = single_gaussian(x, *popt)
            r_squared = calculate_r_squared(y, y_fit)
            return popt, r_squared
        except RuntimeError:
            # Fallback to simple statistics
            mu = np.mean(x)
            sigma = np.std(x)
            return np.array([mu, sigma]), 0.0

    def _extract_dual_dirac(self, timing_offsets: np.ndarray) -> Dict[str, float]:
        """
        Extract RJ and DJ using dual-Dirac model.

        The dual-Dirac model decomposes jitter into:
        - RJ (Random Jitter): Gaussian distributed, characterized by sigma
        - DJ (Deterministic Jitter): Bimodal distribution, characterized by peak-to-peak

        Algorithm:
        1. Build histogram of timing offsets
        2. Fit both single and dual Gaussian distributions
        3. Compare fit quality (R-squared improvement)
        4. If dual fit is significantly better (>5% R² improvement), use dual model
        5. Otherwise, use single model

        Args:
            timing_offsets: Timing offsets wrapped to [-0.5, 0.5)

        Returns:
            Dictionary containing:
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds)
            - fit_quality: R-squared value of fit (0-1)
            - fit_method: 'dual-gaussian' or 'single-gaussian'
        """
        # Build histogram
        hist, bin_edges = np.histogram(timing_offsets, bins=200, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit single Gaussian
        params_single, r2_single = self._fit_single_gaussian(bin_centers, hist)

        # Fit dual Gaussian
        params_dual, r2_dual = self._fit_dual_gaussian(bin_centers, hist)

        # Compare fit quality
        # Use dual Gaussian only if it provides significant improvement (>5% R²)
        r2_improvement = r2_dual - r2_single
        use_dual = r2_improvement > 0.05 and len(params_dual) > 2

        if use_dual and len(params_dual) > 2:
            # Dual Gaussian (RJ + DJ)
            a1, mu1, sigma1, a2, mu2, sigma2 = params_dual
            rj_sigma = (sigma1 + sigma2) / 2 * self.ui
            dj_pp = abs(mu2 - mu1) * self.ui
            fit_method = 'dual-gaussian'
            r_squared = r2_dual
        else:
            # Single Gaussian (pure RJ)
            mu, sigma = params_single
            rj_sigma = sigma * self.ui
            dj_pp = 0.0
            fit_method = 'single-gaussian'
            r_squared = r2_single

        return {
            'rj_sigma': rj_sigma,
            'dj_pp': dj_pp,
            'fit_quality': r_squared,
            'fit_method': fit_method
        }

    def _extract_single_gaussian(self, timing_offsets: np.ndarray
                                 ) -> Dict[str, float]:
        """
        Extract RJ using single-Gaussian fitting (for pure RJ cases).

        This method is used when DJ is negligible and the distribution
        is unimodal (single peak).

        Args:
            timing_offsets: Timing offsets wrapped to [-0.5, 0.5)

        Returns:
            Dictionary containing:
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds, always 0)
            - fit_quality: R-squared value of fit (0-1)
            - fit_method: 'single-gaussian'
        """
        # Build histogram
        hist, bin_edges = np.histogram(timing_offsets, bins=200, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit single Gaussian
        params, r_squared = self._fit_single_gaussian(bin_centers, hist)
        mu, sigma = params

        return {
            'rj_sigma': sigma * self.ui,
            'dj_pp': 0.0,
            'fit_quality': r_squared,
            'fit_method': 'single-gaussian'
        }

    def _extract_tail_fit(self, timing_offsets: np.ndarray,
                         target_ber: float = 1e-12) -> Dict[str, float]:
        """
        Extract RJ and DJ using tail-fitting method.

        The tail-fitting method is useful for non-bimodal DJ distributions
        where the dual-Dirac model may not be appropriate.

        Algorithm:
        1. Build high-resolution histogram
        2. Extract tail regions (beyond 2 sigma from mean)
        3. Fit Gaussian to tails to extract RJ
        4. Estimate DJ from CDF width (5th to 95th percentile)

        Args:
            timing_offsets: Timing offsets wrapped to [-0.5, 0.5)
            target_ber: Target bit error rate for TJ calculation

        Returns:
            Dictionary containing:
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds)
            - fit_quality: R-squared value of fit (always 0 for tail-fit)
            - fit_method: 'tail-fit'
        """
        # Build high-resolution histogram
        hist, bin_edges = np.histogram(timing_offsets, bins=400, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Extract tails (beyond 1.5 sigma)
        mean_offset = np.mean(timing_offsets)
        std_offset = np.std(timing_offsets)
        tail_mask = np.abs(bin_centers - mean_offset) > 1.5 * std_offset

        # Check if we have enough tail data
        if np.sum(tail_mask) < 10:
            # Not enough tail data, fall back to single Gaussian
            return self._extract_single_gaussian(timing_offsets)

        # Fit Gaussian to tails
        def gaussian(x, mu, sigma):
            """Gaussian probability density function."""
            return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        try:
            popt, _ = curve_fit(gaussian, bin_centers[tail_mask], hist[tail_mask],
                              p0=[mean_offset, std_offset], maxfev=10000)
            mu, sigma = popt

            rj_sigma = sigma * self.ui

            # Estimate DJ from CDF width (5th to 95th percentile)
            cdf = np.cumsum(hist) * (bin_edges[1] - bin_edges[0])
            left_idx = np.searchsorted(cdf, 0.05 * cdf[-1])
            right_idx = np.searchsorted(cdf, 0.95 * cdf[-1])
            dj_pp = (bin_centers[right_idx] - bin_centers[left_idx]) * self.ui

            return {
                'rj_sigma': rj_sigma,
                'dj_pp': dj_pp,
                'fit_quality': 0.0,  # Not applicable for tail-fit
                'fit_method': 'tail-fit'
            }
        except RuntimeError:
            # Fallback to single Gaussian if tail fit fails
            return self._extract_single_gaussian(timing_offsets)

    def _detect_periodic_jitter(self, timing_offsets: np.ndarray
                               ) -> Dict[str, Any]:
        """
        Detect periodic jitter (PJ) using frequency domain analysis.

        Periodic jitter appears as narrowband peaks in the power spectral density (PSD).
        This method uses Welch's method to compute the PSD and detects significant peaks.

        Args:
            timing_offsets: Timing offsets wrapped to [-0.5, 0.5)

        Returns:
            Dictionary containing:
            - detected: Boolean indicating if PJ was detected
            - frequencies: List of detected frequencies (Hz)
            - amplitudes: List of detected amplitudes (seconds)
            - count: Number of detected PJ components
        """
        # Need sufficient data for frequency analysis
        if len(timing_offsets) < 200:
            return {
                'detected': False,
                'frequencies': [],
                'amplitudes': [],
                'count': 0
            }

        # Compute PSD using Welch's method
        fs = 1.0 / self.ui  # Sampling rate
        # Use configured nperseg, but ensure it doesn't exceed data length
        nperseg = min(self.psd_nperseg, len(timing_offsets))
        f, Pxx = welch(timing_offsets, fs=fs, nperseg=nperseg)

        # Detect peaks (periodic jitter appears as narrowband peaks)
        peak_threshold = np.mean(Pxx) + 3 * np.std(Pxx)
        peaks, _ = find_peaks(Pxx, height=peak_threshold)

        if len(peaks) == 0:
            return {
                'detected': False,
                'frequencies': [],
                'amplitudes': [],
                'count': 0
            }

        # Filter out DC component and very low frequencies (< 1 MHz)
        valid_mask = f[peaks] > 1e6
        valid_peaks = peaks[valid_mask]

        if len(valid_peaks) == 0:
            return {
                'detected': False,
                'frequencies': [],
                'amplitudes': [],
                'count': 0
            }

        # Extract peak information
        pj_frequencies = f[valid_peaks]
        # Convert PSD peak to amplitude estimate
        pj_amplitudes = np.sqrt(Pxx[valid_peaks] * f[valid_peaks])

        return {
            'detected': True,
            'frequencies': pj_frequencies.tolist(),
            'amplitudes': pj_amplitudes.tolist(),
            'count': len(pj_frequencies)
        }

    def get_jitter_distribution_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get jitter distribution data for CSV export.

        Returns:
            Tuple of (time_offsets_seconds, probabilities):
            - time_offsets_seconds: Time offset values in seconds
            - probabilities: Probability density values

        Raises:
            ValueError: If extract() has not been called yet
        """
        if self._last_histogram is None or self._last_bin_edges is None:
            raise ValueError("No jitter analysis data available. Call extract() first.")

        # Convert bin centers from UI to seconds
        bin_centers = (self._last_bin_edges[:-1] + self._last_bin_edges[1:]) / 2
        time_offsets_seconds = bin_centers * self.ui
        probabilities = self._last_histogram

        return time_offsets_seconds, probabilities