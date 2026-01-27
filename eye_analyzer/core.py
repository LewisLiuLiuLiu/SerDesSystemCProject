"""
Core Eye Analysis Module for EyeAnalyzer

This module provides the main EyeAnalyzer class for performing eye diagram analysis,
including eye diagram construction, eye height/width calculation, and visualization.
"""

import os
from typing import Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import welch

from .utils import (
    validate_ui, validate_bins, validate_input_arrays, 
    save_metrics_json, save_metrics_json_spec, create_output_directory, q_function,
    save_hist2d_csv, save_psd_csv, save_pdf_csv, save_jitter_distribution_csv,
    format_metrics_to_spec
)
from .jitter import JitterDecomposer
from .io import auto_load_waveform


class EyeAnalyzer:
    """
    Eye diagram analyzer for SerDes link simulation data.

    This class performs eye diagram analysis including:
    - Eye diagram construction (phase normalization + 2D histogram)
    - Eye height and eye width calculation
    - Visualization and metrics export

    Example:
        >>> analyzer = EyeAnalyzer(ui=2.5e-11, ui_bins=128, amp_bins=128)
        >>> metrics = analyzer.analyze(time_array, value_array)
        >>> analyzer.save_results(metrics, output_dir='results/')
    """

    def __init__(self, ui: float, ui_bins: int = 128, amp_bins: int = 128,
                 jitter_method: str = 'dual-dirac',
                 measure_length: Optional[float] = None,
                 sampling: str = 'phase-lock',
                 hist2d_normalize: bool = True,
                 psd_nperseg: int = 16384,
                 linearity_threshold: float = 0.1,
                 output_image_format: str = 'png',
                 output_image_dpi: int = 300,
                 save_csv_data: bool = False,
                 csv_data_path: str = 'eye_analysis_data'):
        """
        Initialize the EyeAnalyzer.

        Args:
            ui: Unit interval in seconds (e.g., 2.5e-11 for 10Gbps)
            ui_bins: Number of bins for phase axis (default: 128)
            amp_bins: Number of bins for amplitude axis (default: 128)
            jitter_method: Jitter extraction method
                          ('dual-dirac', 'tail-fit', 'auto', default: 'dual-dirac')
            measure_length: Duration in seconds to analyze from end of waveform.
                           If None, use all data (default: None)
            sampling: Sampling phase estimation strategy
                     ('peak', 'zero-cross', 'phase-lock', default: 'phase-lock')
            hist2d_normalize: Whether to normalize 2D histogram to probability density
                             (default: True)
            psd_nperseg: Number of samples per segment for PSD calculation
                        (default: 16384)
            linearity_threshold: Amplitude threshold for linearity error calculation
                                (default: 0.1)
            output_image_format: Output image format ('png', 'svg', 'pdf', default: 'png')
            output_image_dpi: Output image resolution in DPI (default: 300)
            save_csv_data: Whether to save CSV auxiliary data (default: False)
            csv_data_path: Directory path for CSV data output (default: 'eye_analysis_data')

        Raises:
            ValueError: If parameters are invalid
        """
        validate_ui(ui)
        validate_bins(ui_bins, "ui_bins")
        validate_bins(amp_bins, "amp_bins")
        
        # Validate sampling strategy
        valid_sampling = ['peak', 'zero-cross', 'phase-lock']
        if sampling not in valid_sampling:
            raise ValueError(f"Invalid sampling '{sampling}'. Valid options: {valid_sampling}")
        
        # Validate image format
        valid_formats = ['png', 'svg', 'pdf']
        if output_image_format not in valid_formats:
            raise ValueError(f"Invalid output_image_format '{output_image_format}'. Valid options: {valid_formats}")

        self.ui = ui
        self.ui_bins = ui_bins
        self.amp_bins = amp_bins
        self.jitter_method = jitter_method
        self.measure_length = measure_length
        self.sampling = sampling
        self.hist2d_normalize = hist2d_normalize
        self.psd_nperseg = psd_nperseg
        self.linearity_threshold = linearity_threshold
        self.output_image_format = output_image_format
        self.output_image_dpi = output_image_dpi
        self.save_csv_data = save_csv_data
        self.csv_data_path = csv_data_path
        
        # Data provenance tracking
        self._total_samples = 0
        self._analyzed_samples = 0
        self._sampling_rate = 0.0
        self._duration = 0.0
        self._dat_path = ''

        # Initialize jitter decomposer with psd_nperseg
        self._jitter_decomposer = JitterDecomposer(ui, jitter_method, psd_nperseg)

    def analyze(self, time_array: np.ndarray, value_array: np.ndarray,
             target_ber: float = 1e-12) -> Dict[str, Any]:
        """
        Perform eye diagram analysis with jitter decomposition.

        Args:
            time_array: Time array in seconds
            value_array: Signal value array in volts
            target_ber: Target bit error rate for TJ calculation (default: 1e-12)

        Returns:
            Dictionary containing analysis metrics:
            - eye_height: Eye height in volts
            - eye_width: Eye width in unit intervals (UI)
            - eye_area: Eye opening area (V * UI)
            - linearity_error: Linearity error (normalized)
            - optimal_sampling_phase: Optimal sampling phase (0-1)
            - optimal_threshold: Optimal decision threshold (V)
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds)
            - tj_at_ber: Total jitter at target BER (seconds)
            - target_ber: Target BER used for calculation
            - q_factor: Q function value at target BER
            - fit_method: Jitter extraction method used
            - fit_quality: R-squared value of fit (0-1)
            - pj_info: Periodic jitter detection info
            - signal_mean: Signal mean value (V)
            - signal_rms: Signal RMS value (V)
            - signal_peak_to_peak: Signal peak-to-peak value (V)
            - psd_peak_freq: PSD peak frequency (Hz)
            - psd_peak_value: PSD peak value (dB)
            - total_samples: Total number of input samples
            - analyzed_samples: Number of samples after truncation
            - sampling_rate: Estimated sampling rate (Hz)
            - duration: Analysis duration (s)

        Raises:
            ValueError: If input arrays are invalid
        """
        validate_input_arrays(time_array, value_array)
        
        # Track data provenance
        self._total_samples = len(time_array)
        if len(time_array) > 1:
            self._sampling_rate = 1.0 / (time_array[1] - time_array[0])
        self._duration = time_array[-1] - time_array[0] if len(time_array) > 1 else 0.0

        # Step 0: Truncate waveform if measure_length is specified
        time_array, value_array = self._truncate_waveform(time_array, value_array)
        self._analyzed_samples = len(time_array)

        print(f"  Processing {len(time_array)} samples...")

        # Step 1: Normalize phase using configured sampling strategy
        phase_array = self._normalize_phase(time_array)

        # Step 2: Build eye diagram (2D histogram)
        hist2d, xedges, yedges = self._build_eye_diagram(phase_array, value_array)

        print(f"  Eye diagram shape: {hist2d.shape}")

        # Step 3: Compute eye metrics
        eye_height = self._compute_eye_height(hist2d, yedges)
        eye_width = self._compute_eye_width(hist2d, xedges)
        eye_area = self._compute_eye_area(hist2d, xedges, yedges)
        linearity_error = self._compute_linearity_error(hist2d, xedges, yedges)
        optimal_phase, optimal_threshold = self._compute_optimal_phase_and_threshold(hist2d, xedges, yedges)

        print(f"  Eye height: {eye_height*1000:.2f} mV")
        print(f"  Eye width: {eye_width:.3f} UI")
        print(f"  Eye area: {eye_area*1000:.2f} mV*UI")

        # Step 4: Extract jitter using decomposer
        print(f"  Extracting jitter components (BER={target_ber:.0e}, method={self.jitter_method})...")
        jitter_metrics = self._jitter_decomposer.extract(phase_array, value_array, target_ber)

        print(f"  RJ sigma: {jitter_metrics['rj_sigma']*1e12:.2f} ps")
        print(f"  DJ pp: {jitter_metrics['dj_pp']*1e12:.2f} ps")
        print(f"  TJ@{target_ber:.0e}: {jitter_metrics['tj_at_ber']*1e12:.2f} ps")
        print(f"  Fit method: {jitter_metrics['fit_method']}")

        # Step 5: Compute signal quality metrics
        signal_quality = self._compute_signal_quality(value_array)

        # Store internal data for visualization and CSV export
        self._hist2d = hist2d
        self._xedges = xedges
        self._yedges = yedges
        self._value_array = value_array
        self._phase_array = phase_array

        # Combine all metrics
        metrics = {
            'eye_height': float(eye_height),
            'eye_width': float(eye_width),
            'eye_area': float(eye_area),
            'linearity_error': float(linearity_error),
            'optimal_sampling_phase': float(optimal_phase),
            'optimal_threshold': float(optimal_threshold),
            **jitter_metrics,
            **signal_quality,
            # Data provenance
            'total_samples': self._total_samples,
            'analyzed_samples': self._analyzed_samples,
            'sampling_rate': self._sampling_rate,
            'duration': self._duration
        }

        return metrics

    def save_results(self, metrics: Dict[str, Any], output_dir: str = '.') -> None:
        """
        Save analysis results to files.

        Args:
            metrics: Analysis metrics dictionary
            output_dir: Output directory path
        """
        create_output_directory(output_dir)

        # Build metadata for JSON spec format
        metadata = {
            'dat_path': self._dat_path,
            'ui': self.ui,
            'ui_bins': self.ui_bins,
            'amp_bins': self.amp_bins,
            'measure_length': self.measure_length
        }

        # Save metrics JSON using spec format
        metrics_path = os.path.join(output_dir, 'eye_metrics.json')
        save_metrics_json_spec(metrics, metadata, metrics_path)
        print(f"  Saved metrics to: {metrics_path}")

        # Save eye diagram image
        if hasattr(self, '_hist2d'):
            image_ext = self.output_image_format
            image_path = os.path.join(output_dir, f'eye_diagram.{image_ext}')
            self._plot_eye_diagram(metrics, image_path)
            print(f"  Saved image to: {image_path}")

        # Save CSV data if enabled
        if self.save_csv_data:
            self._save_csv_data(output_dir, metrics)

    def _normalize_phase(self, time_array: np.ndarray) -> np.ndarray:
        """
        Normalize time to phase in [0, 1) range.

        Phase is computed as: phi = (t % UI) / UI

        Args:
            time_array: Time array in seconds

        Returns:
            Phase array normalized to [0, 1)
        """
        return (time_array % self.ui) / self.ui

    def _build_eye_diagram(self, phase_array: np.ndarray,
                          amplitude_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 2D histogram eye diagram.

        Args:
            phase_array: Phase array in [0, 1)
            amplitude_array: Amplitude array in volts

        Returns:
            Tuple of (hist2d, xedges, yedges):
            - hist2d: 2D histogram matrix (shape: ui_bins x amp_bins)
            - xedges: Phase bin edges (length: ui_bins + 1)
            - yedges: Amplitude bin edges (length: amp_bins + 1)
        """
        hist2d, xedges, yedges = np.histogram2d(
            phase_array,
            amplitude_array,
            bins=[self.ui_bins, self.amp_bins],
            density=self.hist2d_normalize
        )

        return hist2d, xedges, yedges

    def _compute_eye_height(self, hist2d: np.ndarray,
                           yedges: np.ndarray) -> float:
        """
        Compute eye height from 2D histogram.

        Eye height is the maximum vertical opening at the optimal sampling phase.
        This simplified version finds the maximum vertical extent of non-zero density.

        Args:
            hist2d: 2D histogram matrix
            yedges: Amplitude bin edges

        Returns:
            Eye height in volts
        """
        # Find phase with maximum density (optimal sampling phase)
        phase_density = np.sum(hist2d, axis=1)
        optimal_phase_idx = np.argmax(phase_density)

        # Find amplitude range at optimal phase
        amplitude_profile = hist2d[optimal_phase_idx, :]

        # Find first and last non-zero bins
        nonzero_indices = np.where(amplitude_profile > 0)[0]

        if len(nonzero_indices) == 0:
            return 0.0

        y_min_idx = nonzero_indices[0]
        y_max_idx = nonzero_indices[-1]

        # Compute eye height
        eye_height = yedges[y_max_idx + 1] - yedges[y_min_idx]

        return eye_height

    def _compute_eye_width(self, hist2d: np.ndarray,
                          xedges: np.ndarray) -> float:
        """
        Compute eye width from 2D histogram.

        Eye width is the maximum horizontal opening at the optimal threshold.
        This simplified version finds the maximum horizontal extent of non-zero density.

        Args:
            hist2d: 2D histogram matrix
            xedges: Phase bin edges

        Returns:
            Eye width in unit intervals (UI)
        """
        # Find amplitude with maximum density (optimal threshold)
        amplitude_density = np.sum(hist2d, axis=0)
        optimal_amp_idx = np.argmax(amplitude_density)

        # Find phase range at optimal amplitude
        phase_profile = hist2d[:, optimal_amp_idx]

        # Find first and last non-zero bins
        nonzero_indices = np.where(phase_profile > 0)[0]

        if len(nonzero_indices) == 0:
            return 0.0

        x_min_idx = nonzero_indices[0]
        x_max_idx = nonzero_indices[-1]

        # Compute eye width
        eye_width = xedges[x_max_idx + 1] - xedges[x_min_idx]

        return eye_width

    def _plot_eye_diagram(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Plot and save eye diagram visualization.

        Args:
            metrics: Analysis metrics dictionary
            output_path: Path to save the image file
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot 2D histogram as heatmap
        im = ax.imshow(
            self._hist2d.T,
            origin='lower',
            aspect='auto',
            extent=[self._xedges[0], self._xedges[-1], self._yedges[0], self._yedges[-1]],
            cmap='hot'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density', rotation=270, labelpad=20)

        # Set labels and title
        ax.set_xlabel('Phase (UI)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title('Eye Diagram')

        # Add metrics annotation
        textstr = (
            f"UI: {self.ui:.2e} s\n"
            f"Eye Height: {metrics['eye_height']*1000:.2f} mV\n"
            f"Eye Width: {metrics['eye_width']:.3f} UI\n"
            f"Eye Area: {metrics.get('eye_area', 0)*1000:.2f} mV*UI"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save figure with configured DPI
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_image_dpi, bbox_inches='tight')
        plt.close()

    # ========================================================================
    # New methods per EyeAnalyzer.md specification
    # ========================================================================

    def _truncate_waveform(self, time_array: np.ndarray, 
                           value_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Truncate waveform to measure_length from the end.

        Per EyeAnalyzer.md: Use measure_length to extract stable data from the end
        of the waveform, avoiding startup transients.

        Args:
            time_array: Full time array in seconds
            value_array: Full signal value array in volts

        Returns:
            Tuple of (truncated_time_array, truncated_value_array)
        """
        if self.measure_length is None:
            return time_array, value_array

        total_duration = time_array[-1] - time_array[0]
        
        if self.measure_length >= total_duration:
            print(f"  Warning: measure_length ({self.measure_length:.2e}s) >= total duration "
                  f"({total_duration:.2e}s), using all data")
            return time_array, value_array

        # Calculate start time for truncation
        t_start = time_array[-1] - self.measure_length
        
        # Find index where time >= t_start
        start_idx = np.searchsorted(time_array, t_start)
        
        return time_array[start_idx:], value_array[start_idx:]

    def _estimate_sampling_phase(self, phase_array: np.ndarray, 
                                  value_array: np.ndarray) -> float:
        """
        Estimate optimal sampling phase using configured strategy.

        Strategies per EyeAnalyzer.md:
        - 'peak': Sample at signal amplitude peak (max SNR)
        - 'zero-cross': Sample at zero-crossing points (for clock recovery)
        - 'phase-lock': Use ideal mid-UI sampling (0.5)

        Args:
            phase_array: Phase array in [0, 1)
            value_array: Amplitude array in volts

        Returns:
            Optimal sampling phase in [0, 1)
        """
        if self.sampling == 'phase-lock':
            # Default: mid-UI sampling
            return 0.5
        
        elif self.sampling == 'peak':
            # Find phase with maximum absolute amplitude
            # Bin data by phase and find bin with highest peak amplitude
            hist, bin_edges = np.histogram(phase_array, bins=self.ui_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            max_amp_per_bin = np.zeros(self.ui_bins)
            for i in range(self.ui_bins):
                mask = (phase_array >= bin_edges[i]) & (phase_array < bin_edges[i+1])
                if np.any(mask):
                    max_amp_per_bin[i] = np.max(np.abs(value_array[mask]))
            
            optimal_idx = np.argmax(max_amp_per_bin)
            return bin_centers[optimal_idx]
        
        elif self.sampling == 'zero-cross':
            # Find phase closest to zero-crossing
            threshold = 0.0
            crossing_indices = np.where(np.diff(np.signbit(value_array - threshold)))[0]
            
            if len(crossing_indices) == 0:
                return 0.5  # Fallback
            
            crossing_phases = phase_array[crossing_indices]
            # Return median crossing phase
            return float(np.median(crossing_phases))
        
        return 0.5  # Default fallback

    def _compute_eye_area(self, hist2d: np.ndarray, 
                          xedges: np.ndarray, yedges: np.ndarray) -> float:
        """
        Compute eye opening area.

        Per EyeAnalyzer.md:
        eye_area = integral over eye opening region of H(phi, V) dphi dV

        This is approximated as the sum of density in the eye opening region
        multiplied by the bin area.

        Args:
            hist2d: 2D histogram matrix
            xedges: Phase bin edges
            yedges: Amplitude bin edges

        Returns:
            Eye opening area in V * UI
        """
        # Find the eye opening region (low density area in center)
        # Use threshold-based detection
        if np.max(hist2d) == 0:
            return 0.0
        
        # Normalize if not already normalized
        if self.hist2d_normalize:
            density_threshold = np.mean(hist2d) * 0.1
        else:
            density_threshold = np.max(hist2d) * 0.01
        
        # Find eye opening as region with density below threshold
        eye_opening_mask = hist2d < density_threshold
        
        # Calculate bin dimensions
        dx = xedges[1] - xedges[0]  # Phase bin width (UI)
        dy = yedges[1] - yedges[0]  # Amplitude bin width (V)
        bin_area = dx * dy
        
        # Sum area of eye opening bins
        eye_area = np.sum(eye_opening_mask) * bin_area
        
        return float(eye_area)

    def _compute_linearity_error(self, hist2d: np.ndarray,
                                  xedges: np.ndarray, yedges: np.ndarray) -> float:
        """
        Compute linearity error in the eye opening region.

        Per EyeAnalyzer.md:
        linearity_error = RMS(V_actual - V_linear_fit)

        This measures signal distortion by fitting a linear model to the
        eye opening region.

        Args:
            hist2d: 2D histogram matrix
            xedges: Phase bin edges
            yedges: Amplitude bin edges

        Returns:
            Linearity error (normalized, 0-1 range)
        """
        if not hasattr(self, '_value_array') or self._value_array is None:
            return 0.0
        
        value_array = self._value_array
        
        # Calculate expected linear behavior
        v_max = np.max(value_array)
        v_min = np.min(value_array)
        v_range = v_max - v_min
        
        if v_range == 0:
            return 0.0
        
        # Focus on samples within linearity_threshold of the signal range
        threshold_range = v_range * self.linearity_threshold
        
        # For a differential signal centered at 0, measure deviation from ideal
        # The "ideal" is the signal staying at either high or low level
        # Linearity error captures transition region distortion
        
        # Simple approach: measure RMS of deviations from ±v_max/2 levels
        high_level = v_max / 2
        low_level = v_min / 2
        
        # Calculate deviation from nearest expected level
        deviations = np.minimum(
            np.abs(value_array - high_level),
            np.abs(value_array - low_level)
        )
        
        # Normalize by signal range
        linearity_error = np.sqrt(np.mean(deviations ** 2)) / v_range
        
        return float(linearity_error)

    def _compute_optimal_phase_and_threshold(self, hist2d: np.ndarray,
                                              xedges: np.ndarray, 
                                              yedges: np.ndarray) -> Tuple[float, float]:
        """
        Compute optimal sampling phase and decision threshold.

        Per EyeAnalyzer.md:
        - optimal_sampling_phase: Phase with maximum eye height
        - optimal_threshold: Amplitude level for best BER

        Args:
            hist2d: 2D histogram matrix
            xedges: Phase bin edges
            yedges: Amplitude bin edges

        Returns:
            Tuple of (optimal_phase, optimal_threshold)
        """
        bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
        bin_centers_y = (yedges[:-1] + yedges[1:]) / 2
        
        # Find optimal phase (column with minimum density in center)
        # This corresponds to the widest eye opening
        center_amp_idx = len(bin_centers_y) // 2
        center_range = slice(center_amp_idx - 5, center_amp_idx + 5)
        
        if center_amp_idx >= 5 and center_amp_idx + 5 <= len(bin_centers_y):
            center_density = np.sum(hist2d[:, center_range], axis=1)
            optimal_phase_idx = np.argmin(center_density)
        else:
            # Fallback: use middle phase
            optimal_phase_idx = len(bin_centers_x) // 2
        
        optimal_phase = float(bin_centers_x[optimal_phase_idx])
        
        # Find optimal threshold (amplitude with minimum density at optimal phase)
        phase_profile = hist2d[optimal_phase_idx, :]
        
        # The optimal threshold is typically at the center of the eye
        # Find the amplitude with minimum density
        if np.any(phase_profile > 0):
            # Find center of low-density region
            threshold_idx = np.argmin(phase_profile)
            optimal_threshold = float(bin_centers_y[threshold_idx])
        else:
            optimal_threshold = float(np.mean(bin_centers_y))
        
        return optimal_phase, optimal_threshold

    def _compute_signal_quality(self, value_array: np.ndarray) -> Dict[str, float]:
        """
        Compute signal quality metrics.

        Per EyeAnalyzer.md, computes:
        - mean: Signal mean value
        - rms: Signal RMS value
        - peak_to_peak: Signal peak-to-peak amplitude
        - psd_peak_freq: Peak frequency in PSD
        - psd_peak_value: Peak value in PSD (dB)

        Args:
            value_array: Signal amplitude array in volts

        Returns:
            Dictionary with signal quality metrics
        """
        # Basic statistics
        signal_mean = float(np.mean(value_array))
        signal_rms = float(np.sqrt(np.mean(value_array ** 2)))
        signal_peak_to_peak = float(np.max(value_array) - np.min(value_array))
        
        # PSD analysis
        psd_peak_freq = 0.0
        psd_peak_value = 0.0
        
        if len(value_array) > self.psd_nperseg:
            try:
                fs = self._sampling_rate if self._sampling_rate > 0 else 1.0 / self.ui
                nperseg = min(self.psd_nperseg, len(value_array))
                f, Pxx = welch(value_array, fs=fs, nperseg=nperseg)
                
                # Store for CSV export
                self._psd_frequencies = f
                self._psd_values = Pxx
                
                # Find peak (excluding DC)
                if len(f) > 1:
                    # Skip DC component
                    peak_idx = np.argmax(Pxx[1:]) + 1
                    psd_peak_freq = float(f[peak_idx])
                    # Convert to dB
                    psd_peak_value = float(10 * np.log10(Pxx[peak_idx] + 1e-20))
            except Exception:
                pass
        
        # Compute PDF for CSV export
        pdf_values, pdf_edges = np.histogram(value_array, bins=256, density=True)
        pdf_centers = (pdf_edges[:-1] + pdf_edges[1:]) / 2
        self._pdf_amplitudes = pdf_centers
        self._pdf_values = pdf_values
        
        return {
            'signal_mean': signal_mean,
            'signal_rms': signal_rms,
            'signal_peak_to_peak': signal_peak_to_peak,
            'psd_peak_freq': psd_peak_freq,
            'psd_peak_value': psd_peak_value
        }

    def _save_csv_data(self, output_dir: str, metrics: Dict[str, Any]) -> None:
        """
        Save auxiliary analysis data to CSV files.

        Per EyeAnalyzer.md, saves:
        - hist2d.csv: 2D density matrix
        - psd.csv: Power spectral density
        - pdf.csv: Probability density function
        - jitter_distribution.csv: Jitter timing distribution

        Args:
            output_dir: Output directory path
            metrics: Analysis metrics dictionary
        """
        csv_dir = os.path.join(output_dir, self.csv_data_path)
        create_output_directory(csv_dir)
        
        print(f"  Saving CSV data to: {csv_dir}")
        
        # Save hist2d
        if hasattr(self, '_hist2d'):
            save_hist2d_csv(self._hist2d, self._xedges, self._yedges,
                           os.path.join(csv_dir, 'hist2d.csv'))
        
        # Save PSD
        if hasattr(self, '_psd_frequencies') and hasattr(self, '_psd_values'):
            save_psd_csv(self._psd_frequencies, self._psd_values,
                        os.path.join(csv_dir, 'psd.csv'))
        
        # Save PDF
        if hasattr(self, '_pdf_amplitudes') and hasattr(self, '_pdf_values'):
            save_pdf_csv(self._pdf_amplitudes, self._pdf_values,
                        os.path.join(csv_dir, 'pdf.csv'))
        
        # Save jitter distribution
        try:
            time_offsets, probabilities = self._jitter_decomposer.get_jitter_distribution_data()
            save_jitter_distribution_csv(time_offsets, probabilities,
                                         os.path.join(csv_dir, 'jitter_distribution.csv'))
        except ValueError:
            pass  # No jitter data available


# ============================================================================
# Convenience function per EyeAnalyzer.md specification
# ============================================================================

def analyze_eye(dat_path: Optional[str] = None,
                waveform_array: Optional[np.ndarray] = None,
                ui: Optional[float] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Convenience function for eye diagram analysis.

    This is the main entry point per EyeAnalyzer.md specification.
    Supports both file-based and memory-based input.

    Args:
        dat_path: Path to waveform data file (SystemC-AMS tabular format)
        waveform_array: Memory waveform array, shape (N, 2), columns: time, value
                       (mutually exclusive with dat_path)
        ui: Unit interval in seconds (required)
        **kwargs: Additional arguments passed to EyeAnalyzer:
            - ui_bins: Phase axis resolution (default: 128)
            - amp_bins: Amplitude axis resolution (default: 128)
            - measure_length: Duration to analyze from end (default: None)
            - target_ber: Target BER for TJ calculation (default: 1e-12)
            - sampling: Sampling strategy ('peak', 'zero-cross', 'phase-lock')
            - jitter_method: Jitter extraction method ('dual-dirac', 'tail-fit', 'auto')
            - hist2d_normalize: Normalize histogram to PDF (default: True)
            - psd_nperseg: PSD segment size (default: 16384)
            - linearity_threshold: Linearity calculation threshold (default: 0.1)
            - output_image_format: Image format ('png', 'svg', 'pdf')
            - output_image_dpi: Image resolution (default: 300)
            - save_csv_data: Save CSV auxiliary data (default: False)
            - csv_data_path: CSV output directory (default: 'eye_analysis_data')

    Returns:
        Dictionary containing all analysis metrics in EyeAnalyzer.md format

    Raises:
        ValueError: If neither dat_path nor waveform_array is provided,
                   or if both are provided, or if ui is not specified

    Example:
        >>> # From file
        >>> metrics = analyze_eye(dat_path='results.dat', ui=2.5e-11)
        
        >>> # From memory array
        >>> metrics = analyze_eye(waveform_array=data, ui=2.5e-11, measure_length=2.5e-6)
    """
    # Validate input
    if dat_path is None and waveform_array is None:
        raise ValueError("Either dat_path or waveform_array must be provided")
    
    if dat_path is not None and waveform_array is not None:
        raise ValueError("Only one of dat_path or waveform_array can be provided")
    
    if ui is None:
        raise ValueError("ui (unit interval) must be specified")
    
    # Extract target_ber from kwargs
    target_ber = kwargs.pop('target_ber', 1e-12)
    
    # Load waveform data
    if dat_path is not None:
        time_array, value_array = auto_load_waveform(dat_path)
    else:
        # waveform_array shape: (N, 2) with columns [time, value]
        if waveform_array.ndim != 2 or waveform_array.shape[1] < 2:
            raise ValueError("waveform_array must have shape (N, 2) with columns [time, value]")
        time_array = waveform_array[:, 0]
        value_array = waveform_array[:, 1]
    
    # Create analyzer with remaining kwargs
    analyzer = EyeAnalyzer(ui=ui, **kwargs)
    
    # Store dat_path for metadata
    analyzer._dat_path = dat_path or ''
    
    # Run analysis
    metrics = analyzer.analyze(time_array, value_array, target_ber=target_ber)
    
    return metrics