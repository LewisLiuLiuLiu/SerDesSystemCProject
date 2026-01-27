"""
Core Eye Analysis Module for EyeAnalyzer

This module provides the main EyeAnalyzer class for performing eye diagram analysis,
including eye diagram construction, eye height/width calculation, and visualization.
"""

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import validate_ui, validate_bins, validate_input_arrays, save_metrics_json, create_output_directory, q_function
from .jitter import JitterDecomposer


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
                 jitter_method: str = 'dual-dirac'):
        """
        Initialize the EyeAnalyzer.

        Args:
            ui: Unit interval in seconds (e.g., 2.5e-11 for 10Gbps)
            ui_bins: Number of bins for phase axis (default: 128)
            amp_bins: Number of bins for amplitude axis (default: 128)
            jitter_method: Jitter extraction method
                          ('dual-dirac', 'tail-fit', 'auto', default: 'dual-dirac')

        Raises:
            ValueError: If parameters are invalid
        """
        validate_ui(ui)
        validate_bins(ui_bins, "ui_bins")
        validate_bins(amp_bins, "amp_bins")

        self.ui = ui
        self.ui_bins = ui_bins
        self.amp_bins = amp_bins
        self.jitter_method = jitter_method

        # Initialize jitter decomposer
        self._jitter_decomposer = JitterDecomposer(ui, jitter_method)

    def analyze(self, time_array: np.ndarray, value_array: np.ndarray,
             target_ber: float = 1e-12) -> Dict[str, float]:
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
            - rj_sigma: Random jitter standard deviation (seconds)
            - dj_pp: Deterministic jitter peak-to-peak (seconds)
            - tj_at_ber: Total jitter at target BER (seconds)
            - target_ber: Target BER used for calculation
            - q_factor: Q function value at target BER
            - fit_method: Jitter extraction method used
            - fit_quality: R-squared value of fit (0-1)
            - pj_info: Periodic jitter detection info

        Raises:
            ValueError: If input arrays are invalid
        """
        validate_input_arrays(time_array, value_array)

        print(f"  Processing {len(time_array)} samples...")

        # Step 1: Normalize phase
        phase_array = self._normalize_phase(time_array)

        # Step 2: Build eye diagram (2D histogram)
        hist2d, xedges, yedges = self._build_eye_diagram(phase_array, value_array)

        print(f"  Eye diagram shape: {hist2d.shape}")

        # Step 3: Compute eye metrics
        eye_height = self._compute_eye_height(hist2d, yedges)
        eye_width = self._compute_eye_width(hist2d, xedges)

        print(f"  Eye height: {eye_height*1000:.2f} mV")
        print(f"  Eye width: {eye_width:.3f} UI")

        # Step 4: Extract jitter using decomposer
        print(f"  Extracting jitter components (BER={target_ber:.0e}, method={self.jitter_method})...")
        jitter_metrics = self._jitter_decomposer.extract(phase_array, value_array, target_ber)

        print(f"  RJ sigma: {jitter_metrics['rj_sigma']*1e12:.2f} ps")
        print(f"  DJ pp: {jitter_metrics['dj_pp']*1e12:.2f} ps")
        print(f"  TJ@{target_ber:.0e}: {jitter_metrics['tj_at_ber']*1e12:.2f} ps")
        print(f"  Fit method: {jitter_metrics['fit_method']}")

        # Store internal data for visualization
        self._hist2d = hist2d
        self._xedges = xedges
        self._yedges = yedges

        # Combine all metrics
        metrics = {
            'eye_height': float(eye_height),
            'eye_width': float(eye_width),
            **jitter_metrics
        }

        return metrics

    def save_results(self, metrics: Dict[str, float], output_dir: str = '.') -> None:
        """
        Save analysis results to files.

        Args:
            metrics: Analysis metrics dictionary
            output_dir: Output directory path
        """
        create_output_directory(output_dir)

        # Save metrics JSON
        metrics_path = os.path.join(output_dir, 'eye_metrics.json')
        save_metrics_json(metrics, metrics_path)
        print(f"  Saved metrics to: {metrics_path}")

        # Save eye diagram image
        if hasattr(self, '_hist2d'):
            image_path = os.path.join(output_dir, 'eye_diagram.png')
            self._plot_eye_diagram(metrics, image_path)
            print(f"  Saved image to: {image_path}")

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
            bins=[self.ui_bins, self.amp_bins]
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

    def _plot_eye_diagram(self, metrics: Dict[str, float], output_path: str) -> None:
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
            f"Eye Width: {metrics['eye_width']:.3f} UI"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()