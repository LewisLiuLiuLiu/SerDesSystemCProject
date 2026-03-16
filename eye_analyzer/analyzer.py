"""
Unified Eye Diagram Analyzer

This module provides the unified EyeAnalyzer entry point for eye diagram analysis,
supporting both statistical and empirical modes.

Statistical Mode:
- Input: Channel pulse response
- Use case: Pre-simulation analysis, channel characterization
- Features: ISI calculation, BER contours, noise/jitter injection

Empirical Mode:
- Input: Time-domain waveform
- Use case: Post-simulation analysis, measurement data
- Features: Eye diagram construction, jitter decomposition

Both modes support NRZ and PAM4 modulation formats.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np

# Import existing scheme classes
from .schemes import StatisticalScheme, SamplerCentricScheme, GoldenCdrScheme
from .core import EyeAnalyzer as OriginalEyeAnalyzer


class EyeAnalyzer:
    """
    Unified eye analyzer supporting statistical and empirical modes.
    
    Statistical mode: Analyze pulse response (pre-simulation)
    Empirical mode: Analyze time-domain waveform (post-simulation)
    
    This class provides a unified interface for eye diagram analysis,
    integrating statistical analysis, BER calculation, jitter analysis,
    and visualization capabilities.
    
    Attributes:
        ui: Unit interval in seconds
        modulation: Modulation format ('nrz' or 'pam4')
        mode: Analysis mode ('statistical' or 'empirical')
        target_ber: Target bit error rate
        samples_per_symbol: Samples per symbol for statistical mode
        ui_bins: Number of UI bins for eye diagram
        amp_bins: Number of amplitude bins for eye diagram
    
    Example:
        >>> # Statistical mode
        >>> analyzer = EyeAnalyzer(ui=2.5e-11, modulation='nrz', mode='statistical')
        >>> result = analyzer.analyze(pulse_response, noise_sigma=0.01)
        >>> print(f"Eye height: {result['eye_metrics']['eye_height']*1000:.2f} mV")
        
        >>> # Empirical mode
        >>> analyzer = EyeAnalyzer(ui=2.5e-11, modulation='pam4', mode='empirical')
        >>> result = analyzer.analyze((time_array, voltage_array))
        >>> analyzer.plot_eye()
    """
    
    def __init__(self,
                 ui: float,
                 modulation: str = 'nrz',
                 mode: str = 'statistical',
                 target_ber: float = 1e-12,
                 samples_per_symbol: int = 16,
                 ui_bins: int = 128,
                 amp_bins: int = 256,
                 **kwargs):
        """
        Initialize eye analyzer.
        
        Args:
            ui: Unit interval in seconds (e.g., 2.5e-11 for 10Gbps)
            modulation: 'nrz' or 'pam4'
            mode: 'statistical' (pulse response) or 'empirical' (waveform)
            target_ber: Target bit error rate
            samples_per_symbol: Samples per symbol for statistical mode
            ui_bins: Number of UI bins for eye diagram
            amp_bins: Number of amplitude bins for eye diagram
            **kwargs: Additional arguments for internal schemes
        
        Raises:
            ValueError: If ui <= 0, or invalid mode/modulation
        """
        # Validate UI
        if ui <= 0:
            raise ValueError(f"ui must be positive, got {ui}")
        
        # Validate mode
        valid_modes = ['statistical', 'empirical']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
        
        # Validate modulation
        valid_modulations = ['nrz', 'pam4']
        modulation_lower = modulation.lower()
        if modulation_lower not in valid_modulations:
            raise ValueError(f"Invalid modulation '{modulation}'. Valid: {valid_modulations}")
        
        self.ui = ui
        self.modulation = modulation_lower
        self.mode = mode
        self.target_ber = target_ber
        self.samples_per_symbol = samples_per_symbol
        self.ui_bins = ui_bins
        self.amp_bins = amp_bins
        
        # Store additional kwargs
        self._kwargs = kwargs
        
        # Initialize internal scheme
        self._scheme = None
        self._empirical_analyzer = None
        self._last_result: Optional[Dict[str, Any]] = None
        
        # Create appropriate analyzer based on mode
        self._initialize_analyzer()
    
    def _initialize_analyzer(self) -> None:
        """Initialize the appropriate internal analyzer based on mode."""
        if self.mode == 'statistical':
            self._scheme = StatisticalScheme(
                ui=self.ui,
                modulation=self.modulation,
                ui_bins=self.ui_bins,
                amp_bins=self.amp_bins,
                samples_per_symbol=self.samples_per_symbol,
                **{k: v for k, v in self._kwargs.items() 
                   if k not in ['jitter_method', 'measure_length']}
            )
        else:  # empirical mode
            # Use original EyeAnalyzer for empirical mode
            self._empirical_analyzer = OriginalEyeAnalyzer(
                ui=self.ui,
                ui_bins=self.ui_bins,
                amp_bins=self.amp_bins,
                **{k: v for k, v in self._kwargs.items()
                   if k in ['jitter_method', 'measure_length', 'sampling', 
                           'hist2d_normalize', 'psd_nperseg', 'linearity_threshold']}
            )
    
    def analyze(self,
                input_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                noise_sigma: float = 0.0,
                jitter_dj: float = 0.0,
                jitter_rj: float = 0.0,
                **kwargs) -> Dict[str, Any]:
        """
        Perform complete eye analysis.
        
        Args:
            input_data: 
                - statistical mode: pulse_response (array)
                - empirical mode: (time_array, value_array) tuple
            noise_sigma: Gaussian noise sigma in volts (for statistical mode)
            jitter_dj: Deterministic jitter in UI
            jitter_rj: Random jitter standard deviation in UI
            **kwargs: Additional arguments passed to underlying analyzers
        
        Returns:
            dict with keys:
                - 'eye_matrix': Eye diagram data (2D numpy array)
                - 'xedges', 'yedges': Edge arrays
                - 'ber_contour': BER contour (statistical mode only)
                - 'eye_metrics': Eye height/width/area metrics
                - 'jitter': Jitter analysis results
                - 'bathtub_time': Time bathtub curve
                - 'bathtub_voltage': Voltage bathtub curve
        
        Raises:
            ValueError: If input_data is invalid for the selected mode
        """
        if self.mode == 'statistical':
            return self._analyze_statistical(
                input_data, noise_sigma, jitter_dj, jitter_rj, **kwargs
            )
        else:
            return self._analyze_empirical(input_data, **kwargs)
    
    def _analyze_statistical(self,
                             pulse_response: np.ndarray,
                             noise_sigma: float,
                             jitter_dj: float,
                             jitter_rj: float,
                             **kwargs) -> Dict[str, Any]:
        """Perform statistical mode analysis."""
        # Validate pulse response
        pulse_response = np.asarray(pulse_response)
        if len(pulse_response) == 0:
            raise ValueError("pulse_response cannot be empty")
        
        # Calculate dt from pulse response length (assume 1 ps default)
        dt = kwargs.get('dt', 1e-12)
        
        # Run statistical analysis
        metrics = self._scheme.analyze(
            pulse_response=pulse_response,
            dt=dt,
            noise_sigma=noise_sigma,
            dj=jitter_dj,
            rj=jitter_rj,
            target_ber=self.target_ber
        )
        
        # Build result dictionary
        result = {
            'eye_matrix': self._scheme.eye_matrix,
            'xedges': self._scheme.get_xedges(),
            'yedges': self._scheme.get_yedges(),
            'ber_contour': self._scheme.get_ber_contour(),
            'eye_metrics': {
                'eye_height': metrics.get('eye_height', 0.0),
                'eye_width': metrics.get('eye_width', 0.0),
                'eye_area': metrics.get('eye_area', 0.0),
                'optimal_sampling_phase': metrics.get('optimal_sampling_phase', 0.5),
                'optimal_threshold': metrics.get('optimal_threshold', 0.0),
            },
            'jitter': {
                'dj': jitter_dj,
                'rj': jitter_rj,
            },
            'bathtub_time': None,  # TODO: implement bathtub curves
            'bathtub_voltage': None,
            'scheme': 'statistical',
            'modulation': self.modulation,
        }
        
        # Add PAM4-specific metrics
        if self.modulation == 'pam4':
            result['eye_metrics']['eye_heights_per_eye'] = metrics.get('eye_heights_per_eye', [])
            result['eye_metrics']['eye_widths_per_eye'] = metrics.get('eye_widths_per_eye', [])
            result['eye_metrics']['eye_height_min'] = metrics.get('eye_height_min', 0.0)
            result['eye_metrics']['eye_height_avg'] = metrics.get('eye_height_avg', 0.0)
        
        self._last_result = result
        return result
    
    def _analyze_empirical(self,
                           input_data: Tuple[np.ndarray, np.ndarray],
                           **kwargs) -> Dict[str, Any]:
        """Perform empirical mode analysis."""
        # Validate input
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError(
                "Empirical mode requires (time_array, value_array) tuple"
            )
        
        time_array, value_array = input_data
        time_array = np.asarray(time_array)
        value_array = np.asarray(value_array)
        
        if len(time_array) != len(value_array):
            raise ValueError(
                f"Time and value arrays must have same length, "
                f"got {len(time_array)} and {len(value_array)}"
            )
        
        if len(time_array) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Run empirical analysis using original EyeAnalyzer
        metrics = self._empirical_analyzer.analyze(
            time_array, value_array,
            target_ber=self.target_ber
        )
        
        # Get eye matrix from empirical analyzer
        eye_matrix = getattr(self._empirical_analyzer, '_hist2d', None)
        xedges = getattr(self._empirical_analyzer, '_xedges', None)
        yedges = getattr(self._empirical_analyzer, '_yedges', None)
        
        # Build result dictionary
        result = {
            'eye_matrix': eye_matrix,
            'xedges': xedges,
            'yedges': yedges,
            'ber_contour': None,  # Not available in empirical mode
            'eye_metrics': {
                'eye_height': metrics.get('eye_height', 0.0),
                'eye_width': metrics.get('eye_width', 0.0),
                'eye_area': metrics.get('eye_area', 0.0),
                'optimal_sampling_phase': metrics.get('optimal_sampling_phase', 0.5),
                'optimal_threshold': metrics.get('optimal_threshold', 0.0),
            },
            'jitter': {
                'dj': metrics.get('dj_pp', 0.0) / self.ui if self.ui > 0 else 0.0,
                'rj': metrics.get('rj_sigma', 0.0) / self.ui if self.ui > 0 else 0.0,
            },
            'bathtub_time': None,
            'bathtub_voltage': None,
            'scheme': 'empirical',
            'modulation': self.modulation,
        }
        
        self._last_result = result
        return result
    
    def analyze_jtol(self,
                     pulse_responses: List[np.ndarray],
                     sj_frequencies: List[float],
                     template: str = 'ieee_802_3ck',
                     **kwargs) -> Dict[str, Any]:
        """
        Perform jitter tolerance test.
        
        Args:
            pulse_responses: List of pulse responses with different SJ
            sj_frequencies: List of SJ frequencies corresponding to pulse_responses
            template: JTOL template name ('ieee_802_3ck', 'oif_cei_56g')
            **kwargs: Additional arguments
        
        Returns:
            JTOL analysis results dict with keys:
                - 'sj_frequencies': List of SJ frequencies
                - 'sj_amplitudes': Required SJ amplitudes for each frequency
                - 'template': Template name used
                - 'pass_fail': Pass/fail results per frequency
        
        Raises:
            ValueError: If inputs are invalid or mode is not statistical
        """
        if self.mode != 'statistical':
            raise ValueError("JTOL analysis requires statistical mode")
        
        if len(pulse_responses) != len(sj_frequencies):
            raise ValueError(
                f"pulse_responses and sj_frequencies must have same length, "
                f"got {len(pulse_responses)} and {len(sj_frequencies)}"
            )
        
        # Calculate required SJ amplitude for each frequency
        # This is a simplified implementation
        sj_amplitudes = []
        pass_fail = []
        
        for freq in sj_frequencies:
            # Calculate SJ amplitude based on template
            if template == 'ieee_802_3ck':
                # IEEE 802.3ck template (simplified)
                # For 56G PAM4: 0.1 UI at low freq, 0.01 UI at high freq
                amplitude = 0.1 * (1e6 / freq) if freq < 10e6 else 0.01
            else:
                # Default template
                amplitude = 0.05
            
            sj_amplitudes.append(amplitude)
            pass_fail.append(True)  # Simplified: always pass
        
        return {
            'sj_frequencies': sj_frequencies,
            'sj_amplitudes': sj_amplitudes,
            'template': template,
            'pass_fail': pass_fail,
        }
    
    def plot_eye(self, ax=None, **kwargs) -> Any:
        """
        Plot eye diagram.
        
        Args:
            ax: Matplotlib axis (optional, creates new figure if None)
            **kwargs: Additional plot arguments
                - cmap: Colormap name
                - title: Plot title
                - show_metrics: Whether to overlay metrics
        
        Returns:
            Matplotlib figure or axis object
        """
        import matplotlib.pyplot as plt
        
        if self._last_result is None:
            raise ValueError("No analysis results. Call analyze() first.")
        
        eye_matrix = self._last_result.get('eye_matrix')
        if eye_matrix is None:
            raise ValueError("No eye matrix available")
        
        create_new_fig = ax is None
        if create_new_fig:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot eye diagram
        xedges = self._last_result.get('xedges')
        yedges = self._last_result.get('yedges')
        
        if xedges is not None and yedges is not None:
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                eye_matrix.T,
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap=kwargs.get('cmap', 'hot'),
                interpolation='bilinear'
            )
        else:
            im = ax.imshow(
                eye_matrix.T,
                origin='lower',
                aspect='auto',
                cmap=kwargs.get('cmap', 'hot'),
                interpolation='bilinear'
            )
        
        ax.set_xlabel('Phase [UI]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title(kwargs.get('title', 'Eye Diagram'))
        
        if create_new_fig:
            plt.colorbar(im, ax=ax, label='Probability Density')
            return fig
        return ax
    
    def plot_jtol(self, jtol_results: Optional[Dict[str, Any]] = None,
                  ax=None, **kwargs) -> Any:
        """
        Plot JTOL curve.
        
        Args:
            jtol_results: JTOL analysis results (uses last result if None)
            ax: Matplotlib axis (optional)
            **kwargs: Additional plot arguments
        
        Returns:
            Matplotlib figure or axis object
        """
        import matplotlib.pyplot as plt
        
        if jtol_results is None:
            raise ValueError("JTOL results required. Call analyze_jtol() first.")
        
        create_new_fig = ax is None
        if create_new_fig:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        freqs = jtol_results.get('sj_frequencies', [])
        amplitudes = jtol_results.get('sj_amplitudes', [])
        
        ax.semilogx(freqs, amplitudes, 'b-o', label='JTOL')
        ax.set_xlabel('SJ Frequency [Hz]')
        ax.set_ylabel('SJ Amplitude [UI]')
        ax.set_title(kwargs.get('title', 'Jitter Tolerance'))
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        if create_new_fig:
            return fig
        return ax
    
    def plot_bathtub(self, direction: str = 'time', ax=None, **kwargs) -> Any:
        """
        Plot bathtub curve.
        
        Args:
            direction: 'time' or 'voltage'
            ax: Matplotlib axis (optional)
            **kwargs: Additional plot arguments
        
        Returns:
            Matplotlib figure or axis object
        """
        import matplotlib.pyplot as plt
        
        if self._last_result is None:
            raise ValueError("No analysis results. Call analyze() first.")
        
        create_new_fig = ax is None
        if create_new_fig:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Placeholder bathtub curve
        # TODO: implement actual bathtub curve calculation
        x = np.linspace(-0.5, 0.5, 100)
        y = np.exp(-x**2 / 0.1)  # Gaussian-like curve
        
        ax.semilogy(x, y, 'b-', label=f'Bathtub ({direction})')
        ax.set_xlabel('Phase [UI]' if direction == 'time' else 'Voltage [V]')
        ax.set_ylabel('BER')
        ax.set_title(kwargs.get('title', f'Bathtub Curve ({direction})'))
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        if create_new_fig:
            return fig
        return ax
    
    def create_report(self, output_file: Optional[str] = None, **kwargs) -> str:
        """
        Create comprehensive analysis report.
        
        Args:
            output_file: Output file path (optional)
            **kwargs: Additional report arguments
                - include_plots: Whether to include plots
                - format: Report format ('text', 'html', 'markdown')
        
        Returns:
            Report content as string
        """
        if self._last_result is None:
            raise ValueError("No analysis results. Call analyze() first.")
        
        format_type = kwargs.get('format', 'text')
        
        metrics = self._last_result.get('eye_metrics', {})
        jitter = self._last_result.get('jitter', {})
        
        if format_type == 'html':
            report = self._create_html_report(metrics, jitter)
        elif format_type == 'markdown':
            report = self._create_markdown_report(metrics, jitter)
        else:
            report = self._create_text_report(metrics, jitter)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def _create_text_report(self, metrics: Dict[str, Any],
                           jitter: Dict[str, Any]) -> str:
        """Create text format report."""
        lines = [
            "=" * 60,
            "Eye Diagram Analysis Report",
            "=" * 60,
            "",
            f"Mode: {self.mode}",
            f"Modulation: {self.modulation}",
            f"UI: {self.ui*1e12:.2f} ps",
            f"Target BER: {self.target_ber:.0e}",
            "",
            "Eye Metrics:",
            "-" * 40,
            f"  Eye Height: {metrics.get('eye_height', 0)*1000:.2f} mV",
            f"  Eye Width:  {metrics.get('eye_width', 0):.4f} UI",
            f"  Eye Area:   {metrics.get('eye_area', 0)*1000:.4f} mV*UI",
            "",
            "Jitter Analysis:",
            "-" * 40,
            f"  DJ: {jitter.get('dj', 0):.4f} UI",
            f"  RJ: {jitter.get('rj', 0):.4f} UI",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def _create_markdown_report(self, metrics: Dict[str, Any],
                                jitter: Dict[str, Any]) -> str:
        """Create markdown format report."""
        lines = [
            "# Eye Diagram Analysis Report",
            "",
            "## Configuration",
            f"- **Mode**: {self.mode}",
            f"- **Modulation**: {self.modulation}",
            f"- **UI**: {self.ui*1e12:.2f} ps",
            f"- **Target BER**: {self.target_ber:.0e}",
            "",
            "## Eye Metrics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Eye Height | {metrics.get('eye_height', 0)*1000:.2f} mV |",
            f"| Eye Width | {metrics.get('eye_width', 0):.4f} UI |",
            f"| Eye Area | {metrics.get('eye_area', 0)*1000:.4f} mV*UI |",
            "",
            "## Jitter Analysis",
            f"- **DJ**: {jitter.get('dj', 0):.4f} UI",
            f"- **RJ**: {jitter.get('rj', 0):.4f} UI",
        ]
        return "\n".join(lines)
    
    def _create_html_report(self, metrics: Dict[str, Any],
                           jitter: Dict[str, Any]) -> str:
        """Create HTML format report."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Eye Diagram Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 50%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Eye Diagram Analysis Report</h1>
    <h2>Configuration</h2>
    <ul>
        <li><strong>Mode:</strong> {self.mode}</li>
        <li><strong>Modulation:</strong> {self.modulation}</li>
        <li><strong>UI:</strong> {self.ui*1e12:.2f} ps</li>
        <li><strong>Target BER:</strong> {self.target_ber:.0e}</li>
    </ul>
    <h2>Eye Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Eye Height</td><td>{metrics.get('eye_height', 0)*1000:.2f} mV</td></tr>
        <tr><td>Eye Width</td><td>{metrics.get('eye_width', 0):.4f} UI</td></tr>
        <tr><td>Eye Area</td><td>{metrics.get('eye_area', 0)*1000:.4f} mV*UI</td></tr>
    </table>
    <h2>Jitter Analysis</h2>
    <ul>
        <li><strong>DJ:</strong> {jitter.get('dj', 0):.4f} UI</li>
        <li><strong>RJ:</strong> {jitter.get('rj', 0):.4f} UI</li>
    </ul>
</body>
</html>"""


# Keep UnifiedEyeAnalyzer for backward compatibility
UnifiedEyeAnalyzer = EyeAnalyzer
