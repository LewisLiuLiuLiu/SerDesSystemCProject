"""
Unified Eye Diagram Analyzer

This module provides the unified entry point for eye diagram analysis,
supporting both Sampler-Centric and Golden CDR schemes.
"""

from typing import Dict, Any, Optional
import numpy as np

from .schemes import SamplerCentricScheme, GoldenCdrScheme, StatisticalScheme
from .visualization import plot_eye_diagram, save_eye_diagram


class UnifiedEyeAnalyzer:
    """
    Unified entry point for eye diagram analysis.
    
    This class provides a simple interface to three analysis schemes:
    - 'sampler_centric': Streaming scheme based on CDR sampling timestamps
    - 'golden_cdr': Standard scheme based on ideal clock
    - 'statistical': Statistical eye analysis based on pulse response
    
    Example:
        >>> # Sampler-Centric scheme
        >>> analyzer = UnifiedEyeAnalyzer(ui=2.5e-11, scheme='sampler_centric')
        >>> metrics = analyzer.analyze(time, voltage, sampler_timestamps)
        >>> analyzer.plot('eye_sampler.png')
        
        >>> # Golden CDR scheme
        >>> analyzer = UnifiedEyeAnalyzer(ui=2.5e-11, scheme='golden_cdr')
        >>> metrics = analyzer.analyze(time, voltage)
        >>> analyzer.plot('eye_golden.png')
        
        >>> # Statistical scheme
        >>> analyzer = UnifiedEyeAnalyzer(ui=2.5e-11, scheme='statistical')
        >>> metrics = analyzer.analyze(pulse_response)
    """
    
    def __init__(self, ui: float, scheme: str = 'sampler_centric',
                 ui_bins: int = 128, amp_bins: int = 256, **kwargs):
        """
        Initialize the unified analyzer.
        
        Args:
            ui: Unit interval in seconds
            scheme: Analysis scheme ('sampler_centric' or 'golden_cdr')
            ui_bins: Number of bins for time/phase axis (default: 128)
            amp_bins: Number of bins for amplitude axis
                     (default: 256 for sampler_centric, 128 for golden_cdr)
            **kwargs: Additional scheme-specific arguments
                - interp_method: Interpolation method for sampler_centric
                - jitter_method: Jitter extraction method for golden_cdr
                
        Raises:
            ValueError: If scheme is invalid
        """
        valid_schemes = ['sampler_centric', 'golden_cdr', 'statistical']
        if scheme not in valid_schemes:
            raise ValueError(
                f"Invalid scheme '{scheme}'. "
                f"Valid schemes: {valid_schemes}"
            )
        
        self.ui = ui
        self.scheme_name = scheme
        self._metrics: Optional[Dict[str, Any]] = None
        
        # Create appropriate scheme object
        if scheme == 'sampler_centric':
            interp_method = kwargs.get('interp_method', 'cubic')
            self._scheme = SamplerCentricScheme(
                ui=ui, 
                ui_bins=ui_bins, 
                amp_bins=amp_bins,
                interp_method=interp_method
            )
        elif scheme == 'golden_cdr':
            jitter_method = kwargs.get('jitter_method', 'dual-dirac')
            # Golden CDR typically uses equal bins for both axes
            if amp_bins == 256:  # If using default, adjust for golden_cdr
                amp_bins = 128
            self._scheme = GoldenCdrScheme(
                ui=ui,
                ui_bins=ui_bins,
                amp_bins=amp_bins,
                jitter_method=jitter_method
            )
        else:  # statistical
            modulation = kwargs.get('modulation', 'nrz')
            samples_per_symbol = kwargs.get('samples_per_symbol', 8)
            noise_sigma = kwargs.get('noise_sigma', 0.0)
            jitter_dj = kwargs.get('jitter_dj', 0.0)
            jitter_rj = kwargs.get('jitter_rj', 0.0)
            target_ber = kwargs.get('target_ber', 1e-12)
            self._scheme = StatisticalScheme(
                ui=ui,
                modulation=modulation,
                samples_per_symbol=samples_per_symbol,
                ui_bins=ui_bins,
                amp_bins=amp_bins,
                noise_sigma=noise_sigma,
                jitter_dj=jitter_dj,
                jitter_rj=jitter_rj,
                target_ber=target_ber
            )
    
    def analyze(self, data: np.ndarray,
                voltage_array: Optional[np.ndarray] = None,
                sampler_timestamps: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform eye diagram analysis.
        
        Args:
            data: For 'sampler_centric' and 'golden_cdr': time array in seconds.
                  For 'statistical': pulse response array.
            voltage_array: Voltage array in volts (required for sampler_centric/golden_cdr)
            sampler_timestamps: CDR sampling timestamps (required for sampler_centric)
            **kwargs: Additional arguments passed to the scheme
                - For 'statistical': noise_sigma, jitter_dj, jitter_rj, target_ber
            
        Returns:
            Dictionary containing analysis metrics
            
        Raises:
            ValueError: If required arguments are missing
        """
        if self.scheme_name == 'sampler_centric':
            if voltage_array is None:
                raise ValueError("voltage_array is required for sampler_centric scheme")
            self._metrics = self._scheme.analyze(
                data, voltage_array, 
                sampler_timestamps=sampler_timestamps,
                **kwargs
            )
        elif self.scheme_name == 'golden_cdr':
            if voltage_array is None:
                raise ValueError("voltage_array is required for golden_cdr scheme")
            self._metrics = self._scheme.analyze(
                data, voltage_array,
                **kwargs
            )
        else:  # statistical
            # Statistical scheme takes pulse_response directly
            self._metrics = self._scheme.analyze(
                data,  # pulse_response
                **kwargs
            )
        
        return self._metrics
    
    def get_eye_matrix(self) -> Optional[np.ndarray]:
        """
        Get the eye diagram 2D matrix.
        
        Returns:
            2D numpy array or None if not analyzed
        """
        return self._scheme.get_eye_matrix()
    
    def plot(self, output_path: str, **kwargs) -> None:
        """
        Generate and save eye diagram visualization.
        
        Args:
            output_path: Output file path
            **kwargs: Additional arguments passed to plot_eye_diagram
                - dpi: Output DPI (default: 300)
                - figsize: Figure size (default: (12, 8))
                - smooth_sigma: Gaussian smoothing sigma (default: 1.0)
                
        Raises:
            ValueError: If analyze() has not been called
        """
        if self._metrics is None:
            raise ValueError("No analysis results. Call analyze() first.")
        
        save_eye_diagram(self._scheme, self._metrics, output_path, **kwargs)
    
    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """Get the last analysis metrics."""
        return self._metrics
