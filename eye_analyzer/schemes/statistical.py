"""
Statistical Eye Diagram Analysis Scheme.

This module implements the StatisticalScheme class for eye diagram analysis
based on channel pulse response. It integrates multiple statistical components:
- PulseResponseProcessor: Process channel pulse response
- ISICalculator: Calculate ISI probability distributions
- NoiseInjector: Inject Gaussian noise into PDFs
- JitterInjector: Inject Dual-Dirac jitter
- BERCalculator: Calculate BER contours and eye metrics

Supports both NRZ and PAM4 modulation formats.

Reference:
- OIF-CEI-04.0 specification for statistical eye methodology
- Statistical eye analysis based on pulse response convolution
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np

from .base import BaseScheme
from ..modulation import ModulationFormat, create_modulation, NRZ, PAM4
from ..statistical.pulse_response import PulseResponseProcessor
from ..statistical.isi_calculator import ISICalculator, ModulationFormat as ISIModulationFormat
from ..statistical.noise_injector import NoiseInjector
from ..statistical.jitter_injector import JitterInjector
from ..statistical.ber_calculator import BERCalculator


class StatisticalScheme(BaseScheme):
    """
    Statistical Eye Diagram Analysis Scheme.
    
    This scheme constructs eye diagrams using statistical analysis of channel
    pulse response. It calculates ISI (Inter-Symbol Interference) distributions,
    injects noise and jitter, and computes BER contours.
    
    Key characteristics:
    - Input: Channel pulse response
    - Method: Statistical convolution of ISI PDFs
    - Noise: Gaussian noise injection via PDF convolution
    - Jitter: Dual-Dirac jitter model (DJ + RJ)
    - Output: Eye diagram matrix, BER contours, eye metrics
    
    Attributes:
        ui: Unit interval in seconds
        modulation: Modulation format (NRZ, PAM4, etc.)
        ui_bins: Number of bins for time axis
        amp_bins: Number of bins for amplitude axis
        samples_per_symbol: Samples per UI for ISI calculation
        sample_size: Number of symbols to sample from pulse tail
        noise_injector: Optional NoiseInjector instance
        jitter_injector: Optional JitterInjector instance
        
    Example:
        >>> scheme = StatisticalScheme(ui=2.5e-11, modulation='pam4')
        >>> pulse_response = load_pulse_response('channel.s4p')
        >>> metrics = scheme.analyze(
        ...     pulse_response=pulse_response,
        ...     dt=1e-12,
        ...     noise_sigma=0.01,
        ...     dj=0.05,
        ...     rj=0.01,
        ...     target_ber=1e-12
        ... )
        >>> print(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
    """
    
    def __init__(self,
                 ui: float,
                 modulation: Union[str, ModulationFormat] = 'nrz',
                 ui_bins: int = 128,
                 amp_bins: int = 256,
                 samples_per_symbol: int = 16,
                 sample_size: int = 32,
                 noise_injector: Optional[NoiseInjector] = None,
                 jitter_injector: Optional[JitterInjector] = None,
                 vh_size: int = 2048):
        """
        Initialize the Statistical scheme.
        
        Args:
            ui: Unit interval in seconds (e.g., 2.5e-11 for 40 Gbps)
            modulation: Modulation format ('nrz', 'pam4') or ModulationFormat object
            ui_bins: Number of bins for time axis (default: 128)
            amp_bins: Number of bins for amplitude axis (default: 256)
            samples_per_symbol: Samples per UI for ISI calculation (default: 16)
            sample_size: Number of symbols to sample from pulse tail (default: 32)
            noise_injector: Optional NoiseInjector instance
            jitter_injector: Optional JitterInjector instance
            vh_size: Number of voltage histogram bins for ISI (default: 2048)
        """
        super().__init__(ui, modulation, ui_bins, amp_bins)
        
        self.samples_per_symbol = samples_per_symbol
        self.sample_size = sample_size
        self.vh_size = vh_size
        
        # Store or create injectors
        self.noise_injector = noise_injector
        self.jitter_injector = jitter_injector
        
        # Create component instances
        self._pulse_processor = PulseResponseProcessor()
        self._isi_calculator: Optional[ISICalculator] = None
        self._ber_calculator: Optional[BERCalculator] = None
        
        # Internal state
        self._time_slices: Optional[np.ndarray] = None
        self._voltage_bins: Optional[np.ndarray] = None
        self._eye_pdf: Optional[np.ndarray] = None
        self._ber_contour: Optional[np.ndarray] = None
    
    def _get_isi_modulation_format(self) -> ISIModulationFormat:
        """Convert modulation format to ISICalculator format."""
        if self.modulation.name == 'nrz':
            return ISIModulationFormat.NRZ
        elif self.modulation.name == 'pam4':
            return ISIModulationFormat.PAM4
        else:
            # Default to NRZ for unknown formats
            return ISIModulationFormat.NRZ
    
    def _initialize_components(self) -> None:
        """Initialize ISI calculator and BER calculator."""
        isi_format = self._get_isi_modulation_format()
        
        self._isi_calculator = ISICalculator(
            modulation_format=isi_format,
            samples_per_symbol=self.samples_per_symbol,
            vh_size=self.vh_size,
            sample_size=self.sample_size,
            upsampling=1
        )
        
        self._ber_calculator = BERCalculator(
            modulation=self.modulation,
            signal_amplitude=1.0
        )
    
    def analyze(self,
                time_array: Optional[np.ndarray] = None,
                voltage_array: Optional[np.ndarray] = None,
                pulse_response: Optional[np.ndarray] = None,
                dt: float = 1e-12,
                noise_sigma: float = 0.0,
                dj: float = 0.0,
                rj: float = 0.0,
                target_ber: float = 1e-12,
                **kwargs) -> Dict[str, Any]:
        """
        Perform statistical eye diagram analysis.
        
        Args:
            time_array: Time array (optional, not used in statistical scheme)
            voltage_array: Voltage array (optional, not used in statistical scheme)
            pulse_response: Channel pulse response array (required)
            dt: Time step of pulse response in seconds (default: 1e-12)
            noise_sigma: Gaussian noise standard deviation in volts (default: 0.0)
            dj: Deterministic jitter in UI (default: 0.0)
            rj: Random jitter standard deviation in UI (default: 0.0)
            target_ber: Target BER for eye opening calculation (default: 1e-12)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary containing:
            - eye_height: Eye height in volts (NRZ) or mean height (PAM4)
            - eye_width: Eye width in UI (NRZ) or mean width (PAM4)
            - eye_area: Eye area in V*UI
            - eye_heights_per_eye: List of eye heights per eye (PAM4)
            - eye_widths_per_eye: List of eye widths per eye (PAM4)
            - scheme: 'statistical'
            - modulation: Modulation format name
            - target_ber: Target BER used for calculations
            - noise_sigma: Noise sigma used
            - dj: DJ used
            - rj: RJ used
            
        Raises:
            ValueError: If pulse_response is not provided or invalid
        """
        # Validate pulse response
        if pulse_response is None:
            raise ValueError(
                "pulse_response is required for StatisticalScheme. "
                "Please provide channel pulse response."
            )
        
        pulse_response = np.asarray(pulse_response)
        
        if len(pulse_response) == 0:
            raise ValueError("pulse_response cannot be empty")
        
        if np.all(pulse_response == 0):
            raise ValueError("pulse_response cannot be all zeros")
        
        # Initialize components
        self._initialize_components()
        
        # Step 1: Process pulse response
        processed_pulse = self._pulse_processor.process(
            pulse=pulse_response,
            dt=dt,
            upsampling=1,
            diff_signal=True
        )
        
        # Step 2: Calculate ISI PDFs
        isi_result = self._isi_calculator.calculate(
            pulse_response=processed_pulse,
            method='convolution'
        )
        
        pdf_list = isi_result['pdf_list']
        self._voltage_bins = isi_result['voltage_bins']
        self._time_slices = isi_result['time_slices']
        
        # Step 3: Build eye PDF matrix
        self._eye_pdf = self._build_eye_pdf(pdf_list)
        
        # Step 4: Inject noise if specified
        if noise_sigma > 0 or self.noise_injector is not None:
            self._eye_pdf = self._apply_noise(self._eye_pdf, noise_sigma)
        
        # Step 5: Inject jitter if specified
        if dj > 0 or rj > 0 or self.jitter_injector is not None:
            self._eye_pdf = self._apply_jitter(self._eye_pdf, dj, rj)
        
        # Set eye_matrix for base class compatibility
        # ISI PDF is (time_slices, voltage_bins), eye_matrix expects (ui_bins, amp_bins)
        # Use interpolation to match target dimensions
        self.eye_matrix = self._resample_to_target_bins(self._eye_pdf)
        
        # Set edges to match eye_matrix dimensions
        self._xedges = np.linspace(-0.5, 0.5, self.ui_bins + 1)
        self._yedges = np.linspace(
            self._voltage_bins.min() if self._voltage_bins is not None else -1,
            self._voltage_bins.max() if self._voltage_bins is not None else 1,
            self.amp_bins + 1
        )
        
        # Update voltage range for base class methods
        if self._voltage_bins is not None and len(self._voltage_bins) > 0:
            self._v_min = self._voltage_bins.min()
            self._v_max = self._voltage_bins.max()
        
        # Step 6: Calculate BER contour
        self._ber_contour = self._ber_calculator.calculate_ber_contour(
            eye_pdf=self._eye_pdf,
            voltage_bins=self._voltage_bins
        )
        
        # Step 7: Compute eye metrics
        if self.modulation.name == 'pam4':
            return self._compute_metrics_pam4(target_ber, noise_sigma, dj, rj)
        else:
            return self._compute_metrics_nrz(target_ber, noise_sigma, dj, rj)
    
    def _build_eye_pdf(self, pdf_list: List[np.ndarray]) -> np.ndarray:
        """
        Build eye PDF matrix from ISI PDF list.
        
        Args:
            pdf_list: List of PDF arrays per time slice
            
        Returns:
            2D eye PDF matrix with shape (time_slices, voltage_bins)
        """
        n_time = len(pdf_list)
        n_voltage = len(pdf_list[0]) if pdf_list else 0
        
        eye_pdf = np.zeros((n_time, n_voltage))
        
        for i, pdf in enumerate(pdf_list):
            eye_pdf[i, :] = pdf
        
        return eye_pdf
    
    def _resample_to_target_bins(self, eye_pdf: np.ndarray) -> np.ndarray:
        """
        Resample eye PDF to target ui_bins x amp_bins dimensions.
        
        Args:
            eye_pdf: Input eye PDF with shape (time_slices, voltage_bins)
            
        Returns:
            Resampled eye matrix with shape (ui_bins, amp_bins)
        """
        from scipy.ndimage import zoom
        
        n_time, n_voltage = eye_pdf.shape
        
        # Calculate zoom factors
        zoom_time = self.ui_bins / n_time if n_time > 0 else 1
        zoom_voltage = self.amp_bins / n_voltage if n_voltage > 0 else 1
        
        # Use scipy.ndimage.zoom for resampling
        if zoom_time != 1 or zoom_voltage != 1:
            resampled = zoom(eye_pdf, (zoom_time, zoom_voltage), order=1)
            # Ensure exact dimensions (zoom may produce slightly different size)
            if resampled.shape != (self.ui_bins, self.amp_bins):
                from scipy.ndimage import resize
                # Use simple interpolation to exact size
                x_old = np.linspace(0, 1, n_time)
                y_old = np.linspace(0, 1, n_voltage)
                x_new = np.linspace(0, 1, self.ui_bins)
                y_new = np.linspace(0, 1, self.amp_bins)
                
                from scipy.interpolate import RegularGridInterpolator
                interpolator = RegularGridInterpolator(
                    (x_old, y_old), eye_pdf, 
                    bounds_error=False, fill_value=0
                )
                xx, yy = np.meshgrid(x_new, y_new, indexing='ij')
                resampled = interpolator((xx, yy))
        else:
            resampled = eye_pdf
        
        return resampled
    
    def _apply_noise(self, eye_pdf: np.ndarray, noise_sigma: float) -> np.ndarray:
        """
        Apply Gaussian noise to eye PDF.
        
        Args:
            eye_pdf: Input eye PDF matrix
            noise_sigma: Noise standard deviation in volts
            
        Returns:
            Noise-injected eye PDF
        """
        # Use stored injector or create new one
        if self.noise_injector is not None and noise_sigma == 0:
            injector = self.noise_injector
        else:
            sigma = noise_sigma if noise_sigma > 0 else 0.001
            injector = NoiseInjector(sigma=sigma)
        
        result = np.zeros_like(eye_pdf)
        
        # Apply noise to each time slice
        for t in range(eye_pdf.shape[0]):
            result[t, :] = injector.inject_noise(
                isi_pdf=eye_pdf[t, :],
                voltage_bins=self._voltage_bins
            )
        
        return result
    
    def _apply_jitter(self, eye_pdf: np.ndarray, dj: float, rj: float) -> np.ndarray:
        """
        Apply Dual-Dirac jitter to eye PDF.
        
        Args:
            eye_pdf: Input eye PDF matrix
            dj: Deterministic jitter in UI
            rj: Random jitter standard deviation in UI
            
        Returns:
            Jitter-injected eye PDF
        """
        # Use stored injector or create new one
        if self.jitter_injector is not None and dj == 0 and rj == 0:
            injector = self.jitter_injector
        else:
            use_dj = dj if dj > 0 else 0.0
            use_rj = rj if rj > 0 else 0.0
            injector = JitterInjector(dj=use_dj, rj=use_rj)
        
        # Create time bins in UI units
        time_bins = np.linspace(-0.5, 0.5, eye_pdf.shape[0])
        
        return injector.inject_jitter(eye_pdf=eye_pdf, time_bins=time_bins)
    
    def _compute_metrics_nrz(self, target_ber: float, 
                             noise_sigma: float, dj: float, rj: float) -> Dict[str, Any]:
        """
        Compute eye metrics for NRZ modulation.
        
        Args:
            target_ber: Target BER level
            noise_sigma: Noise sigma used
            dj: DJ used
            rj: RJ used
            
        Returns:
            Dictionary of metrics
        """
        # Find eye openings at target BER
        openings = self._ber_calculator.find_eye_openings(
            ber_contour=self._ber_contour,
            voltage_bins=self._voltage_bins,
            time_slices=self._time_slices,
            target_ber=target_ber
        )
        
        eye_height = openings.get('mean_height_v', 0.0)
        eye_width = openings.get('mean_width_ui', 0.0)
        eye_area = eye_height * eye_width
        
        # Also compute using base class methods
        eye_height_base = self._compute_eye_height()
        eye_width_base = self._compute_eye_width()
        
        # Use the more reliable value
        if eye_height <= 0 and eye_height_base > 0:
            eye_height = eye_height_base
        if eye_width <= 0 and eye_width_base > 0:
            eye_width = eye_width_base
        
        return {
            'eye_height': float(eye_height),
            'eye_width': float(eye_width),
            'eye_area': float(eye_area),
            'eye_heights': openings.get('eye_heights_v', []),
            'eye_widths': openings.get('eye_widths_ui', []),
            'modulation': 'nrz',
            'scheme': 'statistical',
            'target_ber': target_ber,
            'noise_sigma': noise_sigma,
            'dj': dj,
            'rj': rj,
            'samples_per_symbol': self.samples_per_symbol,
            'sample_size': self.sample_size
        }
    
    def _compute_metrics_pam4(self, target_ber: float,
                              noise_sigma: float, dj: float, rj: float) -> Dict[str, Any]:
        """
        Compute eye metrics for PAM4 modulation.
        
        Args:
            target_ber: Target BER level
            noise_sigma: Noise sigma used
            dj: DJ used
            rj: RJ used
            
        Returns:
            Dictionary of metrics including per-eye measurements
        """
        # Find eye openings at target BER
        openings = self._ber_calculator.find_eye_openings(
            ber_contour=self._ber_contour,
            voltage_bins=self._voltage_bins,
            time_slices=self._time_slices,
            target_ber=target_ber
        )
        
        eye_heights = openings.get('eye_heights_v', [])
        eye_widths = openings.get('eye_widths_ui', [])
        
        # If BER contour method returns empty, use base class methods
        if not eye_heights:
            eye_heights = [self._compute_eye_height()]
        if not eye_widths:
            eye_widths = [self._compute_eye_width()]
        
        # Calculate statistics
        if eye_heights:
            eye_height_min = min(eye_heights)
            eye_height_avg = sum(eye_heights) / len(eye_heights)
        else:
            eye_height_min = 0.0
            eye_height_avg = 0.0
        
        if eye_widths:
            eye_width_min = min(eye_widths)
            eye_width_avg = sum(eye_widths) / len(eye_widths)
        else:
            eye_width_min = 0.0
            eye_width_avg = 0.0
        
        eye_area = eye_height_avg * eye_width_avg
        
        return {
            'eye_heights_per_eye': eye_heights,
            'eye_widths_per_eye': eye_widths,
            'eye_height': float(eye_height_avg),
            'eye_height_min': float(eye_height_min),
            'eye_height_avg': float(eye_height_avg),
            'eye_width': float(eye_width_avg),
            'eye_width_min': float(eye_width_min),
            'eye_width_avg': float(eye_width_avg),
            'eye_area': float(eye_area),
            'modulation': 'pam4',
            'num_eyes': 3,
            'scheme': 'statistical',
            'target_ber': target_ber,
            'noise_sigma': noise_sigma,
            'dj': dj,
            'rj': rj,
            'samples_per_symbol': self.samples_per_symbol,
            'sample_size': self.sample_size
        }
    
    def get_xedges(self) -> np.ndarray:
        """
        Get X-axis bin edges (time in UI units).
        
        Returns:
            Array of X-axis bin edges with length ui_bins + 1
        """
        # Always return edges matching ui_bins
        if self._xedges is not None and len(self._xedges) == self.ui_bins + 1:
            return self._xedges
        
        # Default: return uniform edges over 1 UI centered at 0
        return np.linspace(-0.5, 0.5, self.ui_bins + 1)
    
    def get_yedges(self) -> np.ndarray:
        """
        Get Y-axis bin edges (voltage).
        
        Returns:
            Array of Y-axis bin edges with length amp_bins + 1
        """
        # Always return edges matching amp_bins
        if self._yedges is not None and len(self._yedges) == self.amp_bins + 1:
            return self._yedges
        
        # Default: create from voltage range
        if self._voltage_bins is not None and len(self._voltage_bins) > 0:
            v_min = self._voltage_bins.min()
            v_max = self._voltage_bins.max()
            return np.linspace(v_min, v_max, self.amp_bins + 1)
        
        return np.linspace(-1, 1, self.amp_bins + 1)
    
    def get_ber_contour(self) -> Optional[np.ndarray]:
        """
        Get the BER contour matrix.
        
        Returns:
            2D BER contour array or None if not analyzed
        """
        return self._ber_contour
    
    def get_eye_pdf(self) -> Optional[np.ndarray]:
        """
        Get the eye PDF matrix (before BER calculation).
        
        Returns:
            2D eye PDF array or None if not analyzed
        """
        return self._eye_pdf
