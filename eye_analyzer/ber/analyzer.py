"""BERAnalyzer - Unified interface for BER analysis components.

This module provides the BERAnalyzer class as a unified high-level interface
for all BER (Bit Error Rate) analysis components:

- BERContour: BER contour generation from eye diagram PDFs
- BathtubCurve: Bathtub curve analysis (BER vs. time/voltage)
- QFactor: BER to Q-factor conversions
- JTolTemplate: Industry standard jitter tolerance templates
- JitterTolerance: Jitter tolerance testing

The BERAnalyzer simplifies common analysis workflows by providing a single
entry point for all BER-related analyses.

Example:
    >>> from eye_analyzer.ber import BERAnalyzer
    >>> import numpy as np
    >>> 
    >>> # Create analyzer with desired configuration
    >>> analyzer = BERAnalyzer(modulation='pam4', target_ber=1e-12)
    >>> 
    >>> # Analyze eye diagram
    >>> eye_pdf = np.random.rand(128, 256)  # Your eye data
    >>> voltage_bins = np.linspace(-2, 2, 256)
    >>> time_slices = np.linspace(0, 1, 128)
    >>> 
    >>> result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
    >>> print(f"Eye height: {result['eye_height']:.3f} V")
    >>> print(f"Eye width: {result['eye_width']:.3f} UI")
    >>> 
    >>> # Analyze bathtub curves
    >>> bathtub = analyzer.analyze_bathtub(
    ...     eye_pdf, voltage_bins, time_slices, direction='time'
    ... )
    >>> 
    >>> # BER to Q conversion
    >>> q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
    >>> print(f"Q-factor: {q:.2f}")
    >>> 
    >>> # Get industry standard template
    >>> template = analyzer.get_template('ieee_802_3ck')
"""

import numpy as np
from typing import Optional, Dict, Union, List, Any
import logging

from .contour import BERContour
from .bathtub import BathtubCurve
from .qfactor import QFactor
from .template import JTolTemplate
from .jtol import JitterTolerance
from ..modulation import ModulationFormat, NRZ, PAM4

# Setup logging
logger = logging.getLogger(__name__)


class BERAnalyzer:
    """Unified interface for BER (Bit Error Rate) analysis.
    
    The BERAnalyzer class provides a high-level interface that integrates
    all BER analysis components into a single, easy-to-use API.
    
    Supported analyses:
    - Eye diagram BER contour generation
    - Eye dimension extraction (width, height, area)
    - Bathtub curve analysis (time and voltage)
    - BER to Q-factor conversion
    - Industry standard JTol template access
    - Jitter tolerance testing
    
    Attributes:
        modulation: Modulation format ('nrz' or 'pam4')
        target_ber: Default target bit error rate
        signal_amplitude: Signal amplitude for scaling
        _contour: Internal BERContour instance
        _bathtub: Internal BathtubCurve instance
        _qfactor: Internal QFactor instance
        _jtol: Internal JitterTolerance instance
    
    Example:
        >>> analyzer = BERAnalyzer(modulation='nrz', target_ber=1e-12)
        >>> 
        >>> # Complete eye analysis
        >>> result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
        >>> 
        >>> # Bathtub analysis
        >>> bathtub = analyzer.analyze_bathtub(
        ...     eye_pdf, voltage_bins, time_slices, direction='time'
        ... )
        >>> 
        >>> # BER-Q conversion
        >>> q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
    """
    
    # Valid modulation formats
    VALID_MODULATIONS = ['nrz', 'pam4']
    
    # Valid template names
    VALID_TEMPLATES = ['ieee_802_3ck', 'oif_cei_112g', 'jedec_ddr5', 'pcie_gen6']
    
    def __init__(
        self,
        modulation: str = 'nrz',
        target_ber: float = 1e-12,
        signal_amplitude: float = 1.0
    ):
        """Initialize BERAnalyzer.
        
        Args:
            modulation: Modulation format. Options: 'nrz', 'pam4'.
                Default is 'nrz'.
            target_ber: Default target bit error rate for analyses.
                Default is 1e-12.
            signal_amplitude: Signal amplitude for level scaling.
                Default is 1.0.
        
        Raises:
            ValueError: If modulation is not a valid format.
        
        Example:
            >>> analyzer = BERAnalyzer(modulation='pam4', target_ber=1e-15)
            >>> analyzer = BERAnalyzer(modulation='nrz', signal_amplitude=0.8)
        """
        if modulation.lower() not in self.VALID_MODULATIONS:
            raise ValueError(
                f"Invalid modulation '{modulation}'. "
                f"Valid options: {self.VALID_MODULATIONS}"
            )
        
        self.modulation = modulation.lower()
        self.target_ber = target_ber
        self.signal_amplitude = signal_amplitude
        
        # Create modulation format instance
        if self.modulation == 'nrz':
            mod_format = NRZ()
        else:  # pam4
            mod_format = PAM4()
        
        # Initialize component instances
        self._contour = BERContour(
            modulation=mod_format,
            signal_amplitude=signal_amplitude
        )
        self._bathtub = BathtubCurve(
            modulation=mod_format,
            signal_amplitude=signal_amplitude
        )
        self._qfactor = QFactor()
        self._jtol = JitterTolerance(
            modulation=self.modulation,
            target_ber=target_ber
        )
        
        logger.debug(
            f"BERAnalyzer initialized: modulation={self.modulation}, "
            f"target_ber={self.target_ber}, signal_amplitude={self.signal_amplitude}"
        )
    
    def analyze_eye(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: Optional[float] = None
    ) -> Dict[str, Any]:
        """Perform complete eye diagram analysis.
        
        Calculates BER contour and extracts eye dimensions including width,
        height, and area at the specified target BER level.
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins).
                Each row should be normalized (sum to 1).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values (typically in UI, 0 to 1).
            target_ber: Target BER level for eye dimension extraction.
                If None, uses the target_ber from initialization.
                Default is None.
        
        Returns:
            Dictionary containing:
                - 'ber_contour': BER contour matrix (same shape as eye_pdf)
                - 'eye_height': Eye height at target BER in volts
                - 'eye_width': Eye width at target BER in UI
                - 'eye_area': Eye area (width * height) at target BER
                - 'eye_dimensions': Full eye dimensions dictionary from
                  get_eye_dimensions()
        
        Raises:
            ValueError: If eye_pdf is empty or dimensions don't match.
        
        Example:
            >>> analyzer = BERAnalyzer()
            >>> result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
            >>> print(f"Eye height: {result['eye_height']:.3f} V")
            >>> print(f"Eye width: {result['eye_width']:.3f} UI")
        """
        if target_ber is None:
            target_ber = self.target_ber
        
        # Validate inputs
        if eye_pdf.size == 0:
            raise ValueError("eye_pdf cannot be empty")
        
        if eye_pdf.shape[1] != len(voltage_bins):
            raise ValueError(
                f"eye_pdf second dimension ({eye_pdf.shape[1]}) must match "
                f"voltage_bins length ({len(voltage_bins)})"
            )
        
        if eye_pdf.shape[0] != len(time_slices):
            raise ValueError(
                f"eye_pdf first dimension ({eye_pdf.shape[0]}) must match "
                f"time_slices length ({len(time_slices)})"
            )
        
        logger.debug(
            f"Analyzing eye: shape={eye_pdf.shape}, target_ber={target_ber}"
        )
        
        # Calculate BER contour
        ber_contour = self._contour.calculate(
            eye_pdf=eye_pdf,
            voltage_bins=voltage_bins,
            target_bers=[target_ber]
        )
        
        # Get eye dimensions
        eye_dims = self._contour.get_eye_dimensions(
            ber_contour=ber_contour,
            voltage_bins=voltage_bins,
            time_slices=time_slices,
            target_ber=target_ber
        )
        
        result = {
            'ber_contour': ber_contour,
            'eye_height': eye_dims['eye_height_v'],
            'eye_width': eye_dims['eye_width_ui'],
            'eye_area': eye_dims['eye_area_ui_v'],
            'eye_dimensions': eye_dims,
        }
        
        logger.debug(
            f"Eye analysis complete: height={result['eye_height']:.3f}V, "
            f"width={result['eye_width']:.3f}UI"
        )
        
        return result
    
    def get_eye_dimensions(
        self,
        ber_contour: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: Optional[float] = None
    ) -> Dict[str, Any]:
        """Extract eye dimensions from BER contour.
        
        Analyzes a BER contour matrix to find eye openings at the specified
        target BER level. Returns detailed eye dimension information.
        
        Args:
            ber_contour: BER contour matrix with shape (time_slices, voltage_bins).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values.
            target_ber: Target BER level for eye opening detection.
                If None, uses the target_ber from initialization.
                Default is None.
        
        Returns:
            Dictionary containing:
                - 'eye_width_ui': Width of the eye opening in UI units
                - 'eye_width_samples': Width in number of time samples
                - 'eye_height_v': Height of the eye opening in volts
                - 'eye_height_samples': Height in number of voltage samples
                - 'eye_area_ui_v': Eye area (width * height)
                - 'num_eyes': Number of eye openings detected
                - 'all_widths': List of widths for all detected eyes
                - 'all_heights': List of heights for all detected eyes
        
        Example:
            >>> eye_result = analyzer.analyze_eye(eye_pdf, voltage_bins, time_slices)
            >>> dims = analyzer.get_eye_dimensions(
            ...     eye_result['ber_contour'], voltage_bins, time_slices
            ... )
            >>> print(f"Detected {dims['num_eyes']} eyes")
        """
        if target_ber is None:
            target_ber = self.target_ber
        
        return self._contour.get_eye_dimensions(
            ber_contour=ber_contour,
            voltage_bins=voltage_bins,
            time_slices=time_slices,
            target_ber=target_ber
        )
    
    def analyze_bathtub(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        direction: str = 'time',
        target_ber: Optional[float] = None
    ) -> Dict[str, Any]:
        """Perform bathtub curve analysis.
        
        Generates bathtub curves showing BER as a function of either:
        - Time (phase): Fixed voltage level, scan across time
        - Voltage: Fixed time (phase), scan across voltage
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values.
            direction: Direction of bathtub curve. Options:
                - 'time': BER vs. time at fixed voltage (default)
                - 'voltage': BER vs. voltage at fixed time
            target_ber: Target BER level for reference.
                If None, uses the target_ber from initialization.
                Default is None.
        
        Returns:
            For direction='time':
                - 'time': Array of time values
                - 'ber_left': BER values for left side of eye
                - 'ber_right': BER values for right side of eye
                - 'voltage_level': The voltage level used for the scan
            
            For direction='voltage':
                - 'voltage': Array of voltage values
                - 'ber_upper': BER values for upper side of eye
                - 'ber_lower': BER values for lower side of eye
                - 'time_value': The time value used for the scan
        
        Raises:
            ValueError: If direction is not 'time' or 'voltage'.
        
        Example:
            >>> # Time bathtub
            >>> tb = analyzer.analyze_bathtub(
            ...     eye_pdf, voltage_bins, time_slices, direction='time'
            ... )
            >>> 
            >>> # Voltage bathtub
            >>> vb = analyzer.analyze_bathtub(
            ...     eye_pdf, voltage_bins, time_slices, direction='voltage'
            ... )
        """
        if target_ber is None:
            target_ber = self.target_ber
        
        if direction.lower() not in ['time', 'voltage']:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Valid options: 'time', 'voltage'"
            )
        
        logger.debug(f"Analyzing bathtub curve: direction={direction}")
        
        if direction.lower() == 'time':
            result = self._bathtub.calculate_time_bathtub(
                eye_pdf=eye_pdf,
                voltage_bins=voltage_bins,
                time_slices=time_slices,
                target_ber=target_ber
            )
        else:  # voltage
            result = self._bathtub.calculate_voltage_bathtub(
                eye_pdf=eye_pdf,
                voltage_bins=voltage_bins,
                time_slices=time_slices,
                target_ber=target_ber
            )
        
        return result
    
    def convert_ber_q(
        self,
        value: float,
        direction: str = 'ber_to_q'
    ) -> float:
        """Convert between BER and Q-factor.
        
        Performs conversion between Bit Error Rate (BER) and Q-factor using
        the complementary error function relationship:
        
            Q = sqrt(2) * erfcinv(2 * BER)
            BER = 0.5 * erfc(Q / sqrt(2))
        
        Args:
            value: Value to convert. Either BER (if direction='ber_to_q')
                or Q-factor (if direction='q_to_ber').
            direction: Conversion direction. Options:
                - 'ber_to_q': Convert BER to Q-factor (default)
                - 'q_to_ber': Convert Q-factor to BER
        
        Returns:
            Converted value. Q-factor if direction='ber_to_q',
            BER if direction='q_to_ber'.
        
        Raises:
            ValueError: If direction is invalid or value is out of range.
        
        Example:
            >>> # BER to Q
            >>> q = analyzer.convert_ber_q(1e-12, direction='ber_to_q')
            >>> print(f"Q = {q:.2f}")  # Q ≈ 7.03
            >>> 
            >>> # Q to BER
            >>> ber = analyzer.convert_ber_q(7.03, direction='q_to_ber')
            >>> print(f"BER = {ber:.2e}")  # BER ≈ 1e-12
        """
        if direction.lower() not in ['ber_to_q', 'q_to_ber']:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Valid options: 'ber_to_q', 'q_to_ber'"
            )
        
        if direction.lower() == 'ber_to_q':
            return self._qfactor.ber_to_q(value)
        else:  # q_to_ber
            return self._qfactor.q_to_ber(value)
    
    def get_template(self, template_name: str) -> JTolTemplate:
        """Get an industry standard JTol template.
        
        Returns a JTolTemplate instance for the specified standard.
        
        Args:
            template_name: Name of the template. Options:
                - 'ieee_802_3ck': IEEE 802.3ck (200G/400G Ethernet)
                - 'oif_cei_112g': OIF-CEI-112G
                - 'jedec_ddr5': JEDEC DDR5
                - 'pcie_gen6': PCIe Gen6
        
        Returns:
            JTolTemplate instance for the specified standard.
        
        Raises:
            ValueError: If template_name is not a valid template.
        
        Example:
            >>> template = analyzer.get_template('ieee_802_3ck')
            >>> sj_limit = template.get_sj_limit(1e6)  # SJ limit at 1 MHz
        """
        if template_name not in self.VALID_TEMPLATES:
            raise ValueError(
                f"Invalid template '{template_name}'. "
                f"Valid templates: {self.VALID_TEMPLATES}"
            )
        
        return JTolTemplate(template_name)
    
    def analyze_jtol(
        self,
        pulse_response: np.ndarray,
        sj_frequencies: np.ndarray,
        template: str = 'ieee_802_3ck',
        noise_sigma: float = 0.0,
        rj: float = 0.0,
        eye_analyzer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform jitter tolerance (JTol) analysis.
        
        Measures the maximum tolerable sinusoidal jitter (SJ) at various
        modulation frequencies and compares against industry standard templates.
        
        Args:
            pulse_response: Channel pulse response array. Required for
                statistical eye analysis.
            sj_frequencies: Array of SJ modulation frequencies in Hz.
            template: Template name for comparison. Options:
                - 'ieee_802_3ck': IEEE 802.3ck (200G/400G Ethernet) (default)
                - 'oif_cei_112g': OIF-CEI-112G
                - 'jedec_ddr5': JEDEC DDR5
                - 'pcie_gen6': PCIe Gen6
            noise_sigma: Gaussian noise sigma to apply during measurement.
                Default is 0.0.
            rj: Random jitter to apply during measurement (in UI).
                Default is 0.0.
            eye_analyzer: Optional StatisticalScheme instance. If None,
                a new instance will be created.
        
        Returns:
            Dictionary containing:
                - 'frequencies': Array of test frequencies in Hz
                - 'sj_limits': Array of measured SJ limits in UI
                - 'template_limits': Array of template SJ limits in UI
                - 'margins': Array of margins (measured - template) in UI
                - 'pass_fail': List of Pass/Fail per frequency point
                - 'overall_pass': True if all points pass, False otherwise
                - 'modulation': Modulation format used
                - 'target_ber': Target BER used
        
        Raises:
            ValueError: If pulse_response is None or template is invalid.
        
        Example:
            >>> import numpy as np
            >>> analyzer = BERAnalyzer(modulation='nrz')
            >>> 
            >>> pulse_response = np.exp(-np.linspace(0, 1e-9, 100) / 100e-12)
            >>> sj_freqs = np.logspace(5, 9, 20)  # 100 kHz to 1 GHz
            >>> 
            >>> results = analyzer.analyze_jtol(
            ...     pulse_response=pulse_response,
            ...     sj_frequencies=sj_freqs,
            ...     template='ieee_802_3ck'
            ... )
            >>> 
            >>> print(f"Overall Pass: {results['overall_pass']}")
            >>> print(f"Min margin: {min(results['margins']):.3f} UI")
        """
        if pulse_response is None:
            raise ValueError("pulse_response is required for JTOL analysis")
        
        if template not in self.VALID_TEMPLATES:
            raise ValueError(
                f"Invalid template '{template}'. "
                f"Valid templates: {self.VALID_TEMPLATES}"
            )
        
        logger.info(
            f"Starting JTOL analysis: {len(sj_frequencies)} frequency points, "
            f"template={template}"
        )
        
        # Create eye analyzer if not provided
        if eye_analyzer is None:
            from ..schemes.statistical import StatisticalScheme
            
            # Calculate UI from pulse response
            # Assume pulse_response spans one UI
            ui = 1.0  # Default UI
            
            eye_analyzer = StatisticalScheme(
                ui=ui,
                modulation=self.modulation
            )
        
        # Perform JTOL measurement
        results = self._jtol.measure_jtol(
            eye_analyzer=eye_analyzer,
            sj_frequencies=sj_frequencies,
            template=template,
            pulse_response=pulse_response,
            noise_sigma=noise_sigma,
            rj=rj
        )
        
        logger.info(
            f"JTOL analysis complete: overall_pass={results['overall_pass']}"
        )
        
        return results
    
    def get_eye_opening_at_ber(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: Optional[float] = None,
        bathtub_type: str = 'time'
    ) -> Dict[str, float]:
        """Get eye opening dimensions at a specific BER level.
        
        Convenience method to extract eye opening dimensions directly
        from bathtub curves at a target BER level.
        
        Args:
            eye_pdf: Eye diagram PDF matrix.
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values.
            target_ber: Target BER level. If None, uses the target_ber
                from initialization. Default is None.
            bathtub_type: 'time' for eye width, 'voltage' for eye height.
                Default is 'time'.
        
        Returns:
            Dictionary containing:
                - 'opening': Eye opening (width in UI or height in V)
                - 'center': Center of the eye opening
                - 'edges': Tuple of (lower_edge, upper_edge)
        
        Example:
            >>> result = analyzer.get_eye_opening_at_ber(
            ...     eye_pdf, voltage_bins, time_slices, target_ber=1e-12
            ... )
            >>> print(f"Eye width at 1e-12: {result['opening']:.3f} UI")
        """
        if target_ber is None:
            target_ber = self.target_ber
        
        return self._bathtub.get_eye_opening_at_ber(
            eye_pdf=eye_pdf,
            voltage_bins=voltage_bins,
            time_slices=time_slices,
            target_ber=target_ber,
            bathtub_type=bathtub_type
        )
    
    def __repr__(self) -> str:
        """String representation of BERAnalyzer."""
        return (
            f"BERAnalyzer("
            f"modulation='{self.modulation}', "
            f"target_ber={self.target_ber}, "
            f"signal_amplitude={self.signal_amplitude})"
        )
    
    def __str__(self) -> str:
        """Readable string representation of BERAnalyzer."""
        return (
            f"BERAnalyzer:\n"
            f"  Modulation: {self.modulation.upper()}\n"
            f"  Target BER: {self.target_ber:.0e}\n"
            f"  Signal Amplitude: {self.signal_amplitude} V"
        )
