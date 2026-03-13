"""
BER Contour class for eye diagram analysis.

This module provides the BERContour class as a high-level interface for:
- BER (Bit Error Rate) contour calculation from eye diagram PDFs
- Eye dimension extraction (width, height) at target BER levels

The BERContour class encapsulates the BERCalculator from the statistical
module and provides a simplified, unified API for eye analysis.

Example:
    >>> from eye_analyzer.ber import BERContour
    >>> from eye_analyzer.modulation import PAM4
    >>> import numpy as np
    >>> 
    >>> # Create BER contour analyzer
    >>> analyzer = BERContour(modulation=PAM4())
    >>> 
    >>> # Generate or load eye PDF (time_slices x voltage_bins)
    >>> eye_pdf = np.random.rand(128, 256)
    >>> eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
    >>> voltage_bins = np.linspace(-2, 2, 256)
    >>> 
    >>> # Calculate BER contour
    >>> ber_contour = analyzer.calculate(
    ...     eye_pdf, 
    ...     voltage_bins, 
    ...     target_bers=[1e-12, 1e-9]
    ... )
    >>> 
    >>> # Extract eye dimensions
    >>> dims = analyzer.get_eye_dimensions(
    ...     ber_contour[0],  # First BER level
    ...     voltage_bins,
    ...     np.linspace(0, 1, 128),  # Time slices
    ...     target_ber=1e-12
    ... )
"""

import numpy as np
from typing import Optional, List, Dict, Union, Tuple

from eye_analyzer.statistical.ber_calculator import BERCalculator
from eye_analyzer.modulation import ModulationFormat, NRZ, PAM4


class BERContour:
    """
    High-level interface for BER contour calculation and eye dimension analysis.
    
    This class encapsulates the BERCalculator and provides a simplified API
    for:
    1. Calculating BER contours from eye diagram PDFs
    2. Extracting eye dimensions (width, height) at target BER levels
    
    Supports both NRZ and PAM4 modulation formats.
    
    Attributes:
        modulation: Modulation format instance (NRZ, PAM4, etc.)
        _calculator: Internal BERCalculator instance
    
    Example:
        >>> contour = BERContour(modulation=PAM4())
        >>> ber_matrix = contour.calculate(eye_pdf, voltage_bins)
        >>> dims = contour.get_eye_dimensions(ber_matrix, voltage_bins, time_slices)
    """
    
    def __init__(
        self,
        modulation: Optional[ModulationFormat] = None,
        signal_amplitude: float = 1.0
    ):
        """
        Initialize BERContour analyzer.
        
        Args:
            modulation: Modulation format instance. If None, defaults to NRZ.
            signal_amplitude: Amplitude of the signal for scaling levels.
        """
        if modulation is None:
            modulation = NRZ()
        
        self.modulation = modulation
        self._calculator = BERCalculator(
            modulation=modulation,
            signal_amplitude=signal_amplitude
        )
    
    def calculate(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        target_bers: Optional[List[float]] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Calculate BER contour matrix from eye diagram PDF.
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins).
                Each row should be normalized (sum to 1).
            voltage_bins: Array of voltage bin center values.
            target_bers: List of target BER levels for contour extraction.
                If None or single value, returns single contour matrix.
                If multiple values, returns list of contours.
                Default: [1e-12]
        
        Returns:
            If target_bers has one element or None: BER contour matrix 
                with same shape as eye_pdf.
            If target_bers has multiple elements: List of BER contour matrices,
                one for each target BER level.
        
        Raises:
            ValueError: If eye_pdf dimensions don't match voltage_bins,
                or if eye_pdf is empty.
        
        Example:
            >>> contour = BERContour()
            >>> ber_contour = contour.calculate(eye_pdf, voltage_bins)
            >>> # ber_contour[t, v] = BER at time t, voltage v
        """
        if target_bers is None:
            target_bers = [1e-12]
        
        # Validate inputs
        if eye_pdf.size == 0:
            raise ValueError("eye_pdf cannot be empty")
        
        if eye_pdf.shape[1] != len(voltage_bins):
            raise ValueError(
                f"eye_pdf second dimension ({eye_pdf.shape[1]}) must match "
                f"voltage_bins length ({len(voltage_bins)})"
            )
        
        # Calculate the BER contour using encapsulated BERCalculator
        ber_contour = self._calculator.calculate_ber_contour(
            eye_pdf=eye_pdf,
            voltage_bins=voltage_bins,
            use_oif_method=True
        )
        
        # If single target BER, return the contour directly
        if len(target_bers) == 1:
            return ber_contour
        
        # For multiple target BERs, we return the same contour
        # (the contour contains all BER values, extraction happens in get_eye_dimensions)
        return [ber_contour for _ in target_bers]
    
    def get_eye_dimensions(
        self,
        ber_contour: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: float = 1e-12
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Extract eye dimensions (width and height) from BER contour.
        
        Analyzes the BER contour matrix to find eye openings at the specified
        target BER level. Returns eye width in UI and eye height in volts.
        
        Args:
            ber_contour: BER contour matrix with shape (time_slices, voltage_bins).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values (typically in UI, 0 to 1).
            target_ber: Target BER level for eye opening detection.
                Default: 1e-12
        
        Returns:
            Dictionary containing:
                - 'eye_width_ui': Width of the eye opening in UI units
                - 'eye_width_samples': Width in number of time samples
                - 'eye_height_v': Height of the eye opening in volts
                - 'eye_height_samples': Height in number of voltage samples
                - 'eye_area_ui_v': Eye area (width * height) if available
                - 'num_eyes': Number of eye openings detected
                - 'all_widths': List of widths for all detected eyes (PAM4 has 3)
                - 'all_heights': List of heights for all detected eyes
        
        Example:
            >>> contour = BERContour(modulation=PAM4())
            >>> ber_matrix = contour.calculate(eye_pdf, voltage_bins)
            >>> dims = contour.get_eye_dimensions(
            ...     ber_matrix, voltage_bins, time_slices, target_ber=1e-12
            ... )
            >>> print(f"Eye width: {dims['eye_width_ui']:.3f} UI")
            >>> print(f"Eye height: {dims['eye_height_v']:.3f} V")
        """
        n_time, n_voltage = ber_contour.shape
        
        # Find eye width at voltage center
        center_v_idx = n_voltage // 2
        ber_horizontal = ber_contour[:, center_v_idx]
        
        # Find horizontal crossings (for eye width)
        above_target_h = ber_horizontal > target_ber
        h_crossings = np.where(np.diff(above_target_h.astype(int)) != 0)[0]
        
        # Calculate eye widths
        eye_widths = []
        for i in range(0, len(h_crossings) - 1, 2):
            if i + 1 < len(h_crossings):
                width = time_slices[h_crossings[i+1]] - time_slices[h_crossings[i]]
                eye_widths.append(width)
        
        # Find eye height at time center
        center_t_idx = n_time // 2
        ber_vertical = ber_contour[center_t_idx, :]
        
        # Find vertical crossings (for eye height)
        above_target_v = ber_vertical > target_ber
        v_crossings = np.where(np.diff(above_target_v.astype(int)) != 0)[0]
        
        # Calculate eye heights
        eye_heights = []
        for i in range(0, len(v_crossings) - 1, 2):
            if i + 1 < len(v_crossings):
                height = voltage_bins[v_crossings[i+1]] - voltage_bins[v_crossings[i]]
                eye_heights.append(height)
        
        # Determine primary eye dimensions
        # For NRZ: single eye
        # For PAM4: three eyes, report average or main eye
        eye_width_ui = np.mean(eye_widths) if eye_widths else 0.0
        eye_height_v = np.mean(eye_heights) if eye_heights else 0.0
        
        # Calculate width and height in samples
        eye_width_samples = 0
        if len(h_crossings) >= 2:
            # Find the widest eye
            max_width_samples = 0
            for i in range(0, len(h_crossings) - 1, 2):
                if i + 1 < len(h_crossings):
                    width_samples = h_crossings[i+1] - h_crossings[i]
                    if width_samples > max_width_samples:
                        max_width_samples = width_samples
            eye_width_samples = max_width_samples
        
        eye_height_samples = 0
        if len(v_crossings) >= 2:
            # Find the tallest eye
            max_height_samples = 0
            for i in range(0, len(v_crossings) - 1, 2):
                if i + 1 < len(v_crossings):
                    height_samples = v_crossings[i+1] - v_crossings[i]
                    if height_samples > max_height_samples:
                        max_height_samples = height_samples
            eye_height_samples = max_height_samples
        
        # Calculate eye area
        eye_area_ui_v = eye_width_ui * eye_height_v
        
        # Count number of eyes detected
        num_eyes = min(len(eye_widths), len(eye_heights))
        
        result = {
            'eye_width_ui': float(eye_width_ui),
            'eye_width_samples': int(eye_width_samples),
            'eye_height_v': float(eye_height_v),
            'eye_height_samples': int(eye_height_samples),
            'eye_area_ui_v': float(eye_area_ui_v),
            'num_eyes': int(num_eyes),
            'all_widths': [float(w) for w in eye_widths],
            'all_heights': [float(h) for h in eye_heights],
        }
        
        return result
    
    def get_contour_at_ber(
        self,
        ber_contour: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the contour line at a specific BER level.
        
        Returns the upper and lower voltage boundaries of the eye opening
        at the specified BER level for each time slice.
        
        Args:
            ber_contour: BER contour matrix.
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values.
            target_ber: Target BER level for contour extraction.
        
        Returns:
            Tuple of (upper_boundary, lower_boundary) arrays,
            each with length equal to time_slices.
        """
        n_time, n_voltage = ber_contour.shape
        
        upper_boundary = np.zeros(n_time)
        lower_boundary = np.zeros(n_time)
        
        for t in range(n_time):
            ber_slice = ber_contour[t, :]
            
            # Find where BER crosses target level
            above_target = ber_slice > target_ber
            crossings = np.where(np.diff(above_target.astype(int)) != 0)[0]
            
            if len(crossings) >= 2:
                # For each time slice, find the eye opening
                # In a well-formed eye, crossings come in pairs
                max_opening = 0
                upper_idx = crossings[0]
                lower_idx = crossings[1]
                
                for i in range(0, len(crossings) - 1, 2):
                    if i + 1 < len(crossings):
                        opening = crossings[i+1] - crossings[i]
                        if opening > max_opening:
                            max_opening = opening
                            upper_idx = crossings[i]
                            lower_idx = crossings[i+1]
                
                upper_boundary[t] = voltage_bins[upper_idx]
                lower_boundary[t] = voltage_bins[lower_idx]
            else:
                # No clear eye opening at this time slice
                upper_boundary[t] = np.nan
                lower_boundary[t] = np.nan
        
        return upper_boundary, lower_boundary
