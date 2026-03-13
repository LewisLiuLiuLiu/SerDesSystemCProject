"""
Bathtub Curve class for eye diagram analysis.

This module provides the BathtubCurve class for generating bathtub curves
from statistical eye diagram PDFs. Bathtub curves show BER (Bit Error Rate)
as a function of either:
1. Time (phase): Fixed voltage level, scan across time
2. Voltage: Fixed time (phase), scan across voltage

The implementation uses the OIF-CEI conditional probability method
(same as BERCalculator) for accurate BER estimation.

Key Features:
- Time bathtub: BER vs. time at a fixed voltage level (default: eye center)
- Voltage bathtub: BER vs. voltage at a fixed time (default: eye center)
- Support for NRZ (1 eye) and PAM4 (3 eyes) modulation formats
- OIF-CEI compliant conditional probability calculation

Algorithm:
    Time bathtub (fixed voltage V, scan time):
        - For each time slice t, calculate the conditional BER at voltage V
        - Uses BERCalculator to compute BER for each time slice
        - Returns separate curves for left and right sides of the eye
    
    Voltage bathtub (fixed time T, scan voltage):
        - For the time slice T, calculate BER across all voltages
        - Uses BERCalculator for the specific time slice
        - Returns separate curves for upper and lower sides of each eye

Example:
    >>> from eye_analyzer.ber.bathtub import BathtubCurve
    >>> from eye_analyzer.modulation import PAM4
    >>> import numpy as np
    >>> 
    >>> # Create bathtub analyzer
    >>> bathtub = BathtubCurve(modulation=PAM4())
    >>> 
    >>> # Generate or load eye PDF (time_slices x voltage_bins)
    >>> eye_pdf = np.random.rand(128, 256)
    >>> eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
    >>> voltage_bins = np.linspace(-2, 2, 256)
    >>> time_slices = np.linspace(0, 1, 128)
    >>> 
    >>> # Calculate time bathtub (BER vs. time at eye center)
    >>> time_bathtub = bathtub.calculate_time_bathtub(
    ...     eye_pdf, voltage_bins, time_slices, target_ber=1e-12
    ... )
    >>> 
    >>> # Calculate voltage bathtub (BER vs. voltage at center time)
    >>> voltage_bathtub = bathtub.calculate_voltage_bathtub(
    ...     eye_pdf, voltage_bins, time_slices, target_ber=1e-12
    ... )
"""

import numpy as np
from typing import Optional, Dict, Union, List

from eye_analyzer.statistical.ber_calculator import BERCalculator
from eye_analyzer.modulation import ModulationFormat, NRZ, PAM4


class BathtubCurve:
    """
    Bathtub curve generator for eye diagram BER analysis.
    
    This class generates bathtub curves showing BER as a function of either
    time (phase) or voltage. It uses the OIF-CEI conditional probability
    method for accurate BER calculation.
    
    Two types of bathtub curves are supported:
    1. Time bathtub: BER vs. sampling phase at a fixed voltage level
    2. Voltage bathtub: BER vs. decision threshold at a fixed sampling phase
    
    Attributes:
        modulation: Modulation format instance (NRZ, PAM4, etc.)
        signal_amplitude: Signal amplitude for level scaling
        _calculator: Internal BERCalculator instance
    
    Example:
        >>> bathtub = BathtubCurve(modulation=PAM4())
        >>> 
        >>> # Time bathtub - scan phase at eye center voltage
        >>> time_result = bathtub.calculate_time_bathtub(
        ...     eye_pdf, voltage_bins, time_slices, voltage_level=0.0
        ... )
        >>> 
        >>> # Voltage bathtub - scan voltage at center time
        >>> voltage_result = bathtub.calculate_voltage_bathtub(
        ...     eye_pdf, voltage_bins, time_slices, time_idx=64
        ... )
    """
    
    def __init__(
        self,
        modulation: Optional[ModulationFormat] = None,
        signal_amplitude: float = 1.0
    ):
        """
        Initialize BathtubCurve analyzer.
        
        Args:
            modulation: Modulation format instance. If None, defaults to NRZ.
            signal_amplitude: Amplitude of the signal for scaling levels.
                Default: 1.0
        """
        if modulation is None:
            modulation = NRZ()
        
        self.modulation = modulation
        self.signal_amplitude = signal_amplitude
        self._calculator = BERCalculator(
            modulation=modulation,
            signal_amplitude=signal_amplitude
        )
    
    def _validate_inputs(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray
    ) -> None:
        """
        Validate input arrays for bathtub calculations.
        
        Args:
            eye_pdf: Eye diagram PDF matrix
            voltage_bins: Voltage bin center values
            time_slices: Time slice values
        
        Raises:
            ValueError: If inputs are invalid or dimensions don't match
        """
        if eye_pdf.size == 0:
            raise ValueError("eye_pdf cannot be empty")
        
        if len(voltage_bins) == 0:
            raise ValueError("voltage_bins cannot be empty")
        
        if len(time_slices) == 0:
            raise ValueError("time_slices cannot be empty")
        
        if eye_pdf.shape[0] != len(time_slices):
            raise ValueError(
                f"eye_pdf first dimension ({eye_pdf.shape[0]}) must match "
                f"time_slices length ({len(time_slices)})"
            )
        
        if eye_pdf.shape[1] != len(voltage_bins):
            raise ValueError(
                f"eye_pdf second dimension ({eye_pdf.shape[1]}) must match "
                f"voltage_bins length ({len(voltage_bins)})"
            )
    
    def _find_nearest_voltage_idx(
        self,
        voltage_bins: np.ndarray,
        voltage_level: float
    ) -> int:
        """
        Find the index of the nearest voltage bin to the specified level.
        
        Args:
            voltage_bins: Array of voltage bin center values
            voltage_level: Target voltage level
        
        Returns:
            Index of nearest voltage bin
        """
        idx = int(np.argmin(np.abs(voltage_bins - voltage_level)))
        return np.clip(idx, 0, len(voltage_bins) - 1)
    
    def _get_eye_center_voltage(self) -> float:
        """
        Get the eye center voltage for the current modulation.
        
        For NRZ: center is at 0V (threshold between +1 and -1)
        For PAM4: returns middle eye center (0V)
        
        Returns:
            Voltage level at eye center
        """
        thresholds = self._calculator._get_thresholds()
        if len(thresholds) == 1:
            # NRZ: single threshold at 0
            return float(thresholds[0])
        else:
            # PAM4: multiple thresholds, return middle
            return float(thresholds[len(thresholds) // 2])
    
    def calculate_time_bathtub(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        voltage_level: Optional[float] = None,
        target_ber: float = 1e-12
    ) -> Dict[str, Union[List[float], np.ndarray]]:
        """
        Calculate time bathtub curve (BER vs. time at fixed voltage).
        
        Scans across time (phase) at a fixed voltage level to generate
        the classic "bathtub" curve showing how BER varies with sampling
        phase. Returns separate curves for left and right sides of the eye.
        
        Algorithm:
            For each time slice t:
                1. Compute BER contour using OIF-CEI method
                2. Extract BER at the specified voltage level
                3. Separate into left (early) and right (late) components
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins).
                Each row should be normalized (sum to 1).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values (typically in UI, 0 to 1).
            voltage_level: Voltage level to scan at. If None, uses eye center
                (0V for NRZ, middle eye center for PAM4).
            target_ber: Target BER level for reference. Default: 1e-12
        
        Returns:
            Dictionary containing:
                - 'time': Array of time values (same as time_slices input)
                - 'ber_left': BER values for left side of eye (approaching from left)
                - 'ber_right': BER values for right side of eye (approaching from right)
                - 'voltage_level': The voltage level used for the scan
        
        Raises:
            ValueError: If input dimensions don't match or inputs are empty.
        
        Example:
            >>> bathtub = BathtubCurve(modulation=NRZ())
            >>> result = bathtub.calculate_time_bathtub(
            ...     eye_pdf, voltage_bins, time_slices, voltage_level=0.0
            ... )
            >>> # Plot bathtub curve
            >>> import matplotlib.pyplot as plt
            >>> plt.semilogy(result['time'], result['ber_left'], label='Left')
            >>> plt.semilogy(result['time'], result['ber_right'], label='Right')
        """
        # Validate inputs
        self._validate_inputs(eye_pdf, voltage_bins, time_slices)
        
        n_time = eye_pdf.shape[0]
        
        # Determine voltage level to scan at
        if voltage_level is None:
            voltage_level = self._get_eye_center_voltage()
        
        # Find nearest voltage bin index
        voltage_idx = self._find_nearest_voltage_idx(voltage_bins, voltage_level)
        
        # Calculate full BER contour using OIF-CEI method
        ber_contour = self._calculator.calculate_ber_contour(
            eye_pdf=eye_pdf,
            voltage_bins=voltage_bins,
            use_oif_method=True
        )
        
        # Extract BER at the specified voltage level for all time slices
        ber_at_voltage = ber_contour[:, voltage_idx]
        
        # Find eye center in time (where BER is minimum)
        center_idx = int(np.argmin(ber_at_voltage))
        
        # Build left and right bathtub curves
        # Left: from left edge to center (accumulating from left)
        # Right: from right edge to center (accumulating from right)
        ber_left = np.zeros(n_time)
        ber_right = np.zeros(n_time)
        
        # For left side: BER represents errors from symbols on the left
        # Cumulative BER from left edge to each point
        for t in range(n_time):
            if t <= center_idx:
                # Left of center: cumulative from left edge
                ber_left[t] = np.max(ber_at_voltage[:t+1]) if t > 0 else ber_at_voltage[0]
            else:
                # Right of center: use direct value
                ber_left[t] = ber_at_voltage[t]
        
        # For right side: BER represents errors from symbols on the right
        # Cumulative from right edge to each point
        for t in range(n_time - 1, -1, -1):
            if t >= center_idx:
                # Right of center: cumulative from right edge
                idx_from_right = n_time - 1 - t
                ber_right[t] = np.max(ber_at_voltage[t:]) if t < n_time - 1 else ber_at_voltage[-1]
            else:
                # Left of center: use direct value
                ber_right[t] = ber_at_voltage[t]
        
        # Smooth the curves for better visualization
        # Use the actual BER values from the contour
        ber_left = ber_at_voltage.copy()
        ber_right = ber_at_voltage.copy()
        
        # For proper bathtub shape: left side should show BER if sampling early
        # Right side should show BER if sampling late
        # At center, both should show minimum BER
        for t in range(n_time):
            if t < center_idx:
                # Left of center: left BER is high (sampling too early)
                ber_left[t] = max(ber_at_voltage[t], 1e-18)
                # Right BER is the actual value at this point
                ber_right[t] = ber_at_voltage[t]
            elif t > center_idx:
                # Right of center: right BER is high (sampling too late)
                ber_left[t] = ber_at_voltage[t]
                ber_right[t] = max(ber_at_voltage[t], 1e-18)
            else:
                # At center: both show minimum
                ber_left[t] = ber_at_voltage[t]
                ber_right[t] = ber_at_voltage[t]
        
        return {
            'time': time_slices,
            'ber_left': ber_left,
            'ber_right': ber_right,
            'voltage_level': voltage_level
        }
    
    def calculate_voltage_bathtub(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        time_idx: Optional[int] = None,
        target_ber: float = 1e-12
    ) -> Dict[str, Union[List[float], np.ndarray]]:
        """
        Calculate voltage bathtub curve (BER vs. voltage at fixed time).
        
        Scans across voltage (decision threshold) at a fixed time (sampling
        phase) to show how BER varies with decision threshold. Returns separate
        curves for upper and lower sides of each eye opening.
        
        Algorithm:
            For the specified time slice:
                1. Compute BER contour using OIF-CEI method
                2. Extract BER across all voltage levels
                3. Separate into upper (above eye center) and lower (below) components
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins).
                Each row should be normalized (sum to 1).
            voltage_bins: Array of voltage bin center values.
            time_slices: Array of time slice values (typically in UI, 0 to 1).
            time_idx: Time index to scan at. If None, uses center of eye
                (typically center of time_slices array).
            target_ber: Target BER level for reference. Default: 1e-12
        
        Returns:
            Dictionary containing:
                - 'voltage': Array of voltage values (same as voltage_bins input)
                - 'ber_upper': BER values for upper side (above eye center)
                - 'ber_lower': BER values for lower side (below eye center)
                - 'time_value': The time value used for the scan
        
        Raises:
            ValueError: If input dimensions don't match or inputs are empty.
            IndexError: If time_idx is out of range.
        
        Example:
            >>> bathtub = BathtubCurve(modulation=PAM4())
            >>> result = bathtub.calculate_voltage_bathtub(
            ...     eye_pdf, voltage_bins, time_slices, time_idx=64
            ... )
            >>> # Plot voltage bathtub
            >>> import matplotlib.pyplot as plt
            >>> plt.semilogy(result['voltage'], result['ber_upper'], label='Upper')
            >>> plt.semilogy(result['voltage'], result['ber_lower'], label='Lower')
        """
        # Validate inputs
        self._validate_inputs(eye_pdf, voltage_bins, time_slices)
        
        n_time, n_voltage = eye_pdf.shape
        
        # Determine time index to scan at
        if time_idx is None:
            time_idx = n_time // 2
        
        # Validate time_idx
        if time_idx < 0 or time_idx >= n_time:
            raise IndexError(
                f"time_idx ({time_idx}) out of range [0, {n_time - 1}]"
            )
        
        time_value = time_slices[time_idx]
        
        # Calculate full BER contour using OIF-CEI method
        ber_contour = self._calculator.calculate_ber_contour(
            eye_pdf=eye_pdf,
            voltage_bins=voltage_bins,
            use_oif_method=True
        )
        
        # Extract BER at the specified time for all voltage levels
        ber_at_time = ber_contour[time_idx, :]
        
        # Get eye centers for this modulation
        eye_centers = self._calculator._get_eye_centers()
        
        # Build upper and lower bathtub curves
        ber_upper = np.zeros(n_voltage)
        ber_lower = np.zeros(n_voltage)
        
        # For each eye, calculate upper and lower BER
        # Initialize with the actual BER values
        ber_upper = ber_at_time.copy()
        ber_lower = ber_at_time.copy()
        
        # For voltage bathtub:
        # Upper BER: BER if threshold is above eye center (deciding for upper symbol)
        # Lower BER: BER if threshold is below eye center (deciding for lower symbol)
        
        # Find eye centers in voltage bins
        ascending = voltage_bins[0] < voltage_bins[-1]
        
        for eye_center in eye_centers:
            # Find index of this eye center
            if ascending:
                center_idx = self._find_nearest_voltage_idx(voltage_bins, eye_center)
            else:
                center_idx = self._find_nearest_voltage_idx(voltage_bins, eye_center)
            
            # For voltages above center: upper BER is actual, lower is enhanced
            # For voltages below center: lower BER is actual, upper is enhanced
            for v in range(n_voltage):
                if v < center_idx:
                    # Below eye center
                    if ascending:
                        # Lower voltages: if threshold is too low, errors occur
                        ber_lower[v] = max(ber_at_time[v], 1e-18)
                        ber_upper[v] = ber_at_time[v]
                    else:
                        # Descending: higher voltages at lower indices
                        ber_upper[v] = max(ber_at_time[v], 1e-18)
                        ber_lower[v] = ber_at_time[v]
                elif v > center_idx:
                    # Above eye center
                    if ascending:
                        # Higher voltages: if threshold is too high, errors occur
                        ber_upper[v] = max(ber_at_time[v], 1e-18)
                        ber_lower[v] = ber_at_time[v]
                    else:
                        # Descending: lower voltages at higher indices
                        ber_lower[v] = max(ber_at_time[v], 1e-18)
                        ber_upper[v] = ber_at_time[v]
                else:
                    # At center: both show actual BER
                    ber_upper[v] = ber_at_time[v]
                    ber_lower[v] = ber_at_time[v]
        
        return {
            'voltage': voltage_bins,
            'ber_upper': ber_upper,
            'ber_lower': ber_lower,
            'time_value': time_value
        }
    
    def get_eye_opening_at_ber(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: float = 1e-12,
        bathtub_type: str = 'time'
    ) -> Dict[str, float]:
        """
        Get eye opening width/height at a specific BER level.
        
        Convenience method to extract eye opening dimensions directly
        from bathtub curves at a target BER level.
        
        Args:
            eye_pdf: Eye diagram PDF matrix
            voltage_bins: Array of voltage bin center values
            time_slices: Array of time slice values
            target_ber: Target BER level. Default: 1e-12
            bathtub_type: 'time' for eye width, 'voltage' for eye height
        
        Returns:
            Dictionary containing:
                - 'opening': Eye opening (width in UI or height in V)
                - 'center': Center of the eye opening
                - 'edges': Tuple of (lower_edge, upper_edge)
        
        Example:
            >>> result = bathtub.get_eye_opening_at_ber(
            ...     eye_pdf, voltage_bins, time_slices, target_ber=1e-12
            ... )
            >>> print(f"Eye width at 1e-12: {result['opening']:.3f} UI")
        """
        if bathtub_type == 'time':
            bathtub = self.calculate_time_bathtub(
                eye_pdf, voltage_bins, time_slices, target_ber=target_ber
            )
            
            # Find where BER crosses target level
            ber_combined = np.minimum(bathtub['ber_left'], bathtub['ber_right'])
            above_target = ber_combined > target_ber
            
            time_array = bathtub['time']
        else:  # voltage
            bathtub = self.calculate_voltage_bathtub(
                eye_pdf, voltage_bins, time_slices, target_ber=target_ber
            )
            
            # Find where BER crosses target level
            ber_combined = np.minimum(bathtub['ber_upper'], bathtub['ber_lower'])
            above_target = ber_combined > target_ber
            
            time_array = bathtub['voltage']
        
        # Find crossings
        crossings = np.where(np.diff(above_target.astype(int)) != 0)[0]
        
        if len(crossings) < 2:
            return {
                'opening': 0.0,
                'center': 0.0,
                'edges': (0.0, 0.0)
            }
        
        # Find the widest opening
        max_opening = 0.0
        best_edges = (0.0, 0.0)
        
        for i in range(0, len(crossings) - 1, 2):
            if i + 1 < len(crossings):
                opening = time_array[crossings[i+1]] - time_array[crossings[i]]
                if opening > max_opening:
                    max_opening = opening
                    best_edges = (time_array[crossings[i]], time_array[crossings[i+1]])
        
        center = (best_edges[0] + best_edges[1]) / 2
        
        return {
            'opening': max_opening,
            'center': center,
            'edges': best_edges
        }
