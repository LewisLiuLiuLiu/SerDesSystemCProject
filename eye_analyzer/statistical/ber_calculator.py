"""
BER Calculator implementing strict OIF-CEI conditional probability.

This module provides the BERCalculator class for computing Bit Error Rate
contours from statistical eye diagram PDFs. It implements the OIF-CEI-04.0
specification for conditional probability BER calculation.

Key Features:
- Strict OIF-CEI conditional probability calculation (not simplified)
- Support for NRZ (1 eye) and PAM4 (3 eyes) modulation formats
- BER contour generation for eye diagram analysis
- Eye width/height extraction at target BER levels

Algorithm (from pystateye reference implementation):
    For NRZ (M=2):
        - Single eye centered at 0
        - CDF_0 = cumulative sum from eye center to upper voltages
        - CDF_1 = cumulative sum (reversed) from eye center to lower voltages
        - BER contour = concatenate([reverse(CDF_1), CDF_0])
    
    For PAM4 (M=4):
        - Three eyes with centers at thresholds between 4 levels
        - 4 signal levels: scaled from pulse response amplitude
        - 3 eye centers: at midpoints between levels
        - Each eye region has independent CDF calculation
        - CDFs are concatenated to form full contour
        
        Note: pystateye orders levels from positive to negative (descending),
        and voltage bins are also typically ordered from positive to negative.

Reference:
- OIF-CEI-04.0 specification, Section 2.C.5 and 2.B.2
- pystateye implementation: https://github.com/pystateye/statistical_eye

Example:
    >>> from eye_analyzer.statistical.ber_calculator import BERCalculator
    >>> from eye_analyzer.modulation import PAM4
    >>> import numpy as np
    >>> 
    >>> calc = BERCalculator(modulation=PAM4())
    >>> 
    >>> # Create eye PDF matrix (time_slices x voltage_bins)
    >>> eye_pdf = np.random.rand(64, 512)
    >>> eye_pdf = eye_pdf / eye_pdf.sum(axis=1, keepdims=True)
    >>> 
    >>> voltage_bins = np.linspace(-4, 4, 512)
    >>> ber_contour = calc.calculate_ber_contour(eye_pdf, voltage_bins)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union

from eye_analyzer.modulation import ModulationFormat, NRZ, PAM4


class BERCalculator:
    """
    Bit Error Rate calculator for statistical eye diagrams.
    
    Implements the OIF-CEI-04.0 specification for BER calculation using
    conditional probability. Supports NRZ and PAM4 modulation formats.
    
    This implementation follows the pystateye reference implementation
    which uses CDF-based BER contour calculation.
    
    Attributes:
        modulation: Modulation format instance (NRZ, PAM4, etc.)
        signal_levels: Normalized signal voltage levels (ordered descending)
        symbol_probabilities: Prior probability of each symbol
    
    Example:
        >>> calc = BERCalculator(modulation=PAM4())
        >>> ber_contour = calc.calculate_ber_contour(eye_pdf, voltage_bins)
    """
    
    def __init__(
        self,
        modulation: ModulationFormat,
        signal_levels: Optional[np.ndarray] = None,
        symbol_probabilities: Optional[np.ndarray] = None,
        signal_amplitude: float = 1.0
    ):
        """
        Initialize BER Calculator.
        
        Args:
            modulation: Modulation format instance (NRZ, PAM4, etc.)
            signal_levels: Optional custom signal levels. If None, uses
                standard levels for the modulation format.
                Note: levels are ordered from positive to negative (pystateye convention)
            symbol_probabilities: Optional custom symbol probabilities.
                If None, uses equal probability for all symbols.
            signal_amplitude: Amplitude of the signal (main cursor amplitude).
                Used to scale signal levels.
        """
        self.modulation = modulation
        self.signal_amplitude = signal_amplitude
        
        # Set signal levels
        if signal_levels is not None:
            self.signal_levels = np.array(signal_levels, dtype=float)
        else:
            # Use standard levels based on modulation format
            self.signal_levels = self._get_standard_levels()
        
        # Scale levels by signal amplitude
        self.signal_levels = self.signal_levels * signal_amplitude
        
        # Order from positive to negative (pystateye convention)
        self.signal_levels = np.sort(self.signal_levels)[::-1]
        
        # Set symbol probabilities (priors)
        if symbol_probabilities is not None:
            self.symbol_probabilities = np.array(symbol_probabilities, dtype=float)
            # Order to match signal_levels
            self.symbol_probabilities = self.symbol_probabilities[np.argsort(self.signal_levels)[::-1]]
            # Normalize to sum to 1
            self.symbol_probabilities = self.symbol_probabilities / np.sum(self.symbol_probabilities)
        else:
            # Equal probability for all symbols
            n_levels = len(self.signal_levels)
            self.symbol_probabilities = np.ones(n_levels) / n_levels
        
        # Validate
        if len(self.symbol_probabilities) != len(self.signal_levels):
            raise ValueError(
                f"symbol_probabilities length ({len(self.symbol_probabilities)}) "
                f"must match signal_levels length ({len(self.signal_levels)})"
            )
    
    def _get_standard_levels(self) -> np.ndarray:
        """
        Get standard signal levels for the modulation format.
        
        Returns:
            Array of signal levels in normalized units.
            - NRZ: [1, -1] (ordered descending)
            - PAM4: [1, 1/3, -1/3, -1] (ordered descending)
        """
        if self.modulation.name == 'nrz':
            # NRZ: 2 levels at 1 and -1 (descending)
            return np.array([1.0, -1.0])
        elif self.modulation.name == 'pam4':
            # PAM4: 4 levels at 1, 1/3, -1/3, -1 (descending)
            return np.array([1.0, 1.0/3.0, -1.0/3.0, -1.0])
        else:
            # Default: use modulation levels, sorted descending
            levels = self.modulation.get_levels().astype(float)
            return np.sort(levels)[::-1]
    
    def _get_thresholds(self) -> np.ndarray:
        """
        Get decision thresholds between signal levels.
        
        Returns:
            Array of threshold voltages (ordered from high to low).
            - NRZ: [0]
            - PAM4: [2/3, 0, -2/3] * amplitude (for levels [1, 1/3, -1/3, -1])
        """
        # Thresholds are midpoints between adjacent levels
        thresholds = (self.signal_levels[:-1] + self.signal_levels[1:]) / 2
        return thresholds
    
    def _get_eye_centers(self) -> np.ndarray:
        """
        Get the voltage centers of each eye opening.
        
        Returns:
            Array of eye center voltages (ordered from high to low).
            - NRZ: [0]
            - PAM4: [2/3, 0, -2/3] * amplitude
        """
        return self._get_thresholds()
    
    def _find_nearest_idx(self, voltage_bins: np.ndarray, value: float, 
                          ascending: bool = True) -> int:
        """
        Find index of nearest voltage bin to value.
        
        Args:
            voltage_bins: Array of voltage bin values
            value: Target value to find
            ascending: If True, assume voltage_bins is in ascending order.
                      If False, assume descending (pystateye convention).
        """
        if not ascending:
            # For descending arrays, flip the logic
            idx = int(np.argmin(np.abs(voltage_bins - value)))
        else:
            idx = int(np.argmin(np.abs(voltage_bins - value)))
        return idx
    
    def _calculate_ber_for_slice_nrz(
        self,
        pdf: np.ndarray,
        voltage_bins: np.ndarray
    ) -> np.ndarray:
        """
        Calculate BER contour for NRZ (single eye).
        
        Algorithm (from pystateye lines 262-267):
        - Eye center at threshold = 0
        - CDF_1 = cumulative sum from eye center upward (symbol 1 region)
        - CDF_0 = cumulative sum from eye center downward, reversed (symbol 0 region)
        - BER = concatenate([reverse(CDF_0), CDF_1])
        
        Args:
            pdf: Probability density function array (normalized, sum=1)
            voltage_bins: Voltage bin center values
        
        Returns:
            BER contour array with same shape as pdf
        """
        n_points = len(pdf)
        
        # Determine if voltage_bins is ascending or descending
        ascending = voltage_bins[0] < voltage_bins[-1]
        
        # Find eye center index (threshold at 0)
        eye_centers = self._get_eye_centers()
        idx_center = self._find_nearest_idx(voltage_bins, eye_centers[0], ascending)
        idx_center = np.clip(idx_center, 0, n_points - 1)
        
        if ascending:
            # Voltage bins: low to high (-2, ..., +2)
            # CDF from eye center to upper voltages
            cdf_upper = np.cumsum(pdf[idx_center:])
            
            # CDF from eye center to lower voltages - reversed
            lower_pdf = pdf[:idx_center]
            if len(lower_pdf) > 0:
                cdf_lower = np.cumsum(lower_pdf[::-1])
                cdf_lower = cdf_lower[::-1]
            else:
                cdf_lower = np.array([])
            
            contour = np.concatenate([cdf_lower, cdf_upper])
        else:
            # Voltage bins: high to low (+2, ..., -2) - pystateye style
            # CDF from eye center to lower voltages (which is upward in array index)
            cdf_lower = np.cumsum(pdf[idx_center:])
            
            # CDF from eye center to upper voltages - reversed
            upper_pdf = pdf[:idx_center]
            if len(upper_pdf) > 0:
                cdf_upper = np.cumsum(upper_pdf[::-1])
                cdf_upper = cdf_upper[::-1]
            else:
                cdf_upper = np.array([])
            
            contour = np.concatenate([cdf_upper, cdf_lower])
        
        return contour
    
    def _calculate_ber_for_slice_pam4(
        self,
        pdf: np.ndarray,
        voltage_bins: np.ndarray
    ) -> np.ndarray:
        """
        Calculate BER contour for PAM4 (three eyes).
        
        Algorithm (from pystateye lines 268-280):
        
        Signal levels (4): [1, 1/3, -1/3, -1] * amplitude (descending)
        Eye centers (3): [2/3, 0, -2/3] * amplitude (descending)
        
        Eye structure (from high voltage to low voltage):
            Level 1       Level 1/3     Level -1/3    Level -1
               |             |             |             |
               |   Eye 0     |   Eye 1     |   Eye 2     |
               |  (upper)    |  (middle)   |  (lower)    |
               |             |             |             |
            Thresh 2/3    Thresh 0      Thresh -2/3
        
        CDF calculations (following pystateye lines 269-278 exactly):
        - cdf_11: from eye_center[0] to end (toward higher voltages - which is DOWN in array if descending)
        - cdf_10_part1: from level[1] to eye_center[0], reversed
        - cdf_10_part2: from eye_center[1] to level[1], forward
        - cdf_01_part1: from level[2] to eye_center[1], reversed
        - cdf_01_part2: from eye_center[2] to level[2], forward
        - cdf_00: from start to eye_center[2], reversed
        
        Final contour: concatenate([reverse(cdf_00), cdf_01_part2, reverse(cdf_01_part1), 
                                    cdf_10_part2, reverse(cdf_10_part1), cdf_11])
        
        Args:
            pdf: Probability density function array
            voltage_bins: Voltage bin center values
        
        Returns:
            BER contour array
        """
        n_points = len(pdf)
        
        # Determine if voltage_bins is ascending or descending
        ascending = voltage_bins[0] < voltage_bins[-1]
        
        # Get indices for eye centers (3) and levels (4)
        eye_centers = self._get_eye_centers()  # [2/3, 0, -2/3] * amp
        levels = self.signal_levels  # [1, 1/3, -1/3, -1] * amp
        
        if ascending:
            # For ascending bins, we can flip the array, compute, then flip back
            # Or adjust indices - let's adjust indices
            idx_eye_centers = [self._find_nearest_idx(voltage_bins, ec, True) for ec in eye_centers]
            idx_levels = [self._find_nearest_idx(voltage_bins, lvl, True) for lvl in levels]
            
            # Sort indices to match pystateye's expectations
            # pystateye expects: eye_centers sorted from high to low
            # In ascending voltage: high voltages are at the END of the array
            # So we need to reverse the logic
            idx_eye_centers = idx_eye_centers[::-1]  # Now from low to high voltage index
            idx_levels = idx_levels[::-1]
        else:
            # Descending voltage bins - matches pystateye directly
            idx_eye_centers = [self._find_nearest_idx(voltage_bins, ec, False) for ec in eye_centers]
            idx_levels = [self._find_nearest_idx(voltage_bins, lvl, False) for lvl in levels]
        
        # Ensure indices are within bounds
        idx_eye_centers = [np.clip(idx, 0, n_points - 1) for idx in idx_eye_centers]
        idx_levels = [np.clip(idx, 0, n_points - 1) for idx in idx_levels]
        
        # Following pystateye lines 269-278:
        # In pystateye's coordinate system (descending voltage):
        # idx_eye_centers[0] = upper eye center (highest voltage)
        # idx_eye_centers[1] = middle eye center
        # idx_eye_centers[2] = lower eye center (lowest voltage)
        # idx_levels[0..3] = 4 signal levels from high to low
        
        if ascending:
            # In ascending voltage bins:
            # Low voltages at idx 0, high voltages at idx -1
            # We need to map pystateye's indices to our coordinate system
            
            # Let's work with a flipped copy and then flip back
            pdf_flipped = pdf[::-1]
            vb_flipped = voltage_bins[::-1]  # Now descending
            
            # Recalculate indices for flipped array
            idx_eye_centers = [self._find_nearest_idx(vb_flipped, ec, False) for ec in eye_centers]
            idx_levels = [self._find_nearest_idx(vb_flipped, lvl, False) for lvl in levels]
            
            # Calculate contour on flipped array
            contour_flipped = self._calculate_pam4_contour_descending(
                pdf_flipped, idx_eye_centers, idx_levels
            )
            
            # Flip back to match original orientation
            contour = contour_flipped[::-1]
        else:
            # Descending voltage bins - direct calculation
            contour = self._calculate_pam4_contour_descending(
                pdf, idx_eye_centers, idx_levels
            )
        
        # Ensure correct length
        if len(contour) != n_points:
            if len(contour) < n_points:
                contour = np.pad(contour, (0, n_points - len(contour)), mode='edge')
            else:
                contour = contour[:n_points]
        
        return contour
    
    def _calculate_pam4_contour_descending(
        self,
        pdf: np.ndarray,
        idx_eye_centers: List[int],
        idx_levels: List[int]
    ) -> np.ndarray:
        """
        Calculate PAM4 contour for descending voltage bins.
        
        This is the core implementation matching pystateye exactly.
        """
        # Ensure proper ordering (high to low)
        # idx_eye_centers should be [high_center, mid_center, low_center]
        # idx_levels should be [level_high, level_mid_high, level_mid_low, level_low]
        
        n_points = len(pdf)
        
        # Line 270: cdf_11 from eye_center[0] to end
        if idx_eye_centers[0] < n_points:
            cdf_11 = np.cumsum(pdf[idx_eye_centers[0]:])
        else:
            cdf_11 = np.array([0.0])
        
        # Line 272: cdf_10_part1 from level[1] to eye_center[0], reversed
        if idx_eye_centers[0] > idx_levels[1]:
            cdf_10_part1 = np.cumsum(pdf[idx_levels[1]:idx_eye_centers[0]][::-1])[::-1]
        else:
            cdf_10_part1 = np.array([])
        
        # Line 273: cdf_10_part2 from eye_center[1] to level[1], forward
        if idx_levels[1] > idx_eye_centers[1]:
            cdf_10_part2 = np.cumsum(pdf[idx_eye_centers[1]:idx_levels[1]+1])
        else:
            cdf_10_part2 = np.array([])
        
        # Line 275: cdf_01_part1 from level[2] to eye_center[1], reversed
        if idx_eye_centers[1] > idx_levels[2]:
            cdf_01_part1 = np.cumsum(pdf[idx_levels[2]:idx_eye_centers[1]][::-1])[::-1]
        else:
            cdf_01_part1 = np.array([])
        
        # Line 276: cdf_01_part2 from eye_center[2] to level[2], forward
        if idx_levels[2] > idx_eye_centers[2]:
            cdf_01_part2 = np.cumsum(pdf[idx_eye_centers[2]:idx_levels[2]+1])
        else:
            cdf_01_part2 = np.array([])
        
        # Line 278: cdf_00 from start to eye_center[2], reversed
        if idx_eye_centers[2] > 0:
            cdf_00 = np.cumsum(pdf[:idx_eye_centers[2]][::-1])[::-1]
        else:
            cdf_00 = np.array([])
        
        # Line 280: concatenate all regions
        # Order: reverse(cdf_00), cdf_01_part2, reverse(cdf_01_part1), 
        #        cdf_10_part2, reverse(cdf_10_part1), cdf_11
        parts = []
        
        if len(cdf_00) > 0:
            parts.append(cdf_00[::-1])
        if len(cdf_01_part2) > 0:
            parts.append(cdf_01_part2)
        if len(cdf_01_part1) > 0:
            parts.append(cdf_01_part1[::-1])
        if len(cdf_10_part2) > 0:
            parts.append(cdf_10_part2)
        if len(cdf_10_part1) > 0:
            parts.append(cdf_10_part1[::-1])
        if len(cdf_11) > 0:
            parts.append(cdf_11)
        
        if parts:
            contour = np.concatenate(parts)
        else:
            contour = np.zeros(n_points)
        
        return contour
    
    def _calculate_ber_for_slice(
        self,
        pdf: np.ndarray,
        voltage_bins: np.ndarray
    ) -> np.ndarray:
        """
        Calculate BER for a single time slice using OIF-CEI conditional probability.
        
        Args:
            pdf: Probability density function array (normalized, sum=1)
            voltage_bins: Voltage bin center values (same length as pdf)
        
        Returns:
            BER array with same shape as pdf
        """
        if len(pdf) != len(voltage_bins):
            raise ValueError(
                f"pdf length ({len(pdf)}) must match voltage_bins length ({len(voltage_bins)})"
            )
        
        if len(pdf) == 0:
            raise ValueError("pdf cannot be empty")
        
        # Dispatch based on number of levels
        n_levels = len(self.signal_levels)
        
        if n_levels == 2:
            return self._calculate_ber_for_slice_nrz(pdf, voltage_bins)
        elif n_levels == 4:
            return self._calculate_ber_for_slice_pam4(pdf, voltage_bins)
        else:
            # Generic fallback
            return self._calculate_ber_for_slice_nrz(pdf, voltage_bins)
    
    def calculate_ber_contour(
        self,
        eye_pdf: np.ndarray,
        voltage_bins: np.ndarray,
        use_oif_method: bool = True
    ) -> np.ndarray:
        """
        Calculate BER contour matrix for the entire eye diagram.
        
        Args:
            eye_pdf: Eye diagram PDF matrix with shape (time_slices, voltage_bins)
            voltage_bins: Voltage bin center values
            use_oif_method: If True, use strict OIF-CEI method; if False,
                use simplified min(CDF, 1-CDF)*2 method for comparison
        
        Returns:
            BER contour matrix with same shape as eye_pdf
        """
        if eye_pdf.shape[1] != len(voltage_bins):
            raise ValueError(
                f"eye_pdf second dimension ({eye_pdf.shape[1]}) must match "
                f"voltage_bins length ({len(voltage_bins)})"
            )
        
        n_time = eye_pdf.shape[0]
        
        ber_contour = np.zeros_like(eye_pdf)
        
        if use_oif_method:
            for t in range(n_time):
                pdf_slice = eye_pdf[t, :]
                pdf_sum = np.sum(pdf_slice)
                if pdf_sum > 0:
                    pdf_slice = pdf_slice / pdf_sum
                    ber_contour[t, :] = self._calculate_ber_for_slice(pdf_slice, voltage_bins)
                else:
                    ber_contour[t, :] = 0
        else:
            for t in range(n_time):
                pdf_slice = eye_pdf[t, :]
                pdf_sum = np.sum(pdf_slice)
                if pdf_sum > 0:
                    pdf_slice = pdf_slice / pdf_sum
                    cdf = np.cumsum(pdf_slice)
                    ber_contour[t, :] = np.minimum(cdf, 1 - cdf) * 2
                else:
                    ber_contour[t, :] = 0
        
        return ber_contour
    
    def find_eye_width(
        self,
        ber_contour: np.ndarray,
        time_slices: np.ndarray,
        target_ber: float = 1e-12,
        eye_center_voltage_idx: Optional[int] = None
    ) -> Dict[str, Union[float, List]]:
        """
        Find eye width at target BER level (backward compatibility alias).
        
        This method is kept for backward compatibility.
        Use find_eye_openings() for comprehensive measurements.
        """
        n_time, n_voltage = ber_contour.shape
        
        if eye_center_voltage_idx is None:
            eye_center_voltage_idx = n_voltage // 2
        
        # Get BER values along eye center horizontal line
        ber_center_line = ber_contour[:, eye_center_voltage_idx]
        
        # Find where BER crosses target level
        above_target = ber_center_line > target_ber
        crossings = np.where(np.diff(above_target.astype(int)) != 0)[0]
        
        if len(crossings) < 2:
            return {'eye_width_ui': 0.0, 'crossings': []}
        
        # Pair up crossings to find eye openings
        eye_widths = []
        for i in range(0, len(crossings) - 1, 2):
            if i + 1 < len(crossings):
                width = time_slices[crossings[i+1]] - time_slices[crossings[i]]
                eye_widths.append({
                    'width': width,
                    'start_idx': crossings[i],
                    'end_idx': crossings[i+1]
                })
        
        if not eye_widths:
            return {'eye_width_ui': 0.0, 'crossings': crossings.tolist()}
        
        # Find widest eye
        widest = max(eye_widths, key=lambda x: x['width'])
        
        return {
            'eye_width_ui': widest['width'],
            'eye_width_samples': widest['end_idx'] - widest['start_idx'],
            'start_idx': widest['start_idx'],
            'end_idx': widest['end_idx'],
            'all_widths': [w['width'] for w in eye_widths],
            'crossings': crossings.tolist()
        }
    
    def find_eye_height(
        self,
        ber_contour: np.ndarray,
        voltage_bins: np.ndarray,
        target_ber: float = 1e-12,
        time_center_idx: Optional[int] = None
    ) -> Dict[str, Union[List[float], float, List]]:
        """
        Find eye height at target BER level (backward compatibility alias).
        
        This method is kept for backward compatibility.
        Use find_eye_openings() for comprehensive measurements.
        """
        n_time, n_voltage = ber_contour.shape
        
        if time_center_idx is None:
            time_center_idx = n_time // 2
        
        # Get BER values along time center vertical line
        ber_center_line = ber_contour[time_center_idx, :]
        
        # Find where BER crosses target level
        above_target = ber_center_line > target_ber
        crossings = np.where(np.diff(above_target.astype(int)) != 0)[0]
        
        if len(crossings) < 2:
            return {'eye_heights_v': [], 'eye_heights_mean_v': 0, 'crossings': []}
        
        # For PAM4, there are 3 eye openings, so we expect multiple crossings
        eye_heights = []
        for i in range(0, len(crossings) - 1, 2):
            if i + 1 < len(crossings):
                height = voltage_bins[crossings[i+1]] - voltage_bins[crossings[i]]
                eye_heights.append({
                    'height': height,
                    'start_idx': crossings[i],
                    'end_idx': crossings[i+1]
                })
        
        return {
            'eye_heights_v': [eh['height'] for eh in eye_heights],
            'eye_heights_mean_v': np.mean([eh['height'] for eh in eye_heights]) if eye_heights else 0,
            'crossings': crossings.tolist()
        }
    
    def find_eye_openings(
        self,
        ber_contour: np.ndarray,
        voltage_bins: np.ndarray,
        time_slices: np.ndarray,
        target_ber: float = 1e-12
    ) -> Dict:
        """
        Find eye openings at target BER level.
        
        Returns eye width, height, and area measurements.
        """
        n_time, n_voltage = ber_contour.shape
        
        # Find eye width at center voltage
        center_v_idx = n_voltage // 2
        ber_horizontal = ber_contour[:, center_v_idx]
        
        # Find crossings for eye width
        above_target = ber_horizontal > target_ber
        h_crossings = np.where(np.diff(above_target.astype(int)) != 0)[0]
        
        # Find eye height at center time
        center_t_idx = n_time // 2
        ber_vertical = ber_contour[center_t_idx, :]
        
        # Find crossings for eye height
        above_target_v = ber_vertical > target_ber
        v_crossings = np.where(np.diff(above_target_v.astype(int)) != 0)[0]
        
        # Calculate widths and heights
        eye_widths = []
        for i in range(0, len(h_crossings) - 1, 2):
            if i + 1 < len(h_crossings):
                width = time_slices[h_crossings[i+1]] - time_slices[h_crossings[i]]
                eye_widths.append(width)
        
        eye_heights = []
        for i in range(0, len(v_crossings) - 1, 2):
            if i + 1 < len(v_crossings):
                height = voltage_bins[v_crossings[i+1]] - voltage_bins[v_crossings[i]]
                eye_heights.append(height)
        
        return {
            'eye_widths_ui': eye_widths,
            'eye_heights_v': eye_heights,
            'mean_width_ui': np.mean(eye_widths) if eye_widths else 0,
            'mean_height_v': np.mean(eye_heights) if eye_heights else 0,
        }


def calculate_ber_simplified(
    eye_pdf: np.ndarray,
    voltage_bins: np.ndarray
) -> np.ndarray:
    """
    Calculate BER using simplified min(CDF, 1-CDF)*2 method.
    
    This is the INCORRECT simplified method for comparison purposes only.
    The OIF-CEI standard requires conditional probability calculation.
    """
    n_time = eye_pdf.shape[0]
    ber_contour = np.zeros_like(eye_pdf)
    
    for t in range(n_time):
        pdf_slice = eye_pdf[t, :]
        pdf_sum = np.sum(pdf_slice)
        if pdf_sum > 0:
            pdf_slice = pdf_slice / pdf_sum
            cdf = np.cumsum(pdf_slice)
            ber_contour[t, :] = np.minimum(cdf, 1 - cdf) * 2
    
    return ber_contour
