"""
Golden CDR Eye Diagram Analysis Scheme - Fixed Version with 2-UI Display

This module implements the Golden CDR (ideal clock-based) eye diagram analysis
using MATLAB-style reshape method with 2-UI centered display.

Key characteristics:
- Time reference: Ideal clock t_golden[m] = m * UI as 0 UI
- Observation window: 2 UI ([-0.5, 1.5) centered eye pattern)
- Eye center at 0.5 UI is positioned at the center of the display
- Core algorithm: MATLAB-style reshape + 2-UI centered accumulation
- Jitter analysis: TIE extraction + Dual-Dirac model
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from scipy.interpolate import PchipInterpolator

from .base import BaseScheme
from ..jitter import JitterDecomposer
from ..modulation import ModulationFormat


class GoldenCdrScheme(BaseScheme):
    """
    Golden CDR Eye Diagram Analysis Scheme (Fixed with 2-UI Display)
    
    This scheme constructs standard eye diagrams based on an ideal clock,
    displaying 2 UI window with the eye centered in the middle.
    
    The analysis:
    1. Segments waveform by UI periods using reshape
    2. Duplicates edge samples to create centered 2-UI eye pattern
    3. Accumulates to 2D histogram covering [-0.5, 1.5) UI range
    
    Display layout:
    - Left half [-0.5, 0): 2nd half of previous UI [0.5, 1)
    - Center [0, 1): Complete current UI [0, 1)
    - Right half [1, 1.5): 1st half of next UI [0, 0.5)
    
    Attributes:
        ui: Unit interval in seconds
        ui_bins: Number of bins for phase axis (default: 128, covers 2 UI)
        amp_bins: Number of bins for amplitude axis (default: 128)
        jitter_method: Jitter extraction method ('dual-dirac', 'tail-fit', 'auto')
        
    Example:
        >>> scheme = GoldenCdrScheme(ui=2.5e-11, ui_bins=128, amp_bins=128)
        >>> metrics = scheme.analyze(time_array, voltage_array)
        >>> print(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
        >>> print(f"RJ: {metrics['rj_sigma']*1e12:.2f} ps")
    """
    
    def __init__(self, 
                 ui: float, 
                 modulation: Union[str, ModulationFormat] = 'nrz',
                 ui_bins: int = 128, 
                 amp_bins: int = 128,
                 jitter_method: str = 'dual-dirac'):
        """
        Initialize the Golden CDR scheme.
        
        Args:
            ui: Unit interval in seconds
            modulation: Modulation format ('nrz', 'pam4') or ModulationFormat object
            ui_bins: Number of bins for phase axis (default: 128, covers 2 UI)
            amp_bins: Number of bins for amplitude axis (default: 128)
            jitter_method: Jitter extraction method
                          ('dual-dirac', 'tail-fit', 'auto')
                          Default: 'dual-dirac'
        """
        super().__init__(ui, modulation, ui_bins, amp_bins)
        self.jitter_method = jitter_method
        self._jitter_decomposer = JitterDecomposer(ui, jitter_method)
        
    def analyze(self, time_array: np.ndarray, voltage_array: np.ndarray,
                target_ber: float = 1e-12,
                **kwargs) -> Dict[str, Any]:
        """
        Perform Golden CDR eye diagram analysis with 2-UI centered display.
        
        Builds eye diagram covering [-0.5, 1.5) UI range:
        - The eye center (0.5 UI) is at the center of the display
        - Left side shows transition from previous bit
        - Right side shows transition to next bit
        
        Args:
            time_array: Time array in seconds
            voltage_array: Voltage array in volts
            target_ber: Target BER for TJ calculation (default: 1e-12)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary containing eye metrics and jitter decomposition
        """
        # Validate inputs
        self._validate_input_arrays(time_array, voltage_array)
        
        # Get voltage range
        self._v_min = voltage_array.min()
        self._v_max = voltage_array.max()
        v_range = self._v_max - self._v_min
        v_margin = v_range * 0.1
        
        # Step 1: Calculate samples per UI and upsample if needed
        dt = time_array[1] - time_array[0]
        samples_per_ui = int(round(self.ui / dt))
        
        # Upsample if sampling rate is too low
        if samples_per_ui < 4:
            upsample_factor = int(np.ceil(4 / samples_per_ui))
            time_array, voltage_array = self._upsample_waveform(
                time_array, voltage_array, upsample_factor
            )
            dt = time_array[1] - time_array[0]
            samples_per_ui = int(round(self.ui / dt))
        
        # Step 2: MATLAB-style reshape - segment waveform by UI periods
        n_ui_periods = len(voltage_array) // samples_per_ui
        if n_ui_periods < 2:  # Need at least 2 UI for 2-UI display
            return self._empty_metrics(target_ber)
            
        trimmed_length = n_ui_periods * samples_per_ui
        wave_matrix = voltage_array[:trimmed_length].reshape(n_ui_periods, samples_per_ui)
        
        # Step 3: Build 2-UI centered eye matrix
        # The eye matrix covers [-0.5, 1.5) UI with eye center at 0.5 UI (display center)
        self.eye_matrix = self._build_eye_matrix_2ui_centered(wave_matrix, samples_per_ui)
        
        # Set up edges for 2-UI range [-0.5, 1.5)
        self._xedges = np.linspace(-0.5, 1.5, self.ui_bins + 1)
        self._yedges = np.linspace(self._v_min - v_margin, self._v_max + v_margin, self.amp_bins + 1)
        
        # Step 4: Compute eye metrics based on modulation
        if self.modulation.name == 'pam4':
            return self._compute_metrics_pam4(voltage_array, target_ber)
        else:
            return self._compute_metrics_nrz(voltage_array, target_ber)
    
    def _compute_metrics_nrz(self, voltage_array: np.ndarray, 
                             target_ber: float) -> Dict[str, Any]:
        """Compute eye metrics for NRZ modulation."""
        eye_height = self._compute_eye_height()
        eye_width = self._compute_eye_width()
        eye_area = self._compute_eye_area(eye_height, eye_width)
        
        # Jitter decomposition
        phase = (np.arange(len(voltage_array)) * self.ui / len(voltage_array)) % self.ui / self.ui
        try:
            jitter_metrics = self._jitter_decomposer.extract(
                phase, voltage_array, target_ber
            )
        except (ValueError, RuntimeError):
            jitter_metrics = self._default_jitter_metrics(target_ber)
        
        return {
            'eye_height': float(eye_height),
            'eye_width': float(eye_width),
            'eye_area': float(eye_area),
            'modulation': 'nrz',
            'scheme': 'golden_cdr',
            **jitter_metrics
        }
    
    def _compute_metrics_pam4(self, voltage_array: np.ndarray,
                              target_ber: float) -> Dict[str, Any]:
        """Compute eye metrics for PAM4 modulation (3 eyes)."""
        # Compute per-eye metrics
        eye_heights = self._compute_eye_heights_per_eye()
        eye_widths = self._compute_eye_widths_per_eye()
        
        # Aggregate metrics
        eye_height_min = min(eye_heights) if eye_heights else 0.0
        eye_height_avg = sum(eye_heights) / len(eye_heights) if eye_heights else 0.0
        eye_width_min = min(eye_widths) if eye_widths else 0.0
        eye_width_avg = sum(eye_widths) / len(eye_widths) if eye_widths else 0.0
        eye_area = eye_height_avg * eye_width_avg
        
        # Jitter decomposition (use center eye threshold)
        thresholds = self.modulation.get_thresholds()
        center_threshold = thresholds[1] if len(thresholds) > 1 else 0.0
        
        phase = (np.arange(len(voltage_array)) * self.ui / len(voltage_array)) % self.ui / self.ui
        try:
            jitter_metrics = self._jitter_decomposer.extract(
                phase, voltage_array, target_ber
            )
        except (ValueError, RuntimeError):
            jitter_metrics = self._default_jitter_metrics(target_ber)
        
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
            'scheme': 'golden_cdr',
            **jitter_metrics
        }
    
    def _compute_eye_heights_per_eye(self) -> list:
        """Compute eye height for each PAM4 eye."""
        if self.eye_matrix is None or self._yedges is None:
            return [0.0, 0.0, 0.0]
        
        thresholds = self.modulation.get_thresholds()
        eye_heights = []
        
        y_centers = (self._yedges[:-1] + self._yedges[1:]) / 2
        
        for threshold in thresholds:
            height = self._compute_eye_height_at_threshold(threshold, y_centers)
            eye_heights.append(height)
        
        return eye_heights
    
    def _compute_eye_widths_per_eye(self) -> list:
        """Compute eye width for each PAM4 eye at respective decision thresholds."""
        if self.eye_matrix is None or self._xedges is None:
            return [0.0, 0.0, 0.0]
        
        eye_widths = []
        thresholds = self.modulation.get_thresholds()
        y_centers = (self._yedges[:-1] + self._yedges[1:]) / 2
        
        for threshold in thresholds:
            # Find the eye region around this threshold
            # Look for low density region (eye opening) at this threshold level
            
            # Find y indices near the threshold
            threshold_idx = np.argmin(np.abs(y_centers - threshold))
            
            # Get phase profile at this threshold level
            phase_profile = self.eye_matrix[:, threshold_idx]
            
            if phase_profile.max() == 0:
                eye_widths.append(0.0)
                continue
            
            # Normalize
            phase_profile_norm = phase_profile / phase_profile.max()
            
            # Find low-density region (eye opening) using threshold
            density_threshold = 0.3  # 30% of max density
            
            # Find center (lowest density)
            center_idx = len(phase_profile) // 2
            
            # Search left from center for boundary
            left_idx = center_idx
            for i in range(center_idx, -1, -1):
                if phase_profile_norm[i] > density_threshold:
                    left_idx = i
                    break
            else:
                left_idx = 0
            
            # Search right from center for boundary  
            right_idx = center_idx
            for i in range(center_idx, len(phase_profile_norm)):
                if phase_profile_norm[i] > density_threshold:
                    right_idx = i
                    break
            else:
                right_idx = len(phase_profile_norm) - 1
            
            # Calculate width
            x_centers = (self._xedges[:-1] + self._xedges[1:]) / 2
            eye_width = x_centers[right_idx] - x_centers[left_idx]
            eye_widths.append(max(0.0, eye_width))
        
        return eye_widths
    
    def _compute_eye_height_at_threshold(self, threshold: float, 
                                         y_centers: np.ndarray) -> float:
        """Compute eye height at a specific decision threshold."""
        if self.eye_matrix is None:
            return 0.0
        
        # Find optimal phase (column with maximum density)
        phase_density = np.sum(self.eye_matrix, axis=1)
        optimal_phase_idx = np.argmax(phase_density)
        
        # Get amplitude profile at optimal phase
        amplitude_profile = self.eye_matrix[optimal_phase_idx, :]
        
        if np.sum(amplitude_profile) == 0:
            return 0.0
        
        # Find upper and lower eyes relative to threshold
        upper_mask = y_centers > threshold
        lower_mask = y_centers < threshold
        
        if not np.any(upper_mask) or not np.any(lower_mask):
            return 0.0
        
        # Compute weighted centroids for upper and lower eyes
        upper_centroid = np.average(y_centers[upper_mask], 
                                    weights=amplitude_profile[upper_mask])
        lower_centroid = np.average(y_centers[lower_mask],
                                    weights=amplitude_profile[lower_mask])
        
        return float(upper_centroid - lower_centroid)
    
    def _upsample_waveform(self, time_array: np.ndarray, 
                          voltage_array: np.ndarray,
                          factor: int) -> tuple[np.ndarray, np.ndarray]:
        """Upsample waveform using PCHIP interpolation."""
        dt = time_array[1] - time_array[0]
        dt_new = dt / factor
        time_new = np.arange(time_array[0], time_array[-1], dt_new)
        
        interpolator = PchipInterpolator(time_array, voltage_array)
        voltage_new = interpolator(time_new)
        
        return time_new, voltage_new
    
    def _build_eye_matrix_2ui_centered(self, wave_matrix: np.ndarray, 
                                       samples_per_ui: int) -> np.ndarray:
        """
        Build 2-UI centered eye matrix with proper edge continuity.
        
        Creates eye diagram covering [-0.5, 1.5) UI range:
        - [-0.5, 0): 2nd half of PREVIOUS UI [0.5, 1)
        - [0, 1): Complete current UI [0, 1)
        - [1, 1.5): 1st half of NEXT UI [0, 0.5)
        
        Key fix: Uses adjacent UI data for left/right edges to ensure 
        smooth, continuous eye diagram without gaps.
        """
        n_ui_periods = wave_matrix.shape[0]
        
        eye_matrix = np.zeros((self.ui_bins, self.amp_bins))
        
        # Original phase points (0 to 1, samples_per_ui points)
        phase_orig = np.linspace(0, 1, samples_per_ui, endpoint=False)
        
        # Target phase grid for 2-UI range [-0.5, 1.5)
        phase_target_2ui = np.linspace(-0.5, 1.5, self.ui_bins, endpoint=False)
        
        # Voltage edges for binning
        v_margin = (self._v_max - self._v_min) * 0.1
        yedges = np.linspace(self._v_min - v_margin, self._v_max + v_margin, self.amp_bins + 1)
        
        # Process each UI period (skip first and last for adjacent data availability)
        for i in range(1, n_ui_periods - 1):
            # Get waveforms for current, previous, and next UI
            wave_prev = wave_matrix[i-1, :]      # For [-0.5, 0)
            wave_curr = wave_matrix[i, :]        # For [0, 1)
            wave_next = wave_matrix[i+1, :]      # For [1, 1.5)
            
            # Create interpolators for each UI
            interp_prev = PchipInterpolator(phase_orig, wave_prev)
            interp_curr = PchipInterpolator(phase_orig, wave_curr)
            interp_next = PchipInterpolator(phase_orig, wave_next)
            
            # Accumulate to eye matrix
            for j, phase_2ui in enumerate(phase_target_2ui):
                # Select appropriate interpolator based on phase region
                if phase_2ui < 0:
                    # [-0.5, 0): Use PREVIOUS UI's [0.5, 1) region
                    phase_orig_val = phase_2ui + 1.0  # Map to [0.5, 1)
                    v = interp_prev(phase_orig_val)
                elif phase_2ui >= 1.0:
                    # [1, 1.5): Use NEXT UI's [0, 0.5) region
                    phase_orig_val = phase_2ui - 1.0  # Map to [0, 0.5)
                    v = interp_next(phase_orig_val)
                else:
                    # [0, 1): Use CURRENT UI
                    phase_orig_val = phase_2ui
                    v = interp_curr(phase_orig_val)
                
                # Accumulate to eye matrix
                v_idx = np.digitize(v, yedges) - 1
                v_idx = np.clip(v_idx, 0, self.amp_bins - 1)
                eye_matrix[j, v_idx] += 1
        
        return eye_matrix
    
    def _empty_metrics(self, target_ber: float) -> Dict[str, Any]:
        """Return empty metrics when no data."""
        return {
            'eye_height': 0.0,
            'eye_width': 0.0,
            'eye_area': 0.0,
            'scheme': 'golden_cdr',
            'rj_sigma': 0.0,
            'dj_pp': 0.0,
            'tj_at_ber': 0.0,
            'target_ber': target_ber,
            'q_factor': 0.0,
            'fit_method': 'failed',
            'fit_quality': 0.0,
            'pj_info': {'detected': False, 'frequencies': [], 
                       'amplitudes': [], 'count': 0}
        }
    
    def _default_jitter_metrics(self, target_ber: float) -> Dict[str, Any]:
        """Return default jitter metrics on failure."""
        return {
            'rj_sigma': 0.0,
            'dj_pp': 0.0,
            'tj_at_ber': 0.0,
            'target_ber': target_ber,
            'q_factor': 0.0,
            'fit_method': 'failed',
            'fit_quality': 0.0,
            'pj_info': {'detected': False, 'frequencies': [], 
                       'amplitudes': [], 'count': 0}
        }
    
    def get_xedges(self) -> np.ndarray:
        """
        Get X-axis bin edges (phase in [-0.5, 1.5]).
        
        Returns:
            Array of X-axis bin edges for 2-UI centered display
        """
        if self._xedges is None:
            return np.linspace(-0.5, 1.5, self.ui_bins + 1)
        return self._xedges
    
    def get_yedges(self) -> np.ndarray:
        """
        Get Y-axis bin edges (voltage).
        
        Returns:
            Array of Y-axis bin edges
        """
        if self._yedges is None:
            return np.linspace(-1, 1, self.amp_bins + 1)
        return self._yedges
