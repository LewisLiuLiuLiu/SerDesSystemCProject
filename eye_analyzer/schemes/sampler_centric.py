"""
Sampler-Centric Eye Diagram Analysis Scheme - Fixed Version

This module implements the streaming/sampler-centric eye diagram analysis
using uniform resampling within each UI window to avoid phase discretization.

Key characteristics:
- Time reference: CDR sampling moment t_sampler[k] = 0 UI
- Observation window: 2 UI ([-1, +1] relative to sampling moment)
- Core algorithm: Per-sample window extraction + uniform resampling to fixed grid
- Jitter analysis: Eye diagram horizontal spread directly reflects sampling jitter
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

from .base import BaseScheme
from ..interpolation import get_valid_sampler_indices
from ..modulation import ModulationFormat


class SamplerCentricScheme(BaseScheme):
    """
    Sampler-Centric Eye Diagram Analysis Scheme (Fixed)
    
    This scheme constructs eye diagrams using CDR sampling timestamps as the
    absolute time reference. Each sampling moment defines the center (0 UI)
    of a 2-UI observation window.
    
    Fixed implementation uses uniform resampling within each window to ensure
    proper eye diagram construction regardless of original sampling rate.
    
    Attributes:
        ui: Unit interval in seconds
        ui_bins: Number of bins for time axis (default: 128 for 2 UI window)
        amp_bins: Number of bins for amplitude axis (default: 256)
        interp_method: Interpolation method ('cubic', 'linear', 'pchip')
        
    Example:
        >>> scheme = SamplerCentricScheme(ui=2.5e-11, ui_bins=128, amp_bins=256)
        >>> metrics = scheme.analyze(time_array, voltage_array, sampler_timestamps)
        >>> print(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
    """
    
    def __init__(self, 
                 ui: float, 
                 modulation: Union[str, ModulationFormat] = 'nrz',
                 ui_bins: int = 128, 
                 amp_bins: int = 256,
                 interp_method: str = 'cubic'):
        """
        Initialize the Sampler-Centric scheme.
        
        Args:
            ui: Unit interval in seconds
            modulation: Modulation format ('nrz', 'pam4') or ModulationFormat object
            ui_bins: Number of bins for time axis (default: 128)
                    These bins cover the 2-UI window [-1, +1]
            amp_bins: Number of bins for amplitude axis (default: 256)
            interp_method: Interpolation method ('cubic', 'linear', 'pchip')
                          Default: 'cubic' (CubicSpline)
        """
        super().__init__(ui, modulation, ui_bins, amp_bins)
        self.interp_method = interp_method
        self._num_valid_samples = 0
        self._num_skipped_samples = 0
        
    def analyze(self, time_array: np.ndarray, voltage_array: np.ndarray,
                sampler_timestamps: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform sampler-centric eye diagram analysis (fixed version).
        
        Args:
            time_array: Time array in seconds
            voltage_array: Voltage array in volts
            sampler_timestamps: CDR sampling timestamps in seconds
                               If None, will raise ValueError
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary containing:
            - eye_height: Eye height in volts
            - eye_width: Eye width in UI (relative to 2-UI window)
            - eye_area: Eye area in V*UI
            - scheme: 'sampler_centric'
            - num_valid_samples: Number of valid samples processed
            - num_skipped_samples: Number of boundary samples skipped
            - total_samples: Total number of sampling timestamps
            
        Raises:
            ValueError: If sampler_timestamps is None or invalid
        """
        # Validate inputs
        self._validate_input_arrays(time_array, voltage_array)
        
        if sampler_timestamps is None:
            raise ValueError(
                "sampler_timestamps is required for Sampler-Centric scheme. "
                "Please provide CDR sampling timestamps."
            )
        
        if not isinstance(sampler_timestamps, np.ndarray):
            sampler_timestamps = np.array(sampler_timestamps)
            
        if len(sampler_timestamps) < 1:
            raise ValueError("sampler_timestamps must have at least 1 element")
        
        # Get waveform time range
        t_start = time_array[0]
        t_end = time_array[-1]
        
        # Get voltage range for histogram
        self._v_min = voltage_array.min()
        self._v_max = voltage_array.max()
        v_range = self._v_max - self._v_min
        v_margin = v_range * 0.1
        
        # Initialize eye matrix and edges
        self.eye_matrix = np.zeros((self.ui_bins, self.amp_bins))
        self._xedges = np.linspace(-1, 1, self.ui_bins + 1)
        self._yedges = np.linspace(self._v_min - v_margin, 
                                   self._v_max + v_margin, 
                                   self.amp_bins + 1)
        
        # Get valid sample indices (skip boundary samples)
        valid_indices = get_valid_sampler_indices(
            sampler_timestamps, t_start, t_end, self.ui
        )
        
        self._num_valid_samples = len(valid_indices)
        self._num_skipped_samples = len(sampler_timestamps) - self._num_valid_samples
        
        if self._num_valid_samples == 0:
            return self._empty_metrics(sampler_timestamps)
        
        # Key fix: Create uniform phase grid for resampling
        # This ensures each window contributes to all phase bins uniformly
        uniform_phase_grid = np.linspace(-1, 1, self.ui_bins)
        
        # Process each valid sampling moment
        for idx in valid_indices:
            t_sample = sampler_timestamps[idx]
            
            # Step 1: Extract 2-UI window
            window_time, window_voltage = self._extract_window(
                time_array, voltage_array, t_sample
            )
            
            if len(window_time) < 4:  # Need at least 4 points for cubic
                continue
            
            # Step 2: Normalize time to [-1, +1]
            t_normalized = (window_time - t_sample) / self.ui
            
            # Step 3: Resample to uniform grid (KEY FIX)
            resampled_voltage = self._resample_to_uniform_grid(
                t_normalized, window_voltage, uniform_phase_grid
            )
            
            if resampled_voltage is None:
                continue
            
            # Step 4: Accumulate to eye matrix
            self._accumulate_to_eye_matrix(resampled_voltage)
        
        # Compute metrics based on modulation
        if self.modulation.name == 'pam4':
            return self._compute_metrics_pam4(sampler_timestamps)
        else:
            return self._compute_metrics_nrz(sampler_timestamps)
    
    def _extract_window(self, time_array: np.ndarray,
                       voltage_array: np.ndarray,
                       t_center: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract 2-UI window centered at t_center.
        Window covers [t_center - UI, t_center + UI].
        """
        t_start = t_center - self.ui
        t_end = t_center + self.ui
        
        # Find indices within window
        start_idx = np.searchsorted(time_array, t_start)
        end_idx = np.searchsorted(time_array, t_end, side='right')
        
        if end_idx - start_idx < 2:
            return np.array([]), np.array([])
        
        return time_array[start_idx:end_idx], voltage_array[start_idx:end_idx]
    
    def _resample_to_uniform_grid(self,
                                  t_normalized: np.ndarray,
                                  voltage: np.ndarray,
                                  target_phase_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Resample waveform to uniform phase grid with smooth boundary handling.
        
        Uses PCHIP interpolation with linear extrapolation for boundaries
        to ensure smooth eye diagram edges without discontinuities.
        """
        # Remove duplicates if any
        unique_indices = np.unique(t_normalized, return_index=True)[1]
        t_unique = t_normalized[unique_indices]
        v_unique = voltage[unique_indices]
        
        # Sort by time
        sort_idx = np.argsort(t_unique)
        t_sorted = t_unique[sort_idx]
        v_sorted = v_unique[sort_idx]
        
        if len(t_sorted) < 4:
            return None
        
        # Interpolate to target grid with extrapolation
        try:
            # Use PCHIP with extrapolation for smooth boundaries
            interpolator = PchipInterpolator(t_sorted, v_sorted, extrapolate=True)
            
            # Interpolate all points (including extrapolation outside data range)
            result = interpolator(target_phase_grid)
            
            return result
            
        except Exception:
            return None
    
    def _accumulate_to_eye_matrix(self, resampled_voltage: np.ndarray) -> None:
        """
        Accumulate resampled trace to eye matrix.
        
        Now resampled_voltage has exactly ui_bins points, one for each phase bin.
        """
        for i, voltage in enumerate(resampled_voltage):
            if np.isnan(voltage):
                continue
            
            v_idx = np.digitize(voltage, self._yedges) - 1
            v_idx = np.clip(v_idx, 0, self.amp_bins - 1)
            self.eye_matrix[i, v_idx] += 1
    
    def _empty_metrics(self, sampler_timestamps: np.ndarray) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            'eye_height': 0.0,
            'eye_width': 0.0,
            'eye_area': 0.0,
            'scheme': 'sampler_centric',
            'num_valid_samples': 0,
            'num_skipped_samples': len(sampler_timestamps),
            'total_samples': len(sampler_timestamps)
        }
    
    def _compute_metrics_nrz(self, sampler_timestamps: np.ndarray) -> Dict[str, Any]:
        """Compute eye metrics for NRZ modulation."""
        eye_height = self._compute_eye_height()
        eye_width = self._compute_eye_width()
        eye_area = self._compute_eye_area(eye_height, eye_width)
        
        return {
            'eye_height': float(eye_height),
            'eye_width': float(eye_width),
            'eye_area': float(eye_area),
            'modulation': 'nrz',
            'scheme': 'sampler_centric',
            'num_valid_samples': self._num_valid_samples,
            'num_skipped_samples': self._num_skipped_samples,
            'total_samples': len(sampler_timestamps)
        }
    
    def _compute_metrics_pam4(self, sampler_timestamps: np.ndarray) -> Dict[str, Any]:
        """Compute eye metrics for PAM4 modulation (3 eyes)."""
        # Compute per-eye metrics using thresholds
        eye_heights = self._compute_eye_heights_per_eye()
        eye_widths = self._compute_eye_widths_per_eye()
        
        # Aggregate metrics
        eye_height_min = min(eye_heights) if eye_heights else 0.0
        eye_height_avg = sum(eye_heights) / len(eye_heights) if eye_heights else 0.0
        eye_width_min = min(eye_widths) if eye_widths else 0.0
        eye_width_avg = sum(eye_widths) / len(eye_widths) if eye_widths else 0.0
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
            'scheme': 'sampler_centric',
            'num_valid_samples': self._num_valid_samples,
            'num_skipped_samples': self._num_skipped_samples,
            'total_samples': len(sampler_timestamps)
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
        """Compute eye width for each PAM4 eye."""
        if self.eye_matrix is None or self._xedges is None:
            return [0.0, 0.0, 0.0]
        
        eye_widths = []
        thresholds = self.modulation.get_thresholds()
        y_centers = (self._yedges[:-1] + self._yedges[1:]) / 2
        
        for threshold in thresholds:
            # Find the eye region around this threshold
            threshold_idx = np.argmin(np.abs(y_centers - threshold))
            
            # Get phase profile at this threshold level
            phase_profile = self.eye_matrix[:, threshold_idx]
            
            if phase_profile.max() == 0:
                eye_widths.append(0.0)
                continue
            
            # Normalize
            phase_profile_norm = phase_profile / phase_profile.max()
            
            # Find low-density region (eye opening)
            density_threshold = 0.3
            center_idx = len(phase_profile) // 2
            
            # Search left
            left_idx = center_idx
            for i in range(center_idx, -1, -1):
                if phase_profile_norm[i] > density_threshold:
                    left_idx = i
                    break
            else:
                left_idx = 0
            
            # Search right
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
        
        # Find optimal phase
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
        
        # Compute weighted centroids
        try:
            upper_centroid = np.average(y_centers[upper_mask],
                                        weights=amplitude_profile[upper_mask])
            lower_centroid = np.average(y_centers[lower_mask],
                                        weights=amplitude_profile[lower_mask])
            return float(upper_centroid - lower_centroid)
        except ZeroDivisionError:
            return 0.0
    
    def get_xedges(self) -> np.ndarray:
        """
        Get X-axis bin edges (normalized time in [-1, +1]).
        
        Returns:
            Array of X-axis bin edges
        """
        if self._xedges is None:
            return np.linspace(-1, 1, self.ui_bins + 1)
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
