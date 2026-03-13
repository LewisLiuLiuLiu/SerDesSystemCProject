"""
Base Scheme for Eye Diagram Analysis

This module defines the abstract base class for eye diagram analysis schemes.
Both SamplerCentricScheme and GoldenCdrScheme inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from ..modulation import ModulationFormat, create_modulation


class BaseScheme(ABC):
    """
    Abstract base class for eye diagram analysis schemes.
    
    This class defines the common interface and shared functionality for
    eye diagram analysis. Subclasses must implement the analyze() method
    and provide scheme-specific logic.
    
    Attributes:
        ui: Unit interval in seconds
        modulation: Modulation format (NRZ, PAM4, etc.)
        ui_bins: Number of bins for time/phase axis
        amp_bins: Number of bins for amplitude axis
        eye_matrix: 2D histogram matrix for eye diagram
        _xedges: X-axis bin edges (time or phase)
        _yedges: Y-axis bin edges (voltage)
    """
    
    def __init__(self, 
                 ui: float, 
                 modulation: Union[str, ModulationFormat] = 'nrz',
                 ui_bins: int = 128, 
                 amp_bins: int = 256):
        """
        Initialize the base scheme.
        
        Args:
            ui: Unit interval in seconds (e.g., 2.5e-11 for 40 Gbps)
            modulation: Modulation format, either string ('nrz', 'pam4') or 
                       ModulationFormat object. Defaults to 'nrz'.
            ui_bins: Number of bins for time/phase axis (default: 128)
            amp_bins: Number of bins for amplitude axis (default: 256)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if ui <= 0:
            raise ValueError(f"UI must be positive, got {ui}")
        if ui_bins < 2:
            raise ValueError(f"ui_bins must be at least 2, got {ui_bins}")
        if amp_bins < 2:
            raise ValueError(f"amp_bins must be at least 2, got {amp_bins}")
            
        self.ui = ui
        self.ui_bins = ui_bins
        self.amp_bins = amp_bins
        
        # Support both string and ModulationFormat object
        if isinstance(modulation, str):
            self.modulation = create_modulation(modulation)
        else:
            self.modulation = modulation
            
        self.eye_matrix: Optional[np.ndarray] = None
        self._xedges: Optional[np.ndarray] = None
        self._yedges: Optional[np.ndarray] = None
        self._v_min: float = 0.0
        self._v_max: float = 0.0
        
    @abstractmethod
    def analyze(self, time_array: np.ndarray, voltage_array: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        Perform eye diagram analysis.
        
        This method must be implemented by subclasses to provide
        scheme-specific analysis logic.
        
        Args:
            time_array: Time array in seconds (64x oversampled)
            voltage_array: Voltage array in volts
            **kwargs: Additional scheme-specific arguments
            
        Returns:
            Dictionary containing analysis metrics:
            - eye_height: Eye height in volts
            - eye_width: Eye width in UI
            - eye_area: Eye area in V*UI
            - scheme: Scheme name string
            - Additional scheme-specific metrics
        """
        pass
    
    @abstractmethod
    def get_xedges(self) -> np.ndarray:
        """
        Get X-axis bin edges.
        
        Returns:
            Array of X-axis bin edges (normalized time or phase)
        """
        pass
    
    @abstractmethod
    def get_yedges(self) -> np.ndarray:
        """
        Get Y-axis bin edges.
        
        Returns:
            Array of Y-axis bin edges (voltage)
        """
        pass
    
    def get_eye_matrix(self) -> Optional[np.ndarray]:
        """
        Get the eye diagram 2D matrix.
        
        Returns:
            2D numpy array of shape (ui_bins, amp_bins) or None if not analyzed
        """
        return self.eye_matrix
    
    def _compute_eye_height(self) -> float:
        """
        Compute eye height from the eye diagram matrix.
        
        Eye height is defined as the vertical opening at the center
        of the eye diagram (optimal sampling phase).
        
        Returns:
            Eye height in volts
        """
        if self.eye_matrix is None or self._yedges is None:
            return 0.0
            
        # Find center column (optimal sampling position)
        center_idx = self.ui_bins // 2
        center_col = self.eye_matrix[center_idx, :]
        
        # Normalize column to probability
        total = center_col.sum()
        if total == 0:
            return 0.0
        prob = center_col / total
        
        # Find upper and lower rails
        y_centers = (self._yedges[:-1] + self._yedges[1:]) / 2
        
        # Find high and low levels using weighted average
        mid_voltage = (self._v_max + self._v_min) / 2
        
        high_mask = y_centers > mid_voltage
        low_mask = y_centers < mid_voltage
        
        if not np.any(high_mask) or not np.any(low_mask):
            return 0.0
        
        high_prob = prob[high_mask]
        low_prob = prob[low_mask]
        
        if high_prob.sum() == 0 or low_prob.sum() == 0:
            return 0.0
        
        # Find eye boundaries using cumulative distribution
        # Upper boundary: 99th percentile of low distribution
        # Lower boundary: 1st percentile of high distribution
        high_voltages = y_centers[high_mask]
        low_voltages = y_centers[low_mask]
        
        # Sort by voltage
        high_sorted_idx = np.argsort(high_voltages)
        low_sorted_idx = np.argsort(low_voltages)[::-1]  # Descending
        
        high_prob_sorted = high_prob[high_sorted_idx]
        low_prob_sorted = low_prob[low_sorted_idx]
        
        high_voltages_sorted = high_voltages[high_sorted_idx]
        low_voltages_sorted = low_voltages[low_sorted_idx]
        
        # Find 1st percentile boundary
        high_cumsum = np.cumsum(high_prob_sorted) / high_prob_sorted.sum()
        low_cumsum = np.cumsum(low_prob_sorted) / low_prob_sorted.sum()
        
        high_boundary_idx = np.searchsorted(high_cumsum, 0.01)
        low_boundary_idx = np.searchsorted(low_cumsum, 0.01)
        
        if high_boundary_idx >= len(high_voltages_sorted):
            high_boundary_idx = len(high_voltages_sorted) - 1
        if low_boundary_idx >= len(low_voltages_sorted):
            low_boundary_idx = len(low_voltages_sorted) - 1
            
        eye_top = high_voltages_sorted[high_boundary_idx]
        eye_bottom = low_voltages_sorted[low_boundary_idx]
        
        eye_height = eye_top - eye_bottom
        return max(0.0, eye_height)
    
    def _compute_eye_width(self) -> float:
        """
        Compute eye width from the eye diagram matrix.
        
        Eye width is defined as the horizontal opening at the
        optimal decision threshold.
        
        Returns:
            Eye width in UI (0-1 for Golden CDR, 0-2 for Sampler-Centric)
        """
        if self.eye_matrix is None or self._xedges is None:
            return 0.0
            
        # Sum across voltage axis to get phase distribution
        phase_profile = self.eye_matrix.sum(axis=1)
        
        # Find the eye opening region (low density area)
        if phase_profile.max() == 0:
            return 0.0
            
        # Normalize
        phase_profile = phase_profile / phase_profile.max()
        
        # Find threshold (e.g., 10% of max)
        threshold = 0.1
        
        # Find consecutive low-density region around center
        center_idx = self.ui_bins // 2
        
        # Search left from center
        left_idx = center_idx
        for i in range(center_idx, -1, -1):
            if phase_profile[i] > threshold:
                left_idx = i + 1
                break
        else:
            left_idx = 0
            
        # Search right from center
        right_idx = center_idx
        for i in range(center_idx, self.ui_bins):
            if phase_profile[i] > threshold:
                right_idx = i - 1
                break
        else:
            right_idx = self.ui_bins - 1
        
        # Calculate width in UI
        x_centers = (self._xedges[:-1] + self._xedges[1:]) / 2
        eye_width = x_centers[right_idx] - x_centers[left_idx]
        
        return max(0.0, eye_width)
    
    def _compute_eye_area(self, eye_height: float, eye_width: float) -> float:
        """
        Compute eye area from eye height and width.
        
        Args:
            eye_height: Eye height in volts
            eye_width: Eye width in UI
            
        Returns:
            Eye area in V*UI
        """
        return eye_height * eye_width
    
    def _validate_input_arrays(self, time_array: np.ndarray, 
                               voltage_array: np.ndarray) -> None:
        """
        Validate input arrays.
        
        Args:
            time_array: Time array
            voltage_array: Voltage array
            
        Raises:
            ValueError: If arrays are invalid
        """
        if not isinstance(time_array, np.ndarray):
            raise ValueError("time_array must be a numpy array")
        if not isinstance(voltage_array, np.ndarray):
            raise ValueError("voltage_array must be a numpy array")
        if len(time_array) != len(voltage_array):
            raise ValueError(
                f"Array length mismatch: time={len(time_array)}, "
                f"voltage={len(voltage_array)}"
            )
        if len(time_array) < 2:
            raise ValueError("Arrays must have at least 2 elements")
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("time_array must be strictly increasing")
