"""Pulse response processor for statistical eye diagram analysis.

This module provides the PulseResponseProcessor class for processing
channel pulse responses, supporting both NRZ and PAM4 modulation formats.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal

from eye_analyzer.modulation import ModulationFormat, NRZ


class PulseResponseProcessor:
    """Process channel pulse response for statistical eye analysis.
    
    Features:
    - DC offset removal
    - Window extraction
    - Differential signal conversion
    - Upsampling (default 16x)
    - Main cursor detection
    - Voltage range estimation
    
    Supports both NRZ and PAM4 modulation formats.
    """
    
    def __init__(self):
        """Initialize pulse response processor."""
        self._main_cursor_idx: Optional[int] = None
        self._processed_response: Optional[np.ndarray] = None
    
    def process(
        self,
        pulse: np.ndarray,
        dt: float = 1.0,
        upsampling: int = 16,
        diff_signal: bool = False,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """Process pulse response.
        
        Args:
            pulse: Raw pulse response array
            dt: Time step (seconds)
            upsampling: Upsampling factor (default 16)
            diff_signal: Apply differential signaling factor (divide by 2)
            window_size: Window size in samples after upsampling
            
        Returns:
            Processed pulse response array
            
        Raises:
            ValueError: If pulse is all zeros or invalid
        """
        pulse = np.asarray(pulse)
        
        if np.all(pulse == 0):
            raise ValueError("Pulse response cannot be all zeros")
        
        if len(pulse) == 0:
            raise ValueError("Pulse response cannot be empty")
        
        # Step 1: Remove DC offset (first sample should be baseline)
        dc_offset = pulse[0]
        pulse_dc_removed = pulse - dc_offset
        
        # Step 2: Apply differential signal factor if requested
        if diff_signal:
            pulse_dc_removed = pulse_dc_removed * 0.5
        
        # Step 3: Upsample
        if upsampling > 1:
            # Use scipy.signal.resample for proper upsampling
            new_length = len(pulse_dc_removed) * upsampling
            pulse_upsampled = signal.resample(pulse_dc_removed, new_length)
        else:
            pulse_upsampled = pulse_dc_removed
            upsampling = 1
        
        # Step 4: Extract window around main cursor
        if window_size is not None:
            # Find main cursor in upsampled domain
            main_idx = self.find_main_cursor(pulse_upsampled)
            half_window = window_size // 2
            start = max(0, main_idx - half_window)
            end = min(len(pulse_upsampled), start + window_size)
            start = max(0, end - window_size)  # Adjust start if end is clipped
            result = pulse_upsampled[start:end]
        else:
            result = pulse_upsampled
        
        self._processed_response = result
        return result
    
    def find_main_cursor(self, pulse: np.ndarray) -> int:
        """Find main cursor index (maximum absolute amplitude).
        
        Args:
            pulse: Pulse response array
            
        Returns:
            Index of main cursor
        """
        pulse = np.asarray(pulse)
        self._main_cursor_idx = int(np.argmax(np.abs(pulse)))
        return self._main_cursor_idx
    
    def estimate_voltage_range(
        self,
        pulse: np.ndarray,
        modulation: ModulationFormat,
        multiplier: float = 2.0
    ) -> Tuple[float, float]:
        """Estimate voltage range for eye diagram display.
        
        Args:
            pulse: Processed pulse response
            modulation: Modulation format (NRZ, PAM4, etc.)
            multiplier: Display range multiplier (default 2.0)
            
        Returns:
            Tuple of (v_min, v_max)
        """
        pulse = np.asarray(pulse)
        
        # Find maximum amplitude
        max_amp = np.max(np.abs(pulse))
        
        # Get modulation levels
        levels = modulation.get_levels()
        max_level = np.max(np.abs(levels))
        
        # Calculate display range
        # For PAM4: levels are [-3, -1, 1, 3], max_level = 3
        # Display should show full signal range considering modulation levels
        display_range = max_amp * max_level * multiplier / 3.0
        
        # For PAM4: symmetric around 0
        # For NRZ: symmetric around 0
        v_min = -display_range
        v_max = display_range
        
        return v_min, v_max
    
    def get_main_cursor_idx(self) -> Optional[int]:
        """Get cached main cursor index.
        
        Returns:
            Main cursor index or None if not processed
        """
        return self._main_cursor_idx
    
    def get_processed_response(self) -> Optional[np.ndarray]:
        """Get cached processed response.
        
        Returns:
            Processed response or None if not processed
        """
        return self._processed_response
