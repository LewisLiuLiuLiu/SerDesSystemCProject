"""Pulse response preprocessing for statistical eye analysis."""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional
from ..modulation import ModulationFormat


class PulseResponseProcessor:
    """Process channel pulse response for statistical eye analysis.
    
    This class handles preprocessing of raw pulse response data including:
    - DC offset removal
    - Window extraction (non-zero region)
    - Differential signaling compensation
    - Upsampling for better visualization
    
    Example:
        >>> processor = PulseResponseProcessor()
        >>> pulse = processor.process(
        ...     raw_pulse,
        ...     modulation=PAM4(),
        ...     diff_signal=True,
        ...     upsampling=16
        ... )
    """
    
    def process(self,
                pulse_response: np.ndarray,
                modulation: Optional[ModulationFormat] = None,
                diff_signal: bool = True,
                upsampling: int = 16,
                interpolation_type: str = 'linear') -> np.ndarray:
        """Process pulse response.
        
        Args:
            pulse_response: Raw pulse response array from channel simulation
            modulation: Modulation format (optional, for validation)
            diff_signal: If True, multiply by 0.5 for differential signaling
            upsampling: Upsampling factor for better visualization
            interpolation_type: 'linear' or 'cubic' interpolation
            
        Returns:
            Processed pulse response array
            
        Raises:
            ValueError: If pulse response is all zeros or invalid
        """
        # Remove DC offset
        pulse = np.array(pulse_response)
        pulse = pulse - pulse[0]
        
        # Extract non-zero window
        window = np.where(pulse != 0)[0]
        if len(window) == 0:
            raise ValueError("Pulse response is all zeros")
        
        window_start = max(0, window[0] - 1)
        window_end = min(len(pulse), window[-1] + 2)
        pulse = pulse[window_start:window_end]
        
        # Apply differential signaling factor
        if diff_signal:
            pulse = pulse * 0.5
        
        # Upsample
        if upsampling > 1:
            x = np.linspace(0, len(pulse) - 1, len(pulse))
            f = interp1d(x, pulse, kind=interpolation_type)
            x_new = np.linspace(0, len(pulse) - 1, len(pulse) * upsampling)
            pulse = f(x_new)
        
        return pulse
    
    def find_main_cursor(self, pulse: np.ndarray) -> int:
        """Find index of main cursor (peak amplitude).
        
        Args:
            pulse: Processed pulse response
            
        Returns:
            Index of main cursor (peak absolute amplitude)
        """
        return int(np.argmax(np.abs(pulse)))
    
    def estimate_voltage_range(self, pulse: np.ndarray, 
                               modulation: ModulationFormat,
                               multiplier: float = 2.0) -> tuple:
        """Estimate voltage range for eye diagram display.
        
        Args:
            pulse: Processed pulse response
            modulation: Modulation format
            multiplier: Window size multiplier
            
        Returns:
            Tuple of (v_min, v_max) for display range
        """
        idx_main = self.find_main_cursor(pulse)
        A_max = np.abs(pulse[idx_main])
        
        # For PAM4, main cursor amplitude corresponds to outer levels
        # Need to account for all modulation levels
        if modulation:
            levels = modulation.get_levels()
            max_level = np.max(np.abs(levels))
            # Scale by max level (3 for PAM4, 1 for NRZ)
            A_display = A_max * max_level * multiplier / 3.0
        else:
            A_display = A_max * multiplier
        
        return -A_display, A_display
