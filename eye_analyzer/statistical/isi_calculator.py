"""
ISI (Inter-Symbol Interference) Calculator.

This module provides the ISICalculator class for computing ISI probability
density functions (PDFs) from pulse responses. It supports two methods:

1. Convolution method (_calculate_by_convolution): Fast approximation using
   PDF convolution. Complexity is O(N*L) where N is sample size and L is
   the number of voltage bins.

2. Brute force method (_calculate_by_brute_force): Exact calculation by
   enumerating all symbol combinations. Complexity is O(M^N) where M is
   the modulation level and N is sample size.

Supports NRZ and PAM4 modulation formats.

Reference:
- OIF-CEI-04.0 specification for statistical eye methodology
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import interp1d


class ModulationFormat(Enum):
    """Modulation format enumeration."""
    NRZ = 2  # 2-level: -1, 1
    PAM4 = 4  # 4-level: -1, -1/3, 1/3, 1


class ISICalculator:
    """
    Calculator for ISI (Inter-Symbol Interference) probability distributions.
    
    This class computes the ISI PDF at each time slice within a symbol period
    using either convolution-based or brute-force enumeration methods.
    
    Attributes:
        modulation_format: Modulation format (NRZ or PAM4)
        samples_per_symbol: Number of samples per UI (unit interval)
        vh_size: Number of vertical histogram bins (must be even)
        sample_size: Maximum number of symbols to sample from pulse response
        upsampling: Upsampling factor for improved visualization
        interpolation_type: Type of interpolation ('linear' or 'cubic')
        A_window_multiplier: Multiplier for voltage window around main cursor
    """
    
    def __init__(
        self,
        modulation_format: ModulationFormat = ModulationFormat.PAM4,
        samples_per_symbol: int = 8,
        vh_size: int = 2048,
        sample_size: int = 16,
        upsampling: int = 1,
        interpolation_type: str = 'linear',
        A_window_multiplier: float = 2.0
    ):
        """
        Initialize ISI Calculator.
        
        Args:
            modulation_format: Modulation format (NRZ or PAM4)
            samples_per_symbol: Samples per symbol/UI
            vh_size: Number of voltage histogram bins (should be even)
            sample_size: Number of symbols to sample from pulse tail
            upsampling: Upsampling factor for time domain interpolation
            interpolation_type: Interpolation type ('linear' or 'cubic')
            A_window_multiplier: Voltage window multiplier relative to main cursor
        """
        self.modulation_format = modulation_format
        self.samples_per_symbol = samples_per_symbol
        self.vh_size = vh_size
        self.sample_size = sample_size
        self.upsampling = upsampling
        self.interpolation_type = interpolation_type
        self.A_window_multiplier = A_window_multiplier
        
        # Validate vh_size is even
        if self.vh_size % 2 != 0:
            self.vh_size += 1
    
    def get_modulation_levels(self) -> np.ndarray:
        """
        Get modulation level values.
        
        Returns:
            Array of modulation levels:
            - NRZ: [-1, 1]
            - PAM4: [-1, -1/3, 1/3, 1]
        """
        if self.modulation_format == ModulationFormat.NRZ:
            return np.array([-1.0, 1.0])
        elif self.modulation_format == ModulationFormat.PAM4:
            return np.array([-1.0, -1.0/3.0, 1.0/3.0, 1.0])
        else:
            raise ValueError(f"Unknown modulation format: {self.modulation_format}")
    
    def _prepare_pulse_response(self, pulse_response: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Prepare pulse response for ISI calculation.
        
        Args:
            pulse_response: Raw pulse response array
            
        Returns:
            Tuple of (processed pulse, main cursor index)
            
        Raises:
            ValueError: If pulse response is empty or invalid
        """
        if len(pulse_response) == 0:
            raise ValueError("Pulse response cannot be empty")
        
        pulse_response = np.array(pulse_response, dtype=float)
        
        # Remove DC offset
        pulse_response = pulse_response - pulse_response[0]
        
        # Extract non-zero window
        nonzero_indices = np.nonzero(pulse_response)[0]
        if len(nonzero_indices) == 0:
            # If all zeros, use full array
            pulse_input = pulse_response
        else:
            window_start = max(0, nonzero_indices[0] - 1)
            window_end = min(len(pulse_response), nonzero_indices[-1] + 2)
            pulse_input = pulse_response[window_start:window_end]
        
        # Apply differential signaling (half amplitude)
        pulse_input = pulse_input * 0.5
        
        # Upsample if requested
        if self.upsampling > 1:
            x = np.linspace(0, len(pulse_input) - 1, num=len(pulse_input))
            f = interp1d(x, pulse_input, kind=self.interpolation_type)
            x_new = np.linspace(0, len(pulse_input) - 1, 
                               num=len(pulse_input) * self.upsampling)
            pulse_input = f(x_new)
        
        # Find main cursor (peak)
        idx_main = np.argmax(np.abs(pulse_input))
        
        return pulse_input, idx_main
    
    def _create_voltage_bins(self, pulse_main_amplitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create voltage histogram bins.
        
        Args:
            pulse_main_amplitude: Amplitude of main cursor
            
        Returns:
            Tuple of (bin_edges, bin_centers)
        """
        A_window_min = abs(pulse_main_amplitude) * -self.A_window_multiplier
        A_window_max = abs(pulse_main_amplitude) * self.A_window_multiplier
        
        # Create symmetric bins around zero
        half_size = self.vh_size // 2
        
        # Upper half (excluding zero)
        bin_edges_up = np.linspace(0, A_window_max, half_size + 1)[1:]
        
        # Lower half (including zero)
        bin_edges_down = np.linspace(A_window_min, 0, half_size + 1)
        
        # Combine
        bin_edges = np.concatenate((bin_edges_down, bin_edges_up))
        
        # Calculate bin centers
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        return bin_edges, bin_centers
    
    def _sample_pulse_points(
        self,
        pulse_input: np.ndarray,
        idx_main: int,
        idx_offset: int
    ) -> np.ndarray:
        """
        Sample points from pulse response for ISI calculation.
        
        Samples points at symbol-spaced intervals around the main cursor,
        offset by idx_offset from the center.
        
        Args:
            pulse_input: Processed pulse response
            idx_main: Index of main cursor
            idx_offset: Offset from center for this time slice
            
        Returns:
            Array of sampled amplitudes with shape (num_samples, num_levels)
        """
        samples_per_symbol = self.samples_per_symbol * self.upsampling
        idx_sampled_center = idx_main + idx_offset
        
        sampled_points = []
        
        # Sample pre-cursors (including main cursor at i=0)
        i = 0
        while idx_sampled_center - i * samples_per_symbol >= 0:
            idx_sample = idx_sampled_center - i * samples_per_symbol
            # Ensure index is within bounds
            if idx_sample < len(pulse_input):
                sampled_points.append(idx_sample)
            i += 1
        
        # Sample post-cursors (excluding main cursor)
        j = 1
        while idx_sampled_center + j * samples_per_symbol < len(pulse_input):
            idx_sample = idx_sampled_center + j * samples_per_symbol
            # Ensure index is within bounds
            if idx_sample < len(pulse_input):
                sampled_points.append(idx_sample)
            j += 1
        
        # Limit sample size
        sampled_points = sampled_points[:self.sample_size]
        
        # Handle empty or invalid case
        if len(sampled_points) == 0:
            # Return empty array with correct shape
            return np.array([]).reshape(0, len(self.get_modulation_levels()))
        
        # Get amplitudes and create level matrix
        sampled_amps = np.array([pulse_input[i] for i in sampled_points]).reshape(-1, 1)
        levels = self.get_modulation_levels().reshape(1, -1)
        
        # Multiply amplitudes by modulation levels
        # Result shape: (num_samples, num_levels)
        return sampled_amps @ levels
    
    def _calculate_by_convolution(
        self,
        pulse_response: np.ndarray
    ) -> List[np.ndarray]:
        """
        Calculate ISI PDF using convolution method.
        
        This is a fast approximation that convolves individual cursor PDFs.
        Complexity is O(N * L^2) where N is sample size and L is vh_size.
        
        Args:
            pulse_response: Pulse response array
            
        Returns:
            List of PDF arrays, one per time slice within a symbol period
        """
        pulse_input, idx_main = self._prepare_pulse_response(pulse_response)
        bin_edges, _ = self._create_voltage_bins(pulse_input[idx_main])
        
        samples_per_symbol = self.samples_per_symbol * self.upsampling
        window_size = samples_per_symbol
        
        pdf_list = []
        
        # Calculate for each time slice within one symbol period
        for idx in range(-window_size // 2, window_size // 2):
            # Sample pulse at this time offset
            sampled_amps = self._sample_pulse_points(pulse_input, idx_main, idx)
            
            if len(sampled_amps) == 0:
                # No samples, create uniform PDF
                pdf = np.ones(self.vh_size) / self.vh_size
                pdf_list.append(pdf)
                continue
            
            # Create histogram for first cursor
            pdf, _ = np.histogram(sampled_amps[0], bin_edges)
            pdf = pdf / (np.sum(pdf) + 1e-15)  # Normalize
            
            # Convolve with subsequent cursors
            for j in range(1, len(sampled_amps)):
                pdf_cursor, _ = np.histogram(sampled_amps[j], bin_edges)
                pdf_cursor = pdf_cursor / (np.sum(pdf_cursor) + 1e-15)
                
                # Convolve and renormalize
                pdf = np.convolve(pdf, pdf_cursor, mode='same')
                pdf = pdf / (np.sum(pdf) + 1e-15)
            
            pdf_list.append(pdf)
        
        return pdf_list
    
    def _calculate_by_brute_force(
        self,
        pulse_response: np.ndarray
    ) -> List[np.ndarray]:
        """
        Calculate ISI PDF using brute force enumeration.
        
        This is an exact calculation that enumerates all possible symbol
        combinations. Complexity is O(M^N) where M is modulation levels
        and N is sample size. Use with caution for large sample sizes.
        
        Args:
            pulse_response: Pulse response array
            
        Returns:
            List of PDF arrays, one per time slice within a symbol period
        """
        pulse_input, idx_main = self._prepare_pulse_response(pulse_response)
        bin_edges, _ = self._create_voltage_bins(pulse_input[idx_main])
        
        samples_per_symbol = self.samples_per_symbol * self.upsampling
        window_size = samples_per_symbol
        
        num_levels = len(self.get_modulation_levels())
        
        # Warn about computation time for large configurations
        max_combinations_warning = 10_000_000
        
        pdf_list = []
        
        for idx in range(-window_size // 2, window_size // 2):
            # Sample pulse at this time offset
            sampled_amps = self._sample_pulse_points(pulse_input, idx_main, idx)
            
            if len(sampled_amps) == 0:
                pdf = np.ones(self.vh_size) / self.vh_size
                pdf_list.append(pdf)
                continue
            
            # Check computation size
            num_samples = len(sampled_amps)
            num_combinations = num_levels ** num_samples
            
            if num_combinations > max_combinations_warning:
                raise RuntimeError(
                    f"Brute force would require {num_combinations} combinations. "
                    f"Reduce sample_size or use convolution method."
                )
            
            # Generate all combinations using meshgrid
            # Each row of sampled_amps contains amplitudes for one cursor position
            # across all modulation levels
            grids = np.meshgrid(*[sampled_amps[i] for i in range(num_samples)],
                               indexing='ij')
            all_combs = np.array([g.flatten() for g in grids]).T
            
            # Sum along axis to get all possible ISI values
            A = np.sum(all_combs, axis=1)
            
            # Create histogram
            pdf, _ = np.histogram(A, bin_edges)
            pdf = pdf / (np.sum(pdf) + 1e-15)  # Normalize
            
            pdf_list.append(pdf)
        
        return pdf_list
    
    def calculate(
        self,
        pulse_response: np.ndarray,
        method: str = 'convolution'
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ISI PDF for the given pulse response.
        
        Args:
            pulse_response: Pulse response array (should be pre-processed)
            method: Calculation method ('convolution' or 'brute_force')
            
        Returns:
            Dictionary containing:
            - 'pdf_list': List of PDF arrays per time slice
            - 'voltage_bins': Voltage bin center values
            - 'time_slices': Time slice values within one UI
            
        Raises:
            ValueError: If method is unknown or pulse response is invalid
        """
        # Prepare to get voltage bins and time slices
        pulse_input, idx_main = self._prepare_pulse_response(pulse_response)
        _, voltage_bins = self._create_voltage_bins(pulse_input[idx_main])
        
        samples_per_symbol = self.samples_per_symbol * self.upsampling
        time_slices = np.arange(-samples_per_symbol // 2, samples_per_symbol // 2)
        time_slices = time_slices / samples_per_symbol  # Normalize to UI
        
        # Calculate ISI PDFs
        if method == 'convolution':
            pdf_list = self._calculate_by_convolution(pulse_response)
        elif method == 'brute_force':
            pdf_list = self._calculate_by_brute_force(pulse_response)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'convolution' or 'brute_force'.")
        
        return {
            'pdf_list': pdf_list,
            'voltage_bins': voltage_bins,
            'time_slices': time_slices
        }
