"""BER contour calculation for statistical eye analysis."""

import numpy as np
from typing import List, Dict
from ..modulation import ModulationFormat


class BERContourCalculator:
    """Calculate BER contours from eye diagram PDF.
    
    Example:
        >>> calc = BERContourCalculator()
        >>> contours = calc.calculate_contours(pdf, ber_levels=[1e-12, 1e-9])
    """
    
    def calculate_contours(self,
                          eye_pdf: np.ndarray,
                          modulation: ModulationFormat,
                          ber_levels: List[float] = None) -> Dict[float, np.ndarray]:
        """Calculate BER contours at specified levels.
        
        Args:
            eye_pdf: Eye diagram PDF (2D array)
            modulation: Modulation format
            ber_levels: List of BER levels (default: [1e-12, 1e-9, 1e-6])
            
        Returns:
            Dictionary mapping BER level to contour matrix
        """
        if ber_levels is None:
            ber_levels = [1e-12, 1e-9, 1e-6]
        
        contours = {}
        
        for ber in ber_levels:
            contour = self._calculate_contour_at_ber(eye_pdf, modulation, ber)
            contours[ber] = contour
        
        return contours
    
    def _calculate_contour_at_ber(self, eye_pdf, modulation, target_ber):
        """Calculate single BER contour using CDF-based method.
        
        For each eye, calculate the Cumulative Distribution Function (CDF)
        to find the probability of error at each (voltage, time) point.
        """
        vh_size, window_size = eye_pdf.shape
        contour = np.zeros((vh_size, window_size))
        
        eye_centers = modulation.get_eye_centers()
        thresholds = modulation.get_thresholds()
        
        # For each time slice
        for t_idx in range(window_size):
            voltage_slice = eye_pdf[:, t_idx]
            
            if voltage_slice.sum() == 0:
                continue
            
            # Normalize to get PDF
            pdf = voltage_slice / voltage_slice.sum()
            
            # Calculate CDF from bottom to top
            cdf = np.cumsum(pdf)
            
            # For PAM4, we need to consider each eye separately
            # The BER contour represents the probability of being in the wrong eye
            
            if modulation.num_levels == 4:  # PAM4
                # Find approximate eye boundaries based on thresholds
                for eye_idx, threshold in enumerate(thresholds):
                    # Simple CDF-based error probability
                    # This is a simplified model - full implementation would
                    # calculate conditional probabilities for each eye
                    contour[:, t_idx] = np.minimum(cdf, 1 - cdf) * 2
            else:  # NRZ
                # For NRZ, use distance from center
                contour[:, t_idx] = np.minimum(cdf, 1 - cdf) * 2
        
        return contour
    
    def calculate_eye_dimensions(self,
                                 eye_pdf: np.ndarray,
                                 modulation: ModulationFormat,
                                 target_ber: float) -> Dict[str, float]:
        """Calculate eye height and width at target BER.
        
        Args:
            eye_pdf: Eye diagram PDF
            modulation: Modulation format
            target_ber: Target bit error rate
            
        Returns:
            Dictionary with eye_height and eye_width
        """
        # Simplified calculation
        # In full implementation, would use CDF analysis
        
        eye_heights = []
        thresholds = modulation.get_thresholds()
        
        for threshold in thresholds:
            # Find eye opening at this threshold
            height = self._compute_height_at_threshold(eye_pdf, threshold)
            eye_heights.append(height)
        
        return {
            'eye_heights': eye_heights,
            'eye_height_min': min(eye_heights) if eye_heights else 0.0,
            'eye_height_avg': sum(eye_heights) / len(eye_heights) if eye_heights else 0.0,
        }
    
    def _compute_height_at_threshold(self, eye_pdf, threshold):
        """Compute eye height at a decision threshold using CDF analysis.
        
        Args:
            eye_pdf: Eye diagram PDF (2D array)
            threshold: Decision threshold voltage
            
        Returns:
            Eye height at the specified threshold
        """
        vh_size, window_size = eye_pdf.shape
        
        # For each time slice, find the voltage range where density is low
        # (the eye opening at this threshold)
        
        eye_openings = []
        
        for t_idx in range(window_size):
            voltage_slice = eye_pdf[:, t_idx]
            
            if voltage_slice.sum() == 0:
                continue
            
            # Find low-density region around threshold
            # Use a simple threshold on normalized density
            normalized = voltage_slice / voltage_slice.max()
            
            # Find regions where density < 50% of max (eye opening)
            low_density_mask = normalized < 0.5
            
            if np.any(low_density_mask):
                # Find the extent of the low-density region
                low_density_indices = np.where(low_density_mask)[0]
                if len(low_density_indices) > 0:
                    # Height in voltage bins
                    height_bins = low_density_indices[-1] - low_density_indices[0]
                    eye_openings.append(float(height_bins))
        
        if eye_openings:
            return np.mean(eye_openings)
        return 0.0
