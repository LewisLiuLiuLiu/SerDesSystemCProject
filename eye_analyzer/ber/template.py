#!/usr/bin/env python3
"""
JTolTemplate - Industry Standard Jitter Tolerance Templates.

Defines standard JTol templates for various high-speed serial interfaces:
- IEEE 802.3ck (200G/400G Ethernet)
- OIF-CEI-112G
- JEDEC DDR5
- PCIe Gen6

Each template defines:
- Frequency range (Hz)
- SJ (Sinusoidal Jitter) amplitude limits (UI)
- Interpolation method
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from scipy import interpolate


class JTolTemplate:
    """
    Industry standard Jitter Tolerance (JTol) template.
    
    Defines SJ (Sinusoidal Jitter) amplitude limits as a function of
    modulation frequency for compliance testing of high-speed serial links.
    
    Attributes:
        template_name: Name of the template standard
        TEMPLATES: Dictionary of all available templates
    
    Example:
        >>> template = JTolTemplate('ieee_802_3ck')
        >>> sj_limit = template.get_sj_limit(1e6)  # SJ limit at 1 MHz
        >>> freqs = np.array([1e5, 1e6, 1e7])
        >>> sj_limits = template.evaluate(freqs)  # Batch evaluation
    """
    
    TEMPLATES: Dict[str, Dict] = {
        'ieee_802_3ck': {
            'name': 'IEEE 802.3ck (200G/400G Ethernet)',
            'description': '200Gb/s and 400Gb/s Ethernet JTol template',
            # Frequency points in Hz
            'frequencies': np.array([1e4, 1e5, 1e6, 10e6, 100e6, 1e9, 4e9]),
            # SJ amplitude limits in UI
            'sj_limits': np.array([0.1, 0.1, 0.1, 0.1, 0.01, 0.001, 0.0005]),
            'interpolation': 'log',
        },
        'oif_cei_112g': {
            'name': 'OIF-CEI-112G',
            'description': 'Common Electrical Interface 112G JTol template',
            # Frequency points in Hz
            'frequencies': np.array([1e4, 1e5, 1e6, 4e6, 10e6, 100e6, 1e9, 4e9]),
            # SJ amplitude limits in UI
            'sj_limits': np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.005, 0.0005, 0.00025]),
            'interpolation': 'log',
        },
        'jedec_ddr5': {
            'name': 'JEDEC DDR5',
            'description': 'DDR5 memory interface JTol template',
            # Frequency points in Hz
            'frequencies': np.array([1e4, 1e5, 1e6, 4e6, 10e6, 50e6, 200e6, 800e6]),
            # SJ amplitude limits in UI
            'sj_limits': np.array([0.15, 0.15, 0.15, 0.15, 0.075, 0.03, 0.015, 0.0075]),
            'interpolation': 'log',
        },
        'pcie_gen6': {
            'name': 'PCIe Gen6',
            'description': 'PCI Express Generation 6 JTol template',
            # Frequency points in Hz
            'frequencies': np.array([1e4, 1e5, 1e6, 10e6, 100e6, 500e6, 2e9, 4e9]),
            # SJ amplitude limits in UI
            'sj_limits': np.array([0.12, 0.12, 0.12, 0.12, 0.012, 0.0024, 0.0006, 0.0003]),
            'interpolation': 'log',
        },
    }
    
    def __init__(self, template_name: str = 'ieee_802_3ck'):
        """
        Initialize JTol template.
        
        Args:
            template_name: Name of the template to use. Options:
                - 'ieee_802_3ck': IEEE 802.3ck (200G/400G Ethernet)
                - 'oif_cei_112g': OIF-CEI-112G
                - 'jedec_ddr5': JEDEC DDR5
                - 'pcie_gen6': PCIe Gen6
        
        Raises:
            ValueError: If template_name is not a known template.
        
        Example:
            >>> template = JTolTemplate('ieee_802_3ck')
            >>> template = JTolTemplate('oif_cei_112g')
        """
        if template_name not in self.TEMPLATES:
            valid_templates = list(self.TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Valid templates: {valid_templates}"
            )
        
        self.template_name = template_name
        self._template = self.TEMPLATES[template_name]
        self._frequencies = self._template['frequencies']
        self._sj_limits = self._template['sj_limits']
        self._interpolation = self._template['interpolation']
        
        # Create interpolation function
        self._interp_func = self._create_interpolator()
    
    def _create_interpolator(self) -> interpolate.interp1d:
        """
        Create interpolation function for SJ limits.
        
        Returns:
            SciPy interpolation function.
        """
        freqs = self._frequencies
        sj_limits = self._sj_limits
        
        if self._interpolation == 'log':
            # Use log-log interpolation for frequency-decay relationship
            log_freqs = np.log10(freqs)
            log_sj = np.log10(sj_limits)
            
            return interpolate.interp1d(
                log_freqs,
                log_sj,
                kind='linear',
                bounds_error=False,
                fill_value=(log_sj[0], log_sj[-1])
            )
        else:
            # Linear interpolation
            return interpolate.interp1d(
                freqs,
                sj_limits,
                kind='linear',
                bounds_error=False,
                fill_value=(sj_limits[0], sj_limits[-1])
            )
    
    def get_sj_limit(self, frequency: float) -> float:
        """
        Get SJ amplitude limit at specified frequency.
        
        Returns the maximum allowed Sinusoidal Jitter (SJ) amplitude
        in UI (Unit Intervals) at the given modulation frequency.
        
        Args:
            frequency: Modulation frequency in Hz. Must be positive.
        
        Returns:
            SJ amplitude limit in UI.
        
        Raises:
            ValueError: If frequency is negative.
        
        Example:
            >>> template = JTolTemplate('ieee_802_3ck')
            >>> sj_limit = template.get_sj_limit(1e6)  # SJ limit at 1 MHz
            >>> print(f"SJ limit at 1 MHz: {sj_limit:.4f} UI")
        """
        if frequency < 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        
        if frequency == 0:
            # Return the low frequency limit
            return float(self._sj_limits[0])
        
        if self._interpolation == 'log':
            log_freq = np.log10(frequency)
            log_sj = self._interp_func(log_freq)
            return float(10 ** log_sj)
        else:
            return float(self._interp_func(frequency))
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """
        Get the frequency range supported by this template.
        
        Returns:
            Tuple of (f_min, f_max) in Hz.
        
        Example:
            >>> template = JTolTemplate('ieee_802_3ck')
            >>> f_min, f_max = template.get_frequency_range()
            >>> print(f"Range: {f_min/1e6:.1f} MHz to {f_max/1e9:.1f} GHz")
        """
        return (float(self._frequencies[0]), float(self._frequencies[-1]))
    
    def evaluate(self, frequencies: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Batch evaluate SJ limits at multiple frequencies.
        
        Args:
            frequencies: Array or list of frequencies in Hz.
        
        Returns:
            Array of SJ amplitude limits in UI, same shape as input.
        
        Example:
            >>> template = JTolTemplate('ieee_802_3ck')
            >>> freqs = np.array([1e5, 1e6, 1e7, 1e8])
            >>> sj_limits = template.evaluate(freqs)
        """
        frequencies = np.asarray(frequencies)
        
        if frequencies.size == 0:
            return np.array([])
        
        # Handle scalar input
        if frequencies.ndim == 0:
            return np.array([self.get_sj_limit(float(frequencies))])
        
        # Vectorized evaluation
        if self._interpolation == 'log':
            log_freqs = np.log10(frequencies)
            log_sj = self._interp_func(log_freqs)
            return 10 ** log_sj
        else:
            return self._interp_func(frequencies)
    
    def get_template_info(self) -> Dict:
        """
        Get information about the current template.
        
        Returns:
            Dictionary with template information.
        
        Example:
            >>> template = JTolTemplate('ieee_802_3ck')
            >>> info = template.get_template_info()
            >>> print(info['name'])
            >>> print(info['description'])
        """
        return {
            'name': self._template['name'],
            'description': self._template['description'],
            'template_name': self.template_name,
            'frequency_range': self.get_frequency_range(),
            'interpolation': self._interpolation,
        }
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """
        List all available template names.
        
        Returns:
            List of template name strings.
        
        Example:
            >>> templates = JTolTemplate.list_templates()
            >>> print(templates)
            ['ieee_802_3ck', 'oif_cei_112g', 'jedec_ddr5', 'pcie_gen6']
        """
        return list(cls.TEMPLATES.keys())
    
    def __repr__(self) -> str:
        """String representation of the template."""
        return (
            f"JTolTemplate('{self.template_name}')"
            f" - {self._template['name']}"
        )
