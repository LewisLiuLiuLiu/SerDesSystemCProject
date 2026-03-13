"""Modulation format abstraction layer for eye_analyzer."""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Union
import numpy as np


class ModulationFormat(ABC):
    """Abstract base class for modulation formats.
    
    Supports: NRZ, PAM3, PAM4, PAM6, PAM8
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Format name."""
        pass
    
    @property
    @abstractmethod
    def num_levels(self) -> int:
        """Number of signal levels (M)."""
        pass
    
    @property
    def num_eyes(self) -> int:
        """Number of eye openings (M-1)."""
        return self.num_levels - 1
    
    @abstractmethod
    def get_levels(self) -> np.ndarray:
        """Return signal level array.
        
        PAM4: [-3, -1, 1, 3]
        NRZ:  [-1, 1]
        """
        pass
    
    @abstractmethod
    def get_thresholds(self) -> np.ndarray:
        """Return decision thresholds between levels.
        
        PAM4: [-2, 0, 2]
        NRZ:  [0]
        """
        pass
    
    @abstractmethod
    def get_eye_centers(self) -> np.ndarray:
        """Return center voltage of each eye opening.
        
        PAM4: [-2, 0, 2] (upper/middle/lower eye)
        NRZ:  [0]
        """
        pass
    
    def get_level_names(self) -> List[str]:
        """Return names for each signal level."""
        return [f"LV{i}" for i in range(self.num_levels)]


class PAM4(ModulationFormat):
    """PAM4 modulation: 4 levels at -3, -1, 1, 3."""
    
    @property
    def name(self) -> str:
        return 'pam4'
    
    @property
    def num_levels(self) -> int:
        return 4
    
    def get_levels(self) -> np.ndarray:
        return np.array([-3, -1, 1, 3])
    
    def get_thresholds(self) -> np.ndarray:
        return np.array([-2, 0, 2])
    
    def get_eye_centers(self) -> np.ndarray:
        return np.array([-2, 0, 2])
    
    def get_level_names(self) -> List[str]:
        return ['LV3', 'LV2', 'LV1', 'LV0']


class NRZ(ModulationFormat):
    """NRZ/PAM2 modulation: 2 levels at -1, 1."""
    
    @property
    def name(self) -> str:
        return 'nrz'
    
    @property
    def num_levels(self) -> int:
        return 2
    
    def get_levels(self) -> np.ndarray:
        return np.array([-1, 1])
    
    def get_thresholds(self) -> np.ndarray:
        return np.array([0])
    
    def get_eye_centers(self) -> np.ndarray:
        return np.array([0])


# Registry for factory function
MODULATION_REGISTRY: Dict[str, Type[ModulationFormat]] = {
    'nrz': NRZ,
    'pam4': PAM4,
}


def create_modulation(name: str) -> ModulationFormat:
    """Factory function to create modulation format instances.
    
    Args:
        name: Modulation format name ('nrz', 'pam4', etc.)
        
    Returns:
        ModulationFormat instance
        
    Raises:
        ValueError: If modulation name is not recognized
    """
    if name not in MODULATION_REGISTRY:
        available = list(MODULATION_REGISTRY.keys())
        raise ValueError(f"Unknown modulation '{name}'. Available: {available}")
    return MODULATION_REGISTRY[name]()
