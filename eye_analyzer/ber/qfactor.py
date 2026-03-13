"""Q-factor and BER conversion utilities.

This module provides conversion between Q-factor and Bit Error Rate (BER)
using the complementary error function relationship.

Formulas:
    Q = sqrt(2) * erfcinv(2 * BER)
    BER = 0.5 * erfc(Q / sqrt(2))

where erfc is the complementary error function and erfcinv is its inverse.
"""

import math
from typing import Dict
from scipy.special import erfcinv, erfc


class QFactor:
    """Q-factor and BER conversion utilities.
    
    The Q-factor is a measure of signal quality in digital communications.
    It relates to the Bit Error Rate (BER) through the complementary error
    function, assuming Gaussian noise distribution.
    
    Example:
        >>> from eye_analyzer.ber.qfactor import QFactor
        >>> q = QFactor.ber_to_q(1e-12)  # Get Q for BER = 1e-12
        >>> print(f"Q = {q:.2f}")  # Q ≈ 7.03
        >>> ber = QFactor.q_to_ber(7.03)  # Get BER for Q = 7.03
        >>> print(f"BER = {ber:.2e}")  # BER ≈ 1e-12
    """

    @staticmethod
    def ber_to_q(ber: float) -> float:
        """Convert Bit Error Rate (BER) to Q-factor.
        
        Uses the formula:
            Q = sqrt(2) * erfcinv(2 * BER)
        
        Args:
            ber: Bit Error Rate, must be in range (0, 0.5]
            
        Returns:
            Q-factor value
            
        Raises:
            ValueError: If BER is not in valid range (0, 0.5]
            
        Example:
            >>> QFactor.ber_to_q(1e-12)
            7.034...
        """
        if ber <= 0 or ber > 0.5:
            raise ValueError(f"BER must be in range (0, 0.5], got {ber}")
        
        return math.sqrt(2) * erfcinv(2 * ber)

    @staticmethod
    def q_to_ber(q: float) -> float:
        """Convert Q-factor to Bit Error Rate (BER).
        
        Uses the formula:
            BER = 0.5 * erfc(Q / sqrt(2))
        
        Args:
            q: Q-factor value, must be positive
            
        Returns:
            Bit Error Rate
            
        Raises:
            ValueError: If Q is not positive
            
        Example:
            >>> QFactor.q_to_ber(7.03)
            1.0e-12
        """
        if q <= 0:
            raise ValueError(f"Q must be positive, got {q}")
        
        return 0.5 * erfc(q / math.sqrt(2))

    @staticmethod
    def get_common_values() -> Dict[float, float]:
        """Get common BER to Q-factor reference values.
        
        Returns a dictionary mapping common BER values to their
        corresponding Q-factors.
        
        Returns:
            Dictionary with BER as keys and Q-factor as values
            
        Example:
            >>> values = QFactor.get_common_values()
            >>> values[1e-12]
            7.03
        """
        return {
            1e-3: 3.09,
            1e-6: 4.75,
            1e-9: 5.99,
            1e-12: 7.03,
            1e-15: 7.94,
        }
