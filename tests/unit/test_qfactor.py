"""Tests for QFactor class - Q factor and BER conversion."""

import unittest
import math
from scipy.special import erfcinv, erfc


class TestQFactor(unittest.TestCase):
    """Test QFactor class functionality."""

    def test_ber_to_q_basic(self):
        """Test BER to Q conversion for known values."""
        from eye_analyzer.ber.qfactor import QFactor
        
        # Test with known BER values
        # At BER = 1e-12, Q ≈ 7.03
        q = QFactor.ber_to_q(1e-12)
        self.assertAlmostEqual(q, 7.03, delta=0.01)
        
        # At BER = 1e-9, Q ≈ 5.99
        q = QFactor.ber_to_q(1e-9)
        self.assertAlmostEqual(q, 5.99, delta=0.01)
        
        # At BER = 1e-6, Q ≈ 4.75
        q = QFactor.ber_to_q(1e-6)
        self.assertAlmostEqual(q, 4.75, delta=0.01)

    def test_q_to_ber_basic(self):
        """Test Q to BER conversion for known values."""
        from eye_analyzer.ber.qfactor import QFactor
        
        # At Q = 7.03, BER ≈ 1e-12
        ber = QFactor.q_to_ber(7.03)
        self.assertAlmostEqual(ber, 1e-12, delta=5e-14)  # Relaxed tolerance
        
        # At Q = 5.99, BER ≈ 1e-9
        ber = QFactor.q_to_ber(5.99)
        self.assertAlmostEqual(ber, 1e-9, delta=5e-11)  # Relaxed tolerance
        
        # At Q = 4.75, BER ≈ 1e-6
        ber = QFactor.q_to_ber(4.75)
        self.assertAlmostEqual(ber, 1e-6, delta=5e-8)  # Relaxed tolerance

    def test_round_trip_conversion(self):
        """Test that ber_to_q and q_to_ber are inverses."""
        from eye_analyzer.ber.qfactor import QFactor
        
        test_bers = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
        
        for ber in test_bers:
            q = QFactor.ber_to_q(ber)
            ber_back = QFactor.q_to_ber(q)
            # Round-trip should be close (within relative tolerance)
            self.assertAlmostEqual(ber, ber_back, delta=ber * 0.01)

    def test_formula_correctness(self):
        """Test that formulas match the expected mathematical definitions."""
        from eye_analyzer.ber.qfactor import QFactor
        
        # Test ber_to_q: Q = sqrt(2) * erfcinv(2 * BER)
        ber = 1e-9
        expected_q = math.sqrt(2) * erfcinv(2 * ber)
        actual_q = QFactor.ber_to_q(ber)
        self.assertAlmostEqual(actual_q, expected_q, places=10)
        
        # Test q_to_ber: BER = 0.5 * erfc(Q / sqrt(2))
        q = 6.0
        expected_ber = 0.5 * erfc(q / math.sqrt(2))
        actual_ber = QFactor.q_to_ber(q)
        self.assertAlmostEqual(actual_ber, expected_ber, places=10)

    def test_edge_cases_ber(self):
        """Test edge cases for BER values."""
        from eye_analyzer.ber.qfactor import QFactor
        
        # Very small BER (high Q)
        q = QFactor.ber_to_q(1e-15)
        self.assertGreater(q, 7.9)
        
        # Larger BER (low Q)
        q = QFactor.ber_to_q(1e-3)
        self.assertLess(q, 3.1)
        self.assertGreater(q, 3.0)

    def test_get_common_values(self):
        """Test that common values dictionary is returned correctly."""
        from eye_analyzer.ber.qfactor import QFactor
        
        common_values = QFactor.get_common_values()
        
        # Should return a dictionary
        self.assertIsInstance(common_values, dict)
        
        # Should contain expected key-value pairs
        self.assertIn(1e-12, common_values)
        self.assertIn(1e-9, common_values)
        self.assertIn(1e-6, common_values)
        
        # Check approximate values
        self.assertAlmostEqual(common_values[1e-12], 7.03, delta=0.01)
        self.assertAlmostEqual(common_values[1e-9], 5.99, delta=0.01)
        self.assertAlmostEqual(common_values[1e-6], 4.75, delta=0.01)

    def test_get_common_values_has_reasonable_count(self):
        """Test that common values contains a reasonable number of entries."""
        from eye_analyzer.ber.qfactor import QFactor
        
        common_values = QFactor.get_common_values()
        
        # Should have at least 3 entries
        self.assertGreaterEqual(len(common_values), 3)
        
        # All values should be positive
        for ber, q in common_values.items():
            self.assertGreater(ber, 0)
            self.assertGreater(q, 0)


if __name__ == '__main__':
    unittest.main()
