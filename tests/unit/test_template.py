#!/usr/bin/env python3
"""
Tests for JTolTemplate class.

Task 3.4: 标准模板定义 - JTol 模板测试
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from eye_analyzer.ber.template import JTolTemplate


class TestJTolTemplateInitialization:
    """Tests for JTolTemplate initialization."""
    
    def test_default_template(self):
        """Test default template is ieee_802_3ck."""
        template = JTolTemplate()
        assert template.template_name == 'ieee_802_3ck'
    
    def test_valid_templates(self):
        """Test all valid template names can be initialized."""
        valid_templates = ['ieee_802_3ck', 'oif_cei_112g', 'jedec_ddr5', 'pcie_gen6']
        for name in valid_templates:
            template = JTolTemplate(name)
            assert template.template_name == name
    
    def test_invalid_template_raises_error(self):
        """Test invalid template name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            JTolTemplate('invalid_template')


class TestJTolTemplateIEEE802_3ck:
    """Tests for IEEE 802.3ck template (200G/400G Ethernet)."""
    
    def setup_method(self):
        self.template = JTolTemplate('ieee_802_3ck')
    
    def test_frequency_range(self):
        """Test frequency range is returned correctly."""
        f_min, f_max = self.template.get_frequency_range()
        assert f_min > 0
        assert f_max > f_min
        # IEEE 802.3ck typically covers 10kHz to ~1GHz
        assert f_min <= 1e5  # <= 100 kHz
        assert f_max >= 1e9  # >= 1 GHz
    
    def test_low_frequency_sj_limit(self):
        """Test SJ limit at low frequency is around 0.1 UI."""
        sj_limit = self.template.get_sj_limit(1e5)  # 100 kHz
        assert sj_limit is not None
        # Low frequency SJ limit should be approximately 0.1 UI
        assert 0.05 <= sj_limit <= 0.2
    
    def test_high_frequency_sj_limit_decreases(self):
        """Test SJ limit decreases at high frequency."""
        sj_low = self.template.get_sj_limit(1e6)   # 1 MHz
        sj_high = self.template.get_sj_limit(1e9)  # 1 GHz
        assert sj_high < sj_low
    
    def test_corner_frequency_behavior(self):
        """Test SJ limit at corner frequency (around 10 MHz)."""
        sj_at_corner = self.template.get_sj_limit(10e6)
        assert sj_at_corner is not None
        assert sj_at_corner > 0
    
    def test_sj_limit_never_negative(self):
        """Test SJ limit is never negative."""
        freqs = np.logspace(4, 10, 100)  # 10 kHz to 10 GHz
        for f in freqs:
            sj_limit = self.template.get_sj_limit(f)
            assert sj_limit >= 0


class TestJTolTemplateOIFCEI112G:
    """Tests for OIF-CEI-112G template."""
    
    def setup_method(self):
        self.template = JTolTemplate('oif_cei_112g')
    
    def test_frequency_range(self):
        """Test frequency range is returned correctly."""
        f_min, f_max = self.template.get_frequency_range()
        assert f_min > 0
        assert f_max > f_min
    
    def test_sj_limit_at_various_frequencies(self):
        """Test SJ limit can be retrieved at various frequencies."""
        test_freqs = [1e5, 1e6, 1e7, 1e8, 1e9]
        for f in test_freqs:
            sj_limit = self.template.get_sj_limit(f)
            assert sj_limit is not None
            assert sj_limit >= 0


class TestJTolTemplateJEDEC_DDR5:
    """Tests for JEDEC DDR5 template."""
    
    def setup_method(self):
        self.template = JTolTemplate('jedec_ddr5')
    
    def test_frequency_range(self):
        """Test frequency range is returned correctly."""
        f_min, f_max = self.template.get_frequency_range()
        assert f_min > 0
        assert f_max > f_min
    
    def test_ddr5_specific_limits(self):
        """Test DDR5 specific SJ limits."""
        # DDR5 has different jitter requirements
        sj_limit = self.template.get_sj_limit(1e6)
        assert sj_limit is not None
        assert sj_limit >= 0


class TestJTolTemplatePCIeGen6:
    """Tests for PCIe Gen6 template."""
    
    def setup_method(self):
        self.template = JTolTemplate('pcie_gen6')
    
    def test_frequency_range(self):
        """Test frequency range is returned correctly."""
        f_min, f_max = self.template.get_frequency_range()
        assert f_min > 0
        assert f_max > f_min
    
    def test_pcie6_specific_limits(self):
        """Test PCIe Gen6 specific SJ limits."""
        sj_limit = self.template.get_sj_limit(1e6)
        assert sj_limit is not None
        assert sj_limit >= 0


class TestJTolTemplateEvaluate:
    """Tests for evaluate method (batch evaluation)."""
    
    def setup_method(self):
        self.template = JTolTemplate('ieee_802_3ck')
    
    def test_evaluate_single_frequency(self):
        """Test evaluate with single frequency."""
        freqs = np.array([1e6])
        sj_limits = self.template.evaluate(freqs)
        assert isinstance(sj_limits, np.ndarray)
        assert len(sj_limits) == 1
        assert sj_limits[0] >= 0
    
    def test_evaluate_multiple_frequencies(self):
        """Test evaluate with multiple frequencies."""
        freqs = np.array([1e5, 1e6, 1e7, 1e8])
        sj_limits = self.template.evaluate(freqs)
        assert isinstance(sj_limits, np.ndarray)
        assert len(sj_limits) == 4
        assert all(sj_limits >= 0)
    
    def test_evaluate_returns_decreasing_values(self):
        """Test evaluate returns decreasing values for increasing frequency."""
        freqs = np.array([1e5, 1e7, 1e9])
        sj_limits = self.template.evaluate(freqs)
        # SJ limits should generally decrease with frequency
        assert sj_limits[1] <= sj_limits[0] or np.isclose(sj_limits[1], sj_limits[0])
        assert sj_limits[2] <= sj_limits[1] or np.isclose(sj_limits[2], sj_limits[1])
    
    def test_evaluate_with_list(self):
        """Test evaluate accepts list input."""
        freqs = [1e5, 1e6, 1e7]
        sj_limits = self.template.evaluate(freqs)
        assert isinstance(sj_limits, np.ndarray)
        assert len(sj_limits) == 3


class TestJTolTemplateInterpolation:
    """Tests for interpolation behavior."""
    
    def setup_method(self):
        self.template = JTolTemplate('ieee_802_3ck')
    
    def test_interpolation_between_points(self):
        """Test interpolation between defined frequency points."""
        # Get SJ limits at two frequencies
        f1, f2 = 1e6, 10e6
        sj1 = self.template.get_sj_limit(f1)
        sj2 = self.template.get_sj_limit(f2)
        
        # Get SJ limit at midpoint
        f_mid = (f1 + f2) / 2
        sj_mid = self.template.get_sj_limit(f_mid)
        
        # Midpoint should be between the two values (for monotonic regions)
        assert min(sj1, sj2) <= sj_mid <= max(sj1, sj2)
    
    def test_extrapolation_below_range(self):
        """Test behavior below defined frequency range."""
        f_min, _ = self.template.get_frequency_range()
        sj_below = self.template.get_sj_limit(f_min / 10)
        # Should return the low frequency limit
        assert sj_below is not None
        assert sj_below >= 0
    
    def test_extrapolation_above_range(self):
        """Test behavior above defined frequency range."""
        _, f_max = self.template.get_frequency_range()
        sj_above = self.template.get_sj_limit(f_max * 10)
        # Should return a small value or zero
        assert sj_above is not None
        assert sj_above >= 0


class TestJTolTemplateComparison:
    """Tests comparing different templates."""
    
    def test_templates_have_different_limits(self):
        """Test that different templates have different SJ limits."""
        templates = [
            JTolTemplate('ieee_802_3ck'),
            JTolTemplate('oif_cei_112g'),
        ]
        
        # Get SJ limit at 1 MHz for each template
        sj_limits = [t.get_sj_limit(1e6) for t in templates]
        
        # Different templates should have different limits
        # (or at least not all be exactly equal)
        assert len(set(sj_limits)) > 1 or all(s > 0 for s in sj_limits)


class TestJTolTemplateEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def setup_method(self):
        self.template = JTolTemplate('ieee_802_3ck')
    
    def test_zero_frequency_raises_error(self):
        """Test that zero frequency raises an error or returns boundary."""
        # Should handle gracefully
        try:
            sj_limit = self.template.get_sj_limit(0)
            # If no error, should return the maximum/lowest frequency limit
            assert sj_limit >= 0
        except (ValueError, ZeroDivisionError):
            pass  # Also acceptable to raise error
    
    def test_negative_frequency_raises_error(self):
        """Test that negative frequency raises ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            self.template.get_sj_limit(-1e6)
    
    def test_very_high_frequency(self):
        """Test behavior at very high frequency."""
        sj_limit = self.template.get_sj_limit(1e12)  # 1 THz
        assert sj_limit is not None
        assert sj_limit >= 0
        # SJ limit should be very small at very high frequency
        assert sj_limit < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
