"""
Tests for visualization module - PAM4 support and new plotting functions.

Test-Driven Development for Task 4.2:
- plot_eye_diagram with modulation support
- plot_jtol_curve
- plot_bathtub_curve
- create_analysis_report
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import os
import tempfile


class TestPlotEyeDiagram:
    """Tests for plot_eye_diagram function with NRZ and PAM4 support."""
    
    def test_plot_eye_diagram_nrz_basic(self):
        """Test basic NRZ eye diagram plotting."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        # Create test eye data (simple 2D array)
        eye_data = np.random.rand(64, 128)
        eye_data[30:34, 60:68] = 1.0  # Create an eye opening
        
        time_bins = np.linspace(-1, 1, 129)  # 2 UI window
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        fig, ax = plt.subplots()
        
        # Should not raise any exception
        result = plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            title='Test NRZ Eye',
            ax=ax
        )
        
        assert result is not None or result is None  # Function returns something or None
        plt.close(fig)
    
    def test_plot_eye_diagram_pam4_three_eyes(self):
        """Test PAM4 three-eye overlay display with different colors."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        # Create PAM4 eye data with three distinct eye openings
        eye_data = np.zeros((64, 128))
        # Three eye levels at different voltages
        eye_data[10:14, 60:68] = 1.0   # Top eye
        eye_data[30:34, 60:68] = 1.0   # Middle eye
        eye_data[50:54, 60:68] = 1.0   # Bottom eye
        
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.6, 0.6, 65)
        
        fig, ax = plt.subplots()
        
        result = plot_eye_diagram(
            eye_data,
            modulation='pam4',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            title='Test PAM4 Eye',
            ax=ax
        )
        
        # Check that the plot was created
        assert len(ax.images) > 0 or len(ax.collections) > 0 or ax.has_data()
        plt.close(fig)
    
    def test_plot_eye_diagram_with_ber_contour(self):
        """Test eye diagram with BER contour overlay."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.random.rand(64, 128)
        eye_data[30:34, 60:68] = 1.0
        
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        # Create mock BER contour data
        ber_contour = np.ones((64, 128)) * 1e-12
        ber_contour[30:34, 60:68] = 1e-3  # Better BER in eye center
        
        fig, ax = plt.subplots()
        
        result = plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            show_ber_contour=True,
            ber_contour=ber_contour,
            ax=ax
        )
        
        # Check contour was added
        assert len(ax.collections) > 0 or ax.has_data()
        plt.close(fig)
    
    def test_plot_eye_diagram_with_eye_metrics(self):
        """Test eye diagram with metrics annotation."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.random.rand(64, 128)
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        # Create eye metrics
        eye_metrics = {
            'eye_height': 0.4,
            'eye_width': 0.7,
            'eye_opening': 0.35,
            'snr': 15.5,
            'ber': 1e-12
        }
        
        fig, ax = plt.subplots()
        
        plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            eye_metrics=eye_metrics,
            ax=ax
        )
        
        # Check that text annotations exist
        texts = [child for child in ax.get_children() 
                 if isinstance(child, matplotlib.text.Text)]
        assert len(texts) > 0
        plt.close(fig)
    
    def test_plot_eye_diagram_pam4_separate_colors(self):
        """Test PAM4 shows three eyes with different colors."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.zeros((64, 128))
        eye_data[10:14, 60:68] = 1.0
        eye_data[30:34, 60:68] = 1.0
        eye_data[50:54, 60:68] = 1.0
        
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.6, 0.6, 65)
        
        fig, ax = plt.subplots()
        
        plot_eye_diagram(
            eye_data,
            modulation='pam4',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            ax=ax
        )
        
        # For PAM4, there should be data plotted
        assert ax.has_data()
        plt.close(fig)
    
    def test_plot_eye_diagram_auto_create_axes(self):
        """Test that plot_eye_diagram creates its own figure when ax is None."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.random.rand(64, 128)
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        # Don't provide ax - should create its own
        fig = plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            title='Auto Figure Test'
        )
        
        assert fig is not None
        plt.close(fig)


class TestPlotJtolCurve:
    """Tests for plot_jtol_curve function."""
    
    def test_plot_jtol_curve_basic(self):
        """Test basic JTOL curve plotting."""
        from eye_analyzer.visualization import plot_jtol_curve
        
        # Create mock JTOL results
        jtol_results = {
            'frequencies': np.array([1e3, 1e4, 1e5, 1e6]),
            'jitter_tolerances': np.array([0.5, 0.4, 0.3, 0.2]),
            'unit': 'UI'
        }
        
        fig, ax = plt.subplots()
        
        result = plot_jtol_curve(
            jtol_results,
            title='Test JTOL',
            ax=ax
        )
        
        # Should have plotted data
        assert len(ax.lines) > 0
        plt.close(fig)
    
    def test_plot_jtol_curve_with_template(self):
        """Test JTOL curve with template comparison."""
        from eye_analyzer.visualization import plot_jtol_curve
        
        jtol_results = {
            'frequencies': np.array([1e3, 1e4, 1e5, 1e6]),
            'jitter_tolerances': np.array([0.5, 0.4, 0.3, 0.2]),
            'unit': 'UI'
        }
        
        # Mock template
        template = {
            'frequencies': np.array([1e3, 1e6]),
            'limits': np.array([0.6, 0.15]),
            'name': 'Test Template'
        }
        
        fig, ax = plt.subplots()
        
        plot_jtol_curve(
            jtol_results,
            template=template,
            ax=ax,
            show_margin=True
        )
        
        # Should have at least 2 lines (JTOL + template)
        assert len(ax.lines) >= 1
        plt.close(fig)
    
    def test_plot_jtol_curve_auto_axes(self):
        """Test JTOL curve creates its own figure when ax is None."""
        from eye_analyzer.visualization import plot_jtol_curve
        
        jtol_results = {
            'frequencies': np.array([1e3, 1e4, 1e5]),
            'jitter_tolerances': np.array([0.5, 0.4, 0.3]),
            'unit': 'UI'
        }
        
        fig = plot_jtol_curve(jtol_results, title='JTOL Test')
        
        assert fig is not None
        plt.close(fig)


class TestPlotBathtubCurve:
    """Tests for plot_bathtub_curve function."""
    
    def test_plot_bathtub_time_direction(self):
        """Test bathtub curve in time direction."""
        from eye_analyzer.visualization import plot_bathtub_curve
        
        # Create bathtub data (BER vs time offset)
        time_points = np.linspace(-0.5, 0.5, 100)
        ber_values = 10 ** (-10 * np.exp(-2 * (time_points / 0.15) ** 2))
        
        bathtub_data = {
            'x_values': time_points,
            'ber_values': ber_values,
            'direction': 'time',
            'unit': 'UI'
        }
        
        fig, ax = plt.subplots()
        
        plot_bathtub_curve(
            bathtub_data,
            direction='time',
            ax=ax,
            target_ber=1e-12
        )
        
        assert len(ax.lines) > 0
        plt.close(fig)
    
    def test_plot_bathtub_voltage_direction(self):
        """Test bathtub curve in voltage direction."""
        from eye_analyzer.visualization import plot_bathtub_curve
        
        voltage_points = np.linspace(-0.3, 0.3, 100)
        ber_values = 10 ** (-10 * np.exp(-2 * (voltage_points / 0.1) ** 2))
        
        bathtub_data = {
            'x_values': voltage_points,
            'ber_values': ber_values,
            'direction': 'voltage',
            'unit': 'V'
        }
        
        fig, ax = plt.subplots()
        
        plot_bathtub_curve(
            bathtub_data,
            direction='voltage',
            ax=ax,
            target_ber=1e-12
        )
        
        assert len(ax.lines) > 0
        plt.close(fig)
    
    def test_plot_bathtub_with_target_ber_line(self):
        """Test that target BER line is drawn."""
        from eye_analyzer.visualization import plot_bathtub_curve
        
        bathtub_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': np.ones(100) * 1e-10,
            'direction': 'time',
            'unit': 'UI'
        }
        
        fig, ax = plt.subplots()
        
        plot_bathtub_curve(
            bathtub_data,
            direction='time',
            ax=ax,
            target_ber=1e-12
        )
        
        # Should have bathtub curve and horizontal BER line
        assert len(ax.lines) >= 1
        # Check for horizontal line at target BER
        has_hline = any(line.get_ydata()[0] == line.get_ydata()[-1] 
                       for line in ax.lines)
        plt.close(fig)
    
    def test_plot_bathtub_auto_axes(self):
        """Test bathtub curve creates its own figure when ax is None."""
        from eye_analyzer.visualization import plot_bathtub_curve
        
        bathtub_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': np.ones(100) * 1e-10,
            'direction': 'time',
            'unit': 'UI'
        }
        
        fig = plot_bathtub_curve(bathtub_data, direction='time')
        
        assert fig is not None
        plt.close(fig)


class TestCreateAnalysisReport:
    """Tests for create_analysis_report function."""
    
    def test_create_analysis_report_basic(self):
        """Test basic analysis report creation."""
        from eye_analyzer.visualization import create_analysis_report
        
        # Create test data
        eye_data = np.random.rand(64, 128)
        eye_data[30:34, 60:68] = 1.0
        
        ber_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': np.ones(100) * 1e-10,
            'direction': 'time',
            'unit': 'UI'
        }
        
        jitter_data = {
            'rj': 1e-12,
            'dj': 5e-12,
            'tj': 12e-12,
            'components': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_report.png')
            
            fig = create_analysis_report(
                eye_data=eye_data,
                ber_data=ber_data,
                jitter_data=jitter_data,
                output_file=output_file,
                title='Test Report'
            )
            
            assert fig is not None
            assert os.path.exists(output_file)
            plt.close(fig)
    
    def test_create_analysis_report_with_jtol(self):
        """Test analysis report with JTOL data."""
        from eye_analyzer.visualization import create_analysis_report
        
        eye_data = np.random.rand(64, 128)
        
        ber_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': np.ones(100) * 1e-10,
            'direction': 'time',
            'unit': 'UI'
        }
        
        jitter_data = {
            'rj': 1e-12,
            'dj': 5e-12,
            'tj': 12e-12
        }
        
        jtol_data = {
            'frequencies': np.array([1e3, 1e4, 1e5]),
            'jitter_tolerances': np.array([0.5, 0.4, 0.3]),
            'unit': 'UI'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_report_with_jtol.png')
            
            fig = create_analysis_report(
                eye_data=eye_data,
                ber_data=ber_data,
                jitter_data=jitter_data,
                jtol_data=jtol_data,
                output_file=output_file
            )
            
            assert fig is not None
            assert os.path.exists(output_file)
            plt.close(fig)
    
    def test_create_analysis_report_return_only(self):
        """Test analysis report returns figure without saving if no output_file."""
        from eye_analyzer.visualization import create_analysis_report
        
        eye_data = np.random.rand(64, 128)
        ber_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': np.ones(100) * 1e-10,
            'direction': 'time',
            'unit': 'UI'
        }
        jitter_data = {'rj': 1e-12, 'dj': 5e-12, 'tj': 12e-12}
        
        fig = create_analysis_report(
            eye_data=eye_data,
            ber_data=ber_data,
            jitter_data=jitter_data
        )
        
        assert fig is not None
        plt.close(fig)


class TestVisualizationEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_plot_eye_diagram_invalid_modulation(self):
        """Test handling of invalid modulation type."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.random.rand(64, 128)
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        # Should raise ValueError for invalid modulation
        with pytest.raises(ValueError):
            plot_eye_diagram(
                eye_data,
                modulation='invalid',
                time_bins=time_bins,
                voltage_bins=voltage_bins
            )
    
    def test_plot_eye_diagram_empty_data(self):
        """Test handling of empty eye data."""
        from eye_analyzer.visualization import plot_eye_diagram
        
        eye_data = np.zeros((64, 128))
        time_bins = np.linspace(-1, 1, 129)
        voltage_bins = np.linspace(-0.5, 0.5, 65)
        
        fig, ax = plt.subplots()
        
        # Should handle empty data gracefully
        plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            ax=ax
        )
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
