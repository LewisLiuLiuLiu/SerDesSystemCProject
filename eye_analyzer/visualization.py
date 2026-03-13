"""
Eye Diagram Visualization Module

This module provides visualization utilities for eye diagram analysis,
using MATLAB-style colormap with imshow for both Golden CDR and Sampler-Centric schemes.

Features:
- MATLAB-style colormap (Blue -> Green -> Yellow -> Red -> White)
- Unified imshow-based visualization for all schemes
- 2-UI centered display
- Multiple output formats (PNG, SVG, PDF)
- PAM4 three-eye overlay display with different colors
- BER contour overlay
- Eye metrics annotation
- JTOL and Bathtub curve plotting
- Comprehensive analysis report generation

New in PAM4 Support:
- PAM4 modulation with three-eye overlay using distinct colors
- BER contour overlay on eye diagrams
- Eye metrics annotation
- JTOL curve plotting with template comparison
- Bathtub curve plotting (time and voltage directions)
- Multi-panel analysis report generation
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter


# ============================================================================
# Colormap Functions
# ============================================================================

def create_matlab_colormap() -> LinearSegmentedColormap:
    """
    Create MATLAB-style colormap for eye diagram.
    
    MATLAB color mapping logic:
    - PD=0: White (1,1,1) - background
    - PD>0: 
        R = max(2*PD - 1, 0)
        G = min(2*PD, 2 - 2*PD)
        B = max(1 - 2*PD, 0)
    
    Color progression:
    - Low density (PD≈0): Blue (0,0,1)
    - Medium density (PD≈0.5): Green (0,1,0)
    - High density (PD≈0.75): Yellow/Red (1,1,0)/(1,0,0)
    - Peak density (PD≈1): White (1,1,1)
    
    Returns:
        LinearSegmentedColormap object
    """
    # Create 256-level colormap based on MATLAB algorithm
    n_colors = 256
    colors = []
    
    for i in range(n_colors):
        # Normalize to [0, 1]
        pd = i / (n_colors - 1)
        
        # MATLAB color calculation (background is handled separately)
        if pd == 0:
            r, g, b = 1.0, 1.0, 1.0  # White for zero
        else:
            r = max(2 * pd - 1, 0)
            g = min(2 * pd, 2 - 2 * pd)
            b = max(1 - 2 * pd, 0)
        
        colors.append((r, g, b))
    
    return LinearSegmentedColormap.from_list('matlab_eye', colors, N=n_colors)


def create_pam4_colormaps() -> List[LinearSegmentedColormap]:
    """
    Create three distinct colormaps for PAM4 three-eye display.
    
    Returns:
        List of 3 LinearSegmentedColormap objects for top, middle, bottom eyes
        - Top eye: Blue-based colormap
        - Middle eye: Green-based colormap
        - Bottom eye: Red/Orange-based colormap
    """
    colormaps = []
    
    # Top eye: Blue -> Cyan -> White
    n_colors = 256
    blue_colors = []
    for i in range(n_colors):
        pd = i / (n_colors - 1)
        if pd == 0:
            r, g, b = 1.0, 1.0, 1.0
        else:
            r = min(pd * 0.5, 1.0)
            g = min(pd * 0.8, 1.0)
            b = min(0.5 + pd * 0.5, 1.0)
        blue_colors.append((r, g, b))
    colormaps.append(LinearSegmentedColormap.from_list('pam4_top', blue_colors, N=n_colors))
    
    # Middle eye: Green -> Yellow -> White
    green_colors = []
    for i in range(n_colors):
        pd = i / (n_colors - 1)
        if pd == 0:
            r, g, b = 1.0, 1.0, 1.0
        else:
            r = min(pd * 0.8, 1.0)
            g = min(0.4 + pd * 0.6, 1.0)
            b = min(pd * 0.3, 1.0)
        green_colors.append((r, g, b))
    colormaps.append(LinearSegmentedColormap.from_list('pam4_middle', green_colors, N=n_colors))
    
    # Bottom eye: Red -> Orange -> White
    red_colors = []
    for i in range(n_colors):
        pd = i / (n_colors - 1)
        if pd == 0:
            r, g, b = 1.0, 1.0, 1.0
        else:
            r = min(0.5 + pd * 0.5, 1.0)
            g = min(pd * 0.6, 1.0)
            b = min(pd * 0.3, 1.0)
        red_colors.append((r, g, b))
    colormaps.append(LinearSegmentedColormap.from_list('pam4_bottom', red_colors, N=n_colors))
    
    return colormaps


# ============================================================================
# Main Plotting Functions
# ============================================================================

def plot_eye_diagram(eye_data: np.ndarray,
                     modulation: str = 'nrz',
                     time_bins: Optional[np.ndarray] = None,
                     voltage_bins: Optional[np.ndarray] = None,
                     show_ber_contour: bool = False,
                     ber_contour: Optional[np.ndarray] = None,
                     eye_metrics: Optional[Dict[str, Any]] = None,
                     title: Optional[str] = None,
                     ax: Optional[plt.Axes] = None,
                     smooth_sigma: float = 1.0,
                     **kwargs) -> Optional[plt.Figure]:
    """
    Plot eye diagram with support for NRZ and PAM4 modulation.
    
    NRZ: Single eye display using MATLAB-style colormap
    PAM4: Three-eye overlay display using different colors for each eye level
    
    Args:
        eye_data: Eye diagram matrix (2D array) or list of 3 matrices for PAM4
        modulation: 'nrz' or 'pam4'
        time_bins: Time axis bin edges (UI normalized, e.g., [-1, 1])
        voltage_bins: Voltage axis bin edges (in volts)
        show_ber_contour: Whether to overlay BER contour lines
        ber_contour: BER contour data (2D array matching eye_data shape)
        eye_metrics: Eye diagram metrics for annotation (dict with eye_height, 
                    eye_width, snr, ber, etc.)
        title: Chart title
        ax: Matplotlib axes object (optional, creates new figure if None)
        smooth_sigma: Gaussian smoothing sigma (default: 1.0)
        **kwargs: Additional arguments for backward compatibility
    
    Returns:
        Figure object if ax is None, None otherwise
    
    Raises:
        ValueError: If modulation is not 'nrz' or 'pam4'
    
    Example:
        # NRZ eye
        plot_eye_diagram(eye_data, modulation='nrz', 
                        time_bins=np.linspace(-1, 1, 129),
                        voltage_bins=np.linspace(-0.5, 0.5, 65))
        
        # PAM4 three eyes
        plot_eye_diagram([eye_top, eye_middle, eye_bottom], modulation='pam4',
                        time_bins=time_bins, voltage_bins=voltage_bins)
    """
    modulation = modulation.lower()
    if modulation not in ('nrz', 'pam4'):
        raise ValueError(f"Invalid modulation '{modulation}'. Must be 'nrz' or 'pam4'.")
    
    # Create figure if ax not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        created_fig = True
    else:
        fig = ax.figure
    
    # Default bins if not provided
    if time_bins is None:
        time_bins = np.linspace(-1, 1, eye_data.shape[1] + 1)
    if voltage_bins is None:
        voltage_bins = np.linspace(-0.6, 0.6, eye_data.shape[0] + 1)
    
    # Convert time to picoseconds for display
    ui_ps = 1e12  # Assume UI is normalized to 1, convert to ps scale
    time_ps = time_bins * ui_ps
    
    if modulation == 'nrz':
        _plot_nrz_eye(ax, eye_data, time_ps, voltage_bins, 
                     show_ber_contour, ber_contour, smooth_sigma)
    else:  # PAM4
        _plot_pam4_eye(ax, eye_data, time_ps, voltage_bins,
                      show_ber_contour, ber_contour, smooth_sigma)
    
    # Add metrics annotation if provided
    if eye_metrics:
        _add_eye_metrics_text(ax, eye_metrics)
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        mod_str = 'PAM4' if modulation == 'pam4' else 'NRZ'
        ax.set_title(f'Statistical Eye Diagram ({mod_str})', fontsize=14)
    
    ax.set_xlabel('Time [ps]', fontsize=12)
    ax.set_ylabel('Voltage [V]', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    if created_fig:
        plt.tight_layout()
        return fig
    return None


def _plot_nrz_eye(ax: plt.Axes, eye_data: np.ndarray,
                  time_ps: np.ndarray, voltage_bins: np.ndarray,
                  show_ber_contour: bool, ber_contour: Optional[np.ndarray],
                  smooth_sigma: float) -> None:
    """Plot NRZ single eye."""
    # Data preprocessing
    hist_display = eye_data.T.copy()
    if smooth_sigma > 0:
        hist_display = gaussian_filter(hist_display, sigma=smooth_sigma)
    
    # Normalize to [0, 1] for colormap
    hist_max = hist_display.max()
    if hist_max > 0:
        hist_normalized = hist_display / hist_max
    else:
        hist_normalized = hist_display
    
    # Create MATLAB-style colormap
    cmap = create_matlab_colormap()
    
    # Plot using imshow
    extent = [time_ps[0], time_ps[-1], voltage_bins[0], voltage_bins[-1]]
    im = ax.imshow(
        hist_normalized,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation='bilinear'
    )
    
    # Add BER contour if requested
    if show_ber_contour and ber_contour is not None:
        _add_ber_contour(ax, ber_contour, extent)
    
    # Add zero voltage reference line
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # Add UI center line
    center_time = (time_ps[0] + time_ps[-1]) / 2
    ax.axvline(x=center_time, color='red', linestyle='-', 
              alpha=0.8, linewidth=1.5, label='Center')


def _plot_pam4_eye(ax: plt.Axes, eye_data: Union[np.ndarray, List[np.ndarray]],
                   time_ps: np.ndarray, voltage_bins: np.ndarray,
                   show_ber_contour: bool, ber_contour: Optional[np.ndarray],
                   smooth_sigma: float) -> None:
    """Plot PAM4 three-eye overlay with different colors."""
    # Handle both single array and list of 3 arrays
    if isinstance(eye_data, list):
        if len(eye_data) != 3:
            raise ValueError("PAM4 requires exactly 3 eye data arrays")
        eye_segments = eye_data
    else:
        # Split single array into three voltage regions
        n_rows = eye_data.shape[0]
        row_third = n_rows // 3
        eye_segments = [
            eye_data[2*row_third:, :],      # Top eye (highest voltage)
            eye_data[row_third:2*row_third, :],  # Middle eye
            eye_data[:row_third, :]         # Bottom eye (lowest voltage)
        ]
        # Adjust voltage bins for each segment
        v_third = (voltage_bins[-1] - voltage_bins[0]) / 3
        voltage_bins_list = [
            voltage_bins[2*row_third:],
            voltage_bins[row_third:2*row_third+1],
            voltage_bins[:row_third+1]
        ]
    
    colormaps = create_pam4_colormaps()
    extent = [time_ps[0], time_ps[-1], voltage_bins[0], voltage_bins[-1]]
    
    # Plot each eye with different colormap
    for i, (eye_segment, cmap) in enumerate(zip(eye_segments, colormaps)):
        hist_display = eye_segment.T.copy()
        if smooth_sigma > 0:
            hist_display = gaussian_filter(hist_display, sigma=smooth_sigma)
        
        hist_max = hist_display.max()
        if hist_max > 0:
            hist_normalized = hist_display / hist_max
        else:
            hist_normalized = hist_display
        
        # Calculate extent for this segment
        if isinstance(eye_data, list):
            # Use full voltage range for each
            seg_extent = extent
        else:
            # Calculate segment-specific extent
            v_min = voltage_bins[0] + i * (voltage_bins[-1] - voltage_bins[0]) / 3
            v_max = voltage_bins[0] + (i + 1) * (voltage_bins[-1] - voltage_bins[0]) / 3
            seg_extent = [time_ps[0], time_ps[-1], v_min, v_max]
        
        # Plot with alpha blending for overlay effect
        ax.imshow(
            hist_normalized,
            origin='lower',
            aspect='auto',
            extent=seg_extent,
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation='bilinear',
            alpha=0.9
        )
    
    # Add BER contour if requested
    if show_ber_contour and ber_contour is not None:
        _add_ber_contour(ax, ber_contour, extent)
    
    # Add reference lines for PAM4 levels
    v_range = voltage_bins[-1] - voltage_bins[0]
    for level in [0.25, 0.5, 0.75]:
        y_line = voltage_bins[0] + v_range * level
        ax.axhline(y=y_line, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Add center line
    center_time = (time_ps[0] + time_ps[-1]) / 2
    ax.axvline(x=center_time, color='red', linestyle='-', 
              alpha=0.8, linewidth=1.5, label='Center')


def _add_ber_contour(ax: plt.Axes, ber_contour: np.ndarray, 
                     extent: List[float]) -> None:
    """Add BER contour overlay to eye diagram."""
    # Define BER levels for contours
    levels = [1e-12, 1e-9, 1e-6, 1e-3]
    
    # Plot contours
    cs = ax.contour(
        ber_contour.T,
        levels=levels,
        colors=['red', 'orange', 'yellow', 'green'],
        linewidths=1.5,
        extent=extent,
        alpha=0.7
    )
    
    # Add contour labels
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0e')


def _add_eye_metrics_text(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """Add eye metrics text box to the plot."""
    lines = []
    
    if 'eye_height' in metrics:
        lines.append(f"Eye Height: {metrics['eye_height']*1000:.2f} mV")
    if 'eye_width' in metrics:
        lines.append(f"Eye Width: {metrics['eye_width']:.3f} UI")
    if 'eye_opening' in metrics:
        lines.append(f"Eye Opening: {metrics['eye_opening']*1000:.2f} mV")
    if 'snr' in metrics:
        lines.append(f"SNR: {metrics['snr']:.1f} dB")
    if 'ber' in metrics:
        lines.append(f"BER: {metrics['ber']:.2e}")
    
    if lines:
        textstr = '\n'.join(lines)
        props = dict(boxstyle='round', facecolor='white', 
                    edgecolor='gray', alpha=0.85)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')


# ============================================================================
# JTOL Curve Plotting
# ============================================================================

def plot_jtol_curve(jtol_results: Dict[str, Any],
                    template: Optional[Union[str, Dict[str, Any]]] = None,
                    title: str = 'Jitter Tolerance',
                    ax: Optional[plt.Axes] = None,
                    show_margin: bool = True,
                    **kwargs) -> Optional[plt.Figure]:
    """
    Plot JTOL (Jitter Tolerance) curve with template comparison.
    
    Args:
        jtol_results: JTOL analysis results dictionary containing:
            - frequencies: Array of jitter frequencies (Hz)
            - jitter_tolerances: Array of tolerated jitter amplitudes
            - unit: Unit of jitter tolerance ('UI' or 'ps')
        template: Template name (e.g., 'PCIE4', 'USB4') or template dict with:
            - frequencies: Template frequency points
            - limits: Template jitter limits
            - name: Template name
        title: Chart title
        ax: Matplotlib axes object (optional, creates new figure if None)
        show_margin: Whether to show margin region between JTOL and template
        **kwargs: Additional plotting arguments
    
    Returns:
        Figure object if ax is None, None otherwise
    
    Example:
        jtol_results = {
            'frequencies': np.array([1e3, 1e4, 1e5, 1e6]),
            'jitter_tolerances': np.array([0.5, 0.4, 0.3, 0.2]),
            'unit': 'UI'
        }
        plot_jtol_curve(jtol_results, template='PCIE4')
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.figure
    
    freqs = jtol_results['frequencies']
    jtol_values = jtol_results['jitter_tolerances']
    unit = jtol_results.get('unit', 'UI')
    
    # Plot JTOL curve
    ax.semilogx(freqs, jtol_values, 'b-o', linewidth=2, 
               markersize=6, label='Measured JTOL')
    
    # Plot template if provided
    if template is not None:
        if isinstance(template, str):
            template_data = _get_standard_jtol_template(template)
        else:
            template_data = template
        
        if template_data:
            template_freqs = template_data['frequencies']
            template_limits = template_data['limits']
            template_name = template_data.get('name', 'Template')
            
            ax.semilogx(template_freqs, template_limits, 'r--', 
                       linewidth=2, label=f'{template_name} Spec')
            
            # Show margin region
            if show_margin:
                # Interpolate to common frequency points
                f_common = np.logspace(
                    np.log10(max(freqs.min(), template_freqs.min())),
                    np.log10(min(freqs.max(), template_freqs.max())),
                    100
                )
                jtol_interp = np.interp(f_common, freqs, jtol_values)
                template_interp = np.interp(f_common, template_freqs, template_limits)
                
                ax.fill_between(f_common, jtol_interp, template_interp,
                               where=(jtol_interp >= template_interp),
                               alpha=0.2, color='green', label='Pass Margin')
                ax.fill_between(f_common, jtol_interp, template_interp,
                               where=(jtol_interp < template_interp),
                               alpha=0.2, color='red', label='Fail Region')
    
    ax.set_xlabel('Jitter Frequency [Hz]', fontsize=12)
    ax.set_ylabel(f'Jitter Tolerance [{unit}]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    if created_fig:
        plt.tight_layout()
        return fig
    return None


def _get_standard_jtol_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get standard JTOL template data by name."""
    templates = {
        'PCIE4': {
            'frequencies': np.array([1e4, 1e5, 1e6, 1e7, 1e8]),
            'limits': np.array([0.35, 0.35, 0.15, 0.05, 0.05]),
            'name': 'PCIe 4.0'
        },
        'PCIE5': {
            'frequencies': np.array([1e4, 1e5, 1e6, 1e7, 1e8]),
            'limits': np.array([0.30, 0.30, 0.12, 0.04, 0.04]),
            'name': 'PCIe 5.0'
        },
        'USB4': {
            'frequencies': np.array([1e4, 1e5, 1e6, 1e7, 1e8]),
            'limits': np.array([0.40, 0.40, 0.18, 0.06, 0.06]),
            'name': 'USB4'
        },
    }
    return templates.get(template_name.upper())


# ============================================================================
# Bathtub Curve Plotting
# ============================================================================

def plot_bathtub_curve(bathtub_data: Dict[str, Any],
                       direction: str = 'time',
                       title: Optional[str] = None,
                       ax: Optional[plt.Axes] = None,
                       target_ber: float = 1e-12,
                       **kwargs) -> Optional[plt.Figure]:
    """
    Plot bathtub curve (BER vs time or voltage offset).
    
    Args:
        bathtub_data: Bathtub analysis results dictionary containing:
            - x_values: Array of time/voltage offsets
            - ber_values: Array of BER values
            - direction: 'time' or 'voltage' (override parameter)
            - unit: Unit of x_values ('UI', 'ps', or 'V')
        direction: 'time' or 'voltage' - determines x-axis label
        title: Chart title (auto-generated if None)
        ax: Matplotlib axes object (optional, creates new figure if None)
        target_ber: Target BER line to draw (default: 1e-12)
        **kwargs: Additional plotting arguments
    
    Returns:
        Figure object if ax is None, None otherwise
    
    Example:
        bathtub_data = {
            'x_values': np.linspace(-0.5, 0.5, 100),
            'ber_values': ber_values,
            'direction': 'time',
            'unit': 'UI'
        }
        plot_bathtub_curve(bathtub_data, direction='time', target_ber=1e-12)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.figure
    
    x_values = bathtub_data['x_values']
    ber_values = np.array(bathtub_data['ber_values'])
    data_direction = bathtub_data.get('direction', direction)
    unit = bathtub_data.get('unit', 'UI' if data_direction == 'time' else 'V')
    
    # Clip BER values to avoid log(0)
    ber_values = np.clip(ber_values, 1e-18, 1.0)
    
    # Plot bathtub curve
    ax.semilogy(x_values, ber_values, 'b-', linewidth=2, label='BER')
    
    # Add target BER line
    ax.axhline(y=target_ber, color='r', linestyle='--', 
              linewidth=1.5, label=f'Target BER = {target_ber:.0e}')
    
    # Find and annotate eye opening at target BER
    above_threshold = ber_values < target_ber
    if np.any(above_threshold):
        crossing_indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]
        if len(crossing_indices) >= 2:
            left_edge = x_values[crossing_indices[0]]
            right_edge = x_values[crossing_indices[-1]]
            eye_opening = right_edge - left_edge
            
            # Add annotation
            ax.axvline(x=left_edge, color='g', linestyle=':', alpha=0.7)
            ax.axvline(x=right_edge, color='g', linestyle=':', alpha=0.7)
            ax.annotate(f'Eye Opening @ {target_ber:.0e}\n= {eye_opening:.3f} {unit}',
                       xy=((left_edge + right_edge) / 2, target_ber),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='green'))
    
    # Set labels
    if data_direction == 'time':
        xlabel = f'Time Offset [{unit}]'
        default_title = 'Bathtub Curve (Time)'
    else:
        xlabel = f'Voltage Offset [{unit}]'
        default_title = 'Bathtub Curve (Voltage)'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(title if title else default_title, fontsize=14)
    
    ax.set_ylim(bottom=1e-18, top=1.0)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    if created_fig:
        plt.tight_layout()
        return fig
    return None


# ============================================================================
# Analysis Report Generation
# ============================================================================

def create_analysis_report(eye_data: np.ndarray,
                          ber_data: Dict[str, Any],
                          jitter_data: Dict[str, Any],
                          jtol_data: Optional[Dict[str, Any]] = None,
                          output_file: Optional[str] = None,
                          title: str = 'Eye Analysis Report',
                          modulation: str = 'nrz',
                          time_bins: Optional[np.ndarray] = None,
                          voltage_bins: Optional[np.ndarray] = None,
                          eye_metrics: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """
    Create comprehensive analysis report with multiple subplots.
    
    The report includes:
    - Eye diagram (main plot)
    - Bathtub curve
    - Jitter decomposition (if jitter_data provided)
    - JTOL curve (if jtol_data provided)
    
    Args:
        eye_data: Eye diagram matrix (2D array)
        ber_data: Bathtub analysis results
        jitter_data: Jitter decomposition results
        jtol_data: Optional JTOL analysis results
        output_file: Output file path (if None, only returns figure)
        title: Report title
        modulation: 'nrz' or 'pam4'
        time_bins: Time axis bin edges for eye diagram
        voltage_bins: Voltage axis bin edges for eye diagram
        eye_metrics: Eye diagram metrics for annotation
    
    Returns:
        Matplotlib Figure object
    
    Example:
        fig = create_analysis_report(
            eye_data=eye_matrix,
            ber_data=bathtub_results,
            jitter_data=jitter_results,
            jtol_data=jtol_results,
            output_file='report.png'
        )
    """
    # Determine subplot layout based on available data
    if jtol_data is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_eye = fig.add_subplot(gs[0, 0])
        ax_bathtub = fig.add_subplot(gs[0, 1])
        ax_jitter = fig.add_subplot(gs[1, 0])
        ax_jtol = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_eye = fig.add_subplot(gs[0, 0])
        ax_bathtub = fig.add_subplot(gs[0, 1])
        ax_jitter = fig.add_subplot(gs[1, :])
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Eye Diagram
    plot_eye_diagram(
        eye_data,
        modulation=modulation,
        time_bins=time_bins,
        voltage_bins=voltage_bins,
        eye_metrics=eye_metrics,
        title='Statistical Eye Diagram',
        ax=ax_eye
    )
    
    # 2. Bathtub Curve
    plot_bathtub_curve(
        ber_data,
        direction=ber_data.get('direction', 'time'),
        title='Bathtub Curve',
        ax=ax_bathtub,
        target_ber=1e-12
    )
    
    # 3. Jitter Decomposition
    _plot_jitter_decomposition(ax_jitter, jitter_data)
    
    # 4. JTOL Curve (if provided)
    if jtol_data is not None:
        plot_jtol_curve(
            jtol_data,
            title='Jitter Tolerance',
            ax=ax_jtol,
            show_margin=True
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig


def _plot_jitter_decomposition(ax: plt.Axes, jitter_data: Dict[str, Any]) -> None:
    """Plot jitter decomposition histogram and components."""
    components = jitter_data.get('components', {})
    
    if 'histogram' in components and 'bins' in components:
        # Plot histogram
        hist = components['histogram']
        bins = components['bins']
        ax.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.6, 
              color='blue', edgecolor='black', label='Jitter Histogram')
    
    # Plot RJ (Gaussian) component
    if 'rj' in jitter_data:
        rj = jitter_data['rj']
        # Generate Gaussian approximation
        x = np.linspace(-5*rj, 5*rj, 200) if rj > 0 else np.linspace(-1e-12, 1e-12, 200)
        gaussian = np.exp(-x**2 / (2 * rj**2)) / (rj * np.sqrt(2 * np.pi)) if rj > 0 else np.zeros_like(x)
        ax.plot(x, gaussian, 'r-', linewidth=2, label=f'RJ (σ={rj*1e12:.2f} ps)')
    
    # Annotate key metrics
    metrics_text = []
    if 'rj' in jitter_data:
        metrics_text.append(f"RJ: {jitter_data['rj']*1e12:.2f} ps")
    if 'dj' in jitter_data:
        metrics_text.append(f"DJ: {jitter_data['dj']*1e12:.2f} ps")
    if 'tj' in jitter_data:
        metrics_text.append(f"TJ: {jitter_data['tj']*1e12:.2f} ps")
    
    if metrics_text:
        textstr = '\n'.join(metrics_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Jitter Decomposition', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


# ============================================================================
# Legacy Functions (Backward Compatibility)
# ============================================================================

def plot_eye_diagram_legacy(eye_matrix: np.ndarray,
                            xedges: np.ndarray,
                            yedges: np.ndarray,
                            ui: float,
                            scheme: str,
                            metrics: Dict[str, Any],
                            output_path: str,
                            dpi: int = 300,
                            figsize: Tuple[float, float] = (12, 8),
                            smooth_sigma: float = 1.0) -> None:
    """
    Legacy eye diagram plotting function for backward compatibility.
    
    Uses the new plot_eye_diagram internally but maintains old interface.
    
    Args:
        eye_matrix: 2D density matrix (ui_bins x amp_bins)
        xedges: Time/phase axis bin edges
        yedges: Voltage axis bin edges (in volts)
        ui: Unit interval in seconds
        scheme: 'sampler_centric' or 'golden_cdr'
        metrics: Analysis metrics dictionary
        output_path: Output file path
        dpi: Output image DPI (default: 300)
        figsize: Figure size in inches (default: (12, 8))
        smooth_sigma: Gaussian smoothing sigma (default: 1.0)
    """
    # Convert xedges to time_bins format
    time_bins = xedges / ui if ui > 0 else xedges  # Normalize to UI
    voltage_bins = yedges
    
    # Convert scheme-based labels
    if scheme == 'sampler_centric':
        title_suffix = 'Sampler-Centric'
    else:
        title_suffix = 'Golden CDR'
    
    # Convert metrics format
    eye_metrics = {
        'eye_height': metrics.get('eye_height', 0),
        'eye_width': metrics.get('eye_width', 0),
    }
    
    fig = plot_eye_diagram(
        eye_data=eye_matrix,
        modulation='nrz',
        time_bins=time_bins,
        voltage_bins=voltage_bins,
        eye_metrics=eye_metrics,
        title=f'Statistical Eye Diagram ({title_suffix})',
        smooth_sigma=smooth_sigma
    )
    
    fig.set_size_inches(figsize[0], figsize[1])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _add_eye_boundaries(ax, scheme: str, ui_ps: float, x_range: tuple = None) -> None:
    """
    Add UI boundary lines and center markers to the plot.
    
    Args:
        ax: Matplotlib axes object
        scheme: 'sampler_centric' or 'golden_cdr'
        ui_ps: Unit interval in picoseconds
        x_range: Optional (xmin, xmax) to determine boundaries
    """
    if scheme == 'sampler_centric':
        # Sampler-Centric: 2-UI window [-1, +1]
        boundaries = [-1, -0.5, 0, 0.5, 1]
        for b in boundaries:
            x_line = b * ui_ps
            if b == 0:
                ax.axvline(x=x_line, color='red', linestyle='-', 
                          alpha=0.8, linewidth=1.5, label='Sampling moment')
            else:
                ax.axvline(x=x_line, color='gray', linestyle='--', 
                          alpha=0.4, linewidth=0.8)
    else:
        # Golden CDR: 2-UI centered window [-0.5, 1.5]
        # Eye center at 0.5 UI (center of display)
        boundaries = [-0.5, 0, 0.5, 1.0, 1.5]
        for b in boundaries:
            x_line = b * ui_ps
            if b == 0.5:
                # Eye center - ideal sampling point (red solid)
                ax.axvline(x=x_line, color='red', linestyle='-', 
                          alpha=0.8, linewidth=1.5, label='Eye center')
            elif b == 0 or b == 1.0:
                # UI boundaries (gray solid)
                ax.axvline(x=x_line, color='gray', linestyle='-', 
                          alpha=0.6, linewidth=1.0)
            else:
                # Window edges (gray dashed)
                ax.axvline(x=x_line, color='gray', linestyle='--', 
                          alpha=0.4, linewidth=0.8)


def _add_metrics_text(ax, metrics: Dict[str, Any], ui: float, scheme: str) -> None:
    """
    Add metrics text box to the plot (legacy version).
    """
    ui_ps = ui * 1e12
    data_rate = 1 / ui / 1e9
    
    lines = [
        f"Data Rate: {data_rate:.1f} Gbps",
        f"UI: {ui_ps:.1f} ps",
        f"Eye Height: {metrics.get('eye_height', 0)*1000:.2f} mV",
        f"Eye Width: {metrics.get('eye_width', 0):.3f} UI",
        f"Scheme: {scheme}",
    ]
    
    if scheme == 'golden_cdr':
        rj = metrics.get('rj_sigma', 0)
        dj = metrics.get('dj_pp', 0)
        if rj > 0 or dj > 0:
            lines.append(f"RJ: {rj*1e12:.2f} ps")
            lines.append(f"DJ: {dj*1e12:.2f} ps")
    
    textstr = '\n'.join(lines)
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')


def save_eye_diagram(scheme_obj, metrics: Dict[str, Any], output_path: str,
                     **kwargs) -> None:
    """
    Convenience function to save eye diagram from a scheme object.
    
    Args:
        scheme_obj: SamplerCentricScheme or GoldenCdrScheme object
        metrics: Analysis metrics dictionary
        output_path: Output file path
        **kwargs: Additional arguments passed to plot_eye_diagram_legacy
    """
    eye_matrix = scheme_obj.get_eye_matrix()
    if eye_matrix is None:
        raise ValueError("No eye matrix available. Run analyze() first.")
    
    plot_eye_diagram_legacy(
        eye_matrix=eye_matrix,
        xedges=scheme_obj.get_xedges(),
        yedges=scheme_obj.get_yedges(),
        ui=scheme_obj.ui,
        scheme=metrics.get('scheme', 'unknown'),
        metrics=metrics,
        output_path=output_path,
        **kwargs
    )


# Keep original function name for backward compatibility
# Map the old signature to legacy function
_original_plot_eye_diagram = plot_eye_diagram_legacy

# Export the new function as primary interface
__all__ = [
    'plot_eye_diagram',
    'plot_jtol_curve',
    'plot_bathtub_curve',
    'create_analysis_report',
    'create_matlab_colormap',
    'create_pam4_colormaps',
    'save_eye_diagram',
    # Legacy exports
    'plot_eye_diagram_legacy',
]
