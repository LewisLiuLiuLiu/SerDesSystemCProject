#!/usr/bin/env python3
"""
Generate eye diagram for DFE output

Usage:
    python plot_dfe_eye.py [dfe_csv_file] [output_dir]
    
Examples:
    python plot_dfe_eye.py                                    # Auto-detect latest *_dfe.csv
    python plot_dfe_eye.py build/nrz_diff_full_dfe.csv        # Specific file
    python plot_dfe_eye.py build/nrz_diff_full_dfe.csv myout  # Custom output dir
"""
import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eye_analyzer import EyeAnalyzer, plot_eye_diagram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def find_latest_dfe_file():
    """Find the most recent *_dfe.csv file in build directory"""
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
    pattern = os.path.join(build_dir, '*_dfe.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    # Return most recently modified file
    return max(files, key=os.path.getmtime)

def resolve_ui_from_metadata(waveform_file):
    """Resolve UI (seconds) from <prefix>_metadata.json next to the CSV.

    2026-10: UI used to be hardcoded to 100 ps (10 Gbps), which distorts the
    eye for any other data rate. TBs write <prefix>_metadata.json with ui_ps.
    """
    import json
    base = os.path.basename(waveform_file)
    prefix = base.rsplit('_', 1)[0] if '_' in base else os.path.splitext(base)[0]
    meta = os.path.join(os.path.dirname(os.path.abspath(waveform_file)),
                        prefix + '_metadata.json')
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                d = json.load(f)
            if d.get('ui_ps'):
                print(f"  UI from metadata ({os.path.basename(meta)}): {d['ui_ps']:.1f} ps")
                return float(d['ui_ps']) * 1e-12
        except Exception as e:
            print(f"  Warning: failed to read {meta}: {e}")
    print("  Warning: UI metadata not found; falling back to 100 ps (10 Gbps)")
    return 100e-12


def main():
    # Optional --ui-ps override (consumed before positional parsing)
    ui_override = None
    if '--ui-ps' in sys.argv:
        k = sys.argv.index('--ui-ps')
        ui_override = float(sys.argv[k + 1]) * 1e-12
        del sys.argv[k:k + 2]

    # Parse arguments
    if len(sys.argv) > 1:
        dfe_file = sys.argv[1]
    else:
        dfe_file = find_latest_dfe_file()
        if dfe_file:
            print(f"Auto-detected: {dfe_file}")
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output_eye'
    
    # Check if file exists
    if not dfe_file or not os.path.exists(dfe_file):
        print(f"Error: DFE file not found!")
        print("Usage: python plot_dfe_eye.py [dfe_csv_file] [output_dir]")
        print("Please run the simulation first: ./bin/nrz_link_tb")
        return 1
    
    print("="*60)
    print("DFE Output Eye Diagram Analysis")
    print("="*60)
    print(f"Input file: {dfe_file}")
    
    # Load waveform data
    print("\nLoading waveform data...")
    df = pd.read_csv(dfe_file)
    time_array = df['time_s'].values
    
    # Support both old format (voltage_v) and new format (voltage_diff)
    if 'voltage_diff' in df.columns:
        voltage_array = df['voltage_diff'].values
        print("  Using differential voltage (voltage_diff)")
    elif 'voltage_v' in df.columns:
        voltage_array = df['voltage_v'].values
        print("  Using single-ended voltage (voltage_v)")
    else:
        print(f"Error: No voltage column found. Available columns: {list(df.columns)}")
        return 1
    
    print(f"  Total samples: {len(time_array)}")
    print(f"  Time range: {time_array[0]:.3e} s to {time_array[-1]:.3e} s")
    print(f"  Duration: {time_array[-1] - time_array[0]:.3e} s")
    print(f"  Voltage range: {voltage_array.min():.3f} V to {voltage_array.max():.3f} V")
    
    # UI: metadata-driven (2026-10), with --ui-ps override
    ui = ui_override if ui_override else resolve_ui_from_metadata(dfe_file)
    
    print(f"\nAnalyzing eye diagram...")
    print(f"  UI: {ui*1e12:.1f} ps ({1/ui/1e9:.1f} Gbps)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer and run analysis
    analyzer = EyeAnalyzer(
        ui=ui,
        modulation='nrz',
        mode='empirical',  # Use empirical mode for waveform-based analysis
        sampling='phase-lock',
        interpolate_factor=16,  # 16x interpolation for smooth transitions
        center_eye=True,  # Center the eye pattern to show complete transitions
        n_ui_display=2.0,  # Display 2 UI to show both left and right transitions
        use_bresenham=True,  # Use Bresenham rasterization for continuous eye traces
    )
    
    # Analyze eye diagram (empirical mode requires tuple input)
    metrics = analyzer.analyze(
        (time_array, voltage_array),
        target_ber=1e-12
    )
    
    # Plot and save eye diagram
    output_file = os.path.join(output_dir, 'eye_diagram.png')
    
    # Get eye diagram data from metrics
    if 'eye_matrix' in metrics:
        eye_data = metrics['eye_matrix']
        time_bins = metrics.get('xedges', np.linspace(-1, 1, eye_data.shape[1] + 1))
        voltage_bins = metrics.get('yedges', np.linspace(-0.3, 0.3, eye_data.shape[0] + 1))
        eye_metrics = metrics.get('eye_metrics', {})
        
        plot_eye_diagram(
            eye_data,
            modulation='nrz',
            time_bins=time_bins,
            voltage_bins=voltage_bins,
            eye_metrics=eye_metrics,
            title=f'DFE Output Eye Diagram ({1/ui/1e9:.1f} Gbps NRZ)',
            smooth_sigma=0.0  # No smoothing for Bresenham rasterized eye to preserve sharp transitions
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nEye diagram saved to: {output_file}")
    else:
        print("Warning: Could not extract eye diagram data")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    # Extract metrics from nested structure
    eye_m = metrics.get('eye_metrics', {})
    jitter_m = metrics.get('jitter', {})
    
    print(f"\nEye Metrics:")
    print(f"  Eye Height:     {eye_m.get('eye_height', 0)*1000:.2f} mV")
    print(f"  Eye Width:      {eye_m.get('eye_width', 0):.3f} UI ({eye_m.get('eye_width', 0)*ui*1e12:.2f} ps)")
    print(f"  Eye Area:       {eye_m.get('eye_area', 0)*1000:.2f} mV*UI")
    print(f"\nJitter Metrics:")
    print(f"  RJ (sigma):     {jitter_m.get('rj_sigma', 0)*1e12:.2f} ps")
    print(f"  DJ (pp):        {jitter_m.get('dj_pp', 0)*1e12:.2f} ps")
    print(f"  TJ @ 1e-12:     {jitter_m.get('tj_at_ber', 0)*1e12:.2f} ps")
    print(f"\nSignal Quality:")
    print(f"  Voltage Range:  {voltage_array.min()*1000:.2f} mV to {voltage_array.max()*1000:.2f} mV")
    print(f"  Peak-to-peak:   {(voltage_array.max() - voltage_array.min())*1000:.2f} mV")
    
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"  - eye_diagram.png")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
