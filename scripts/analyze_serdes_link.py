#!/usr/bin/env python3
"""
SerDes Link EyeAnalyzer Script

Analyzes waveform data from serdes_link_tb simulation.

Usage:
    python analyze_serdes_link.py [output_prefix]
    
Example:
    python analyze_serdes_link.py nrz_10g
    (默认前缀 nrz_10g；读取 build/<prefix>_*.csv 与 build/<prefix>_metadata.json)
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add eye_analyzer to path
try:
    from eye_analyzer import EyeAnalyzer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'eye_analyzer'))
    from eye_analyzer import EyeAnalyzer


def load_metadata(prefix):
    """Load simulation metadata JSON."""
    metadata_file = f"build/{prefix}_metadata.json"
    with open(metadata_file, 'r') as f:
        return json.load(f)


def analyze_waveform(prefix, signal_name, ui):
    """Analyze a single waveform file using eye_analyzer."""
    csv_file = f"build/{prefix}_{signal_name}.csv"
    
    print(f"\n{'='*60}")
    print(f"Analyzing {signal_name.upper()} waveform")
    print(f"{'='*60}")
    
    # Load data
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    time = data[:, 0]
    voltage = data[:, 1]
    
    print(f"Loaded {len(time)} samples")
    print(f"Time range: {time[0]*1e9:.3f} ns to {time[-1]*1e9:.3f} ns")
    print(f"Voltage range: {voltage.min()*1000:.2f} mV to {voltage.max()*1000:.2f} mV")
    
    # Create EyeAnalyzer
    analyzer = EyeAnalyzer(
        ui=ui,
        ui_bins=128,
        amp_bins=128,
        jitter_method='dual-dirac',
        output_image_dpi=300
    )
    
    # Analyze - handle jitter analysis failure gracefully
    try:
        metrics = analyzer.analyze(time, voltage)
        
        print(f"\nEye Metrics:")
        print(f"  Eye height: {metrics.get('eye_height', 0)*1000:.2f} mV")
        print(f"  Eye width:  {metrics.get('eye_width', 0)*1e12:.2f} ps")
        print(f"  Eye area:   {metrics.get('eye_area', 0)*1e12:.2f} mV*UI")
        
    except Exception as e:
        print(f"  Warning: {e}")
        print(f"  Continuing with eye diagram only...")
        
        # Manual eye diagram construction with 2-UI centered display
        phase_array = (time % ui) / ui
        
        # Build 2-UI centered eye diagram [-0.5, 1.5)
        phase_extended = []
        volt_extended = []
        for p, v in zip(phase_array, voltage):
            phase_extended.append(p)  # Original [0, 1)
            volt_extended.append(v)
            if p >= 0.5:  # Duplicate [0.5, 1) to [-0.5, 0)
                phase_extended.append(p - 1.0)
                volt_extended.append(v)
            if p < 0.5:   # Duplicate [0, 0.5) to [1, 1.5)
                phase_extended.append(p + 1.0)
                volt_extended.append(v)
        
        phase_extended = np.array(phase_extended)
        volt_extended = np.array(volt_extended)
        
        # 2-UI histogram with centered range
        hist2d, xedges, yedges = np.histogram2d(
            phase_extended, volt_extended,
            bins=[128, 128],
            range=[[-0.5, 1.5], [voltage.min()*1.1, voltage.max()*1.1]]
        )
        analyzer._hist2d = hist2d
        analyzer._xedges = xedges
        analyzer._yedges = yedges
        
        # Compute basic metrics
        eye_height = voltage.max() - voltage.min()
        metrics = {
            'eye_height': eye_height,
            'eye_width': 1.0,
            'eye_area': eye_height * 1.0
        }
        print(f"  Eye height: {eye_height*1000:.2f} mV")
    
    # Save eye diagram to output_eye/
    output_dir = Path("output_eye")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = output_dir / f'{prefix}_{signal_name}_eye.png'
    analyzer._plot_eye_diagram(metrics, str(image_path))
    print(f"  Saved eye diagram: {image_path}")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "nrz_10g"
    
    print(f"SerDes Link Analysis - Output prefix: {prefix}")
    
    # Load metadata (schema written by tb/top/nrz_link_tb)
    try:
        metadata = load_metadata(prefix)
        ui = metadata.get('ui_ps', 100.0) * 1e-12
        data_rate = metadata.get('data_rate_gbps', 10.0) * 1e9
        
        print(f"\nSimulation Parameters:")
        print(f"  Data rate: {data_rate/1e9:.2f} Gbps")
        print(f"  UI: {ui*1e12:.2f} ps")
        print(f"  Channel: {metadata.get('channel_method', 'unknown')}")
        
    except FileNotFoundError:
        print(f"Warning: Metadata file not found, using default UI=100ps")
        ui = 100e-12
    
    # Analyze each signal
    signals = ['tx', 'channel', 'dfe']
    
    for signal in signals:
        try:
            analyze_waveform(prefix, signal, ui)
        except FileNotFoundError:
            print(f"\nWarning: {prefix}_{signal}.csv not found, skipping...")
        except Exception as e:
            print(f"\nError analyzing {signal}: {e}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
