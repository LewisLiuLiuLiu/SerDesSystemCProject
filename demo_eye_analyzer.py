#!/usr/bin/env python3
"""
EyeAnalyzer Demo Script

This script demonstrates how to use the EyeAnalyzer module.
It generates synthetic eye diagram data and performs analysis.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eye_analyzer import EyeAnalyzer

def generate_test_data(ui=2.5e-11, num_ui=10000, noise_level=0.01):
    """
    Generate synthetic eye diagram data.

    Args:
        ui: Unit interval in seconds
        num_ui: Number of UI to generate
        noise_level: Noise standard deviation in volts

    Returns:
        Tuple of (time_array, value_array)
    """
    # Generate time array
    time_array = np.arange(num_ui) * ui

    # Generate PRBS-like binary signal
    np.random.seed(42)
    value_array = np.random.choice([0.4, -0.4], size=num_ui)

    # Add noise
    value_array += np.random.normal(0, noise_level, size=num_ui)

    return time_array, value_array

def main():
    """Main demo function."""
    print("="*60)
    print("EyeAnalyzer Demo")
    print("="*60)

    # Parameters
    ui = 2.5e-11  # 10Gbps
    num_ui = 10000
    noise_level = 0.01

    print(f"\nGenerating test data:")
    print(f"  UI: {ui:.2e} s (10 Gbps)")
    print(f"  Number of UI: {num_ui}")
    print(f"  Noise level: {noise_level} V")

    # Generate test data
    time_array, value_array = generate_test_data(ui, num_ui, noise_level)

    print(f"\nData statistics:")
    print(f"  Time range: {time_array[0]:.2e} to {time_array[-1]:.2e} s")
    print(f"  Value range: {value_array.min():.3f} to {value_array.max():.3f} V")
    print(f"  Value mean: {value_array.mean():.3f} V")
    print(f"  Value std: {value_array.std():.3f} V")

    # Create analyzer
    print(f"\nInitializing EyeAnalyzer...")
    analyzer = EyeAnalyzer(ui=ui, ui_bins=128, amp_bins=128)

    # Perform analysis
    print(f"\nAnalyzing eye diagram...")
    metrics = analyzer.analyze(time_array, value_array)

    # Save results
    output_dir = 'demo_results'
    print(f"\nSaving results to '{output_dir}/'...")
    analyzer.save_results(metrics, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nEye Metrics:")
    print(f"  Eye Height: {metrics['eye_height']*1000:.2f} mV")
    print(f"  Eye Width:  {metrics['eye_width']:.3f} UI")
    print(f"\nOutput Files:")
    print(f"  📄 {os.path.join(output_dir, 'eye_metrics.json')}")
    print(f"  🖼️  {os.path.join(output_dir, 'eye_diagram.png')}")
    print("="*60)

if __name__ == "__main__":
    main()