#!/usr/bin/env python3
"""
EyeAnalyzer Command Line Interface

A command-line tool for analyzing eye diagrams from SystemC-AMS simulation output.

Usage:
    python scripts/analyze_eye.py <input_file> --ui <ui_value> [options]

Example:
    python scripts/analyze_eye.py build/bin/simple_link.dat --ui 2.5e-11
    python scripts/analyze_eye.py build/tb/ctle/ctle_tran_prbs.csv --ui 2.5e-11 --output-dir results/
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eye_analyzer import EyeAnalyzer, auto_load_waveform


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='EyeAnalyzer - SerDes Link Eye Diagram Analysis Tool (Basic Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze SystemC-AMS tabular output
  python scripts/analyze_eye.py build/bin/simple_link.dat --ui 2.5e-11

  # Analyze CSV output with custom resolution
  python scripts/analyze_eye.py build/tb/ctle/ctle_tran_prbs.csv --ui 2.5e-11 --ui-bins 256 --amp-bins 256

  # Specify output directory
  python scripts/analyze_eye.py results.dat --ui 2.5e-11 --output-dir analysis_results/
        """
    )

    # Required arguments
    parser.add_argument('input_file', help='Input file path (.dat or .csv format)')

    # Required parameters
    parser.add_argument('--ui', type=float, required=True,
                       help='Unit interval in seconds (e.g., 2.5e-11 for 10Gbps)')

    # Optional parameters
    parser.add_argument('--ui-bins', type=int, default=128,
                       help='Phase axis resolution (default: 128)')
    parser.add_argument('--amp-bins', type=int, default=128,
                       help='Amplitude axis resolution (default: 128)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for results (default: current directory)')

    # Parse arguments
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    print("="*60)
    print("EyeAnalyzer - SerDes Link Eye Diagram Analysis Tool")
    print("="*60)
    print(f"\nInput file: {args.input_file}")
    print(f"UI: {args.ui:.2e} s")
    print(f"Resolution: {args.ui_bins} x {args.amp_bins}")
    print(f"Output directory: {args.output_dir}")

    try:
        # Load waveform data
        print(f"\n📊 Loading waveform data...")
        time_array, value_array = auto_load_waveform(args.input_file)
        print(f"  Loaded {len(time_array)} samples")
        print(f"  Time range: {time_array[0]:.2e} to {time_array[-1]:.2e} s")
        print(f"  Value range: {value_array.min():.3f} to {value_array.max():.3f} V")

        # Create analyzer
        print(f"\n🔬 Initializing EyeAnalyzer...")
        analyzer = EyeAnalyzer(
            ui=args.ui,
            ui_bins=args.ui_bins,
            amp_bins=args.amp_bins
        )

        # Perform analysis
        print(f"\n📈 Analyzing eye diagram...")
        metrics = analyzer.analyze(time_array, value_array)

        # Save results
        print(f"\n💾 Saving results...")
        analyzer.save_results(metrics, args.output_dir)

        # Print summary
        print("\n" + "="*60)
        print("✅ Analysis Complete!")
        print("="*60)
        print(f"\nResults Summary:")
        print(f"  Eye Height: {metrics['eye_height']*1000:.2f} mV")
        print(f"  Eye Width:  {metrics['eye_width']:.3f} UI")
        print(f"\nOutput Files:")
        print(f"  📄 {os.path.join(args.output_dir, 'eye_metrics.json')}")
        print(f"  🖼️  {os.path.join(args.output_dir, 'eye_diagram.png')}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()