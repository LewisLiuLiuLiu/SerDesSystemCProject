#!/usr/bin/env python3
"""
Plot DFE taps evolution over time

Usage:
    python plot_dfe_taps.py [dfe_taps_csv_file] [output_dir]
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parse arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'build/nrz_10g_dfe_taps.csv'
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output_eye'
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return 1
    
    print("="*60)
    print("DFE Taps Evolution Plot")
    print("="*60)
    print(f"Input file: {csv_file}")
    
    # Load data
    print("\nLoading DFE taps data...")
    df = pd.read_csv(csv_file)
    
    time_us = df['time_s'].values * 1e6  # Convert to us
    tap1 = df['tap1'].values
    tap2 = df['tap2'].values
    tap3 = df['tap3'].values
    tap4 = df['tap4'].values
    tap5 = df['tap5'].values
    
    print(f"  Total samples: {len(time_us)}")
    print(f"  Time range: {time_us[0]:.3f} us to {time_us[-1]:.3f} us")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DFE Taps Evolution (10 Gbps NRZ)', fontsize=14, fontweight='bold')
    
    taps = [tap1, tap2, tap3, tap4, tap5]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot individual taps
    for i, (tap, color) in enumerate(zip(taps, colors)):
        ax = axes[i // 3, i % 3]
        ax.plot(time_us, tap, color=color, linewidth=0.8)
        ax.set_xlabel('Time [us]')
        ax.set_ylabel(f'Tap {i+1} Coefficient')
        ax.set_title(f'Tap {i+1}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Show final value
        final_val = tap[-1]
        ax.axhline(y=final_val, color='r', linestyle=':', alpha=0.5, label=f'Final: {final_val:.4f}')
        ax.legend(loc='best', fontsize=8)
    
    # Plot all taps together in the last subplot
    ax = axes[1, 2]
    for i, (tap, color) in enumerate(zip(taps, colors)):
        ax.plot(time_us, tap, color=color, linewidth=0.8, label=f'Tap {i+1}')
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('Tap Coefficient')
    ax.set_title('All Taps Combined')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'dfe_taps_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    # Print final values
    print("\n" + "="*60)
    print("Final Tap Coefficients:")
    print("="*60)
    for i, tap in enumerate(taps):
        print(f"  Tap {i+1}: {tap[-1]:.6f}")
    
    # Print convergence info (skip first 10% for steady state)
    steady_start = int(len(tap1) * 0.1)
    print("\n" + "="*60)
    print("Convergence Analysis (last 90% of simulation):")
    print("="*60)
    for i, tap in enumerate(taps):
        steady_state = tap[steady_start:]
        mean_val = np.mean(steady_state)
        std_val = np.std(steady_state)
        print(f"  Tap {i+1}: Mean = {mean_val:.6f}, Std = {std_val:.6f}")
    
    plt.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
