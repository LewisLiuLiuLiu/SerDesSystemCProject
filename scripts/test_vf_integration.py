#!/usr/bin/env python3
"""
Integration Test: Python VectorFittingPY -> C++ Channel workflow

Test flow:
1. Load S4P file (peters_01_0605_B12_thru.s4p)
2. Run VectorFittingPY to fit the data (12-16 poles)
3. Export state-space matrices to JSON
4. Validate correlation > 0.95
5. Check matrix dimensions are valid (no NaN/Inf)

Usage:
    python3 scripts/test_vf_integration.py
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Add scripts directory to path for importing vector_fitting
sys.path.insert(0, str(Path(__file__).parent))
from vector_fitting import VectorFitting


class VectorFittingPY:
    """
    Wrapper for VectorFitting with state-space export.
    Uses the existing vector_fitting.py implementation.
    """
    
    def __init__(self, order: int = 12):
        """
        Initialize VectorFittingPY.
        
        Args:
            order: Number of poles (fitting order), typically 12-16
        """
        self.order = order
        self.vf = None
        self.freq = None
        self.H_data = None
        self._fitted = False
        
        # Results
        self.poles = None
        self.residues = None
        self.d = 0.0
        self.e = 0.0
        self.correlation = 0.0
        self.rmse = 0.0
        
    def fit(self, s4p_file: str, port_pair: Tuple[int, int] = (1, 0),
            max_freq: float = 20e9, decimate: int = 8) -> Dict:
        """
        Fit S-parameter data using VectorFitting.
        
        Args:
            s4p_file: Path to S4P file
            port_pair: (output_port, input_port) tuple, default (1, 0) for S21
            max_freq: Maximum frequency for fitting (Hz)
            decimate: Decimation factor for faster fitting
            
        Returns:
            Dictionary with fitting results
        """
        try:
            import skrf
        except ImportError:
            raise ImportError("scikit-rf is required. Install with: pip install scikit-rf")
        
        print(f"\n[1] Loading S4P file: {s4p_file}")
        nw_full = skrf.Network(s4p_file)
        nports = nw_full.nports
        print(f"    Original: {len(nw_full.f)} points, "
              f"{nw_full.f[0]/1e9:.2f}-{nw_full.f[-1]/1e9:.2f} GHz, {nports} ports")
        
        # Decimate for faster fitting
        nw = nw_full[nw_full.f < max_freq][::decimate]
        print(f"    Decimated: {len(nw.f)} points for fitting (<{max_freq/1e9:.0f} GHz)")
        
        # Extract S-parameter
        out_port, in_port = port_pair
        self.H_data = nw.s[:, out_port, in_port]
        self.freq = nw.f
        
        # Run vector fitting using existing implementation
        print(f"\n[2] Running VectorFitting with {self.order} poles...")
        
        self.vf = VectorFitting(order=self.order, max_iterations=10, tolerance=1e-6)
        result = self.vf.fit(self.freq, self.H_data, remove_delay=True)
        
        # Store results
        self.poles = result['poles']
        self.residues = result['residues']
        self.d = result['d']
        self.e = result['h']  # h is proportional term in vector_fitting.py
        
        # Calculate correlation
        self._calculate_metrics()
        self._fitted = True
        
        print(f"    Fitting complete!")
        print(f"    Poles: {len(self.poles)}")
        print(f"    Constant d: {self.d:.6e}")
        print(f"    Proportional e: {self.e:.6e}")
        print(f"    Estimated delay: {result['delay']*1e12:.2f} ps")
        
        return {
            'poles': self.poles,
            'residues': self.residues,
            'd': self.d,
            'e': self.e,
            'correlation': self.correlation,
            'rmse': self.rmse,
            'order': self.order,
            'delay': result['delay']
        }
    
    def _calculate_metrics(self):
        """Calculate correlation and RMSE between original and fitted data."""
        # Evaluate fitted response
        H_fit = self.vf.evaluate(self.freq)
        
        # Calculate magnitude in dB
        H_mag = 20 * np.log10(np.abs(self.H_data) + 1e-20)
        H_fit_mag = 20 * np.log10(np.abs(H_fit) + 1e-20)
        
        # Correlation
        self.correlation = np.corrcoef(H_mag, H_fit_mag)[0, 1]
        
        # RMSE in dB
        self.rmse = np.sqrt(np.mean((H_mag - H_fit_mag)**2))
    
    def to_state_space(self, fs: float = 80e9) -> Dict:
        """
        Convert pole-residue form to state-space representation.
        
        State-space form:
            x_dot = A*x + B*u
            y = C*x + D*u + E*u_dot
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before to_state_space()")
        
        n = len(self.poles)
        
        # A matrix: diagonal matrix of poles
        A_complex = np.diag(self.poles)
        
        # Convert to JSON-serializable format (complex as dict)
        A = []
        for i in range(n):
            row = []
            for j in range(n):
                val = A_complex[i, j]
                row.append({'real': float(val.real), 'imag': float(val.imag)})
            A.append(row)
        
        # B matrix: column vector of ones
        B = []
        for i in range(n):
            B.append([{'real': 1.0, 'imag': 0.0}])
        
        # C matrix: row vector of residues
        C = [[]]
        for r in self.residues:
            C[0].append({'real': float(r.real), 'imag': float(r.imag)})
        
        # D matrix
        D = [[{'real': float(self.d), 'imag': 0.0}]]
        
        # E matrix (proportional term)
        E = [[{'real': float(self.e), 'imag': 0.0}]]
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'E': E,
            'order': n,
            'fs': fs
        }
    
    def validate_matrices(self, state_space: Dict) -> Dict:
        """Validate state-space matrices."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dimensions': {}
        }
        
        def to_complex_array(data):
            """Convert list of dicts to numpy complex array."""
            if isinstance(data, list):
                if len(data) == 0:
                    return np.array([])
                if isinstance(data[0], dict):
                    return np.array([d['real'] + 1j*d['imag'] for d in data])
                elif isinstance(data[0], list):
                    rows = []
                    for row in data:
                        row_vals = [d['real'] + 1j*d['imag'] for d in row]
                        rows.append(row_vals)
                    return np.array(rows)
            return np.array(data)
        
        A = to_complex_array(state_space['A'])
        B = to_complex_array(state_space['B'])
        C = to_complex_array(state_space['C'])
        D = to_complex_array(state_space['D'])
        E = to_complex_array(state_space['E'])
        
        n = A.shape[0]
        results['dimensions'] = {
            'A': A.shape,
            'B': B.shape,
            'C': C.shape,
            'D': D.shape,
            'E': E.shape
        }
        
        # Check dimensions
        if A.shape != (n, n):
            results['errors'].append(f"A matrix has wrong shape: {A.shape}")
        if B.shape != (n, 1):
            results['errors'].append(f"B matrix has wrong shape: {B.shape}")
        if C.shape != (1, n):
            results['errors'].append(f"C matrix has wrong shape: {C.shape}")
        if D.shape != (1, 1):
            results['errors'].append(f"D matrix has wrong shape: {D.shape}")
        if E.shape != (1, 1):
            results['errors'].append(f"E matrix has wrong shape: {E.shape}")
        
        # Check for NaN/Inf
        for name, matrix in [('A', A), ('B', B), ('C', C), ('D', D), ('E', E)]:
            if np.any(np.isnan(matrix)):
                results['errors'].append(f"{name} matrix contains NaN values")
            if np.any(np.isinf(matrix)):
                results['errors'].append(f"{name} matrix contains Inf values")
        
        # Check pole stability
        pole_real_parts = np.real(np.diag(A))
        unstable_count = np.sum(pole_real_parts > 0)
        if unstable_count > 0:
            results['warnings'].append(f"{unstable_count} poles are in RHP (unstable)")
        
        results['valid'] = len(results['errors']) == 0
        results['unstable_poles'] = int(unstable_count)
        
        return results
    
    def export_to_json(self, output_file: str, fs: float = 80e9) -> str:
        """Export state-space representation to JSON file."""
        state_space = self.to_state_space(fs)
        
        config = {
            'version': '2.1-vf',
            'method': 'state_space',
            'fs': fs,
            'state_space': {
                'A': state_space['A'],
                'B': state_space['B'],
                'C': state_space['C'],
                'D': state_space['D'],
                'E': state_space['E']
            },
            'metadata': {
                'order': self.order,
                'correlation': float(self.correlation),
                'rmse_db': float(self.rmse),
                'poles_count': len(self.poles)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[3] Exported state-space to: {output_file}")
        return output_file


def run_integration_test(s4p_file: str = None, 
                         output_file: str = 'test_config_state_space.json',
                         order: int = 12) -> Dict:
    """Run full integration test."""
    print("="*70)
    print("Integration Test: Python VF -> C++ Channel Workflow")
    print("="*70)
    
    # Find S4P file if not specified
    if s4p_file is None:
        s4p_files = list(Path('.').glob('*.s4p')) + list(Path('..').glob('*.s4p'))
        if not s4p_files:
            print("\n✗ ERROR: No S4P file found")
            sys.exit(1)
        s4p_file = str(s4p_files[0])
    
    s4p_path = Path(s4p_file)
    if not s4p_path.exists():
        print(f"\n✗ ERROR: S4P file not found: {s4p_file}")
        sys.exit(1)
    
    # Step 1 & 2: Load and fit
    vf = VectorFittingPY(order=order)
    try:
        fit_result = vf.fit(s4p_file)
    except Exception as e:
        print(f"\n✗ ERROR: Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Export state-space
    vf.export_to_json(output_file)
    
    # Step 4: Validate correlation > 0.95
    print(f"\n[4] Validating correlation...")
    print(f"    Correlation: {vf.correlation:.4f}")
    print(f"    Target: > 0.95")
    correlation_pass = vf.correlation > 0.95
    if correlation_pass:
        print(f"    ✓ PASS: Correlation > 0.95")
    else:
        print(f"    ✗ FAIL: Correlation <= 0.95")
    
    # Step 5: Validate matrix dimensions
    print(f"\n[5] Validating state-space matrices...")
    state_space = vf.to_state_space()
    validation = vf.validate_matrices(state_space)
    
    print(f"    Dimensions:")
    for name, dim in validation['dimensions'].items():
        print(f"      {name}: {dim}")
    
    if validation['valid']:
        print(f"    ✓ PASS: All matrices are valid (no NaN/Inf)")
    else:
        print(f"    ✗ FAIL: Matrix validation errors:")
        for error in validation['errors']:
            print(f"      - {error}")
    
    if validation['warnings']:
        print(f"    ⚠ WARNINGS:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
    
    # Summary
    all_pass = correlation_pass and validation['valid']
    
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"S4P File:           {s4p_file}")
    print(f"Fitting Order:      {order}")
    print(f"Correlation:        {vf.correlation:.4f} {'✓' if correlation_pass else '✗'}")
    print(f"RMSE:               {vf.rmse:.2f} dB")
    print(f"Matrix Validation:  {'✓ PASS' if validation['valid'] else '✗ FAIL'}")
    print(f"Unstable Poles:     {validation['unstable_poles']}")
    print(f"Output File:        {output_file}")
    print(f"{'='*70}")
    print(f"OVERALL:            {'✓ ALL TESTS PASS' if all_pass else '✗ SOME TESTS FAILED'}")
    print(f"{'='*70}")
    
    return {
        'success': all_pass,
        'correlation': float(vf.correlation),
        'correlation_pass': correlation_pass,
        'rmse_db': float(vf.rmse),
        'matrix_valid': validation['valid'],
        'unstable_poles': validation['unstable_poles'],
        'output_file': output_file,
        'dimensions': validation['dimensions']
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integration test for Python VF -> C++ Channel workflow'
    )
    parser.add_argument('--s4p', type=str, default=None,
                        help='Path to S4P file (default: auto-detect)')
    parser.add_argument('--output', type=str, default='test_config_state_space.json',
                        help='Output JSON file path')
    parser.add_argument('--order', type=int, default=12,
                        help='Number of poles for fitting (12-16, default: 12)')
    parser.add_argument('--max-freq', type=float, default=20e9,
                        help='Maximum frequency for fitting (Hz, default: 20e9)')
    
    args = parser.parse_args()
    
    # Validate order range
    if args.order < 12 or args.order > 16:
        print(f"Warning: Order {args.order} is outside recommended range (12-16)")
    
    result = run_integration_test(
        s4p_file=args.s4p,
        output_file=args.output,
        order=args.order
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)
