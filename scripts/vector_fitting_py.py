"""
Vector Fitting algorithm implementation (Relaxed VF) for S-parameter fitting.

Based on vectfit3.m by Bjorn Gustavsen (SINTEF Energy Research).
This is a clean implementation of the Fast Relaxed Vector Fitting algorithm.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorFitting:
    """
    Vector Fitting algorithm with relaxed pole identification.
    
    The algorithm approximates frequency response H(s) as:
    H(s) = sum(r_k / (s - p_k)) + d + s*e
    
    using the relaxed Vector Fitting method which ensures better
    convergence by avoiding the trivial solution.
    """
    
    def __init__(self, order: int = 8, max_iterations: int = 10, 
                 tolerance: float = 1e-6, relax: bool = True,
                 stable: bool = True, asymp: int = 2):
        """
        Initialize Vector Fitting.
        
        Args:
            order: Number of poles (fitting order)
            max_iterations: Maximum VF iterations
            tolerance: Convergence tolerance for poles
            relax: Use relaxed nontriviality constraint
            stable: Force unstable poles to left half-plane
            asymp: Asymptotic behavior (1=D=0,E=0; 2=D!=0,E=0; 3=D!=0,E!=0)
        """
        self.order = order
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.relax = relax
        self.stable = stable
        self.asymp = asymp
        
        # Results
        self.poles = None
        self.residues = None
        self.d = 0.0
        self.e = 0.0
        
        # Normalization
        self._s_scale = 1.0
        
    def _initialize_poles(self, s: np.ndarray) -> np.ndarray:
        """
        Initialize starting poles distributed across frequency range.
        
        Args:
            s: Complex frequency array (rad/s)
            
        Returns:
            Initial pole array (complex)
        """
        s_min = np.min(np.abs(s))
        s_max = np.max(np.abs(s))
        
        # Store normalization factor
        self._s_scale = s_max
        
        # Create complex conjugate pairs
        n_pairs = self.order // 2
        n_real = self.order % 2
        
        poles = []
        
        if n_pairs > 0:
            # Log-spaced frequencies
            f_start = s_min / (2 * np.pi * 10)  # Start one decade below
            f_end = s_max / (2 * np.pi)
            
            freqs = np.logspace(np.log10(f_start), np.log10(f_end), n_pairs)
            
            for f in freqs:
                omega = 2 * np.pi * f
                sigma = -omega * 0.1  # 10% damping
                poles.append(sigma + 1j * omega)
                poles.append(sigma - 1j * omega)
        
        if n_real > 0:
            f_mid = np.sqrt(s_min * s_max) / (2 * np.pi)
            poles.append(-2 * np.pi * f_mid * 0.1)
        
        return np.array(poles, dtype=complex)
    
    def _build_cindex(self, poles: np.ndarray) -> np.ndarray:
        """
        Build cindex array to identify complex conjugate pairs.
        
        Args:
            poles: Pole array
            
        Returns:
            cindex: 0=real, 1=complex first, 2=complex second
        """
        N = len(poles)
        cindex = np.zeros(N, dtype=int)
        
        for m in range(N):
            if np.imag(poles[m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m-1] == 0 or cindex[m-1] == 2:
                        cindex[m] = 1
                        if m + 1 < N:
                            cindex[m+1] = 2
                    else:
                        cindex[m] = 2
        
        return cindex
    
    def _build_Dk(self, s: np.ndarray, poles: np.ndarray, 
                  cindex: np.ndarray) -> np.ndarray:
        """
        Build Dk matrix for pole identification.
        
        Dk contains basis functions: 1/(s - p_k) for real poles,
        and separated real/imaginary parts for complex poles.
        
        Args:
            s: Complex frequency array
            poles: Current poles
            cindex: Complex index array
            
        Returns:
            Dk matrix [Ns, N+offs] where offs depends on asymp setting
        """
        Ns = len(s)
        N = len(poles)
        
        # Determine offset for asymptotic terms
        if self.asymp == 1:
            offs = 0
        elif self.asymp == 2:
            offs = 1  # D term
        else:  # asymp == 3
            offs = 2  # D + E*s terms
        
        Dk = np.zeros((Ns, N + offs), dtype=complex)
        
        # Build pole basis functions
        for m in range(N):
            if cindex[m] == 0:  # Real pole
                Dk[:, m] = 1.0 / (s - poles[m])
            elif cindex[m] == 1:  # Complex pole, first part
                Dk[:, m] = 1.0 / (s - poles[m]) + 1.0 / (s - np.conj(poles[m]))
                Dk[:, m+1] = 1j / (s - poles[m]) - 1j / (s - np.conj(poles[m]))
            # cindex[m] == 2 is handled with m+1 above
        
        # Add asymptotic terms
        if self.asymp == 2:
            Dk[:, N] = 1.0
        elif self.asymp == 3:
            Dk[:, N] = 1.0
            Dk[:, N+1] = s
        
        return Dk
    
    def _pole_identification(self, s: np.ndarray, H: np.ndarray,
                            poles: np.ndarray, weight: np.ndarray,
                            opts: Dict) -> np.ndarray:
        """
        Pole identification using relaxed Vector Fitting.
        
        This is the core pole relocation step. It solves for sigma(s)
        such that sigma(s)*H(s) is approximated by a rational function
        with the same poles. The zeros of sigma become the new poles.
        
        Algorithm:
        1. Build Dk matrix using current poles
        2. Build linear system with relaxed nontriviality constraint
        3. QR decomposition
        4. Solve for sigma coefficients
        5. Compute zeros of sigma (new poles)
        6. Stabilize if requested
        
        Args:
            s: Complex frequency array (rad/s)
            H: Frequency response (Nc, Ns)
            poles: Current poles
            weight: Weight array
            opts: Options dict
            
        Returns:
            new_poles: Relocated poles
        """
        Ns = len(s)
        N = len(poles)
        Nc = H.shape[0] if H.ndim > 1 else 1
        
        if H.ndim == 1:
            H = H.reshape(1, -1)
        
        # Build cindex
        cindex = self._build_cindex(poles)
        
        # Build Dk matrix
        Dk = self._build_Dk(s, poles, cindex)
        
        # Determine offset
        if self.asymp == 1:
            offs = 0
        elif self.asymp == 2:
            offs = 1
        else:
            offs = 2
        
        # Common weight flag
        common_weight = weight.ndim == 1 or weight.shape[0] == 1
        
        # Scale for relaxation (from vectfit3.m line 304-312)
        scale = 0.0
        for m in range(Nc):
            if common_weight:
                scale += np.linalg.norm(weight * H[m, :])**2
            else:
                scale += np.linalg.norm(weight[m, :] * H[m, :])**2
        scale = np.sqrt(scale) / Ns
        
        # Tolerance thresholds
        TOLlow = 1e-18
        TOLhigh = 1e18
        
        # Relaxed pole identification
        if opts.get('relax', True):
            AA = np.zeros((Nc * (N + 1), N + 1))
            bb = np.zeros(Nc * (N + 1))
            
            for n in range(Nc):
                # Build system matrix A
                A = np.zeros((Ns, (N + offs) + (N + 1)), dtype=complex)
                
                # Select weight
                if common_weight:
                    weig = weight
                else:
                    weig = weight[n, :] if weight.ndim > 1 else weight
                
                # Left block: Dk for H fitting
                for m in range(N + offs):
                    A[:, m] = weig * Dk[:, m]
                
                # Right block: -H*Dk for sigma fitting
                inda = N + offs
                for m in range(N + 1):
                    A[:, inda + m] = -weig * Dk[:, m] * H[n, :]
                
                # Stack real and imaginary parts
                A_real = np.vstack([A.real, A.imag])
                
                # Add relaxation constraint (integral criterion for sigma)
                # This ensures sigma is not identically zero
                if n == Nc - 1:
                    offset_row = 2 * Ns
                    A_real = np.vstack([A_real, np.zeros(N + offs + N + 1)])
                    for mm in range(N + 1):
                        A_real[offset_row, inda + mm] = scale * np.sum(Dk[:, mm].real)
                
                # QR decomposition
                Q, R = np.linalg.qr(A_real, mode='reduced')
                
                # Extract R22 block (corresponding to sigma coefficients)
                ind1 = N + offs
                ind2 = N + offs + N + 1
                R22 = R[ind1:ind2, ind1:ind2]
                
                AA[n*(N+1):(n+1)*(N+1), :] = R22
                
                # Right-hand side from relaxation
                if n == Nc - 1:
                    bb[n*(N+1):(n+1)*(N+1)] = Q[-1, ind1:ind2] * Ns * scale
            
            # Column scaling for better conditioning
            Escale = np.zeros(AA.shape[1])
            for col in range(AA.shape[1]):
                col_norm = np.linalg.norm(AA[:, col])
                if col_norm > 1e-15:
                    Escale[col] = 1.0 / col_norm
                    AA[:, col] *= Escale[col]
            
            # Solve least squares
            x = np.linalg.lstsq(AA, bb, rcond=None)[0]
            x = x * Escale
        else:
            x = np.zeros(N + 1)
            x[-1] = 1.0  # D = 1
        
        # Check if relaxation produced bad D value
        Dnew = x[-1]
        if not opts.get('relax', True) or abs(Dnew) < TOLlow or abs(Dnew) > TOLhigh:
            # Solve without relaxation, fixing D
            if not opts.get('relax', True):
                Dnew = 1.0
            elif abs(Dnew) < TOLlow:
                Dnew = np.sign(Dnew) * TOLlow if Dnew != 0 else TOLlow
            elif abs(Dnew) > TOLhigh:
                Dnew = np.sign(Dnew) * TOLhigh
            
            AA = np.zeros((Nc * N, N))
            bb = np.zeros(Nc * N)
            
            for n in range(Nc):
                A = np.zeros((Ns, (N + offs) + N), dtype=complex)
                
                if common_weight:
                    weig = weight
                else:
                    weig = weight[n, :] if weight.ndim > 1 else weight
                
                # Left block
                for m in range(N + offs):
                    A[:, m] = weig * Dk[:, m]
                
                # Right block (only N columns now, excluding D)
                inda = N + offs
                for m in range(N):
                    A[:, inda + m] = -weig * Dk[:, m] * H[n, :]
                
                b = Dnew * weig * H[n, :]
                
                # Stack real and imaginary
                A_real = np.vstack([A.real, A.imag])
                b_real = np.hstack([b.real, b.imag])
                
                # QR decomposition
                Q, R = np.linalg.qr(A_real, mode='reduced')
                
                ind1 = N + offs
                ind2 = N + offs + N
                R22 = R[ind1:ind2, ind1:ind2]
                AA[n*N:(n+1)*N, :] = R22
                bb[n*N:(n+1)*N] = Q[:, ind1:ind2].T @ b_real
            
            # Column scaling
            Escale = np.zeros(AA.shape[1])
            for col in range(AA.shape[1]):
                col_norm = np.linalg.norm(AA[:, col])
                if col_norm > 1e-15:
                    Escale[col] = 1.0 / col_norm
                    AA[:, col] *= Escale[col]
            
            # Solve
            x = np.linalg.lstsq(AA, bb, rcond=None)[0]
            x = x * Escale
            x = np.append(x, Dnew)
        
        # Extract sigma coefficients
        C = x[:-1]  # Residues for sigma
        D = x[-1]   # Direct term for sigma
        
        # Convert C back to complex representation
        C_complex = np.zeros(N, dtype=complex)
        for m in range(N):
            if cindex[m] == 0:
                C_complex[m] = C[m]
            elif cindex[m] == 1:
                r1 = C[m]
                r2 = C[m+1]
                C_complex[m] = r1 + 1j * r2
                if m + 1 < N:
                    C_complex[m+1] = r1 - 1j * r2
            # cindex[m] == 2 is handled with m-1
        
        # Build state-space for sigma and compute zeros
        # ZER = LAMBD - B*C.T/D
        LAMBD = np.diag(poles)
        B = np.ones(N)
        
        # Modify LAMBD and B for complex pairs (real form)
        for n in range(N):
            if n < N - 1 and cindex[n] == 1:
                # Convert complex pair to 2x2 real block
                real_part = np.real(poles[n])
                imag_part = np.imag(poles[n])
                LAMBD[n, n] = real_part
                LAMBD[n, n+1] = imag_part
                LAMBD[n+1, n] = -imag_part
                LAMBD[n+1, n+1] = real_part
                B[n] = 2
                B[n+1] = 0
                # C should be real for first, imag for second
                C_complex[n] = np.real(C_complex[n])
                C_complex[n+1] = np.imag(C_complex[n])
        
        # Compute zeros: ZER = A - B*C/D
        ZER = LAMBD - np.outer(B, C_complex) / D
        
        # New poles are eigenvalues of ZER
        new_poles = np.linalg.eigvals(ZER)
        
        # Stabilize: move unstable poles to left half-plane
        if opts.get('stable', True):
            unstables = np.real(new_poles) > 0
            new_poles[unstables] -= 2 * np.real(new_poles[unstables])
        
        # Sort: real poles first, then complex pairs
        new_poles = self._sort_poles(new_poles)
        
        return new_poles
    
    def _sort_poles(self, poles: np.ndarray) -> np.ndarray:
        """
        Sort poles: real poles first, then complex by imaginary part.
        
        Args:
            poles: Unsorted poles
            
        Returns:
            Sorted poles
        """
        # Separate real and complex
        real_mask = np.abs(np.imag(poles)) < 1e-10 * (np.abs(poles) + 1e-10)
        real_poles = np.real(poles[real_mask])
        complex_poles = poles[~real_mask]
        
        # Sort real poles
        real_poles = np.sort(real_poles)
        
        # Sort complex poles by imaginary part
        if len(complex_poles) > 0:
            imag_parts = np.imag(complex_poles)
            idx = np.argsort(np.abs(imag_parts))
            complex_poles = complex_poles[idx]
        
        # Combine: real first, then complex
        result = np.concatenate([real_poles, complex_poles])
        
        return result.astype(complex)
    
    def fit(self, s: np.ndarray, H: np.ndarray, 
            weight: Optional[np.ndarray] = None) -> Dict:
        """
        Fit frequency response data using Vector Fitting.
        
        Args:
            s: Complex frequency array (rad/s)
            H: Frequency response (Ns,) or (Nc, Ns)
            weight: Optional weight array
            
        Returns:
            Dictionary with poles, residues, d, e, etc.
        """
        logger.info(f"Starting Vector Fitting with order={self.order}")
        
        # Ensure H is 2D
        if H.ndim == 1:
            H = H.reshape(1, -1)
        
        Ns = len(s)
        Nc = H.shape[0]
        
        # Default weights
        if weight is None:
            weight = np.ones(Ns)
        
        # Initialize poles
        self.poles = self._initialize_poles(s)
        logger.debug(f"Initial poles: {self.poles}")
        
        # Build options dict
        opts = {
            'relax': self.relax,
            'stable': self.stable,
            'asymp': self.asymp
        }
        
        # Iterative pole relocation
        for iteration in range(self.max_iterations):
            # Pole identification
            new_poles = self._pole_identification(s, H, self.poles, weight, opts)
            
            # Check convergence
            pole_change = np.max(np.abs(new_poles - self.poles))
            rel_change = pole_change / (np.max(np.abs(self.poles)) + 1e-10)
            
            logger.debug(f"Iteration {iteration + 1}: rel_change = {rel_change:.2e}")
            
            self.poles = new_poles
            
            if rel_change < self.tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Final residue identification (to be implemented)
        # For now, use simple least squares
        self.residues, self.d, self.e = self._solve_residues(s, H[0, :])
        
        return {
            'poles': self.poles,
            'residues': self.residues,
            'd': self.d,
            'e': self.e,
            'order': self.order
        }
    
    def _solve_residues(self, s: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Solve for residues given fixed poles.
        
        Args:
            s: Complex frequency array
            H: Frequency response
            
        Returns:
            residues, d, e
        """
        N = len(self.poles)
        Ns = len(s)
        
        # Build system matrix
        A = np.zeros((Ns, N + 2), dtype=complex)
        
        for k, p in enumerate(self.poles):
            A[:, k] = 1.0 / (s - p)
        
        A[:, N] = 1.0  # d term
        A[:, N+1] = s  # e term
        
        # Stack real and imaginary
        A_real = np.vstack([A.real, A.imag])
        b_real = np.hstack([H.real, H.imag])
        
        # Solve
        x = np.linalg.lstsq(A_real, b_real, rcond=None)[0]
        
        residues = x[:N]
        d = x[N]
        e = x[N+1]
        
        return residues, d, e


def fit_vector_fitting(s: np.ndarray, H: np.ndarray, order: int = 8,
                       max_iterations: int = 10, tolerance: float = 1e-6,
                       **kwargs) -> Dict:
    """
    Convenience function for Vector Fitting.
    
    Args:
        s: Complex frequency array (rad/s)
        H: Frequency response
        order: Fitting order
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        **kwargs: Additional options for VectorFitting
        
    Returns:
        Fitting results dictionary
    """
    vf = VectorFitting(order=order, max_iterations=max_iterations,
                       tolerance=tolerance, **kwargs)
    return vf.fit(s, H)
