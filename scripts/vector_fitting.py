"""
Vector Fitting algorithm implementation (Relaxed VF) for S-parameter fitting.

Based on vectfit3.m by Bjorn Gustavsen (SINTEF Energy Research).
This is a clean implementation of the Fast Relaxed Vector Fitting algorithm.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def estimate_delay(s: np.ndarray, H: np.ndarray) -> float:
    """
    Estimate a pure transport delay from the unwrapped phase.

    Real channels carry a delay tau (ns-scale) that a low-order rational fit
    cannot represent: the phase rotates by many multiples of 2*pi, and fitting
    the delayed response directly collapses the magnitude LS (2026-10 finding:
    a 0.5ns delay already destroys an 8-pole fit). The standard remedy is to
    de-embed the delay before VF and keep it as an explicit parameter.

    Weighted linear fit of phi(w) ~= phi0 - tau*w with weight |H|^2, restricted
    to points well above the noise floor (deep notches are excluded).

    Returns:
        Delay in seconds (>= 0).
    """
    w = np.asarray(s).imag
    mag = np.abs(H)
    if len(w) < 4 or not np.all(np.isfinite(mag)):
        return 0.0
    phase = np.unwrap(np.angle(H))
    # Exclude deep notches / noise floor
    mask = mag > max(1e-2 * np.max(mag), 1e-30)
    if np.count_nonzero(mask) < 4:
        mask = np.ones_like(mag, dtype=bool)
    wgt = mag[mask] ** 2
    # Weighted linear least squares: phase = a*w + b, tau = -a
    A = np.vstack([w[mask], np.ones(np.count_nonzero(mask))]).T
    Aw = A * np.sqrt(wgt)[:, None]
    bw = phase[mask] * np.sqrt(wgt)
    a, _b = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    tau = float(-a)
    return max(tau, 0.0)


def poles_residues_to_tf(poles: np.ndarray, residues: np.ndarray,
                         d: float = 0.0, e: float = 0.0
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert pole-residue form H(s) = sum(r_k/(s-p_k)) + d + s*e
    to normalized transfer-function polynomials.

    To keep coefficients numerically well-conditioned at high orders (raw
    coefficient form overflows double around n>=40 for mm-wave poles), the
    polynomials are expressed in the normalized variable u = s / s_scale:

        H_rational(s) = polyval(num, u) / polyval(den, u)

    Returns (num, den, s_scale) with real coefficients, highest degree first.
    The transport delay (if any) is NOT included here — keep it separately.
    """
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    n = len(poles)

    scale = float(np.max(np.abs(poles))) if n > 0 else 1.0
    if scale <= 0.0:
        scale = 1.0
    ps = poles / scale
    den_s = np.poly(ps) if n > 0 else np.array([1.0 + 0j])

    # Degree-(n+1) array (highest first) for:
    #   scale^(n-1) * [ sum_i r_i prod_{j!=i}(u-p'_j)
    #                   + d*scale*den_s(u) + e*scale^2*u*den_s(u) ]
    Np = np.zeros(n + 2, dtype=complex)
    sum_r = np.zeros(max(n, 1), dtype=complex)   # degree n-1
    for i in range(n):
        others = np.delete(ps, i)
        prod = np.poly(others) if len(others) > 0 else np.array([1.0 + 0j])
        sum_r += residues[i] * prod
    Np[2:] += sum_r                    # deg n-1 -> indices 2..n+1
    Np[1:] += (d * scale) * den_s      # deg n   -> indices 1..n+1
    if e != 0.0:
        Np += (e * scale * scale) * np.concatenate((den_s, [0.0]))  # deg n+1

    num_c = Np / scale                 # H = num_c(u)/den_s(u)
    imag_ratio = np.max(np.abs(num_c.imag)) / (np.max(np.abs(num_c.real)) + 1e-30)
    if imag_ratio > 1e-6:
        logger.warning(f"num coefficients not conjugate-symmetric: imag ratio {imag_ratio:.2e}")

    num = num_c.real
    # Trim a leading zero when e == 0 (num degree is then n, not n+1)
    if abs(num[0]) < 1e-30 and e == 0.0:
        num = num[1:]
    return num, den_s.real, scale


def evaluate_fit(s: np.ndarray, result: Dict, delay: Optional[float] = None
                 ) -> np.ndarray:
    """
    Evaluate a VF result at complex frequencies s (rad/s).

    H(s) = sum(r_k/(s-p_k)) + d + e*s, multiplied by exp(-s*delay) when a
    transport delay is present (delay=0 disables it).
    """
    s = np.asarray(s, dtype=complex)
    Hm = np.full(s.shape, complex(result.get('d', 0.0)))
    if result.get('e', 0.0) != 0.0:
        Hm = Hm + result['e'] * s
    for p, r in zip(result['poles'], result['residues']):
        Hm = Hm + r / (s - p)
    tau = result.get('delay', 0.0) if delay is None else delay
    if tau:
        Hm = Hm * np.exp(-s * tau)
    return Hm


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
        self.delay = 0.0

        # Normalization
        self._s_scale = 1.0

    def _initialize_poles(self, s: np.ndarray) -> np.ndarray:
        """
        Initialize starting poles distributed across frequency range.

        Based on vectfit3 ex4a.m: linearly spaced complex conjugate pairs
        with small damping (1% of frequency).

        Args:
            s: Complex frequency array (rad/s)

        Returns:
            Initial pole array (complex)
        """
        s_min = np.min(np.abs(s))
        s_max = np.max(np.abs(s))

        # Store normalization factor
        self._s_scale = s_max

        # Create complex conjugate pairs (all poles are complex pairs)
        n_pairs = self.order // 2

        poles = []

        if n_pairs > 0:
            # LINEARLY spaced frequencies (like ex4a.m), not log-spaced
            # bet = linspace(w(1), w(Ns), N/2)
            # Avoid DC (omega=0) to prevent zero poles -> division by zero
            omega_min = max(s_min, s_max * 1e-3)
            omega_max = s_max

            omegas = np.linspace(omega_min, omega_max, n_pairs)

            for omega in omegas:
                # alf = -bet * 1e-2  (1% damping like ex4a.m)
                sigma = -omega * 0.01
                poles.append(sigma + 1j * omega)
                poles.append(sigma - 1j * omega)

        # If odd order, add one real pole at the middle
        if self.order % 2 == 1:
            omega_mid = (s_min + s_max) / 2
            poles.append(-omega_mid * 0.01)

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
        # Following vectfit3.m lines 484-498 exactly
        LAMBD = np.diag(poles).astype(complex)
        B = np.ones(N)
        C_sigma = C_complex.copy()

        # Convert to real state-space form for complex pairs
        m = 0
        while m < N:
            if m < N - 1 and cindex[m] == 1:
                # Complex conjugate pair - convert to 2x2 real block
                p_real = np.real(poles[m])
                p_imag = np.imag(poles[m])

                # LAMBD block: [p_real, p_imag; -p_imag, p_real]
                LAMBD[m, m] = p_real
                LAMBD[m, m+1] = p_imag
                LAMBD[m+1, m] = -p_imag
                LAMBD[m+1, m+1] = p_real

                # B: [2; 0]
                B[m] = 2
                B[m+1] = 0

                # C: [real(C), imag(C)]
                c_val = C_sigma[m]
                C_sigma[m] = np.real(c_val)
                C_sigma[m+1] = np.imag(c_val)

                m += 2
            else:
                m += 1

        # Compute zeros: ZER = LAMBD - B*C.T/D
        ZER = LAMBD - np.outer(B, C_sigma) / D

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
            weight: Optional[np.ndarray] = None,
            extract_delay: bool = True) -> Dict:
        """
        Fit frequency response data using Vector Fitting.

        Args:
            s: Complex frequency array (rad/s)
            H: Frequency response (Ns,) or (Nc, Ns)
            weight: Optional weight array
            extract_delay: Estimate and de-embed a pure transport delay before
                fitting (recommended for real channels; 2026-10). The delay is
                returned as result['delay'] and re-applied when evaluating.

        Returns:
            Dictionary with poles, residues, d, e, num, den, delay, order and
            fit-error metrics (mse, max_error, rel_rms).
        """
        logger.info(f"Starting Vector Fitting with order={self.order}")

        # Ensure H is 2D
        if H.ndim == 1:
            H = H.reshape(1, -1)
        H_orig = H.copy()

        Ns = len(s)
        Nc = H.shape[0]

        # ---- Transport delay extraction / de-embedding (2026-10) ----
        # A pure delay rotates the phase by up to hundreds of radians and makes
        # the rational LS collapse (magnitude error >10dB even at order 60).
        # De-embed it, fit the smooth residual, and keep it as a parameter.
        delay = 0.0
        if extract_delay:
            delay = estimate_delay(s, H[0, :])
            if delay > 0.0:
                H = H * np.exp(s * delay)   # remove e^{-s*tau}
                logger.info(f"Estimated transport delay: {delay*1e9:.3f} ns (de-embedded)")
        self.delay = delay

        # Default weights: 1/sqrt(|f|) like ex4a.m
        # weight=1./sqrt(abs(f)) balances accuracy across magnitude variations
        if weight is None:
            # Use inverse square root of magnitude for weighting
            abs_H = np.abs(H[0, :])
            weight = 1.0 / np.sqrt(np.maximum(abs_H, 1e-15))

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

        result = {
            'poles': self.poles,
            'residues': self.residues,
            'd': self.d,
            'e': self.e,
            'order': self.order,
            'delay': delay,
        }

        # ---- Fit-error metrics on the ORIGINAL (delayed) response ----
        Hm = evaluate_fit(s, result)
        err = Hm - H_orig[0, :]
        result['mse'] = float(np.mean(np.abs(err) ** 2))
        result['max_error'] = float(np.max(np.abs(err)))
        result['rel_rms'] = float(np.linalg.norm(err) / (np.linalg.norm(H_orig[0, :]) + 1e-30))
        # dB error where the response is not in a deep null
        mag_orig = np.abs(H_orig[0, :])
        mask = mag_orig > max(1e-2 * np.max(mag_orig), 1e-30)
        if np.count_nonzero(mask) > 0:
            db_err = np.abs(20*np.log10(np.abs(Hm[mask]) + 1e-20)
                            - 20*np.log10(mag_orig[mask] + 1e-20))
            result['median_error_db'] = float(np.median(db_err))
            result['max_error_db'] = float(np.max(db_err))
        else:
            result['median_error_db'] = float('nan')
            result['max_error_db'] = float('nan')

        # ---- Rational transfer function export (analysis / rational configs) ----
        # Normalized variable u = s / s_scale:
        #   H_rational(s) = polyval(num, u) / polyval(den, u)
        num, den, s_scale = poles_residues_to_tf(
            self.poles, self.residues, self.d, self.e)
        result['num'] = num
        result['den'] = den
        result['s_scale'] = s_scale

        return result

    def _solve_residues(self, s: np.ndarray, H: np.ndarray,
                        weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """
        Solve for residues given fixed poles using real basis functions.

        Based on vectfit3.m residue identification (lines 577-712):
        - For complex pole pairs, use basis: 1/(s-p)+1/(s-p*) and j/(s-p)-j/(s-p*)
        - Apply column scaling to fix ill-conditioning
        - Convert back to complex residues

        Args:
            s: Complex frequency array
            H: Frequency response
            weight: Optional per-frequency weights (length Ns). Applied to both
                    the real and imaginary equation rows; None preserves the
                    historical unweighted residue solve.

        Returns:
            residues (complex), d, e
        """
        N = len(self.poles)
        Ns = len(s)

        # Build cindex to identify complex conjugate pairs
        cindex = self._build_cindex(self.poles)

        # Build system matrix with REAL basis functions (vectfit3.m style)
        # For real poles: 1/(s-p)
        # For complex pairs: 1/(s-p)+1/(s-p*) and j/(s-p)-j/(s-p*)
        A = np.zeros((Ns, N + 2), dtype=complex)

        for m in range(N):
            if cindex[m] == 0:  # Real pole
                A[:, m] = 1.0 / (s - self.poles[m])
            elif cindex[m] == 1:  # Complex pole, first part
                p = self.poles[m]
                p_conj = np.conj(p)
                A[:, m] = 1.0 / (s - p) + 1.0 / (s - p_conj)
                A[:, m + 1] = 1j / (s - p) - 1j / (s - p_conj)
            # cindex[m] == 2 is filled by m-1

        # Add d and e terms
        A[:, N] = 1.0      # d term
        A[:, N + 1] = s    # e term (proportional)

        # Stack real and imaginary parts
        A_real = np.vstack([A.real, A.imag])
        b_real = np.hstack([H.real, H.imag])

        # Optional weighting (T-S4/A1): a per-frequency weight applied to both
        # the real and imaginary rows of the residue LS. Without it the residue
        # solve is absolute-error least squares (historical behaviour).
        if weight is not None:
            w = np.asarray(weight, dtype=float).ravel()
            if len(w) != Ns:
                raise ValueError(f"residue weight length {len(w)} != Ns {Ns}")
            ww = np.concatenate([w, w])
            A_real = A_real * ww[:, None]
            b_real = b_real * ww

        # Column scaling to fix ill-conditioning (vectfit3.m lines 624-628)
        Escale = np.linalg.norm(A_real, axis=0)
        Escale[Escale < 1e-15] = 1.0
        A_scaled = A_real / Escale

        # Solve least squares
        x = np.linalg.lstsq(A_scaled, b_real, rcond=None)[0]

        # Undo scaling
        x = x / Escale

        # Extract and convert residues back to complex (vectfit3.m lines 705-712)
        residues = np.zeros(N, dtype=complex)
        for m in range(N):
            if cindex[m] == 0:  # Real pole -> real residue
                residues[m] = x[m]
            elif cindex[m] == 1:  # Complex pair
                r1, r2 = x[m], x[m + 1]
                residues[m] = r1 + 1j * r2
                residues[m + 1] = r1 - 1j * r2  # Conjugate

        d = x[N]
        e = x[N + 1]

        return residues, d, e


    def to_state_space(self) -> Dict:
        """
        Convert pole-residue representation to real state-space form.

        State-space form:
            x_dot = A*x + B*u
            y = C*x + D*u + E*u_dot

        Complex conjugate pole pairs are converted to real 2x2 blocks.

        Returns:
            Dictionary with A, B, C, D, E matrices (all real)
        """
        if self.poles is None or self.residues is None:
            raise RuntimeError("Must call fit() before to_state_space()")

        n = len(self.poles)

        # Build real state-space matrices
        # For complex conjugate pairs, use 2x2 real blocks
        A_blocks = []
        B_blocks = []
        C_blocks = []

        i = 0
        state_idx = 0
        while i < n:
            p = self.poles[i]
            r = self.residues[i]

            if np.abs(p.imag) > 1e-12:
                # Complex conjugate pair
                # Find conjugate
                if i + 1 < n and np.abs(self.poles[i+1] - p.conj()) < 1e-6:
                    p_conj = self.poles[i+1]
                    r_conj = self.residues[i+1]
                else:
                    # Find closest conjugate
                    p_conj = p.conj()
                    r_conj = r.conj()

                sigma = p.real
                omega = p.imag
                alpha = r.real
                beta = r.imag

                # 2x2 real block for A
                A_block = [[sigma, omega], [-omega, sigma]]
                B_block = [[2.0], [0.0]]
                C_block = [[alpha, -beta]]

                A_blocks.append(A_block)
                B_blocks.append(B_block)
                C_blocks.append(C_block)

                i += 2
                state_idx += 2
            else:
                # Real pole
                A_block = [[p.real]]
                B_block = [[1.0]]
                C_block = [[r.real]]

                A_blocks.append(A_block)
                B_blocks.append(B_block)
                C_blocks.append(C_block)

                i += 1
                state_idx += 1

        # Assemble full matrices
        n_states = sum(len(b) for b in A_blocks)

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, 1))
        C = np.zeros((1, n_states))

        idx = 0
        for i, (Ab, Bb, Cb) in enumerate(zip(A_blocks, B_blocks, C_blocks)):
            m = len(Ab)
            A[idx:idx+m, idx:idx+m] = Ab
            B[idx:idx+m, 0] = np.array(Bb).flatten()
            C[0, idx:idx+m] = np.array(Cb).flatten()
            idx += m

        D = np.array([[float(self.d)]])
        E = np.array([[float(self.e)]])

        return {
            'A': A.tolist(),
            'B': B.tolist(),
            'C': C.tolist(),
            'D': D.tolist(),
            'E': E.tolist(),
            'n_states': n_states,
            'n_outputs': 1
        }




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


# ==============================================================================
# Delay Utility Functions
# ==============================================================================

def estimate_delay_phase(freq: np.ndarray, H_row: np.ndarray) -> float:
    """
    Estimate propagation delay from phase slope (linear regression).

    tau = -d(phase)/d(omega), solved by least-squares fit of
    phase vs omega across the full frequency range.

    Args:
        freq: Frequency array (Hz)
        H_row: Complex S-parameter response (1-D)

    Returns:
        tau: Estimated propagation delay (seconds)
    """
    phase = np.unwrap(np.angle(H_row))
    omega = 2.0 * np.pi * freq
    slope, _ = np.polyfit(omega, phase, 1)
    return float(-slope)


def estimate_delay_impulse(freq: np.ndarray, H_row: np.ndarray) -> float:
    """
    Estimate propagation delay from impulse response peak position.

    This method is more accurate for dispersive channels than phase-slope,
    because it finds the actual "time of arrival" of the main energy pulse.

    Args:
        freq: Frequency array (Hz), must be uniformly spaced
        H_row: Complex S-parameter response (1-D)

    Returns:
        tau: Estimated propagation delay (seconds)
    """
    # Zero-pad for better time resolution
    n_fft = len(H_row) * 32
    impulse = np.fft.irfft(H_row, n=n_fft)
    df = freq[1] - freq[0]
    dt = 1.0 / (n_fft * df)
    # Search only in the first quarter (causal response)
    peak_idx = int(np.argmax(np.abs(impulse[:n_fft // 4])))
    return float(peak_idx * dt)


def remove_delay(freq: np.ndarray, H: np.ndarray, tau: float) -> np.ndarray:
    """
    Remove propagation delay from a frequency response.

    H_comp(f) = H(f) * exp(+j * 2*pi * f * tau)

    Args:
        freq: Frequency array (Hz)
        H: Complex response, shape (Ns,) or (Nc, Ns)
        tau: Delay to remove (seconds)

    Returns:
        H_compensated: Delay-removed response (same shape as H)
    """
    exp_fac = np.exp(1j * 2.0 * np.pi * freq * tau)
    if H.ndim == 1:
        return H * exp_fac
    return H * exp_fac[np.newaxis, :]


def restore_delay(freq: np.ndarray, H_fit: np.ndarray, tau: float) -> np.ndarray:
    """
    Restore propagation delay to a fitted frequency response.

    H_restored(f) = H_fit(f) * exp(-j * 2*pi * f * tau)

    Args:
        freq: Frequency array (Hz)
        H_fit: Fitted complex response (same shape as original)
        tau: Delay to restore (seconds)

    Returns:
        H_restored: Response with delay restored
    """
    return remove_delay(freq, H_fit, -tau)


# ==============================================================================
# SParamModel: High-level S-parameter fitting with MIMO state-space export
# ==============================================================================

class SParamModel:
    """
    High-level S-parameter model builder.

    Supports arbitrary sNp files (s2p, s4p, s8p, s12p, s16p, ...),
    automatic port mapping (differential pair detection), optional
    differential (mixed-mode) conversion, delay extraction, columnwise
    Vector Fitting with shared poles (ex4b.m strategy), and MIMO
    state-space JSON export for C++ simulation.

    Typical usage::

        model = SParamModel()
        model.load_snp('channel.s12p')
        # Override auto-detection if needed:
        # model.set_port_map(diff_pairs=[(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)])
        model.to_differential()
        model.fit(order=50, extract_delay=True)
        model.summary()
        model.export_json('channel.json')
    """

    def __init__(self):
        # Raw data
        self.freq: Optional[np.ndarray] = None
        self.s_raw: Optional[np.ndarray] = None    # (N_freq, N_port, N_port)
        self.s_active: Optional[np.ndarray] = None  # working matrix
        self.n_ports_raw: int = 0
        self.filename: str = ""

        # System config (for auto fmax calculation)
        self._config: Optional[Dict] = None
        self._symbol_rate: Optional[float] = None  # Hz
        self._fmax_multiplier: float = 1.5  # fmax = multiplier × symbol_rate

        # Port mapping
        self.diff_pairs: Optional[List[Tuple[int, int]]] = None  # [(p+,p-) 1-based]
        self.tx_diff_ports: Optional[List[int]] = None           # 0-based in diff space
        self.rx_diff_ports: Optional[List[int]] = None
        self._is_differential: bool = False

        # Selected port pairs for fitting (0-based indices in s_active)
        self.selected_pairs: Optional[List[Tuple[int, int]]] = None

        # Fitting results
        self.shared_poles: Optional[np.ndarray] = None
        self.residues_matrix: Optional[np.ndarray] = None  # (N_pairs, N_poles) complex
        self.d_vector: Optional[np.ndarray] = None          # (N_pairs,) real
        self.e_vector: Optional[np.ndarray] = None          # (N_pairs,) real
        self.delay_map: Optional[Dict] = None               # {(out,in): tau_s}

    # --------------------------------------------------------------------------
    # System Configuration
    # --------------------------------------------------------------------------

    def load_config(
        self,
        config_path: str = 'config/default.json',
        fmax_multiplier: float = 1.5
    ) -> 'SParamModel':
        """
        Load SerDes system configuration to auto-calculate fmax from symbol rate.

        The symbol rate is derived from global.UI (unit interval) in the config:
            symbol_rate = 1 / UI
            fmax = fmax_multiplier × symbol_rate

        Args:
            config_path: Path to the JSON config file (default: 'config/default.json')
            fmax_multiplier: Multiplier for symbol rate to get fmax (default: 1.5)
                - 1.0: Nyquist frequency
                - 1.5: Covers main lobe + first sidelobe (recommended for NRZ)
                - 2.0: More conservative

        Returns:
            self (for method chaining)
        """
        import json
        import os

        # Handle relative paths
        if not os.path.isabs(config_path):
            # Try relative to workspace root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(script_dir)
            config_path = os.path.join(workspace_root, config_path)

        with open(config_path, 'r') as f:
            self._config = json.load(f)

        self._fmax_multiplier = fmax_multiplier

        # Extract UI and compute symbol rate
        ui = self._config.get('global', {}).get('UI')
        if ui is not None and ui > 0:
            self._symbol_rate = 1.0 / ui
            fmax_auto = self._fmax_multiplier * self._symbol_rate
            logger.info(
                f"Config loaded: UI={ui*1e12:.2f}ps → symbol_rate={self._symbol_rate/1e9:.1f}Gbps → "
                f"fmax={fmax_auto/1e9:.1f}GHz (×{fmax_multiplier})"
            )
        else:
            logger.warning("Config does not contain global.UI; fmax auto-calculation disabled")

        return self

    def get_auto_fmax(self) -> Optional[float]:
        """
        Get automatically calculated fmax from loaded config.

        Returns:
            fmax in Hz, or None if config not loaded or UI not available.
        """
        if self._symbol_rate is not None:
            return self._fmax_multiplier * self._symbol_rate
        return None

    # --------------------------------------------------------------------------
    # Loading
    # --------------------------------------------------------------------------

    def load_snp(self, filename: str) -> 'SParamModel':
        """
        Load an sNp file using scikit-rf.

        Supports any port count: s2p, s4p, s8p, s12p, s16p, etc.
        After loading, auto-detection of differential pairs is attempted.

        Args:
            filename: Path to the sNp file.

        Returns:
            self (for method chaining)
        """
        try:
            import skrf as rf
        except ImportError:
            raise ImportError("scikit-rf is required: pip install scikit-rf")

        nw = rf.Network(filename)
        self.filename = filename
        self.freq = nw.f.copy()                # (N_freq,)
        self.s_raw = nw.s.copy()              # (N_freq, N_port, N_port)
        self.n_ports_raw = nw.number_of_ports
        self.s_active = self.s_raw.copy()
        self._is_differential = False

        logger.info(
            f"Loaded {filename}: {self.n_ports_raw} ports, "
            f"{len(self.freq)} pts, "
            f"{self.freq[0]/1e9:.2f}-{self.freq[-1]/1e9:.2f} GHz"
        )

        self.auto_detect_port_map(verbose=True)
        return self

    # --------------------------------------------------------------------------
    # Port Mapping
    # --------------------------------------------------------------------------

    def set_port_map(
        self,
        diff_pairs: List[Tuple[int, int]],
        tx_ports: Optional[List[int]] = None,
        rx_ports: Optional[List[int]] = None,
    ) -> 'SParamModel':
        """
        Manually specify differential pair port mapping.

        Args:
            diff_pairs: List of (p_plus, p_minus) tuples using 1-based port
                        numbers. Example: [(1,2),(3,4),(5,6)] for a 6-port
                        file with 3 differential pairs.
            tx_ports: 1-based indices into diff_pairs that are TX.
                      Example: [1,2] means diff pairs 0 and 1 are TX.
            rx_ports: 1-based indices into diff_pairs that are RX.

        Returns:
            self
        """
        for pp, pm in diff_pairs:
            if pp < 1 or pp > self.n_ports_raw or pm < 1 or pm > self.n_ports_raw:
                raise ValueError(
                    f"Port indices must be 1..{self.n_ports_raw}, got ({pp},{pm})"
                )
        flat = [p for pair in diff_pairs for p in pair]
        if len(set(flat)) != len(flat):
            raise ValueError("Duplicate port numbers in diff_pairs")

        self.diff_pairs = list(diff_pairs)
        self.tx_diff_ports = [p - 1 for p in tx_ports] if tx_ports else None
        self.rx_diff_ports = [p - 1 for p in rx_ports] if rx_ports else None
        logger.info(f"Port map set manually: {len(diff_pairs)} diff pairs")
        return self

    def auto_detect_port_map(self, verbose: bool = True) -> 'SParamModel':
        """
        Auto-detect differential pair mapping using a heuristic CMRR test.

        Tries two common port-ordering conventions:
          - Interleaved : (1,2),(3,4),(5,6),...  (most common, e.g. PCIe)
          - Half-split  : (1,N/2+1),(2,N/2+2),... (e.g. CtoCassembly6dB.s12p)

        Selects the scheme that maximises |Sdd|/|Scc| at mid-band.
        TX/RX direction is inferred from asymmetry in forward transmission.

        Results are printed and can be overridden with set_port_map().

        Returns:
            self
        """
        if self.s_raw is None:
            return self

        N = self.n_ports_raw
        if N < 2 or N % 2 != 0:
            if verbose:
                print(f"[PortMap] {N} ports (odd or <2); diff detection skipped")
            return self

        N_diff = N // 2
        mid = len(self.freq) // 2
        S_mid = self.s_raw[mid]  # (N, N) snapshot at mid-band

        schemes: Dict[str, List[Tuple[int, int]]] = {
            'interleaved': [(2 * i + 1, 2 * i + 2) for i in range(N_diff)],
            'half-split':  [(i + 1, N_diff + i + 1) for i in range(N_diff)],
        }

        best_scheme = 'interleaved'
        best_score = -np.inf
        best_pairs: List[Tuple[int, int]] = schemes['interleaved']

        for name, pairs in schemes.items():
            score = 0.0
            for pp, pm in pairs:
                pi, mi = pp - 1, pm - 1
                Sdd = (S_mid[pi, pi] - S_mid[pi, mi]
                       - S_mid[mi, pi] + S_mid[mi, mi]) / 2.0
                Scc = (S_mid[pi, pi] + S_mid[pi, mi]
                       + S_mid[mi, pi] + S_mid[mi, mi]) / 2.0
                score += abs(Sdd) / (abs(Scc) + 1e-15)
            score /= N_diff
            if score > best_score:
                best_score = score
                best_scheme = name
                best_pairs = pairs

        self.diff_pairs = best_pairs

        # Detect TX/RX from differential transmission asymmetry
        tx_ports, rx_ports = self._detect_tx_rx(best_pairs)
        self.tx_diff_ports = tx_ports
        self.rx_diff_ports = rx_ports

        if verbose:
            print(f"\n[PortMap] Auto-detection result:")
            print(f"  Scheme  : {best_scheme}  (CMRR score = {best_score:.2f})")
            print(f"  Pairs   : {best_pairs}")
            if tx_ports is not None:
                print(f"  TX diff : {[i + 1 for i in tx_ports]} (1-based in diff space)")
                print(f"  RX diff : {[i + 1 for i in rx_ports]} (1-based in diff space)")
            else:
                print(f"  TX/RX   : undetermined")
            print(f"  [Use set_port_map() to override if incorrect]")

        return self

    def _detect_tx_rx(
        self,
        pairs: List[Tuple[int, int]],
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Infer TX/RX direction from differential transmission strength."""
        N_diff = len(pairs)
        if N_diff < 2:
            return None, None

        tx_score = np.zeros(N_diff)
        for j, (pj_1, mj_1) in enumerate(pairs):
            pj, mj = pj_1 - 1, mj_1 - 1
            for i, (pi_1, mi_1) in enumerate(pairs):
                if i == j:
                    continue
                pi, mi = pi_1 - 1, mi_1 - 1
                # |Sdd(i,j)| averaged over frequency
                Sdd_ij = (
                    self.s_raw[:, pi, pj] - self.s_raw[:, pi, mj]
                    - self.s_raw[:, mi, pj] + self.s_raw[:, mi, mj]
                ) / 2.0
                tx_score[j] += float(np.mean(np.abs(Sdd_ij)))

        thr = float(np.mean(tx_score))
        if thr < 1e-6:
            return None, None

        tx = [i for i in range(N_diff) if tx_score[i] > thr * 0.5]
        rx = [i for i in range(N_diff) if tx_score[i] <= thr * 0.5]
        if not tx or not rx:
            return None, None
        return tx, rx

    def print_port_info(self) -> None:
        """Print port mapping, per-path delay and insertion loss summary."""
        if self.s_raw is None:
            print("No data loaded")
            return

        print(f"\n=== Port Information: {self.filename} ===")
        print(f"Ports  : {self.n_ports_raw} single-ended")
        print(
            f"Freq   : {self.freq[0]/1e9:.3f} - {self.freq[-1]/1e9:.3f} GHz "
            f"({len(self.freq)} pts)"
        )

        if self.diff_pairs:
            print(f"\nDifferential pairs ({len(self.diff_pairs)}):")
            for i, (pp, pm) in enumerate(self.diff_pairs):
                role = ""
                if self.tx_diff_ports and i in self.tx_diff_ports:
                    role = " [TX]"
                elif self.rx_diff_ports and i in self.rx_diff_ports:
                    role = " [RX]"
                print(f"  Pair {i + 1}: P{pp}+ / P{pm}-{role}")

        N_act = self.s_active.shape[1]
        mid10 = max(1, len(self.freq) // 10)
        print(f"\nActive matrix ({N_act}x{N_act}) – significant paths:")
        for oi in range(N_act):
            for ii in range(N_act):
                if oi == ii:
                    continue
                s_path = self.s_active[:, oi, ii]
                if np.mean(np.abs(s_path)) < 0.005:
                    continue
                tau = estimate_delay_phase(self.freq, s_path)
                il_low  = 20 * np.log10(max(abs(s_path[mid10]), 1e-15))
                il_high = 20 * np.log10(max(abs(s_path[-1]),    1e-15))
                print(
                    f"  S({oi + 1},{ii + 1}): τ={tau * 1e9:.3f}ns, "
                    f"IL@low={il_low:.1f}dB, IL@high={il_high:.1f}dB"
                )

    # --------------------------------------------------------------------------
    # Differential Conversion
    # --------------------------------------------------------------------------

    def to_differential(self) -> 'SParamModel':
        """
        Convert single-ended S-matrix to the differential Sdd submatrix.

        Applies the mixed-mode transformation  Smm = M * S_reordered * M^T
        where M is the block-diagonal orthogonal matrix that maps each
        (P+, P-) pair to (differential, common-mode) signals.
        The top-left N_diff x N_diff block of Smm is the Sdd matrix.

        If set_port_map() has not been called, auto_detect_port_map() is run
        first.

        Returns:
            self
        """
        if self.diff_pairs is None:
            self.auto_detect_port_map(verbose=True)
        if self.diff_pairs is None:
            raise RuntimeError("No differential pairs available; call set_port_map()")

        pairs = self.diff_pairs
        N_diff = len(pairs)
        N_se = 2 * N_diff
        N_freq = len(self.freq)

        # Reorder ports: [p0+, p0-, p1+, p1-, ...]
        port_order = np.array(
            [item for pp, pm in pairs for item in (pp - 1, pm - 1)]
        )
        # Vectorised port reorder: shape (N_freq, N_se, N_se)
        s_reord = self.s_raw[:, port_order, :][:, :, port_order]

        # Build orthogonal block-diagonal M  (M^{-1} = M^T)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        M = np.zeros((N_se, N_se))
        for i in range(N_diff):
            M[i,          2 * i]     =  inv_sqrt2   # diff row
            M[i,          2 * i + 1] = -inv_sqrt2
            M[N_diff + i, 2 * i]     =  inv_sqrt2   # common-mode row
            M[N_diff + i, 2 * i + 1] =  inv_sqrt2
        M_inv = M.T  # orthogonal matrix

        # Smm = M @ S_reord @ M^T  (einsum broadcasts over frequency axis)
        s_mm = np.einsum('ij,fjk,kl->fil', M, s_reord, M_inv)

        # Extract Sdd submatrix
        self.s_active = s_mm[:, :N_diff, :N_diff]
        self._is_differential = True

        logger.info(
            f"Differential conversion: {self.n_ports_raw} SE → "
            f"{N_diff}×{N_diff} Sdd"
        )
        return self

    # --------------------------------------------------------------------------
    # Port Selection
    # --------------------------------------------------------------------------

    def select_ports(self, port_pairs) -> 'SParamModel':
        """
        Select which port pairs to fit.

        Args:
            port_pairs:
                'all'       – all N×N elements of the active matrix.
                'all_thru'  – off-diagonal elements with mean |S| ≥ 0.005
                              (-46 dB threshold).
                [(o,i),...] – explicit list of 1-based (out, in) indices.

        Returns:
            self
        """
        N = self.s_active.shape[1]

        if port_pairs == 'all':
            self.selected_pairs = [(oi, ii) for oi in range(N) for ii in range(N)]

        elif port_pairs == 'all_thru':
            pairs = []
            for oi in range(N):
                for ii in range(N):
                    if oi == ii:
                        continue
                    if np.mean(np.abs(self.s_active[:, oi, ii])) >= 0.005:
                        pairs.append((oi, ii))
            self.selected_pairs = pairs

        else:
            # Convert 1-based to 0-based
            self.selected_pairs = [(o - 1, i - 1) for o, i in port_pairs]

        logger.info(f"Selected {len(self.selected_pairs)} port pairs for fitting")
        return self

    # --------------------------------------------------------------------------
    # Fitting  (Phase 1: shared poles | Phase 2: per-column residues)
    # --------------------------------------------------------------------------

    def fit(
        self,
        order: int = 50,
        port_pairs=None,
        extract_delay: bool = True,
        fmax: Optional[Union[float, str]] = None,
        n_iter_poles: int = 5,
        n_iter_res: int = 1,
        weight: Optional[np.ndarray] = None,
        weight_residues: bool = False,
    ) -> 'SParamModel':
        """
        Columnwise Vector Fitting with shared poles (ex4b.m strategy).

        Phase 1
            Optimize shared poles from all selected columns jointly.

        Phase 2
            With poles fixed, solve residues independently for each column.

        If extract_delay=True, propagation delay is estimated using the
        **impulse-peak method** (more accurate for dispersive channels) and
        removed before fitting.  Delay is restored during evaluate().

        **Key for low-order fitting**: Set fmax to bandlimit the fit to your
        actual signal bandwidth.  For example:
          - 10 Gbps NRZ  -> fmax=5e9   (Nyquist ~5 GHz)  -> 20-30 poles
          - 28 Gbps NRZ  -> fmax=15e9                    -> 30-50 poles
          - 56 Gbps PAM4 -> fmax=14e9                    -> 30-50 poles

        With bandlimiting + delay extraction, 20-50 poles can achieve < 1 dB
        accuracy for most SerDes applications.

        Args:
            order: Number of poles.
            port_pairs: Passed to select_ports(). None -> 'all_thru'.
            extract_delay: Remove delay using impulse-peak method.
            fmax: Maximum frequency to fit (Hz). Options:
                  - None: Use full bandwidth of S-parameter data.
                  - float: Explicit frequency limit in Hz.
                  - 'auto': Auto-calculate from config (1.5 × symbol_rate).
                            Requires load_config() to be called first.
            n_iter_poles: Pole-optimization iterations (Phase 1).
            n_iter_res: Unused (residues solved in one LS step).
            weight: Manual weight array (Ns,). Default: 1/sqrt(mean|H|).
            weight_residues: If True, also use `weight` in the per-column residue
                least-squares (T-S4/A1). Default False keeps the historical
                unweighted residue solve.

        Returns:
            self
        """
        if port_pairs is not None:
            self.select_ports(port_pairs)
        if self.selected_pairs is None:
            self.select_ports('all_thru')
        if not self.selected_pairs:
            raise ValueError("No port pairs selected for fitting")

        # ── Handle fmax='auto' ─────────────────────────────────────────────────
        if fmax == 'auto':
            fmax = self.get_auto_fmax()
            if fmax is None:
                raise ValueError(
                    "fmax='auto' requires load_config() to be called first "
                    "with a config containing global.UI"
                )
            logger.info(f"Auto fmax from config: {fmax/1e9:.1f} GHz")

        freq   = self.freq
        omega  = 2.0 * np.pi * freq
        s      = 1j * omega
        N_freq = len(freq)
        N_pairs = len(self.selected_pairs)

        # Collect H: (N_pairs, N_freq)
        H_all = np.stack(
            [self.s_active[:, oi, ii] for oi, ii in self.selected_pairs]
        )

        # ── Bandlimit to fmax (key for low-order fitting with delay extraction) ─
        if fmax is not None and fmax < freq[-1]:
            bw_mask = freq <= fmax
            freq_fit   = freq[bw_mask]
            omega_fit  = omega[bw_mask]
            s_fit      = s[bw_mask]
            H_fit_all  = H_all[:, bw_mask]
            self._fmax_used = float(fmax)
            logger.info(f"Bandlimited to {fmax/1e9:.1f} GHz ({bw_mask.sum()} pts)")
        else:
            freq_fit   = freq
            omega_fit  = omega
            s_fit      = s
            H_fit_all  = H_all
            self._fmax_used = None

        # ── Delay extraction using impulse-peak method ───────────────────────
        self.delay_map = {}
        H_comp = H_fit_all.copy()

        if extract_delay:
            for k, (oi, ii) in enumerate(self.selected_pairs):
                # Use impulse-peak method (more accurate for dispersive channels)
                tau = estimate_delay_impulse(freq_fit, H_fit_all[k])
                tau = float(np.clip(tau, 0.0, 100e-9))
                self.delay_map[(oi, ii)] = tau
                H_comp[k] = remove_delay(freq_fit, H_fit_all[k], tau)

            taus = list(self.delay_map.values())
            logger.info(
                f"Delays (impulse-peak): min={min(taus)*1e9:.2f}ns  "
                f"max={max(taus)*1e9:.2f}ns"
            )

        # ── Remove DC point (freq=0) to avoid pole-at-zero singularity ────────
        dc_mask = freq_fit > 0
        if not dc_mask.all():
            freq_vf  = freq_fit[dc_mask]
            s_vf     = s_fit[dc_mask]
            H_vf     = H_comp[:, dc_mask]
            logger.info(f"Skipped {(~dc_mask).sum()} DC point(s) for VF")
        else:
            freq_vf = freq_fit
            s_vf    = s_fit
            H_vf    = H_comp

        # Store for evaluate()
        self._freq_vf = freq_vf

        # ── Default weight ───────────────────────────────────────────────────
        if weight is None:
            avg_mag = np.mean(np.abs(H_vf), axis=0)
            weight  = 1.0 / np.sqrt(np.maximum(avg_mag, 1e-15))

        # ── Phase 1: shared-pole optimisation ───────────────────────────────
        # Pass ALL columns simultaneously to _pole_identification so it
        # builds the joint QR system (like ex4b.m passing the full column
        # set at once).  This gives much better shared poles than fitting
        # only the magnitude envelope.
        vf = VectorFitting(order=order, max_iterations=1,
                           relax=True, stable=True, asymp=2)
        vf.poles = vf._initialize_poles(s_vf)
        opts = {'relax': True, 'stable': True}

        logger.info(
            f"Phase 1: optimising {order} shared poles "
            f"({n_iter_poles} iters, {N_pairs} columns)"
        )
        for _ in range(n_iter_poles):
            vf.poles = vf._pole_identification(s_vf, H_vf, vf.poles, weight, opts)

        self.shared_poles = vf.poles.copy()

        # ── Phase 2: per-column residue identification ───────────────────────
        self.residues_matrix = np.zeros((N_pairs, order), dtype=complex)
        self.d_vector = np.zeros(N_pairs)
        self.e_vector = np.zeros(N_pairs)

        logger.info(f"Phase 2: solving residues for {N_pairs} columns...")
        vf_res = VectorFitting(order=order, asymp=2)
        vf_res.poles = self.shared_poles.copy()

        for k, (oi, ii) in enumerate(self.selected_pairs):
            rw = weight if weight_residues else None
            residues, d, e = vf_res._solve_residues(s_vf, H_vf[k], weight=rw)
            self.residues_matrix[k] = residues
            self.d_vector[k] = float(d)
            self.e_vector[k] = float(e)

        logger.info("Fitting complete")
        return self

    # --------------------------------------------------------------------------
    # State-Space and Export
    # --------------------------------------------------------------------------

    def to_state_space(self) -> Dict:
        """
        Build MIMO state-space from shared poles.

        For a true MIMO system with N_inputs and N_outputs:
        - A : (n_states, n_states)  block-diagonal, shared
        - B : (n_states, N_inputs)  one column per unique input
        - C : (N_outputs, n_states) one row per unique output
        - D : (N_outputs, N_inputs) feedthrough matrix
        - E : (N_outputs, N_inputs) derivative matrix

        Returns:
            dict with MIMO state-space matrices.
        """
        if self.shared_poles is None:
            raise RuntimeError("Must call fit() before to_state_space()")

        poles  = self.shared_poles
        N_poles = len(poles)
        N_pairs = len(self.selected_pairs)

        # Determine unique inputs and outputs from selected_pairs
        unique_inputs = sorted(set(ii for oi, ii in self.selected_pairs))
        unique_outputs = sorted(set(oi for oi, ii in self.selected_pairs))
        N_inputs = len(unique_inputs)
        N_outputs = len(unique_outputs)

        # Map from port index to matrix index
        input_map = {port: idx for idx, port in enumerate(unique_inputs)}
        output_map = {port: idx for idx, port in enumerate(unique_outputs)}

        # Build cindex for block type identification
        vf_tmp = VectorFitting(order=N_poles)
        vf_tmp.poles = poles
        cindex = vf_tmp._build_cindex(poles)

        # ── Assemble A (block-diagonal) and B_col (standard excitation) ─────
        A_blocks, B_entries = [], []
        i = 0
        while i < N_poles:
            p = poles[i]
            if abs(p.imag) > 1e-10:             # complex conjugate pair
                s_r, o_r = p.real, abs(p.imag)
                A_blocks.append([[s_r, o_r], [-o_r, s_r]])
                B_entries.extend([2.0, 0.0])
                i += 2
            else:                               # real pole
                A_blocks.append([[p.real]])
                B_entries.append(1.0)
                i += 1

        n_states = sum(len(ab) for ab in A_blocks)
        A = np.zeros((n_states, n_states))
        idx = 0
        for ab in A_blocks:
            m = len(ab)
            A[idx:idx + m, idx:idx + m] = ab
            idx += m

        B_col = np.array(B_entries, dtype=float)  # (n_states,)

        # ── Build MIMO B matrix (n_states, N_inputs) ────────────────────────
        # Each input gets its own copy of the standard excitation vector
        B = np.zeros((n_states, N_inputs))
        for j in range(N_inputs):
            B[:, j] = B_col

        # ── Build MIMO C matrix (N_outputs, n_states) ───────────────────────
        # And MIMO D, E matrices (N_outputs, N_inputs)
        C = np.zeros((N_outputs, n_states))
        D = np.zeros((N_outputs, N_inputs))
        E = np.zeros((N_outputs, N_inputs))

        for k, (oi, ii) in enumerate(self.selected_pairs):
            out_idx = output_map[oi]
            in_idx = input_map[ii]

            residues = self.residues_matrix[k]

            # Fill C row for this output (accumulate if multiple inputs to same output)
            j_st = 0
            j_pl = 0
            while j_pl < N_poles:
                p = poles[j_pl]
                r = residues[j_pl]
                if abs(p.imag) > 1e-10:         # complex pair block
                    C[out_idx, j_st]     += r.real
                    C[out_idx, j_st + 1] += r.imag if p.imag > 0 else -r.imag
                    j_st += 2
                    j_pl += 2
                else:                           # real pole
                    C[out_idx, j_st] += r.real
                    j_st += 1
                    j_pl += 1

            # Fill D and E for this (output, input) pair
            D[out_idx, in_idx] = self.d_vector[k]
            E[out_idx, in_idx] = self.e_vector[k]

        return {
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
            'n_states':  n_states,
            'n_pairs':   N_pairs,
            'n_inputs':  N_inputs,
            'n_outputs': N_outputs,
            'port_pairs': self.selected_pairs,
            'delay_map':  self.delay_map or {},
        }

    def export_json(self, filename: str, fs: float = 80e9,
                     active_inputs: Optional[List[int]] = None,
                     active_outputs: Optional[List[int]] = None) -> None:
        """
        Export MIMO state-space to JSON for C++ channel simulation.

        The JSON contains the full State Space model plus port selection config.
        C++ side can use the full model and select which inputs/outputs to use.

        JSON layout::

            {
              "version": "3.0",
              "method": "state_space",
              "fs": <sampling_freq>,
              "full_model": {
                "n_diff_ports": <N>,
                "n_outputs": <N_pairs>,
                "n_states": <n_states>,
                "port_pairs": [[out,in], ...],
                "delay_s": [tau0, ...],
                "state_space": {"A", "B", "C", "D", "E"}
              },
              "port_config": {
                "active_inputs": [0, ...],
                "active_outputs": [0, ...]
              },
              "metadata": {...}
            }

        Args:
            filename: Output JSON file path.
            fs: Simulation sampling frequency (Hz).
            active_inputs: Which input ports to use (0-based). None = all.
            active_outputs: Which outputs to use (0-based). None = all.
        """
        import json

        ss = self.to_state_space()
        n_diff = self.s_active.shape[1] if self.s_active is not None else 1

        # Default: use all inputs and outputs
        if active_inputs is None:
            active_inputs = list(range(n_diff))
        if active_outputs is None:
            active_outputs = list(range(ss['n_pairs']))

        delay_list = [
            float(ss['delay_map'].get(tuple(p), 0.0))
            for p in ss['port_pairs']
        ]

        config = {
            'version': '3.0',
            'method': 'state_space',
            'fs': float(fs),
            'full_model': {
                'n_diff_ports': n_diff,
                'n_outputs': int(ss['n_outputs']),
                'n_states': int(ss['n_states']),
                'port_pairs': [[int(o), int(i)] for o, i in ss['port_pairs']],
                'delay_s': delay_list,
                'state_space': {
                    'A': ss['A'].tolist(),
                    'B': ss['B'].tolist(),
                    'C': ss['C'].tolist(),
                    'D': ss['D'].tolist(),
                    'E': ss['E'].tolist(),
                },
            },
            'port_config': {
                'active_inputs': active_inputs,
                'active_outputs': active_outputs,
            },
            'metadata': {
                'n_raw_ports': self.n_ports_raw,
                'is_differential': self._is_differential,
                'diff_pairs': self.diff_pairs,
                'freq_min_ghz': float(self.freq[0] / 1e9) if self.freq is not None else 0.0,
                'freq_max_ghz': float(self.freq[-1] / 1e9) if self.freq is not None else 0.0,
            },
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        max_delay = max(delay_list) if delay_list else 0.0
        print(f"[Export] Saved: {filename}")
        print(
            f"  States={ss['n_states']}, Outputs={ss['n_pairs']}, "
            f"Active inputs={len(active_inputs)}, Active outputs={len(active_outputs)}, "
            f"Max delay={max_delay * 1e9:.3f} ns"
        )
        logger.info(f"Exported MIMO state-space to {filename}")

    # --------------------------------------------------------------------------
    # Evaluation and Diagnostics
    # --------------------------------------------------------------------------

    def evaluate(
        self, freq_eval: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate the fitted model at given frequencies.

        Reconstructs H(f) = sum_k r_k/(j*2*pi*f - p_k) + d + e*j*2*pi*f
        and restores the extracted delay.

        Args:
            freq_eval: Frequency array (Hz). None → original frequencies.

        Returns:
            H_fit: (N_pairs, N_freq_eval) complex array.
        """
        if self.shared_poles is None:
            raise RuntimeError("Must call fit() before evaluate()")

        if freq_eval is None:
            freq_eval = self.freq
        s_eval  = 1j * 2.0 * np.pi * freq_eval
        N_freq  = len(freq_eval)
        N_pairs = len(self.selected_pairs)

        poles  = self.shared_poles
        vf_tmp = VectorFitting(order=len(poles))
        vf_tmp.poles = poles
        cindex = vf_tmp._build_cindex(poles)

        H_fit = np.zeros((N_pairs, N_freq), dtype=complex)

        for k, (oi, ii) in enumerate(self.selected_pairs):
            residues = self.residues_matrix[k]
            H_k = np.zeros(N_freq, dtype=complex)

            for m in range(len(poles)):
                if cindex[m] == 0:           # real pole
                    H_k += residues[m] / (s_eval - poles[m])
                elif cindex[m] == 1:         # complex conjugate pair
                    p, r = poles[m], residues[m]
                    H_k += r / (s_eval - p) + r.conj() / (s_eval - p.conj())

            H_k += self.d_vector[k]
            if self.e_vector[k] != 0.0:
                H_k += self.e_vector[k] * s_eval

            # Restore delay
            tau = (self.delay_map or {}).get((oi, ii), 0.0)
            if tau != 0.0:
                H_k = restore_delay(freq_eval, H_k, tau)

            H_fit[k] = H_k

        return H_fit

    def rms_error_db(self) -> np.ndarray:
        """
        Compute RMS error in dB between fitted and original data.

        Only computes error within the fmax bandwidth used for fitting,
        since the model is only valid in that range.

        Returns:
            rms_errors: (N_pairs,) array of per-pair RMS errors in dB.
        """
        # Evaluate only in fitting bandwidth
        if self._fmax_used is not None:
            freq_eval = self.freq[self.freq <= self._fmax_used]
        else:
            freq_eval = self.freq

        H_fit   = self.evaluate(freq_eval)
        N_pairs = len(self.selected_pairs)
        rms_errors = np.zeros(N_pairs)

        # Mask for original data to match freq_eval
        if self._fmax_used is not None:
            mask = self.freq <= self._fmax_used
        else:
            mask = np.ones(len(self.freq), dtype=bool)

        for k, (oi, ii) in enumerate(self.selected_pairs):
            H_orig = self.s_active[mask, oi, ii]
            # RMS in linear domain (more meaningful)
            diff = np.abs(H_fit[k]) - np.abs(H_orig)
            rms_lin = np.sqrt(np.mean(diff**2))
            ref_rms = np.sqrt(np.mean(np.abs(H_orig)**2))
            rms_errors[k] = 20 * np.log10(max(rms_lin / ref_rms, 1e-15))

        return rms_errors

    def summary(self) -> None:
        """
        Print fitting summary: poles, delays, RMS errors per port pair.
        """
        if self.shared_poles is None:
            print("Model not fitted yet")
            return

        errors = self.rms_error_db()
        print(f"\n=== SParamModel Fitting Summary ===")
        print(f"File       : {self.filename}")
        print(f"Poles      : {len(self.shared_poles)}")
        print(f"Pairs      : {len(self.selected_pairs)}")
        print(f"Diff mode  : {self._is_differential}")
        print(f"")
        for k, (oi, ii) in enumerate(self.selected_pairs):
            tau = (self.delay_map or {}).get((oi, ii), 0.0)
            print(
                f"  S({oi + 1},{ii + 1}): RMS={errors[k]:.3f} dB, "
                f"delay={tau * 1e9:.3f} ns"
            )
        print(f"  Mean RMS : {float(np.mean(errors)):.3f} dB")
