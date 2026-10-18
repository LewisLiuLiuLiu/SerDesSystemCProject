#!/usr/bin/env python3
"""
Generate a SystemC-AMS state-space channel config from a Touchstone file
(2026-10, delay-aware, per-pair fitting).

Pipeline per (output, input) pair:
  1. delay-aware Vector Fitting with the s-proportional term constrained to zero
     (physically required for S-parameters: |S(inf)| is finite; the free-e fit
     produced huge cancelling terms that the C++ finite-difference E handling
     could not reproduce faithfully)
  2. exact block-diagonal realization of H(s) = sum r/(s-p) + d
     (real poles -> 1-state blocks, conjugate pairs -> 2-state blocks)
  3. exact ZOH discretization (Ad, Bd) for the requested simulation timestep
  4. JSON matching ChannelSParamTdf's full_model schema: shared A/B/C/D,
     port_pairs, delay_s, optional discrete_state_space

Note on delays: the C++ channel applies `delay_s` per output using the
(active_output, active_inputs[0]) pair; per-input skew inside a MIMO mix is not
modeled. For a single driven input this is exact.

For shared-pole MIMO fitting with all pairs jointly, see SParamModel in
vector_fitting.py; the per-pair route here is used when maximum accuracy for a
specific thru path matters (e.g. the S21 channel check).

Usage:
  python3 scripts/s4p_to_state_space.py peters.s4p --out config/peters_ss_vf12.json
      [--ports 0,1] [--order 12] [--dt 1e-11] [--allow-e]

Requires: numpy, scipy, scikit-rf.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from scipy.linalg import expm


def pair_realization(poles, residues, d):
    """Realize H(s)=sum r/(s-p)+d as (A,B,C) real matrices.

    Real pole p, residue r:             A=[p],          B=[1],   C=[r]
    Conjugate pair p=a+jb (b>0), r=c+jd:
        A=[[2a, -(a^2+b^2)], [1, 0]], B=[1,0]^T,
        C=[2c, -2*(c*a + d*b)]
    (denominator s^2-2as+a^2+b^2, numerator 2c*s - 2(ca+db))
    """
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    n = len(poles)
    blocks = []
    used = np.zeros(n, dtype=bool)
    tol = 1e-6

    for i in range(n):
        if used[i]:
            continue
        p = poles[i]
        r = residues[i]
        if abs(p.imag) < tol:
            blocks.append((np.array([[p.real]]), np.array([1.0]), np.array([r.real])))
            used[i] = True
        else:
            j = None
            for k in range(i + 1, n):
                if not used[k] and abs(poles[k] - np.conj(p)) < tol * max(1.0, abs(p)):
                    j = k
                    break
            if j is None:
                raise SystemExit(f"unpaired complex pole {p} in fit result")
            a, b = p.real, abs(p.imag)
            if p.imag < 0:
                p = poles[j]
                r = residues[j]
            c, dd = r.real, r.imag
            A = np.array([[2.0 * a, -(a * a + b * b)],
                          [1.0, 0.0]])
            B = np.array([1.0, 0.0])
            C = np.array([2.0 * c, -2.0 * (c * a + dd * b)])
            blocks.append((A, B, C))
            used[i] = True
            used[j] = True

    n_states = sum(A.shape[0] for A, _, _ in blocks)
    A_full = np.zeros((n_states, n_states))
    B_full = np.zeros((n_states, 1))
    C_full = np.zeros((1, n_states))
    row = 0
    for A, B, C in blocks:
        m = A.shape[0]
        A_full[row:row + m, row:row + m] = A
        B_full[row:row + m, 0] = B
        C_full[0, row:row + m] = C
        row += m
    return A_full, B_full, C_full, float(d)


def exact_zoh(A, B, dt):
    """Exact zero-order-hold discretization."""
    Ad = expm(A * dt)
    n = A.shape[0]
    aug = np.zeros((n + B.shape[1], n + B.shape[1]))
    aug[:n, :n] = A * dt
    aug[:n, n:] = B * dt
    Bd = expm(aug)[:n, n:]
    return Ad, Bd


def main():
    parser = argparse.ArgumentParser(
        description="Touchstone -> delay-aware state-space channel config")
    parser.add_argument("touchstone")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ports", default="",
                        help="Comma-separated port indices to model (default: all)")
    parser.add_argument("--order", type=int, default=12,
                        help="VF poles per (out,in) pair (default 12)")
    parser.add_argument("--dt", type=float, default=1e-11,
                        help="Discretization timestep in seconds (default 1e-11)")
    parser.add_argument("--allow-e", action="store_true",
                        help="Allow the (unphysical) s-proportional fit term")
    parser.add_argument("--active-inputs", default="",
                        help="Active input physical ports (default: all modeled ports)")
    parser.add_argument("--active-outputs", default="",
                        help="Active output physical ports (default: all modeled ports)")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vector_fitting import VectorFitting

    import skrf
    nw = skrf.Network(args.touchstone)
    nports_all = nw.nports
    ports = ([int(p) for p in args.ports.split(",")] if args.ports
             else list(range(nports_all)))
    for p in ports:
        if p >= nports_all:
            raise SystemExit(f"port {p} out of range (file has {nports_all})")
    n = len(ports)
    active_in = ([int(p) for p in args.active_inputs.split(",")] if args.active_inputs
                 else list(range(n)))
    active_out = ([int(p) for p in args.active_outputs.split(",")] if args.active_outputs
                  else list(range(n)))
    for p in active_in + active_out:
        if p < 0 or p >= n:
            raise SystemExit(f"active port {p} outside modeled ports {ports} "
                             "(indices are positions in the --ports list)")
    freq = nw.frequency.f
    s = 2j * np.pi * freq
    print(f"Touchstone: {args.touchstone}  ({nports_all} ports)")
    print(f"Modeling ports {ports} -> {n} x {n} matrix, order {args.order}/pair")
    print(f"Active: inputs {active_in} (TX pair), outputs {active_out} (RX pair)")

    # Fit only the pairs used by the active I/O sets (other rows/cols stay zero)
    pairs = [(m, i) for m in active_out for i in active_in]
    blocks = []
    total_states = 0
    delays = []
    delay_of = {}
    dc_mag = {}
    d_in = np.zeros((n, n))
    print(f"\n{'pair':>10} {'delay_ns':>9} {'rel_rms':>9} {'medDB*':>7} {'states':>7}")
    print("-" * 50)
    for (m, i) in pairs:
        H = nw.s[:, ports[m], ports[i]]
        vf = VectorFitting(order=args.order)
        r = vf.fit(s, H, force_e_zero=not args.allow_e)
        A, B, C, d = pair_realization(r["poles"], r["residues"], r["d"])
        blocks.append((m, i, A, B, C))
        total_states += A.shape[0]
        delays.append(r["delay"])
        delay_of[(m, i)] = r["delay"]
        dc_mag[(m, i)] = np.abs(H[0])
        d_in[m, i] = d
        print(f"S{ports[m]}{ports[i]:<8} {r['delay']*1e9:9.3f} {r['rel_rms']:9.2e} "
              f"{r['median_error_db']:7.2f} {A.shape[0]:7d}")

    # C++ applies one delay per output, looked up at (output, active_inputs[0]).
    # Use the *dominant* pair's delay for that key (e.g. the thru path rather
    # than a weak FEXT pair); the other pairs into the same output differ by
    # only a few tens of ps at <=1% of the signal amplitude.
    if active_in:
        for m in active_out:
            candidates = [(dc_mag[(m, i)], i) for i in active_in if (m, i) in dc_mag]
            if not candidates:
                continue
            _, dom_i = max(candidates)
            key = (m, active_in[0])
            if key in delay_of and abs(delay_of[key] - delay_of[(m, dom_i)]) > 1e-15:
                delay_of[key] = delay_of[(m, dom_i)]
                print(f"  [delay] output {m}: using dominant path ({m},{dom_i}) "
                      f"delay {delay_of[key]*1e9:.3f} ns for the C++ lookup key {key}")
        delays = [delay_of[(m, i)] for (m, i) in pairs]

    A_full = np.zeros((total_states, total_states))
    B_full = np.zeros((total_states, n))
    C_full = np.zeros((n, total_states))
    row = 0
    for (m, i, A, B, C) in blocks:
        k = A.shape[0]
        A_full[row:row + k, row:row + k] = A
        B_full[row:row + k, i] = B.ravel()
        C_full[m, row:row + k] = C.ravel()
        row += k
    D_full = d_in
    E_full = np.zeros((n, n))          # e forced to zero by default

    Ad, Bd = exact_zoh(A_full, B_full, args.dt)

    cfg = {
        "version": "2.0",
        "method": "state_space",
        "fs": 1.0 / args.dt,
        "full_model": {
            "n_diff_ports": n,
            "n_outputs": n,
            "n_states": total_states,
            "port_pairs": [[m, i] for (m, i) in pairs],
            "delay_s": [float(x) for x in delays],
            "state_space": {
                "A": A_full.tolist(),
                "B": B_full.tolist(),
                "C": C_full.tolist(),
                "D": D_full.tolist(),
                "E": E_full.tolist(),
            },
            "discrete_state_space": {
                "dt": args.dt,
                "A": Ad.tolist(),
                "B": Bd.tolist(),
                "C": C_full.tolist(),
                "D": D_full.tolist(),
                "E": E_full.tolist(),
            },
        },
        "port_config": {
            "active_inputs": active_in,
            "active_outputs": active_out,
        },
        "metadata": {
            "source": os.path.basename(args.touchstone),
            "generator": "scripts/s4p_to_state_space.py (delay-aware VF, e=0)",
            "order_per_pair": args.order,
            "ports": ports,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print("-" * 50)
    print(f"Total states: {total_states} ({len(pairs)} pairs)")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
