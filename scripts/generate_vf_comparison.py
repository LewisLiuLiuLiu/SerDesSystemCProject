#!/usr/bin/env python3
"""
Vector Fitting quality report for a real Touchstone file (2026-10 rewrite).

Replaces the previous version, which drew synthetic np.random "fitting errors"
for paper figures. Everything here is computed from the given S-parameter file:

  * delay-aware Vector Fitting (delay estimated from phase slope, de-embedded,
    refit, re-applied for evaluation)
  * order sweep with relative-RMS and dB error metrics
  * usable-band statistics (|S21| >= max|S21|/10, i.e. within 20 dB of peak)
  * measured-vs-fitted magnitude and unwrapped phase

Usage:
  python3 scripts/generate_vf_comparison.py <file.s4p> [--sparam S21]
      [--orders 4,6,8,12,16,20] [--out output_eye]

Outputs (default output_eye/):
  vf_error_vs_order.png   - relative RMS / median dB error vs order
  vf_fit_vs_measured.png  - measured vs fitted |S| and phase (best order)

Requires: numpy, matplotlib, scikit-rf.
"""

import argparse
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sparam(path, sparam):
    """Load S-parameter from a Touchstone file. Returns freq (Hz), H (complex)."""
    import skrf
    nw = skrf.Network(path)
    a = int(sparam[1]) - 1
    b = int(sparam[2]) - 1
    if a >= nw.nports or b >= nw.nports:
        raise SystemExit(f"{sparam} out of range for an {nw.nports}-port file")
    return nw.frequency.f.copy(), nw.s[:, a, b].copy()


def usable_mask(mag_db, rel_db=-20.0):
    """Points within rel_db of the peak magnitude."""
    return mag_db >= (np.max(mag_db) + rel_db)


def main():
    parser = argparse.ArgumentParser(
        description="Delay-aware Vector Fitting quality report (real data only)")
    parser.add_argument("touchstone", help="Touchstone file (.s2p/.s4p)")
    parser.add_argument("--sparam", default="S21", help="S-parameter to fit (default S21)")
    parser.add_argument("--orders", default="4,6,8,12,16,20",
                        help="Comma-separated pole counts (default 4,6,8,12,16,20)")
    parser.add_argument("--out", default="output_eye", help="Output directory")
    args = parser.parse_args()

    logging.disable(logging.WARNING)   # VF logs are noisy for sweeps
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vector_fitting import VectorFitting, evaluate_fit

    if not os.path.exists(args.touchstone):
        raise SystemExit(f"File not found: {args.touchstone}")

    freq, H = load_sparam(args.touchstone, args.sparam)
    s = 2j * np.pi * freq
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)
    band = usable_mask(mag_db)
    orders = [int(o) for o in args.orders.split(",") if o.strip()]

    print("=" * 66)
    print(f"VF quality report: {args.touchstone}  ({args.sparam})")
    print(f"points={len(freq)}  band={freq[0]/1e9:.3f}-{freq[-1]/1e9:.3f} GHz")
    print("=" * 66)
    print(f"{'order':>5} {'delay_ns':>9} {'relRSS':>9} {'medDB*':>7} "
          f"{'p90DB*':>7} {'maxDB*':>7}")
    print("-" * 66)

    rows = []
    best = None
    for order in orders:
        vf = VectorFitting(order=order)
        r = vf.fit(s, H)
        Hm = evaluate_fit(s, r)
        dberr = np.abs(20*np.log10(np.abs(Hm) + 1e-20) - 20*np.log10(np.abs(H) + 1e-20))
        med = float(np.median(dberr[band]))
        p90 = float(np.percentile(dberr[band], 90))
        mx = float(np.max(dberr[band]))
        rows.append(dict(order=order, delay=r["delay"], rel=r["rel_rms"],
                         med=med, p90=p90, max=mx, result=r, fit=Hm))
        print(f"{order:5d} {r['delay']*1e9:9.3f} {r['rel_rms']:9.3e} "
              f"{med:7.2f} {p90:7.2f} {mx:7.2f}")
        if best is None or r["rel_rms"] < best["rel"]:
            best = rows[-1]
    print("-" * 66)
    print("* dB error within the usable band (|S| >= max|S| - 20 dB), "
          f"{np.count_nonzero(band)} of {len(freq)} points")

    os.makedirs(args.out, exist_ok=True)

    # ---- Figure 1: error vs order ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    o = [row["order"] for row in rows]
    axes[0].semilogy(o, [row["rel"] * 100 for row in rows], "o-")
    axes[0].set_xlabel("Number of poles")
    axes[0].set_ylabel("Relative RMS error [%]")
    axes[0].set_title("Full-band relative RMS vs order")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(o, [row["med"] for row in rows], "o-", label="median")
    axes[1].plot(o, [row["p90"] for row in rows], "s--", label="90th percentile")
    axes[1].plot(o, [row["max"] for row in rows], "^:", label="max")
    axes[1].set_xlabel("Number of poles")
    axes[1].set_ylabel("dB error (usable band)")
    axes[1].set_title("Magnitude error within 20 dB of peak")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Delay-aware Vector Fitting: {os.path.basename(args.touchstone)} "
                 f"({args.sparam})", fontsize=13)
    fig.tight_layout()
    f1 = os.path.join(args.out, "vf_error_vs_order.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: measured vs fitted (best order) ----
    br = best
    f_ghz = freq / 1e9
    Hm = br["fit"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(f_ghz, mag_db, "k-", lw=1.8, label="measured")
    axes[0].plot(f_ghz, 20*np.log10(np.abs(Hm) + 1e-20), "r--", lw=1.4,
                 label=f"VF order {br['order']}")
    axes[0].axhline(np.max(mag_db) - 20, color="gray", ls=":", lw=1,
                    label="usable-band floor (-20 dB)")
    axes[0].set_ylabel("|S| [dB]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"{args.sparam}: measured vs fitted (tau = {br['delay']*1e9:.3f} ns, "
        f"median {br['med']:.2f} dB in band)")

    ph_m = np.unwrap(np.angle(H))
    ph_f = np.unwrap(np.angle(Hm))
    axes[1].plot(f_ghz, ph_m, "k-", lw=1.5, label="measured")
    axes[1].plot(f_ghz, ph_f, "r--", lw=1.2, label="fitted")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("Unwrapped phase [rad]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(args.out, "vf_fit_vs_measured.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {f1}\nSaved: {f2}")
    print("All plotted values are computed from the input file — no synthetic data.")


if __name__ == "__main__":
    main()
