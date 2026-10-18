#!/usr/bin/env python3
"""
BER bathtub / noise sweep for nrz_ber_tb (2026-10).

Runs nrz_ber_tb for each sampling phase (cdr.sample_point) or sampler noise
sigma, computes offline PRBS31 BER, and saves a CSV + semilogy plot.

Examples:
  # phase bathtub (fixed noise from the overlay)
  python3 scripts/ber_bathtub.py --channel config/peters_ss_channel_ts4.json \
      --overlay config/ts4_a2_ber_diff.json --sweep phase \
      --start 0.35 --stop 0.95 --step 0.05 --ui 20000 \
      --out output_eye/ber_bathtub_ts4.png

  # noise sweep at the optimum phase
  python3 scripts/ber_bathtub.py --channel config/peters_ss_channel_ts4.json \
      --overlay config/ts4_a2_ber_diff.json --sweep noise \
      --start 0.15 --stop 0.55 --step 0.10 --phase 0.65 --ui 20000 \
      --out output_eye/ber_noise_ts4.png

Notes:
- The overlay should have sampler noise enabled (for bathtubs) and the same
  FFE/equalizer settings for all points.
- sample_point must be swept per channel model: the optimum shifts by the
  fractional part of the model's transport delay (integer UI is absorbed by
  the offline lag search).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "ts3"))
from ber_from_decisions import prbs31_bits, window  # noqa: E402


def ber_from_bits(bits, seed=12345, lag_max=300):
    prbs = prbs31_bits(len(bits) + lag_max + 10, seed)
    best = None
    for inv in (False, True):
        b = (1 - bits) if inv else bits
        for lag in range(0, lag_max + 1):
            r = window(b, prbs, lag, lag)
            if r is None:
                continue
            errs, n = r
            if best is None or errs / n < best[0]:
                best = (errs / n, errs, n, lag, inv)
    return best


def run_point(channel, overlay, ui, decisions_prefix, sweep, value, phase, tmpdir):
    ov = copy.deepcopy(overlay)
    if sweep == "phase":
        ov.setdefault("cdr", {})["sample_point"] = float(value)
    else:
        ov.setdefault("rx", {}).setdefault("sampler", {})
        ov["rx"]["sampler"]["noise_sigma"] = float(value)
        ov["rx"]["sampler"]["noise_enable"] = True
        if phase is not None:
            ov.setdefault("cdr", {})["sample_point"] = float(phase)
    ov_path = os.path.join(tmpdir, "overlay.json")
    with open(ov_path, "w") as f:
        json.dump(ov, f)
    cmd = [os.path.join(ROOT, "build", "bin", "nrz_ber_tb"), "ss", channel,
           "--config", ov_path, "-d", str(ui), "-o", decisions_prefix]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=1800)
    arr = np.loadtxt(decisions_prefix + "_decisions.csv", delimiter=",",
                     skiprows=1, ndmin=2)
    bits = arr[:, 1].astype(np.uint8)
    ber, errs, n, lag, inv = ber_from_bits(bits)
    return ber, errs, n, lag, inv


def main():
    ap = argparse.ArgumentParser(description="BER bathtub / noise sweep")
    ap.add_argument("--channel", required=True, help="state-space channel JSON")
    ap.add_argument("--overlay", required=True, help="SystemParams overlay JSON")
    ap.add_argument("--sweep", choices=["phase", "noise"], required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--step", type=float, required=True)
    ap.add_argument("--ui", type=int, default=20000)
    ap.add_argument("--phase", type=float, default=None,
                    help="fixed sample_point for a noise sweep (recommended)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--tag", default="", help="prefix for temporary decision files")
    args = ap.parse_args()

    overlay = json.load(open(args.overlay))
    n_pts = int(round((args.stop - args.start) / args.step)) + 1
    values = [round(args.start + i * args.step, 4) for i in range(n_pts)]
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for v in values:
            prefix = os.path.join(tmpdir, f"{args.tag}_{args.sweep}_{v}")
            ber, errs, n, lag, inv = run_point(
                args.channel, overlay, args.ui, prefix,
                args.sweep, v, args.phase, tmpdir)
            print(f"{args.sweep}={v:.4g}: BER={ber:.3e} "
                  f"({errs}/{n}, lag={lag}, inv={inv})")
            rows.append((v, ber, errs, n, lag, inv))

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("value,ber,errors,n,lag,inv\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6e},{r[2]},{r[3]},{r[4]},{int(r[5])}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array([r[0] for r in rows])
    ys = np.array([max(r[1], 1.0 / max(r[3], 1)) for r in rows])  # 0 -> 1/n mark
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(xs, ys, "o-")
    ax.set_xlabel("Sampling phase [UI]" if args.sweep == "phase"
                  else "Sampler noise sigma [V]")
    ax.set_ylabel("BER")
    ax.set_title(f"BER {'bathtub' if args.sweep == 'phase' else 'vs noise'} — "
                 f"{os.path.basename(args.channel)}")
    ax.grid(True, which="both", alpha=0.3)
    if args.sweep == "phase" and args.phase is None:
        i = int(np.argmin([r[1] for r in rows]))
        ax.axvline(xs[i], color="r", ls="--", lw=1,
                   label=f"optimum ~{xs[i]:.2f} UI, BER {rows[i][1]:.2e}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
