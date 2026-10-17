#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 helper: sampler-input vertical eye opening vs sampling phase.

Input: nrz_link_tb `*_dfe.csv` (time_s, voltage_p, voltage_n, voltage_diff).
The waveform is folded at UI; for each phase bin the vertical opening is
q01(high cluster) - q99(low cluster). The best phase is the centre of the
widest vertical opening, used to set CDR sample_point (frozen PI).

This is a diagnostic/calibration helper; it does not change contract semantics.
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ui-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--skip-ui", type=float, default=10.0,
                    help="skip the startup transient (default 10 UI)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    t = d["time_s"]
    v = d["voltage_diff"]
    ui = a.ui_ps * 1e-12
    keep = t >= a.skip_ui * ui
    t, v = t[keep], v[keep]
    ph = np.mod(t / ui, 1.0)
    bins = np.round(ph * a.bins).astype(int) % a.bins

    rows = []
    for b in range(a.bins):
        x = v[bins == b]
        hi, lo = x[x > 0], x[x < 0]
        if len(hi) < 5 or len(lo) < 5:
            rows.append(dict(bin=b, phase_ps=b * a.ui_ps / a.bins,
                             opening=0.0, hi_p1=0.0, lo_p99=0.0))
            continue
        q_hi = float(np.percentile(hi, 1))
        q_lo = float(np.percentile(lo, 99))
        rows.append(dict(bin=b, phase_ps=b * a.ui_ps / a.bins,
                         opening=q_hi - q_lo, hi_p1=q_hi, lo_p99=q_lo))

    best = max(rows, key=lambda r: r["opening"])
    out = dict(csv=a.csv, ui_ps=a.ui_ps, bins=a.bins, skip_ui=a.skip_ui,
               samples=int(len(t)), v_min=float(v.min()), v_max=float(v.max()),
               best_phase_ps=best["phase_ps"], opening=best["opening"],
               sample_point=best["phase_ps"] / a.ui_ps,
               phases=rows)
    print(json.dumps({k: out[k] for k in ("csv", "samples", "v_min", "v_max",
                                          "best_phase_ps", "opening", "sample_point")},
                     indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
