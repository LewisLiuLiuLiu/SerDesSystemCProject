#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse S1 (zero-jitter sigma scan) decisions against v3 E-group values."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_from_decisions import prbs31_bits, window  # noqa: E402

RUNS = [("022", 0.22, "E1_低噪声", 9.8944e-06),
        ("030", 0.30, "E2_噪声0.30V", 6.6299e-04),
        ("035", 0.35, "E3_噪声0.35V", 2.7622e-03),
        ("050", 0.50, "E4_高噪声", 2.4646e-02)]
E10 = 2.7604e-03  # zero-jitter, sigma=0.35
ROOT = "/tmp/serdes_ts3_e101025/build/run_ts3/s1"
SEED = 12345
LAG = 2  # from the zero-noise anchor

def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return a[:, 1].astype(np.uint8) if a.ndim == 2 else np.array([a], dtype=np.uint8)[:, 1]

rows = []
for tag, sn, v3name, v3ber in RUNS:
    p = os.path.join(ROOT, tag, f"ts3_s1_{tag}_decisions.csv")
    if not os.path.exists(p):
        rows.append(dict(tag=tag, sigma=sn, status="missing"))
        continue
    bits = load(p)
    prbs = prbs31_bits(len(bits) + 20, SEED)
    r = window(bits, prbs, LAG, 0)
    rs = window(bits, prbs, LAG, 5000)
    ber = r[0] / r[1]
    bers = rs[0] / rs[1]
    nerr = r[0]
    stat_rel = (nerr ** -0.5) if nerr > 0 else float("inf")
    band = max(0.10, 3 * stat_rel)
    ratio = ber / v3ber
    rows.append(dict(tag=tag, sigma=sn, n=int(r[1]), errors=int(nerr), ber=ber,
                     ber_steady=bers, v3=v3name, v3ber=v3ber, ratio=ratio,
                     stat_rel=stat_rel, band=band,
                     pass_band=bool(abs(ratio - 1) <= band)))
    print(f"σ={sn:.2f}: N={r[1]} errs={nerr} BER={ber:.4e} (steady {bers:.4e}) "
          f"v3({v3name})={v3ber:.4e} ratio={ratio:.3f} band=±{band:.3f} "
          f"{'PASS' if abs(ratio-1)<=band else 'FAIL'}")

# E10 zero-jitter comparison at sigma=0.35
p = os.path.join(ROOT, "035", "ts3_s1_035_decisions.csv")
if os.path.exists(p):
    bits = load(p)
    prbs = prbs31_bits(len(bits) + 20, SEED)
    r = window(bits, prbs, LAG, 0)
    ber = r[0] / r[1]
    nerr = r[0]
    stat_rel = (nerr ** -0.5) if nerr > 0 else float("inf")
    band = max(0.10, 3 * stat_rel)
    ratio = ber / E10
    print(f"σ=0.35 vs v3 E10 (zero jitter): BER={ber:.4e} ratio={ratio:.3f} "
          f"band=±{band:.3f} {'PASS' if abs(ratio-1)<=band else 'FAIL'}")
    print(json.dumps(dict(e10=dict(ber=ber, ratio=ratio, band=band,
                                   pass_band=bool(abs(ratio-1) <= band))), indent=2))

print(json.dumps(rows, indent=2))
