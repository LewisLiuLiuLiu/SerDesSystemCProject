#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse S3 jitter-subset decisions (SystemC per-edge EDGE-01) vs v3 E5-E9/E11
(per-symbol conditional approximation — formal same-sequence ratio belongs to T-P6)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_from_decisions import prbs31_bits, window  # noqa: E402

ROOT = "/tmp/serdes_ts3_phase2/build/run_ts3/s3"
SEED = 12345
LAG = 2
V3 = {
    "E5": ("RJ 1ps", 2.7673e-03),
    "E6": ("RJ 2ps", 2.7881e-03),
    "E7": ("DCD 0.5ps", 2.7639e-03),
    "E8": ("SJ 1ps", 2.7655e-03),
    "E9": ("RJ0.5+DCD0.5+SJ1", 2.7673e-03),
    "E11": ("RJ10+DCD5+SJ5", 4.8091e-03),
}

def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return a[:, 1].astype(np.uint8) if a.ndim == 2 else np.array([a], dtype=np.uint8)[:, 1]

rows = []
for tag, (desc, v3ber) in V3.items():
    p = os.path.join(ROOT, tag, f"ts3_s3_{tag}_decisions.csv")
    if not os.path.exists(p):
        rows.append(dict(tag=tag, status="missing"))
        continue
    bits = load(p)
    prbs = prbs31_bits(len(bits) + 20, SEED)
    r = window(bits, prbs, LAG, 0)
    rs = window(bits, prbs, LAG, 5000)
    ber, nerr = r[0] / r[1], r[0]
    stat_rel = (nerr ** -0.5) if nerr > 0 else float("inf")
    band = max(0.10, 3 * stat_rel)
    ratio = ber / v3ber
    rows.append(dict(tag=tag, desc=desc, n=int(r[1]), errors=int(nerr), ber=ber,
                     ber_steady=rs[0] / rs[1], v3ber=v3ber, ratio=ratio,
                     stat_rel=stat_rel, band=band))
    print(f"{tag:3s} {desc:20s} N={r[1]} errs={nerr} BER={ber:.4e} (steady {rs[0]/rs[1]:.4e}) "
          f"v3(per-symbol)={v3ber:.4e} ratio={ratio:.3f} band=±{band:.3f}")

print(json.dumps(rows, indent=2))
