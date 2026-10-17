#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-S3 / A3 — 离线 BER：最小 TB 导出的每 UI 判决 vs PRBS31 重生成。

判决来源：nrz_ber_tb 的 UiDecisionRecorder（CDR data_trigger 频闪，1/UI）。
判定：bits[k] 对应 PRBS 位 k-L（L = 数据通路整数 UI 延迟，由零噪声锚点扫描确定；
      同时扫描极性翻转）。统计窗口：全对齐窗口 + 去前 2000/5000 UI（采集瞬态）窗口。
"""

import argparse
import json

import numpy as np


def prbs31_bits(n, seed):
    """Replicates WaveGenerationTdf: initialize() consumes one bit, then one bit per UI."""
    mask = 0x7FFFFFFF
    state = (0x7FFFFFFF ^ (seed & mask)) & mask
    if state == 0:
        state = 0x7FFFFFFF

    def gen():
        nonlocal state
        fb = ((state >> 30) ^ (state >> 27)) & 1
        state = ((state << 1) | fb) & mask
        return state & 1

    gen()  # consumed in initialize()
    return np.array([gen() for _ in range(n)], dtype=np.uint8)


def window(bits, prbs, lag, start_ui):
    k0 = max(int(start_ui), int(lag))
    if k0 >= len(bits):
        return None
    errs = int(np.sum(bits[k0:] != prbs[k0 - lag: len(bits) - lag]))
    return errs, int(len(bits) - k0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--lag-max", type=int, default=300)
    ap.add_argument("--lag", type=int, default=None, help="fixed lag (skip search)")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arr = np.loadtxt(a.decisions, delimiter=",", skiprows=1, dtype=np.int64)
    bits = arr[:, 1].astype(np.uint8) if arr.ndim == 2 else np.array([arr], dtype=np.uint8)[:, 1]
    N = len(bits)
    prbs = prbs31_bits(N + a.lag_max + 10, a.seed)

    if a.lag is None:
        best = None
        for inv in (False, True):
            b = (1 - bits) if inv else bits
            for lag in range(0, a.lag_max + 1):
                r = window(b, prbs, lag, lag)
                if r is None:
                    continue
                errs, n = r
                if best is None or errs / n < best[0]:
                    best = (errs / n, errs, n, lag, inv)
        ber, errs, n, lag, inv = best
    else:
        lag, inv = a.lag, a.invert
        b = (1 - bits) if inv else bits
        errs, n = window(b, prbs, lag, lag)
        ber = errs / n

    b = (1 - bits) if inv else bits
    res = {"decisions": int(N), "lag": int(lag), "invert": bool(inv),
           "errors": int(errs), "n_compared": int(n), "ber": float(ber), "windows": {}}
    for skip in (0, 2000, 5000):
        r = window(b, prbs, lag, skip)
        if r:
            res["windows"][f"skip_first_{skip}_ui"] = {
                "errors": int(r[0]), "n": int(r[1]), "ber": float(r[0] / r[1])}

    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
