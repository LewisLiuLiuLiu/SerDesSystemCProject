#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-S3 / A2 — 由 peters s4p 重生成 SystemC 信道 JSON（F-016 裁决路径，v3）

方法（运行前固定）
  * 契约端口映射（formula_contract §2.2）：近端差分对 (1,3)、远端差分对 (2,4)；
    SDD21 = (S21 − S23 − S41 + S43)/2。
  * 工具：scripts/vector_fitting.py（SParamModel）单对拟合：
      load_snp → set_port_map([(1,3),(2,4)]) → to_differential()（2×2 Sdd）
      → fit(order=32, port_pairs=[(2,1)], extract_delay=True, fmax=15 GHz,
            n_iter_poles=8, weight=1/max(|H|,1e-4))。
    该工具 to_state_space() 在 N_inputs>1 的 MIMO 组合存在缺陷（B 各列相同、
    C 行按输出累积所有输入 residue）——本脚本只取单对结果，不受影响（见报告 §A2）。
  * e 项保留：C++ 侧已修复为使用 E 矩阵（channel_sparam.cpp: y += E·du/dt）。
  * delay_s 存档但 C++ 不参与运算（常数延迟，对 BER 无影响）。
  * 2×2 对称差分实现（nrz_link_tb: in0/in1=P/N，out0/out1=P/N）：
        B2=[B/2,-B/2], C2=[C;-C],
        D2=(d/2)[[1,-1],[-1,1]], E2=(e/2)[[1,-1],[-1,1]]
    则 H=(h0/2)[[1,-1],[-1,1]]，差模 (out0-out1)=h0·(in0-in1)，共模=0；h0=SDD21。

固定参数：ORDER=32, FMAX=15 GHz, FS_EXPORT=500 GHz。
"""

import argparse
import json
import os
import sys

import numpy as np

ORDER = 32
FMAX = 15e9
FS_EXPORT = 500e9


def eval_ss(A, B, C, D, E, f, delay=0.0):
    jw = 2j * np.pi * f
    X = np.linalg.solve(jw[:, None, None] * np.eye(A.shape[0])[None] - A[None],
                        np.broadcast_to(B[None], (len(f),) + B.shape))
    H = np.einsum("on,fni->foi", C, X)[:, 0, 0] + D[0, 0] + jw * E[0, 0]
    if delay:
        H = H * np.exp(-jw * delay)
    return H


def stats(H, ref, f):
    d_db = 20 * np.log10(np.abs(H)) - 20 * np.log10(np.abs(ref))
    ph = np.unwrap(np.angle(H / ref))
    coef = np.polyfit(f, ph, 1)
    resid = np.degrees(ph - np.polyval(coef, f))
    T = 100e-12
    tr = 20e-12
    pulse = T * np.sinc(f * T) * np.exp(-(np.pi * (tr / 2.326) * f) ** 2)
    num = np.trapezoid(np.abs(H - ref) ** 2 * np.abs(pulse) ** 2, f)
    den = np.trapezoid(np.abs(ref) ** 2 * np.abs(pulse) ** 2, f)
    return dict(mag_rms_db=float(np.sqrt(np.mean(d_db ** 2))),
                mag_max_db=float(np.max(np.abs(d_db))),
                ph_rms_deg=float(np.sqrt(np.mean(resid ** 2))),
                ph_max_deg=float(np.max(np.abs(resid))),
                energy_rel=float(np.sqrt(num / den)),
                delay_removed_ps=float(-coef[0] / (2 * np.pi) * 1e12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-report", default=None)
    args = ap.parse_args()

    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, scripts_dir)
    from vector_fitting import SParamModel  # noqa: E402

    import skrf as rf
    net = rf.Network(args.s4p)
    f = np.asarray(net.f, float).ravel()
    S = np.asarray(net.s, complex)
    band = (f >= 0.05e9) & (f <= FMAX)
    f = f[band]
    H_ref = 0.5 * (S[band, 1, 0] - S[band, 1, 2] - S[band, 3, 0] + S[band, 3, 2])

    model = SParamModel()
    model.load_snp(args.s4p)
    model.set_port_map(diff_pairs=[(1, 3), (2, 4)])
    model.to_differential()
    w = 1.0 / np.maximum(np.abs(H_ref), 1e-4)
    model.fit(order=ORDER, port_pairs=[(2, 1)], extract_delay=True,
              fmax=FMAX, n_iter_poles=8, weight=w)
    model.summary()

    ss = model.to_state_space()
    A = np.asarray(ss["A"], float)
    B = np.asarray(ss["B"], float)
    C = np.asarray(ss["C"], float)
    D = np.asarray(ss["D"], float)
    E = np.asarray(ss["E"], float)
    tau = float(ss["delay_map"].get((1, 0), 0.0))

    H_sto = eval_ss(A, B, C, D, E, f, delay=tau)
    H_cpp = eval_ss(A, B, C, D, E, f, delay=0.0)
    rep = dict(s4p=args.s4p, order=ORDER, fmax_GHz=FMAX / 1e9,
               n_states=int(A.shape[0]), delay_s=tau, e_value=float(E[0, 0]),
               weight="1/max(|SDD21|, 1e-4)",
               fit_stored_with_delay=stats(H_sto, H_ref, f),
               cpp_semantics_ABCED_no_delay=stats(H_cpp, H_ref, f))
    print("[gen] fit quality (stored, delay restored):",
          json.dumps(rep["fit_stored_with_delay"], indent=2))
    print("[gen] fit quality (C++ semantics: A,B,C,D,E; delay ignored):",
          json.dumps(rep["cpp_semantics_ABCED_no_delay"], indent=2))

    # symmetric differential 2x2 realization
    B2 = np.hstack([B / 2.0, -B / 2.0])
    C2 = np.vstack([C, -C])
    S2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    D2 = (D[0, 0] / 2.0) * S2
    E2 = (E[0, 0] / 2.0) * S2

    out = {
        "version": "3.0",
        "method": "state_space",
        "fs": FS_EXPORT,
        "full_model": {
            "n_diff_ports": 2,          # C++ parses this as n_inputs
            "n_outputs": 2,
            "n_states": int(A.shape[0]),
            "port_pairs": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "delay_s": [tau, tau, tau, tau],
            "state_space": {
                "A": A.tolist(), "B": B2.tolist(), "C": C2.tolist(),
                "D": D2.tolist(), "E": E2.tolist(),
            },
        },
        "port_config": {"active_inputs": [0, 1], "active_outputs": [0, 1]},
        "metadata": {
            "n_raw_ports": 4,
            "is_differential": True,
            "diff_pairs": [[1, 3], [2, 4]],
            "freq_min_ghz": 0.05,
            "freq_max_ghz": FMAX / 1e9,
            "contract_port_mapping": "(S21 - S23 - S41 + S43)/2",
            "realization": ("symmetric differential pair: H=(h0/2)[[1,-1],[-1,1]]; "
                            "differential transfer out0-out1 = h0*(in0-in1); common mode = 0"),
            "generator": "scripts/ts3/gen_peters_ss.py (v3, e kept; C++ E implemented)",
            "vector_fitting_order": ORDER,
            "delay_s": tau,
            "fit_report": rep,
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[gen] written:", args.out)
    if args.fit_report:
        with open(args.fit_report, "w") as fh:
            json.dump(rep, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
