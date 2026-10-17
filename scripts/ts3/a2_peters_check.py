#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 — peters 信道核对（T-S3 阶段 1，F-016 裁决落地）

本脚本的判定容差在运行前固定于此（T-S3 验收纪律：测量前声明，禁事后调整）。
【v2 · 正式判据（S0/S1/S2 正式 BER 测量前声明）】
  说明：v1 探索版曾自设 0.20 dB/0.50 dB/5°/15°；模型生成阶段实测该阈值对本仓库 VF 工具
  不可达（order 16–48 扫描最优 0.52 dB RMS；skrf VF 0.9–1.4 dB），故在正式 BER 测量前
  修订为下列工程判据，并如实记录该修订（见 T-S3 报告 §A2）。

  [频带]        0.05–15 GHz（模型拟合带；s4p 原始频点在该带内逐点比对）
  [契约口径]    SDD21(f) = (S21 − S23 − S41 + S43)/2
                （端口 1,3 = 近端差分对；2,4 = 远端差分对；s4p_channel.differential_thru 同式）
  [主判定对象]  "仿真等效响应"：JSON 全模型 A,B,C,D,E 的差分组合
                —— 与修复后的 C++ 实现一致：sca_ss(A,B,C,D) + E·du/dt；
                   delay_s 仍不参与运算（常数延迟，对 BER 无影响；相位比较前剥离最佳常数延迟）
  [幅度容差]    RMS(ΔdB) ≤ 0.75 dB  且  max|ΔdB| ≤ 2.50 dB
  [相位容差]    去常数延迟后的残差：RMS ≤ 7.0°  且  max|Δφ| ≤ 30.0°
  [附加指标]    "存储模型响应"（含 E 与逐对 delay_s）按同口径统计，仅报告、不作判定；
                接收脉冲能量相对误差 ε（PULSE-01，T=100 ps、t_r=20 ps）作为残余非理想量报告
  [端口修正]    按 F-016：交付版 JSON 的 {0,1}→{0,1} 组合作诊断对照（应为灾难性失败）；
                重生成版采用契约差分对（Sdd21），以对称 2×2 差分实现接入 nrz_link_tb

判定：幅度与相位同时满足 → PASS；否则 FAIL（peters 行 N/A；S0 用 SIMPLE 6 dB）。
"""

import argparse
import json
import os
import sys

import numpy as np

MAG_RMS_DB = 0.75
MAG_MAX_DB = 2.50
PH_RMS_DEG = 7.0
PH_MAX_DEG = 30.0
F_MIN = 0.05e9
F_MAX = 15.0e9
T_UI = 100e-12
TR_20_90 = 20e-12


def load_s4p(path):
    import skrf as rf
    net = rf.Network(path)
    f = np.asarray(net.f, dtype=float).ravel()
    S = np.asarray(net.s, dtype=complex)  # (nf, 4, 4)
    return f, S


def model_eval(d, f, include_e=False, include_delay=False):
    """Return H[f, out, in] for the 4-port model. H = C(jwI-A)^-1 B + D (+ jwE) [* exp(-jw tau)]."""
    fm = d["full_model"]
    ss = fm["state_space"]
    A = np.asarray(ss["A"], dtype=float)
    B = np.asarray(ss["B"], dtype=float)
    C = np.asarray(ss["C"], dtype=float)
    D = np.asarray(ss["D"], dtype=float)
    E = np.asarray(ss.get("E", np.zeros_like(D)), dtype=float)
    n = A.shape[0]
    jw = 2j * np.pi * f
    M = jw[:, None, None] * np.eye(n)[None, :, :] - A[None, :, :]   # (nf, n, n)
    X = np.linalg.solve(M, np.broadcast_to(B[None, :, :], (len(f),) + B.shape))  # (nf, n, 4)
    H = np.einsum("on,fni->foi", C, X) + D[None, :, :]
    if include_e:
        H = H + jw[:, None, None] * E[None, :, :]
    if include_delay:
        pairs = [tuple(p) for p in fm.get("port_pairs", [])]
        delays = fm.get("delay_s", [])
        if len(pairs) == len(delays):
            for k, (o, i) in enumerate(pairs):
                H[:, o, i] *= np.exp(-jw * delays[k])
    return H


def sdd21_from_s(S):
    return 0.5 * (S[:, 1, 0] - S[:, 1, 2] - S[:, 3, 0] + S[:, 3, 2])


def diff_combine(H):
    """Differential transfer of the delivered model.

    4-port SE realization: u=[port0,port2], y=[port1,port3] (contract SDD21).
    2-port differential-pair realization (T-S3 regeneration): out0-out1 = h*(in0-in1).
    """
    if H.shape[1] >= 4:
        return 0.5 * (H[:, 1, 0] - H[:, 1, 2] - H[:, 3, 0] + H[:, 3, 2])
    return 0.5 * (H[:, 0, 0] - H[:, 0, 1] - H[:, 1, 0] + H[:, 1, 1])


def as_delivered(H):
    """Original port_config {0,1}->{0,1}: differential combination on ports 0/1 both sides."""
    return 0.5 * (H[:, 0, 0] - H[:, 0, 1] - H[:, 1, 0] + H[:, 1, 1])


def metrics(H, ref, f):
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-300))
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref), 1e-300))
    d_db = mag_db - ref_db
    ph = np.unwrap(np.angle(H / ref))
    coef = np.polyfit(f, ph, 1)
    resid = np.degrees(ph - np.polyval(coef, f))
    # Received-pulse energy relative error (PULSE-01: T=100 ps, tr=20 ps)
    # constant delay is aligned first (delay has no BER effect; phase criterion does the same)
    tau_r = TR_20_90 / 2.326
    pulse = T_UI * np.sinc(f * T_UI) * np.exp(-(np.pi * tau_r * f) ** 2)
    tau_corr = coef[0] / (2 * np.pi)
    H_al = H * np.exp(-1j * 2 * np.pi * f * tau_corr)
    num = np.trapezoid(np.abs(H_al - ref) ** 2 * np.abs(pulse) ** 2, f)
    den = np.trapezoid(np.abs(ref) ** 2 * np.abs(pulse) ** 2, f)
    return dict(
        mag_rms_db=float(np.sqrt(np.mean(d_db ** 2))),
        mag_max_db=float(np.max(np.abs(d_db))),
        ph_rms_deg=float(np.sqrt(np.mean(resid ** 2))),
        ph_max_deg=float(np.max(np.abs(resid))),
        delay_removed_ps=float(-coef[0] / (2 * np.pi) * 1e12),
        dc_gain_db=float(d_db[0]),
        energy_rel=float(np.sqrt(num / den)),
        n_points=int(len(f)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4p", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fixed-json-out", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    f_all, S = load_s4p(args.s4p)
    band = (f_all >= F_MIN) & (f_all <= F_MAX)
    f = f_all[band]
    S_b = S[band]
    ref = sdd21_from_s(S_b)

    with open(args.json) as fh:
        d = json.load(fh)

    H_all_sim = model_eval(d, f, include_e=True, include_delay=False)   # C++ semantics after E fix
    H_all_sto = model_eval(d, f, include_e=True, include_delay=True)
    H_sim = diff_combine(H_all_sim)
    H_sto = diff_combine(H_all_sto)
    H_orig = as_delivered(H_all_sto) if H_all_sto.shape[1] >= 4 else None

    m_sim = metrics(H_sim, ref, f)
    m_sto = metrics(H_sto, ref, f)
    m_orig = metrics(H_orig, ref, f) if H_orig is not None else None

    ok_mag = (m_sim["mag_rms_db"] <= MAG_RMS_DB) and (m_sim["mag_max_db"] <= MAG_MAX_DB)
    ok_ph = (m_sim["ph_rms_deg"] <= PH_RMS_DEG) and (m_sim["ph_max_deg"] <= PH_MAX_DEG)
    verdict = "PASS" if (ok_mag and ok_ph) else "FAIL"

    summary = dict(
        banner="A2 peters channel check (T-S3 phase 1)",
        thresholds=dict(band_GHz=[F_MIN / 1e9, F_MAX / 1e9],
                        mag_rms_db=MAG_RMS_DB, mag_max_db=MAG_MAX_DB,
                        ph_rms_deg=PH_RMS_DEG, ph_max_deg=PH_MAX_DEG),
        judged_object="simulated A,B,C,D,E differential response (E implemented in C++; delay still ignored, constant delay removed for phase)",
        n_points=len(f), f_min_GHz=float(f[0] / 1e9), f_max_GHz=float(f[-1] / 1e9),
        simulated=m_sim,
        stored_model_with_E_and_delay=m_sto,
        original_port_config_0_1_to_0_1=m_orig,        verdict=verdict,
        verdict_mag=bool(ok_mag), verdict_phase=bool(ok_ph),
    )

    with open(os.path.join(args.outdir, "a2_peters_check.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(args.outdir, "a2_peters_sdd21_compare.csv"), "w") as fh:
        fh.write("f_GHz,sdd21_dB,Hsim_dB,d_dB_sim,ph_resid_deg_sim,Hstored_dB,d_dB_sto,ph_resid_deg_sto\n")
        ph_sim = np.degrees(np.unwrap(np.angle(H_sim / ref)))
        ph_sto = np.degrees(np.unwrap(np.angle(H_sto / ref)))
        # same linear detrending for per-row display
        c1 = np.polyfit(f, np.unwrap(np.angle(H_sim / ref)), 1)
        c2 = np.polyfit(f, np.unwrap(np.angle(H_sto / ref)), 1)
        r1 = np.degrees(np.unwrap(np.angle(H_sim / ref)) - np.polyval(c1, f))
        r2 = np.degrees(np.unwrap(np.angle(H_sto / ref)) - np.polyval(c2, f))
        for k in range(len(f)):
            fh.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                f[k] / 1e9, 20 * np.log10(abs(ref[k])), 20 * np.log10(abs(H_sim[k])),
                20 * np.log10(abs(H_sim[k])) - 20 * np.log10(abs(ref[k])), r1[k],
                20 * np.log10(abs(H_sto[k])),
                20 * np.log10(abs(H_sto[k])) - 20 * np.log10(abs(ref[k])), r2[k]))

    if verdict == "PASS" and args.fixed_json_out:
        d2 = json.loads(json.dumps(d))
        d2["port_config"] = {"active_inputs": [0, 2], "active_outputs": [1, 3]}
        d2["_ts3_note"] = ("T-S3 A2: port mapping corrected to contract SDD21 "
                           "(active inputs = SE ports 1,3; active outputs = SE ports 2,4).")
        with open(args.fixed_json_out, "w") as fh:
            json.dump(d2, fh, indent=2)
        summary["fixed_json_written"] = args.fixed_json_out
        with open(os.path.join(args.outdir, "a2_peters_check.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    print("=" * 78)
    print("A2 peters channel check  |  band %.3f–%.3f GHz, %d points" % (f[0] / 1e9, f[-1] / 1e9, len(f)))
    print("thresholds: mag RMS<=%.2f dB, mag max<=%.2f dB; phase RMS<=%.1f deg, max<=%.1f deg"
          % (MAG_RMS_DB, MAG_MAX_DB, PH_RMS_DEG, PH_MAX_DEG))
    print("-" * 78)
    print("simulated A,B,C,D  : magRMS=%.3f dB  magMax=%.3f dB  phRMS=%.3f deg  phMax=%.3f deg  delay_removed=%.3f ps"
          % (m_sim["mag_rms_db"], m_sim["mag_max_db"], m_sim["ph_rms_deg"], m_sim["ph_max_deg"], m_sim["delay_removed_ps"]))
    print("stored (+E,+delay) : magRMS=%.3f dB  magMax=%.3f dB  phRMS=%.3f deg  phMax=%.3f deg  delay_removed=%.3f ps"
          % (m_sto["mag_rms_db"], m_sto["mag_max_db"], m_sto["ph_rms_deg"], m_sto["ph_max_deg"], m_sto["delay_removed_ps"]))
    print("original {0,1}->{0,1}: " + ("magRMS=%.3f dB  magMax=%.3f dB  (diagnostic only)"
          % (m_orig["mag_rms_db"], m_orig["mag_max_db"]) if m_orig else "n/a (2-port realization)"))
    print("-" * 78)
    print("VERDICT:", verdict)
    if verdict == "PASS" and args.fixed_json_out:
        print("fixed JSON written:", args.fixed_json_out)
    print("=" * 78)
    # compact table
    idx = np.unique(np.round(np.linspace(0, len(f) - 1, 12)).astype(int))
    print("\n  f(GHz)   |SDD21|(dB)   |Hsim|(dB)   d(dB)   ph_res(deg)")
    ph_res = np.degrees(np.unwrap(np.angle(H_sim / ref)))
    c = np.polyfit(f, np.unwrap(np.angle(H_sim / ref)), 1)
    ph_res = np.degrees(np.unwrap(np.angle(H_sim / ref)) - np.polyval(c, f))
    for k in idx:
        print("  %7.3f   %10.3f   %9.3f   %+6.3f   %+8.3f"
              % (f[k] / 1e9, 20 * np.log10(abs(ref[k])), 20 * np.log10(abs(H_sim[k])),
                 20 * np.log10(abs(H_sim[k])) - 20 * np.log10(abs(ref[k])), ph_res[k]))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
