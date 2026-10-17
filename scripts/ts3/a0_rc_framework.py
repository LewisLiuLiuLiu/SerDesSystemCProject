#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-S4 / A0 — 合成一阶 RC 信道的框架侧参考（v3 链 + MC 旁证）

两窗口同一解析式：H(f) = G / (1 + j f / fp)，G = 10^(-6/20)（−6 dB），fp = 20 GHz
（= nrz_link_tb 默认信道参数）。**SystemC 侧不用 SIMPLE**（SIMPLE 是后向欧拉，
20 GHz 处频率翘曲；本实验会把该数值误差误判为信道差异），而用 1 态状态空间
dx/dt = -ωc x + [0.5,-0.5]·u，y = [Gωc,-Gωc]·x，差分传递函数同样恰为 G/(1+jf/fp)。
CTLE/FFE 与 S0/S1 同结构：CTLE (fz,fp) = (2 GHz, 10 GHz)；5 抽头 FFE 由本信道
按 s4p_channel.design_ffe 重新设计（零抖动；PULSE-01 T=100 ps, t_r=20 ps）。

窗口处置：RC 无延迟，CTLE 又带高频提升，单比特响应主峰落在 t≈0 附近的负时间侧
（IFFT 周期窗回绕到数组末端），main_cursor_analysis 的 ±20 ns 索引会越界。
这里只在**分析前**把 1 ps / 0.25 ps 响应圆周平移 ROLL=40 ns（整数采样点，不引入
任何频率域插值），使主峰落在窗口中段；H(f) 本身不加延迟，因此与 SystemC 侧
逐频点同一个解析式，平移不改变 cursor 取值与 BER。

输出：v3 链 BER（M=600, K=12, τ=0）、MC 旁证（5e5 bits）、FFE 权重、
GATE-01/02、cursor 窗收敛诊断，以及供 SystemC 使用的稀疏 201-tap 向量。
"""

import argparse
import json
import os
import sys

import numpy as np

SIM = "/Volumes/YT_SSD/workspace/serdesResearch/simulation"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(HERE))

import s4p_channel as ch          # noqa: E402
import validate_channel as vc    # noqa: E402
import v3_prototype as v3        # noqa: E402
import mc_ber as mc              # noqa: E402

G_DB = -6.0
FP_HZ = 20e9
CTLE = (2e9, 10e9)
FFE_TAPS = 5
UI_SAMPLES = 50
SIGMAS = (0.22, 0.30, 0.35, 0.50)
ROLL = int(round(40e-9 / ch.DT_H))   # 40 ns，纯圆周平移（1 ps 网格整数点）


def build():
    f_ext = np.arange(0.0, ch.F_MAX_EXT + ch.FREQ_STEP / 2, ch.FREQ_STEP)
    G = 10.0 ** (G_DB / 20.0)
    H_rc = G / (1.0 + 1j * f_ext / FP_HZ)
    H0 = H_rc * ch.ctle_spectrum(f_ext, CTLE[0], CTLE[1])

    _, h0_raw = ch.single_bit_response(f_ext, H0, vc.T, vc.TR)
    i0_raw = int(np.argmax(np.abs(h0_raw)))          # 主峰可能贴窗口边缘
    h0 = np.roll(h0_raw, ROLL)                       # 仅分析用居中
    ca0 = ch.main_cursor_analysis(h0, vc.T)
    w, offs = ch.design_ffe(ca0["cursors"], FFE_TAPS)

    H = H0 * ch.ffe_spectrum(f_ext, w, offs, vc.T)
    _, h_raw = ch.single_bit_response(f_ext, H, vc.T, vc.TR)
    i_raw = int(np.argmax(np.abs(h_raw)))
    h = np.roll(h_raw, ROLL)
    ca = ch.main_cursor_analysis(h, vc.T)

    hf_raw = v3.build_h_fine(H, f_ext, vc.T, vc.TR, v3.PAD)
    hfc, ic = v3.centered(hf_raw, i_raw * v3.PAD)    # 与 1 ps 主峰同一采样相位
    return dict(f_ext=f_ext, H=H, h_raw=h_raw, i_raw=i_raw, h=h, ca=ca, ca0=ca0,
                i0_raw=i0_raw, w=w, offs=offs, hf=hfc, ic=ic)


def v3_ber(res, sigma_n):
    taus, wt = v3.tau_measure(0.0, 0.0, 0.0, 1, 1)
    C = v3.cursor_matrix(res["hf"], res["ic"], taus, 600, v3.DT)
    ber, dg = v3.chain(C, wt, sigma_n, 12, "v1")
    return float(ber), dg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mc-bits", type=int, default=500000)
    ap.add_argument("--rc-json", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    res = build()
    w, offs = res["w"], res["offs"]
    print("FFE weights:", w.tolist(), "offsets:", offs.tolist())
    print("ca0 c0=%.6f  ca c0=%.6f  i_opt=%d  (raw peak i=%d, ROLL=%d)"
          % (res["ca0"]["c0"], res["ca"]["c0"], res["ca"]["i_opt"], res["i_raw"], ROLL))
    assert res["ca"]["i_opt"] == (res["i_raw"] + ROLL) % ch.N_FFT, "roll/peak 口径不一致"
    wd = ch.window_declaration(res["h"], vc.T)
    print("window:", {k: wd[k] for k in ("converged", "last_step_rel", "monotone",
                                        "impl_deficit_rel")})

    # ---- 1-state RC state-space JSON (symmetric differential pair) ----
    if args.rc_json:
        w0 = 2.0 * np.pi * FP_HZ
        G = 10.0 ** (G_DB / 20.0)
        A = np.array([[-w0]])
        B = np.array([[0.5, -0.5]])          # 1 state × 2 inputs
        C = np.array([[G * w0], [-G * w0]])
        D = np.zeros((2, 2)); E = np.zeros((2, 2))
        # sanity: differential response vs analytic
        fchk = np.arange(0.05e9, 15e9 + 1e6, 10e6)
        jw = 2j * np.pi * fchk
        X = np.linalg.solve(jw[:, None, None] * np.eye(1)[None] - A[None],
                            np.broadcast_to(B[None], (len(fchk), 1, 2)))
        Hss = np.einsum("on,fni->foi", C, X) + D[None]
        hd = 0.5 * (Hss[:, 0, 0] - Hss[:, 0, 1] - Hss[:, 1, 0] + Hss[:, 1, 1])
        ref = G / (1.0 + jw / w0)
        ddb = 20 * np.log10(np.abs(hd) / np.abs(ref))
        print(f"[RC-SS] max|ΔdB| vs analytic (0.05–15 GHz): {np.max(np.abs(ddb)):.6f} dB")
        rc = {"version": "3.0", "method": "state_space", "fs": 500e9,
              "full_model": {"n_diff_ports": 2, "n_outputs": 2, "n_states": 1,
                             "port_pairs": [[0, 0], [1, 1]],
                             "delay_s": [0.0, 0.0, 0.0, 0.0],
                             "state_space": {"A": A.tolist(), "B": B.tolist(), "C": C.tolist(),
                                             "D": D.tolist(), "E": E.tolist()},
                             # T-S4: exact ZOH realization (2 ps TB timestep).
                             "discrete_state_space": {
                                 "dt": 2e-12,
                                 "A": [[float(np.exp(A[0, 0] * 2e-12))]],
                                 "B": (((1.0 - np.exp(A[0, 0] * 2e-12)) / (-A[0, 0])) * B).tolist(),
                                 "C": C.tolist(), "D": D.tolist(), "E": E.tolist()}},
              "port_config": {"active_inputs": [0, 1], "active_outputs": [0, 1]},
              "metadata": {"is_differential": True, "contract": "H(f)=G/(1+jf/fp)",
                           "G_dB": G_DB, "fp_GHz": FP_HZ / 1e9,
                           "realization": "1-state RC, symmetric differential pair",
                           "ss_vs_analytic_max_dB": float(np.max(np.abs(ddb)))}}
        with open(args.rc_json, "w") as fh:
            json.dump(rc, fh, indent=2)
        print("written:", args.rc_json)

    rows = []
    for sn in SIGMAS:
        ber, dg = v3_ber(res, sn)
        # MC 旁证（同 h_eff、零抖动）
        rng = np.random.RandomState(7000 + int(round(sn * 100)))
        N = args.mc_bits
        bits = (rng.randint(0, 2, N).astype(float) * 2.0 - 1.0)
        noise = rng.normal(0.0, sn, N)
        jitter = np.zeros(N)
        errs, nclip = mc.mc_ber_exact(bits, res["h_raw"], vc.T, ch.DT_H,
                                      res["i_raw"], jitter, noise)
        ber_mc = errs / N
        rows.append(dict(sigma_n=sn, ber_v3=ber, ber_mc=float(ber_mc),
                         mc_errs=int(errs), n_bits=int(N), mc_clip=int(nclip),
                         neff=float(dg["neff_wmean"])))
        print(f"σ={sn:.2f}: v3={ber:.6e}  MC={ber_mc:.6e}  errs={errs}  N={N}")

    # GATE（τ=0，操作窗）
    taus0, w0 = v3.tau_measure(0.0, 0.0, 0.0, 1, 1)
    C0 = v3.cursor_matrix(res["hf"], res["ic"], taus0, 600, v3.DT)[0]
    gate = v3.gate_channel(C0)

    taps = [0.0] * (1 + UI_SAMPLES * (len(w) - 1))
    for k, wk in enumerate(w):
        taps[UI_SAMPLES * k] = float(wk)

    out = dict(
        channel=dict(formula="H(f)=G/(1+j f/fp)", G_dB=G_DB, fp_GHz=FP_HZ / 1e9),
        ctle=dict(fz_GHz=CTLE[0] / 1e9, fp_GHz=CTLE[1] / 1e9),
        pulse=dict(T_ps=vc.T * 1e12, tr_ps=vc.TR * 1e12),
        ffe=dict(weights=[float(x) for x in w], offsets=[int(x) for x in offs],
                 taps_201_sparse=taps),
        c0=float(res["ca"]["c0"]), gate=gate,
        window=wd, roll_samples=int(ROLL), raw_peak_index=int(res["i_raw"]),
        results=rows,
    )
    path = os.path.join(args.outdir, "a0_rc_framework.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
