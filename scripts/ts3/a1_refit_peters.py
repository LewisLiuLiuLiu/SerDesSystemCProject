#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 / A1 — peters SDD21 重拟合（低频谱加权 + 加权残差 + 高阶）。

目标（用户 T-S4 指令，运行前固定）：
  * |Δ| @ 50 MHz < 0.1 dB
  * 接收脉冲能量相对误差 ε < 1%（PULSE-01：T=100 ps，tr=20 ps）
  * A2 原判据不放松：mag RMS ≤0.75 dB、max ≤2.50 dB；phase RMS ≤7°、max ≤30°

方法：
  * 契约端口映射 SDD21=(S21−S23−S41+S43)/2（同 s4p_channel.differential_thru）
  * 传播延迟用 s4p 原始均匀 10 MHz 网格的 impulse-peak 估计（3.961 ns），
    先从 H 中剥离；拟合时 extract_delay=False，避免增强网格影响延迟估计。
    C++ 端不使用 delay_s（常数延迟，对 BER 无影响）；A2 相位判据同样剥离常数延迟。
  * 低频谱加权：w(f) = 1/max(|H|,1e-4) · (1 + 8·exp(−(f/5 GHz)²))
  * 残差加权：weight_residues=True（T-S4 对 vector_fitting.py 的向后兼容扩展；
    False 时保持历史未加权残差解）
  * order=80，n_iter_poles=40；只拟合 0.05–15 GHz 实测点（不注入合成 DC 点）。
  * 导出与 T-S3 相同的对称 2×2 差分实现：
        B2=[B/2,−B/2], C2=[C;−C], D2=(d/2)[[1,−1],[−1,1]], E2=(e/2)[...].

运行后由 a2_peters_check.py 对导出的 JSON 做独立复核（同一判据）。
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

from vector_fitting import SParamModel, estimate_delay_impulse, remove_delay  # noqa: E402
from a2_peters_check import metrics, sdd21_from_s  # noqa: E402

F_MIN = 0.05e9
F_MAX = 15.0e9
ORDER = 80
N_ITER_POLES = 40
BOOST_A = 8.0
BOOST_FC = 5.0e9
FS_EXPORT = 500e9

# Fixed thresholds (A2, not relaxed)
MAG_RMS_DB = 0.75
MAG_MAX_DB = 2.50
PH_RMS_DEG = 7.0
PH_MAX_DEG = 30.0
DC_TARGET_DB = 0.10
ENERGY_TARGET = 0.01


def eval_ss(A, B, C, D, E, f, delay=0.0):
    jw = 2j * np.pi * f
    X = np.linalg.solve(jw[:, None, None] * np.eye(A.shape[0])[None] - A[None],
                        np.broadcast_to(B[None], (len(f),) + B.shape))
    H = np.einsum("on,fni->foi", C, X)[:, 0, 0] + D[0, 0] + jw * E[0, 0]
    if delay:
        H = H * np.exp(-jw * delay)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-report", default=None)
    ap.add_argument("--order", type=int, default=ORDER)
    ap.add_argument("--n-iter-poles", type=int, default=N_ITER_POLES)
    ap.add_argument("--boost-a", type=float, default=BOOST_A)
    ap.add_argument("--boost-fc-ghz", type=float, default=BOOST_FC / 1e9)
    ap.add_argument("--discrete-dt-ps", type=float, default=2.0,
                    help="ZOH discrete export timestep (must match the TB timestep)")
    args = ap.parse_args()

    import skrf as rf
    net = rf.Network(args.s4p)
    f_all = np.asarray(net.f, float).ravel()
    S = np.asarray(net.s, complex)
    band = (f_all >= F_MIN) & (f_all <= F_MAX)
    f = f_all[band]
    H_ref = sdd21_from_s(S[band])

    # Delay from the raw uniform grid; strip before fitting.
    tau = float(estimate_delay_impulse(f, H_ref))
    H_comp = remove_delay(f, H_ref, tau)

    weight = (1.0 / np.maximum(np.abs(H_ref), 1e-4)) * (
        1.0 + args.boost_a * np.exp(-(f / (args.boost_fc_ghz * 1e9)) ** 2))

    model = SParamModel()
    model.n_ports_raw = 4
    model._is_differential = True
    model.diff_pairs = [(1, 3), (2, 4)]
    model.freq = f.copy()
    model.s_active = np.zeros((len(f), 2, 2), dtype=complex)
    model.s_active[:, 1, 0] = H_comp        # SDD21 (output 2 <- input 1, 0-based)
    model.fit(order=args.order, port_pairs=[(2, 1)], extract_delay=False,
              fmax=F_MAX, n_iter_poles=args.n_iter_poles,
              weight=weight, weight_residues=True)
    model.summary()

    ss = model.to_state_space()
    A = np.asarray(ss["A"], float)
    B = np.asarray(ss["B"], float)
    C = np.asarray(ss["C"], float)
    D = np.asarray(ss["D"], float)
    E = np.asarray(ss["E"], float)

    H_sim = eval_ss(A, B, C, D, E, f, delay=0.0)
    H_sto = eval_ss(A, B, C, D, E, f, delay=tau)
    m_sim = metrics(H_sim, H_ref, f)
    m_sto = metrics(H_sto, H_ref, f)

    ok_a2 = (m_sim["mag_rms_db"] <= MAG_RMS_DB and m_sim["mag_max_db"] <= MAG_MAX_DB
             and m_sim["ph_rms_deg"] <= PH_RMS_DEG and m_sim["ph_max_deg"] <= PH_MAX_DEG)
    ok_dc = abs(m_sim["dc_gain_db"]) < DC_TARGET_DB
    ok_energy = m_sim["energy_rel"] < ENERGY_TARGET
    verdict = "PASS" if (ok_a2 and ok_dc and ok_energy) else "FAIL"

    # Symmetric differential 2x2 realization (same convention as T-S3)
    B2 = np.hstack([B / 2.0, -B / 2.0])
    C2 = np.vstack([C, -C])
    S2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    D2 = (D[0, 0] / 2.0) * S2
    E2 = (E[0, 0] / 2.0) * S2

    # T-S4: exact zero-order-hold discrete realization of the symmetric 2x2
    # system.  The C++ ChannelSParamTdf uses it when the TDF timestep matches
    # dt; this removes the sca_ss DAE solver's phase error (see report).
    dt = args.discrete_dt_ps * 1e-12
    n = A.shape[0]
    Mz = np.zeros((n + 2, n + 2))
    Mz[:n, :n] = A
    Mz[:n, n:] = B2
    Z = expm(Mz * dt)
    Ad = Z[:n, :n]
    Bd = Z[:n, n:]

    rep = dict(
        s4p=args.s4p, order=args.order, n_iter_poles=args.n_iter_poles,
        n_states=int(A.shape[0]), delay_s=tau, e_value=float(E[0, 0]),
        weight=f"1/max(|SDD21|,1e-4)*(1+{args.boost_a:g}*exp(-(f/{args.boost_fc_ghz:g}GHz)^2))",
        weight_residues=True, fit_band_GHz=[F_MIN / 1e9, F_MAX / 1e9],
        judged="C++ semantics A,B,C,D,E differential; delay_s ignored; constant delay removed for phase",
        simulated=m_sim, stored_with_delay=m_sto,
        targets=dict(dc_abs_db=DC_TARGET_DB, energy_rel=ENERGY_TARGET,
                     mag_rms_db=MAG_RMS_DB, mag_max_db=MAG_MAX_DB,
                     ph_rms_deg=PH_RMS_DEG, ph_max_deg=PH_MAX_DEG),
        checks=dict(a2=bool(ok_a2), dc=bool(ok_dc), energy=bool(ok_energy)),
        verdict=verdict,
    )
    print("[A1] fit report:", json.dumps(rep, indent=2))

    out = {
        "version": "3.0",
        "method": "state_space",
        "fs": FS_EXPORT,
        "full_model": {
            "n_diff_ports": 2,
            "n_outputs": 2,
            "n_states": int(A.shape[0]),
            "port_pairs": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "delay_s": [tau, tau, tau, tau],
            "state_space": {
                "A": A.tolist(), "B": B2.tolist(), "C": C2.tolist(),
                "D": D2.tolist(), "E": E2.tolist(),
            },
            # Optional exact ZOH realization used by the C++ channel when the
            # TDF timestep equals dt (T-S4 implementation fix).
            "discrete_state_space": {
                "dt": dt, "A": Ad.tolist(), "B": Bd.tolist(),
                "C": C2.tolist(), "D": D2.tolist(), "E": E2.tolist(),
            },
        },
        "port_config": {"active_inputs": [0, 1], "active_outputs": [0, 1]},
        "metadata": {
            "n_raw_ports": 4,
            "is_differential": True,
            "diff_pairs": [[1, 3], [2, 4]],
            "contract_port_mapping": "(S21 - S23 - S41 + S43)/2",
            "fit_band_GHz": [F_MIN / 1e9, F_MAX / 1e9],
            "realization": ("symmetric differential pair: H=(h0/2)[[1,-1],[-1,1]]; "
                            "differential transfer out0-out1 = h0*(in0-in1); common mode = 0"),
            "generator": "scripts/ts3/a1_refit_peters.py (T-S4/A1)",
            "fit_parameters": dict(order=args.order, n_iter_poles=args.n_iter_poles,
                                   boost_a=args.boost_a, boost_fc_GHz=args.boost_fc_ghz,
                                   weight_residues=True, extract_delay=False),
            "delay_s": tau,
            "fit_report": rep,
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[A1] written:", args.out)
    if args.fit_report:
        with open(args.fit_report, "w") as fh:
            json.dump(rep, fh, indent=2)
        print("[A1] report written:", args.fit_report)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
