#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 / A3 SystemC 配置：A1 重拟合信道 + A2 erf 20 ps 边沿 + E 组 FFE。

final : 80-state A1 fit + erf20 + transparent driver + exact ZOH channel
        + exact ZOH first-order CTLE + linear CTLE/VGA, sample_point=0.28 UI
diag  : 同一信道/脉形，但 continuous CTLE (sca_ltf_nd) + saturated CTLE/VGA
        (±10 V, T-S3 实况)；用于复现 A3 的中间台阶，不是最终配置。
A2 pair: sigma=0.50 V, 1e5 UI, on/off, continuous CTLE + saturated RX；复现 A2 表。
信道由 TB CLI `ss config/peters_ss_channel_ts4.json` 提供（内含 discrete ZOH block）。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_configs import build  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
SIGMAS = (0.22, 0.30, 0.35, 0.50)


def cfg(on, sigma, duration, sample_point, linear_rx=False, discrete_ctle=False):
    c = build(True, sigma, duration)
    c["wave"]["edge_mode"] = "erf" if on else "ideal"
    c["wave"]["edge_tr"] = 20e-12
    c["tx"]["driver"]["poles"] = [] if on else [17.5e9]
    c["cdr"]["sample_point"] = float(sample_point)
    c["channel"]["attenuation_db"] = 6.0
    c["channel"]["bandwidth_hz"] = 20e9
    c["rx"]["ctle"]["discrete"] = bool(discrete_ctle)
    sat = 1e6 if linear_rx else 10.0
    c["rx"]["ctle"]["sat_min"] = -sat
    c["rx"]["ctle"]["sat_max"] = sat
    c["rx"]["vga"]["sat_min"] = -sat
    c["rx"]["vga"]["sat_max"] = sat
    c["_comment"] = (
        f"T-S4/A3 {'ON' if on else 'OFF'} sigma={sigma} V, peters A1 fit, "
        f"E-group FFE, sample_point={sample_point} UI, "
        f"CTLE discrete={bool(discrete_ctle)}, RX sat={'linear' if linear_rx else '+/-10 V'}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "config"))
    ap.add_argument("--sample-point", type=float, default=0.28,
                    help="A3 BER optimum with discrete ZOH channel+CTLE (0.28 UI)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for sn in SIGMAS:
        tag = "%03d" % int(round(sn * 100))
        p = os.path.join(args.outdir, f"ts4_a3_final_{tag}.json")
        json.dump(cfg(True, sn, 5e-5, args.sample_point,
                      linear_rx=True, discrete_ctle=True), open(p, "w"), indent=2)
        print("wrote", p)
        p = os.path.join(args.outdir, f"ts4_a3_diag_{tag}.json")
        json.dump(cfg(True, sn, 5e-5, args.sample_point,
                      linear_rx=False, discrete_ctle=False), open(p, "w"), indent=2)
        print("wrote", p)

    # A2 statistically meaningful on/off pair at sigma=0.50, 1e5 UI
    for on, tag, sp in ((True, "on", args.sample_point), (False, "off", 0.12)):
        p = os.path.join(args.outdir, f"ts4_a2_ber_050_{tag}.json")
        json.dump(cfg(on, 0.50, 1e-4, sp, linear_rx=False, discrete_ctle=False),
                  open(p, "w"), indent=2)
        print("wrote", p)


if __name__ == "__main__":
    main()
