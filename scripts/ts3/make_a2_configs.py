#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 / A2 SystemC 配置：TX 20 ps 边沿开关对照。

on  : wave.edge_mode="erf", edge_tr=20 ps；driver.poles=[]（透明，边沿由波形源定义）
off : wave.edge_mode="ideal"；driver.poles=[17.5 GHz]（T-S3 的一阶极点近似）

其余与 S0/S1 完全一致：peters A1 重拟合信道（TB CLI `ss peters_ss_channel_ts4.json`）、
E 组 5 抽头 FFE（201 稀疏 taps）、CTLE 2/10 GHz、VGA 透明、DFE/AGC 关闭、
CDR PI 冻结、sampler 噪声按 σ_n。眼图配置为零噪声 2000 UI。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_configs import build  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))


def make(on, sigma, duration, sample_point):
    c = build(True, sigma, duration)
    c["wave"]["edge_mode"] = "erf" if on else "ideal"
    c["wave"]["edge_tr"] = 20e-12
    c["tx"]["driver"]["poles"] = [] if on else [17.5e9]
    c["cdr"]["sample_point"] = float(sample_point)
    c["channel"]["attenuation_db"] = 6.0
    c["channel"]["bandwidth_hz"] = 20e9
    c["_comment"] = (
        f"T-S4/A2 {'ON (erf 20 ps, driver transparent)' if on else 'OFF (driver pole 17.5 GHz)'}; "
        f"peters A1 fit channel via ss; E-group FFE; 5e5 UI BER / 2000 UI eye")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "config"))
    ap.add_argument("--sample-point-on", type=float, default=0.10)
    ap.add_argument("--sample-point-off", type=float, default=0.10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for on, sp, tag in ((True, args.sample_point_on, "on"), (False, args.sample_point_off, "off")):
        eye = make(on, 0.0, 2e-6, sp)
        eye["rx"]["sampler"]["noise_enable"] = False
        eye["rx"]["sampler"]["noise_sigma"] = 0.0
        p = os.path.join(args.outdir, f"ts4_a2_eye_{tag}.json")
        json.dump(eye, open(p, "w"), indent=2)
        print("wrote", p)

        ber = make(on, 0.35, 1e-4, sp)
        p = os.path.join(args.outdir, f"ts4_a2_ber_{tag}.json")
        json.dump(ber, open(p, "w"), indent=2)
        print("wrote", p)


if __name__ == "__main__":
    main()
