#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 / A0 SystemC 配置：RC 解析信道 + 与框架同源的 5 抽头 FFE。

配置基底复用 T-S3 的 make_configs.build()（冻结 CDR、透明 VGA、DFE/AGC 关闭、
driver pole 17.5 GHz、CTLE 2/10 GHz），只替换：
  * tx.ffe.taps  ← a0_rc_framework.json 的 FFE 权重（50 样本/UI 稀疏 201 taps）
  * cdr.sample_point（由 A0 零噪声眼图标定后传入）
  * sampler.noise_sigma / enable（σ_n 四档）
  * 时长（眼图 2000 UI；BER 5e5 UI）
信道本身由 `nrz_ber_tb ss config/ts3_a0_rc_ss.json` 提供，不在本文件内。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_configs import build  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
FRAMEWORK_JSON = "/Volumes/YT_SSD/workspace/serdesResearch/docs/systemc_evidence_TS4_20261017/a0_rc_framework.json"
UI_SAMPLES = 50
SIGMAS = (0.22, 0.30, 0.35, 0.50)


def load_taps(path=FRAMEWORK_JSON):
    d = json.load(open(path))
    taps = d["ffe"]["taps_201_sparse"]
    assert len(taps) == 201 and sum(1 for x in taps if x != 0.0) == 5
    return taps, d["ffe"]["weights"], d["ffe"]["offsets"]


def common(cfg, taps, sample_point):
    cfg["tx"]["ffe"]["taps"] = taps
    cfg["cdr"]["sample_point"] = float(sample_point)
    cfg["channel"]["attenuation_db"] = 6.0       # 仅摘要显示；实际用 SS JSON
    cfg["channel"]["bandwidth_hz"] = 20e9
    cfg["wave"]["jitter"] = {"RJ_sigma": 0.0, "SJ_freq": [], "SJ_pp": []}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-point", type=float, default=0.1)
    ap.add_argument("--outdir", default=os.path.join(REPO, "config"))
    ap.add_argument("--framework-json", default=FRAMEWORK_JSON)
    args = ap.parse_args()

    taps, weights, offs = load_taps(args.framework_json)
    os.makedirs(args.outdir, exist_ok=True)

    eye = common(build(False, 0.0, 2e-6), taps, args.sample_point)
    eye["_comment"] = ("T-S4/A0 eye calibration: RC 1-state SS, zero noise, 2000 UI; "
                       "FFE weights from a0_rc_framework.json")
    eye_path = os.path.join(args.outdir, "ts3_a0_eye.json")
    json.dump(eye, open(eye_path, "w"), indent=2)
    print("wrote", eye_path)

    for sn in SIGMAS:
        cfg = common(build(True, sn, 5e-5), taps, args.sample_point)
        cfg["_comment"] = (f"T-S4/A0 sigma_n={sn} V, RC 1-state SS, 5e5 UI, "
                           f"sample_point={args.sample_point} UI; FFE={weights}")
        path = os.path.join(args.outdir, "ts3_a0_s1_%03d.json" % int(round(sn * 100)))
        json.dump(cfg, open(path, "w"), indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
