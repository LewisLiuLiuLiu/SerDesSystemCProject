#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-S3 配置生成器：由 config/default.json 生成 S0/S1/S2/锚点配置。

物理配置（对齐 v3 E 组，零抖动阶段）：
  * 信道：STATE_SPACE（peters_ss_channel_ts3.json，CLI `ss` 指定，不在此文件）
  * FFE：5 抽头（Python E 组设计权重，offset -1..3），以 201 长稀疏 taps 实现
    UI 间隔（50 样本/UI @2 ps）；因果化仅引入 1 UI 常数延迟（与 BER 无关）
  * 驱动：dc_gain=1.0（S2D 差分 ±x → 信道侧差分 ±x），sat_mode=none，
          pole=17.5 GHz（一阶近似框架 PULSE-01 的 t_r=20 ps；残差在报告量化）
  * CTLE：FILT-01 参数 fz=2 GHz / fp=10 GHz，dc_gain=1.0，饱和放宽（线性）
  * VGA：透明（dc_gain=1.0, pole=1e13, sat 放宽）；AGC 关闭（initial_gain=1.0）
  * DFE：关闭（adaption.dfe.enabled=false；summer taps 全 0）
  * 采样器：理想判决（resolution=1e-6, hysteresis=0, threshold=0）；
           噪声注入 = v3 σ_n（判决点等效），seed 固定 12345
  * CDR：默认 PI（kp=0.01, ki=1e-4, sample_point=0.5），实际相位由 CDR 内部环控制
  * 时钟：IDEAL
"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(REPO, "config", "default.json")

# Python E-group FFE design weights (results_real_channel.json meta.ffe; offsets -1..3)
W = [-0.18940785142168523, 1.5385698606271185, -0.10127970157613488,
     -0.0021582021029717376, 0.025005851099668033]
UI_SAMPLES = 50  # 100 ps UI / 2 ps timestep


def ffe_taps_ui_spaced():
    taps = [0.0] * (1 + UI_SAMPLES * (len(W) - 1))
    for k, w in enumerate(W):
        taps[UI_SAMPLES * k] = w
    return taps


def build(noise_enable, noise_sigma, duration_s, eq=True, noise_seed=12345,
          driver_pole=17.5e9, sat=10.0):
    c = json.load(open(BASE))
    c["global"].update({"Fs": 500e9, "UI": 1e-10, "duration": duration_s, "seed": 12345})
    c["wave"]["type"] = "PRBS31"
    c["wave"]["single_pulse"] = 0.0
    c["wave"]["jitter"] = {"RJ_sigma": 0.0, "SJ_freq": [], "SJ_pp": []}

    c["tx"]["ffe"]["taps"] = ffe_taps_ui_spaced() if eq else [1.0]
    c["tx"]["mux_lane"] = 0
    d = c["tx"]["driver"]
    d.update({"dc_gain": 1.0, "vswing": 0.8, "vcm_out": 0.6, "output_impedance": 50.0,
              "poles": [driver_pole], "sat_mode": "none", "vlin": 1.0})
    d["psrr"]["enable"] = False
    d["imbalance"].update({"gain_mismatch": 0.0, "skew": 0.0})
    d["slew_rate"]["enable"] = False

    ct = c["rx"]["ctle"]
    ct.update({"zeros": ([2e9] if eq else []), "poles": ([10e9] if eq else []),
               "dc_gain": 1.0, "vcm_out": 0.6, "offset_enable": False, "vos": 0.0,
               "noise_enable": False, "vnoise_sigma": 0.0,
               "sat_min": -sat, "sat_max": sat})
    vg = c["rx"]["vga"]
    vg.update({"zeros": [], "poles": [1e13], "dc_gain": 1.0, "vcm_out": 0.6,
               "offset_enable": False, "vos": 0.0,
               "noise_enable": False, "vnoise_sigma": 0.0,
               "sat_min": -sat, "sat_max": sat})
    sm = c["rx"]["sampler"]
    sm.update({"threshold": 0.0, "hysteresis": 0.0, "sample_delay": 0.0,
               "phase_source": "phase", "resolution": 1e-6,
               "offset_enable": False, "offset_value": 0.0,
               "noise_enable": bool(noise_enable), "noise_sigma": float(noise_sigma),
               "noise_seed": noise_seed})
    ds = c["rx"]["dfe_summer"]
    ds.update({"tap_coeffs": [0.0] * 5, "ui": 1e-10, "vcm_out": 0.0, "vtap": 0.05,
               "map_mode": "pm1", "enable": True, "sat_enable": False,
               "sat_min": -sat, "sat_max": sat})

    cdr = c["cdr"]
    # T-S3 alignment with v3's static t_opt: freeze the CDR PI (kp=ki=0) and use a
    # fixed sampling point measured from the vertical eye opening (max gap at ~10 ps
    # for the E-group chain; CTLE group delay shifts the eye centre by ~-40 ps).
    # (CDR "false lock" at -1.5 ps observed and documented as a residual non-ideality.)
    cdr["pi"].update({"kp": 0.0, "ki": 0.0, "edge_threshold": 0.5, "adaptive_threshold": False})
    cdr["pai"].update({"resolution": 1e-12, "range": 5e-11})
    cdr.update({"ui": 1e-10, "sample_point": (0.1 if eq else 0.5), "debug_enable": False})

    c["clock"]["type"] = "IDEAL"
    c["clock"]["frequency"] = 1e10

    a = c["adaption"]
    a.update({"Fs": 500e9, "UI": 1e-10, "seed": 12345, "update_mode": "multi-rate",
              "fast_update_period": 2e-10, "slow_update_period": 2e-7})
    a["agc"].update({"enabled": False, "target_amplitude": 0.4, "kp": 0.1, "ki": 100.0,
                     "gain_min": 0.5, "gain_max": 8.0, "rate_limit": 10.0,
                     "initial_gain": 1.0})
    a["dfe"].update({"enabled": False, "num_taps": 5, "algorithm": "sign-lms",
                     "mu": 1e-5, "leakage": 1e-6, "initial_taps": [0.0] * 5,
                     "tap_min": -1.0, "tap_max": 1.0, "stats_period": 4096,
                     "mu_startup": 1e-5, "mu_acquire": 1e-6, "mu_track": 1e-7})
    a["vref_adapt"]["enabled"] = False
    a["cdr_pi"].update({"enabled": False, "initial_phase": 0.0})
    a["safety"].update({"freeze_on_error": False, "rollback_enable": False})
    return c


def jitter_cfg(rj_ps, dcd_ps, sj_amp_ps, sigma=0.35, duration_s=1e-5):
    """S3 jitter subset (EDGE-01). sj_amp_ps is the SJ *peak amplitude* (matches
    mc_ber.make_jitter's sj_amp); stored as SJ_pp = 2*A in JitterParams."""
    c = build(True, sigma, duration_s)
    j = c["wave"]["jitter"]
    j["enable"] = True
    j["RJ_sigma"] = rj_ps * 1e-12
    j["DCD_delta"] = dcd_ps * 1e-12
    if sj_amp_ps > 0.0:
        j["SJ_freq"] = [10e6]          # validate_channel.py: SJ_FREQ = 10 MHz
        j["SJ_pp"] = [2.0 * sj_amp_ps * 1e-12]
    else:
        j["SJ_freq"] = []
        j["SJ_pp"] = []
    j["SJ_phase"] = 0.7                # validate_channel.py: seed_phase = 0.7 rad
    j["edge_csv"] = ""                 # set by the TB from the output prefix
    return c


def main():
    outdir = os.path.join(REPO, "config")
    files = {
        "ts3_anchor.json": build(False, 0.0, 2e-6),
        "ts3_s1_022.json": build(True, 0.22, 5e-5),
        "ts3_s1_030.json": build(True, 0.30, 5e-5),
        "ts3_s1_035.json": build(True, 0.35, 5e-5),
        "ts3_s1_050.json": build(True, 0.50, 5e-5),
        "ts3_s2_bare.json": build(True, 0.10, 2e-6, eq=False),
        # Phase 2 / S3 (v3 E5-E9, E11 equivalents; 1e5 UI each)
        "ts3_s3_E5.json": jitter_cfg(1.0, 0.0, 0.0),
        "ts3_s3_E6.json": jitter_cfg(2.0, 0.0, 0.0),
        "ts3_s3_E7.json": jitter_cfg(0.5, 0.5, 0.0),
        "ts3_s3_E8.json": jitter_cfg(0.5, 0.0, 1.0),
        "ts3_s3_E9.json": jitter_cfg(0.5, 0.5, 1.0),
        "ts3_s3_E11.json": jitter_cfg(10.0, 5.0, 5.0),
    }
    for name, cfg in files.items():
        path = os.path.join(outdir, name)
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
