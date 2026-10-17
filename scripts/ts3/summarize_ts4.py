#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-S4 final evidence summary (A0/A1/A2/A3; read-only)."""

import argparse
import json
import math
import os

S = ("022", "030", "035", "050")
FW_A3 = {"022": 9.8944e-6, "030": 6.6299e-4, "035": 2.7622e-3, "050": 2.4646e-2}
FW_A0 = {"022": 1.955539889664507e-05, "030": 9.148810966876890e-04,
         "035": 3.402104971446699e-03, "050": 2.654618113179658e-02}
T3 = {"022": (3.600e-5, 18), "030": (1.214e-3, 607),
      "035": (4.378e-3, 2189), "050": (3.223e-2, 16114)}


def jl(p):
    with open(p) as f:
        return json.load(f)


def row(s, r, fw):
    band = max(0.10, 3.0 / math.sqrt(max(r["errors"], 1)))
    ratio = r["ber"] / fw[s]
    return (int(s) / 100.0, r["ber"], r["errors"], fw[s], ratio, band,
            "PASS" if abs(ratio - 1) <= band else "FAIL")


def table(title, rows):
    print("\n## " + title)
    print("| sigma_n | SystemC BER | errs | v3 BER | ratio | band | verdict |")
    print("|---|---|---|---|---|---|---|")
    for x in rows:
        print("| %.2f | %.4e | %d | %.4e | %.3f | %.1f%% | %s |"
              % (x[0], x[1], x[2], x[3], x[4], 100 * x[5], x[6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev", required=True)
    a = ap.parse_args()
    ev = a.ev

    a0 = [row(s, jl(os.path.join(ev, "artifacts", f"a0_s1_{s}_ber.json")), FW_A0) for s in S]
    a0on = [row(s, jl(os.path.join(ev, "artifacts_disc", f"a0_on_final_{s}_ber.json")), FW_A0) for s in S]
    table("A0 continuous sca_ss/simple original", a0)
    table("A0 ON: discrete ZOH channel+CTLE, erf, linear RX, 0.32 UI", a0on)

    a3 = {}
    for tag, prefix in (("discrete channel, saturated CTLE (0.28 UI)", "a3_final"),
                        ("discrete channel+CTLE, linear RX (0.28 UI)", "a3_ctle")):
        a3[tag] = [row(s, jl(os.path.join(ev, "artifacts_disc", f"{prefix}_{s}_ber.json")), FW_A3)
                   for s in S]
        table("A3 " + tag, a3[tag])
    print("\n## A3 T-S3 original")
    print("| sigma_n | SystemC BER | errs | v3 BER | ratio | band | verdict |")
    print("|---|---|---|---|---|---|---|")
    for s in S:
        ber, errs = T3[s]
        band = max(0.10, 3.0 / math.sqrt(errs))
        ratio = ber / FW_A3[s]
        print("| %.2f | %.4e | %d | %.4e | %.3f | %.1f%% | %s |"
              % (int(s) / 100, ber, errs, FW_A3[s], ratio, 100 * band,
                 "PASS" if abs(ratio - 1) <= band else "FAIL"))

    fit = jl(os.path.join(ev, "10_a1_fit.json"))
    print("\n## A1 fit")
    print("simulated:", json.dumps(fit["simulated"], indent=2))
    print("checks:", fit["checks"], "verdict:", fit["verdict"])
    chk = jl(os.path.join(ev, "11_a1_a2_check", "a2_peters_check.json"))
    print("independent A2:", chk["verdict"])

    print("\n## A3 phase trade-off (discrete channel+CTLE, linear RX)")
    for s in ("022", "050"):
        vals = []
        for p in ("024", "025", "026", "027", "028"):
            r = jl(os.path.join(ev, "artifacts_disc", f"a3_ctle_ph{p}_{s}_ber.json"))
            vals.append("%s:%.4e(%d)" % (p, r["ber"], r["errors"]))
        print("sigma", int(s) / 100, " ".join(vals))


if __name__ == "__main__":
    main()
