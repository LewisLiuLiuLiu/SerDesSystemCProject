#!/usr/bin/env python3
"""
Compare the simulated channel transfer function against a Touchstone S-parameter.

2026-10 rewrite: the previous version compared |FFT(vout)| directly against |S21|
without normalizing by the input spectrum — physically meaningless (the waveform
CSV now records input_V and output_V, so the true transfer function is available):

    H_sim(f) = FFT(vout * w) / FFT(vin * w)

with a Hann window and the initial transient trimmed. Magnitude (dB) and
unwrapped phase are compared against S21 over the *usable band*
(|S21| within 20 dB of its peak, and below the Touchstone's fmax).

Usage:
  python3 scripts/compare_frequency_response.py <file.s4p> <waveform.csv>
      [--sparam S21] [--trim 0.2] [--band-db -20] [--plot out.png]

Exit code 0 when the fit passes (median |dB error| <= 1.5 dB and
90th percentile <= 3.0 dB in the usable band), 1 otherwise.
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_touchstone(path, sparam):
    import skrf
    nw = skrf.Network(path)
    a = int(sparam[1]) - 1
    b = int(sparam[2]) - 1
    if a >= nw.nports or b >= nw.nports:
        raise SystemExit(f"{sparam} out of range for {nw.nports}-port file")
    return nw.frequency.f.copy(), nw.s[:, a, b].copy()


def load_waveform(path):
    with open(path) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if "input_V" not in header or "output_V" not in header:
        raise SystemExit(
            f"waveform must contain input_V/output_V (have: {header}); "
            "rerun the channel testbench with the current code")
    return (data[:, 0],
            data[:, header.index("input_V")],
            data[:, header.index("output_V")])


def compare(s4p_file, waveform_file, sparam="S21", trim=0.2, band_db=-20.0,
            plot_file=None):
    f_orig, s_orig = load_touchstone(s4p_file, sparam)
    t, vin, vout = load_waveform(waveform_file)

    dt = t[1] - t[0]
    n0 = int(len(t) * trim)
    vin = vin[n0:]
    vout = vout[n0:]
    n = len(vin)
    w = np.hanning(n)
    X = np.fft.rfft(vin * w)
    Y = np.fft.rfft(vout * w)
    freq = np.fft.rfftfreq(n, dt)

    # Simulation frequency grid limited to the Touchstone range
    fmax = f_orig[-1]
    band = (freq > 50e6) & (freq <= fmax)
    xmag = np.abs(X)
    # Skip excitation nulls (PRBS lines with negligible energy)
    valid = band & (xmag > np.max(xmag[band]) * 1e-3)
    fv = freq[valid]
    H = Y[valid] / X[valid]

    # Interpolate the reference (real + imag to keep phase)
    s_ref = (np.interp(fv, f_orig, s_orig.real)
             + 1j * np.interp(fv, f_orig, s_orig.imag))

    # Usable band: reference magnitude within band_db of its peak
    mag_ref_db = 20 * np.log10(np.abs(s_ref) + 1e-20)
    usable = mag_ref_db >= (np.max(mag_ref_db) + band_db)
    if np.count_nonzero(usable) < 20:
        raise SystemExit("not enough usable-band points; check the data")

    f_u = fv[usable]
    mag_sim_db = 20 * np.log10(np.abs(H[usable]) + 1e-20)
    mag_ref_u = mag_ref_db[usable]
    dberr = np.abs(mag_sim_db - mag_ref_u)

    # Phase (unwrapped), aligned by removing a constant offset (fit delay free)
    ph_sim = np.unwrap(np.angle(H[usable]))
    ph_ref = np.unwrap(np.angle(s_ref[usable]))
    dphi = ph_sim - ph_ref
    dphi = dphi - np.median(dphi)

    med_db = float(np.median(dberr))
    p90_db = float(np.percentile(dberr, 90))
    rmse_db = float(np.sqrt(np.mean(dberr ** 2)))
    med_phase = float(np.median(np.abs(dphi)))

    print(f"Reference: {s4p_file} ({sparam})")
    print(f"Waveform : {waveform_file} (trim={trim:.0%}, window=Hann)")
    print(f"Usable band: {f_u[0]/1e9:.2f}-{f_u[-1]/1e9:.2f} GHz, "
          f"{np.count_nonzero(usable)} points (|S| >= peak{band_db:+.0f} dB)")
    print(f"Magnitude error: median={med_db:.2f} dB, p90={p90_db:.2f} dB, "
          f"RMSE={rmse_db:.2f} dB")
    print(f"Phase error (offset removed): median={med_phase:.3f} rad")

    if plot_file:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        axes[0].plot(f_u / 1e9, mag_ref_u, "k-", lw=1.5, label=f"{sparam} reference")
        axes[0].plot(f_u / 1e9, mag_sim_db, "r--", lw=1.2, label="simulated H=Y/X")
        axes[0].set_ylabel("Magnitude [dB]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Channel transfer function check: median {med_db:.2f} dB "
                          f"(p90 {p90_db:.2f} dB)")
        axes[1].plot(f_u / 1e9, ph_ref, "k-", lw=1.2, label="reference")
        axes[1].plot(f_u / 1e9, ph_sim - np.median(ph_sim - ph_ref), "r--", lw=1.0,
                     label="simulated")
        axes[1].set_xlabel("Frequency [GHz]")
        axes[1].set_ylabel("Unwrapped phase [rad]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {plot_file}")

    passed = (med_db <= 1.5) and (p90_db <= 3.0)
    print(f"RESULT: {'PASS' if passed else 'FAIL'} "
          f"(median <= 1.5 dB, p90 <= 3.0 dB)")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulated channel H(f)=Y/X against Touchstone S-parameters")
    parser.add_argument("s4p", help="Touchstone file (.s2p/.s4p)")
    parser.add_argument("waveform", help="Channel TB CSV (time_s,input_V,output_V)")
    parser.add_argument("--sparam", default="S21", help="S-parameter to compare (default S21)")
    parser.add_argument("--trim", type=float, default=0.2,
                        help="Fraction of the record to trim as transient (default 0.2)")
    parser.add_argument("--band-db", type=float, default=-20.0,
                        help="Usable-band floor relative to peak (default -20 dB)")
    parser.add_argument("--plot", default=None, help="Optional PNG output path")
    args = parser.parse_args()
    ok = compare(args.s4p, args.waveform, args.sparam, args.trim,
                 args.band_db, args.plot)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
