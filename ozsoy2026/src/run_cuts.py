"""Simulated counterparts of the measured cuts of Figs. 10 and 12.

The paper shows once-per-revolution (OPR) acceleration and its spectrum for
labelled cuts.  The experiments cannot be repeated here; instead each cut is
simulated with the reproduced gains and classified with the same reading the
paper uses (OPR samples settle / spectrum shows only tooth-passing harmonics).
    python run_cuts.py --tag main
"""
import argparse
import json
import os
import numpy as np
import params as P
from tdsim import simulate
from tdboundary import classify

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VMAX = P.F_MAX / P.GA_GAIN

# (label, figure, rpm, depth mm, controller, measured outcome in the paper)
CUTS = [
    ("10A", "Fig10", 1300, 8.0, "map", "stable"),
    ("10B", "Fig10", 1300, 10.0, "map", "chatter"),
    ("12A", "Fig12", 2800, 0.5, "none", "stable"),
    ("12B", "Fig12", 2800, 1.5, "none", "chatter"),
    ("12C", "Fig12", 2700, 3.0, "single", "stable"),
    ("12D", "Fig12", 2700, 4.0, "single", "chatter"),
]


def accel(run):
    """Acceleration from the simulated velocity (central difference)."""
    v, t = run["v"], run["t"]
    return np.gradient(v, t)


def spectrum(sig, dt):
    n = len(sig)
    w = np.hanning(n)
    S = np.abs(np.fft.rfft((sig - sig.mean()) * w)) * 2 / w.sum()
    return np.fft.rfftfreq(n, dt), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nrev", type=int, default=200)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
    sg = np.load(os.path.join(RES, f"single_gain_{a.tag}.npz"))
    P.set_damping(float(d["zeta"]))
    gmap = d["g_sade"] if "g_sade" in d else d["g_ref"]
    out, summary = {}, []
    for lab, fig, rpm, depth, ctrl, measured in CUTS:
        if ctrl == "map":
            g = float(np.interp(rpm, d["rpm"], gmap))
        elif ctrl == "single":
            g = float(sg["g_ref"])
        else:
            g = 0.0
        c = classify(rpm, depth * 1e-3, g, vmax=VMAX if g > 0 else 0.0, nrev=a.nrev)
        run = c["run"]
        acc = accel(run)
        spr = P.STEPS_PER_REV
        dt = run["t"][1] - run["t"][0]
        tail = acc[len(acc) // 2:]
        f, S = spectrum(tail, dt)
        ftp = P.N_TEETH * rpm / 60.0
        # dominant non-harmonic peak
        harm = np.min(np.abs(f[:, None] - ftp * np.arange(1, 40)[None, :]), axis=1) < 2.0
        band = (f > 20) & (f < 400)
        jn = np.argmax(np.where(band & ~harm, S, 0))
        jh = np.argmax(np.where(band & harm, S, 0))
        pred = "chatter" if c["chatter"] else "stable"
        row = dict(label=lab, figure=fig, rpm=rpm, depth_mm=depth, controller=ctrl, gain=g,
                   measured=measured, predicted=pred, agree=pred == measured,
                   slope_per_rev=c["slope"], level=c["level"],
                   peak_nonharmonic_hz=float(f[jn]), peak_nonharmonic=float(S[jn]),
                   peak_harmonic_hz=float(f[jh]), peak_harmonic=float(S[jh]),
                   max_fact=float(np.abs(run["fact"]).max()))
        summary.append(row)
        out[lab] = dict(t=run["t"][::4], acc=acc[::4], opr_t=run["t"][::spr], opr=acc[::spr],
                        f=f[f < 450], S=S[f < 450], ftp=ftp)
        print(f"{lab}: {rpm} rpm {depth} mm {ctrl} g={g:.0f} -> {pred} (paper: {measured}); "
              f"chatter peak {f[jn]:.1f} Hz amp {S[jn]:.3g}, harmonic {f[jh]:.1f} Hz amp {S[jh]:.3g}")
    np.savez_compressed(os.path.join(RES, f"cuts_{a.tag}.npz"),
                        **{f"{k}_{kk}": vv for k, v in out.items() for kk, vv in v.items()})
    with open(os.path.join(RES, f"cuts_{a.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
