"""Simulated counterparts of the measured cuts (Figs. 9-12) and of every test point.

The experiments cannot be repeated.  Two things are done instead:
 1. every one of the 21 published test points (verification/targets/
    experimental_points.csv) is classified by the zero-order limit with
    (a) the paper's own gain at that speed (Fig. 8, 237 V s/m, or none) and
    (b) the reproduced gain, and with (b) also in the time domain
    (D1 detector, 9 V command limit);
 2. the six labelled cuts of Figs. 10 and 12 are simulated over 300
    revolutions; the once-per-revolution samples and the spectrum of the
    acceleration are stored, with the dominant non-synchronous frequency.
    python run_cuts.py --tag main
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool
import numpy as np
import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
POINTS = os.path.join(HERE, "..", "verification", "targets", "experimental_points.csv")
DIG = os.path.join(HERE, "..", "digitized")
VMAX = P.F_MAX / P.GA_GAIN
PAPER_SINGLE = 237.0
LABELLED = {("Fig9", "A"): "10A", ("Fig9", "B"): "10B", ("Fig11", "A"): "12A",
            ("Fig11", "B"): "12B", ("Fig11", "C"): "12C", ("Fig11", "D"): "12D"}


def spectrum(sig, dt):
    w = np.hanning(len(sig))
    S = np.abs(np.fft.rfft((sig - sig.mean()) * w)) * 2 / w.sum()
    return np.fft.rfftfreq(len(sig), dt), S


def work(job):
    P.set_damping(job["zeta"])
    from zoa import a_lim
    from tdboundary import classify
    out = dict(job)
    for key in ("paper", "ours"):
        g = job[f"g_{key}"]
        al, fc = a_lim(job["rpm"], g, return_freq=True)
        out[f"alim_{key}"] = al * 1e3
        out[f"fc_{key}"] = fc
        out[f"pred_{key}"] = "stable" if job["depth_mm"] < al * 1e3 else "chatter"
    g = job["g_ours"]
    c = classify(job["rpm"], job["depth_mm"] * 1e-3, g, vmax=VMAX if g > 0 else 0.0)
    out["pred_ours_td"] = "stable" if c["stable"] else "chatter"
    out["max_fact_td"] = float(np.abs(c["run"]["fact"]).max())
    run = c["run"]
    if job["cut"]:
        t, v = run["t"], run["v"]
        acc = np.gradient(v, t)
        dt = t[1] - t[0]
        tail = acc[len(acc) // 2:]
        f, S = spectrum(tail, dt)
        ftp = P.N_TEETH * job["rpm"] / 60.0
        fs = job["rpm"] / 60.0
        sync = np.min(np.abs(f[:, None] - fs * np.arange(1, 60)[None, :]), axis=1) < 1.0
        band = (f > 20) & (f < 400)
        jn = int(np.argmax(np.where(band & ~sync, S, 0)))
        jh = int(np.argmax(np.where(band & sync, S, 0)))
        out.update(f_nonsync=float(f[jn]), amp_nonsync=float(S[jn]),
                   f_sync=float(f[jh]), amp_sync=float(S[jh]), ftp=ftp)
        spr = run["spr"]
        sel = f < 450
        out["_series"] = dict(t=t[::8], acc=acc[::8], opr_t=t[::spr], opr=acc[::spr],
                              f=f[sel], S=S[sel])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
    sg = np.load(os.path.join(RES, f"single_gain_{a.tag}.npz"))
    zeta = float(d["zeta"])
    gmap = d["g_sade"] if "g_sade" in d else d["g_ref"]
    g8 = np.loadtxt(os.path.join(DIG, "fig8_gain_points.csv"), delimiter=",", skiprows=1)
    jobs = []
    with open(POINTS) as fh:
        for r in csv.DictReader(fh):
            rpm, depth = float(r["rpm"]), float(r["depth_mm"])
            ctrl = r["control"]
            if ctrl.startswith("mapped"):
                gp, go, kind = float(np.interp(rpm, g8[:, 0], g8[:, 1])), float(np.interp(rpm, d["rpm"], gmap)), "map"
            elif ctrl.startswith("single"):
                gp, go, kind = PAPER_SINGLE, float(sg["g_ref"]), "single"
            else:
                gp, go, kind = 0.0, 0.0, "none"
            jobs.append(dict(figure=r["figure"], label=r["label"], rpm=rpm, depth_mm=depth,
                             controller=kind, measured=r["state"], g_paper=gp, g_ours=go,
                             zeta=zeta, cut=LABELLED.get((r["figure"], r["label"]), "")))
    with Pool(a.workers) as pool:
        rows = pool.map(work, jobs)
    series = {}
    for r in rows:
        s = r.pop("_series", None)
        if s:
            series.update({f"{r['cut']}_{k}": v for k, v in s.items()})
    np.savez_compressed(os.path.join(RES, f"cuts_{a.tag}.npz"), **series)
    with open(os.path.join(RES, f"cuts_{a.tag}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    agree = {k: sum(r[f"pred_{k}"] == r["measured"] for r in rows) for k in ("paper", "ours", "ours_td")}
    print(f"{len(rows)} test points; agreement with the measured outcome: "
          f"ZOA at the paper's gains {agree['paper']}, ZOA at reproduced gains {agree['ours']}, "
          f"time domain at reproduced gains {agree['ours_td']}")
    for r in rows:
        print(f"{r['figure']:5s} {r['label'] or '-':1s} {r['controller']:6s} {r['rpm']:6.0f} rpm "
              f"{r['depth_mm']:5.1f} mm  measured {r['measured']:7s} | paper g={r['g_paper']:6.0f} "
              f"a_lim={r['alim_paper']:6.2f} -> {r['pred_paper']:7s} | ours g={r['g_ours']:6.0f} "
              f"a_lim={r['alim_ours']:6.2f} -> {r['pred_ours']:7s} TD {r['pred_ours_td']}"
              + (f" | non-sync {r['f_nonsync']:.1f} Hz ({r['amp_nonsync']:.3g}) vs sync "
                 f"{r['f_sync']:.1f} Hz ({r['amp_sync']:.3g})" if r["cut"] else ""))


if __name__ == "__main__":
    main()
