"""Simulated outcome of the cutting tests of Ozsoy et al. 2025 with the clipped DVF model of this repository.

Purpose.  The 2025 article reports, for DVF control, (i) controlled cuts on its stability lobe diagram
(its Fig. 6; structure of its Table 2 = src/params.py, gain 253 V s/m) and (ii) saturation tests at
1700-2200 rpm (its Figs. 10 and 12; re-clamped structure 128.1 Hz, 1.48 %, 1.13e7 N/m) with the measured
amplitude of the command in N, 'Sat' where it saturated, and the outcome (chatter, marginal) where not
stable.  This script predicts every test point with the model of the manuscript.

Per point (speed n, depth a, gain g; structure as above):
  closed form   a_lim(0) and a_lim(g) (src/zoa.py, zero-order); the settled forced demand K_a g max|x_p'|
                (N; src/feasibility.forced_demand: 40 tooth harmonics of the helical force, no
                regeneration, exact for the tooth-periodic forced orbit), its tooth-harmonic amplitude and
                sqrt(2) RMS (forced_map.demand_components); residual authority 27 N - demand.
  simulator     src/tdsim (time domain, 30 axial layers, full surface memory NREG = 0, loss of contact,
                command clipped at +-9 V = +-27 N), two entries: 'face' (the tool enters the work across
                its radius at the feed: tdboundary.classify(..., edge=True)) and 'step' (full depth from
                rest); classified by tdboundary.classify (classifier D1, 300 revolutions after full
                engagement).  Recorded: chatter, the peak unclipped demand K_a g |x'| over the last 50
                revolutions (settled) and over the entry, the fraction of the last 50 revolutions with the
                command on the clip, and the peak displacement.
Inputs: points_fig10.json and points_fig6.json in this directory (digitised from the article, see
digitise_notes.md), each a list of {rpm, depth_mm, ...measured fields}.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/ozsoy2025/sim2025.py [--workers 3] [--only fig10|fig6]
Writes sim2025_fig10.json and sim2025_fig6.json (cache sim2025_cache.jsonl).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup2025 as S  # noqa: E402

CACHE = os.path.join(HERE, "sim2025_cache.jsonl")
VMAX = 9.0
SETTLED_REVS = 50
PROTO = "sim2025-v1"


def key_of(case, rpm, depth_mm, g, entry):
    return json.dumps([PROTO, case, round(float(rpm), 6), round(float(depth_mm), 6), round(float(g), 6), entry])


def job(args):
    case, rpm, depth_mm, g, entry = args
    import params as P
    import tdboundary as TB
    S.apply_structure(**(S.RECLAMPED if case == "fig10" else S.NOMINAL))
    t0 = time.time()
    a = depth_mm * 1e-3
    kw = dict(edge=True) if entry == "face" else {}
    c = TB.classify(float(rpm), a, float(g), vmax=VMAX, nreg=0, **kw)
    r = c["run"]
    spr = r["spr"]
    v = np.asarray(r["v"])
    vc = np.asarray(r["vcmd"])
    dem = P.GA_GAIN * float(g) * np.abs(v)                    # unclipped demand, N
    n_end = len(v) - 1
    s0 = max(0, n_end - SETTLED_REVS * spr)
    start = int(np.ceil(TB.entry_revs(**kw))) * spr
    out = dict(chatter=bool(c["chatter"]), mu=float(c["mu"]) if np.isfinite(c["mu"]) else None,
               settled_peak_demand_N=float(dem[s0:].max()),
               entry_peak_demand_N=float(dem[:max(start, spr)].max()),
               run_peak_demand_N=float(dem.max()),
               settled_clip_fraction=float(np.mean(np.abs(vc[s0:]) >= VMAX - 1e-9)) if g > 0 else 0.0,
               run_clipped=bool(np.any(np.abs(vc) >= VMAX - 1e-9)) if g > 0 else False,
               max_x_mm=float(r["max_x"] * 1e3), diverged=bool(r["diverged"]))
    return list(args), out, round(time.time() - t0, 1)


def closed_form(case, rpm, depth_mm, g):
    import zoa
    import params as P
    from forced_map import demand_components
    S.apply_structure(**(S.RECLAMPED if case == "fig10" else S.NOMINAL))
    pk, h1, r2 = demand_components(float(rpm), float(g), np.array([depth_mm * 1e-3]))
    return dict(a_lim0_mm=float(zoa.a_lim(rpm, 0.0) * 1e3), a_lim_g_mm=float(zoa.a_lim(rpm, g) * 1e3),
                demand_N=float(pk[0]), h1_N=float(h1[0]), rms2_N=float(r2[0]),
                residual_authority_N=float(P.F_MAX - pk[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--g", type=float, default=S.G_FIG6)
    ap.add_argument("--only", default=None, choices=[None, "fig10", "fig6"])
    a = ap.parse_args()
    cases = [c for c in ("fig10", "fig6") if a.only in (None, c)]
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                rr = json.loads(line)
                cache[key_of(*rr["k"])] = rr["v"]
    pts = {}
    jobs = []
    for case in cases:
        p = os.path.join(HERE, f"points_{case}.json")
        pts[case] = json.load(open(p))["points"]
        for q in pts[case]:
            g = q.get("g", a.g)
            for entry in ("face", "step"):
                k = (case, q["rpm"], q["depth_mm"], g, entry)
                if key_of(*k) not in cache and k not in jobs:
                    jobs.append(k)
    print(f"{len(jobs)} simulations to run", flush=True)
    if jobs:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for i, (k, v, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
                fh.write(json.dumps(dict(k=k, v=v, t=dt)) + "\n")
                fh.flush()
                cache[key_of(*k)] = v
                print(f"[{i}/{len(jobs)}] {k}: chatter {v['chatter']}, settled demand {v['settled_peak_demand_N']:.1f} N, "
                      f"clip {v['settled_clip_fraction']:.2f} ({dt} s)", flush=True)
    for case in cases:
        rows = []
        for q in pts[case]:
            g = q.get("g", a.g)
            rec = dict(q, g=g, closed_form=closed_form(case, q["rpm"], q["depth_mm"], g),
                       face=cache[key_of(case, q["rpm"], q["depth_mm"], g, "face")],
                       step=cache[key_of(case, q["rpm"], q["depth_mm"], g, "step")])
            rows.append(rec)
        S.apply_structure(**S.NOMINAL)
        out = dict(note=__doc__.split("\n\n")[0], protocol=PROTO, structure=S.RECLAMPED if case == "fig10" else S.NOMINAL,
                   vmax_V=VMAX, settled_revs=SETTLED_REVS, points=rows)
        with open(os.path.join(HERE, f"sim2025_{case}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote sim2025_{case}.json ({len(rows)} points)")


if __name__ == "__main__":
    main()
