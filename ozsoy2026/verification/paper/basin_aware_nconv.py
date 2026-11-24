"""Collocation order N for the basin-aware gain scan (verification/paper/basin_aware.py).

Purpose.  The basin-aware scan needs the certified kick beta (c2_certificate.certify_kick,
e = unit velocity) at many (speed, gain, depth) points.  The default order
N = nodes_for(rpm) = 16 + 4 ceil(T f_n) (20-32 nodes) costs 5-90 s per certificate, so the
scan uses a reduced order and re-certifies its design points with the default one.  This
script measures what a reduced order costs in accuracy and saves in time.

Protocol.
  A. Linear limit of the collocated averaged model (C2's model, DVF inside A): first depth
     at which the rightmost eigenvalue crosses zero (upward scan by 2 % from 0.05 a_lim,
     then 30 bisections), divided by the ZOA limit zoa.a_lim, for N = 10, 12, 14,
     nodes_for - 8 and nodes_for, at the 21 speeds 500:125:3000 rpm and g = 0, 250, 1000,
     1500, 3000 V s/m (None: no crossing below 2 a_lim).  A fixed small N (10-14) is badly
     wrong at low speed, where the delay spans up to 3.9 structural periods.
  B. Certified kick at ten points (the three points of the manuscript's Table nconv and
     seven others spread over speed, depth and gain) for N = 10, 12, 14, nodes_for - 8 and
     nodes_for, with Clarabel's default tolerances (c2.FAST, used by the scan) and with
     c2.TIGHT (1e-10, used by the manuscript's 48-point C2 campaign), with wall times
     (single core, machine shared with other jobs).  beta is reported as certify_kick
     returns it (command bound of c2.orbit); the rescaling to the rigorous bound of
     c3_periodic.orbit does not depend on N.  The stored values at N, 1.5N, 2N (TIGHT) of
     verification/paper/revision_runs.json (beta_vs_N) are copied for reference.
  The damping is that of every stored result: params.set_damping(gainmap_main.npz zeta).

How to run (from the repository root ozsoy2026/, about 20 min on one core):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/basin_aware_nconv.py --workers 1
Writes verification/paper/basin_aware_nconv.json.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])

SPEEDS = tuple(float(n) for n in np.arange(500.0, 3001.0, 125.0))
LIN_GAINS = (0.0, 250.0, 1000.0, 1500.0, 3000.0)
# (rpm, depth, g): depth > 0 in m, depth < 0 a fraction -depth of a_lim(rpm, g)
POINTS = ((2500.0, -0.9, 1500.0), (1090.0, -0.9, 1500.0), (2870.0, -0.3, 1500.0),
          (500.0, 10e-3, 1500.0), (625.0, 24e-3, 2500.0), (750.0, 12e-3, 1500.0),
          (1250.0, 8e-3, 1000.0), (1750.0, 15e-3, 500.0), (2000.0, 12e-3, 200.0),
          (2500.0, 6e-3, 1000.0))
LABELS = ("10", "12", "14", "nodes_for-8", "nodes_for")


def orders(rpm):
    """label -> N at this speed."""
    import c2_certificate as c2
    nd = c2.nodes_for(rpm)
    return dict(zip(LABELS, (10, 12, 14, nd - 8, nd)))


def _setup():
    import params as P
    P.set_damping(ZETA)


def lin_job(args):
    """Collocated linear limit over the ZOA limit at one (speed, gain), for each N."""
    rpm, g = args
    _setup()
    import c2_certificate as c2
    from zoa import a_lim
    al = a_lim(rpm, g)

    def lead(a, N):
        return float(np.linalg.eigvals(c2.model(rpm, a, g, N)[0]).real.max())

    out = {}
    for N in sorted(set(orders(rpm).values())):
        lo, a = 1e-6, 0.05 * al
        while lead(a, N) < 0 and a <= 2.0 * al:
            lo, a = a, a * 1.02
        if a > 2.0 * al:
            out[N] = None                            # stable up to 2 a_lim: the model is wrong
            continue
        hi = a
        for _ in range(30):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if lead(m, N) < 0 else (lo, m)
        out[N] = lo / al
    return dict(rpm=rpm, g=g, a_lim_mm=al * 1e3, nodes_for=c2.nodes_for(rpm),
                ratio={lab: out[N] for lab, N in orders(rpm).items()})


def beta_job(args):
    rpm, dep, g, N, setting = args
    _setup()
    import c2_certificate as c2
    from zoa import a_lim
    a = dep if dep > 0 else -dep * a_lim(rpm, g)
    st = dict(FAST=c2.FAST, TIGHT=c2.TIGHT)[setting]
    t = time.time()
    beta, note = c2.certify_kick(rpm, a, g, N=N, settings=st)
    return (rpm, dep, g, N, setting), dict(a_mm=a * 1e3, beta_mmps=beta * 1e3, note=note, time_s=time.time() - t)


def stats(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return dict(min=float(np.min(v)), max=float(np.max(v)), median=float(np.median(v)), n=len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    lin_jobs = [(n, g) for n in SPEEDS for g in LIN_GAINS]
    beta_jobs = sorted({(n, d, g, N, s) for (n, d, g) in POINTS for N in orders(n).values() for s in ("FAST", "TIGHT")},
                       key=lambda j: -(j[3] ** 3) * (3 if j[4] == "TIGHT" else 1))       # longest first
    with Pool(a.workers) as pool:
        lin = pool.map(lin_job, lin_jobs, chunksize=4)
        res = dict(pool.map(beta_job, beta_jobs, chunksize=1))
    # A: summary per label
    lin_sum = {}
    for lab in LABELS:
        v = np.array([np.inf if r["ratio"][lab] is None else r["ratio"][lab] for r in lin])
        bad = np.abs(v - 1) > 1e-3
        lin_sum[lab] = dict(min=float(v.min()), max=float(v.max()), n_off_by_more_than_1e3=int(bad.sum()),
                            speeds_off_by_more_than_1e3=sorted({r["rpm"] for r, b in zip(lin, bad) if b}))
    # B: rows per point and label, ratios to the default order
    rows = []
    for (n, d, g) in POINTS:
        Ns = orders(n)
        ref = {s: res[(n, d, g, Ns["nodes_for"], s)]["beta_mmps"] for s in ("FAST", "TIGHT")}
        for lab in LABELS:
            for s in ("FAST", "TIGHT"):
                r = res[(n, d, g, Ns[lab], s)]
                rows.append(dict(rpm=n, a_mm=r["a_mm"], g=g, label=lab, N=Ns[lab], settings=s,
                                 beta_mmps=r["beta_mmps"], time_s=round(r["time_s"], 2), note=r["note"],
                                 over_default_same_settings=r["beta_mmps"] / ref[s] if ref[s] > 0 else None,
                                 over_default_tight=r["beta_mmps"] / ref["TIGHT"] if ref["TIGHT"] > 0 else None))
    summ = {}
    for lab in LABELS:
        for s in ("FAST", "TIGHT"):
            sel = [r for r in rows if r["label"] == lab and r["settings"] == s]
            summ[f"{lab} {s}"] = dict(over_default_same_settings=stats([r["over_default_same_settings"] for r in sel]),
                                      over_default_tight=stats([r["over_default_tight"] for r in sel]),
                                      time_s=stats([r["time_s"] for r in sel]))
    rev = json.load(open(os.path.join(HERE, "revision_runs.json")))
    out = dict(note=__doc__, zeta=ZETA, linear_limit=dict(rows=lin, summary=lin_sum),
               beta=dict(rows=rows, summary=summ), stored_beta_vs_N_tight=rev.get("beta_vs_N"),
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, "basin_aware_nconv.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(linear_limit=lin_sum, beta=summ), indent=1, default=float))
    for r in rows:
        print(f"{r['rpm']:6.0f} {r['a_mm']:6.2f} mm g {r['g']:6.0f} {r['label']:12s} N={r['N']:2d} {r['settings']:5s} "
              f"beta {r['beta_mmps']:8.4f} mm/s  {r['time_s']:6.1f} s  [{r['note']}]")


if __name__ == "__main__":
    main()
