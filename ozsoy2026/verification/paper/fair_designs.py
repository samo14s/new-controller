"""Reference designs for the like-for-like comparison of the basin-aware chart (review round 2, 4.1).

At the 21 speeds of the basin-aware chart (500:125:3000 rpm), every reference design gets the same
gain range, g <= 3000 V s/m, and the same operating depth rule as the authority schedule of
Section 3: the checked depth a_chk = min(kappa a_lim(g), a_TD(g)), kappa = 0.9, where a_TD is the
step-entry limit with the 9 V clip and the full surface memory (src/run_feas_certify.py; every
depth up to a_chk settles after a step entry, scanned in 8 % steps from 0.225 a_lim and bisected).

  pub       the published condition, Eq. (13) of the manuscript, as reproduced in Section 3
            (src/gainmap.reference: dense scan of g in [0, 3000] with the step-entry demand at
            a_lim, then bisection on the edges of the feasible pockets).  Its own depth is a_lim
            (no margin); a_chk applies the common rule.
  auth0     the authority condition, Eq. (14), voltage margin 0 (steady command <= 9 V up to
            kappa a_lim): feasibility.edge_gain(g_hi = 3000, guard = 1.0).  Taken from the
            basin-aware scan cache ('compare', g3000), which recomputed it with the same code.
  auth10    the same with a 10 % voltage margin (steady command <= 8.1 V, guard = 0.9), the margin
            of the basin-aware chart.
  auth1500  the schedule of the manuscript (g <= 1500, margin 0), from the same cache, for
            continuity with Sections 3-6.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/fair_designs.py [--workers 4]
Writes verification/paper/fair_designs.json (resumable: one cache line per speed and design in
fair_designs_cache.jsonl).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "fair_designs.json")
CACHE = os.path.join(HERE, "fair_designs_cache.jsonl")
SPEEDS = tuple(float(n) for n in np.arange(500, 3001, 125))
KAPPA = 0.9
G_HI = 3000.0
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])


def _setup():
    import params as P
    P.set_damping(ZETA)
    return P


def checked_depth(rpm, g):
    """a_lim, kappa a_lim, a_TD and a_chk (mm) with the step-entry check of run_feas_certify."""
    P = _setup()
    import feasibility as F
    from tdboundary import boundary
    from zoa import a_lim
    vmax = P.F_MAX / P.GA_GAIN
    al = a_lim(rpm, g)
    td = boundary(rpm, g, vmax=vmax if g > 0 else 0.0, a_lo=0.25 * KAPPA * al, a_hi=KAPPA * al,
                  growth=1.08, nreg=F.NREG)
    return dict(a_lim=al * 1e3, kappa_a_lim=KAPPA * al * 1e3, a_td=td * 1e3, a_chk=min(KAPPA * al, td) * 1e3)


def job(args):
    rpm, kind = args
    P = _setup()
    t = time.time()
    if kind == "pub":
        import gainmap
        r = gainmap.reference(rpm, kind="force", g_hi=G_HI)
        g = float(r["g"])
        rec = dict(g=g, j_published_mm=float(r["a"]) * 1e3,
                   n_feasible_scan=int(np.sum(np.asarray(r["scan_f"]) > 0)), n_scan=int(len(r["scan_g"])))
    elif kind == "auth10":
        import feasibility as F
        g = float(F.edge_gain(rpm, g_hi=G_HI, guard=0.9))
        rec = dict(g=g, guard=0.9)
    else:
        raise ValueError(kind)
    rec.update(checked_depth(rpm, g))
    rec.update(rpm=rpm, kind=kind, seconds=round(time.time() - t, 1))
    return rec


def load_cache():
    out = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r["rpm"], r["kind"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cache = load_cache()
    todo = [(n, k) for k in ("pub", "auth10") for n in SPEEDS if (n, k) not in cache]
    # longest jobs (published scan) first
    print(f"{len(todo)} jobs to run, {len(cache)} cached", flush=True)
    if todo:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for rec in pool.imap_unordered(job, todo):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cache[(rec["rpm"], rec["kind"])] = rec
                print(f"{rec['kind']} {rec['rpm']:.0f} rpm: g {rec['g']:.1f}, a_lim {rec['a_lim']:.2f}, "
                      f"a_chk {rec['a_chk']:.2f} mm ({rec['seconds']} s)", flush=True)
    ba = json.load(open(os.path.join(HERE, "basin_aware.json")))
    out = dict(note=__doc__.split("\n\n")[0], kappa=KAPPA, g_hi=G_HI, zeta=ZETA, speeds={})
    for n in SPEEDS:
        ps = ba["per_speed"][f"{n:.0f}"]
        d = dict(pub=cache.get((n, "pub")), auth10=cache.get((n, "auth10")))
        for tag, src in (("auth0", "auth_g3000"), ("auth1500", "auth_g1500")):
            s = ps[src]
            d[tag] = dict(g=s["g"], a_lim=s["a_lim_mm"], a_chk=s["a_chk_mm"], source=f"basin_aware.json {src}",
                          kappa_a_lim=s.get("kappa_a_lim_mm", KAPPA * s["a_lim_mm"]))
        out["speeds"][f"{n:.0f}"] = d
    rows = [v for v in out["speeds"].values() if all(v[k] for k in ("pub", "auth10"))]

    def mean(tag, key):
        return float(np.mean([v[tag][key] for v in rows]))
    out["summary"] = dict(n_speeds_complete=len(rows),
                          mean_a_chk_mm={t: mean(t, "a_chk") for t in ("pub", "auth0", "auth10", "auth1500")},
                          mean_a_lim_mm={t: mean(t, "a_lim") for t in ("pub", "auth0", "auth10", "auth1500")},
                          mean_g={t: mean(t, "g") for t in ("pub", "auth0", "auth10", "auth1500")})
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
