"""Critical kicks of the model at the saturation tests of Ozsoy et al. 2025 that lie between the two linear limits.

Purpose.  Between the uncontrolled limit a_lim(0) and the controlled limit a_lim(g) the clipped DVF cut
can be bistable (Section 4 of the manuscript): the size of its basin, the critical kick, decides whether
a disturbance ends in chatter.  For every test of the article's Fig. 10 (re-clamped structure,
g = 253 V s/m) whose depth lies between the two limits of the model, this script computes the simulated
critical kick with the protocol of Section 4 (c1_unsafe_zone._critical_kick: face entry, full surface
memory, 9 V clip, kick at revolution 100, 300-revolution window, classifier D1; kicks of 0.1 mm/s to
3 m/s searched upwards and bisected; inf = no chatter up to 3 m/s, nan = the cut does not settle), at
the kick phases 0, 4, 8 and 12 of 16, the smallest kept.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/ozsoy2025/kicks2025.py [--workers 3]
Writes kicks2025.json (cache kicks2025_cache.jsonl).
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

CACHE = os.path.join(HERE, "kicks2025_cache.jsonl")
PHASES = (0, 4, 8, 12)


def job(args):
    rpm, depth_mm, g, ph = args
    S.apply_structure(**S.RECLAMPED)
    import c1_unsafe_zone as C
    t = time.time()
    v = float(C._critical_kick((float(rpm), depth_mm * 1e-3, float(g), C.VMAX, int(ph), S.RECLAMPED["zeta"])))
    return list(args), v, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cmp = json.load(open(os.path.join(HERE, "compare2025.json")))["fig10"]["rows"]
    pts = [r for r in cmp if r["position"].startswith("between")]
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line)
                cache[json.dumps(r["k"])] = r["v"]
    jobs = [[r["rpm"], r["depth_mm"], 253.0, ph] for r in pts for ph in PHASES]
    todo = [j for j in jobs if json.dumps(j) not in cache]
    print(f"{len(pts)} points, {len(todo)} kick searches to run", flush=True)
    if todo:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for k, v, dt in pool.imap_unordered(job, todo):
                fh.write(json.dumps(dict(k=k, v=v, t=dt)) + "\n")
                fh.flush()
                cache[json.dumps(k)] = v
                print(f"  {k}: {v} ({dt} s)", flush=True)
    rows = []
    for r in pts:
        vals = [cache[json.dumps([r["rpm"], r["depth_mm"], 253.0, ph])] for ph in PHASES]
        eff = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in vals]
        crit = min(eff)
        rows.append(dict(rpm=r["rpm"], depth_mm=r["depth_mm"], measured=r["measured"], measured_outcome=r["measured_outcome"],
                         demand_N=r["demand_N"], residual_authority_N=r["residual_authority_N"],
                         crit_by_phase_mmps=[None if not np.isfinite(x) else x * 1e3 for x in eff],
                         crit_mmps=None if not np.isfinite(crit) else crit * 1e3, crit_inf=bool(np.isinf(crit))))
    out = dict(note=__doc__.split("\n\n")[0], phases=list(PHASES), rows=rows)
    with open(os.path.join(HERE, "kicks2025.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(r["rpm"], r["depth_mm"], r["measured"], r["measured_outcome"], round(r["demand_N"], 1), r["crit_mmps"], "inf" if r["crit_inf"] else "")


if __name__ == "__main__":
    main()
