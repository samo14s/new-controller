"""Simulated critical kicks and forced-command bounds at the design points of the margin chart (ba_kappa.json).

Purpose.  fair_compare.py compares the basin-aware chart (basin_aware.json) with the reference
designs by the same metrics.  The chart with the depth margin a <= 0.9 a_lim (ba_kappa.py) needs the
same simulated metric: the critical kick at its design points (rpm, g_rob, a_rob), smallest over the
kick phases 0, 4, 8, 12 of 16, with the protocol of the basin-aware validation and of fair_compare.py
(basin_aware.kick_job(('crit', rpm, a, g, ph)): face entry, kick at revolution 100, 300-revolution
window, 9 V clip, full surface memory, classifier D1), and the rigorous forced-command bound
basin_aware.bbar_of.  The default-order certified kick at a_rob is already in ba_kappa.json
(designs[b].beta_ref_at_a_rob); the entries are in ba_entry_kappa.json.

Jobs already computed with the same key by basin_aware.py (validate.jsonl) or fair_compare.py
(fair_compare_cache.jsonl) are reused.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/bak_crit.py [--workers 1]
Writes verification/paper/bak_crit.json (cache bak_crit_cache.jsonl).
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
OUT = os.path.join(HERE, "bak_crit.json")
CACHE = os.path.join(HERE, "bak_crit_cache.jsonl")
PHASES = (0, 4, 8, 12)


def kkey(args):
    return json.dumps([args[0], round(float(args[1]), 9), round(float(args[2]) * 1e3, 6), round(float(args[3]), 6)] +
                      list(args[4:]))


def stored_jobs():
    """crit results with the same job arguments from basin_aware's validate cache and fair_compare's cache."""
    out = {}
    p = os.path.join(HERE, "basin_aware_cache", "050bc4a99e", "validate.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("k")
            v = r.get("v")
            if isinstance(v, dict) and "v" in v:                # stored as {"v": value, "t": seconds}
                v = v["v"]
            if isinstance(k, list) and k and k[0] == "crit" and isinstance(v, (int, float)):
                out[kkey(k)] = float(v)
    p = os.path.join(HERE, "fair_compare_cache.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("k") or r.get("key")
            v = r.get("v") if "v" in r else r.get("value")
            if isinstance(v, dict) and "v" in v:
                v = v["v"]
            if isinstance(k, list) and k and k[0] == "crit" and isinstance(v, (int, float)):
                out[kkey(k)] = float(v)
    return out


def job(args):
    import basin_aware as BA
    BA._setup()
    t = time.time()
    if args[0] == "bbar":
        _, rpm, a, g = args
        return list(args), float(BA.bbar_of(rpm, g, a)), round(time.time() - t, 1)
    k, v, dt = BA.kick_job(tuple(args))
    return list(args), v, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    bk = json.load(open(os.path.join(HERE, "ba_kappa.json")))
    pts = {}
    for k, v in bk["speeds"].items():
        for b, d in v["designs"].items():
            if d.get("g") is None or not d.get("a_rob"):
                continue
            pts.setdefault((float(k), float(d["g"]), float(d["a_rob"]) * 1e-3), []).append(b)
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line)
                cache[kkey(r["k"])] = r["v"]
    st = stored_jobs()
    jobs, reused = [], 0
    for (n, g, a_) in pts:
        for job_args in [("bbar", n, a_, g)] + [("crit", n, a_, g, ph) for ph in PHASES]:
            kk = kkey(job_args)
            if kk in cache:
                continue
            if job_args[0] == "crit" and kk in st:
                cache[kk] = st[kk]
                reused += 1
                continue
            jobs.append(job_args)
    print(f"{len(pts)} design points, {len(jobs)} jobs to run, {reused} reused from stored caches", flush=True)
    if jobs:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for i, (k, v, dt) in enumerate(pool.imap_unordered(job, sorted(jobs, key=lambda j: j[0] != "crit")), 1):
                fh.write(json.dumps(dict(k=k, v=v, t=dt)) + "\n")
                fh.flush()
                cache[kkey(k)] = v
                print(f"[{i}/{len(jobs)}] {k}: {v} ({dt} s)", flush=True)
    per = {}
    for (n, g, a_), bs in sorted(pts.items()):
        ck = [cache.get(kkey(("crit", n, a_, g, ph))) for ph in PHASES]
        missing = any(c is None for c in ck)
        vals = [np.nan if c is None else float(c) for c in ck]          # nan: unsettled; inf: no chatter up to 3 m/s
        if missing:
            kind, crit = "missing", None
        elif any(np.isnan(x) for x in vals):
            kind, crit = "unsettled", 0.0                               # counted as 0, as in fair_compare.py
        elif all(np.isinf(x) for x in vals):
            kind, crit = "inf", None
        else:
            kind, crit = "exact", min(x for x in vals if np.isfinite(x)) * 1e3
        rec = dict(rpm=n, g=g, a_mm=a_ * 1e3, betas=bs, bbar_V=cache.get(kkey(("bbar", n, a_, g))),
                   crit_by_phase_mmps=[None if not np.isfinite(x) else x * 1e3 for x in vals],
                   crit_mmps=crit, crit_kind=kind)
        for b in bs:
            per.setdefault(f"{n:.0f}", {})[b] = rec
    out = dict(note=__doc__.split("\n\n")[0], phases=list(PHASES), speeds=per)
    crit = {}
    for b in ("0.001", "0.002", "0.005"):
        xs = [per[k][b]["crit_mmps"] for k in per if b in per[k] and per[k][b]["crit_mmps"] is not None]
        crit[b] = dict(n=len(xs), median_mmps=float(np.median(xs)) if xs else None, min_mmps=float(min(xs)) if xs else None,
                       n_inf=sum(1 for k in per if b in per[k] and per[k][b]["crit_kind"] == "inf"))
    out["summary"] = crit
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    main()
