"""The reference gains judged by the criteria of the basin-aware chart (review round 2, 4.1).

Purpose.  The basin-aware chart chooses, per speed, the gain that maximises the depth up to which
every grid depth is admissible (linear limit, forced-command bound <= 8.1 V, certified kick
>= beta_req; Section 6).  The reference schedules choose their gain by other rules (fair_designs.py:
the published condition 'pub', the authority condition with a voltage margin of 0 or 10 % and
g <= 3000, 'auth0' and 'auth10', and the manuscript's g <= 1500 schedule 'auth1500').  Judging the
reference gains by the chart's own criteria separates the effect of the selection rule from the
effect of the gain range: a_rob(n, g_ref; beta_req) is the depth that the reference gain could
certify, to be compared with a_rob(n; beta_req) = max over g.

Protocol.  basin_aware.scan_gain(rpm, g) (upward depth scan 2, 4, ..., 32 mm with the reduced
collocation order N_red and the solver defaults, stopping at the first depth not admissible for
1 mm/s) and basin_aware.evaluate(rows, beta_req) for beta_req = 1, 2, 5 mm/s; rows of the
basin-aware scan cache are reused where the same (rpm, g) was scanned.  Reduced order, not
re-certified with the default order (the default order lowers such certificates by at most 7 %,
basin_aware.json nred_vs_default).

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/ref_under_ba.py [--workers 2]
Writes verification/paper/ref_under_ba.json (cache ref_under_ba_cache.jsonl).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "ref_under_ba.json")
CACHE = os.path.join(HERE, "ref_under_ba_cache.jsonl")
SCAN_CACHE = os.path.join(HERE, "basin_aware_cache", "050bc4a99e")
DESIGNS = ("pub", "auth0", "auth10", "auth1500")
BETAS = (1.0, 2.0, 5.0)


def stored_scan(rpm, g):
    p = os.path.join(SCAN_CACHE, f"speed_{rpm:.0f}.jsonl")
    if not os.path.exists(p):
        return None
    for line in open(p):
        r = json.loads(line)
        k = r["k"]
        if k[0] == "scan" and abs(k[1] - rpm) < 1e-9 and abs(k[2] - g) < 1e-6:
            return r["v"]
    return None


def job(args):
    rpm, g = args
    import basin_aware as BA
    BA._setup()
    t = time.time()
    v = stored_scan(rpm, g)
    src = "basin_aware scan cache"
    if v is None:
        v = BA.scan_gain(rpm, g)
        src = "computed"
    return dict(rpm=rpm, g=g, scan=v, source=src, seconds=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    fd = json.load(open(os.path.join(HERE, "fair_designs.json")))
    ba = json.load(open(os.path.join(HERE, "basin_aware.json")))
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line)
                cache[(r["rpm"], r["g"])] = r
    jobs = sorted({(float(n), float(v[d]["g"])) for n, v in ((float(k), v) for k, v in fd["speeds"].items())
                   for d in DESIGNS} - set(cache))
    print(f"{len(jobs)} (speed, gain) scans to run, {len(cache)} cached", flush=True)
    if jobs:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for r in pool.imap_unordered(job, jobs):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                cache[(r["rpm"], r["g"])] = r
                print(f"{r['rpm']:.0f} rpm g {r['g']:.1f}: {len(r['scan']['rows'])} rows ({r['source']}, {r['seconds']} s)",
                      flush=True)
    import basin_aware as BA
    out = dict(note=__doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1], speeds={})
    for k, v in fd["speeds"].items():
        n = float(k)
        rec = {}
        for d in DESIGNS:
            g = float(v[d]["g"])
            sc = cache[(n, g)]["scan"]
            r = dict(g=g, a_lim_mm=sc["a_lim"], a_chk_mm=v[d]["a_chk"])
            for b in BETAS:
                a_rob, mins, fail = BA.evaluate(sc["rows"], b * 1e-3)
                r[f"beta_{b:g}"] = dict(a_rob_mm=a_rob, min_beta_mmps=float(mins) * 1e3,
                                        binding=BA.binding(fail, b * 1e-3, a_rob) if fail is not None else ["a_phys"])
            rec[d] = r
        rec["ba"] = {f"beta_{b:g}": dict(g=ba["per_speed"][k]["designs"][f"{b:g}"]["g"],
                                         a_rob_mm=ba["per_speed"][k]["designs"][f"{b:g}"]["a_rob_mm"]) for b in BETAS}
        out["speeds"][k] = rec
    import numpy as np
    summ = {}
    for d in DESIGNS + ("ba",):
        summ[d] = {f"beta_{b:g}": dict(mean_a_rob_mm=float(np.mean([out["speeds"][k][d][f"beta_{b:g}"]["a_rob_mm"]
                                                                   for k in out["speeds"]])))
                   for b in BETAS}
    out["summary"] = summ
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
