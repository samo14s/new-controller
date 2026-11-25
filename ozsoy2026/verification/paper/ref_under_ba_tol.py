"""The reference gains under the chart's conditions with a 10 mV tolerance on the voltage test (round 2).

Purpose.  ref_under_ba.py judged the reference gains by the chart's conditions, among them the voltage
margin bbar <= 8.1 V on the rigorous bound bbar of the forced command.  The authority rule with the
10 % margin (auth10) chooses its gain so that its steady command, sampled on 80 depths with 40
harmonics, reaches 8.1 V at some depth: at 1250, 1375, 1500, 2375 and 2500 rpm the rigorous bound
then exceeds 8.1 V by 0.5-2.3 mV at one grid depth (4 or 6 mm), the upward scan stops there, and the
gain counts as certifiable only up to 2 or 4 mm.  That is a numerical tie, not a property of the rule.
This script repeats the judgement with the voltage test relaxed to bbar <= 8.1 V + TOL, TOL = 10 mV,
everything else unchanged (linear limit, certificate >= beta_req on the reduced collocation order, grid
2:2:32 mm, scan stopped at the first depth not admissible for 1 mm/s), for every reference gain:
the stored scan (ref_under_ba_cache.jsonl) is reused up to its last row and continued, with
basin_aware.bbar_of and basin_aware.certificate, where it stopped only on the strict voltage test.
The chart is not changed: none of its stored scan rows has 8.1 < bbar <= 8.11 V (checked here), so the
tolerance would not alter any of its evaluated gains.  Both the strict and the tolerant depths are
reported, with and without the depth margin a <= 0.9 a_lim (ba_kappa.kappa_rows).

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/ref_under_ba_tol.py [--workers 2]
Writes verification/paper/ref_under_ba_tol.json (cache ref_under_ba_tol_cache.jsonl).
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOL = 0.010                     # V
OUT = os.path.join(HERE, "ref_under_ba_tol.json")
CACHE = os.path.join(HERE, "ref_under_ba_tol_cache.jsonl")
DESIGNS = ("pub", "auth0", "auth10", "auth1500")
BETAS = (1.0, 2.0, 5.0)


def tolerant(rows, vlim):
    out = []
    for r in rows:
        r = dict(r)
        r["volt_strict"] = bool(r["volt"])
        r["volt"] = bool(r["bbar"] <= vlim + TOL)
        out.append(r)
    return out


def needs_continuation(rows, vlim, beta_min):
    last = rows[-1]
    return (not last["volt_strict"]) and last["volt"] and last["lin"] and last.get("beta", 0.0) >= beta_min


def job(args):
    rpm, g, scan = args
    import basin_aware as BA
    BA._setup()
    t0 = time.time()
    rows = tolerant(scan["rows"], BA.VLIM)
    added = 0
    go = needs_continuation(rows, BA.VLIM, min(BA.BETA_REQ))
    while go and rows[-1]["d"] < max(BA.DEPTHS):
        d = rows[-1]["d"] + 2.0
        a = d * 1e-3
        lin = bool(d < scan["a_lim"])
        bb = BA.bbar_of(rpm, g, a)
        row = dict(d=d, lin=lin, bbar=bb, volt_strict=bool(bb <= BA.VLIM), volt=bool(bb <= BA.VLIM + TOL))
        if lin and bb < BA.VMAX:
            row.update(BA.certificate(rpm, g, a, scan["N"], "FAST", bb))
        rows.append(row)
        added += 1
        go = bool(lin and row["volt"] and row.get("beta", 0.0) >= min(BA.BETA_REQ))
    return dict(rpm=rpm, g=g, scan=dict(scan, rows=rows), added_rows=added, seconds=round(time.time() - t0, 1))


def chart_near_ties():
    near = []
    for p in sorted(glob.glob(os.path.join(HERE, "basin_aware_cache", "*", "speed_*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["k"][0] == "scan" and r["v"]:
                for row in r["v"]["rows"]:
                    b = row.get("bbar")
                    if b is not None and 8.1 < b <= 8.1 + TOL:
                        near.append([os.path.basename(os.path.dirname(p)), r["k"][1], r["k"][2], row["d"], b])
    return near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import basin_aware as BA
    from ba_kappa import kappa_rows
    fd = json.load(open(os.path.join(HERE, "fair_designs.json")))["speeds"]
    stored = {}
    for line in open(os.path.join(HERE, "ref_under_ba_cache.jsonl")):
        r = json.loads(line)
        stored[(float(r["rpm"]), round(float(r["g"]), 6))] = r["scan"]
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line)
            cache[(float(r["rpm"]), round(float(r["g"]), 6))] = r
    todo = []
    for k in sorted(fd, key=float):
        for d in DESIGNS:
            key = (float(k), round(float(fd[k][d]["g"]), 6))
            if key in cache or any(t[:2] == key for t in todo):
                continue
            todo.append((key[0], key[1], stored[key]))
    print(f"{len(todo)} (speed, gain) scans to extend", flush=True)
    if todo:
        with Pool(a.workers) as pool, open(CACHE, "a") as fh:
            for r in pool.imap_unordered(job, todo):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                cache[(float(r["rpm"]), round(float(r["g"]), 6))] = r
                if r["added_rows"]:
                    print(f"  {r['rpm']:.0f} rpm g {r['g']:.1f}: {r['added_rows']} rows added ({r['seconds']} s)", flush=True)
    per = {}
    for k in sorted(fd, key=float):
        per[k] = {}
        for d in DESIGNS:
            key = (float(k), round(float(fd[k][d]["g"]), 6))
            sc = cache[key]["scan"]
            strict_rows = [dict(r, volt=r["volt_strict"]) for r in sc["rows"]]
            rec = dict(g=key[1], added_rows=cache[key]["added_rows"])
            for name, rows in (("strict", strict_rows), ("tol", sc["rows"])):
                for kap in (1.0, 0.9):
                    rr = rows if kap == 1.0 else kappa_rows(dict(sc, rows=rows), kap)["rows"]
                    for b in BETAS:
                        ar, mins, fail = BA.evaluate(rr, b * 1e-3)
                        rec[f"{name}_k{kap:g}_b{b:g}"] = dict(a_rob_mm=ar, min_beta_mmps=(mins * 1e3 if np.isfinite(mins) else None))
            per[k][d] = rec
    summ = {}
    for d in DESIGNS:
        s = {}
        for name in ("strict", "tol"):
            for kap in (1.0, 0.9):
                for b in BETAS:
                    key = f"{name}_k{kap:g}_b{b:g}"
                    s[key] = float(np.mean([per[k][d][key]["a_rob_mm"] for k in per]))
        s["speeds_changed"] = [k for k in per if any(per[k][d][f"tol_k1_b{b:g}"]["a_rob_mm"] != per[k][d][f"strict_k1_b{b:g}"]["a_rob_mm"] for b in BETAS)]
        summ[d] = s
    out = dict(note=__doc__.split("\n\n")[0], tol_V=TOL, chart_rows_with_8p1_lt_bbar_le_8p11=chart_near_ties(),
               speeds=per, summary=summ)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for d in DESIGNS:
        s = summ[d]
        print(d, "strict k1", [round(s[f'strict_k1_b{b:g}'], 2) for b in BETAS], "tol k1", [round(s[f'tol_k1_b{b:g}'], 2) for b in BETAS],
              "| strict k0.9", [round(s[f'strict_k0.9_b{b:g}'], 2) for b in BETAS], "tol k0.9", [round(s[f'tol_k0.9_b{b:g}'], 2) for b in BETAS],
              "| changed", s["speeds_changed"])
    print("chart near ties:", len(out["chart_rows_with_8p1_lt_bbar_le_8p11"]))


if __name__ == "__main__":
    main()
