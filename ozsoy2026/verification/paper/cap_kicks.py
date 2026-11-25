"""Certified kicks up to the depth cap: basin-aware chart against the reference gains (review round 2, 4.4).

Purpose.  With a cap of 10 or 16 mm on the depth, the basin-aware chart (ba_cap.json) and the
reference schedules reach the cap at most speeds, so the depths no longer differ; what differs is the
disturbance each design is certified to absorb at every depth up to the cap.  For each cap (10, 16,
32 mm) and speed this script compares
  - the chart: the smallest certified kick over the grid depths up to its capped a_rob
    (ba_cap.json per_cap[cap][rpm][b].min_beta_up_to_a_rob_mmps: default order where re-certified,
    reduced order elsewhere), its depth and gain;
  - each reference gain (fair_designs.json: pub, auth0, auth10, auth1500) under the chart's
    conditions up to the cap, from its reduced-order depth scan (ref_under_ba_cache.jsonl; the scan
    stops at the first depth where the kick falls below 1 mm/s or a condition fails):
      'reaches_cap'   every grid depth up to min(cap, 32) is linearly stable and keeps
                      bbar <= 8.1 V, and the scan reached the cap;
      'kick_below_1'  the scan stopped before the cap on a kick below 1 mm/s with the linear and
                      voltage conditions holding (the smallest kick up to the cap is then below
                      1 mm/s, and is reported as that row's kick, an upper bound on it);
      'fails'         the linear or the voltage condition fails before the cap.
    min_beta_mmps: the smallest reduced-order kick over the grid depths up to the cap (or up to the
    stopping row).
Medians over the speeds; reduced-order kicks overstate default-order ones by at most 7 %.
paired_both_at_cap: the ratio chart / reference of these smallest kicks at the speeds where the chart's
a_rob equals the cap and the reference gain reaches the cap (the same depths on both sides).

Run:  cd ozsoy2026 && python3 verification/paper/cap_kicks.py [--tol]   (no simulation; writes cap_kicks.json,
or with --tol cap_kicks_tol.json from the reference scans of ref_under_ba_tol.py, whose voltage test has a
10 mV tolerance; there the 'volt' flag of every row is the tolerant one)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCANS = "ref_under_ba_cache.jsonl"          # --tol: ref_under_ba_tol_cache.jsonl (voltage test with 10 mV tolerance)
OUT = "cap_kicks.json"
REFS = ("pub", "auth0", "auth10", "auth1500")
CAPS = (10, 16, 32)
BETAS = ("1", "2", "5")


def main():
    bc = json.load(open(os.path.join(HERE, "ba_cap.json")))["per_cap"]
    fd = json.load(open(os.path.join(HERE, "fair_designs.json")))["speeds"]
    scans = {}
    for line in open(os.path.join(HERE, SCANS)):
        r = json.loads(line)
        scans[(float(r["rpm"]), round(float(r["g"]), 6))] = r["scan"]
    out = dict(note=__doc__.split("\n\n")[0], caps={})
    for cap in CAPS:
        per = {}
        for k in sorted(bc[str(cap)], key=float):
            n = float(k)
            rec = dict(ba={b: dict(g=bc[str(cap)][k][b]["g"], a_rob_mm=bc[str(cap)][k][b]["a_rob_mm"],
                                   min_beta_mmps=bc[str(cap)][k][b].get("min_beta_up_to_a_rob_mmps"))
                           for b in BETAS})
            for d in REFS:
                g = float(fd[k][d]["g"])
                rows = scans[(n, round(g, 6))]["rows"]
                kicks, status = [], "reaches_cap"
                last = 0.0
                for r in rows:
                    if r["d"] > cap:
                        break
                    if not (r["lin"] and r["volt"]):
                        status = "fails"
                        break
                    kicks.append(float(r.get("beta", 0.0)) * 1e3)
                    last = r["d"]
                if status == "reaches_cap" and last < min(cap, 32.0):
                    status = "kick_below_1" if kicks and kicks[-1] < 1.0 else "fails"
                rec[d] = dict(g=g, status=status, min_beta_mmps=min(kicks) if kicks else 0.0, last_depth_mm=last)
            per[k] = rec
        summ = {}
        for b in BETAS:
            v = [per[k]["ba"][b]["min_beta_mmps"] for k in per]
            summ[f"ba_{b}"] = dict(median_min_beta_mmps=float(np.median(v)), min_min_beta_mmps=float(min(v)),
                                   mean_a_rob_mm=float(np.mean([per[k]["ba"][b]["a_rob_mm"] for k in per])))
        for d in REFS:
            st = [per[k][d]["status"] for k in per]
            v = [per[k][d]["min_beta_mmps"] for k in per]
            vc = [per[k][d]["min_beta_mmps"] for k in per if per[k][d]["status"] == "reaches_cap"]
            summ[d] = dict(n_reaches_cap=st.count("reaches_cap"), n_kick_below_1=st.count("kick_below_1"),
                           n_fails=st.count("fails"), median_min_beta_mmps_all=float(np.median(v)),
                           median_min_beta_mmps_reaching=float(np.median(vc)) if vc else None)
        paired = {}
        for d in REFS:
            for b in BETAS:
                ks = [k for k in per if per[k][d]["status"] == "reaches_cap" and per[k]["ba"][b]["a_rob_mm"] >= cap - 1e-9]
                r = [per[k]["ba"][b]["min_beta_mmps"] / per[k][d]["min_beta_mmps"] for k in ks]
                paired[f"{d}_vs_ba_{b}"] = dict(
                    n=len(ks), speeds=ks,
                    median_ratio_ba_over_ref=float(np.median(r)) if r else None,
                    min_ratio=float(min(r)) if r else None, max_ratio=float(max(r)) if r else None,
                    n_ba_at_least_ref=int(sum(x >= 0.999 for x in r)),
                    median_ba_mmps=float(np.median([per[k]["ba"][b]["min_beta_mmps"] for k in ks])) if ks else None,
                    median_ref_mmps=float(np.median([per[k][d]["min_beta_mmps"] for k in ks])) if ks else None)
        out["caps"][str(cap)] = dict(per_speed=per, summary=summ, paired_both_at_cap=paired)
    out["scans"] = SCANS
    with open(os.path.join(HERE, OUT), "w") as fh:
        json.dump(out, fh, indent=1)
    for cap in CAPS:
        print(cap, json.dumps(out["caps"][str(cap)]["summary"]))
        print(cap, json.dumps({k: {x: v[x] for x in ("n", "median_ratio_ba_over_ref", "min_ratio", "max_ratio", "n_ba_at_least_ref")}
                               for k, v in out["caps"][str(cap)]["paired_both_at_cap"].items()}))


if __name__ == "__main__":
    import sys
    if "--tol" in sys.argv:
        SCANS, OUT = "ref_under_ba_tol_cache.jsonl", "cap_kicks_tol.json"
    main()
