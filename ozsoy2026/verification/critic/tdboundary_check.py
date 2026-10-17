"""Spot-check src/tdboundary.py (used for Table 3 and the simulated Figs. 10/12 cuts).

1. boundary(rpm, g) at 512 steps/rev against the decay-based (D1, 1000 rev)
   boundaries of verification/tdsim/check3_stability.csv, same simulator.
2. With the 9 V command limit used by run_table3.py: is chatter monotone in
   depth (one stable->chatter switch, as bisection assumes)?  Scan the depth
   at fixed gain and record classify() along the scan.
Writes tdboundary_check.json.
"""
import csv
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P                         # noqa: E402
from tdboundary import boundary, classify  # noqa: E402
from zoa import a_lim                      # noqa: E402

out = {"part1": [], "part2": []}
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "tdsim", "check3_stability.csv"))))
for r in rows:
    n, g = float(r["rpm"]), float(r["g"])
    ref = float(r["a_td512_n1000"])
    b = boundary(n, g)
    out["part1"].append(dict(rpm=n, g=g, a_zoa_mm=float(r["a_zoa"]) * 1e3, a_D1_1000rev_mm=ref * 1e3,
                             a_tdboundary_mm=b * 1e3, rel=(b / ref - 1)))
    print(out["part1"][-1], flush=True)
rel = np.array([x["rel"] for x in out["part1"]])
out["part1_summary"] = dict(n=int(rel.size), min_rel=float(rel.min()), max_rel=float(rel.max()),
                            median_rel=float(np.median(rel)))
print(out["part1_summary"], flush=True)

VMAX = P.F_MAX / P.GA_GAIN
for n, g in [(500, 1479.0), (500, 3000.0), (1300, 1158.0), (2500, 543.0), (2500, 3000.0), (3000, 3000.0)]:
    az = a_lim(n, g)
    depths = np.linspace(0.3, 1.6, 27) * az
    seq = []
    for a in depths:
        c = classify(n, a, g, vmax=VMAX)
        seq.append(int(c["chatter"]))
    switches = int(np.sum(np.abs(np.diff(seq))))
    b = boundary(n, g, vmax=VMAX)
    out["part2"].append(dict(rpm=n, g=g, a_zoa_unclipped_mm=az * 1e3, depths_mm=list(np.round(depths * 1e3, 3)),
                             chatter=seq, n_switches=switches, boundary_vmax_mm=b * 1e3))
    print(n, g, round(az * 1e3, 2), seq, switches, round(b * 1e3, 3), flush=True)
with open(os.path.join(HERE, "tdboundary_check.json"), "w") as fh:
    json.dump(out, fh, indent=1)
