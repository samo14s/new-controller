"""Check 5: what the feasibility check (objective.py, Step 2) actually simulates.

The SADE fitness runs tdsim at a = a_lim(ZOA) for 50 revolutions with 512
steps per revolution.  Check 3 shows that this discretisation moves the
time-domain stability limit below a_lim at low speeds, so the run at a_lim can
be a (slowly growing or saturated) chatter run whose peak demand depends on
nrev and spr.  This script quantifies that for a few speeds and gains,
including the paper's worked example (1300 rpm, g = 1240 V s/m: a_lim = 11.8 mm,
max |F_d| = 29 N).  Demands: 'volt' = max |g xdot| (V), 'force' = 3 x volt (N),
'actual' = max |f_act| (N).  Writes check5_demand_at_alim.csv / .json.
"""
import json
import common
import numpy as np
import params as P
import tdsim
import zoa
import detector as Dt

CASES = [(1300, 1240.0), (500, 600.0), (800, 1200.0), (1300, 600.0), (2000, 600.0),
         (2600, 1200.0), (3000, 600.0)]
rows = []
for rpm, g in CASES:
    al = float(zoa.a_lim(rpm, g))
    row = dict(rpm=rpm, g=g, a_lim_mm=al * 1e3)
    for spr, nrev in ((512, 50), (512, 300), (2048, 50), (2048, 300)):
        r = tdsim.simulate(rpm, al, g, store=True, nrev=nrev, spr=spr)
        c = Dt.classify(r["x"], spr, r["diverged"])
        tag = f"spr{spr}_n{nrev}"
        row[f"volt_{tag}"] = r["max_demand"]
        row[f"force_{tag}"] = P.GA_GAIN * r["max_demand"]
        row[f"actual_{tag}"] = r["max_fact"]
        row[f"max_x_um_{tag}"] = r["max_x"] * 1e6
        if nrev == 300:
            row[f"D1_stable_{tag}"] = c["stable_decay"]
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
          flush=True)

keys = list(rows[0].keys())
with open(common.out("check5_demand_at_alim.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r[k]) for k in keys) + "\n")
with open(common.out("check5_demand_at_alim.json"), "w") as f:
    json.dump(rows, f, indent=1, default=float)
