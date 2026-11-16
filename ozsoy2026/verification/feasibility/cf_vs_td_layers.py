"""Closed-form settled demand (src/feasibility.py) against the simulator's, by
the number of axial layers (audit check of README 8.2).

The certificate runs (30 layers) give a settled demand up to 8 % above the
closed form at low speed.  The closed form is exact for the tooth-periodic
orbit; the gap is the simulator's midpoint quadrature over the depth, and it
closes as the number of layers grows.  Cases: speeds with the largest gaps in
the stored certificate runs, at the gain g_best of feas_g1500 and the
operating depth 0.9 a_lim, without the voltage limit (the limit changes
nothing in the settled tail there).  Part 2 compares the Eq. (2) demand of the
reproduction (the peak over 50 revolutions at a_lim, set by the entry) with
30 and 240 layers.  Writes cf_vs_td_layers.json (settled, entry_peak).
Result: settled +3.7 to +8.7 % with 30 layers, within 0.8 % with 240;
entry peak -0.03 to +0.85 %.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P                       # noqa: E402
import feasibility as F                  # noqa: E402
from tdsim import simulate               # noqa: E402
from zoa import a_lim                    # noqa: E402

RPM = (520.0, 660.0, 1900.0)
LAYERS = (30, 60, 120, 240)
NREV, TAIL = 150, 20

d = np.load(os.path.join(HERE, "..", "..", "results", "feas_g1500.npz"))
out = []
for rpm in RPM:
    g = float(np.interp(rpm, d["rpm"], d["g_best"]))
    a = F.KAPPA * a_lim(rpm, g)
    cf = float(F.forced_demand(rpm, g, [a])[0])
    row = dict(rpm=rpm, g=g, a_mm=a * 1e3, demand_cf=cf, td={})
    for L in LAYERS:
        r = simulate(rpm, a, g, store=True, nrev=NREV, nlay=L, nreg=F.NREG)
        spr = r["spr"]
        td = float(P.GA_GAIN * g * np.abs(r["v"][(NREV - TAIL) * spr:]).max())
        row["td"][L] = dict(demand=td, rel=td / cf - 1)
    out.append(row)
    print(rpm, round(g, 1), round(cf, 4), {L: round(v["rel"] * 100, 3) for L, v in row["td"].items()}, flush=True)

# Part 2: the Eq. (2) demand of the reproduction (peak over a 50-revolution run
# at a_lim, dominated by the entry transient) with 30 and 240 layers, at the
# gain of the reproduction's gain map.
gm = np.load(os.path.join(HERE, "..", "..", "results", "gainmap_main.npz"))
peak = []
for rpm in (520.0, 1300.0, 1940.0, 2500.0):
    g = float(np.interp(rpm, gm["rpm"], gm["g_best"]))
    a = a_lim(rpm, g)
    row = dict(rpm=rpm, g=g, a_mm=a * 1e3)
    for L in (30, 240):
        r = simulate(rpm, a, g, nlay=L)
        row[f"demand_{L}"] = np.inf if r["diverged"] else float(P.GA_GAIN * r["max_demand"])     # N, as objective.demand
    row["rel"] = row["demand_30"] / row["demand_240"] - 1
    peak.append(row)
    print("entry peak", rpm, round(g, 1), round(row["demand_30"], 3), round(row["demand_240"], 3), f"{row['rel'] * 100:+.2f} %", flush=True)
json.dump(dict(settled=out, entry_peak=peak), open(os.path.join(HERE, "cf_vs_td_layers.json"), "w"), indent=1)
