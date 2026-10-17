"""Check 1: integration accuracy and reported maxima of src/tdsim.c.

For a few (rpm, a, g) cases the C simulator (RK4, nsub sub-steps) is compared
with (i) src/tdsim.simulate_exact_python (exact ZOH, same author) and (ii) the
independent exact-ZOH reference ref_sim.ref_linear (tf2ss realisation).  The
summary maxima returned by the C code are compared with maxima recomputed from
the stored histories.  Writes check1_integration.csv / .json.
"""
import json
import warnings
import common
import numpy as np
import ref_sim as R
import tdsim
import zoa

warnings.filterwarnings("ignore", module="scipy.signal")

CASES = [  # rpm, a [m], g [V s/m]
    (500, 1.5e-3, 0.0),
    (500, 3.0e-3, 0.0),
    (1300, 5.0e-3, 600.0),
    (2600, 2.0e-3, 0.0),
    (3000, 10.0e-3, 1200.0),
    (1950, 30.0e-3, 1200.0),
    (800, 14.0e-3, 1200.0),
]
NREV = 20


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


rows = []
for rpm, a, g in CASES:
    ref = R.ref_linear(rpm, a, g, nrev=NREV)
    exw, exv = tdsim.simulate_exact_python(rpm, a, g, nrev=NREV)
    row = dict(rpm=rpm, a_mm=a * 1e3, g=g, a_zoa_mm=float(zoa.a_lim(rpm, g) * 1e3))
    row["exactpy_vs_ref_x"] = rel(exw, ref["w"])
    row["exactpy_vs_ref_v"] = rel(exv, ref["v"])
    for nsub in (1, 2, 4, 8):
        c = tdsim.simulate(rpm, a, g, store=True, nrev=NREV, nsub=nsub)
        row[f"rk4_nsub{nsub}_x"] = rel(c["x"], ref["w"])
        if nsub == 2:
            row["rk4_nsub2_v"] = rel(c["v"], ref["v"])
            row["rk4_nsub2_fact"] = rel(c["fact"], ref["fact"])
            row["rk4_nsub2_fcut"] = rel(c["fcut"][:-1], ref["fcut"][:-1])
            # reported maxima vs histories (C loop records samples 0..nsteps-1)
            n = len(c["x"]) - 1
            row["max_demand_C"] = c["max_demand"]
            row["max_demand_hist"] = float(np.max(np.abs(g * c["v"][:n])))
            row["max_demand_ref"] = float(np.max(np.abs(g * ref["v"][:n])))
            row["max_fact_C"] = c["max_fact"]
            row["max_fact_hist"] = float(np.max(np.abs(c["fact"][:n])))
            row["max_fact_ref"] = float(np.max(np.abs(ref["fact"][:n])))
            row["max_x_C_um"] = c["max_x"] * 1e6
            row["max_x_ref_um"] = float(np.max(np.abs(ref["w"][:n])) * 1e6)
            row["max_v_C"] = c["max_v"]
            # does the tool leave the cut (nonlinearity active)?
            row["diverged"] = c["diverged"]
    rows.append(row)
    print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in row.items()})

keys = list(rows[0].keys())
with open(common.out("check1_integration.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r[k]) for k in keys) + "\n")

summary = dict(
    max_rel_err_exactpy_vs_independent_ref=max(r["exactpy_vs_ref_x"] for r in rows),
    max_rel_err_rk4_nsub2_x=max(r["rk4_nsub2_x"] for r in rows),
    max_rel_err_rk4_nsub2_fact=max(r["rk4_nsub2_fact"] for r in rows),
    convergence_ratio_nsub1_over_nsub2=[r["rk4_nsub1_x"] / r["rk4_nsub2_x"] for r in rows],
    max_rel_err_reported_max_demand_vs_ref=max(
        abs(r["max_demand_C"] - r["max_demand_ref"]) / max(r["max_demand_ref"], 1e-300)
        for r in rows if r["g"] > 0),
    max_rel_err_reported_max_fact_vs_ref=max(
        abs(r["max_fact_C"] - r["max_fact_ref"]) / max(r["max_fact_ref"], 1e-300)
        for r in rows if r["g"] > 0),
    reported_equals_history=all(
        np.isclose(r["max_demand_C"], r["max_demand_hist"], rtol=1e-12)
        and np.isclose(r["max_fact_C"], r["max_fact_hist"], rtol=1e-12) for r in rows),
)
print(json.dumps(summary, indent=1))
with open(common.out("check1_integration.json"), "w") as f:
    json.dump(dict(nrev=NREV, cases=rows, summary=summary), f, indent=1, default=float)
