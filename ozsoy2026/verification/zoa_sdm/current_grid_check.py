"""The ZOA accuracy figures of this folder with the CURRENT src/zoa.py.

dense_rootcheck and compare_12speeds were computed before commit a8fb532,
which also coarsened the frequency grid on 60-600 Hz (0.01 -> 0.02 Hz).
This recomputes the comparisons against the stored independent lobe sweep
and mean-coefficient SDM values with the current solver.
    python current_grid_check.py  ->  current_grid_check.json
"""
import csv
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P   # noqa: E402
import zoa           # noqa: E402

out = {"f_grid_step_60_600_hz": float(np.diff(zoa.F_GRID[1200:1202])[0])}
d = json.load(open(os.path.join(HERE, "dense_sld_z0134.json")))
rp = np.array(d["rpm"])
P.set_damping(0.0134)
for g in ("0", "1500"):
    lob = np.array(d[g]["a_lobes_mm"])
    rel = zoa.sld(rp, float(g)) * 1e3 / lob - 1
    out[f"dense_g{g}"] = dict(max_abs_rel=float(np.abs(rel).max()), median_abs_rel=float(np.median(np.abs(rel))),
                              n_above_1e3=int(np.sum(np.abs(rel) > 1e-3)),
                              stored_pre_fix_max_abs_rel=float(np.abs(np.array(d[g]["a_src_mm"]) / lob - 1).max()))
mx_l = mx_s = 0.0
rows = list(csv.DictReader(open(os.path.join(HERE, "compare_12speeds.csv"))))
for z in (0.0134, 0.0145):
    P.set_damping(z)
    for r in rows:
        if float(r["zeta"]) != z:
            continue
        a = zoa.a_lim(float(r["rpm"]), float(r["g"])) * 1e3
        mx_l = max(mx_l, abs(a / float(r["a_zoa_lobes_mm"]) - 1))
        mx_s = max(mx_s, abs(a / float(r["a_sdm_meanH_mm"]) - 1))
P.set_damping(P.ZETA_PAPER_CURVES)
out["table_96"] = dict(max_abs_rel_vs_lobes=mx_l, max_abs_rel_vs_sdm_mean=mx_s)
json.dump(out, open(os.path.join(HERE, "current_grid_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
