"""SDM convergence in the number of intervals m (straight flute, full actuator)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import numpy as np
import sdm

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for rpm in (500, 1300, 2200, 3000):
    for g in (0, 1500):
        ref = None
        for m in (20, 40, 80, 160, 320):
            a, mu, _ = sdm.boundary(rpm, g, 0.0134, m=m, rtol=1e-6)
            rows.append(dict(rpm=rpm, g=g, m=m, a_mm=a * 1e3))
            print(rpm, g, m, f"{a*1e3:.6f}", flush=True)
for r in rows:
    best = [q for q in rows if q["rpm"] == r["rpm"] and q["g"] == r["g"] and q["m"] == 320][0]
    r["rel_to_m320"] = r["a_mm"] / best["a_mm"] - 1
with open(os.path.join(HERE, "sdm_convergence.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("default m:", {n: sdm.default_m(n) for n in (500, 1300, 2200, 3000)})
