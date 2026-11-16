"""Check of the final simulator scheme (fmode=1), added after the force fix.

Checks 1-6 ran before src/tdsim.c re-evaluated the cutting force at every RK4
stage, so they validate the held-force scheme (fmode=0) only.  This script
covers the scheme used for all results:
  (a) trajectories at 512 steps/rev, fmode 0 and 1, against a converged
      reference: the exact zero-order-hold solution (simulate_exact_python,
      force held over each step) at 8192 and 16384 steps/rev, extrapolated
      to zero step (Richardson, 2 x[16384] - x[8192]; its error is first
      order in the step).  The un-extrapolated 8192-step solution, used as
      the reference before the audit, is itself 0.1-0.6 % off;
  (b) peak demand at a_lim, 512 vs 8192 steps/rev;
  (c) time-domain stability limit (src.tdboundary, D1 detector) against ZOA.
Writes check7_final_scheme.json.
"""
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from tdsim import simulate, simulate_exact_python   # noqa: E402
from tdboundary import boundary                      # noqa: E402
from zoa import a_lim                                # noqa: E402

CASES = [(500, 1.5e-3, 0.0), (1300, 5e-3, 600.0), (2600, 2e-3, 0.0),
         (3000, 10e-3, 1200.0), (1950, 30e-3, 1200.0)]
NREV = 10
out = {"trajectory": [], "demand": [], "boundary": []}
t0 = time.time()
for rpm, a, g in CASES:
    x8, _ = simulate_exact_python(rpm, a, g, nrev=NREV, spr=8192)
    x16, _ = simulate_exact_python(rpm, a, g, nrev=NREV, spr=16384)
    x8, x16 = x8[::16], x16[::32]
    xr = 2 * x16 - x8
    scale = np.abs(xr).max()
    row = dict(rpm=rpm, a_mm=a * 1e3, g=g,
               rel_err_zoh8192=float(np.abs(x8 - xr).max() / scale),
               rel_err_zoh16384=float(np.abs(x16 - xr).max() / scale))
    for fm in (0, 1):
        x = simulate(rpm, a, g, nrev=NREV, spr=512, fmode=fm, store=True)["x"]
        row[f"rel_err_fmode{fm}"] = float(np.abs(x - xr).max() / scale)
    x = simulate(rpm, a, g, nrev=NREV, spr=16384, fmode=1, store=True)["x"][::32]
    row["rel_err_fmode1_16384"] = float(np.abs(x - xr).max() / scale)
    out["trajectory"].append(row)
    print(row, f"{time.time() - t0:.0f}s", flush=True)
for rpm, g in [(500, 600.0), (1000, 630.0), (1300, 991.0), (2500, 535.0)]:
    al = a_lim(rpm, g)
    d512 = simulate(rpm, al, g, spr=512)["max_demand"]
    d8192 = simulate(rpm, al, g, spr=8192)["max_demand"]
    out["demand"].append(dict(rpm=rpm, g=g, a_mm=al * 1e3, demand_512=3 * d512,
                              demand_8192=3 * d8192, rel=float(d512 / d8192 - 1)))
    print(out["demand"][-1], flush=True)
for rpm, g in [(500, 0.0), (500, 1200.0), (1000, 630.0), (1300, 991.0), (1600, 0.0),
               (2500, 0.0), (2500, 535.0), (3000, 712.0)]:
    z = a_lim(rpm, g)
    b = boundary(rpm, g)
    out["boundary"].append(dict(rpm=rpm, g=g, zoa_mm=z * 1e3, td_mm=b * 1e3,
                                rel=float(b / z - 1)))
    print(out["boundary"][-1], flush=True)
with open(os.path.join(HERE, "check7_final_scheme.json"), "w") as fh:
    json.dump(out, fh, indent=1)
