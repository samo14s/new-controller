"""Convergence of the Eq. (2) optimum gain with the time step of the feasibility run.

Definition fixed to kind='force' (3 g|xdot|, all 50 revolutions, depth a_lim(ZOA));
only steps per revolution change.  The tdsim verification showed that the force
held over each step lags by half a step, so at 512 steps/rev the run at
a_lim(ZOA) is already slightly unstable below ~2000 rpm.  Writes
gainmap_spr_convergence.{csv,json}.
"""
import csv
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P                 # noqa: E402
from zoa import a_lim, _gcl_grid   # noqa: E402
from tdsim import simulate         # noqa: E402

SPEEDS = [500, 740, 800, 1000, 1300, 1940, 2500]
SPRS = [512, 1024, 2048, 4096, 8192]
GS = np.arange(0.0, 3000.0 + 1e-9, 50.0)
fig8 = np.loadtxt(os.path.join(HERE, "..", "targets", "fig8_gain_map.csv"), delimiter=",", skiprows=1)
_gcl = {}


def alim(n, g):
    if g not in _gcl:
        _gcl[g] = _gcl_grid(g)
    return a_lim(n, g, gcl=_gcl[g])


def fit(n, g, spr):
    a = alim(n, g)
    if not np.isfinite(a):
        return 0.0, np.inf
    if g == 0:
        return a, 0.0
    r = simulate(n, a, g, spr=spr)
    d = np.inf if r["diverged"] else P.GA_GAIN * r["max_demand"]
    return a, d


rows = []
for n in SPEEDS:
    g8 = float(np.interp(n, fig8[:, 0], fig8[:, 1]))
    for spr in SPRS:
        res = [fit(n, g, spr) for g in GS]
        f = np.array([a if d <= P.F_MAX else 0.0 for a, d in res])
        j = int(np.argmax(f))
        gb, ab = GS[j], f[j]
        if j + 1 < len(GS) and f[j + 1] == 0:
            lo, hi = GS[j], GS[j + 1]
            while hi - lo > 1.0:
                mid = 0.5 * (lo + hi)
                a, d = fit(n, mid, spr)
                if d <= P.F_MAX:
                    lo = mid
                    if a > ab:
                        gb, ab = mid, a
                else:
                    hi = mid
        # demand at the Fig. 8 gain itself
        a8, d8 = fit(n, g8, spr)
        rows.append(dict(rpm=n, spr=spr, g_opt=round(float(gb), 1), a_opt_mm=round(float(ab) * 1e3, 3),
                         g_fig8=round(g8, 1), ratio=round(float(gb) / g8, 3),
                         demand_at_g_fig8_N=round(float(d8), 2), a_zoa_at_g_fig8_mm=round(a8 * 1e3, 3)))
        print(rows[-1], flush=True)
with open(os.path.join(HERE, "gainmap_spr_convergence.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(HERE, "gainmap_spr_convergence.json"), "w") as fh:
    json.dump(dict(kind="force", window="full 50 rev", zeta=P.ZETA, rows=rows), fh, indent=1)
