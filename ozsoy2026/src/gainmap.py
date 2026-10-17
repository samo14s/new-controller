"""Per-speed gain optimisation (Sec. 2.1, Steps 1-5) and a deterministic reference.

For each spindle speed two answers are produced:
  * SADE  - the paper's optimiser on Eq. (2), several seeds;
  * REF   - a dense scan of g over the search interval followed by bisection on
            the feasibility edge.  It needs no tuning and shows whether SADE
            reached the true optimum of the same objective.
"""
import numpy as np
import params as P
from objective import fitness
from sade import sade

G_LO, G_HI = 0.0, 3000.0      # search interval G (V s/m); the paper does not state it
DG = 10.0                     # reference scan step


def reference(rpm, kind="force", g_lo=G_LO, g_hi=G_HI, dg=DG, tol=0.25):
    gs = np.arange(g_lo, g_hi + 0.5 * dg, dg)
    rows = np.array([fitness(rpm, g, kind, detail=True) for g in gs])
    f, al, fd = rows[:, 0], rows[:, 1] * 1e3, rows[:, 2]
    j = int(np.argmax(f))
    g_best, f_best = gs[j], f[j]
    # refine on the feasibility edge just above the best grid point
    if j + 1 < len(gs) and f[j + 1] == 0.0 and f_best > 0:
        lo, hi = gs[j], gs[j + 1]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            fm = fitness(rpm, mid, kind)
            if fm > 0:
                lo = mid
                if fm > f_best:
                    g_best, f_best = mid, fm
            else:
                hi = mid
    return dict(g=g_best, a=f_best, scan_g=gs, scan_f=f, scan_alim=al, scan_fd=fd)


def run_sade(rpm, kind="force", seeds=(1, 2, 3), np_=20, max_gen=50, lp=5,
             window=12, tol=1e-4, g_lo=G_LO, g_hi=G_HI):
    out = []
    for s in seeds:
        # stratified uniform initial population over G (Step 1)
        rng = np.random.default_rng(1000 + s)
        init = g_lo + (np.arange(np_) + rng.random(np_)) / np_ * (g_hi - g_lo)
        r = sade(lambda x: fitness(rpm, x[0], kind), g_lo, g_hi, np_=np_,
                 max_gen=max_gen, lp=lp, window=window, tol=tol, seed=s,
                 init=init[:, None])
        out.append((float(r["x"][0]), r["f"], r["n_eval"], r["gens"]))
    return out
