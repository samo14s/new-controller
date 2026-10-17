"""Brute-force reference optimum of fitness(rpm, g, 'force') over g in [0, 3000].

1. grid step 5 V s/m over [0, 3000];
2. step 1 V s/m over [0, last feasible grid point + 10] (fragmentation check);
3. step 0.1 V s/m in +-5 V s/m around the three best local maxima;
4. bisection (to 0.01 V s/m) on any feasibility edge next to the best point.
Writes bruteforce_<rpm>.csv (all evaluated points) and bruteforce.json.
"""
import time
from multiprocessing import Pool
import numpy as np
from common import CachedFitness, RPMS, G_LO, G_HI, QSTEP, dump_json, HERE
import os


def local_maxima(g, f, k=3):
    idx = [i for i in range(len(f)) if f[i] > 0 and
           (i == 0 or f[i] >= f[i - 1]) and (i == len(f) - 1 or f[i] >= f[i + 1])]
    idx.sort(key=lambda i: -f[i])
    return idx[:k]


def run(rpm):
    t0 = time.time()
    fit = CachedFitness(rpm)
    g5 = np.arange(G_LO, G_HI + 1e-9, 5.0)
    f5 = np.array([fit(g) for g in g5])
    feas5 = g5[f5 > 0]
    gmax_feas = float(feas5.max()) if feas5.size else 0.0
    g1 = np.arange(G_LO, min(G_HI, gmax_feas + 10) + 1e-9, 1.0)
    f1 = np.array([fit(g) for g in g1])
    # fragmentation: number of feasible intervals on the 1 V s/m grid
    feas1 = f1 > 0
    n_islands = int(np.sum(np.diff(feas1.astype(int)) == 1) + (1 if feas1[0] else 0))
    cand = [g1[i] for i in local_maxima(g1, f1, 3)] + [g5[i] for i in local_maxima(g5, f5, 3)]
    best_g, best_f = None, -1.0
    for gc in cand:
        gf = np.arange(max(G_LO, gc - 5), min(G_HI, gc + 5) + 1e-9, 0.1)
        ff = np.array([fit(g) for g in gf])
        j = int(np.argmax(ff))
        if ff[j] > best_f:
            best_g, best_f = float(gf[j]), float(ff[j])
            # bisection on edges next to the best fine point
            for nb in (j + 1, j - 1):
                if 0 <= nb < len(gf) and ff[nb] == 0.0:
                    lo, hi = gf[j], gf[nb]
                    while abs(hi - lo) > 2 * QSTEP:
                        mid = 0.5 * (lo + hi)
                        fm = fit(mid)
                        if fm > 0:
                            lo = mid
                            if fm > best_f:
                                best_g, best_f = float(mid), float(fm)
                        else:
                            hi = mid
    fit.save()
    keys = sorted(fit.cache)
    rows = np.array([(k * QSTEP,) + fit.cache[k] for k in keys])
    rows[:, 2] *= 1e3
    np.savetxt(os.path.join(HERE, f"bruteforce_{rpm}.csv"), rows, delimiter=",",
               header="g_Vs_per_m,fitness_mm,alim_mm,Fd_N", comments="", fmt="%.6g")
    best5 = int(np.argmax(f5))
    return dict(rpm=rpm, g_opt=best_g, a_opt=best_f,
                g_opt_grid5=float(g5[best5]), a_opt_grid5=float(f5[best5]),
                a_unc=float(fit(0.0)), feasible_frac_grid5=float(np.mean(f5 > 0)),
                gmax_feasible_grid5=gmax_feas, n_feasible_intervals_grid1=n_islands,
                feasible_frac_of_G=float(np.sum(feas1) * 1.0 / (G_HI - G_LO)),
                n_eval=fit.n_new, seconds=time.time() - t0)


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(run, RPMS)
    for r in res:
        print(r)
    dump_json(res, "bruteforce.json")
