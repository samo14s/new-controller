"""src/sade.py versus scipy differential_evolution, a textbook SaDE and brute force.

Every optimiser gets the same evaluation budget NP*(G+1).  20 seeds per case.
Functions are written as minimisation problems; sade() maximises -f.
Success: f_best - f* <= tol (smooth functions) or f_best >= 0.99 f*_max (cliff
functions, which are maximisation problems like Eq. (2)).
Writes benchmarks.csv and benchmarks.json.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from common import dump_json, HERE
from sade import sade
from ref_sade import ref_sade
import os

SEEDS = range(1, 21)


# ---------- test functions (minimisation form, global minimum f*) ----------
def rastrigin(x):
    x = np.atleast_1d(x)
    return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    x = np.atleast_1d(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def schwefel(x):
    x = np.atleast_1d(x)
    return 418.9828872724338 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def gramacy_lee(x):
    x = float(np.atleast_1d(x)[0])
    return np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4


def shubert(x):
    x = float(np.atleast_1d(x)[0])
    i = np.arange(1, 6)
    return float(np.sum(i * np.cos((i + 1) * x + i)))


# cliff-type maximisation problems that mimic Eq. (2) (returned negated)
def cliff_edge(x):            # increasing, feasible on [0, 205.55] of [0, 3000]
    x = float(np.atleast_1d(x)[0])
    return -(3.49 + 0.0275 * x) if x <= 205.55 else 0.0


def narrow_low(x):            # feasible only on [0, 67.5] (2.3 % of the box)
    x = float(np.atleast_1d(x)[0])
    return -(29.16 + 0.005 * x) if x <= 67.49 else 0.0


def islands(x):               # fragmented feasibility, best at 1580 (edge)
    x = float(np.atleast_1d(x)[0])
    feas = (x <= 1600) and ((x % 100) <= 80)
    return -(1 + x / 200) if feas else 0.0


def fstar_num(f, lo, hi):
    g = np.linspace(lo, hi, 200001)
    v = np.array([f(t) for t in g])
    j = int(np.argmin(v))
    r = minimize_scalar(f, bounds=(max(lo, g[j] - 1e-4 * (hi - lo)),
                                   min(hi, g[j] + 1e-4 * (hi - lo))),
                        method="bounded", options=dict(xatol=1e-12))
    return min(v[j], r.fun)


CASES = [
    # name, f, d, lo, hi, f*, kind, NP, G
    ("rastrigin1d", rastrigin, 1, -5.12, 5.12, 0.0, "abs", 20, 60),
    ("schwefel1d", schwefel, 1, -500, 500, None, "abs", 20, 60),
    ("gramacy_lee", gramacy_lee, 1, 0.5, 2.5, None, "abs", 20, 60),
    ("shubert1d", shubert, 1, -10, 10, None, "abs", 20, 60),
    ("cliff_edge", cliff_edge, 1, 0, 3000, -(3.49 + 0.0275 * 205.55), "rel", 20, 60),
    ("narrow_low", narrow_low, 1, 0, 3000, -(29.16 + 0.005 * 67.49), "rel", 20, 60),
    ("islands", islands, 1, 0, 3000, -(1 + 1580 / 200), "rel", 20, 60),
    ("rastrigin2d", rastrigin, 2, -5.12, 5.12, 0.0, "abs", 20, 300),
    ("rosenbrock2d", rosenbrock, 2, -5, 10, 0.0, "abs", 20, 300),
    ("rastrigin5d", rastrigin, 5, -5.12, 5.12, 0.0, "abs", 50, 600),
    ("rosenbrock5d", rosenbrock, 5, -5, 10, 0.0, "abs", 50, 600),
]
TOL_ABS = 1e-4


def success(fb, fs, kind):
    if kind == "abs":
        return fb - fs <= TOL_ABS
    return fb <= 0.99 * fs          # fs < 0 (negated maximum)


def brute(f, d, lo, hi, budget):
    n = int(np.floor(budget ** (1.0 / d)))
    h = (hi - lo) / n
    ax = lo + (np.arange(n) + 0.5) * h          # cell-centred grid
    if d == 1:
        v = np.array([f(t) for t in ax])
        j = int(np.argmin(v))
        # refine in the best cell with a bounded scalar search (counted)
        r = minimize_scalar(f, bounds=(ax[j] - h, ax[j] + h), method="bounded",
                            options=dict(xatol=1e-10, maxiter=60))
        return float(v[j]), float(min(v[j], r.fun)), n
    mesh = np.stack(np.meshgrid(*[ax] * d, indexing="ij"), -1).reshape(-1, d)
    v = np.array([f(t) for t in mesh])
    return float(v.min()), float(v.min()), n**d


def main():
    t0 = time.time()
    rows, out = [], []
    for name, f, d, lo, hi, fs, kind, NP, G in CASES:
        if fs is None:
            fs = fstar_num(f, lo, hi)
        lo_v, hi_v = np.full(d, lo), np.full(d, hi)
        budget = NP * (G + 1)
        res = {}
        # src/sade.py: fixed budget (window disabled), and with window=15
        for tag, win in (("sade_src", G + 1), ("sade_src_w15", 15)):
            fb, ne = [], []
            for s in SEEDS:
                r = sade(lambda x: -f(x), lo_v, hi_v, np_=NP, max_gen=G, lp=5,
                         window=win, tol=1e-8, seed=s)
                fb.append(-r["f"])
                ne.append(r["n_eval"])
            res[tag] = (np.array(fb), np.array(ne))
        for tag, lp in (("sade_ref_lp5", 5), ("sade_ref_lp50", 50)):
            fb, ne = [], []
            for s in SEEDS:
                r = ref_sade(f, lo_v, hi_v, np_=NP, max_gen=G, lp=lp, seed=s)
                fb.append(r["f"])
                ne.append(r["n_eval"])
            res[tag] = (np.array(fb), np.array(ne))
        fb, ne = [], []
        for s in SEEDS:
            r = differential_evolution(f, list(zip(lo_v, hi_v)), popsize=max(1, NP // d),
                                       maxiter=G, tol=0, atol=0, polish=False, seed=s)
            fb.append(r.fun)
            ne.append(r.nfev)
        res["scipy_de"] = (np.array(fb), np.array(ne))
        b_grid, b_ref, nb = brute(f, d, lo, hi, budget)
        for tag, (fb, ne) in res.items():
            ok = np.array([success(v, fs, kind) for v in fb])
            row = dict(case=name, d=d, NP=NP, G=G, budget=budget, f_star=fs, method=tag,
                       success=int(ok.sum()), n=len(fb), median_err=float(np.median(fb - fs)),
                       worst_err=float(np.max(fb - fs)), mean_nfe=float(np.mean(ne)))
            out.append(row)
        for tag, val, n in (("brute_grid", b_grid, nb), ("brute_grid+local", b_ref, nb)):
            ok = success(val, fs, kind)
            out.append(dict(case=name, d=d, NP=NP, G=G, budget=budget, f_star=fs, method=tag,
                            success=int(ok), n=1, median_err=float(val - fs),
                            worst_err=float(val - fs), mean_nfe=float(n)))
        print(f"{name:13s} done ({time.time() - t0:.0f} s)", flush=True)
    hdr = "case,d,NP,G,budget,f_star,method,success,n,median_err,worst_err,mean_nfe"
    with open(os.path.join(HERE, "benchmarks.csv"), "w") as fh:
        fh.write(hdr + "\n")
        for r in out:
            fh.write(",".join(str(r[k]) if not isinstance(r[k], float) else f"{r[k]:.6g}"
                              for k in hdr.split(",")) + "\n")
    dump_json(out, "benchmarks.json")
    # compact table
    meths = ["sade_src", "sade_src_w15", "sade_ref_lp5", "sade_ref_lp50", "scipy_de",
             "brute_grid", "brute_grid+local"]
    print("case          " + " ".join(f"{m:>16s}" for m in meths))
    for name, *_ in CASES:
        cells = []
        for m in meths:
            r = next(r for r in out if r["case"] == name and r["method"] == m)
            cells.append(f"{r['success']:>2d}/{r['n']:<2d} {r['median_err']:9.2e}")
        print(f"{name:13s} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
