"""Settings study for src/sade.py on a tabulated copy of the real objective.

The brute-force scan (bruteforce_<rpm>.csv) shows that at 1000, 1950 and 2500 rpm
fitness(rpm, g, 'force') is a_lim(g) (smooth, increasing) on one interval
[0, g_edge] and 0 beyond it.  The surrogate is therefore
    f_s(g) = interp(g; tabulated a_lim)   for g <= g_edge,   0 otherwise,
with g_edge = the brute-force optimum.  It is cheap, so each setting can be run
with many seeds.  sade_real.py checks the baseline on the real objective.

One factor is changed at a time around the baseline
    NP = 20, max_gen = 60, lp = 5, window = 15, tol = 1e-4, stratified init.
Writes surrogate_study.csv / .json.
"""
import json
import os
import time
import numpy as np
from common import RPMS, G_LO, G_HI, HERE, dump_json
from sade import sade

N_SEEDS = 60


def build(rpm, ref):
    d = np.loadtxt(os.path.join(HERE, f"bruteforce_{rpm}.csv"), delimiter=",", skiprows=1)
    g, f, al = d[:, 0], d[:, 1], d[:, 2]
    edge = ref["g_opt"]
    m = g <= edge + 1e-9
    gt, at = g[m], al[m]
    o = np.argsort(gt)
    gt, at = gt[o], at[o]

    def fs(x):
        x = float(x[0])
        if x > edge or x < G_LO:
            return 0.0
        return float(np.interp(x, gt, at))
    return fs


def init_pop(kind, np_, s):
    if kind == "uniform":
        return None
    rng = np.random.default_rng(1000 + s)
    p = G_LO + (np.arange(np_) + rng.random(np_)) / np_ * (G_HI - G_LO)
    if kind == "strat0":
        p[0] = G_LO
    if kind == "lhs_shuffled":       # same design, different stream
        p = G_LO + (rng.permutation(np_) + rng.random(np_)) / np_ * (G_HI - G_LO)
    return p[:, None]


BASE = dict(np=20, gens=60, lp=5, window=15, tol=1e-4, init="strat")
VARIANTS = [
    ("baseline", {}),
    ("init=uniform", dict(init="uniform")),
    ("init=strat0", dict(init="strat0")),
    ("window=8", dict(window=8)),
    ("window=25", dict(window=25)),
    ("no window", dict(window=10**6)),
    ("NP=10", dict(np=10)),
    ("NP=30", dict(np=30)),
    ("gens=100", dict(gens=100)),
    ("lp=20", dict(lp=20)),
    ("tol=1e-6", dict(tol=1e-6)),
    ("uniform,no window", dict(init="uniform", window=10**6)),
    ("rec: strat0,NP=20,G=100,W=25", dict(init="strat0", gens=100, window=25)),
    ("rec: strat0,NP=30,G=100,W=25", dict(init="strat0", np=30, gens=100, window=25)),
]


def main():
    t0 = time.time()
    ref = {r["rpm"]: r for r in json.load(open(os.path.join(HERE, "bruteforce.json")))}
    funs = {rpm: build(rpm, ref[rpm]) for rpm in RPMS}
    rows = []
    for name, ch in VARIANTS:
        cfg = dict(BASE, **ch)
        for rpm in RPMS:
            a_opt, g_opt = ref[rpm]["a_opt"], ref[rpm]["g_opt"]
            gaps, gerr, ne, gens, zero = [], [], [], [], 0
            for s in range(1, N_SEEDS + 1):
                r = sade(funs[rpm], G_LO, G_HI, np_=cfg["np"], max_gen=cfg["gens"],
                         lp=cfg["lp"], window=cfg["window"], tol=cfg["tol"], seed=s,
                         init=init_pop(cfg["init"], cfg["np"], s))
                gaps.append(100 * (1 - r["f"] / a_opt))
                gerr.append(100 * abs(r["x"][0] - g_opt) / g_opt)
                ne.append(r["n_eval"])
                gens.append(r["gens"])
                zero += r["f"] == 0.0
            gaps, gerr = np.array(gaps), np.array(gerr)
            rows.append(dict(variant=name, rpm=rpm, **cfg, n=N_SEEDS,
                             within_1pct_a=int(np.sum(gaps <= 1.0)),
                             within_2pct_g=int(np.sum(gerr <= 2.0)),
                             zero_fitness=int(zero),
                             median_gap_pct=float(np.median(gaps)),
                             p90_gap_pct=float(np.percentile(gaps, 90)),
                             median_gerr_pct=float(np.median(gerr)),
                             mean_nfe=float(np.mean(ne)), mean_gens=float(np.mean(gens))))
            q = rows[-1]
            print(f"{name:30s} {rpm:5d}  a<=1%: {q['within_1pct_a']:3d}/{N_SEEDS}  "
                  f"g<=2%: {q['within_2pct_g']:3d}/{N_SEEDS}  f=0: {q['zero_fitness']:2d}  "
                  f"med gap {q['median_gap_pct']:.3f}%  p90 {q['p90_gap_pct']:.2f}%  "
                  f"nfe {q['mean_nfe']:.0f}  gens {q['mean_gens']:.1f}", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(HERE, "surrogate_study.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    dump_json(rows, "surrogate_study.json")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
