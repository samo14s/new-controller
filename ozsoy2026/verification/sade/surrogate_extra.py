"""(a) Fidelity of the tabulated surrogate: same seeds / init as sade_real_A_strat.
(b) Best-of-3-seeds success (what src/run_gain_map.py reports) for the settings
    used in src/gainmap.run_sade (G=50, window=12), the task baseline (G=60,
    window=15) and the recommended setting (G=60, window=25).
Writes surrogate_extra.json.
"""
import json
import os
import numpy as np
from common import RPMS, G_LO, G_HI, HERE, dump_json
from sade import sade
from surrogate_study import build, init_pop

ref = {r["rpm"]: r for r in json.load(open(os.path.join(HERE, "bruteforce.json")))}
funs = {rpm: build(rpm, ref[rpm]) for rpm in RPMS}
out = {}

# (a) fidelity
real = json.load(open(os.path.join(HERE, "sade_real_A_strat.json")))["runs"]
fid = []
for r in real:
    s = sade(funs[r["rpm"]], G_LO, G_HI, np_=20, max_gen=60, lp=5, window=15, tol=1e-4,
             seed=r["seed"], init=init_pop("strat", 20, r["seed"]))
    fid.append(dict(rpm=r["rpm"], seed=r["seed"], g_real=r["g"], g_surr=float(s["x"][0]),
                    a_real=r["a"], a_surr=s["f"], gens_real=r["gens"], gens_surr=s["gens"]))
    print("fidelity", fid[-1])
out["fidelity"] = fid

# (b) best of 3 seeds
SET = {"gainmap.run_sade (G=50,W=12)": dict(gens=50, window=12),
       "task baseline (G=60,W=15)": dict(gens=60, window=15),
       "recommended (G=60,W=25)": dict(gens=60, window=25)}
N = 60
res = []
for name, c in SET.items():
    for rpm in RPMS:
        a_opt, g_opt = ref[rpm]["a_opt"], ref[rpm]["g_opt"]
        runs = []
        for s in range(1, N + 1):
            r = sade(funs[rpm], G_LO, G_HI, np_=20, max_gen=c["gens"], lp=5,
                     window=c["window"], tol=1e-4, seed=s, init=init_pop("strat", 20, s))
            runs.append((r["f"], r["x"][0]))
        f = np.array([x[0] for x in runs])
        g = np.array([x[1] for x in runs])
        ok1 = 100 * (1 - f / a_opt) <= 1.0
        okg = 100 * np.abs(g - g_opt) / g_opt <= 2.0
        grp = np.arange(N).reshape(-1, 3)
        best = np.array([i[np.argmax(f[i])] for i in grp])
        row = dict(setting=name, rpm=rpm, n_runs=N, single_a1=int(ok1.sum()),
                   single_g2=int(okg.sum()), n_groups=len(grp),
                   best3_a1=int(ok1[best].sum()), best3_g2=int(okg[best].sum()))
        res.append(row)
        print(row, flush=True)
out["best_of_3"] = res
dump_json(out, "surrogate_extra.json")
