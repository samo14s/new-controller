"""src/sade.py on the real reproduction objective fitness(rpm, g, 'force').

    python3 sade_real.py --tag A --init strat   --np 20 --gens 60 --window 15
    python3 sade_real.py --tag B --init uniform --np 20 --gens 60 --window 15

init: strat    stratified uniform sample over [0, 3000] (what src/gainmap.run_sade does)
      uniform  plain uniform sample drawn inside sade() (init=None)
      strat0   stratified, first member replaced by g = 0 (uncontrolled, always feasible)
Writes sade_real_<tag>.csv / .json.  Success = best fitness >= 99 % of the
brute-force optimum in bruteforce.json.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
import numpy as np
from common import CachedFitness, RPMS, G_LO, G_HI, HERE, dump_json
from sade import sade


def make_init(kind, np_, s):
    if kind == "uniform":
        return None
    rng = np.random.default_rng(1000 + s)
    init = G_LO + (np.arange(np_) + rng.random(np_)) / np_ * (G_HI - G_LO)
    if kind == "strat0":
        init[0] = G_LO
    return init[:, None]


def work(job):
    rpm, cfg = job
    fit = CachedFitness(rpm)
    rows = []
    for s in cfg["seeds"]:
        t0 = time.time()
        n0 = fit.n_new
        r = sade(lambda x: fit(x[0]), G_LO, G_HI, np_=cfg["np"], max_gen=cfg["gens"],
                 lp=cfg["lp"], window=cfg["window"], tol=cfg["tol"], seed=s,
                 init=make_init(cfg["init"], cfg["np"], s))
        rows.append(dict(rpm=rpm, seed=s, g=float(r["x"][0]), a=r["f"], gens=r["gens"],
                         n_eval=r["n_eval"], new_eval=fit.n_new - n0,
                         seconds=time.time() - t0, history=r["history"].tolist(),
                         probs=r["probs"].tolist()))
    fit.save()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default="strat")
    ap.add_argument("--np", type=int, default=20)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--lp", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = dict(init=a.init, np=a.np, gens=a.gens, window=a.window, lp=a.lp, tol=a.tol,
               seeds=a.seeds)
    ref = {r["rpm"]: r for r in json.load(open(os.path.join(HERE, "bruteforce.json")))}
    t0 = time.time()
    with Pool(a.workers) as p:
        out = sum(p.map(work, [(rpm, cfg) for rpm in RPMS]), [])
    lines = ["rpm,seed,g_sade,a_sade_mm,g_opt,a_opt_mm,rel_gap_pct,within_1pct,gens,n_eval,new_eval,seconds"]
    summ = {}
    for r in out:
        b = ref[r["rpm"]]
        gap = 100 * (1 - r["a"] / b["a_opt"])
        r.update(g_opt=b["g_opt"], a_opt=b["a_opt"], rel_gap_pct=gap, ok=bool(gap <= 1.0))
        lines.append(f"{r['rpm']},{r['seed']},{r['g']:.3f},{r['a']:.5f},{b['g_opt']:.3f},"
                     f"{b['a_opt']:.5f},{gap:.3f},{int(r['ok'])},{r['gens']},{r['n_eval']},"
                     f"{r['new_eval']},{r['seconds']:.1f}")
        s = summ.setdefault(r["rpm"], dict(ok=0, n=0, gaps=[], gens=[], gerr=[]))
        s["ok"] += r["ok"]
        s["n"] += 1
        s["gaps"].append(gap)
        s["gens"].append(r["gens"])
        s["gerr"].append(r["g"] - b["g_opt"])
    with open(os.path.join(HERE, f"sade_real_{a.tag}.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for rpm, s in summ.items():
        print(f"{rpm} rpm: within 1 %: {s['ok']}/{s['n']}  median gap {np.median(s['gaps']):.2f} %"
              f"  max gap {np.max(s['gaps']):.2f} %  gens {s['gens']}  g-g_opt {np.round(s['gerr'], 1).tolist()}")
    print(f"total {time.time() - t0:.0f} s")
    dump_json(dict(cfg=cfg, runs=out, summary=summ), f"sade_real_{a.tag}.json")


if __name__ == "__main__":
    main()
