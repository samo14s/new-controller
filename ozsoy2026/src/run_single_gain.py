"""Sec. 4 'alternative strategy': one gain for the whole 500-3000 rpm range.

The fixed gain must satisfy Eq. (2)'s feasibility rule at every speed; among
the feasible gains the one with the largest mean a_lim is kept (with a_lim
increasing in g this is also the max-min choice).  Solved two ways:
  REF   from the reference scans of run_gain_map.py (same grid, all speeds),
        refined by bisection on the joint feasibility edge;
  SADE  the paper's optimiser on the joint objective.
    python run_single_gain.py --tag main [--sade]
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _feas_alim(args):
    rpm, g, zeta, kind = args
    P.set_damping(zeta)
    from objective import fitness
    f, al, fd = fitness(rpm, g, kind, detail=True)
    return f > 0, al * 1e3, fd


def joint(g, rpms, zeta, kind, pool):
    res = pool.map(_feas_alim, [(n, g, zeta, kind) for n in rpms])
    ok = np.array([r[0] for r in res])
    al = np.array([r[1] for r in res])
    return ok, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sade", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
    zeta, kind = float(d["zeta"]), str(d["kind"])
    rpms, gs = d["rpm"], d["scan_g"]
    feas = d["scan_f"] > 0
    all_ok = feas.all(axis=0)
    mean_a = d["scan_alim"].mean(axis=0)
    obj = np.where(all_ok, mean_a, 0.0)
    j = int(np.argmax(obj))
    out = dict(scan_g=gs, all_ok=all_ok, mean_a=mean_a, n_feasible_speeds=feas.sum(axis=0))
    # which speeds bind first above the best grid gain
    with Pool(a.workers) as pool:
        lo, hi = gs[j], gs[j + 1] if j + 1 < len(gs) else gs[j]
        while hi - lo > 0.25:
            mid = 0.5 * (lo + hi)
            ok, _ = joint(mid, rpms, zeta, kind, pool)
            lo, hi = (mid, hi) if ok.all() else (lo, mid)
        g_ref = lo
        ok, al = joint(g_ref, rpms, zeta, kind, pool)
        ok_hi, _ = joint(hi, rpms, zeta, kind, pool)
        out.update(g_ref=g_ref, alim_ref=al, binding_rpm=rpms[~ok_hi])
        print(f"REF single gain g={g_ref:.2f} V s/m, mean a_lim={al.mean():.3f} mm, "
              f"min a_lim={al.min():.3f} mm; binding speeds above it: {rpms[~ok_hi]}")
        if a.sade:
            from sade import sade

            def fun(x):
                okx, alx = joint(x[0], rpms, zeta, kind, pool)
                return alx.mean() if okx.all() else 0.0
            runs = []
            for s in (1, 2, 3):
                rng = np.random.default_rng(2000 + s)
                npop = 10
                init = (np.arange(npop) + rng.random(npop)) / npop * 3000.0
                r = sade(fun, 0.0, 3000.0, np_=npop, max_gen=30, lp=5, window=8,
                         tol=1e-4, seed=s, init=init[:, None])
                runs.append((float(r["x"][0]), r["f"], r["n_eval"], r["gens"]))
                print(f"SADE seed {s}: g={r['x'][0]:.2f} mean a={r['f']:.3f} "
                      f"evals={r['n_eval']} gens={r['gens']}", flush=True)
            out["sade_runs"] = np.array(runs)
    np.savez_compressed(os.path.join(RES, f"single_gain_{a.tag}.npz"), **out)


if __name__ == "__main__":
    main()
