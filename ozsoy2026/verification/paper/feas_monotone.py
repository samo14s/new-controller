"""Is the steady authority condition of Section 3 (Eq. feas) monotone in g?

For every speed of the map (500-3000 rpm, step 10) and g = 0, 20, ..., 3000
V s/m, compute F_ss(g) = K_a * max over 80 depths up to 0.9 a_lim(g) of the
settled command (feasibility.demand_envelope, the function the gain map uses)
and record (i) every decrease of F_ss between neighbouring gains, with the
demand where it occurs, and (ii) whether the feasible set {g: F_ss <= 27 N}
is a single interval [0, g_b] on this grid (needed by the bisection of
feasibility.edge_gain).

    OMP_NUM_THREADS=1 python3 verification/paper/feas_monotone.py
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")
GRID = np.arange(0.0, 3000.0 + 1e-9, 20.0)


def one(rpm):
    import params as P
    P.set_damping(float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"]))
    from feasibility import demand_envelope
    F = np.array([P.GA_GAIN * demand_envelope(rpm, g)[1] for g in GRID])
    d = np.diff(F)
    drops = [dict(g_from=float(GRID[i]), g_to=float(GRID[i + 1]), drop_N=float(-d[i]), F_N=float(F[i]))
             for i in np.nonzero(d < 0)[0]]
    feas = F <= P.F_MAX
    # one interval [0, g_b]: once infeasible, never feasible again
    first_bad = np.argmax(~feas) if (~feas).any() else len(feas)
    one_interval = bool(feas[:first_bad].all() and not feas[first_bad:].any())
    return dict(rpm=float(rpm), drops=drops, one_interval=one_interval,
                g_b_grid=float(GRID[first_bad - 1]) if first_bad > 0 else 0.0)


def main():
    t0 = time.time()
    rpms = np.arange(500, 3001, 10)
    with Pool(4) as pool:
        rows = pool.map(one, rpms, chunksize=4)
    drops = [dict(rpm=r["rpm"], **x) for r in rows for x in r["drops"]]
    out = dict(note=__doc__, n_speeds=len(rows), n_gains=len(GRID),
               speeds_with_drops=sorted({d["rpm"] for d in drops}),
               max_drop_N=max((d["drop_N"] for d in drops), default=0.0),
               min_demand_at_drop_N=min((d["F_N"] for d in drops), default=None),
               min_gain_at_drop=min((d["g_from"] for d in drops), default=None),
               all_one_interval=all(r["one_interval"] for r in rows),
               rows=rows, runtime_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, "feas_monotone.json"), "w"), indent=1)
    print({k: out[k] for k in ("n_speeds", "n_gains", "max_drop_N", "min_demand_at_drop_N",
                               "min_gain_at_drop", "all_one_interval", "runtime_s")},
          "speeds with drops:", len(out["speeds_with_drops"]))


if __name__ == "__main__":
    main()
