"""Gain map under the reformulated feasibility condition (feasibility.py).

Per speed: the upper edge of the feasible interval by bisection (reference),
SADE on the same objective (3 seeds, the main run's settings), and a
time-domain certificate at the operating depth kappa*a_lim with the clipped
actuator, starting from rest.
    python run_feasibility_map.py --tag g1500 --gmax 1500 --sade --certify
    python run_feasibility_map.py --tag g3000 --gmax 3000
    python run_feasibility_map.py --tag k095 --gmax 1500 --kappa 0.95
    python run_feasibility_map.py --tag guard085 --gmax 1500 --guard 0.85
    python run_feasibility_map.py --tag g1500 --certify-only   # redo only the time-domain check
Writes ../results/feas_<tag>.npz and .csv
"""
import argparse
import os
import time
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def work(args):
    rpm, zeta, gmax, kappa, guard, do_sade, do_cert = args
    P.set_damping(zeta)
    import feasibility as F
    from zoa import a_lim
    t0 = time.time()
    g_ref = F.edge_gain(rpm, g_hi=gmax, kappa=kappa, guard=guard)
    f_ref, al_ref, d_ref = F.fitness(rpm, g_ref, kappa, guard, detail=True)
    row = dict(rpm=rpm, g_ref=g_ref, a_ref=f_ref, demand_ref=d_ref,
               a_unc=a_lim(rpm, 0.0) * 1e3, at_bound=float(g_ref >= gmax))
    if do_sade:
        from sade import sade
        runs = []
        for s in (1, 2, 3):
            rng = np.random.default_rng(1000 + s)
            init = (np.arange(20) + rng.random(20)) / 20 * gmax
            r = sade(lambda x: F.fitness(rpm, x[0], kappa, guard), 0.0, gmax, np_=20,
                     max_gen=60, lp=5, window=25, tol=1e-4, seed=s, init=init[:, None])
            runs.append((float(r["x"][0]), r["f"], r["n_eval"], r["gens"]))
        best = max(runs, key=lambda q: q[1])
        row.update(g_sade=best[0], a_sade=best[1], sade_runs=np.array(runs))
    g = row["g_sade"] if do_sade and row["a_sade"] > f_ref else g_ref         # the gain of the better run
    row["g_best"] = g
    row["a_best"] = max(f_ref, row.get("a_sade", 0.0))
    if do_cert:
        row.update(cert_fields(rpm, g, kappa))
    row["seconds"] = time.time() - t0
    return row


def cert_fields(rpm, g, kappa):
    import feasibility as F
    c = F.certify(rpm, g, kappa)
    return dict(cert_stable=float(c["stable"]), cert_fact_settled=c["fact_settled"],
                cert_demand_td=c["demand_settled_td"],
                cert_demand_cf=c["demand_closed_form"],
                cert_entry_saturated=float(c["entry_saturated"]))


def cert_work(args):
    rpm, g, zeta, kappa = args
    P.set_damping(zeta)
    t0 = time.time()
    return dict(rpm=rpm, **cert_fields(rpm, g, kappa), seconds=time.time() - t0)


def certify_only(a):
    """Recompute the time-domain check of a stored map at its g_best; every
    other field is kept."""
    path = os.path.join(RES, f"feas_{a.tag}.npz")
    d = dict(np.load(path))
    jobs = [(float(n), float(g), float(d["zeta"]), float(d["kappa"])) for n, g in zip(d["rpm"], d["g_best"])]
    rows, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(cert_work, jobs)):
            rows.append(r)
            print(f"[{i+1}/{len(jobs)}] {r['rpm']:.0f} rpm  cert {'ok' if r['cert_stable'] else 'FAIL'} "
                  f"f_act={r['cert_fact_settled']:.1f}N  ({r['seconds']:.0f}s, total {time.time()-t0:.0f}s)",
                  flush=True)
    for k in rows[0]:
        if k.startswith("cert_"):
            d[k] = np.array([r[k] for r in rows])
    from feasibility import NREG
    d["cert_nreg"] = NREG
    np.savez_compressed(path, **d)
    keys = [k for k in d if np.ndim(d[k]) == 1 and k != "sade_runs"]
    np.savetxt(os.path.join(RES, f"feas_{a.tag}.csv"), np.column_stack([d[k] for k in keys]),
               delimiter=",", header=",".join(keys), comments="", fmt="%.4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zeta", type=float, default=P.ZETA_PAPER_CURVES)
    ap.add_argument("--gmax", type=float, default=1500.0)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--guard", type=float, default=1.0)
    ap.add_argument("--sade", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--certify-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.certify_only:
        return certify_only(a)
    jobs =[(float(n), a.zeta, a.gmax, a.kappa, a.guard, a.sade, a.certify) for n in P.RPM_GRID]
    rows, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(work, jobs)):
            rows.append(r)
            msg = f"[{i+1}/{len(jobs)}] {r['rpm']:.0f} rpm  edge g={r['g_ref']:.1f} a={r['a_ref']:.2f}"
            if a.sade:
                msg += f"  sade g={r['g_sade']:.1f} a={r['a_sade']:.2f}"
            if a.certify:
                msg += f"  cert {'ok' if r['cert_stable'] else 'FAIL'} f_act={r['cert_fact_settled']:.1f}N"
            print(msg + f"  ({r['seconds']:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    keys = [k for k in rows[0] if k not in ("sade_runs", "seconds")]
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    if a.sade:
        data["sade_runs"] = np.array([r["sade_runs"] for r in rows])
    data.update(zeta=a.zeta, gmax=a.gmax, kappa=a.kappa, guard=a.guard)
    np.savez_compressed(os.path.join(RES, f"feas_{a.tag}.npz"), **data)
    np.savetxt(os.path.join(RES, f"feas_{a.tag}.csv"), np.column_stack([data[k] for k in keys]),
               delimiter=",", header=",".join(keys), comments="", fmt="%.4f")


if __name__ == "__main__":
    main()
