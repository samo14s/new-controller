"""Reruns behind three statements of the manuscript that README 4.6 states
without a stored file (paper/sections/s2_model.tex and s3_feasibility.tex).

A. The clip acts on the command, before the actuator dynamics, so the force
   produced can exceed F_max transiently; clipping at 8.1 V instead of 9 V
   keeps it below 27 N and lowers the clipped step-entry limits at
   g = 1500 V s/m.  Protocol of results/table3_main.csv (run_table3.py):
   tdboundary.boundary(rpm, g, vmax, nreg = 0), step entry, D1 classifier.
   The peak actuator force is max_fact of the deepest stable run (300 rev).
B. The optimum of the published condition, Eq. (2), at 2500 rpm when the
   simulated cut starts with a soft entry instead of the step entry: the
   protocol of objective.demand (K_a g max|x'| over 50 unclipped revolutions
   with single regeneration at a_lim(g)) on a 50 V s/m grid.

    OMP_NUM_THREADS=1 python3 verification/paper/support_runs.py
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)
RES = os.path.join(HERE, "..", "..", "results")


def _setup():
    import params as P
    P.set_damping(float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"]))
    return P


def clip_case(args):
    rpm, vmax = args
    P = _setup()
    from tdboundary import boundary, classify
    a = boundary(rpm, 1500.0, vmax=vmax, nreg=0)
    run = classify(rpm, a, 1500.0, vmax, nreg=0)["run"]
    return dict(rpm=rpm, g=1500.0, vmax=vmax, a_td_mm=a * 1e3, stable_at_a=True,
                peak_actuator_force_N=float(run["max_fact"]))


def eq2_case(args):
    entry, g = args
    P = _setup()
    from zoa import a_lim
    from tdsim import simulate
    al = a_lim(2500, g)
    if g == 0.0 or not np.isfinite(al):
        return dict(entry=entry, g=g, a_lim_mm=al * 1e3, demand_N=0.0)
    opts = dict(step={}, ramp1=dict(ramp_f=1.0), ramp10=dict(ramp_f=10.0), face=dict(edge=True))[entry]
    r = simulate(2500, al, g, nrev=P.N_REVS, **opts)
    d = np.inf if r["diverged"] else P.GA_GAIN * r["max_demand"]
    return dict(entry=entry, g=g, a_lim_mm=al * 1e3, demand_N=float(d))


def main():
    t0 = time.time()
    with Pool(4) as pool:
        clip = pool.map(clip_case, [(2500, 9.0), (2500, 8.1), (3000, 9.0), (3000, 8.1)], chunksize=1)
        grid = np.arange(0.0, 3000.0 + 1e-9, 50.0)
        rows = pool.map(eq2_case, [(e, g) for e in ("step", "ramp1", "ramp10", "face") for g in grid], chunksize=4)
    P = _setup()
    opt = {}
    for e in ("step", "ramp1", "ramp10", "face"):
        feas = [r for r in rows if r["entry"] == e and r["demand_N"] <= P.F_MAX]
        best = max(feas, key=lambda r: r["a_lim_mm"])
        opt[e] = dict(g_opt=best["g"], a_lim_mm=best["a_lim_mm"], demand_N=best["demand_N"])
    out = dict(note=__doc__, clip=clip, eq2_optimum_2500rpm=opt, eq2_grid=rows,
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, "support_runs.json"), "w"), indent=1)
    for c in clip:
        print(f"{c['rpm']} rpm, {c['vmax']} V: a_td {c['a_td_mm']:.2f} mm, peak force {c['peak_actuator_force_N']:.1f} N")
    for e, o in opt.items():
        print(f"Eq. (2) optimum at 2500 rpm, {e}: g = {o['g_opt']:.0f} V s/m, a_lim {o['a_lim_mm']:.2f} mm")
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
