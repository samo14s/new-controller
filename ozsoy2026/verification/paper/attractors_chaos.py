"""Is the bounded clipped chatter chaotic?  Twin runs with a perturbed kick.

Purpose.  Appendix A of the manuscript states that clipped chatter 'is
chaotic and separates pointwise'.  attractors_kick.py finds the chatter that a
kick releases bounded and statistically stationary; this script tests whether
it is chaotic (sensitive to initial conditions) or regular (periodic or
quasi-periodic), by following two runs whose kicks differ by a relative 1e-9.
On a regular attractor the separation of the two runs stays at the size of
the initial difference (or grows at most linearly, along the neutral phase of
a torus); on a chaotic one it grows exponentially until it is as large as the
attractor.

Protocol.  Same model and kick protocol as attractors_kick.py (face entry, kick
at revolution 100 at the worst phase of results/c1_basin.npz, full surface
memory, 9 V clip, g = 1500 V s/m, damping 1.34 %), at the four points at
0.9 a_lim(1500) with the 2x kick, 2000 revolutions after the kick; as the
deepest cuts of attractors_bifurcation.py, the step entry at 1.05 a_lim(1500) at
the four speeds (perturbation: depth a (1 + 1e-9)), 1600 revolutions; and the two
bounded clipped-chatter cases of verification/tdsim/check9_full_memory.json on
which the statement rests (step entry, 2350 rpm, g = 1217 V s/m, 14.406 mm, and
2500 rpm, g = 1500, 15 mm), 1600 revolutions.  Stored:
the per-revolution maximum of |x1 - x2|, the separation at the end, its ratio
to the peak-to-peak of x over the last 300 revolutions, and the
growth rate per revolution fitted to log separation while it is between 1e3
times its first value and 1e-2 of the peak-to-peak (none if never).
'chaotic' if the final separation exceeds 10 % of the peak-to-peak, 'regular'
if it stays below 1e-4 of it, else 'undecided'.

Run (2 worker processes, about 3 min):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_chaos.py --workers 2
Writes verification/paper/attractors_chaos.json and attractors_chaos.npz
(per-revolution separation).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import attractors_common as AC

HERE = AC.HERE
EPS = 1e-9
G = 1500.0


def pair(args):
    """Two runs that differ by a relative EPS in the kick (kind 'kick') or in the depth ('step')."""
    kind, rpm, a, phase, kick, nrev_post, g = args
    AC.setup()
    from tdsim import simulate
    from c1_unsafe_zone import KICK_REV, N_PHASE
    t0 = time.time()
    if kind == "kick":
        ks = KICK_REV * AC.SPR + (phase * AC.ND) // N_PHASE
        st = (ks // AC.ND + 1) * AC.ND
        nrev = KICK_REV + nrev_post + 1
        runs = [simulate(rpm, a, g, vmax=AC.VMAX, store=True, nrev=nrev, edge=True, kick_step=ks,
                         kick_dv=kick * f, nreg=0) for f in (1.0, 1.0 + EPS)]
    else:
        st, nrev = 0, nrev_post
        runs = [simulate(rpm, a * f, g, vmax=AC.VMAX, store=True, nrev=nrev, nreg=0) for f in (1.0, 1.0 + EPS)]
    out = dict(kind=kind, rpm=rpm, g=g, a_mm=a * 1e3, phase=phase, kick_m_s=kick, eps=EPS,
               diverged=[bool(r["diverged"]) for r in runs])
    sep = np.full(nrev_post, np.nan)
    if not any(out["diverged"]):
        d = np.abs(runs[0]["x"] - runs[1]["x"])
        for k in range(nrev_post):
            sep[k] = d[st + k * AC.SPR:st + (k + 1) * AC.SPR].max()
        x = runs[0]["x"]
        pp = float(np.ptp(x[(nrev - 300) * AC.SPR:nrev * AC.SPR]))
        res_tooth = float(np.max(np.abs(x[(nrev - 300) * AC.SPR:nrev * AC.SPR] -
                                        x[(nrev - 300) * AC.SPR - AC.ND:nrev * AC.SPR - AC.ND])))
        s0 = sep[1] if sep[1] > 0 else np.nanmin(sep[sep > 0])
        fin = float(np.max(sep[-300:]))
        use = (sep > 1e3 * s0) & (sep < 1e-2 * pp)
        k = np.nonzero(use)[0]
        rate = None
        if k.size >= 10:
            sl = np.polyfit(k, np.log(sep[k]), 1)[0]
            rate = float(sl)
        out.update(pp_x_last300_mm=pp * 1e3, res_tooth_last300_m=res_tooth, sep_first_m=float(s0),
                   sep_final_m=fin, sep_final_over_pp=fin / pp if pp > 0 else None,
                   growth_rate_per_rev=rate,
                   e_folding_revs=(1.0 / rate) if rate and rate > 0 else None,
                   verdict=("forced orbit" if res_tooth < AC.RET_TOL else
                            "chaotic" if fin > 0.1 * pp else "regular" if fin < 1e-4 * pp else "undecided"))
    else:
        out["verdict"] = "diverged"
    out["runtime_s"] = round(time.time() - t0, 1)
    return dict(res=out, sep=sep.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    AC.setup()
    from zoa import a_lim
    d = np.load(os.path.join(AC.RES, "c1_basin.npz"))
    jobs = []
    for i in range(len(d["rpm"])):
        if d["vmax"][i] > 0 and abs(d["frac"][i] - 0.9) < 1e-9:
            dv = d["dv"][i]
            ph = int(np.argmin(np.where(np.isfinite(dv), dv, np.inf)))
            al = a_lim(float(d["rpm"][i]), G)
            jobs.append(("kick", float(d["rpm"][i]), 0.9 * al, ph, float(min(2 * dv[ph], 3.0)), 2000, G))
    for n in (1090.0, 1420.0, 2500.0, 2870.0):
        jobs.append(("step", n, 1.05 * a_lim(n, G), -1, 0.0, 1600, G))
    c9 = json.load(open(os.path.join(AC.ROOT, "verification", "tdsim", "check9_full_memory.json")))
    for c in c9["clipped_chatter"]:
        if not c["options"] and not c["full"]["diverged"]:
            jobs.append(("step", float(c["rpm"]), c["a_mm"] * 1e-3, -1, 0.0, 1600, float(c["g"])))
    with Pool(a.workers) as pool:
        res = pool.map(pair, jobs, chunksize=1)
    out = dict(note=__doc__, zeta=AC.ZETA, build=AC.build_id(), cases=[r["res"] for r in res],
               runtime_s=round(time.time() - t0, 1), workers=a.workers)
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_chaos.json"), "w"), indent=1)
    L = max(len(r["sep"]) for r in res)
    sep = np.full((len(res), L), np.nan, dtype=np.float32)
    for k, r in enumerate(res):
        sep[k, :len(r["sep"])] = r["sep"]
    np.savez_compressed(os.path.join(HERE, "attractors_chaos.npz"), sep=sep,
                        kind=np.array([r["res"]["kind"] for r in res]),
                        rpm=np.array([r["res"]["rpm"] for r in res]),
                        a_mm=np.array([r["res"]["a_mm"] for r in res]), eps=EPS,
                        note="sep: per-revolution max |x1 - x2| [m] after the kick (kick cases) or from t = 0 (step)")
    for r in res:
        o = r["res"]
        print(f"{o['kind']} {o['rpm']:.0f} rpm g {o['g']:.0f} {o['a_mm']:.3f} mm: {o['verdict']}, final sep/pp "
              f"{o.get('sep_final_over_pp')}, rate/rev {o.get('growth_rate_per_rev')}")
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
