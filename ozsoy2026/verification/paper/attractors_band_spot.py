"""Spot check of the 'bistable band' away from the four basin speeds: bounded
chatter or divergence at the deepest cuts of the band.

Purpose.  The manuscript maps the lower edge of the 'bistable band' at 126
speeds (results/c1_band.npz) with the D1 classifier over 300 revolutions, which
counts bounded and divergent chatter alike.  attractors_kick.py and
attractors_bifurcation.py find bounded chatter at the four basin speeds; but
divergence (|x| > 1 cm) is known at 1050 rpm and 0.9 a_lim
(verification/tdsim/check9_full_memory.json, step entry), where the linear limit
is deeper.  This script checks, at the band speeds with the deepest cuts, whether
the motion outside the basin of the forced orbit is bounded chatter (a second
bounded attractor, i.e. bistability) or divergence (loss of attraction to
infinity in the model, i.e. a finite-amplitude instability without a demonstrated
second attractor).

Protocol.  Speeds: those of results/c1_band.npz with a band edge and
a_lim(g) >= 21 mm (the gain g of the gain map results/feas_g1500.npz, as in the
band campaign), and 1050 rpm (the check9 case).  Depths inside the band: 0.9 and
0.98 a_lim(g), and 0.99 a_lim(g) where the band edge is above 0.9 a_lim; only
depths above the band edge (at 1050 rpm, off the 20-rpm band grid, the edge of
the lower neighbour, 1040 rpm, is used).  Initial conditions (1600 revolutions):
step entry from rest; face entry plus a kick of 0.03 m/s or 0.3 m/s at the
start of revolution 100 (phase 0).  9 V clip, full surface memory (nreg = 0),
damping 1.34 %.  Final state over the last 300 revolutions as in
attractors_bifurcation.py: forced / chatter (bounded, stationary within 10 %) /
transient / diverged.

Run (2 worker processes, about 6 min):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_band_spot.py --workers 2
Writes verification/paper/attractors_band_spot.json.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import attractors_common as AC

HERE = AC.HERE
NREV = 1600
KICK_REV = 100
STEADY = 300
ICS = ("step", "kick0.03", "kick0.3")
STAT_TOL = 0.10
A_MIN = 21e-3


def run(args):
    rpm, g, a, ic = args
    AC.setup()
    from tdsim import simulate
    t0 = time.time()
    opts = {} if ic == "step" else dict(edge=True, kick_step=KICK_REV * AC.SPR, kick_dv=float(ic[4:]))
    r = simulate(rpm, a, g, vmax=AC.VMAX, store=True, nrev=NREV, nreg=0, **opts)
    dstep = AC.divergence_step(r)
    out = dict(rpm=rpm, g=g, a_mm=a * 1e3, ic=ic, diverged=dstep is not None,
               diverged_rev=None if dstep is None else dstep / AC.SPR, max_x_mm=float(r["max_x"] * 1e3))
    if dstep is not None:
        out["state"] = "diverged"
    else:
        i0, i1 = (NREV - STEADY) * AC.SPR, NREV * AC.SPR
        s = AC.window_stats(r, i0, i1, rpm)
        out.update({k: s[k] for k in ("pp_x", "rms_x", "rms_V", "clip_frac", "res_tooth", "f_chatter")})
        if s["res_tooth"] < AC.RET_TOL:
            out["state"] = "forced"
        else:
            x = r["x"]
            pa, pb = np.ptp(x[i0:i0 + 100 * AC.SPR]), np.ptp(x[i1 - 100 * AC.SPR:i1])
            h = (i0 + i1) // 2
            ra, rb = np.std(x[i0:h]), np.std(x[h:i1])
            out["state"] = "chatter" if abs(pb / pa - 1) <= STAT_TOL and abs(rb / ra - 1) <= STAT_TOL else "transient"
        out["transient_peak_mm"] = float(np.max(np.abs(r["x"][:(KICK_REV + 200) * AC.SPR])) * 1e3)
    out["runtime_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    AC.setup()
    from zoa import a_lim
    from c1_unsafe_zone import gain_map
    rpm_map, g_map, zeta = gain_map("g1500")
    assert abs(zeta - AC.ZETA) < 1e-12
    b = np.load(os.path.join(AC.RES, "c1_band.npz"))
    cases = []
    for n, gb, edge in zip(b["rpm"], b["g"], b["a_band_low"]):
        if not np.isfinite(edge):
            continue
        al = a_lim(float(n), float(gb))
        if al >= A_MIN:
            cases.append((float(n), float(gb), al, float(edge) * 1e-3))
    g1050 = float(np.interp(1050.0, rpm_map, g_map))
    e1040 = float(b["a_band_low"][np.nonzero(np.isclose(b["rpm"], 1040.0))[0][0]]) * 1e-3
    cases.append((1050.0, g1050, a_lim(1050.0, g1050), e1040))
    cases.sort()
    jobs, plan = [], []
    for n, g, al, edge in cases:
        fr = [0.9, 0.98] + ([0.99] if edge > 0.9 * al else [])
        fr = [f for f in fr if f * al > edge]
        plan.append(dict(rpm=n, g=g, a_lim_mm=al * 1e3, band_edge_mm=edge * 1e3, band_edge_over_alim=edge / al,
                         fracs=fr))
        jobs += [(n, g, f * al, ic) for f in fr for ic in ICS]
    print(f"{len(cases)} speeds, {len(jobs)} runs", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    for p in plan:
        rr = [r for r in res if r["rpm"] == p["rpm"]]
        p["runs"] = rr
        p["states"] = {f"{r['a_mm']:.3f} mm {r['ic']}": r["state"] for r in rr}
        p["any_diverged"] = any(r["state"] == "diverged" for r in rr)
        p["any_bounded_chatter"] = any(r["state"] == "chatter" for r in rr)
        p["bounded_chatter_at_every_depth"] = all(
            any(r["state"] == "chatter" for r in rr if abs(r["a_mm"] - f * p["a_lim_mm"]) < 1e-9)
            for f in p["fracs"])
    counts = {}
    for r in res:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    summ = dict(n_speeds=len(plan), n_runs=len(res), state_counts=counts,
                speeds_with_divergence=[p["rpm"] for p in plan if p["any_diverged"]],
                speeds_bounded_only=[p["rpm"] for p in plan if p["any_bounded_chatter"] and not p["any_diverged"]],
                bounded_pp_x_mm=[min(r["pp_x"] for r in res if r["state"] == "chatter") * 1e3,
                                 max(r["pp_x"] for r in res if r["state"] == "chatter") * 1e3]
                if any(r["state"] == "chatter" for r in res) else None)
    out = dict(note=__doc__, zeta=AC.ZETA, build=AC.build_id(),
               protocol=dict(nrev=NREV, kick_rev=KICK_REV, steady=STEADY, ics=ICS, a_lim_min_m=A_MIN,
                             vmax=AC.VMAX, nreg=0, stationarity_tol=STAT_TOL),
               summary=summ, speeds=plan, runtime_s=round(time.time() - t0, 1), workers=a.workers)
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_band_spot.json"), "w"), indent=1)
    for p in plan:
        print(f"{p['rpm']:.0f} rpm g {p['g']:.0f} a_lim {p['a_lim_mm']:.2f} band {p['band_edge_mm']:.2f}: {p['states']}")
    print(json.dumps(AC.jsonable(summ), indent=1))
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
