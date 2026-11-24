"""Bifurcation-type diagram of the clipped cut against depth at the four C1 basin speeds.

Purpose.  Show over which depths the clipped DVF cut is bistable in the strict
sense, i.e. where a low-amplitude attractor (the forced, tooth-periodic orbit)
and a high-amplitude bounded attractor (clipped chatter) coexist, where the
high branch ends, and whether it diverges anywhere (|x| > 1 cm, the simulator's
abort), as opposed to a basin that is merely bounded.

Protocol.  g = 1500 V s/m, |V| <= 9 V, full surface memory (nreg = 0), damping
of results/gainmap_main.npz (1.34 %); 1090, 1420, 2500 and 2870 rpm.  Depths: 28,
evenly spaced from a_lim(rpm, 0) (uncontrolled limit) to 1.05 a_lim(rpm, 1500).
Four initial conditions (ICs), each run NREV = 1600 revolutions:
  face       face entry (edge): settles on the forced orbit where it is stable
  kick0.3    the same plus a kick of 0.3 m/s added to x' at revolution 100
             (start of a tooth period, phase 0 of c1_unsafe_zone)
  kick0.03   the same with 0.03 m/s
  step       step entry from rest (full depth from t = 0)
Final state over the last STEADY = 300 revolutions:
  forced     max |x(t) - x(t - T)| < 0.01 fz (tooth-periodic: the forced orbit)
  chatter    bounded and off the forced orbit, stationary: peak-to-peak of x over
             the first and the last 100 of the 300 revolutions, and the RMS of x
             over the two halves, within 10 %
  transient  bounded, off the forced orbit, not stationary
  diverged   |x| > 1 cm
The steady amplitude is the peak-to-peak of x over the 300 revolutions.  Also
stored, to compare with the manuscript's 300-revolution D1 verdicts: D1 over
the 300 revolutions after the face entry (revolutions 40-340), after the step
entry (0-300) and after the kick (x[st : 400 * 512 + 1], st = 100 * 512 + 128,
the window of c1_unsafe_zone._kick_chatters).
Refinement: at each speed the lower end of the high branch (lowest grid depth
with a 'chatter' final state) is bisected N_REFINE = 5 times against the grid
depth below it, at each depth with the seven kicks of the band study
(c1_unsafe_zone.BAND_DV, 3 mm/s to 3 m/s, phase 0) on the face entry, taken in
the order of REFINE_ORDER until one ends in 'chatter'.  Being reached by one of
these ICs is sufficient, not necessary, for the high branch to exist, so the
refined end is an upper bound of the true end (fold) of the branch.

Run (2 worker processes, about 25 min):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_bifurcation.py --workers 2
Writes verification/paper/attractors_bifurcation.json and .npz (arrays for a
figure: steady peak-to-peak per depth and IC, final states, per-revolution
peak-to-peak histories, once-per-tooth samples (x, x') of the last 400 tooth periods).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import attractors_common as AC

HERE = AC.HERE
SPEEDS = (1090.0, 1420.0, 2500.0, 2870.0)
G = 1500.0
N_DEPTH = 28
HI = 1.05
NREV = 1600
KICK_REV = 100
STEADY = 300
EDGE_REVS = 40
D1_WIN = 300
N_POINCARE = 400
ICS = ("face", "kick0.3", "kick0.03", "step")
N_REFINE = 5
REFINE_ORDER = (4, 3, 2, 5, 6, 1, 0)     # indices into BAND_DV: 0.3, 0.095, 0.03, 0.95, 3, 0.0095, 0.003 m/s
STAT_TOL = 0.10


def ic_options(ic):
    """simulate() options of an initial condition ('kick<dv>' for any kick size)."""
    if ic == "step":
        return {}
    if ic == "face":
        return dict(edge=True)
    assert ic.startswith("kick")
    return dict(edge=True, kick_step=KICK_REV * AC.SPR, kick_dv=float(ic[4:]))


def run_ic(args):
    """One run: args = (rpm, a, ic).  Final state, statistics, D1 verdict, histories."""
    rpm, a, ic = args
    AC.setup()
    from tdsim import simulate
    from tdboundary import classify_run
    t0 = time.time()
    opts = ic_options(ic)
    r = simulate(rpm, a, G, vmax=AC.VMAX, store=True, nrev=NREV, nreg=0, **opts)
    dstep = AC.divergence_step(r)
    # D1 over the manuscript's 300-revolution window for this IC
    if ic == "step":
        w0 = 0
    elif ic == "face":
        w0 = EDGE_REVS * AC.SPR
    else:
        ks = opts["kick_step"]
        w0 = (ks // AC.ND + 1) * AC.ND
    w1 = (KICK_REV + D1_WIN) * AC.SPR + 1 if ic.startswith("kick") else w0 + D1_WIN * AC.SPR + 1
    d1 = not classify_run(r["x"][w0:w1], AC.SPR, dstep is not None and dstep < w1)["stable"]
    pp_rev, _ = AC.per_rev(r, 0, NREV)
    out = dict(rpm=rpm, a=a, ic=ic, d1_chatter=bool(d1), diverged=dstep is not None,
               diverged_rev=None if dstep is None else dstep / AC.SPR,
               max_x_mm=float(r["max_x"] * 1e3))
    px = pv = np.full(N_POINCARE, np.nan)
    if dstep is not None:
        out["state"] = "diverged"
    else:
        i0, i1 = (NREV - STEADY) * AC.SPR, NREV * AC.SPR
        s = AC.window_stats(r, i0, i1, rpm)
        out.update({k: s[k] for k in ("pp_x", "rms_x", "pp_V", "rms_V", "clip_frac", "res_tooth",
                                      "nonsync_frac", "f_dom", "f_chatter", "f_tooth")})
        if s["res_tooth"] < AC.RET_TOL:
            out["state"] = "forced"
        else:
            x = r["x"]
            pp_a, pp_b = np.ptp(x[i0:i0 + 100 * AC.SPR]), np.ptp(x[i1 - 100 * AC.SPR:i1])
            h = (i0 + i1) // 2
            rms_a, rms_b = np.std(x[i0:h]), np.std(x[h:i1])
            out["pp_ratio_last_first_100rev"] = float(pp_b / pp_a)
            out["rms_ratio_halves"] = float(rms_b / rms_a)
            stat = abs(pp_b / pp_a - 1) <= STAT_TOL and abs(rms_b / rms_a - 1) <= STAT_TOL
            out["state"] = "chatter" if stat else "transient"
        idx = NREV * AC.SPR - AC.ND * np.arange(N_POINCARE, 0, -1)
        px, pv = r["x"][idx].copy(), r["v"][idx].copy()
    out["runtime_s"] = round(time.time() - t0, 2)
    return dict(res=out, pp_rev=pp_rev.astype(np.float32), px=px, pv=pv)


def refine(args):
    """Bisection of the lower end of the bounded high branch at one speed:
    args = (rpm, a_lo, a_hi), a_lo: no IC of the grid ended in chatter, a_hi: one did."""
    rpm, lo, hi = args
    AC.setup()
    from c1_unsafe_zone import BAND_DV
    tried = []
    for _ in range(N_REFINE):
        mid = 0.5 * (lo + hi)
        found = False
        for j in REFINE_ORDER:
            res = run_ic((rpm, mid, f"kick{float(BAND_DV[j])!r}"))["res"]
            tried.append({k: res.get(k) for k in ("a", "ic", "state", "pp_x", "d1_chatter", "diverged",
                                                  "clip_frac", "f_chatter")})
            if res["state"] == "chatter":
                found = True
                break
        if found:
            hi = mid
        else:
            lo = mid
    return dict(rpm=rpm, a_lo=lo, a_hi=hi, runs=tried)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    AC.setup()
    from zoa import a_lim
    from c1_unsafe_zone import BAND_DV
    al0 = np.array([a_lim(n, 0.0) for n in SPEEDS])
    al = np.array([a_lim(n, G) for n in SPEEDS])
    depths = np.array([np.linspace(al0[i], HI * al[i], N_DEPTH) for i in range(len(SPEEDS))])
    jobs = [(n, float(depths[i, j]), ic) for i, n in enumerate(SPEEDS) for j in range(N_DEPTH) for ic in ICS]
    # longest first (chattering, deep cuts), for a balanced pool
    order = sorted(range(len(jobs)), key=lambda k: -jobs[k][1] / al[SPEEDS.index(jobs[k][0])])
    print(f"{len(jobs)} grid runs", flush=True)
    shape = (len(SPEEDS), N_DEPTH, len(ICS))
    res = np.empty(shape, dtype=object)
    pp_rev = np.full(shape + (NREV,), np.nan, dtype=np.float32)
    px = np.full(shape + (N_POINCARE,), np.nan)
    pv = np.full_like(px, np.nan)
    with Pool(a.workers) as pool:
        for k, out in enumerate(pool.imap_unordered(run_ic, [jobs[i] for i in order], chunksize=1)):
            o = out["res"]
            i, j, c = SPEEDS.index(o["rpm"]), int(np.argmin(np.abs(depths[SPEEDS.index(o["rpm"])] - o["a"]))), \
                ICS.index(o["ic"])
            res[i, j, c] = o
            pp_rev[i, j, c], px[i, j, c], pv[i, j, c] = out["pp_rev"], out["px"], out["pv"]
            if (k + 1) % 16 == 0 or k + 1 == len(jobs):
                print(f"[{k + 1}/{len(jobs)}] {time.time() - t0:.0f} s; last: {o['rpm']:.0f} rpm "
                      f"{o['a'] * 1e3:.2f} mm {o['ic']}: {o['state']}", flush=True)
    state = np.vectorize(lambda o: o["state"])(res)
    # lower end of the bounded high branch on the grid, then bisection
    ref_jobs, grid_end = [], {}
    for i, n in enumerate(SPEEDS):
        ch = np.nonzero(np.any(state[i] == "chatter", axis=1))[0]
        grid_end[n] = int(ch[0]) if ch.size else None
        if ch.size and ch[0] > 0:
            ref_jobs.append((n, float(depths[i, ch[0] - 1]), float(depths[i, ch[0]])))
    print("refining", [(n, round(lo * 1e3, 3), round(hi * 1e3, 3)) for n, lo, hi in ref_jobs], flush=True)
    with Pool(min(a.workers, max(1, len(ref_jobs)))) as pool:
        refined = pool.map(refine, ref_jobs, chunksize=1)
    refined = {r["rpm"]: r for r in refined}
    # per-speed digest
    speeds = []
    for i, n in enumerate(SPEEDS):
        st = state[i]
        rows = []
        for j in range(N_DEPTH):
            rows.append(dict(a_mm=depths[i, j] * 1e3, a_over_alim=depths[i, j] / al[i],
                             a_over_alim0=depths[i, j] / al0[i],
                             runs={ICS[c]: {k: v for k, v in res[i, j, c].items() if k not in ("rpm", "a", "ic")}
                                   for c in range(len(ICS))}))
        forced_face = [j for j in range(N_DEPTH) if st[j, ICS.index("face")] == "forced"]
        high = [j for j in range(N_DEPTH) if np.any(st[j] == "chatter")]
        bist = [j for j in high if st[j, ICS.index("face")] == "forced"]
        div = [j for j in range(N_DEPTH) if np.any(st[j] == "diverged")]
        trans = [j for j in range(N_DEPTH) if np.any(st[j] == "transient")]
        step_ch = [j for j in range(N_DEPTH) if st[j, ICS.index("step")] != "forced"]
        hp = [res[i, j, c]["pp_x"] for j in high for c in range(len(ICS)) if st[j, c] == "chatter"]
        holes = [j for j in range(min(high), max(high) + 1) if j not in high] if high else []
        rf = refined.get(n)
        d = dict(
            rpm=n, g=G, a_lim0_mm=al0[i] * 1e3, a_lim_mm=al[i] * 1e3, depths_mm=depths[i] * 1e3,
            low_branch_top_mm=depths[i, max(forced_face)] * 1e3 if forced_face else None,
            low_branch_first_unstable_mm=(depths[i, max(forced_face) + 1] * 1e3
                                          if forced_face and max(forced_face) + 1 < N_DEPTH else None),
            high_branch_grid_end_mm=depths[i, high[0]] * 1e3 if high else None,
            high_branch_grid_below_mm=depths[i, high[0] - 1] * 1e3 if high and high[0] > 0 else None,
            high_branch_refined_mm=[rf["a_lo"] * 1e3, rf["a_hi"] * 1e3] if rf else None,
            high_branch_holes_mm=[depths[i, j] * 1e3 for j in holes],
            bistable_grid_mm=[depths[i, j] * 1e3 for j in bist],
            bistable_range_mm=[depths[i, bist[0]] * 1e3, depths[i, bist[-1]] * 1e3] if bist else None,
            bistable_range_over_alim=[depths[i, bist[0]] / al[i], depths[i, bist[-1]] / al[i]] if bist else None,
            diverged_depths_mm={f"{depths[i, j] * 1e3:.3f}": [ICS[c] for c in range(len(ICS)) if st[j, c] == "diverged"]
                                for j in div},
            transient_depths_mm={f"{depths[i, j] * 1e3:.3f}": [ICS[c] for c in range(len(ICS)) if st[j, c] == "transient"]
                                 for j in trans},
            step_chatter_first_mm=depths[i, step_ch[0]] * 1e3 if step_ch else None,
            step_forced_last_mm=(depths[i, step_ch[0] - 1] * 1e3 if step_ch and step_ch[0] > 0 else None),
            high_branch_pp_x_mm=[min(hp) * 1e3, max(hp) * 1e3] if hp else None,
            d1_vs_final_disagreements=[dict(a_mm=depths[i, j] * 1e3, ic=ICS[c], d1_chatter=res[i, j, c]["d1_chatter"],
                                            state=st[j, c])
                                       for j in range(N_DEPTH) for c in range(len(ICS))
                                       if res[i, j, c]["d1_chatter"] != (st[j, c] != "forced")],
            refine_runs=rf["runs"] if rf else [],
            depth_rows=rows)
        speeds.append(d)
    # a compact table of states (rows: depth, columns: ICs)
    out = dict(note=__doc__, zeta=AC.ZETA, build=AC.build_id(),
               protocol=dict(g=G, vmax=AC.VMAX, nreg=0, n_depth=N_DEPTH, hi=HI, nrev=NREV, kick_rev=KICK_REV,
                             steady_revs=STEADY, ics=ICS, return_tol_m=AC.RET_TOL, stationarity_tol=STAT_TOL,
                             d1_window=D1_WIN, n_refine=N_REFINE,
                             refine_kicks_m_s=[float(BAND_DV[j]) for j in REFINE_ORDER]),
               speeds=speeds, runtime_s=round(time.time() - t0, 1), workers=a.workers)
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_bifurcation.json"), "w"), indent=1)
    getf = np.vectorize(lambda o, k: np.nan if o.get(k) is None else float(o[k]), excluded={1})
    np.savez_compressed(
        os.path.join(HERE, "attractors_bifurcation.npz"),
        rpm=np.array(SPEEDS), g=G, a_lim0=al0, a_lim=al, depths=depths, ics=np.array(ICS), state=state,
        pp_x=getf(res, "pp_x"), rms_x=getf(res, "rms_x"), rms_V=getf(res, "rms_V"),
        clip_frac=getf(res, "clip_frac"), f_chatter=getf(res, "f_chatter"), res_tooth=getf(res, "res_tooth"),
        d1_chatter=np.vectorize(lambda o: bool(o["d1_chatter"]))(res),
        diverged_rev=getf(res, "diverged_rev"), pp_rev=pp_rev, poincare_x=px, poincare_v=pv,
        refine_lo=np.array([refined[n]["a_lo"] if n in refined else np.nan for n in SPEEDS]),
        refine_hi=np.array([refined[n]["a_hi"] if n in refined else np.nan for n in SPEEDS]),
        zeta=AC.ZETA, build=AC.build_id(),
        note="depths [m] (speed, depth); arrays (speed, depth, IC) with ICs face, kick0.3, kick0.03, step; "
             "pp_x: peak-to-peak of x over the last 300 of 1600 revolutions [m] (NaN: diverged); "
             "pp_rev: per-revolution peak-to-peak of x [m]; poincare_x/v: x [m], x' [m/s] at the last 400 "
             "tooth-period boundaries; refine_lo/hi: bisected lower end of the bounded high branch [m]")
    for d in speeds:
        print(f"{d['rpm']:.0f} rpm: a_lim0 {d['a_lim0_mm']:.2f}, a_lim {d['a_lim_mm']:.2f} mm; low branch top "
              f"{d['low_branch_top_mm']}; high branch from {d['high_branch_grid_end_mm']} (refined "
              f"{d['high_branch_refined_mm']}); bistable {d['bistable_range_mm']}; diverged {d['diverged_depths_mm']}; "
              f"transient {d['transient_depths_mm']}; step chatter from {d['step_chatter_first_mm']}")
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
