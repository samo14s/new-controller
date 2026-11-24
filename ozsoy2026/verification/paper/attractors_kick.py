"""Where do kicks outside the basin of the forced orbit end?  (C1 basin points.)

Purpose.  Section 4 of the manuscript calls the clipped cut 'bistable' where a
kick larger than the critical kick dv* ends in chatter, 'bounded or
divergent', as judged by the D1 classifier over the 300 revolutions after the
kick.  That shows that the forced orbit has a bounded basin; it does not show
a second bounded attractor.  This script follows the trajectories for 2000
revolutions after the kick and classifies where they end:
  return                 back on the forced orbit: tooth-periodic in the last
                         quarter, max |x(t) - x(t - T)| < 0.01 fz
  bounded_stationary     bounded chatter whose statistics in the second and in
                         the last quarter agree (RMS of x and of V within 10 %,
                         clipped fraction within 0.05, chatter frequency within 1 Hz)
  bounded_nonstationary  bounded, off the forced orbit, but those statistics differ
  diverged               |x| > 1 cm (the simulator's abort): no bounded attractor reached

Protocol (that of results/c1_basin.npz, src/c1_unsafe_zone._critical_kick):
  the 48 clipped points of c1_basin.npz (vmax = 9 V, g = 1500 V s/m; 1090,
  1420, 2500 and 2870 rpm; 12 depths 0.15-0.98 a_lim(rpm, 1500)); points with no
  finite stored critical kick (no kick of the grid up to 3 m/s ended in chatter)
  are skipped.  Worst phase p* = argmin over the 16 phases of the stored critical
  kick; kicks of 2, 2.5 and 3 times dv*(p*), capped at 3 m/s, added to x' at step
  KICK_REV * 512 + p* * 128 // 16 (revolution 100) of a face entry (edge), with
  the full surface memory (nreg = 0), the 9 V clip, g = 1500 and the damping of
  results/gainmap_main.npz (1.34 %).  The run lasts NPOST = 2000 revolutions after
  the first tooth-period boundary st after the kick (nrev = 2101).  Windows
  after st: Q2 = revolutions 500-1000, Q4 = 1500-2000.  Also stored: the D1
  verdict on the 300 revolutions after the kick (the window of the stored
  critical kick, x[st : 400 * 512 + 1]), the forced orbit before the kick
  (revolution 99), the revolution after which the run is back on the forced
  orbit, the transient peak, and once-per-tooth Poincare samples (x, x') at the
  tooth-period boundaries, of the last 500 tooth periods of every run.
  Long check: the four 0.9 a_lim points with the 2x kick are continued to 4000
  revolutions after the kick; their last quarter (3000-4000) is compared with Q4.

Run (2 worker processes, about 7 min):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_kick.py --workers 2
Writes verification/paper/attractors_kick.json (per point and kick) and
attractors_kick.npz (Poincare samples, per-revolution peak-to-peak of x).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import attractors_common as AC

HERE = AC.HERE
MULTS = (2.0, 2.5, 3.0)
DV_CAP = 3.0
NPOST = 2000
NPOST_LONG = 4000
N_POINCARE = 500
STAT_TOL = dict(rms_x=0.10, rms_V=0.10, clip_frac=0.05, f_chatter_hz=1.0)


def _protocol():
    from c1_unsafe_zone import KICK_REV, WINDOW, N_PHASE
    return KICK_REV, WINDOW, N_PHASE


def classify(r, rpm, st, npost):
    """Outcome and statistics of one kicked run (st: first tooth boundary after the kick)."""
    from tdboundary import classify_run
    KICK_REV, WINDOW, _ = _protocol()
    out = dict()
    dstep = AC.divergence_step(r)
    # D1 verdict over the 300 revolutions after the kick (the stored critical-kick window)
    end_d1 = (KICK_REV + WINDOW) * AC.SPR + 1
    div_d1 = dstep is not None and dstep < end_d1
    out["d1_chatter_300rev"] = not classify_run(r["x"][st:end_d1], AC.SPR, div_d1)["stable"]
    pp_rev, res_rev = AC.per_rev(r, st, npost)
    post = r["x"][st:st + npost * AC.SPR + 1]
    out["max_x_after_kick_mm"] = float(np.max(np.abs(post)) * 1e3)
    out["transient_peak_mm"] = float(np.max(np.abs(r["x"][st:st + 200 * AC.SPR])) * 1e3)
    out["diverged"] = dstep is not None
    out["diverged_rev_after_kick"] = None if dstep is None else (dstep - st) / AC.SPR
    if dstep is not None:
        out["outcome"] = "diverged"
        return out, pp_rev
    q = npost // 4
    win = {name: AC.window_stats(r, st + k0 * q * AC.SPR, st + (k0 + 1) * q * AC.SPR, rpm)
           for name, k0 in (("Q2", 1), ("Q4", 3))}
    out.update(win)
    out["return_rev_after_kick"] = AC.return_rev(res_rev)
    if win["Q4"]["res_tooth"] < AC.RET_TOL:
        out["outcome"] = "return"
        return out, pp_rev
    a, b = win["Q2"], win["Q4"]
    diff = dict(rms_x=abs(b["rms_x"] / a["rms_x"] - 1), rms_V=abs(b["rms_V"] / a["rms_V"] - 1),
                clip_frac=abs(b["clip_frac"] - a["clip_frac"]),
                f_chatter_hz=abs(b["f_chatter"] - a["f_chatter"]))
    out["q2_q4_difference"] = diff
    stat = all(diff[k] <= STAT_TOL[k] for k in STAT_TOL)
    out["outcome"] = "bounded_stationary" if stat else "bounded_nonstationary"
    # regularity of the bounded motion: spread of the per-revolution peak-to-peak in Q4
    pq = pp_rev[3 * q:]
    out["Q4_pp_rev_cv"] = float(np.std(pq) / np.mean(pq))
    return out, pp_rev


def run_one(args):
    """One kicked run: args = (row, rpm, a, g, phase, kick, npost)."""
    row, rpm, a, g, phase, kick, npost = args
    AC.setup()
    from tdsim import simulate
    KICK_REV, _, N_PHASE = _protocol()
    t0 = time.time()
    ks = KICK_REV * AC.SPR + (phase * AC.ND) // N_PHASE
    st = (ks // AC.ND + 1) * AC.ND
    r = simulate(rpm, a, g, vmax=AC.VMAX, store=True, nrev=KICK_REV + npost + 1, edge=True,
                 kick_step=ks, kick_dv=kick, nreg=0)
    out, pp_rev = classify(r, rpm, st, npost)
    # forced orbit before the kick (revolution 99)
    i0 = (KICK_REV - 1) * AC.SPR
    out["pre_kick"] = AC.window_stats(r, i0, i0 + AC.SPR, rpm)
    ipre = (ks // AC.ND - 1) * AC.ND
    pre = (float(r["x"][ipre]), float(r["v"][ipre]))
    if out["diverged"]:
        px = pv = np.full(N_POINCARE, np.nan)
    else:
        idx = st + AC.ND * np.arange(npost * 4 - N_POINCARE, npost * 4)
        px, pv = r["x"][idx].copy(), r["v"][idx].copy()
    out["runtime_s"] = round(time.time() - t0, 2)
    return dict(key=(row, kick, npost), res=out, px=px, pv=pv, pre=pre,
                pp_rev=pp_rev.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    AC.setup()
    from zoa import a_lim
    KICK_REV, WINDOW, N_PHASE = _protocol()
    d = np.load(os.path.join(AC.RES, "c1_basin.npz"))
    assert abs(float(d["zeta"]) - AC.ZETA) < 1e-12 and int(d["nreg"]) == 0
    rows = [i for i in range(len(d["rpm"])) if d["vmax"][i] > 0]
    assert len(rows) == 48
    points, jobs = [], []
    for i in rows:
        rpm, g, frac = float(d["rpm"][i]), float(d["g"][i]), float(d["frac"][i])
        al = a_lim(rpm, g)
        assert abs(al * 1e3 - d["a_lim"][i]) < 1e-6 * d["a_lim"][i]
        dv = d["dv"][i]
        fin = np.isfinite(dv)
        p = dict(row=i, rpm=rpm, g=g, frac=frac, a_mm=frac * al * 1e3, a_lim_mm=al * 1e3,
                 n_finite_phases=int(fin.sum()), n_inf_phases=int(np.isinf(dv).sum()),
                 n_nan_phases=int(np.isnan(dv).sum()))
        if not fin.any():
            p.update(skipped=True, reason="no finite stored critical kick (no kick up to 3 m/s "
                                          "ended in chatter at any of the 16 phases)")
            points.append(p)
            continue
        ph = int(np.argmin(np.where(fin, dv, np.inf)))
        p.update(skipped=False, worst_phase=ph, dv_star_worst_mm_s=float(dv[ph] * 1e3),
                 dv_star_median_mm_s=float(np.median(dv) * 1e3), runs=[])
        for m in MULTS:
            kick = min(m * dv[ph], DV_CAP)
            p["runs"].append(dict(mult=m, kick_m_s=float(kick), capped=bool(m * dv[ph] > DV_CAP)))
            jobs.append((i, rpm, frac * al, g, ph, float(kick), NPOST))
        points.append(p)
    long_pts = [p for p in points if not p["skipped"] and abs(p["frac"] - 0.9) < 1e-9]
    for p in long_pts:
        kick = min(2.0 * p["dv_star_worst_mm_s"] * 1e-3, DV_CAP)
        jobs.append((p["row"], p["rpm"], p["a_mm"] * 1e-3, p["g"], p["worst_phase"], float(kick), NPOST_LONG))
    print(f"{len(jobs)} runs ({sum(1 for p in points if p['skipped'])} of 48 points skipped)", flush=True)
    results = {}
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            results[res["key"]] = res
            o = res["res"]
            print(f"[{k + 1}/{len(jobs)}] row {res['key'][0]} kick {res['key'][1] * 1e3:.2f} mm/s "
                  f"npost {res['key'][2]}: {o['outcome']}" +
                  (f", Q4 pp x {o['Q4']['pp_x'] * 1e3:.3f} mm" if "Q4" in o else "") +
                  f" ({o['runtime_s']:.1f} s)", flush=True)
    # assemble
    act = [p for p in points if not p["skipped"]]
    n_m = len(MULTS)
    px = np.full((len(act), n_m, N_POINCARE), np.nan)
    pv = np.full_like(px, np.nan)
    pre = np.full((len(act), 2), np.nan)
    env = np.full((len(act), n_m, NPOST), np.nan, dtype=np.float32)
    for ia, p in enumerate(act):
        for im, run in enumerate(p["runs"]):
            res = results[(p["row"], run["kick_m_s"], NPOST)]
            run.update(res["res"])
            px[ia, im], pv[ia, im], env[ia, im] = res["px"], res["pv"], res["pp_rev"]
            pre[ia] = res["pre"]
    long_runs, lpx, lpv = [], [], []
    for p in long_pts:
        kick = min(2.0 * p["dv_star_worst_mm_s"] * 1e-3, DV_CAP)
        res = results[(p["row"], float(kick), NPOST_LONG)]
        short = next(r for r in p["runs"] if r["mult"] == 2.0)
        lr = dict(row=p["row"], rpm=p["rpm"], frac=p["frac"], kick_m_s=float(kick), npost=NPOST_LONG,
                  outcome=res["res"]["outcome"], last_quarter=res["res"].get("Q4"),
                  q4_of_2000rev_run=short.get("Q4"), runtime_s=res["res"]["runtime_s"])
        if lr["last_quarter"] and lr["q4_of_2000rev_run"]:
            a4, b4 = lr["q4_of_2000rev_run"], lr["last_quarter"]
            lr["rel_change_rms_x"] = b4["rms_x"] / a4["rms_x"] - 1
            lr["rel_change_rms_V"] = b4["rms_V"] / a4["rms_V"] - 1
            lr["change_f_chatter_hz"] = b4["f_chatter"] - a4["f_chatter"]
        long_runs.append(lr)
        lpx.append(res["px"])
        lpv.append(res["pv"])
    # summary
    allruns = [(p, r) for p in act for r in p["runs"]]
    counts = {}
    for _, r in allruns:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    per_point = {}
    for p in act:
        oc = sorted(set(r["outcome"] for r in p["runs"]))
        per_point[f"{p['rpm']:.0f} rpm {p['frac']:.2f}"] = oc
    bnd = [(p, r) for p, r in allruns if r["outcome"].startswith("bounded")]

    def rng(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return [float(min(vals)), float(max(vals))] if vals else None

    summ = dict(
        n_points=48, n_skipped=sum(1 for p in points if p["skipped"]), n_points_run=len(act),
        n_runs=len(allruns), outcome_counts=counts, outcomes_per_point=per_point,
        d1_chatter_300rev_count=sum(1 for _, r in allruns if r["d1_chatter_300rev"]),
        bounded_Q4_pp_x_mm=rng([r["Q4"]["pp_x"] * 1e3 for _, r in bnd]),
        bounded_Q4_rms_x_mm=rng([r["Q4"]["rms_x"] * 1e3 for _, r in bnd]),
        bounded_Q4_pp_V=rng([r["Q4"]["pp_V"] for _, r in bnd]),
        bounded_Q4_rms_V=rng([r["Q4"]["rms_V"] for _, r in bnd]),
        bounded_Q4_clip_frac=rng([r["Q4"]["clip_frac"] for _, r in bnd]),
        bounded_Q4_f_chatter_hz=rng([r["Q4"]["f_chatter"] for _, r in bnd]),
        bounded_Q4_nonsync_frac=rng([r["Q4"]["nonsync_frac"] for _, r in bnd]),
        transient_peak_mm=rng([r["transient_peak_mm"] for _, r in allruns]),
        per_speed={},
    )
    for n in sorted(set(p["rpm"] for p in act)):
        b = [(p, r) for p, r in bnd if p["rpm"] == n]
        summ["per_speed"][f"{n:.0f}"] = dict(
            fracs_run=[p["frac"] for p in act if p["rpm"] == n],
            fracs_skipped=[p["frac"] for p in points if p["skipped"] and p["rpm"] == n],
            outcomes=sorted(set(r["outcome"] for p, r in allruns if p["rpm"] == n)),
            Q4_pp_x_mm=rng([r["Q4"]["pp_x"] * 1e3 for _, r in b]),
            Q4_rms_V=rng([r["Q4"]["rms_V"] for _, r in b]),
            Q4_clip_frac=rng([r["Q4"]["clip_frac"] for _, r in b]),
            Q4_f_chatter_hz=rng([r["Q4"]["f_chatter"] for _, r in b]),
            # the same attractor from the three kicks: largest relative spread of the Q4 RMS of x
            max_rel_spread_rms_x_over_kicks=max(
                [float(np.ptp([r["Q4"]["rms_x"] for r in p["runs"] if "Q4" in r]) /
                       np.mean([r["Q4"]["rms_x"] for r in p["runs"] if "Q4" in r]))
                 for p in act if p["rpm"] == n and all("Q4" in r for r in p["runs"])] or [np.nan]))
    out = dict(note=__doc__, zeta=AC.ZETA, build=AC.build_id(),
               protocol=dict(kick_rev=KICK_REV, d1_window=WINDOW, n_phase=N_PHASE, mults=MULTS,
                             dv_cap_m_s=DV_CAP, npost=NPOST, npost_long=NPOST_LONG, nreg=0,
                             vmax=AC.VMAX, g=1500.0, entry="face (edge)", spr=AC.SPR,
                             return_tol_m=AC.RET_TOL, stationarity_tol=STAT_TOL,
                             n_poincare=N_POINCARE, source="results/c1_basin.npz rows with vmax > 0"),
               summary=summ, long_runs=long_runs, points=points,
               runtime_s=round(time.time() - t0, 1), workers=a.workers)
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_kick.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(HERE, "attractors_kick.npz"),
        rows=np.array([p["row"] for p in act]), rpm=np.array([p["rpm"] for p in act]),
        frac=np.array([p["frac"] for p in act]), a_mm=np.array([p["a_mm"] for p in act]),
        worst_phase=np.array([p["worst_phase"] for p in act]),
        dv_star=np.array([p["dv_star_worst_mm_s"] * 1e-3 for p in act]), mults=np.array(MULTS),
        kick=np.array([[r["kick_m_s"] for r in p["runs"]] for p in act]),
        outcome=np.array([[r["outcome"] for r in p["runs"]] for p in act]),
        poincare_x=px, poincare_v=pv, pre_kick_xv=pre, pp_rev_after_kick=env,
        long_rows=np.array([p["row"] for p in long_pts]), long_poincare_x=np.array(lpx),
        long_poincare_v=np.array(lpv), zeta=AC.ZETA, build=AC.build_id(),
        note="Poincare samples (x [m], x' [m/s]) at tooth-period boundaries, last 500 tooth periods "
             "of the 2000 revolutions after the kick (long_*: of 4000); pp_rev_after_kick: peak-to-peak "
             "of x per revolution after the kick [m]; pre_kick_xv: forced-orbit Poincare point")
    print(json.dumps(AC.jsonable(summ), indent=1))
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
