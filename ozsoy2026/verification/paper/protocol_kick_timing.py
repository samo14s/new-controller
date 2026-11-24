"""Kick timing: critical kick with the kick at revolution 300 instead of 100.

Reviewer's question: in the C1 kick protocol (src/c1_unsafe_zone.py) the cut is
settled by a face entry (full engagement at revolution 40) and kicked at
revolution 100.  At 0.98 a_lim the slowest mode decays by only 0.985-0.994
per tooth period, so a residual of the entry transient may still be present
at the kick and bias the critical kick.  This script

  (a) measures that residual on the unkicked run (face entry, g = 1500 V s/m,
      |V| <= 9 V, full surface memory nreg = 0, 600 revolutions): per tooth
      period k the largest |x(t) - x(t - T)| and |x'(t) - x'(t - T)|
      (difference between consecutive tooth periods), the classifier's
      once-per-tooth sample e_k = x(t_k) - x(t_k - T), and the deviation from
      the settled orbit, taken as the last tooth period of the run; reported
      for the tooth period that ends at the kick of phase 0 in revolutions
      40 (full engagement), 100 (stored protocol) and 300 (this check), and
      the full per-tooth series in protocol_kick_timing.npz;
  (b) checks that protocol_common.critical_kick reproduces the stored value
      of results/c1_basin.npz exactly with the stored protocol (kick at
      revolution 100, phase 0 of 16, window 300 revolutions);
  (c) recomputes the critical kick at the four C1 speeds (1090, 1420, 2500,
      2870 rpm) at 0.98 a_lim(rpm, 1500), phases 0, 4, 8 and 12 of 16, with
      the kick at revolution 300 and the window after the kick unchanged (300
      revolutions, i.e. runs of 600 revolutions), and compares it with the
      stored values at revolution 100.

Everything else is the stored protocol (DV_GRID upward search from 0.1 mm/s
to 3 m/s, 6 bisections in log dv, classifier D1 of tdboundary.classify_run).

Run (one process, about 10 min on a free core; resumable: units already in
the JSON are skipped, --fresh starts over):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_kick_timing.py
Writes verification/paper/protocol_kick_timing.json (after every unit) and
protocol_kick_timing.npz (residual series).
"""
import argparse
import os
import time

import numpy as np

import protocol_common as PC

OUT = os.path.join(PC.HERE, "protocol_kick_timing.json")
NPZ = os.path.join(PC.HERE, "protocol_kick_timing.npz")
FRAC = 0.98
PHASES = (0, 4, 8, 12)              # of 16
KICK_REV_OLD, KICK_REV_NEW, WINDOW = 100, 300, 300
NREV_RES = 600
REVS_REPORTED = (40, 100, 300)


def residual(rpm):
    """Unkicked run: residual of the entry at the kick times."""
    a = PC.depth(rpm, FRAC)
    t0 = time.time()
    r = PC.simulate(rpm, a, PC.G_DESIGN, vmax=PC.VMAX, store=True, nrev=NREV_RES, edge=True,
                    nreg=PC.NREG)
    assert not r["diverged"]
    x, v = r["x"], r["v"]
    K = NREV_RES * PC.SPR // PC.ND                     # tooth periods in the run
    X = x[:K * PC.ND].reshape(K, PC.ND)
    V = v[:K * PC.ND].reshape(K, PC.ND)
    # tooth period k covers steps k*ND ... (k+1)*ND - 1; diff row k-1 = period k minus period k-1
    dX = np.r_[np.nan, np.abs(np.diff(X, axis=0)).max(axis=1)]
    dV = np.r_[np.nan, np.abs(np.diff(V, axis=0)).max(axis=1)]
    ek = np.r_[np.nan, np.diff(x[::PC.ND][:K + 1])][:K + 1]     # e_k = x(k T) - x((k-1) T), k = 0..K
    devX = np.abs(X - X[-1]).max(axis=1)
    devV = np.abs(V - V[-1]).max(axis=1)
    peakV = np.abs(V).max(axis=1)
    cmd = PC.G_DESIGN * np.abs(V).max(axis=1)                  # |g x'| per tooth period (V)
    rows = {}
    for rev in REVS_REPORTED:
        k = rev * PC.SPR // PC.ND                              # the kick of phase 0 is at the start of period k
        kp = k - 1                                              # the period that ends at the kick
        rows[f"rev{rev}"] = dict(
            tooth_period_index=kp,
            dx_consecutive_um=float(dX[kp] * 1e6), dv_consecutive_mmps=float(dV[kp] * 1e3),
            e_k_at_kick_um=float(abs(ek[k]) * 1e6),
            dev_x_from_orbit_um=float(devX[kp] * 1e6), dev_v_from_orbit_mmps=float(devV[kp] * 1e3),
            orbit_peak_v_mmps=float(peakV[-1] * 1e3), orbit_peak_cmd_V=float(cmd[-1]),
            peak_cmd_V_this_period=float(cmd[kp]))
    # the stored protocol's unkicked window (revolutions 100-400) and the new one (300-600),
    # judged by the classifier: both must be stable for the critical kick to be defined
    win = {}
    for rev in (KICK_REV_OLD, KICK_REV_NEW):
        st = rev * PC.SPR + PC.ND                               # first tooth boundary after a phase-0 kick
        end = (rev + WINDOW) * PC.SPR + 1
        win[f"rev{rev}"] = PC._verdict(x[st:end], False)
    # decay of the residual between the kick times, against the slowest-mode decay of c1_basin
    s = PC.stored_row(rpm, FRAC)
    k100, k300 = KICK_REV_OLD * 4 - 1, KICK_REV_NEW * 4 - 1
    arrays = dict(dX=dX, dV=dV, devX=devX, devV=devV, ek=ek, cmd=cmd)
    return dict(a_mm=a * 1e3, frac=FRAC, rows=rows, unkicked_window=win,
                ratio_dv_consecutive_300_over_100=float(dV[k300] / dV[k100]),
                stored_slowest_mode_decay_per_tooth=s["decay"],
                expected_ratio_from_decay=float(s["decay"] ** (k300 - k100)),
                seconds=time.time() - t0), arrays


def crit(rpm, phase, kick_rev):
    a = PC.depth(rpm, FRAC)
    t0 = time.time()
    val, trace = PC.critical_kick(rpm, a, PC.G_DESIGN, PC.VMAX, phase, 16, kick_rev, WINDOW)
    lo, hi = PC.bracket(trace, val)
    stored = PC.stored_row(rpm, FRAC)["dv"][phase]
    return dict(rpm=rpm, frac=FRAC, phase_of_16=phase, kick_rev=kick_rev, window=WINDOW,
                value_mmps=val * 1e3, bracket_mmps=[lo * 1e3, hi * 1e3],
                stored_rev100_mmps=float(stored * 1e3),
                ratio_to_stored=float(val / stored) if np.isfinite(stored) else None,
                identical_to_stored=bool(val == stored),
                n_runs=len(trace), seconds=time.time() - t0, trace=trace)


def summarize(out):
    S = {}
    for n in PC.SPEEDS:
        key = f"{n:.0f}"
        new = [out["units"].get(f"kick300_{key}_{p}") for p in PHASES]
        if any(u is None for u in new):
            continue
        v_new = np.array([u["value_mmps"] for u in new])
        v_old = np.array([u["stored_rev100_mmps"] for u in new])
        S[key] = dict(phases_of_16=list(PHASES), rev300_mmps=v_new.tolist(), rev100_stored_mmps=v_old.tolist(),
                      ratio=(v_new / v_old).tolist(), min_rev300_mmps=float(v_new.min()),
                      min_rev100_mmps=float(v_old.min()),
                      ratio_of_minima=float(v_new.min() / v_old.min()),
                      max_abs_rel_change=float(np.max(np.abs(v_new / v_old - 1))),
                      stored_min_over_16_phases_mmps=float(PC.stored_row(n, FRAC)["dv"].min() * 1e3))
    if S:
        allr = np.concatenate([S[k]["ratio"] for k in S])
        S["all"] = dict(n_points=len(allr), ratio_min=float(allr.min()), ratio_max=float(allr.max()),
                        ratio_median=float(np.median(allr)),
                        max_abs_rel_change=float(np.max(np.abs(allr - 1))))
    rep = [out["units"].get(f"repro_{n:.0f}") for n in PC.SPEEDS]
    if all(u is not None for u in rep):
        S["reproduction_all_identical"] = bool(all(u["identical_to_stored"] for u in rep))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    out = None if args.fresh else PC.load_json(OUT)
    if out is None:
        out = dict(PC.header(__file__, __doc__),
                   protocol=dict(frac=FRAC, phases_of_16=list(PHASES), kick_rev_stored=KICK_REV_OLD,
                                 kick_rev_new=KICK_REV_NEW, window_after_kick=WINDOW,
                                 entry="face (edge=True), full engagement at revolution 40",
                                 residual_run_revs=NREV_RES),
                   units={})
    arrays = dict(np.load(NPZ)) if (os.path.exists(NPZ) and not args.fresh) else {}
    t_all = time.time()
    units = [(f"residual_{n:.0f}", ("res", n)) for n in PC.SPEEDS]
    units += [(f"repro_{n:.0f}", ("crit", n, 0, KICK_REV_OLD)) for n in PC.SPEEDS]
    units += [(f"kick300_{n:.0f}_{p}", ("crit", n, p, KICK_REV_NEW)) for p in PHASES for n in PC.SPEEDS]
    for key, job in units:
        if key in out["units"]:
            continue
        if job[0] == "res":
            res, arr = residual(job[1])
            for k, v in arr.items():
                arrays[f"{job[1]:.0f}_{k}"] = v
            np.savez_compressed(NPZ, **arrays, tooth_periods_per_rev=4, nrev=NREV_RES)
        else:
            res = crit(*job[1:])
        out["units"][key] = res
        out["summary"] = summarize(out)
        out["runtime_s_this_invocation"] = round(time.time() - t_all, 1)
        PC.save_json(OUT, out)
        msg = res.get("value_mmps", "")
        print(f"{key}: {msg} ({res['seconds']:.1f} s)", flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"] = summarize(out)
    PC.save_json(OUT, out)
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
