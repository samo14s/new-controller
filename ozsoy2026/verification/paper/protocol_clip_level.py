"""Clip level: critical kick with an 8.1 V clip against the stored 9 V protocol.

Reviewer's question: Section 2 states that an 8.1 V clip (10 % of the voltage
kept free) would keep the actuator force below 27 N, and that near the limit
the basin scales with the free voltage V_max - b_max.  What does an 8.1 V
clip do to the simulated basin?

  (a) closed form (src/feasibility.forced_velocity, 40 harmonics): peak
      forced command b_max = g max_t |x'(t)| of the settled orbit at a_lim and
      at 0.9 a_lim (g = 1500 V s/m), on 128 samples per tooth period (as
      results/c1_df.npz) and on 4096, the worst b_max over 400 depths up to
      0.9 a_lim and up to a_lim, and the free voltage V_max - b_max for
      V_max = 9 and 8.1 V; the dual-input describing-function estimate of the
      critical kick at 0.9 a_lim, dv_DF = A*/(g p), with A* from
      c1_unsafe_zone.critical_amplitude at either clip and p the kick-to-mode
      factor stored in results/c1_basin.npz (the 9 V value is checked against
      A_dual of results/c1_df.npz);
  (b) simulation: critical kick at the four C1 speeds (1090, 1420, 2500,
      2870 rpm) at 0.9 a_lim(rpm, 1500), phases 0, 4, 8, 12 of 16, with
      |V| <= 8.1 V and otherwise the stored protocol (face entry, kick at
      revolution 100, 300-revolution window, full surface memory, DV_GRID
      search, 6 bisections), against the stored 9 V values of
      results/c1_basin.npz at the same phases.  The depth and the gain do not
      change with the clip (a_lim is a property of the linear loop).

Run (one process, about 5 min on a free core; resumable, --fresh starts over):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_clip_level.py
Writes verification/paper/protocol_clip_level.json (after every unit).
"""
import argparse
import os
import time

import numpy as np

import protocol_common as PC
from feasibility import forced_velocity

OUT = os.path.join(PC.HERE, "protocol_clip_level.json")
FRAC = 0.9
PHASES = (0, 4, 8, 12)
CLIPS = (9.0, 8.1)
VMAX_NEW = 8.1


def bmax(rpm, a, g=PC.G_DESIGN, n_t=128):
    return float(g * np.abs(forced_velocity(rpm, g, [a], n_t=n_t)[0]).max())


def closed_form():
    import c1_unsafe_zone as C1
    df = np.load(os.path.join(PC.RES, "c1_df.npz"))
    i09 = int(np.argmin(np.abs(df["frac"] - FRAC)))
    out = {}
    g = PC.G_DESIGN
    for j, n in enumerate(PC.SPEEDS):
        al = PC.a_lim(n, g)
        row = dict(a_lim_mm=al * 1e3)
        for lab, a in (("alim", al), ("0p9alim", FRAC * al)):
            b128, b4096 = bmax(n, a, n_t=128), bmax(n, a, n_t=4096)
            row[f"bmax_{lab}_V"] = b4096
            row[f"bmax_{lab}_V_128samples"] = b128
            for v in CLIPS:
                row[f"free_{lab}_{str(v).replace('.', 'p')}V"] = v - b4096
        for lab, top in (("up_to_0p9alim", FRAC * al), ("up_to_alim", al)):
            ds = np.linspace(top / 400, top, 400)
            bb = g * np.abs(forced_velocity(n, g, ds, n_t=1024)).max(axis=1)
            k = int(np.argmax(bb))
            row[f"worst_bmax_{lab}_V"] = float(bb[k])
            row[f"worst_bmax_{lab}_at_frac"] = float(ds[k] / al)
        # describing-function estimate at 0.9 a_lim (as c1_unsafe_zone._df_speed)
        gains = np.linspace(0.0, g, 81)
        alg = np.array([PC.a_lim(n, x) for x in gains])
        a = FRAC * alg[-1]
        ge = C1.effective_gain_needed(alg, gains, a)
        b = -g * forced_velocity(n, g, [a], n_t=PC.ND)[0]
        p = PC.stored_row(n, FRAC)["p_mode"]
        row["df"] = dict(g_eff_needed=ge, p_mode=p)
        for v in CLIPS:
            A = C1.critical_amplitude(ge / g, b, vmax=v)
            row["df"][f"A_star_{str(v).replace('.', 'p')}V"] = A
            row["df"][f"dv_DF_{str(v).replace('.', 'p')}V_mmps"] = A / (g * p) * 1e3
        row["df"]["A_star_9V_stored_c1_df"] = float(df["A_dual"][j, i09])
        row["df"]["c1_df_frac"] = float(df["frac"][i09])
        row["df"]["ratio_dv_DF_8p1_over_9"] = row["df"]["A_star_8p1V"] / row["df"]["A_star_9p0V"]
        out[f"{n:.0f}"] = row
    return out


def crit(rpm, phase):
    a = PC.depth(rpm, FRAC)
    t0 = time.time()
    val, trace = PC.critical_kick(rpm, a, PC.G_DESIGN, VMAX_NEW, phase, 16, 100, 300)
    lo, hi = PC.bracket(trace, val)
    stored = PC.stored_row(rpm, FRAC)["dv"][phase]
    return dict(rpm=rpm, frac=FRAC, phase_of_16=phase, vmax=VMAX_NEW, value_mmps=val * 1e3,
                bracket_mmps=[lo * 1e3, hi * 1e3], stored_9V_mmps=float(stored * 1e3),
                ratio_8p1_over_9=float(val / stored), n_runs=len(trace),
                seconds=time.time() - t0, trace=trace)


def summarize(out):
    S = {}
    for n in PC.SPEEDS:
        key = f"{n:.0f}"
        u = [out["units"].get(f"clip8p1_{key}_{p}") for p in PHASES]
        if any(x is None for x in u):
            continue
        v8 = np.array([x["value_mmps"] for x in u])
        v9 = np.array([x["stored_9V_mmps"] for x in u])
        cf = out["closed_form"][key]
        S[key] = dict(phases_of_16=list(PHASES), kick_8p1V_mmps=v8.tolist(), kick_9V_mmps=v9.tolist(),
                      ratio=(v8 / v9).tolist(), min_8p1V_mmps=float(np.nanmin(v8)) if np.any(np.isfinite(v8)) else None,
                      min_9V_mmps=float(v9.min()),
                      ratio_of_minima=float(np.nanmin(v8) / v9.min()) if np.any(np.isfinite(v8)) else None,
                      n_nan=int(np.isnan(v8).sum()),
                      df_ratio_prediction=cf["df"]["ratio_dv_DF_8p1_over_9"],
                      free_voltage_ratio_at_0p9alim=cf["free_0p9alim_8p1V"] / cf["free_0p9alim_9p0V"],
                      free_voltage_ratio_at_alim=(cf["free_alim_8p1V"] / cf["free_alim_9p0V"]))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    out = None if args.fresh else PC.load_json(OUT)
    if out is None:
        out = dict(PC.header(__file__, __doc__),
                   protocol=dict(frac=FRAC, phases_of_16=list(PHASES), vmax_new=VMAX_NEW, vmax_stored=9.0,
                                 kick_rev=100, window=300, entry="face (edge=True)"),
                   units={})
    t_all = time.time()
    if "closed_form" not in out:
        out["closed_form"] = closed_form()
        PC.save_json(OUT, out)
        print("closed form done", flush=True)
    for p in PHASES:
        for n in PC.SPEEDS:
            key = f"clip8p1_{n:.0f}_{p}"
            if key in out["units"]:
                continue
            res = crit(n, p)
            out["units"][key] = res
            out["summary"] = summarize(out)
            out["runtime_s_this_invocation"] = round(time.time() - t_all, 1)
            PC.save_json(OUT, out)
            print(f"{key}: {res['value_mmps']:.3f} mm/s (9 V: {res['stored_9V_mmps']:.3f}) "
                  f"({res['seconds']:.1f} s)", flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"] = summarize(out)
    PC.save_json(OUT, out)
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
