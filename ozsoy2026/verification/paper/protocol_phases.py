"""Worst phase: critical kick at 64 kick phases per tooth period instead of 16.

Reviewer's question: the stored worst-phase critical kick at 0.98 a_lim
(results/c1_basin.npz: minimum over 16 phases) may miss a narrower minimum
between the sampled phases.  How much does the minimum over phases move with
a finer phase grid?

At the four C1 speeds (1090, 1420, 2500, 2870 rpm), 0.98 a_lim(rpm, 1500),
9 V, the critical kick is computed at the phases p = 0..63 of 64 (kick step
100 * 512 + 2 p; the tooth period has 128 steps), with the stored protocol
otherwise (face entry, kick at revolution 100, 300-revolution window, full
surface memory, DV_GRID upward search, 6 bisections).  The phases p = 4 k are
the stored phases k of 16 (same kick step) and are taken from
results/c1_basin.npz; protocol_kick_timing.py checks that the implementation
reproduces stored values exactly.  The 48 new phases per speed are computed
in two passes, p = 2 mod 4 first (which completes a 32-phase grid at every
speed), then the odd p; the summary reports the minimum over 16, 32 and 64
phases, the phase spread, the impulse share m dv*/(F_y T) of Table 2 of the
manuscript and the ratio to the C2 certificate of results/c2_validate.npz.

Run (one process, about 50 min on a free core; resumable, --fresh starts
over; --pass32 stops after the 32-phase pass):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_phases.py
Writes verification/paper/protocol_phases.json (after every unit) and
protocol_phases.npz (the 4 x 64 critical kicks, nan where not computed).
"""
import argparse
import os
import time

import numpy as np

import protocol_common as PC

OUT = os.path.join(PC.HERE, "protocol_phases.json")
NPZ = os.path.join(PC.HERE, "protocol_phases.npz")
FRAC = 0.98
N_PHASE = 64
PASS32 = [p for p in range(N_PHASE) if p % 4 == 2]
PASS64 = [p for p in range(N_PHASE) if p % 2 == 1]
WN = 2 * np.pi * PC.P.FN


def crit(rpm, p):
    a = PC.depth(rpm, FRAC)
    t0 = time.time()
    val, trace = PC.critical_kick(rpm, a, PC.G_DESIGN, PC.VMAX, p, N_PHASE, 100, 300)
    lo, hi = PC.bracket(trace, val)
    # store a compact trace (the verdicts and classifier numbers)
    return dict(rpm=rpm, phase_of_64=p, kick_step_in_tooth=(p * PC.ND) // N_PHASE, value_mmps=val * 1e3,
                bracket_mmps=[lo * 1e3, hi * 1e3], n_runs=len(trace), seconds=time.time() - t0,
                trace=[[t["dv"], t["chatter"], t["mu"], t["q"], t["dx_end_over_fz"]] for t in trace])


def table(out):
    """(4, 64) array of critical kicks in mm/s, stored phases filled in, nan where missing."""
    T = np.full((len(PC.SPEEDS), N_PHASE), np.nan)
    for i, n in enumerate(PC.SPEEDS):
        T[i, ::4] = PC.stored_row(n, FRAC)["dv"] * 1e3
        for p in range(N_PHASE):
            u = out["units"].get(f"ph64_{n:.0f}_{p}")
            if u is not None:
                T[i, p] = u["value_mmps"]
    return T


def summarize(out):
    T = table(out)
    c2 = np.load(os.path.join(PC.RES, "c2_validate.npz"))
    S = {}
    for i, n in enumerate(PC.SPEEDS):
        s = PC.stored_row(n, FRAC)
        row = T[i]
        m16 = float(np.min(row[::4]))
        have32 = bool(np.all(np.isfinite(row[::2])))
        have64 = bool(np.all(np.isfinite(row)))
        fin = row[np.isfinite(row)]
        mall = float(fin.min())
        k = int(np.nanargmin(row))
        j = int(np.nonzero((c2["rpm"] == n) & np.isclose(c2["frac"], FRAC))[0][0])
        beta = float(c2["cert"][j]) * 1e3
        imp = lambda dv: float(s["mass"] * dv * 1e-3 / s["tooth_impulse"] * 100)
        S[f"{n:.0f}"] = dict(
            n_phases_available=int(np.isfinite(row).sum()),
            min_16_mmps=m16, phase16_of_min=int(np.argmin(row[::4])),
            min_32_mmps=float(np.min(row[::2])) if have32 else None,
            min_64_mmps=float(row.min()) if have64 else None,
            min_available_mmps=mall, phase64_of_min=k,
            rel_change_of_min=(mall / m16 - 1),
            max_16_mmps=float(np.max(row[::4])), max_available_mmps=float(fin.max()),
            spread_16=float(np.max(row[::4]) / m16 - 1), spread_available=float(fin.max() / mall - 1),
            median_16_mmps=float(np.median(row[::4])), median_available_mmps=float(np.median(fin)),
            impulse_pct_min16=imp(m16), impulse_pct_min_available=imp(mall),
            displacement_um_min16=m16 * 1e-3 / WN * 1e6, displacement_um_min_available=mall * 1e-3 / WN * 1e6,
            c2_cert_mmps=beta, min_available_over_c2_cert=mall / beta)
    av = [S[k]["rel_change_of_min"] for k in S]
    S["all"] = dict(largest_drop_of_min=float(min(av)), largest_rise_of_min=float(max(av)),
                    impulse_pct_min_available_range=[min(S[k]["impulse_pct_min_available"] for k in S if k != "all"),
                                                     max(S[k]["impulse_pct_min_available"] for k in S if k != "all")],
                    spread_available_max=float(max(S[k]["spread_available"] for k in S if k != "all")),
                    spread_16_max=float(max(S[k]["spread_16"] for k in S if k != "all")),
                    units_done=len(out["units"]), units_total=len(PC.SPEEDS) * (len(PASS32) + len(PASS64)))
    return S, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--pass32", action="store_true", help="stop after the 32-phase pass")
    args = ap.parse_args()
    out = None if args.fresh else PC.load_json(OUT)
    if out is None:
        out = dict(PC.header(__file__, __doc__),
                   protocol=dict(frac=FRAC, n_phase=N_PHASE, kick_rev=100, window=300, vmax=PC.VMAX,
                                 stored_phases="p = 4k of 64 = phase k of 16, from results/c1_basin.npz",
                                 entry="face (edge=True)"),
                   units={})
    t_all = time.time()
    order = PASS32 + ([] if args.pass32 else PASS64)
    for p in order:
        for n in PC.SPEEDS:
            key = f"ph64_{n:.0f}_{p}"
            if key in out["units"]:
                continue
            res = crit(n, p)
            out["units"][key] = res
            out["summary"], T = summarize(out)
            out["runtime_s_this_invocation"] = round(time.time() - t_all, 1)
            PC.save_json(OUT, out)
            np.savez_compressed(NPZ, rpm=np.array(PC.SPEEDS), phase_of_64=np.arange(N_PHASE), dv_mmps=T,
                                frac=FRAC, g=PC.G_DESIGN, vmax=PC.VMAX)
            print(f"{key}: {res['value_mmps']:.4f} mm/s ({res['seconds']:.1f} s, "
                  f"{len(out['units'])} units, {time.time() - t_all:.0f} s)", flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"], T = summarize(out)
    PC.save_json(OUT, out)
    np.savez_compressed(NPZ, rpm=np.array(PC.SPEEDS), phase_of_64=np.arange(N_PHASE), dv_mmps=T,
                        frac=FRAC, g=PC.G_DESIGN, vmax=PC.VMAX)
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
