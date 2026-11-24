"""Band holes: the bistable band scanned on a depth grid instead of bisected.

Reviewer's question: the lower edge of the bistable band (results/c1_band.npz,
src/c1_unsafe_zone._band_low) was found by six bisection steps on
[0.05, 0.98] a_lim, which assumes that the set of depths at which some kick
ends in chatter is one interval reaching up to 0.98 a_lim (no holes in depth).
Is it?

At nine speeds spread over the range where a band was found (640, 960, 1640,
1820, 1000, 2140, 2800, 1300 and 520 rpm, in this order: first the speeds
where the stored edge jumps between neighbouring speeds or sits at the top of
the bisection interval, 0.98 a_lim, then evenly spread ones), with the gain
of the band campaign (results/c1_band.npz g, the feas_g1500 map), each depth
of the grid 0.05, 0.10, ..., 0.95 and 0.98 a_lim(rpm, g) is tested exactly as
one bisection step of _band_low: the seven kicks of BAND_DV (3 mm/s to
3 m/s, geometric) at the two phases 0 and 1/2 of a tooth period, kick at
revolution 100 of a face entry, 9 V clip, full surface memory, 300-revolution
window, classifier D1; the depth chatters if any of the 14 runs does (the
runs are made in _band_low's order and stop at the first chatter).  A hole is
a grid depth above the stored edge that does not chatter; the converse is a
grid depth below the stored edge that chatters.  The bisection brackets the
edge between two tested depths 0.93/64 = 0.0145 a_lim apart, so a grid depth
inside that bracket is ambiguous and is flagged as such.

Run (one process, about 35 min on a free core; resumable, --fresh starts over):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_band_holes.py
Writes verification/paper/protocol_band_holes.json (after every depth).
"""
import argparse
import os
import time

import numpy as np

import protocol_common as PC

OUT = os.path.join(PC.HERE, "protocol_band_holes.json")
SPEEDS = (640.0, 960.0, 1640.0, 1820.0, 1000.0, 2140.0, 2800.0, 1300.0, 520.0)
FRACS = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()) + (0.98,)
BAND_DV = PC.C1.BAND_DV
BAND_PHASES = PC.C1.BAND_PHASES
BISECT_STEP = (0.98 - 0.05) / 2 ** PC.C1.BAND_BISECT


def band_row(rpm):
    B = np.load(os.path.join(PC.RES, "c1_band.npz"))
    i = int(np.nonzero(B["rpm"] == rpm)[0][0])
    g = float(B["g"][i])
    al = PC.a_lim(rpm, g)
    edge_mm = float(B["a_band_low"][i])
    return g, al, edge_mm


def depth_test(rpm, frac):
    g, al, edge_mm = band_row(rpm)
    a = frac * al
    t0 = time.time()
    runs = []
    chatter = False
    for dv in BAND_DV:
        for p in range(BAND_PHASES):
            ks = PC.kick_step(p, BAND_PHASES, 100)
            o = PC.kick_run(rpm, a, g, PC.VMAX, ks, dv, 100, 300)
            runs.append([float(dv), p, o["chatter"], o["mu"], o["q"], o["dx_end_over_fz"], o["diverged"]])
            if o["chatter"]:
                chatter = True
                break
        if chatter:
            break
    first = [r for r in runs if r[2]]
    return dict(rpm=rpm, g=g, frac=frac, a_mm=a * 1e3, chatter=chatter,
                first_chatter_kick_mps=first[0][0] if first else None,
                first_chatter_phase_of_2=first[0][1] if first else None,
                n_runs=len(runs), seconds=time.time() - t0, runs=runs)


def summarize(out):
    S = {}
    for n in SPEEDS:
        rows = [out["units"].get(f"band_{n:.0f}_{f:.2f}") for f in FRACS]
        done = [r for r in rows if r is not None]
        if not done:
            continue
        g, al, edge_mm = band_row(n)
        ef = edge_mm / (al * 1e3)                    # band_row gives a_lim in m, the edge in mm
        lo_f = ef - BISECT_STEP                       # the last non-chattering bisection depth
        ch = [r["frac"] for r in done if r["chatter"]]
        nc = [r["frac"] for r in done if not r["chatter"]]
        holes = [f for f in nc if f > ef + 1e-9]
        below = [f for f in ch if f < lo_f - 1e-9]
        ambiguous = [r["frac"] for r in done if lo_f - 1e-9 <= r["frac"] <= ef + 1e-9]
        grid_edge = min(ch) if ch else None
        # is the chattering set an interval [grid_edge, 0.98] on the grid?
        interval = bool(ch) and all(r["chatter"] for r in done if r["frac"] >= grid_edge)
        S[f"{n:.0f}"] = dict(g=g, a_lim_mm=al * 1e3, stored_edge_mm=edge_mm, stored_edge_frac=ef,
                             bisection_bracket_frac=[lo_f, ef], depths_done=len(done), depths_total=len(FRACS),
                             chatter_fracs=ch, no_chatter_fracs=nc,
                             holes_above_edge=holes, chatter_below_edge=below, ambiguous_in_bracket=ambiguous,
                             grid_edge_frac=grid_edge, chatter_set_is_interval_on_grid=interval,
                             runs=int(sum(r["n_runs"] for r in done)))
    complete = [k for k in S if S[k]["depths_done"] == len(FRACS)]
    S["all"] = dict(speeds_complete=complete,
                    speeds_with_holes=[k for k in S if k != "all" and S[k]["holes_above_edge"]],
                    speeds_with_chatter_below_edge=[k for k in S if k != "all" and S[k]["chatter_below_edge"]],
                    total_runs=int(sum(S[k]["runs"] for k in S if k != "all")))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    out = None if args.fresh else PC.load_json(OUT)
    if out is None:
        out = dict(PC.header(__file__, __doc__),
                   protocol=dict(speeds=list(SPEEDS), fracs=list(FRACS), band_dv=BAND_DV.tolist(),
                                 band_phases=BAND_PHASES, kick_rev=100, window=300, vmax=PC.VMAX,
                                 gain="results/c1_band.npz g (feas_g1500 map)", entry="face (edge=True)",
                                 bisection_step_frac=BISECT_STEP),
                   units={})
    t_all = time.time()
    for n in SPEEDS:
        for f in FRACS:
            key = f"band_{n:.0f}_{f:.2f}"
            if key in out["units"]:
                continue
            res = depth_test(n, f)
            out["units"][key] = res
            out["summary"] = summarize(out)
            out["runtime_s_this_invocation"] = round(time.time() - t_all, 1)
            PC.save_json(OUT, out)
            print(f"{key}: chatter {res['chatter']} (first kick {res['first_chatter_kick_mps']}, "
                  f"{res['n_runs']} runs, {res['seconds']:.1f} s; total {time.time() - t_all:.0f} s)", flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"] = summarize(out)
    PC.save_json(OUT, out)
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
