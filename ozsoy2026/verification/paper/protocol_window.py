"""Classifier window: critical kick judged over 900 instead of 300 revolutions.

Reviewer's question: near the limit the chatter mode decays (or grows) slowly,
so a run judged over the 300 revolutions after the kick (classifier D1 of
src/tdboundary.classify_run) might be called stable while it would chatter
later, or the reverse; is the stored critical kick at 0.98 a_lim biased?

At the four C1 speeds (1090, 1420, 2500, 2870 rpm), 0.98 a_lim(rpm, 1500),
the phase of the smallest stored critical kick (worst stored phase of
results/c1_basin.npz), 9 V, the critical kick is recomputed with the kick at
revolution 100 (as stored) and a window of 900 revolutions after it (runs of
1000 revolutions); everything else is the stored protocol (DV_GRID upward
search, 6 bisections in log dv, full surface memory).  The classifier is
used unchanged; its criteria are fractions of the window (first 10 % against
the last 20 %, growth fitted over the second half), so with 900 revolutions
every criterion is applied over a span three times longer.  Every run is also
judged over its first 300 revolutions after the kick, which is exactly the
stored protocol's run (the integration is causal), so any disagreement
between the two windows on the same kick is listed.

Run (one process, about 4 min on a free core; resumable, --fresh starts over):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_window.py
Writes verification/paper/protocol_window.json (after every unit).
"""
import argparse
import os
import time

import numpy as np

import protocol_common as PC

OUT = os.path.join(PC.HERE, "protocol_window.json")
FRAC = 0.98
WINDOW_NEW, WINDOW_OLD, KICK_REV = 900, 300, 100


def worst_phase(rpm):
    dv = PC.stored_row(rpm, FRAC)["dv"]
    return int(np.argmin(dv)), float(dv.min())


def crit(rpm):
    p, stored = worst_phase(rpm)
    a = PC.depth(rpm, FRAC)
    t0 = time.time()
    val, trace = PC.critical_kick(rpm, a, PC.G_DESIGN, PC.VMAX, p, 16, KICK_REV, WINDOW_NEW,
                                  extra_windows=(WINDOW_OLD,))
    lo, hi = PC.bracket(trace, val)
    disagree = [dict(dv_mmps=t["dv"] * 1e3, chatter_900=t["chatter"], chatter_300=t[f"w{WINDOW_OLD}"]["chatter"],
                     mu_900=t["mu"], q_900=t["q"], mu_300=t[f"w{WINDOW_OLD}"]["mu"], q_300=t[f"w{WINDOW_OLD}"]["q"])
                for t in trace if t["chatter"] != t[f"w{WINDOW_OLD}"]["chatter"]]
    return dict(rpm=rpm, frac=FRAC, worst_stored_phase_of_16=p, window=WINDOW_NEW, kick_rev=KICK_REV,
                value_mmps=val * 1e3, bracket_mmps=[lo * 1e3, hi * 1e3], stored_300_mmps=stored * 1e3,
                ratio_900_over_300=float(val / stored), identical=bool(val == stored),
                verdict_disagreements_on_same_runs=disagree, n_runs=len(trace),
                seconds=time.time() - t0, trace=trace)


def summarize(out):
    rows = [out["units"].get(f"window900_{n:.0f}") for n in PC.SPEEDS]
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    r = np.array([x["ratio_900_over_300"] for x in rows])
    return dict(n_points=len(rows), ratio_min=float(r.min()), ratio_max=float(r.max()),
                max_abs_rel_change=float(np.max(np.abs(r - 1))),
                n_disagreeing_runs=int(sum(len(x["verdict_disagreements_on_same_runs"]) for x in rows)),
                n_runs=int(sum(x["n_runs"] for x in rows)),
                by_rpm={f"{x['rpm']:.0f}": dict(phase=x["worst_stored_phase_of_16"], w900_mmps=x["value_mmps"],
                                                 w300_stored_mmps=x["stored_300_mmps"],
                                                 ratio=x["ratio_900_over_300"]) for x in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    out = None if args.fresh else PC.load_json(OUT)
    if out is None:
        out = dict(PC.header(__file__, __doc__),
                   protocol=dict(frac=FRAC, phase="worst stored phase of 16", kick_rev=KICK_REV,
                                 window_new=WINDOW_NEW, window_stored=WINDOW_OLD, vmax=PC.VMAX,
                                 entry="face (edge=True)"),
                   units={})
    t_all = time.time()
    for n in PC.SPEEDS:
        key = f"window900_{n:.0f}"
        if key in out["units"]:
            continue
        res = crit(n)
        out["units"][key] = res
        out["summary"] = summarize(out)
        out["runtime_s_this_invocation"] = round(time.time() - t_all, 1)
        PC.save_json(OUT, out)
        print(f"{key}: {res['value_mmps']:.4f} mm/s (stored {res['stored_300_mmps']:.4f}), "
              f"{len(res['verdict_disagreements_on_same_runs'])} disagreeing runs ({res['seconds']:.1f} s)", flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["summary"] = summarize(out)
    PC.save_json(OUT, out)
    print(out["summary"], flush=True)


if __name__ == "__main__":
    main()
