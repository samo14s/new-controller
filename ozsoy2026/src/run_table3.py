"""Table 3: clipped DVF versus the saturation-aware gain at nine speeds.

Clipped DVF: the gain is tuned on a_lim alone (no saturation check) over the
same interval G, then the cut is simulated with the actuator command limited
to F_max / K_a (27 N / 3 N/V = 9 V); its limiting depth must come from the
time domain because the limit makes the loop nonlinear.
Proposed: the saturation-aware gain of the main map, simulated with the same
limit.  Both are reported as time-domain boundaries; the zero-order a_lim of
the proposed gain is listed too.
    python run_table3.py --tag main
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SPEEDS = [500, 560, 600, 800, 1300, 2300, 2500, 2800, 3000]
PAPER = {500: (15.2, 14.9), 560: (8.7, 8.0), 600: (12.2, 11.8), 800: (11.2, 10.8),
         1300: (12.1, 12.0), 2300: (4.5, 4.9), 2500: (4.7, 5.3), 2800: (5.8, 6.6),
         3000: (6.2, 7.1)}
VMAX = P.F_MAX / P.GA_GAIN


def work(args):
    rpm, g_prop, g_clip, zeta = args
    P.set_damping(zeta)
    from tdboundary import boundary
    from zoa import a_lim
    return dict(rpm=rpm, g_prop=g_prop, g_clip=g_clip,
                a_prop_zoa=a_lim(rpm, g_prop) * 1e3,
                a_clip_zoa_unclipped=a_lim(rpm, g_clip) * 1e3,
                a_prop_td=boundary(rpm, g_prop, vmax=VMAX) * 1e3,
                a_clip_td=boundary(rpm, g_clip, vmax=VMAX) * 1e3,
                a_unc_td=boundary(rpm, 0.0) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
    zeta = float(d["zeta"])
    jobs = []
    for n in SPEEDS:
        i = int(np.argmin(np.abs(d["rpm"] - n)))
        g_prop = float(d["g_sade"][i]) if "g_sade" in d else float(d["g_ref"][i])
        g_clip = float(d["scan_g"][int(np.argmax(d["scan_alim"][i]))])
        jobs.append((n, g_prop, g_clip, zeta))
    with Pool(a.workers) as pool:
        rows = pool.map(work, jobs)
    keys = list(rows[0].keys())
    arr = np.array([[r[k] for k in keys] for r in rows])
    np.savetxt(os.path.join(RES, f"table3_{a.tag}.csv"), arr, delimiter=",",
               header=",".join(keys), comments="", fmt="%.3f")
    print(f"{'rpm':>5} {'g_clip':>7} {'clip TD':>8} {'paper':>6} | {'g_prop':>7} "
          f"{'prop TD':>8} {'prop ZOA':>8} {'paper':>6}")
    for r in rows:
        pc, pp = PAPER[r["rpm"]]
        print(f"{r['rpm']:5d} {r['g_clip']:7.0f} {r['a_clip_td']:8.2f} {pc:6.1f} | "
              f"{r['g_prop']:7.0f} {r['a_prop_td']:8.2f} {r['a_prop_zoa']:8.2f} {pp:6.1f}")


if __name__ == "__main__":
    main()
