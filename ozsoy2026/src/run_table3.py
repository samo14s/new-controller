"""Table 3: clipped DVF versus the saturation-aware gain at nine speeds.

Clipped DVF (paper Sec. 4): the gain is tuned on a_lim alone, ignoring the
actuator, and the actuator output is then simply limited during simulation.
Tuned on a_lim alone over the search interval G = [0, G_max], the gain runs to
the upper bound because a_lim grows with g; G_max is not stated in the paper,
so two values are reported (1500 V s/m, just above the largest gain of Fig. 8,
and the reproduction's 3000 V s/m).  The limit is applied to the command,
|V| <= F_max / K_a = 9 V.
Proposed: the saturation-aware gain of the main map.
Each controller gets a time-domain boundary (D1 detector, 9 V limit active);
the zero-order a_lim of the proposed gain is listed as well, since the paper's
Table 3 'proposed' column tracks its Fig. 9 frequency-domain curve.
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
G_CLIP = (1500.0, 3000.0)


def work(args):
    rpm, g_prop, zeta = args
    P.set_damping(zeta)
    from tdboundary import boundary
    from zoa import a_lim
    row = dict(rpm=rpm, g_prop=g_prop, a_unc_zoa=a_lim(rpm, 0.0) * 1e3,
               a_prop_zoa=a_lim(rpm, g_prop) * 1e3,
               a_prop_td=boundary(rpm, g_prop, vmax=VMAX) * 1e3)
    for gc in G_CLIP:
        row[f"a_clip{int(gc)}_zoa_nolimit"] = a_lim(rpm, gc) * 1e3
        row[f"a_clip{int(gc)}_td"] = boundary(rpm, gc, vmax=VMAX) * 1e3
    row["paper_clip"], row["paper_prop"] = PAPER[rpm]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
    zeta = float(d["zeta"])
    gmap = d["g_sade"] if "g_sade" in d else d["g_ref"]
    jobs = [(n, float(np.interp(n, d["rpm"], gmap)), zeta) for n in SPEEDS]
    with Pool(a.workers) as pool:
        rows = pool.map(work, jobs)
    keys = list(rows[0].keys())
    arr = np.array([[r[k] for k in keys] for r in rows])
    np.savetxt(os.path.join(RES, f"table3_{a.tag}.csv"), arr, delimiter=",",
               header=",".join(keys), comments="", fmt="%.3f")
    print(" rpm | paper clip  prop | clip1500 TD  clip3000 TD | g_prop  prop TD  prop ZOA")
    for r in rows:
        print(f"{r['rpm']:4d} | {r['paper_clip']:9.1f} {r['paper_prop']:5.1f} | "
              f"{r['a_clip1500_td']:11.2f} {r['a_clip3000_td']:12.2f} | {r['g_prop']:6.0f} "
              f"{r['a_prop_td']:8.2f} {r['a_prop_zoa']:9.2f}")


if __name__ == "__main__":
    main()
