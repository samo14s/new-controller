"""Orbit limit for the gain sweep of circle_criterion.json (Proposition 1 needs an unclipped forced orbit).

circle_criterion.py applies the orbit limit (first depth at which the forced command exceeds Vmax)
to the scheduled gains (g_best) and to g = 1500 (g1500), not to its gain sweep at the four speeds of
Section 4.4.  This script adds it: for each gain-sweep row, the first of 2000 depths up to a_circ at
which the peak forced command exceeds 9 V, a_circ_eff = min(a_circ, that depth), and the ratio to
a_lim(0).  It also lists the frequency at which the condition fails, by speed range, for the text.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/circle_gainsweep_orbit.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circle_criterion as CC  # noqa: E402


def main():
    c = json.load(open(os.path.join(HERE, "circle_criterion.json")))
    rows = []
    for q in c["gain_sweep"]:
        a_c = q["a_circ_mm"] * 1e-3
        a_k = CC.first_clipped_depth(q["rpm"], q["g"], a_c)
        eff = q["a_circ_mm"] if a_k is None else min(q["a_circ_mm"], a_k * 1e3)
        rows.append(dict(rpm=q["rpm"], g=q["g"], ratio=q["ratio"], first_clipped_depth_mm=None if a_k is None else a_k * 1e3,
                         a_circ_eff_mm=eff, ratio_eff=eff / q["a0_mm"]))
    gb = c["g_best"]["rows"]
    f = np.array([r["f_fail_hz"] for r in gb], float)
    ratio = np.array([r["ratio"] for r in gb], float)
    rpm = np.array([r["rpm"] for r in gb], float)
    low = ratio < 0.9
    out = dict(note=__doc__.split("\n\n")[0], gain_sweep=rows,
               f_fail=dict(all_min_hz=float(f.min()), all_max_hz=float(f.max()), median_hz=float(np.median(f)),
                           ratio_below_0p9_n=int(low.sum()),
                           ratio_below_0p9_rpm=rpm[low].tolist(),
                           ratio_below_0p9_f_min_hz=float(f[low].min()) if low.any() else None,
                           ratio_below_0p9_f_max_hz=float(f[low].max()) if low.any() else None,
                           at_four_speeds={str(int(r)): float(f[rpm == r][0]) for r in (1090, 1420, 2500, 2870)}))
    with open(os.path.join(HERE, "circle_gainsweep_orbit.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(r)
    print(out["f_fail"])


if __name__ == "__main__":
    main()
