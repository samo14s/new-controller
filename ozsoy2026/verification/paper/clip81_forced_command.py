"""Where the forced command of the g = 1500 V s/m design exceeds a clip of 8.1 V (Section 7.2).

For the four speeds of Section 4.4, the peak forced command b_max(a) (closed form, src/feasibility.py
forced_demand / K_a) on 400 depths up to a_lim(n, 1500): the depth intervals where it exceeds 8.1 V,
as fractions of a_lim, and its largest value.  With an 8.1 V clip the forced orbit itself is clipped
on these intervals.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/clip81_forced_command.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")
import params as P  # noqa: E402

P.set_damping(float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"]))
import feasibility as F  # noqa: E402
from zoa import a_lim  # noqa: E402

G, CLIP, NDEP = 1500.0, 8.1, 400


def main():
    out = dict(note=__doc__.split("\n\n")[0], g=G, clip_V=CLIP, n_depths=NDEP, speeds={})
    for n in (1090.0, 1420.0, 2500.0, 2870.0):
        al = a_lim(n, G)
        d = al * np.arange(1, NDEP + 1) / NDEP
        v = F.forced_demand(n, G, d) / P.GA_GAIN
        above = v > CLIP
        iv, k = [], 0
        while k < NDEP:
            if above[k]:
                j = k
                while j + 1 < NDEP and above[j + 1]:
                    j += 1
                iv.append([float(d[k] / al), float(d[j] / al)])
                k = j + 1
            else:
                k += 1
        out["speeds"][f"{n:.0f}"] = dict(a_lim_mm=al * 1e3, max_V=float(v.max()), at_frac=float(d[np.argmax(v)] / al),
                                         V_at_a_lim=float(v[-1]), intervals_above_clip_frac=iv)
    with open(os.path.join(HERE, "clip81_forced_command.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["speeds"], indent=1))


if __name__ == "__main__":
    main()
