"""Onset of saturation: the article's time-domain simulation (open squares of its Fig. 6) against the model.

For each speed of the four squares (nominal structure, g = 253 V s/m) the depth at which the settled forced
demand of the model first exceeds 27 N (scan 0.1-12 mm in steps of 0.05 mm, below the controlled limit),
and the demand at the article's onset depth.
Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/ozsoy2025/onset2025.py   (writes onset2025.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup2025 as S  # noqa: E402
import feasibility as F  # noqa: E402
import zoa  # noqa: E402

S.apply_structure(**S.NOMINAL)
sq = [q for q in json.load(open(os.path.join(HERE, "points_fig6.json")))["points"] if q["type"] == "saturation_onset_sim"]
dd = np.arange(0.1, 12.001, 0.05) * 1e-3
rows = []
for q in sorted(sq, key=lambda q: q["rpm"]):
    n = float(q["rpm"])
    alg = zoa.a_lim(n, S.G_FIG6)
    d = F.forced_demand(n, S.G_FIG6, dd)
    ok = (d > 27.0) & (dd < alg)
    rows.append(dict(rpm=n, article_onset_mm=q["depth_mm"], demand_at_article_onset_N=float(F.forced_demand(n, S.G_FIG6, np.array([q["depth_mm"] * 1e-3]))[0]),
                     model_onset_mm=float(dd[np.argmax(ok)] * 1e3) if ok.any() else None, a_lim_g_mm=float(alg * 1e3),
                     a_lim0_mm=float(zoa.a_lim(n, 0.0) * 1e3)))
json.dump(dict(note=__doc__.split("\n\n")[0], rows=rows), open(os.path.join(HERE, "onset2025.json"), "w"), indent=1)
for r in rows:
    print(r)
