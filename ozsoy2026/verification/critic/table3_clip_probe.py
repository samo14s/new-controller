"""Table 3 probe: what gain could the paper's clipped-DVF column use?

Part A (ZOA only, cheap): for each Table 3 speed, the ratio a_lim(g_c)/a_lim(g_fig8)
for candidate clipped gains g_c in {1500, 2000, 3000} versus the paper's
clipped/proposed ratio.  At low speed the paper says clipping is rarely active,
so the ratio there should follow the unclipped ZOA.
Part B (time domain, decay detector D1 of verification/tdsim, 2048 steps/rev,
300 rev, 9 V command limit): clipped boundaries at 500/1300/2500/3000 rpm for
g_c = 1500 and 3000 and for the Fig. 8 gain.
Writes table3_clip_probe.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tdsim"))
import params as P          # noqa: E402
from zoa import a_lim       # noqa: E402
import detector             # noqa: E402

PAPER = {500: (15.2, 14.9), 560: (8.7, 8.0), 600: (12.2, 11.8), 800: (11.2, 10.8),
         1300: (12.1, 12.0), 2300: (4.5, 4.9), 2500: (4.7, 5.3), 2800: (5.8, 6.6), 3000: (6.2, 7.1)}
fig8 = np.loadtxt(os.path.join(HERE, "..", "targets", "fig8_gain_map.csv"), delimiter=",", skiprows=1)
VMAX = P.F_MAX / P.GA_GAIN
out = {"partA": [], "partB": []}
for n, (pc, pp) in PAPER.items():
    g8 = float(np.interp(n, fig8[:, 0], fig8[:, 1]))
    a8 = a_lim(n, g8) * 1e3
    row = dict(rpm=n, g_fig8=round(g8, 1), a_zoa_g8_mm=round(a8, 3), paper_clip=pc, paper_prop=pp,
               paper_ratio=round(pc / pp, 4))
    for gc in (1500.0, 2000.0, 3000.0):
        row[f"zoa_ratio_g{int(gc)}"] = round(a_lim(n, gc) * 1e3 / a8, 4)
    out["partA"].append(row)
    print(row, flush=True)

for n in (500, 1300, 2500, 3000):
    g8 = float(np.interp(n, fig8[:, 0], fig8[:, 1]))
    for g in (g8, 1500.0, 3000.0):
        az = a_lim(n, g)
        lo_c, hi_c = detector.boundary(n, g, 300, spr=2048, a_guess=az, rtol=5e-3, vmax=VMAX)
        r = dict(rpm=n, g=round(g, 1), a_zoa_unclipped_mm=round(az * 1e3, 3),
                 a_td_clip9V_mm=round(lo_c * 1e3, 3), paper_clip=PAPER[n][0], paper_prop=PAPER[n][1])
        out["partB"].append(r)
        print(r, flush=True)
with open(os.path.join(HERE, "table3_clip_probe.json"), "w") as fh:
    json.dump(out, fh, indent=1)
