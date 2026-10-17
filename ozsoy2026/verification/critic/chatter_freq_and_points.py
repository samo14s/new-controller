"""ZOA chatter frequencies at the labelled cuts of Figs. 10/12 and classification
of all 21 experimental points with the ZOA boundary at (a) the paper's gains
(Fig. 8 map, single gain 237 and the fitted 270) and (b) gains reproduced by
gainmap_spr_convergence (force demand, >= 2048 steps/rev) where available.
Writes chatter_freq_and_points.json.
"""
import csv
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from zoa import a_lim  # noqa: E402

fig8 = np.loadtxt(os.path.join(HERE, "..", "targets", "fig8_gain_map.csv"), delimiter=",", skiprows=1)
g8 = lambda n: float(np.interp(n, fig8[:, 0], fig8[:, 1]))  # noqa: E731
out = {"chatter_freq": [], "points": []}
for lab, n, g, f_meas in [("Fig10B", 1300, g8(1300), 149.5), ("Fig12B", 2800, 0.0, 133.3),
                          ("Fig12D_g237", 2700, 237.0, 138.1), ("Fig12D_g270", 2700, 270.0, 138.1)]:
    a, fc = a_lim(n, g, return_freq=True)
    out["chatter_freq"].append(dict(cut=lab, rpm=n, g=round(g, 1), a_lim_mm=round(a * 1e3, 3),
                                    f_zoa_Hz=round(fc, 2), f_fft_Hz=f_meas, diff_Hz=round(fc - f_meas, 2)))
    print(out["chatter_freq"][-1])
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "targets", "experimental_points.csv"))))
for r in rows:
    n, d, ctrl, st = float(r["rpm"]), float(r["depth_mm"]), r["control"], r["state"]
    gains = {"none": [0.0], "mapped gain": [g8(n)], "single gain 237": [237.0, 270.0]}[ctrl]
    for g in gains:
        a = a_lim(n, g) * 1e3
        pred = "stable" if d < a else "chatter"
        out["points"].append(dict(figure=r["figure"], label=r["label"], control=ctrl, rpm=n, depth_mm=d,
                                  measured=st, g=round(g, 1), a_zoa_mm=round(a, 3), predicted=pred,
                                  agree=pred == st))
ok = {}
for p in out["points"]:
    k = (p["control"], p["g"] if p["control"] == "single gain 237" else "paper")
    ok.setdefault(str(k), []).append(p["agree"])
out["agreement"] = {k: f"{sum(v)}/{len(v)}" for k, v in ok.items()}
for p in out["points"]:
    if not p["agree"]:
        print("disagree:", p)
print(out["agreement"])
with open(os.path.join(HERE, "chatter_freq_and_points.json"), "w") as fh:
    json.dump(out, fh, indent=1)
