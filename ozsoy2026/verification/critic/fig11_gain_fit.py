"""Does the Fig. 11 single-gain SLD equal the ZOA SLD at g = 237 V s/m?

Uses the raster-digitised Fig. 11 curves from verification/targets (controlled
and uncontrolled) and fits the constant gain that best explains the controlled
curve (rms of log ratio, points off steep lobe flanks only).
Writes fig11_gain_fit.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from zoa import sld  # noqa: E402
import params as P   # noqa: E402

rows = np.genfromtxt(os.path.join(HERE, "..", "targets", "fig11_curves.csv"), delimiter=",",
                     names=True, dtype=None, encoding=None)
out = {"zeta": P.ZETA}
for cname in ("uncontrolled", "controlled"):
    sel = rows[rows["curve"] == cname]
    n = sel["rpm"].astype(float)
    a = sel["depth_mm_centre"].astype(float)
    top, bot = sel["depth_mm_run_top"].astype(float), sel["depth_mm_run_bottom"].astype(float)
    keep = (n >= 500) & (n <= 3000) & ((top - bot) < 0.6) & (a > 0.5) & (a < 30)
    n, a = n[keep], a[keep]
    gains = np.arange(0, 401, 5.0) if cname == "controlled" else np.array([0.0])
    res = []
    for g in gains:
        m = sld(n, g) * 1e3
        lr = np.log(a / m)
        res.append((g, float(np.sqrt(np.mean(lr**2))), float(np.median(lr))))
    res = np.array(res)
    j = int(np.argmin(res[:, 1]))
    entry = dict(n_points=int(n.size), best_g=float(res[j, 0]), best_rms_log=res[j, 1],
                 best_median_log=res[j, 2])
    if cname == "controlled":
        i237 = int(np.argmin(np.abs(res[:, 0] - 235)))
        m237 = sld(n, 237.0) * 1e3
        lr = np.log(a / m237)
        entry.update(g237_rms_log=float(np.sqrt(np.mean(lr**2))), g237_median_log=float(np.median(lr)),
                     g237_p90_abs_rel=float(np.percentile(np.abs(np.expm1(lr)), 90)),
                     scan=[[float(x) for x in r] for r in res[::4]])
        mb = sld(n, res[j, 0]) * 1e3
        entry["minima_plot_vs_model_best"] = [float(a.min()), float(mb.min())]
        entry["minima_plot_vs_model_237"] = [float(a.min()), float(m237.min())]
    out[cname] = entry
with open(os.path.join(HERE, "fig11_gain_fit.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "scan"} if isinstance(v, dict) else v
                  for k, v in out.items()}, indent=1))
