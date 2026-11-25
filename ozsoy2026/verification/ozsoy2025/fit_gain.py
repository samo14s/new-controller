"""Linear limits of the model against the digitised stability lobes of Ozsoy et al. 2025; DVF gain of its Fig. 10.

Purpose.  (i) Its Fig. 6 shows the uncontrolled and the DVF-controlled stability limit (g = 253 V s/m,
its Table 1) of the structure of its Table 2, which is the structure of src/params.py: the model's
zoa.a_lim(n, 0) and zoa.a_lim(n, 253) are compared with the digitised curves.  (ii) Its Fig. 10 shows
the same two limits for the re-clamped structure (128.1 Hz, 1.48 %, 1.13e7 N/m) with a gain 'optimised
for this modification' that the article does not state: the uncontrolled curve tests the re-clamped
parameters, and the gain is estimated by least squares of the controlled curve over a grid of gains
(0-600 V s/m in steps of 1), excluding the steep flank of the lobe (where a small error in speed is a
large error in depth: points whose model slope exceeds 0.2 mm/rpm).

Inputs: points_fig6.json and points_fig10.json ('controlled_sld' and 'uncontrolled_sld': lists of
[rpm, mm], null where off the axes).
Run:  cd ozsoy2026 && python3 verification/ozsoy2025/fit_gain.py   (writes fit_gain.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup2025 as S  # noqa: E402
import zoa  # noqa: E402


def curve(c):
    c = [(float(r), float(m)) for r, m in c if r is not None and m is not None]
    return np.array([x[0] for x in c]), np.array([x[1] for x in c])


def compare(rpm, mm, g, slope_max=0.2):
    model = np.array([zoa.a_lim(n, g) * 1e3 for n in rpm])
    eps = 2.0
    slope = np.array([abs(zoa.a_lim(n + eps, g) - zoa.a_lim(n - eps, g)) * 1e3 / (2 * eps) for n in rpm])
    keep = np.isfinite(model) & (slope <= slope_max)
    err = model[keep] - mm[keep]
    return dict(n=int(keep.sum()), rms_mm=float(np.sqrt(np.mean(err ** 2))), median_abs_mm=float(np.median(np.abs(err))),
                max_abs_mm=float(np.max(np.abs(err))), median_rel=float(np.median(np.abs(err) / mm[keep])),
                rows=[[float(r), float(m), float(x)] for r, m, x, k in zip(rpm, mm, model, keep)])


def main():
    out = dict(note=__doc__.split("\n\n")[0])
    p6 = json.load(open(os.path.join(HERE, "points_fig6.json")))
    S.apply_structure(**S.NOMINAL)
    out["fig6"] = dict(uncontrolled=compare(*curve(p6["uncontrolled_sld"]), 0.0),
                       controlled_g253=compare(*curve(p6["controlled_sld"]), S.G_FIG6))
    p10 = json.load(open(os.path.join(HERE, "points_fig10.json")))
    S.apply_structure(**S.RECLAMPED)
    out["fig10_uncontrolled"] = compare(*curve(p10["uncontrolled_sld"]), 0.0)
    rpm, mm = curve(p10["controlled_sld"])
    gains = np.arange(0.0, 601.0, 1.0)
    rms = []
    for g in gains:
        c = compare(rpm, mm, g)
        rms.append(c["rms_mm"])
    j = int(np.argmin(rms))
    out["fig10_gain_fit"] = dict(g_best=float(gains[j]), rms_best_mm=float(rms[j]),
                                 rms_at_253_mm=float(rms[253]), g_within_10pct_of_best_rms=[float(gains[k]) for k in range(len(gains)) if rms[k] <= 1.1 * rms[j]][::10],
                                 at_best=compare(rpm, mm, gains[j]), at_253=compare(rpm, mm, 253.0))
    S.apply_structure(**S.NOMINAL)
    with open(os.path.join(HERE, "fit_gain.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (v if k == "note" else {kk: (vv if not isinstance(vv, dict) else {a: b for a, b in vv.items() if a != "rows"}) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
