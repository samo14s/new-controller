"""Diagnostic: forced demand against the measured command of Ozsoy et al. 2025 for other helix angles.

Purpose.  At 10 and 15 mm the measured command of the saturation tests departs from the forced demand of
the model in a pattern that the article's own frequency-domain model shares (its Fig. 12): the measured
minimum lies near 15 mm, while the tooth-passing force of a 45 degree helix (16 mm cutter, four flutes)
is smallest at the axial pitch pi D / (N tan 45) = 12.6 mm.  This script recomputes the forced demand of
the 36 unsaturated tests (re-clamped structure, g = 253 V s/m) with helix angles 36-46 degrees, everything
else unchanged, as a diagnostic of where the discrepancy comes from (the axial distribution of the
cutting force: helix, or a mode shape that is not uniform over the depth).  It changes nothing in the
model of the manuscript, which keeps the 45 degrees of the article.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/ozsoy2025/helix2025.py   (writes helix2025.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup2025 as S  # noqa: E402
import params as P  # noqa: E402
import feasibility as F  # noqa: E402


def main():
    pts = [q for q in json.load(open(os.path.join(HERE, "points_fig10.json")))["points"]
           if str(q["value"]).strip().lower() != "sat"]
    S.apply_structure(**S.RECLAMPED)
    h0 = P.HELIX
    out = dict(note=__doc__.split("\n\n")[0], n=len(pts), helix={})
    for deg in (36.0, 38.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0):
        P.HELIX = np.deg2rad(deg)
        F._W = F.slice_force_harmonics()
        rat = np.array([F.forced_demand(float(q["rpm"]), 253.0, np.array([q["depth_mm"] * 1e-3]))[0] / float(q["value"])
                        for q in pts])
        out["helix"][f"{deg:g}"] = dict(median_ratio=float(np.median(rat)), q25=float(np.quantile(rat, 0.25)),
                                        q75=float(np.quantile(rat, 0.75)), mean_abs_log=float(np.mean(np.abs(np.log(rat)))),
                                        pitch_mm=float(np.pi * P.D_TOOL / (P.N_TEETH * np.tan(P.HELIX)) * 1e3))
    P.HELIX = h0
    F._W = F.slice_force_harmonics()
    S.apply_structure(**S.NOMINAL)
    best = min(out["helix"], key=lambda k: out["helix"][k]["mean_abs_log"])
    out["best_helix_deg"] = float(best)
    with open(os.path.join(HERE, "helix2025.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["helix"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    print("best", best)


if __name__ == "__main__":
    main()
