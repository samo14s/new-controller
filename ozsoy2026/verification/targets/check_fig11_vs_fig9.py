"""Calibration check of the Fig. 11 raster digitisation.

The uncontrolled SLD is the same curve in Figs. 9 (vector, exact) and 11 (raster).
Compare the Fig. 11 dashed-curve pixels with the Fig. 9 vector curve on its flat
parts (|slope| < 0.02 mm/rpm) where the vertical position is well defined.
"""
import csv, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
v = [r for r in csv.DictReader(open(os.path.join(HERE, "fig9_curves.csv"))) if r["curve"] == "uncontrolled"]
R9 = np.array([float(r["rpm"]) for r in v]); A9 = np.array([float(r["depth_mm"]) for r in v])
o = np.argsort(R9); R9, A9 = R9[o], A9[o]
f = [r for r in csv.DictReader(open(os.path.join(HERE, "fig11_curves.csv"))) if r["curve"] == "uncontrolled"]
R11 = np.array([float(r["rpm"]) for r in f]); A11 = np.array([float(r["depth_mm_centre"]) for r in f])
slope = np.gradient(A9, R9)
s11 = np.interp(R11, R9, np.abs(slope))
m = (s11 < 0.02) & (R11 > 420) & (R11 < 3050)
d = A11[m] - np.interp(R11[m], R9, A9)
# best horizontal shift (rpm) minimising the mismatch on sloped-but-not-steep parts
m2 = (s11 > 0.005) & (s11 < 0.05) & (R11 > 420)
shifts = np.arange(-15, 15.1, 0.5)
cost = [np.median(np.abs(A11[m2] - np.interp(R11[m2] + s, R9, A9))) for s in shifts]
res = dict(n_flat=int(m.sum()), depth_bias_mm=round(float(np.median(d)), 3),
           depth_mad_mm=round(float(np.median(np.abs(d - np.median(d)))), 3),
           depth_p95_abs_mm=round(float(np.percentile(np.abs(d), 95)), 3),
           best_rpm_shift=float(shifts[int(np.argmin(cost))]), n_sloped=int(m2.sum()),
           median_abs_mm_at_best_shift=round(float(min(cost)), 3),
           median_abs_mm_at_zero_shift=round(float(cost[list(shifts).index(0.0)]), 3))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(HERE, "fig11_calibration_check.json"), "w"), indent=1)
