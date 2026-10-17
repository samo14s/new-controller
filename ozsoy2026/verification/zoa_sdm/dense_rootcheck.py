"""Missed/spurious-root check of src/zoa.py::a_lim on a 1-rpm grid.

Reference: the lower envelope of the classic lobe sweep (zoa_lobes.py),
which parametrises the roots by frequency and lobe number instead of
scanning Im a(w) at fixed speed.  Writes dense_rootcheck.csv (summary per
case) and dense_sld_<zeta>_<g>.csv is not written to keep the folder small;
the worst speeds per case are listed instead.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import json
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
import params as P
import zoa
import zoa_lobes as L

RPMS = np.arange(500, 3001, 1.0)
rows = []
dense = {}
for zeta in (0.0134, 0.0145):
    P.set_damping(zeta)
    for g in (0.0, 300.0, 800.0, 1500.0, 3000.0, 6000.0):
        a_src, f_src = zoa.sld(RPMS, g, return_freq=True)
        a_lob, f_lob, k_lob = L.envelope(RPMS, g, zeta)
        rel = a_src / a_lob - 1
        bad = np.abs(rel) > 1e-3
        j = np.argsort(-np.abs(rel))[:3]
        rows.append(dict(
            zeta=zeta, g=g, n_speeds=RPMS.size,
            max_abs_rel=float(np.max(np.abs(rel))),
            median_abs_rel=float(np.median(np.abs(rel))),
            n_rel_gt_1e3=int(bad.sum()),
            n_src_larger=int((rel > 1e-3).sum()),     # src missed the lowest root
            n_src_smaller=int((rel < -1e-3).sum()),   # src found a root the lobes do not have
            max_abs_dfc_Hz=float(np.nanmax(np.abs(f_src - f_lob))),
            worst_rpm=";".join(f"{RPMS[i]:.0f}" for i in j),
            worst_rel=";".join(f"{rel[i]:.2e}" for i in j),
            min_a_mm=float(a_lob.min() * 1e3), max_a_mm=float(a_lob.max() * 1e3),
        ))
        if g in (0.0, 1500.0) and zeta == 0.0134:
            dense[str(int(g))] = dict(a_src_mm=(a_src * 1e3).tolist(), a_lobes_mm=(a_lob * 1e3).tolist(),
                                      fc_src=f_src.tolist(), fc_lobes=f_lob.tolist())
        print(rows[-1], flush=True)
with open(os.path.join(HERE, "dense_rootcheck.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(HERE, "dense_sld_z0134.json"), "w") as fh:
    json.dump(dict(rpm=RPMS.tolist(), **dense), fh)
