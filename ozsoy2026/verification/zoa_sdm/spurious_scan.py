"""How often does src/zoa.py::a_lim return a tooth-passing pole as a root?

At w = 2 pi j / T the factor (1 - e^{-iwT}) vanishes, a(w) has a pole and
Im a(w) changes sign.  zoa.py rejects such sign changes only when
|Im a| >= 0.5 |a| at the left grid point; near the pole a ~ -i K / (w T - 2 pi j)
with K = 1/(c Gcl T w), so |Im a|/|a| = |cos(arg Gcl)| and the filter lets
the pole through whenever arg Gcl lies between -120 and -60 deg, i.e. when a
tooth-passing harmonic falls inside the resonance band of the closed loop.
The interpolated "root" then has an arbitrary (small positive) real part.

Monte Carlo over the paper's speed grid (500:10:3000 rpm) and gains drawn
uniformly in [0, 2000] V s/m; reference = polished lobe sweep.
Writes spurious_scan.csv (flagged cases) and spurious_scan.json (rates).
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
import model as M
import zoa_lobes as L

rng = np.random.default_rng(12345)
NDRAW = 3000
summary = {}
flagged = []
for zeta in (0.0134, 0.0145):
    P.set_damping(zeta)
    rpm = rng.choice(P.RPM_GRID, NDRAW).astype(float)
    gg = rng.uniform(0, 2000, NDRAW)
    nflag = 0
    for n, g in zip(rpm, gg):
        a, fc = zoa.a_lim(n, g, return_freq=True)
        T = 60.0 / (M.NT * n)
        w = 2 * np.pi * fc
        res = abs(1 - L.C_CUT * a * (1 - np.exp(-1j * w * T)) * M.Gcl(1j * w, g, zeta))
        dh = abs(fc * T - np.round(fc * T)) / T          # distance to a tooth harmonic, Hz
        if res > 1e-2:
            nflag += 1
            ref, fref, _ = L.envelope([n], g, zeta)
            flagged.append(dict(zeta=zeta, rpm=n, g=round(g, 3), a_src_mm=a * 1e3, fc_src_Hz=fc,
                                dist_to_harmonic_Hz=dh, residual=res,
                                a_ref_mm=ref[0] * 1e3, fc_ref_Hz=fref[0],
                                rel=a / ref[0] - 1))
            print(flagged[-1], flush=True)
    summary[str(zeta)] = dict(n_draws=NDRAW, n_spurious=nflag, rate=nflag / NDRAW)
# deterministic paper-grid scan at the four check gains
grid = {}
for zeta in (0.0134, 0.0145):
    P.set_damping(zeta)
    for g in (0.0, 300.0, 800.0, 1500.0):
        a, fc = zoa.sld(P.RPM_GRID, g, return_freq=True)
        T = 60.0 / (M.NT * P.RPM_GRID)
        res = np.abs(1 - L.C_CUT * a * (1 - np.exp(-2j * np.pi * fc * T)) * M.Gcl(2j * np.pi * fc, g, zeta))
        grid[f"{zeta}_{int(g)}"] = int(np.sum(res > 1e-2))
summary["paper_grid_10rpm_spurious_counts"] = grid
with open(os.path.join(HERE, "spurious_scan.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
if flagged:
    with open(os.path.join(HERE, "spurious_scan.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(flagged[0].keys()))
        wr.writeheader()
        wr.writerows(flagged)
print(json.dumps(summary, indent=1))
