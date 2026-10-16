"""
Open question of docs/03 section 4.4: does the virtual testbed reproduce the
bending of the FAST edge of the 120 Hz effective region (published slope
b = -0.72 instead of -1), or is its map an exact mu*L hyperbola?

Full FxLMS on the truth plant (no averaging), on the L rows tested in the
paper and a fine mu grid; per row the edges of {t5 < 5 s} are located by
linear interpolation, exactly as the paper's contour plots do.  Then the
same slope statistic as data/digitized/maps_muL_analysis.txt.
Writes results/map_probe.txt.
"""
import numpy as np

from _common import Tee, identify, Testbed
from control import Plants, run_case
import calib

log = Tee('map_probe.txt')
cal = calib.load()
T, M, _, _ = identify(Testbed(zeta=cal['zeta']))
P = Plants(T, M)
KAPPA = cal['mu_scale']
ROWS = [40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140, 144, 150, 160]
CASES = [('120 Hz / S2 / A2', 120.0, 1, [1], np.arange(1.0, 20.01, 0.5)),
         ('280 Hz / S1 / A1', 280.0, 0, [0], np.arange(0.5, 24.01, 0.5))]


def edges(mu, t, thr=5.0):
    inside = t < thr
    if not inside.any():
        return np.nan, np.nan, np.nan
    i0, i1 = np.argmax(inside), len(inside) - 1 - np.argmax(inside[::-1])
    lo = mu[i0] if i0 == 0 else np.interp(thr, [t[i0], t[i0 - 1]], [mu[i0], mu[i0 - 1]])
    hi = mu[i1] if i1 == len(mu) - 1 else np.interp(thr, [t[i1], t[i1 + 1]], [mu[i1], mu[i1 + 1]])
    return lo, hi, mu[np.argmin(t)]


for label, f, s, acts, mus in CASES:
    log(f"\n=== {label}: testbed truth, full FxLMS, mu in paper units x1e-6 ===")
    log("   L   | lower mu*L | upper mu*L | ridge mu*L   (x1e-6)")
    rec = []
    for L in ROWS:
        t = np.array([run_case(P, 'truth', f, s, acts, KAPPA * m * 1e-6, L)['t_c'] for m in mus])
        lo, hi, rd = edges(mus, t)
        hit_edge = np.isfinite(hi) and hi >= mus[-1]
        rec.append((L, lo * L, np.nan if hit_edge else hi * L, rd * L))
        log(f"  {L:4d} | {lo * L:10.0f} | {'axis' if hit_edge else f'{hi * L:10.0f}'} | {rd * L:10.0f}")
    rec = np.array(rec, float)
    for j, name in ((1, 'lower'), (2, 'upper'), (3, 'ridge')):
        ok = np.isfinite(rec[:, j])
        if ok.sum() >= 3:
            Ls, muL = rec[ok, 0], rec[ok, j]
            b = np.polyfit(np.log(Ls), np.log(muL / Ls), 1)[0]
            log(f"  {name:5s} edge: mean mu*L = {muL.mean():.0f}e-6, CV = {100 * muL.std() / muL.mean():.1f} %,"
                f" slope b = {b:+.2f} over L = {Ls.min():.0f}..{Ls.max():.0f}  (paper: see maps_muL_analysis.txt)")
