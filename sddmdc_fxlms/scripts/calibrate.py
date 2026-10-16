"""
Phase 0b -- calibrate the virtual testbed on PUBLISHED numbers only.
=====================================================================
Three steps, each on a disjoint subset of the paper's data, so that the rest
of the paper stays available as out-of-sample validation:

  1. moduli E1, E2, G12 -> the six measured frequencies of Table 4;
  2. electrical gains   -> the six sine amplitudes of Fig. 8 (PZT path) and the
                           two converged actuating voltages of Figs. 11 / 13;
  3. damping zeta and step-size scale mu_scale -> the EXPERIMENTAL curve of
                           Fig. 16(c) only (120 Hz, S2, L = 128, mu swept).
Figs. 16(d), 19, 21 (and the maps 18, 20, 23) are NOT used here: they are the
validation set of scripts/run_sweeps.py.

Step 3 uses the averaged-LTI predictor of theory.py (verified against the
full FxLMS to a few per cent in docs/04) because it costs milliseconds; the
fitted pair is then checked with the full time-domain FxLMS.

Writes results/calibration.txt and results/calibration.json.
"""
import csv
import os

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from _common import Tee, identify, Testbed, ROOT
from plate import build_paper_plate, PLATE, F_EXP
from control import Plants, run_case, lti_t5
import calib

log = Tee('calibration.txt')
READ = os.path.join(ROOT, 'data', 'paper_readings', 'convergence_times_by_eye.csv')


def readings(fig, col='t_exp'):
    rows = []
    with open(READ) as f:
        lines = [l for l in f if not l.startswith('#')]
    hdr = ['fig', 'case', 'sweep', 'fixed', 'x', 't_exp', 't_sim']
    for r in csv.DictReader(lines, fieldnames=hdr):
        if r['fig'] == fig and r[col]:
            rows.append((float(r['x']), float(r[col])))
    return np.array(rows)


# ---------------------------------------------------------------- step 1
log("=== Step 1: moduli -> Table 4 measured frequencies ===")


def freqs(s):
    p = dict(PLATE, E1=PLATE['E1'] * np.exp(s[0]), E2=PLATE['E2'] * np.exp(s[1]),
             G12=PLATE['G12'] * np.exp(s[2]))
    return build_paper_plate(M=12, N=9, n_modes=8, plate=p).freq[:6]


f0 = freqs([0, 0, 0])
r = least_squares(lambda s: np.log(freqs(s) / F_EXP), [0, 0, 0], diff_step=1e-3)
f1 = freqs(r.x)
E1, E2, G12 = (PLATE[k] * np.exp(v) for k, v in zip(('E1', 'E2', 'G12'), r.x))
log(f" Table 2 moduli : f = {np.round(f0, 2)}  max err {100 * np.max(np.abs(f0 / F_EXP - 1)):.1f} %")
log(f" updated moduli : E1 = {E1 / 1e9:.3f} GPa, E2 = {E2 / 1e9:.3f} GPa, G12 = {G12 / 1e9:.3f} GPa")
log(f"                  f = {np.round(f1, 2)}  errors % = {np.round(100 * (f1 / F_EXP - 1), 2)}")
log("                  (paper's FEM: max error 4.43 %).  The truth then pins the six")
log("                  frequencies to Table 4 exactly (testbed.f_override) and keeps these shapes.")

# ---------------------------------------------------------------- step 2
log("\n=== Step 2: electrical gains (values hard-coded in testbed.py) ===")
T = Testbed()
f_anchor = np.array([20, 245, 280, 35, 85, 120.])
exp_amp = np.array([0.225, 0.688, 0.341, 0.0190, 0.410, 0.272])
H = T.frf(f_anchor, out_rows=T.iv)
model = np.r_[np.abs(H[:3, 0, 0]), np.abs(H[3:, 1, 0])] * 5
log(f" S_sens = {T.s_sens:.3g} V/strain, g_E = {T.g[0]:.2f}, g_A = {T.g[1]:.1f}")
log(" Fig. 8 amplitudes, testbed / paper: " + " ".join(f"{a:.2f}" for a in model / exp_amp)
    + "   (geometric mean %.2f)" % np.exp(np.mean(np.log(model / exp_amp))))
log("  -> the PZT path is matched only on average: individual ratios span x0.2..x2.1."
    " The testbed is a plausible specimen, not a replica.")
Hs = T.frf([120., 280.], out_rows=T.iv)
log(f" |Gs| A2->S2 @120 Hz = {abs(Hs[0, 1, 2]):.4f} (paper 0.272/1.570 = {0.272 / 1.570:.4f}),"
    f" A1->S1 @280 Hz = {abs(Hs[1, 0, 1]):.4f} (paper 0.341/1.160 = {0.341 / 1.160:.4f})")
log(" single-channel actuator not stated in the paper: A2->S2 and A1->S1 assumed (nearest pair).")

# ---------------------------------------------------------------- step 3
log("\n=== Step 3: zeta and mu_scale -> Fig. 16(c) experimental curve only ===")
d16c = readings('16c')
keep = d16c[:, 1] < 19.5
mu_p, t_p = d16c[keep, 0], d16c[keep, 1]
log(f" {keep.sum()} points, mu = {mu_p.min():.1e} .. {mu_p.max():.1e}")


def curve(P, scale, mus, L=128, which='truth'):
    return np.array([lti_t5(P, which, 120.0, 1, 1, scale * m, L)[0] for m in mus])


results = []
for zeta in [0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006]:
    T, M, _, _ = identify(Testbed(zeta=zeta))
    P = Plants(T, M)

    def cost(ls):
        t = curve(P, np.exp(ls), mu_p)
        return np.mean((np.log(t) - np.log(t_p))**2)

    res = minimize_scalar(cost, bounds=(np.log(0.05), np.log(2.0)), method='bounded',
                          options=dict(xatol=1e-3))
    rms = np.sqrt(res.fun)
    results.append((zeta, np.exp(res.x), rms))
    log(f" zeta = {zeta:.4f}: best mu_scale = {np.exp(res.x):.4f}, rms log error = {rms:.3f}"
        f"  (~{100 * (np.exp(rms) - 1):.0f} %)")

zb, sb, eb = min(results, key=lambda r: r[2])
# refine zeta locally
zs = np.linspace(max(0.001, zb - 0.0006), zb + 0.0006, 7)
for zeta in zs:
    if any(abs(zeta - r[0]) < 1e-6 for r in results):
        continue
    T, M, _, _ = identify(Testbed(zeta=zeta))
    P = Plants(T, M)
    res = minimize_scalar(lambda ls: np.mean((np.log(curve(P, np.exp(ls), mu_p)) - np.log(t_p))**2),
                          bounds=(np.log(0.05), np.log(2.0)), method='bounded', options=dict(xatol=1e-3))
    results.append((zeta, np.exp(res.x), np.sqrt(res.fun)))
    log(f" zeta = {zeta:.5f}: best mu_scale = {np.exp(res.x):.4f}, rms log error = {np.sqrt(res.fun):.3f}")
zb, sb, eb = min(results, key=lambda r: r[2])
log(f"\n SELECTED: zeta = {zb:.5f}, mu_scale = {sb:.4f} (1/mu_scale = {1 / sb:.2f}), rms log err {eb:.3f}")

# full time-domain check of the selected pair
T, M, _, _ = identify(Testbed(zeta=zb))
P = Plants(T, M)
log("\n check with the full FxLMS (virtual experiment), Fig. 16(c) points:")
log("  mu_paper | paper exp | LTI theory | full FxLMS truth | full FxLMS rom (=simulation) | paper sim")
d16s = readings('16c', 't_sim')
simmap = dict(d16s)
for m, tp in zip(d16c[:, 0], d16c[:, 1]):
    tl = lti_t5(P, 'truth', 120.0, 1, 1, sb * m, 128)[0]
    tf = run_case(P, 'truth', 120.0, 1, 1, sb * m, 128)['t_c']
    tr = run_case(P, 'rom', 120.0, 1, 1, sb * m, 128)['t_c']
    log(f"  {m:.2e} | {tp:6.2f}    | {tl:6.2f}     | {tf:6.2f}           | {tr:6.2f}                      | {simmap.get(m, float('nan')):6.2f}")

calib.save(dict(zeta=float(zb), mu_scale=float(sb), rms_log_err=float(eb),
                E1=float(E1), E2=float(E2), G12=float(G12),
                fitted_on='Fig. 16(c) experimental curve (120 Hz, S2, L=128)',
                grid=[dict(zeta=float(a), mu_scale=float(b), rms=float(c)) for a, b, c in results]))
log(f"\n saved {calib.CAL_FILE}")
