"""
Phase 1b -- Section 5 of the paper on the virtual testbed, out of sample.
=========================================================================
With zeta and mu_scale fitted on Fig. 16(c) ONLY (scripts/calibrate.py), every
other published sweep is a prediction:

  Fig. 16(d)  120 Hz / S2, L swept at mu = 1e-5
  Fig. 19(c,d) 280 Hz / S1, mu at L = 128 and L at mu = 1.2e-5
  Fig. 21(c,d) 85 Hz / S2, two channels, mu at L = 64 and L at mu = 6e-6
  Fig. 17(b,c) the FEM-based compensation path, 120 Hz / S2

For each point three numbers are produced:
  truth  -- full FxLMS on the virtual experiment (truth plant + noise),
  rom    -- full FxLMS on the identified model (the paper's "simulation"),
  theory -- averaged-LTI predictor on the identified model (theory.py), i.e.
            what can be known BEFORE running any adaptive loop.
Writes results/sweeps.txt, results/sweeps.npz, figures/fig_sweeps.png.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import Tee, identify, Testbed, ROOT, FIGS, RESULTS
from control import Plants, run_case, lti_t5, fem_like_model, gain_normalisation, PAPER_G2
import calib

log = Tee('sweeps.txt')
cal = calib.load()
ZETA, KAPPA = cal['zeta'], cal['mu_scale']
log(f"calibration: zeta = {ZETA:.5f}, mu_scale = {KAPPA:.4f}  "
    f"({'calibrated on Fig. 16c' if cal['calibrated'] else 'UNCALIBRATED defaults'})")

T, M, _, _ = identify(Testbed(zeta=ZETA))
P = Plants(T, M)
P_fem = Plants(T, M, comp=fem_like_model(ZETA))
log(f"ROM: p = {M.p}, r = {M.r}")

READ = os.path.join(ROOT, 'data', 'paper_readings', 'convergence_times_by_eye.csv')
with open(READ) as f:
    ROWS = list(csv.DictReader([l for l in f if not l.startswith('#')],
                               fieldnames=['fig', 'case', 'sweep', 'fixed', 'x', 't_exp', 't_sim']))


def paper(fig, filt=None):
    x, te, ts = [], [], []
    for r in ROWS:
        if r['fig'] == fig and (filt is None or filt in r['case']):
            x.append(float(r['x']))
            te.append(float(r['t_exp']) if r['t_exp'] else np.nan)
            ts.append(float(r['t_sim']) if r['t_sim'] else np.nan)
    return np.array(x), np.array(te), np.array(ts)


# (fig, f, sensor, actuators, swept, fixed value, role, plants, filter)
CURVES = [
    ('16c', 120.0, 1, [1], 'mu', 128, 'calibration', P, None),
    ('16d', 120.0, 1, [1], 'L', 1e-5, 'validation', P, None),
    ('19c', 280.0, 0, [0], 'mu', 128, 'validation', P, None),
    ('19d', 280.0, 0, [0], 'L', 1.2e-5, 'validation', P, None),
    ('21c', 85.0, 1, [0, 1], 'mu', 64, 'validation', P, None),
    ('21d', 85.0, 1, [0, 1], 'L', 6e-6, 'validation', P, None),
    ('17b', 120.0, 1, [1], 'mu', 128, 'validation (FEM path)', P_fem, 'FEM'),
    ('17c', 120.0, 1, [1], 'L', 1e-5, 'validation (FEM path)', P_fem, 'FEM'),
]

# --- per-case gain normalisation -------------------------------------------
# Tonal FxLMS adapts at a rate proportional to mu * sum_j |G_s,j(w0)|^2 (with
# G_c = G_s).  The testbed matches the paper's secondary-path gain only on
# average (one g_A for both MFCs), so at each control frequency the published
# CONVERGED voltages give the paper's sum_j |G_j|^2 = d^2 / sum_j |y_j|^2
# (Figs. 11, 13, 14 -- not used anywhere in the fit), and the testbed's comes
# from its FRF.  nu_case rescales mu so that both run at the same adaptation
# gain; it equals 1 for the calibration case by construction.
NU, G2T = gain_normalisation(T)
for f in NU:
    log(f"gain normalisation {f:g} Hz: paper sum|G|^2 = {PAPER_G2[f]:.4f}, testbed = "
        f"{G2T[f]:.4f} -> nu = {NU[f]:.3f}")

store = {}
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
for ax, ax2, (fg, f, s, acts, sw, fixed, role, PP, filt) in zip(axes.T.ravel(), axes2.T.ravel(), CURVES):
    x, te, ts = paper(fg, filt)
    xs = np.unique(np.r_[x, np.linspace(x.min(), x.max(), 25)])
    if sw == 'L':
        xs = np.unique(np.round(xs).astype(int))
    tt, tr, th, tn, thn = [], [], [], [], []
    for xv in xs:
        mu = KAPPA * (xv if sw == 'mu' else fixed)
        L = int(fixed if sw == 'mu' else xv)
        tt.append(run_case(PP, 'truth', f, s, acts, mu, L)['t_c'])
        tr.append(run_case(PP, 'rom', f, s, acts, mu, L)['t_c'])
        th.append(lti_t5(PP, 'rom', f, s, acts, mu, L)[0])
        tn.append(run_case(PP, 'truth', f, s, acts, NU[f] * mu, L)['t_c'])
        thn.append(lti_t5(PP, 'rom', f, s, acts, NU[f] * mu, L)[0])
    tt, tr, th, tn, thn = map(np.array, (tt, tr, th, tn, thn))
    store[fg] = dict(x=xs, truth=tt, rom=tr, theory=th, truth_norm=tn, theory_norm=thn,
                     paper_x=x, paper_exp=te, paper_sim=ts)
    # scores on the paper's abscissae (exclude points capped at 20 s on both sides)
    idx = np.searchsorted(xs, x)
    log(f"\n{fg} ({role}): {f:g} Hz, sensor S{s + 1}, actuators {['A%d' % (a + 1) for a in acts]},"
        f" {sw} swept, fixed {'L' if sw == 'mu' else 'mu'} = {fixed:g}")
    for name, vec, ref in (('truth vs paper exp', tt, te), ('rom vs paper sim ', tr, ts),
                           ('theory vs paper exp', th, te), ('truth vs rom      ', tt, None),
                           (f'NORMALISED (nu={NU[f]:.2f}) truth vs paper exp', tn, te),
                           (f'NORMALISED theory vs paper exp', thn, te)):
        b = vec[idx]
        r_ = tr[idx] if ref is None else ref
        ok = np.isfinite(r_) & ~((b >= 19.5) & (r_ >= 19.5))
        if ok.sum() == 0:
            continue
        err = np.abs(np.log(b[ok] / r_[ok]))
        log(f"   {name}: median |ratio-1| = {100 * (np.exp(np.median(err)) - 1):5.1f} %, "
            f"max = {100 * (np.exp(err.max()) - 1):6.1f} %  ({ok.sum()} pts)")
    for name, vec in (('paper exp', te), ('truth', tt[idx]), ('theory', th[idx]),
                      ('truth/nu', tn[idx]), ('theory/nu', thn[idx])):
        good = np.isfinite(vec)
        if good.any():
            k = np.nanargmin(np.where(good, vec, np.inf))
            eff = x[good & (vec < 5.0)]
            rng = f"[{eff.min():g}, {eff.max():g}]" if eff.size else "none"
            log(f"   {name:9s}: optimum {sw} = {x[k]:g} (t = {vec[k]:.2f} s); t < 5 s for {sw} in {rng}")
    # figure
    ax.plot(x, te, 'o', mfc='none', color='k', label='paper exp')
    if np.isfinite(ts).any():
        ax.plot(x, ts, 's', mfc='none', color='0.5', label='paper sim')
    ax.plot(xs, tt, '-', color='C0', label='testbed truth (virtual exp)')
    ax.plot(xs, tr, '--', color='C3', label='identified model (virtual sim)')
    ax.plot(xs, th, ':', color='C2', lw=2, label='LTI theory (no adaptive run)')
    ax.axhline(5, color='k', lw=0.6, ls='--')
    ax.set_ylim(0, 21)
    ax.set_xlabel(r'$\mu$ (paper units)' if sw == 'mu' else r'$L$')
    ax.set_ylabel('convergence time [s]')
    ax.set_title(f'Fig. {fg} -- {role}', fontsize=10)
    ax.grid(alpha=0.3)
    if fg == '16c':
        ax.legend(fontsize=7)
    ax2.plot(x, te, 'o', mfc='none', color='k', label='paper exp')
    if np.isfinite(ts).any():
        ax2.plot(x, ts, 's', mfc='none', color='0.5', label='paper sim')
    ax2.plot(xs, tn, '-', color='C0', label='testbed truth, gain-normalised')
    ax2.plot(xs, thn, ':', color='C2', lw=2, label='LTI theory, gain-normalised')
    ax2.axhline(5, color='k', lw=0.6, ls='--')
    ax2.set_ylim(0, 21)
    ax2.set_xlabel(r'$\mu$ (paper units)' if sw == 'mu' else r'$L$')
    ax2.set_ylabel('convergence time [s]')
    ax2.set_title(f'Fig. {fg} -- {role}, nu = {NU[f]:.2f}', fontsize=10)
    ax2.grid(alpha=0.3)
    if fg == '16c':
        ax2.legend(fontsize=7)

fig.suptitle(f'Virtual testbed vs published curves (zeta = {ZETA:.4f}, '
             f'mu_testbed = {KAPPA:.3f} mu_paper; only Fig. 16c used for fitting)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_sweeps.png'), dpi=130)
fig2.suptitle('Same, with mu rescaled per case by the published converged voltages '
              '(Figs. 11/13/14): equal adaptation gain sum|G_s|^2', fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGS, 'fig_sweeps_normalised.png'), dpi=130)
np.savez(os.path.join(RESULTS, 'sweeps.npz'), **{f'{k}_{kk}': v for k, d in store.items() for kk, v in d.items()})
log("\nsaved figures/fig_sweeps.png and results/sweeps.npz")
