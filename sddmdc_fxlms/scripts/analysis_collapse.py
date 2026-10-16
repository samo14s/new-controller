"""
H1 on the paper's OWN data -- no model involved.
=================================================
Hypothesis (pre-declared, docs/03): for a tonal disturbance the FxLMS
convergence time depends on (mu, L) only through beta ~ mu * L_eff, where
L_eff = 2 lambda_min(Gamma(L, w)) is L corrected for the finite-length
eigenvalue split (theory.py).  Then each case's two published sweeps -- mu
swept at fixed L, and L swept at fixed mu -- must fall on ONE curve t(mu L_eff).

Test: interpolate the mu-sweep (log-log, as a function of mu*L_eff) at the
mu*L_eff of every L-sweep point inside the overlap, and report the relative
discrepancy.  Pass threshold declared before running: median |discrepancy|
<= 10 % for the corrected variable, for the experimental AND simulated curves
of all three cases.  Points at the 20 s cap are excluded.

Data: data/paper_readings/convergence_times_by_eye.csv (readings of Figs. 16,
17, 19, 21; +-0.1 s).  Writes results/collapse.txt and figures/fig_collapse.png.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import Tee, ROOT, FIGS
import theory

log = Tee('collapse.txt' if os.environ.get('COLLAPSE_SOURCE', 'eye') == 'eye' else 'collapse_pixel.txt')
READ = os.path.join(ROOT, 'data', 'paper_readings', 'convergence_times_by_eye.csv')
FS = 1e4
CASES = [  # (label, f_exc, mu-sweep fig, its L, L-sweep fig, its mu)
    ('120 Hz / S2 (single)', 120.0, '16c', 128, '16d', 1e-5),
    ('280 Hz / S1 (single)', 280.0, '19c', 128, '19d', 1.2e-5),
    ('85 Hz / S2 (two-channel)', 85.0, '21c', 64, '21d', 6e-6),
    ('120 Hz / S2, FEM path (exp)', 120.0, '17b', 128, '17c', 1e-5),
]


def load():
    with open(READ) as f:
        lines = [l for l in f if not l.startswith('#')]
    hdr = ['fig', 'case', 'sweep', 'fixed', 'x', 't_exp', 't_sim']
    return list(csv.DictReader(lines, fieldnames=hdr))


rows = load()


SOURCE = os.environ.get('COLLAPSE_SOURCE', 'eye')     # 'eye' | 'pixel'
DIGIT = os.path.join(ROOT, 'data', 'digitized')
log(f"data source: {'by-eye readings' if SOURCE == 'eye' else 'pixel digitisation (data/digitized/*_best.csv)'}")


def series(fig, col, case_filter=None):
    if SOURCE == 'pixel':
        name = {('t_exp', None): 'Experiment', ('t_sim', None): 'Simulation',
                ('t_exp', 'FEM'): 'Exp_FEM'}[(col, case_filter)]
        out = []
        with open(os.path.join(DIGIT, f'fig{fig}_best.csv')) as fh:
            for r in csv.DictReader(fh):
                if r['series'] == name:
                    out.append((float(r['x_nominal']), float(r['y'])))
        return np.array(sorted(out))
    out = []
    for r in rows:
        if r['fig'] != fig or not r[col]:
            continue
        if case_filter and case_filter not in r['case']:
            continue
        out.append((float(r['x']), float(r[col])))
    return np.array(out)


def leff(L, f):
    return 2 * theory.gram(int(round(L)), 2 * np.pi * f / FS)[1].min()


fig, axes2 = plt.subplots(2, 4, figsize=(17, 8.2))
verdicts = []
raw = []
for ci, (label, f, fmu, Lfix, fL, mufix) in enumerate(CASES):
    log(f"\n=== {label}: {fmu} (mu swept, L = {Lfix})  vs  {fL} (L swept, mu = {mufix:g}) ===")
    cols = ['t_exp'] if fmu == '17b' else ['t_exp', 't_sim']
    for col, mk in zip(cols, ['o', 's']):
        filt = 'FEM' if fmu == '17b' else None
        A = series(fmu, col, filt)
        B = series(fL, col, filt)
        A = A[A[:, 1] < 19.5]
        B = B[B[:, 1] < 19.5]
        for corr in (False, True):
            xa = A[:, 0] * (leff(Lfix, f) if corr else Lfix)
            xb = mufix * np.array([leff(L, f) if corr else L for L in B[:, 0]])
            o = np.argsort(xa)
            xa_s, ta_s = xa[o], A[o, 1]
            inside = (xb >= xa_s.min()) & (xb <= xa_s.max())
            ti = np.exp(np.interp(np.log(xb[inside]), np.log(xa_s), np.log(ta_s)))
            disc = (B[inside, 1] - ti) / ti
            tag = 'mu*L_eff' if corr else 'mu*L    '
            log(f"  {col} {tag}: {inside.sum():2d} overlapping points, median |disc| = "
                f"{100 * np.median(np.abs(disc)):5.1f} %, max = {100 * np.max(np.abs(disc)):5.1f} %"
                f"  [L in {B[inside, 0].min():.0f}..{B[inside, 0].max():.0f}]")
            if not corr:
                raw.append((label, col, np.median(np.abs(disc))))
            if corr:
                verdicts.append((label, col, np.median(np.abs(disc))))
            ax = axes2[int(corr), ci]
            if True:
                c = 'C0' if col == 't_exp' else 'C3'
                ax.loglog(xa, A[:, 1], mk + '-', color=c, mfc='none', ms=5,
                          label=f'{col[2:]}: mu swept (L={Lfix})')
                ax.loglog(xb, B[:, 1], mk, color=c, ms=5, label=f'{col[2:]}: L swept (mu={mufix:g})')
    for row, xl in ((0, r'$\mu\,L$  (raw product)'), (1, r'$\mu\,L_{\rm eff}$,  $L_{\rm eff}=2\lambda_{\min}(\Gamma)$')):
        ax = axes2[row, ci]
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel('convergence time [s]')
        ax.axhline(5, color='k', lw=0.6, ls='--')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7)
fig.suptitle('Ouyang et al. (2026), published convergence times re-plotted against '
             r'$\mu L$ (top) and $\mu L_{\rm eff}$ (bottom); open: $\mu$ swept, filled: $L$ swept', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_collapse.png' if SOURCE == 'eye' else 'fig_collapse_pixel.png'), dpi=140)

log("\n=== Verdict (pre-declared threshold: median |disc| <= 10 % with L_eff) ===")
ok = True
for label, col, m in verdicts:
    passed = m <= 0.10
    ok &= passed
    log(f"  {label:32s} {col}: {100 * m:5.1f} %  {'PASS' if passed else 'FAIL'}")
log(f"  H1 on published data: {'PASS' if ok else 'FAIL'}")

log("\n=== POST-HOC (not pre-declared): the same test with the raw product mu*L ===")
for label, col, m in raw:
    log(f"  {label:32s} {col}: {100 * m:5.1f} %  {'<= 10 %' if m <= 0.10 else '> 10 %'}")
log("  Reading: theory.py shows lambda_min(Gamma) is the decay rate of the SLOW"
    " eigen-direction only; the 5 % time mixes both directions, so L_eff = 2 lambda_min"
    " over-corrects and the exact correction lies between L and L_eff.  The raw"
    " product is the trace of Gamma, i.e. the averaged gain.")
