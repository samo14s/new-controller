"""Paper figures 5-6 regenerated for the lobe-first champions.

Sources: results/lobes_l2.npz (full-resolution lobes of every generation;
the champion keys are read from results/lobe_champions.pkl),
results/timeresp.npz (open loop + benchmark traces) and
results/timeresp_l.npz (champion traces, judge_lobe1 Stage A).
"""
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.28, 'grid.linewidth': 0.6,
    'lines.linewidth': 1.6, 'figure.dpi': 150})
C1, C2, C3, GY = '#7C3AED', '#C2410C', '#0891B2', '0.45'
import os
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, 'paper', 'figs')
RES = os.path.join(ROOT, 'results')

# the paper's members: the gate-passing lobe-first filtered champion, the
# incumbent realizable member (the widened-box lobe winner ps_ac_ri_l2
# fails G1(real) and is reported in the text, not carried in the figures),
# and the delay-exploiting family member (dashed: out of both the DI
# contract and the gate -- the pure-performance corner of the frontier)
KF, KR, KT = 'ps_ac_rfa_l2', 'ps_ac_ri', 'ps_tdc_j'
C4 = '#15803D'
SERIES = (('open', GY, 'open loop'),
          ('mu_tdc', C1, '$\\mu$-TDC (benchmark)'),
          (KF, C3, 'PS-AC-RFA-L'),
          (KR, C2, 'PS-AC-RI'),
          (KT, C4, 'PS-TDC-J (delay-expl.)'))

# ---- Fig 5: stability lobes (champions) --------------------------------
d = np.load(f'{RES}/lobes_l2.npz')
dt_ = np.load(f'{RES}/lobes_tdcj.npz')
rpm = d['rpm']
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for k, c, lab in SERIES:
    v = (d[k] if k in d.files else dt_[k]) * 1e3
    ax.plot(rpm, v, color=c, label=lab,
            lw=1.7 if k != 'open' else 1.3,
            ls='--' if k == KT else '-')
ax.axvline(4900, color='0.25', ls=':', lw=1.0)
ax.text(4900, ax.get_ylim()[1], ' design speed', fontsize=7.5,
        color='0.35', va='top')
ax.set_yscale('log')
ax.set_xlabel('spindle speed (rpm)')
ax.set_ylabel('$a_{\\mathrm{p,lim}}$ (mm)')
ax.legend(frameon=False, ncol=2, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_lobes.pdf'); plt.close(fig)

# ---- Fig 6: time responses (champions) ---------------------------------
d0 = np.load(f'{RES}/timeresp.npz')
dl = np.load(f'{RES}/timeresp_l.npz')
dj = np.load(f'{RES}/timeresp_tdcj.npz')


def tr(k, suff):
    for src in (d0, dl, dj):
        if f'{k}_{suff}' in src.files:
            return src[f'{k}_{suff}']
    raise KeyError(f'{k}_{suff}')


fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.7))
for k, c, lab in SERIES:
    t = tr(k, 't'); env = tr(k, 'env') * 1e6
    a1.plot(t, np.maximum(env, 1e-4), color=c, label=lab,
            lw=1.5 if k != 'open' else 1.2,
            ls='--' if k == KT else '-')
    tdv = float(tr(k, 'div'))
    if np.isfinite(tdv):
        a1.axvline(tdv, color=c, ls='--', lw=0.9, alpha=0.6)
        a1.text(tdv, a1.get_ylim()[1], ' diverges', fontsize=7,
                color=c, va='top', rotation=90)
a1.set_yscale('log')
a1.set_xlabel('time (s)'); a1.set_ylabel('$|y|$ envelope ($\\mu$m)')
a1.legend(frameon=False, fontsize=6.5, loc='lower right')
for k, c, lab in SERIES[1:]:
    a2.plot(tr(k, 't'), tr(k, 'urms'), color=c, label=lab, lw=1.4,
            ls='--' if k == KT else '-')
a2.set_xlabel('time (s)'); a2.set_ylabel('$u_{\\mathrm{rms}}$ (V)')
a2.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_timeresp.pdf'); plt.close(fig)
print('figures 5-6 regenerated for champions', KF, KR)
