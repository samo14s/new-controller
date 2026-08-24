"""Paper figures 5-6 from the stored lobes/timeresp artifacts."""
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
SERIES = (('open', GY, 'open loop'),
          ('mu_tdc', C1, '$\\mu$-TDC (benchmark)'),
          ('ps_ac_rfa', C3, 'PS-AC-RFA r5'),
          ('ps_ac_ri', C2, 'PS-AC-RI'))

# ---- Fig 5: stability lobes --------------------------------------------
d = np.load(f'{RES}/lobes.npz')
rpm = d['rpm']
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for k, c, lab in SERIES:
    ax.plot(rpm, d[k] * 1e3, color=c, label=lab,
            lw=1.7 if k != 'open' else 1.3)
ax.axvline(4900, color='0.25', ls=':', lw=1.0)
ax.text(4900, ax.get_ylim()[1], ' design speed', fontsize=7.5,
        color='0.35', va='top')
ax.set_yscale('log')
ax.set_xlabel('spindle speed (rpm)')
ax.set_ylabel('$a_{\\mathrm{p,lim}}$ (mm)')
ax.legend(frameon=False, ncol=2, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_lobes.pdf'); plt.close(fig)

# ---- Fig 6: time responses ---------------------------------------------
d = np.load(f'{RES}/timeresp.npz')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.7))
for k, c, lab in SERIES:
    t = d[f'{k}_t']; env = d[f'{k}_env'] * 1e6
    a1.plot(t, np.maximum(env, 1e-4), color=c, label=lab,
            lw=1.5 if k != 'open' else 1.2)
    tdv = float(d[f'{k}_div'])
    if np.isfinite(tdv):
        a1.axvline(tdv, color=c, ls='--', lw=0.9, alpha=0.6)
        a1.text(tdv, a1.get_ylim()[1], ' diverges', fontsize=7,
                color=c, va='top', rotation=90)
a1.set_yscale('log')
a1.set_xlabel('time (s)'); a1.set_ylabel('$|y|$ envelope ($\\mu$m)')
a1.legend(frameon=False, fontsize=6.5, loc='lower right')
for k, c, lab in SERIES[1:]:
    a2.plot(d[f'{k}_t'], d[f'{k}_urms'], color=c, label=lab, lw=1.4)
a2.set_xlabel('time (s)'); a2.set_ylabel('$u_{\\mathrm{rms}}$ (V)')
a2.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_timeresp.pdf'); plt.close(fig)
print('figures 5-6 written')
