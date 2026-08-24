"""Paper figures, generated from the stored artifacts (artifact-first)."""
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
C1, C2, C3 = '#7C3AED', '#C2410C', '#0891B2'
GATE = '#9F1239'
import os
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, 'paper', 'figs')
RES = os.path.join(ROOT, 'results')

# ---- Fig 1: the two readings, one scale --------------------------------
members = ['mu-paper witness\n(reference D--K)',
           'PS-AC-R\n(scheduled LQG)',
           'PS-AC-RFA r4\n(filtered, augm.)',
           'PS-AC-RF r2\n(series filter)',
           'PS-AC-RFA r5\n(final member)']
sc = [1.733, 85.8, 10.0, 3.70, 2.26]          # scalar-D complex
tc = [1.205, 85.8, None, None, None]          # tight complex (full-D)
mx = [0.419, 3.79, 0.485, 0.313, 0.383]       # mixed (real reading)
fig, ax = plt.subplots(figsize=(5.5, 2.9))
y = np.arange(len(members))[::-1]
for yi, s, t, m in zip(y, sc, tc, mx):
    xs = [v for v in (s, t, m) if v is not None]
    ax.plot([min(xs), max(xs)], [yi, yi], color='0.82', lw=1.2, zorder=1)
ax.scatter(sc, y, s=34, color=C1, zorder=3, label='complex $\\mu$ (scalar-$D$)')
ax.scatter([v for v in tc if v], [yi for yi, v in zip(y, tc) if v],
           s=34, color=C2, zorder=3, label='complex $\\mu$ (full-$D$)')
ax.scatter(mx, y, s=40, color=C3, zorder=4, marker='D',
           label='mixed real/complex $\\mu$')
ax.axvline(1.0, color=GATE, ls='--', lw=1.4)
ax.text(1.0, len(members) - 0.35, 'gate $\\mu=1$', color=GATE,
        ha='center', va='bottom', fontsize=8)
ax.set_xscale('log'); ax.set_xlim(0.2, 130)
ax.set_yticks(y); ax.set_yticklabels(members)
ax.set_xlabel('peak $\\mu_{\\mathrm{RS}}$ (worst probed node, actuator block included)')
ax.legend(loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_readings.pdf'); plt.close(fig)

# ---- Fig 2: the 21-node sweep ------------------------------------------
d = np.load(f'{RES}/mu_real_21_sup.npz')
x = d['x'] * 1e3
fig, ax = plt.subplots(figsize=(5.5, 2.7))
ax.axhline(1.0, color=GATE, ls='--', lw=1.4)
ax.text(2, 1.02, 'gate $\\mu=1$', color=GATE, fontsize=8, va='bottom')
ax.plot(x, d['wit_peak'], '-o', ms=3.4, color=C1,
        label='mu-paper witness (fixed)')
ax.plot(x, d['r5_peak'], '-o', ms=3.4, color=C3,
        label='PS-AC-RFA r5 (scheduled)')
ax.annotate('0.419', (x[-1], d['wit_peak'][-1]), textcoords='offset points',
            xytext=(-2, 7), fontsize=8, color='0.25')
ax.annotate('0.383', (x[-1], d['r5_peak'][-1]), textcoords='offset points',
            xytext=(-2, -13), fontsize=8, color='0.25')
ax.set_xlabel('scheduling node position $x$ along the feed path (mm)')
ax.set_ylabel('peak mixed $\\mu_{\\mathrm{RS}}$')
ax.set_ylim(0, 1.1); ax.legend(loc='center left', frameon=False)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_sweep21.pdf'); plt.close(fig)

# ---- Fig 3: refinement -- convergence vs the diverging corner ----------
r1 = np.load(f'{RES}/adiab_rigor.npz')
NX = np.array([121, 241, 481, 961])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.6), sharex=True)
ri = [r1[f'ps_ac_ri_{nm}_NX{n}'][0] * 1e3 for nm in ('lo',) for n in NX]
a1.plot(NX, ri, '-o', ms=3.6, color=C1, label='RI, $a_4$ lo')
ri_n = [r1[f'ps_ac_ri_nom_NX{n}'][0] * 1e3 for n in NX]
a1.plot(NX, ri_n, '-s', ms=3.6, color=C2, label='RI, $a_4$ nom/hi')
rf_hi = [r1[f'ps_ac_rfa_hi_NX{n}'][0] * 1e3 for n in NX]
a1.plot(NX, rf_hi, '-^', ms=3.8, color=C3,
        label='RFA @ 0.28, $a_4$ hi')
a1.axhline(4.90, color=GATE, ls='--', lw=1.2)
a1.text(940, 5.3, 'actual feed', color=GATE, fontsize=7.5, ha='right')
a1.set_xscale('log'); a1.set_xticks(NX); a1.set_xticklabels(NX)
a1.set_xlabel('grid nodes $N_x$'); a1.set_ylabel('$v^{*}$ (mm/s)')
a1.set_yscale('log'); a1.legend(frameon=False, fontsize=7)
a1.set_title('secant $v^{*}$ under refinement', fontsize=8.5)
val_lo = [r1[f'ps_ac_ri_lo_NX{n}'][1] * 100 for n in NX]
val_rfnom = [r1[f'ps_ac_rfa_nom_NX{n}'][1] * 100 for n in NX]
val_rfhi = [r1[f'ps_ac_rfa_hi_NX{n}'][1] * 100 for n in NX]
a2.plot(NX, val_lo, '-o', ms=3.6, color=C1, label='RI lo')
a2.plot(NX, val_rfnom, '-s', ms=3.6, color=C2, label='RFA @ 0.28 nom')
a2.plot(NX, val_rfhi, '-^', ms=3.8, color=C3, label='RFA @ 0.28 hi')
a2.set_xscale('log'); a2.set_xticks(NX); a2.set_xticklabels(NX)
a2.set_xlabel('grid nodes $N_x$')
a2.set_ylabel('static interval validity (%)')
a2.set_ylim(-4, 104); a2.legend(frameon=False, fontsize=7)
a2.set_title('per-interval static validation', fontsize=8.5)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_refine.pdf'); plt.close(fig)

# ---- Fig 4: differential rotation density ------------------------------
dd = np.load(f'{RES}/adiab_diff.npz')
fig, ax = plt.subplots(figsize=(5.5, 2.5))
for nm, c, lab in (('lo', C1, '$a_4$ lo'), ('nom', C2, '$a_4$ nom'),
                   ('hi', C3, '$a_4$ hi')):
    s = dd[f'ps_ac_ri_{nm}_sigma']
    xs = np.linspace(0, 100, len(s))
    ax.plot(xs, s, color=c, label=lab, lw=1.3)
ax.set_xlabel('position $x$ (mm)')
ax.set_ylabel('$\\sigma(x)$ (1/m, scaled time)')
ax.set_yscale('log')
ax.legend(frameon=False, title='PS-AC-RI @ 0.10 mm', fontsize=7.5,
          title_fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIGS}/fig_sigma.pdf'); plt.close(fig)
print('figures written')
