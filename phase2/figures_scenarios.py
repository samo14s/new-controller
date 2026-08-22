"""
figures_scenarios.py — the comparison figures for Stages 7-8.
==============================================================
Four scenarios, six metrics, one protocol.  The point of these figures is not
that the proposed controller wins -- on several metrics it does not -- but that
every controller was put through exactly the same thing, and that the ranking
CHANGES with the scenario, which is the real finding.

    fig_s1_nominal    the six metrics at the nominal operating point
    fig_s2_sweep      a_p,lim along the pass, and what the spread costs
    fig_s3_modal      the modal-parameter scenarios, case by case
    fig_s4_delay      a_p,lim against delay, up to 3 tau_0
    fig_s5_worst      the worst case of each scenario, side by side -- the
                      number production is actually limited by
    fig_s6_ranking    how the ranking moves between scenarios

    python figures_scenarios.py -> results/figures_scenarios/*.png
"""
import os
import pickle
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C

OUT = os.path.join(C.RESULTS, 'figures_scenarios')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 1.7, 'figure.constrained_layout.use': True,
})
BOX = dict(fc='white', ec='0.86', lw=.7, pad=3.5)

SHOW = ['fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac']
LAB = dict(fopid='FOPID', lqg='LQG', mu_tdc='$\\mu$-TDC (Du 2024)',
           mu_phys_tdc='$\\mu$-TDC, physics set', ps_ac='PS-AC (proposed)',
           open='no control')
COL = dict(fopid='#C2410C', lqg='#7C3AED', mu_tdc='#0891B2',
           mu_phys_tdc='#15803D', ps_ac='#9F1239', open='0.6')
LS = dict(mu_phys_tdc='--')


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print('  ', p)
    return p


def store():
    with open(os.path.join(C.RESULTS, 'stage78.pkl'), 'rb') as fh:
        return pickle.load(fh)


def _ks(d, prefix='S1'):
    return [k for k in SHOW if f'{prefix}_{k}' in d]


def mm(d, key):
    """a_p,lim in mm.  run_stage78 stores S1/S2 in METRES and S3/S4 already in
    MILLIMETRES; mixing the two silently was worth a factor of a thousand."""
    v = d[key]
    v = v['limits'] if hasattr(v, 'keys') else v
    v = np.asarray(v, float)
    return v * 1e3 if key[:2] in ('S1', 'S2') else v


# ---------------------------------------------------------------------------
def fig_nominal():
    d = store()
    ks = _ks(d)
    met = [('$a_{p,\\mathrm{lim}}$ worst (mm)',
            [mm(d, f'S1_{k}').min() for k in ks], True),
           ('$a_{p,\\mathrm{lim}}$ mean (mm)',
            [mm(d, f'S1_{k}').mean() for k in ks], True),
           ('$A_{\\mathrm{RMS}}$ (μm)',
            [d[f'S1_{k}']['A_rms'] for k in ks], False),
           ('settling $T_s$ (ms)',
            [d[f'S1_{k}']['T_s'] * 1e3 for k in ks], False),
           ('effort $E_u$ (V$^2$s)', [d[f'S1_{k}']['E_u'] for k in ks], False),
           ('peak voltage (V)', [d[f'S1_{k}']['u_max'] for k in ks], False)]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.0))
    for ax, (nm, vals, up) in zip(axes.ravel(), met):
        v = np.array(vals, float)
        ax.bar(np.arange(len(ks)), v, 0.62, color=[COL[k] for k in ks])
        best = int(np.argmax(v) if up else np.argmin(v))
        ax.bar([best], [v[best]], 0.62, color=COL[ks[best]],
               edgecolor='k', lw=1.8)
        ax.set_xticks(np.arange(len(ks)),
                      [LAB[k].split(' (')[0].replace('$\\mu$-TDC, ', '')
                       for k in ks], fontsize=7, rotation=18)
        ax.set_title(nm + ('   ↑ better' if up else '   ↓ better'),
                     fontsize=9)
        for i, x in enumerate(v):
            ax.annotate(f'{x:.3g}', (i, x), textcoords='offset points',
                        xytext=(0, 3), ha='center', fontsize=7)
        ax.set_ylim(0, v.max() * 1.22)
    fig.suptitle(f'Stage 7 — the six metrics at the nominal point '
                 f'({C.RPM_S} rpm, $a_p$ = {C.AP_S*1e3:.1f} mm); '
                 'thick outline = best', fontweight='bold', fontsize=11)
    return save(fig, 'fig_s1_nominal.png')


# ---------------------------------------------------------------------------
def fig_sweep():
    d = store()
    ks = _ks(d, 'S2')
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                               width_ratios=[1.55, 1])
    mins, rngs = [], []
    for k in ['open'] + ks:
        key = f'S2_{k}'
        if key not in d:
            continue
        y = mm(d, key)
        x = np.linspace(0, 100, len(y))
        a.plot(x, y, 'o-', ms=4, color=COL[k], ls=LS.get(k, '-'), label=LAB[k])
        if k != 'open':
            mins.append(y.min()); rngs.append(y.max() - y.min())
    a.axvline(50, color='0.35', lw=.9, ls=':')
    a.annotate('mode-2 node', (50, 0.06), textcoords='offset points',
               xytext=(4, 0), fontsize=8, color='0.35')
    a.set_xlabel('tool position (% of $l_P$)')
    a.set_ylabel('$a_{p,\\mathrm{lim}}$ (mm)')
    a.set_title('Stage 8(a) — the position sweep')
    a.legend(loc='upper left', fontsize=7.8, ncol=2)

    b.scatter(rngs, mins, s=110, color=[COL[k] for k in ks], zorder=3)
    for i, k in enumerate(ks):
        b.annotate(LAB[k].split(' (')[0], (rngs[i], mins[i]),
                   textcoords='offset points', xytext=(0, 9), ha='center',
                   fontsize=7.6, color=COL[k])
    b.set_xlabel('spread across the pass (mm)   ↓ better')
    b.set_ylabel('worst position (mm)   ↑ better')
    b.set_title('The trade the sweep exposes')
    b.text(0.5, 0.04,
           'the upper-LEFT corner is what production wants:\n'
           'a high floor and a flat curve',
           transform=b.transAxes, fontsize=8, color='0.3', ha='center',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_s2_sweep.png')


# ---------------------------------------------------------------------------
def fig_modal():
    d = store()
    ks = _ks(d, 'S3')
    cases = [str(c) for c in d.get('S3_cases', [])]
    M = np.array([mm(d, f'S3_{k}') for k in ks])
    x = np.arange(len(cases)); w = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    for i, k in enumerate(ks):
        ax.bar(x - 0.4 + (i + 0.5) * w, M[i], w, color=COL[k], label=LAB[k])
    if 'S3_open' in d:
        ax.plot(x, mm(d, 'S3_open'), 'k_', ms=22,
                mew=2, label='no control')
    ax.set_xticks(x, cases, fontsize=8)
    ax.set_ylabel('$a_{p,\\mathrm{lim}}$ worst over positions (mm)')
    ax.set_title('Stage 8(b) — modal-parameter variation, case by case')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, M.max() * 1.28)
    for i, k in enumerate(ks):
        j = int(np.argmin(M[i]))
        ax.annotate(f'{M[i][j]:.3f}', (x[j] - 0.4 + (i + 0.5) * w, M[i][j]),
                    textcoords='offset points', xytext=(0, 3), ha='center',
                    fontsize=6.8, color=COL[k], rotation=90)
    ax.text(0.02, 0.955,
            'the annotated bar is each controller\'s WORST case.\n'
            'the ranking here is not the ranking at the nominal point.',
            transform=ax.transAxes, fontsize=8, color='0.3', va='top',
            bbox=BOX)
    return save(fig, 'fig_s3_modal.png')


# ---------------------------------------------------------------------------
def fig_delay():
    d = store()
    ks = _ks(d, 'S4')
    r = np.asarray(d['S4_ratios'], float)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for k in ['open'] + ks:
        key = f'S4_{k}'
        if key not in d:
            continue
        y = mm(d, key)
        ax.plot(r, y, 'o-', ms=5, color=COL[k], ls=LS.get(k, '-'),
                label=f'{LAB[k]}   worst {y.min():.3f}')
    ax.set_xlabel('delay  $\\tau / \\tau_0$')
    ax.set_ylabel('$a_{p,\\mathrm{lim}}$ worst over positions (mm)')
    ax.set_title('Stage 8(c) — increased regenerative delay')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_ylim(0, 1.12)
    ax.text(0.985, 0.955,
            'the delay-independent certificate says every loop survives ANY delay;\n'
            'this asks a different and weaker question — how much DEPTH survives —\n'
            'and it is the one production cares about',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='right',
            va='top', bbox=BOX)
    return save(fig, 'fig_s4_delay.png')


# ---------------------------------------------------------------------------
def fig_worst():
    d = store()
    ks = _ks(d)
    rows = [('nominal', [mm(d, f'S1_{k}').min() for k in ks]),
            ('position sweep', [mm(d, f'S2_{k}').min() for k in ks]),
            ('modal variation', [mm(d, f'S3_{k}').min() for k in ks]),
            ('increased delay', [mm(d, f'S4_{k}').min() for k in ks])]
    M = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    x = np.arange(len(rows)); w = 0.8 / len(ks)
    for i, k in enumerate(ks):
        ax.bar(x - 0.4 + (i + 0.5) * w, M[:, i], w, color=COL[k], label=LAB[k])
    ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
    ax.set_ylabel('worst $a_{p,\\mathrm{lim}}$ in the scenario (mm)')
    ax.set_title('Stage 8 — the number production is actually limited by')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, M.max() * 1.3)
    for j in range(len(rows)):
        b = int(np.argmax(M[j]))
        ax.annotate('best', (x[j] - 0.4 + (b + 0.5) * w, M[j, b]),
                    textcoords='offset points', xytext=(0, 4), ha='center',
                    fontsize=7.4, fontweight='bold', color=COL[ks[b]])
    ov = M.min(axis=0)
    ax.text(0.02, 0.955,
            'worst over ALL four scenarios:  '
            + ',  '.join(f'{LAB[k].split(" (")[0]} {ov[i]:.3f}'
                         for i, k in enumerate(ks)),
            transform=ax.transAxes, fontsize=7.8, color='0.3', va='top',
            bbox=BOX)
    return save(fig, 'fig_s5_worst.png')


# ---------------------------------------------------------------------------
def fig_ranking():
    d = store()
    ks = _ks(d)
    scen = [('nominal', [mm(d, f'S1_{k}').min() for k in ks]),
            ('position', [mm(d, f'S2_{k}').min() for k in ks]),
            ('modal', [mm(d, f'S3_{k}').min() for k in ks]),
            ('delay', [mm(d, f'S4_{k}').min() for k in ks])]
    R = np.array([len(ks) - np.argsort(np.argsort(v)) for _, v in scen])
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(scen))
    for i, k in enumerate(ks):
        ax.plot(x, R[:, i], 'o-', ms=9, color=COL[k], ls=LS.get(k, '-'),
                label=LAB[k], lw=2.0)
    ax.set_xticks(x, [s[0] for s in scen], fontsize=9.5)
    ax.set_yticks(np.arange(1, len(ks) + 1))
    ax.invert_yaxis()
    ax.set_ylabel('rank on worst-case $a_{p,\\mathrm{lim}}$   (1 = best)')
    ax.set_title('Where the ranking moves between scenarios — and where it does not')
    ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1.01, 0.5))
    ax.text(0.02, 0.03,
            'the top two do NOT move: PS-AC and LQG hold ranks 1 and 2 in all\n'
            'four scenarios.  All the movement is between the two $\\mu$ designs —\n'
            'the physics-set one climbs from last to third under increased delay,\n'
            'and the published one falls to last.',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='left',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_s6_ranking.png')


def main():
    print('writing scenario figures ->', OUT)
    fig_nominal(); fig_sweep(); fig_modal()
    fig_delay(); fig_worst(); fig_ranking()


if __name__ == '__main__':
    main()
