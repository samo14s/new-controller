"""
figures_physset.py — the figures for the physics-based uncertainty set.
========================================================================
The set is better as a DESCRIPTION and worse as a DESIGN MODEL.  Both halves
of that sentence are measured, and both are drawn here.  The negative half is
not a footnote: it is the result.

    fig_p1_geometry   coverage gap against spurious content, for all five
                      descriptions -- the picture the whole redesign rests on
    fig_p2_weights    the same sixty weight triples on each description:
                      how much of the weight space is usable at all
    fig_p3_crossmu    mu of every controller on every description, both ways
    fig_p4_ablation   what each structural fact is worth, on its own
    fig_p5_decisive   the control that refutes the weight-choice explanation
    fig_p6_nominal    where reshaping moves the LFT nominal, and why that is a
                      design decision and not only a description

    python figures_physset.py -> results/figures_physset/*.png
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

OUT = os.path.join(C.RESULTS, 'figures_physset')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 1.8, 'figure.constrained_layout.use': True,
})
BOX = dict(fc='white', ec='0.86', lw=.7, pad=3.5)

SETS = ['paper', 'exact', 'phys', 'phys_sym', 'phys_ind']
SLAB = dict(paper='paper — Eqs. (22)–(25)',
            exact='exact — corrected box',
            phys='physics — all three facts',
            phys_sym='ablation: no one-sidedness',
            phys_ind='ablation: no correlation')
SCOL = dict(paper='#9F1239', exact='#B45309', phys='#0E7490',
            phys_sym='#7C3AED', phys_ind='#15803D')


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print('  ', p)
    return p


def _geo():
    p = os.path.join(C.RESULTS, 'set_geometry.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {k: dict(n=int(d[f'{k}_n']), cov=float(d[f'{k}_cov']),
                    spu=float(d[f'{k}_spu']))
            for k in SETS if f'{k}_n' in d.files}


def _weights():
    from matched_weights import load
    R = load()
    out = {}
    for tag, d in R.items():
        if not d:
            continue
        k = tag.replace('mu_', '', 1)
        ok = [v for v in d.values() if v['ok']]
        # the winning design is the FEASIBLE trial with the best objective --
        # the same rule run_musyn uses -- so its mu is the one the tables carry
        win = max(ok, key=lambda v: v['J']) if ok else None
        out[k] = dict(n=len(d), n_ok=len(ok),
                      bestJ=win['J'] if win else np.nan,
                      Ms=float(np.median([v['Ms'] for v in ok])) if ok else np.nan,
                      V=float(np.median([v['V'] for v in ok])) if ok else np.nan,
                      mu=win['mu'] if win else np.nan,
                      raw=d)
    return out


# ---------------------------------------------------------------------------
def fig_geometry():
    g = _geo()
    if g is None:
        print('   (set_geometry.npz missing — run compare_sets.py)')
        return None
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    # the three physics variants sit at almost the same coverage, so their
    # labels are staggered and carry leader lines instead of overlapping
    order = sorted(g, key=lambda k: g[k]['spu'])
    dy = {}
    step = 0
    for k in order:
        near = [j for j in dy if abs(g[j]['spu'] - g[k]['spu']) < 0.06
                and abs(g[j]['cov'] - g[k]['cov']) < 0.006]
        step = (max(dy[j] for j in near) + 30) if near else 20
        dy[k] = step
    for k, v in g.items():
        ax.scatter(100 * v['spu'], 100 * v['cov'], s=190, color=SCOL[k],
                   zorder=3, edgecolor='white', lw=1.4)
        ax.annotate(f'{SLAB[k]}  ({v["n"]} parameters)',
                    (100 * v['spu'], 100 * v['cov']),
                    textcoords='offset points', xytext=(0, dy[k]), ha='center',
                    fontsize=8, color=SCOL[k], fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=SCOL[k], lw=.8,
                                    shrinkA=1, shrinkB=6))
    ax.axhline(0, color='k', lw=1.4)
    ax.set_xlabel('spurious content (%)   — how much of the set the physics '
                  'cannot reach   ↓ better')
    ax.set_ylabel('coverage gap (%)   ↓ better')
    ax.set_title('The two things an uncertainty set can get wrong')
    ax.set_ylim(-1.4, max(100 * v['cov'] for v in g.values()) * 1.75)
    ax.set_xlim(0, max(100 * v['spu'] for v in g.values()) * 1.28)
    ax.text(0.985, 0.955,
            'a non-zero coverage gap VOIDS the guarantee: the design is robust\n'
            'to things the process does not do, and not to things it does.\n'
            'Only the published description has one.',
            transform=ax.transAxes, fontsize=8.5, color='0.2', va='top',
            ha='right', bbox=BOX)
    if 'exact' in g and 'phys' in g:
        ax.annotate('', xy=(100 * g['phys']['spu'], 100 * g['phys']['cov']),
                    xytext=(100 * g['exact']['spu'], 100 * g['exact']['cov']),
                    arrowprops=dict(arrowstyle='->', color='0.45', lw=1.6,
                                    ls='--'))
        mid = (100 * (g['exact']['spu'] + g['phys']['spu']) / 2,
               100 * (g['exact']['cov'] + g['phys']['cov']) / 2)
        ax.annotate(f'{g["exact"]["spu"]/g["phys"]["spu"]:.1f}× tighter\n'
                    'at equal coverage', mid, textcoords='offset points',
                    xytext=(0, -34), ha='center', fontsize=8.5,
                    color='0.35', fontweight='bold')
    return save(fig, 'fig_p1_geometry.png')


# ---------------------------------------------------------------------------
def fig_weights():
    W = _weights()
    ks = [k for k in SETS if k in W]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    share = np.array([100 * W[k]['n_ok'] / W[k]['n'] for k in ks])
    a.barh(np.arange(len(ks)), share, 0.6, color=[SCOL[k] for k in ks])
    a.set_yticks(np.arange(len(ks)), [SLAB[k] for k in ks], fontsize=8)
    a.set_xlabel('percent of the 60 weight triples that yield a design\n'
                 'meeting $M_s\\leq2$, effort $\\leq$ 450 V/N and the pole bound')
    a.set_title('How much of the weight space is usable at all')
    for i, k in enumerate(ks):
        a.annotate(f'{W[k]["n_ok"]}/{W[k]["n"]}   ($M_s$ {W[k]["Ms"]:.2f}, '
                   f'{W[k]["V"]:.0f} V/N)', (share[i], i),
                   textcoords='offset points', xytext=(5, -3), fontsize=7.6)
    a.set_xlim(0, 100)
    a.set_ylim(-0.9, len(ks) - 0.35)
    a.text(0.985, 0.04,
           'the wider space is NOT bought with conservatism: every physics\n'
           'variant lands near $M_s$ = 1.72 and 140 V/N, the same authority\n'
           'as the paper set.  Only the corrected box is forced down to 62 V/N.',
           transform=a.transAxes, fontsize=7.8, color='0.3', ha='right',
           va='bottom', bbox=BOX)

    P, F = W.get('paper', {}).get('raw', {}), W.get('phys', {}).get('raw', {})
    common = sorted(set(P) & set(F))
    both = [k for k in common if P[k]['ok'] and F[k]['ok']]
    onlyF = [k for k in common if F[k]['ok'] and not P[k]['ok']]
    onlyP = [k for k in common if P[k]['ok'] and not F[k]['ok']]
    none = [k for k in common if not P[k]['ok'] and not F[k]['ok']]
    counts = [len(both), len(onlyF), len(onlyP), len(none)]
    lab = ['both\nfeasible', 'physics\nONLY', 'paper\nONLY', 'neither']
    col = ['0.55', SCOL['phys'], SCOL['paper'], '0.85']
    b.bar(np.arange(4), counts, 0.62, color=col)
    b.set_xticks(np.arange(4), lab, fontsize=8.5)
    b.set_ylabel('weight triples (of 60)')
    b.set_title('The same 60 triples, paired one to one')
    for i, c in enumerate(counts):
        b.annotate(str(c), (i, c), textcoords='offset points', xytext=(0, 4),
                   ha='center', fontsize=10, fontweight='bold')
    if both:
        win = sum(F[k]['J'] > P[k]['J'] + 1e-4 for k in both)
        b.text(0.5, 0.72,
               f'on the {len(both)} triples where both work,\n'
               f'the objective is a wash: {win}–{len(both)-win}',
               transform=b.transAxes, fontsize=8.5, color='0.25', ha='center',
               bbox=BOX)
    return save(fig, 'fig_p2_weights.png')


# ---------------------------------------------------------------------------
def fig_crossmu():
    p = os.path.join(C.RESULTS, 'cross_mu.npz')
    if not os.path.exists(p):
        print('   (cross_mu.npz missing)')
        return None
    d = np.load(p, allow_pickle=True)
    rows = [str(x) for x in d['rows']]
    cols = [str(x) for x in d['cols']]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for ax, M, nm, note in zip(
            axes, [d['RS'], d['RP']],
            ['$\\mu_{RS}$ — robust stability only',
             '$\\mu_{RP}$ — stability and performance'],
            ['independent of the performance weights,\n'
             'so this is the clean comparison across rows',
             'measured with ONE common performance\nspecification for every row']):
        im = ax.imshow(M, cmap='RdYlGn_r', aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if M[i, j] > M.mean() else 'black')
        ax.set_xticks(range(len(cols)), [f'evaluated on\nthe {c} set'
                                         for c in cols], fontsize=8.5)
        ax.set_yticks(range(len(rows)), [r.replace('mu_', '$\\mu$ designed\non ')
                                         for r in rows], fontsize=8.5)
        ax.set_title(nm)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.046)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, fontsize=8,
                color='0.3', ha='center', va='top')
    fig.suptitle('mu is a property of a PAIR — so the table is built both ways',
                 fontweight='bold', fontsize=11)
    return save(fig, 'fig_p3_crossmu.png')


# ---------------------------------------------------------------------------
def fig_ablation():
    g, W = _geo(), _weights()
    if g is None:
        return None
    facts = [('one-sidedness', 'phys_sym'), ('correlation', 'phys_ind')]
    met = [('spurious content (%)', lambda k: 100 * g[k]['spu'], False),
           ('best objective $J$', lambda k: W[k]['bestJ'], True),
           ('usable weights (%)', lambda k: 100 * W[k]['n_ok'] / W[k]['n'], True),
           ('$\\mu$ of the winning design', lambda k: W[k]['mu'], False)]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.9))
    for ax, (nm, fn, up) in zip(axes, met):
        base = fn('phys')
        vals = [fn(t) for _, t in facts]
        x = np.arange(len(facts))
        ax.bar(x - 0.2, vals, 0.38, color='0.68', label='fact switched OFF')
        ax.bar(x + 0.2, [base] * len(facts), 0.38, color=SCOL['phys'],
               label='all three ON')
        ax.set_xticks(x, [f for f, _ in facts], fontsize=8.5)
        ax.set_title(nm + ('   ↑ better' if up else '   ↓ better'), fontsize=9)
        for i in range(len(facts)):
            good = (base > vals[i]) if up else (base < vals[i])
            ax.annotate(f'{vals[i]:.3g}', (x[i] - 0.2, vals[i]),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7.6, color='0.35')
            ax.annotate(f'{base:.3g}', (x[i] + 0.2, base),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7.6,
                        color=(SCOL['phys'] if good else '#9F1239'),
                        fontweight='bold')
        lo = min(min(vals), base); hi = max(max(vals), base)
        pad = 0.28 * (hi - lo if hi > lo else abs(hi) + 1)
        ax.set_ylim(min(lo - pad, 0) if lo >= 0 else lo - pad, hi + pad)
        if ax is axes[0]:
            ax.legend(loc='lower left', fontsize=7.6)
    fig.suptitle('What each structural fact is worth, measured by switching it '
                 'off alone   (red = the fact HURTS on that metric)',
                 fontweight='bold', fontsize=10.5)
    return save(fig, 'fig_p4_ablation.png')


# ---------------------------------------------------------------------------
def fig_decisive():
    """The control: physics set at the paper set's own winning weights."""
    with open(os.path.join(C.RESULTS, 'stage56.pkl'), 'rb') as fh:
        s56 = pickle.load(fh)
    mc = os.path.join(C.RESULTS, 'matched_certify.npz')
    rows = [('paper set,\nits own weights', s56['mu_tdc']['ap_inf'] * 1e3,
             s56['mu_tdc']['delta_max'], s56['mu_tdc']['peak'], SCOL['paper']),
            ('physics set,\nits own weights', s56['mu_phys_tdc']['ap_inf'] * 1e3,
             s56['mu_phys_tdc']['delta_max'], s56['mu_phys_tdc']['peak'],
             SCOL['phys'])]
    if os.path.exists(mc):
        d = np.load(mc, allow_pickle=True)
        rows.append(("physics set,\nthe PAPER's weights",
                     float(d['ap_inf']) * 1e3, float(d['delta_max']),
                     float(d['peak']), '#7C3AED'))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0))
    met = [('$a_p^\\infty$  (mm)', 1, False), ('$\\delta_{\\max}$', 2, False),
           ('peak $\\rho$   ↓ better', 3, True)]
    for ax, (nm, idx, down) in zip(axes, met):
        v = np.array([r[idx] for r in rows], float)
        ax.bar(np.arange(len(rows)), v, 0.6, color=[r[4] for r in rows])
        ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set_title(nm + ('' if down else '   ↑ better'), fontsize=9.5)
        for i, x in enumerate(v):
            ax.annotate(f'{x:.4f}', (i, x), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=9,
                        fontweight='bold')
        ax.set_ylim(0, v.max() * 1.25)
        if len(v) == 3:
            ax.annotate(f'{100*(v[2]/v[0]-1):+.0f} %', (2, v[2]),
                        textcoords='offset points', xytext=(0, 17),
                        ha='center', fontsize=9, color='#9F1239',
                        fontweight='bold')
    fig.suptitle('The control that refutes the weight-choice explanation — '
                 'at MATCHED weights the physics set is worse still',
                 fontweight='bold', fontsize=10.5)
    return save(fig, 'fig_p5_decisive.png')


# ---------------------------------------------------------------------------
def fig_nominal():
    """Where reshaping moves the LFT nominal — a design decision, not a description."""
    from plate_model import build_plate
    from uncertain_phys import PhysUncertainSystem
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    U = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN)
    xs = np.linspace(0.0, plate.lp, 401)
    D = np.array([plate.D_row(x, plate.hp)[:2] for x in xs])
    # DD0 of Eq. (24) built in the SAME gauge as the plotted locus, so the two
    # nominals are comparable: entrywise mid-range of D^T D over the edge
    DtD = np.einsum('ni,nj->nij', D, D)
    DD0 = 0.5 * (DtD.max(axis=0) + DtD.min(axis=0))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a.plot(D[:, 0], D[:, 1], color='#0E7490', lw=2.2,
           label='$D(x)$ over the whole edge')
    a.scatter(*U.d_c, s=150, color='#0E7490', zorder=4, edgecolor='white',
              lw=1.4, label='physics nominal $d_c$ — an ATTAINABLE position')
    if DD0 is not None:
        s12 = np.sign(DD0[0, 1]) if DD0[0, 1] != 0 else 1.0
        dn = np.array([np.sqrt(abs(DD0[0, 0])), s12 * np.sqrt(abs(DD0[1, 1]))])
        for sg in (1.0, -1.0):
            a.scatter(dn[0], sg * abs(dn[1]), s=150, marker='s',
                      color='#9F1239', zorder=4, edgecolor='white', lw=1.4,
                      label=("paper nominal from $DD_0$ — reached by NO position"
                             if sg > 0 else None))
    a.axhline(0, color='0.5', lw=.9)
    a.set_xlabel('$D_1(x)$'); a.set_ylabel('$D_2(x)$')
    a.set_title('Where each description puts its nominal')
    a.legend(loc='lower left', fontsize=7.6)
    a.set_xlim(6.15, 7.35)
    a.text(0.985, 0.955,
           'the physics nominal sits where mode 2 has a NODE, so its\n'
           'nominal cutting stiffness on mode 2 is nearly zero.  The\n'
           "paper's is a fictitious average that no position attains —\n"
           'and it forces the controller to damp mode 2 anyway.',
           transform=a.transAxes, fontsize=7.6, color='0.2', ha='right',
           va='top', bbox=BOX)

    f0 = np.array(plate.omega_n[:2]) / (2 * np.pi)
    Minv = np.linalg.inv(U.M_c)
    fc = np.sort(np.sqrt(np.linalg.eigvals(Minv @ U.K_c).real)) / (2 * np.pi)
    lo = np.sort(np.sqrt(np.linalg.eigvals(
        np.linalg.inv(U.M_c - U.M_1 + U.M_2) @ (U.K_c - U.K_1 + U.K_2)).real)) / (2 * np.pi)
    hi = np.sort(np.sqrt(np.linalg.eigvals(
        np.linalg.inv(U.M_c + U.M_1 + U.M_2) @ (U.K_c + U.K_1 + U.K_2)).real)) / (2 * np.pi)
    y = np.arange(2)
    b.barh(y + 0.18, [1.106 * f0[i] - f0[i] / 1.106 for i in range(2)], 0.32,
           left=[f0[i] / 1.106 for i in range(2)], color=SCOL['paper'],
           alpha=.75, label='paper: $\\pm$10 % box, centred on the raw plate')
    b.barh(y - 0.18, [abs(hi[i] - lo[i]) for i in range(2)], 0.32,
           left=[min(lo[i], hi[i]) for i in range(2)], color=SCOL['phys'],
           alpha=.75, label='physics: one-sided, centred at mid-removal')
    for i in range(2):
        b.plot([f0[i]], [y[i] + 0.18], 'k|', ms=14, mew=2)
        b.plot([fc[i]], [y[i] - 0.18], 'k|', ms=14, mew=2)
        b.annotate(f'{f0[i]:.0f}', (f0[i], y[i] + 0.18),
                   textcoords='offset points', xytext=(0, 11), ha='center',
                   fontsize=7.6)
        b.annotate(f'{fc[i]:.0f}', (fc[i], y[i] - 0.18),
                   textcoords='offset points', xytext=(0, -18), ha='center',
                   fontsize=7.6)
    b.set_yticks(y, ['mode 1', 'mode 2'])
    b.set_xlabel('frequency covered by the description (Hz)')
    b.set_title('One-sidedness moves the nominal — by construction')
    b.legend(loc='upper left', fontsize=7.8)
    b.set_ylim(-0.95, 2.05)
    b.text(0.5, 0.03,
           'an LFT nominal is not only a description, it is a DESIGN TARGET.\n'
           'Reshaping puts it on the physical manifold; the tick marks are\n'
           'where each description thinks the plant nominally is.',
           transform=b.transAxes, fontsize=7.8, color='0.2', ha='center',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_p6_nominal.png')


def main():
    print('writing physics-set figures ->', OUT)
    fig_weights(); fig_crossmu(); fig_decisive()
    fig_geometry(); fig_ablation(); fig_nominal()


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
def fig_mechanism():
    """The pre-declared test of the proposed mechanism -- and its refutation."""
    p = os.path.join(C.RESULTS, 'nominal_experiment2.npz')
    if not os.path.exists(p):
        print('   (nominal_experiment2.npz missing)')
        return None
    d = np.load(p)
    w = np.abs(d['W22']); a = d['ap_inf'] * 1e3; dm = d['delta_max']
    fr = d['frac']
    node = np.isnan(fr) | (np.abs(fr - 0.5) < 1e-9)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for axis, y, nm in ((ax, a, '$a_p^\\infty$  (mm)'),
                        (bx, dm, '$\\delta_{\\max}$')):
        axis.scatter(w[~node], y[~node], s=110, color='#0E7490', zorder=3,
                     label='nominal moved OFF the node')
        axis.scatter(w[node], y[node], s=170, color='#9F1239', marker='D',
                     zorder=4, label='nominal ON the mode-2 node')
        axis.set_xscale('log')
        axis.set_xlabel('$|W_c(2,2)|$ — nominal cutting stiffness on mode 2\n'
                        '(the mechanism variable)   →  further from the node')
        axis.set_ylabel(nm)
        r = float(np.corrcoef(np.log10(np.maximum(w, 1e-9)), y)[0, 1])
        axis.set_title(f'{nm}    measured  r = {r:+.3f}', fontsize=10)
        for i in range(len(w)):
            lab = 'centre' if np.isnan(fr[i]) else f'{fr[i]:.2f}'
            axis.annotate(lab, (w[i], y[i]), textcoords='offset points',
                          xytext=(0, 10), ha='center', fontsize=7.4,
                          color='0.35')
        axis.legend(loc='lower right', fontsize=8)
        axis.set_ylim(0, max(y) * 1.42)
    fig.suptitle('The pre-declared test of the proposed mechanism — '
                 'it predicted a STRONG POSITIVE correlation',
                 fontweight='bold', fontsize=10.5)
    ax.text(0.5, 0.035,
            'both correlations come out NEGATIVE, and the best nominal is the natural\n'
            'centre — the one the reshaped set already uses.  The mechanism is\n'
            'contradicted by its own test.',
            transform=ax.transAxes, fontsize=8, color='0.2', ha='center',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_p7_mechanism.png')
