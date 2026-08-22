"""
figures_certify.py — the certificate figures for Stages 5-6.
=============================================================
Everything in Phases 3-4 is numerical evidence.  These are the figures of what
is actually PROVED, and of the one numerical trap that nearly invalidated it.

    fig_k1_crossing   the delay-independent test itself: the curve
                      rho((j w I - A_cl)^-1 A_d,cl) against the threshold 1,
                      for each controller, over the vertex family
    fig_k2_grid       the trap: a plain log grid steps straight over resonances
                      of half-width ~10 rad/s and reports a spectral radius far
                      too small.  Both grids on the same closed loop.
    fig_k3_depth      a_p^inf, the depth certified for EVERY delay and every
                      point of the family, against the nominal Floquet limit
    fig_k4_margin     delta_max, how far the uncertainty set can be inflated
                      before the certificate fails
    fig_k5_family     what the certificate actually quantifies over: the vertex
                      family in position, removal, damping and mid-pass state
    fig_k6_summary    all four certified quantities together, per controller

    python figures_certify.py -> results/figures_certify/*.png
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

import certify2 as CF2
import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = os.path.join(C.RESULTS, 'figures_certify')
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


def _rho_curve(A, Ad, w):
    """rho((j w I - A)^-1 A_d) on a given grid."""
    I = np.eye(A.shape[0])
    out = np.empty(len(w))
    for k, wk in enumerate(w):
        try:
            out[k] = np.abs(np.linalg.eigvals(
                np.linalg.solve(1j * wk * I - A, Ad))).max()
        except np.linalg.LinAlgError:
            out[k] = np.nan
    return out


def _worst_vertex(plant, ctrl, **kw):
    """The vertex with the largest peak, and its grid."""
    V, npl, ws = CF2.vertices(plant, ctrl, **kw)
    best, bp = None, -1.0
    for (A, Ad) in V:
        _, p = CF2.crossing_delays(A, Ad)
        if p > bp and np.isfinite(p):
            bp, best = p, (A, Ad)
    return best, bp, ws


# ---------------------------------------------------------------------------
def fig_crossing(n_pos=5):
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    made = load_controllers(plate, plant)
    kw = dict(n_pos=n_pos, etas=(0.0, C.ETA_MAX), zetas=(C.ZETA_LO, C.ZETA_HI),
              xis=(1.0,))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for k in SHOW:
        if k not in made:
            continue
        (A, Ad), pk, ws = _worst_vertex(plant, made[k](plant), **kw)
        w = CF2._wgrid(A)
        r = _rho_curve(A, Ad, w)
        ax.semilogx(np.maximum(w * ws, 1e-2) / (2 * np.pi), r, color=COL[k],
                    ls=LS.get(k, '-'), label=f'{LAB[k]}   peak {pk:.3f}')
    ax.axhline(1.0, color='k', lw=1.8, ls='--',
               label='threshold: below 1 = stable for EVERY delay')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\rho\left((j\omega I - A_{cl})^{-1}A_{d,cl}\right)$')
    ax.set_title('The delay-independent test, at the worst vertex of the family')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1.35)
    ax.set_xlim(1.0, 2e4)
    ax.text(0.985, 0.06,
            'the test is EXACT, not sufficient: a characteristic root reaches\n'
            'the imaginary axis for some delay if and only if this curve\n'
            'touches 1.  Below it, no delay whatsoever destabilises the loop.',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='right',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_k1_crossing.png')


# ---------------------------------------------------------------------------
def fig_grid():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    made = load_controllers(plate, plant)
    (A, Ad), pk, ws = _worst_vertex(plant, made['open'](plant), n_pos=3,
                                    etas=(0.0,), zetas=(1.0,), xis=(1.0,))
    w_ref = CF2._wgrid(A)
    r_ref = _rho_curve(A, Ad, w_ref)
    w_log = np.logspace(-3, np.log10(5.0 * max(np.abs(np.linalg.eigvals(A)).max(),
                                               1.0)), 400)
    r_log = _rho_curve(A, Ad, w_log)
    f_ref = np.maximum(w_ref * ws, 1e-2) / (2 * np.pi)
    f_log = np.maximum(w_log * ws, 1e-2) / (2 * np.pi)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    a.semilogx(f_ref, r_ref, color='#0E7490', lw=1.6,
               label=f'refined grid ({len(w_ref)} pts)   peak {np.nanmax(r_ref):.2f}')
    a.semilogx(f_log, r_log, 'o-', color='#9F1239', ms=3, lw=1.0, alpha=.85,
               label=f'plain log grid ({len(w_log)} pts)   peak {np.nanmax(r_log):.2f}')
    a.axhline(1.0, color='k', lw=1.5, ls='--')
    a.set_xlabel('frequency (Hz)'); a.set_ylabel(r'$\rho$')
    a.set_xlim(1e2, 5e3)
    a.set_title('The trap — a plain log grid steps over the resonances')
    a.legend(loc='upper left', fontsize=8)

    ev = np.linalg.eigvals(A)
    k = int(np.argmax(np.abs(np.imag(ev))))
    wk = abs(float(np.imag(ev[k]))); half = abs(float(np.real(ev[k])))
    m = (f_ref > (wk - 12 * half) * ws / (2 * np.pi)) & \
        (f_ref < (wk + 12 * half) * ws / (2 * np.pi))
    ml = (f_log > (wk - 12 * half) * ws / (2 * np.pi)) & \
         (f_log < (wk + 12 * half) * ws / (2 * np.pi))
    b.plot(f_ref[m], r_ref[m], color='#0E7490', lw=1.8, label='refined')
    b.plot(f_log[ml], r_log[ml], 'o', color='#9F1239', ms=6, label='log-grid samples')
    b.axhline(1.0, color='k', lw=1.5, ls='--')
    b.set_xlabel('frequency (Hz)'); b.set_ylabel(r'$\rho$')
    b.set_title('Zoom on one mode')
    b.legend(loc='upper left', fontsize=8)
    true_pk = float(np.nanmax(r_ref[m])) if m.sum() else float('nan')
    log_pk = float(np.nanmax(r_log[ml])) if ml.sum() else float('nan')
    b.axhline(true_pk, color='#0E7490', lw=.9, ls=':')
    b.axhline(log_pk, color='#9F1239', lw=.9, ls=':')
    b.text(0.5, 0.055,
           f'the mode is {2*half*ws:.0f} rad/s wide while the log grid is spaced\n'
           f'{(f_log[ml][1]-f_log[ml][0])*2*np.pi if ml.sum()>1 else float("nan"):.0f} '
           'rad/s here, so it lands BESIDE the peak, never on it:\n'
           f'it reports {log_pk:.1f} where the truth is {true_pk:.1f} — '
           f'{true_pk/max(log_pk,1e-9):.1f}× too small.\n'
           'That is enough to turn an unstable verdict into a stable one.',
           transform=b.transAxes, fontsize=8, color='0.2', ha='center',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_k2_grid.png')


# ---------------------------------------------------------------------------
def _store():
    with open(os.path.join(C.RESULTS, 'stage56.pkl'), 'rb') as fh:
        s56 = pickle.load(fh)
    with open(os.path.join(C.RESULTS, 'stage78.pkl'), 'rb') as fh:
        s78 = pickle.load(fh)
    return s56, s78


def fig_depth():
    s56, s78 = _store()
    ks = [k for k in SHOW if k in s56]
    ap = np.array([s56[k]['ap_inf'] for k in ks]) * 1e3
    nom = np.array([np.min(s78[f'S1_{k}']['limits']) for k in ks]) * 1e3
    x = np.arange(len(ks)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.bar(x - w / 2, nom, w, color='0.72', label='nominal Floquet limit (evidence)')
    for i, k in enumerate(ks):
        ax.bar(x[i] + w / 2, ap[i], w, color=COL[k])
    ax.bar([], [], color='0.35', label='$a_p^\\infty$ certified (proof)')
    ax.axhline(np.min(s78['S1_open']['limits']) * 1e3, color='k', lw=1.4,
               ls=':', label='no control')
    ax.set_xticks(x, [LAB[k] for k in ks], fontsize=8, rotation=10)
    ax.set_ylabel('$a_p$  (mm)')
    ax.set_title('What is measured, and what is proved')
    ax.legend(loc='upper left', fontsize=8)
    for i in range(len(ks)):
        ax.annotate(f'{nom[i]:.3f}', (i - w / 2, nom[i]),
                    textcoords='offset points', xytext=(0, 4), ha='center',
                    fontsize=7.6, color='0.4')
        ax.annotate(f'{ap[i]:.4f}', (i + w / 2, ap[i]),
                    textcoords='offset points', xytext=(0, 4), ha='center',
                    fontsize=8, color=COL[ks[i]], fontweight='bold')
    ax.text(0.985, 0.06,
            'the certified depth is far below the measured one, and must be:\n'
            'it holds for EVERY delay, every position, every removal state,\n'
            'every damping in the range — simultaneously',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='right',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_k3_depth.png')


def fig_margin():
    s56, _ = _store()
    ks = [k for k in SHOW if k in s56]
    dm = np.array([s56[k]['delta_max'] for k in ks])
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.barh(np.arange(len(ks)), dm, 0.55, color=[COL[k] for k in ks])
    ax.axvline(1.0, color='k', lw=1.8, ls='--',
               label='the physical set itself')
    ax.set_yticks(np.arange(len(ks)), [LAB[k] for k in ks], fontsize=8)
    ax.set_xlabel('$\\delta_{\\max}$ — how many times the uncertainty set '
                  'can be inflated')
    ax.set_title('Robustness margin, certified')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, max(dm) * 1.25)
    for i, v in enumerate(dm):
        ax.annotate(f'{v:.2f}×', (v, i), textcoords='offset points',
                    xytext=(5, -3), fontsize=9, fontweight='bold',
                    color=COL[ks[i]])
    ax.text(0.985, 0.06,
            'every controller survives more than the physical set —\n'
            'the margin is the factor by which it can grow first',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='right',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_k4_margin.png')


# ---------------------------------------------------------------------------
def fig_family():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    made = load_controllers(plate, plant)
    kw = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
              zetas=(C.ZETA_LO, C.ZETA_HI), xis=(0.5, 1.0))
    V, npl, ws = CF2.vertices(plant, made['ps_ac'](plant), **kw)
    peaks = np.array([CF2.crossing_delays(A, Ad)[1] for (A, Ad) in V])
    peaks = peaks[np.isfinite(peaks)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    a.hist(peaks, bins=24, color='#0E7490', alpha=.85)
    a.axvline(1.0, color='k', lw=1.8, ls='--', label='threshold')
    a.axvline(peaks.max(), color='#9F1239', lw=1.6,
              label=f'worst vertex: {peaks.max():.3f}')
    a.set_xlabel('peak $\\rho$ at the vertex'); a.set_ylabel('count')
    a.set_title(f'PS-AC over all {len(peaks)} vertices')
    a.legend(loc='upper left', fontsize=8)

    txt = [('tool position $x_P$', f'{kw["n_pos"]} points on $[0, l_P]$'),
           ('removed material $\\eta$', f'{len(kw["etas"])} values on '
            f'$[0,\\ {C.ETA_MAX}]$'),
           ('damping scale', f'$\\times${C.ZETA_LO} and $\\times${C.ZETA_HI}'),
           ('mid-pass state $\\xi$', '0.5 and 1.0'),
           ('milling coefficient', f'$[{C.ALPHA_LO},\\ {C.ALPHA_HI}]\\ '
            r'\bar\alpha_4$ hull'),
           ('delay $\\tau$', 'ALL of $[0, \\infty)$')]
    b.axis('off')
    b.set_title('What the certificate quantifies over')
    for i, (nm, v) in enumerate(txt):
        y = 0.94 - i * 0.135
        b.text(0.02, y, nm, fontsize=9.5, fontweight='bold', va='top',
               transform=b.transAxes)
        b.text(0.52, y, v, fontsize=9, va='top', color='0.3',
               transform=b.transAxes)
    b.text(0.02, 0.015,
           f'{len(V)} vertices, all satisfied simultaneously.\n'
           'A simulation samples this space; the certificate covers it.',
           fontsize=8, color='0.3', va='bottom', transform=b.transAxes,
           bbox=BOX)
    return save(fig, 'fig_k5_family.png')


# ---------------------------------------------------------------------------
def fig_summary():
    s56, _ = _store()
    ks = [k for k in SHOW if k in s56]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    # every bar is "higher is better": the crossing peak is shown as its
    # reciprocal, so it reads as a margin like the other two
    metrics = [('$1/\\rho_{peak}$  (threshold margin)',
                [1.0 / s56[k]['peak'] for k in ks], 1.0),
               ('$a_p^\\infty$  (mm)', [s56[k]['ap_inf'] * 1e3 for k in ks], None),
               ('$\\delta_{\\max}$', [s56[k]['delta_max'] for k in ks], 1.0)]
    x = np.arange(len(ks))
    w = 0.26
    for j, (nm, vals, ref) in enumerate(metrics):
        v = np.array(vals, float)
        vn = v / np.max(np.abs(v))
        ax.bar(x + (j - 1) * w, vn, w, label=nm,
               color=['#0E7490', '#C2410C', '#7C3AED'][j], alpha=.9)
        for i in range(len(ks)):
            ax.annotate(f'{v[i]:.3f}', (x[i] + (j - 1) * w, vn[i]),
                        textcoords='offset points', xytext=(0, 4),
                        ha='center', fontsize=7.2, rotation=90)
    ax.set_xticks(x, [LAB[k] for k in ks], fontsize=8, rotation=10)
    ax.set_ylabel('each metric normalised to its own maximum')
    ax.set_ylim(0, 1.34)
    ax.set_title('All three certified quantities — higher is better in every bar')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    lk = ', '.join(f'{LAB[k]}: '
                   + ('yes' if s56[k]['lk'] else
                      ('n/a' if not s56[k].get('lk_attempted', True) else 'no'))
                   for k in ks)
    ax.text(0.5, 0.03, 'common-$P$ Lyapunov–Krasovskii — ' + lk,
            transform=ax.transAxes, fontsize=7.4, color='0.3', ha='center',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_k6_summary.png')


def main():
    print('writing certificate figures ->', OUT)
    fig_depth(); fig_margin(); fig_summary()
    fig_crossing(); fig_grid(); fig_family()


if __name__ == '__main__':
    main()
