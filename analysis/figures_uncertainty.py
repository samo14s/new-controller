"""
figures_uncertainty.py — the audit figures for Phase 2.
========================================================
Eqs. (22)-(25) and the closing sentence of Section 3.2 are the block this whole
contribution turns on.  The audit found the set to be, at the same time, TOO
SMALL where it matters and TOO LARGE where the physics cannot go.  These are the
pictures of that.

    fig_u1_cross      F1 — Eq. (25) keeps the smallest of three terms of an
                      interval product and drops the two larger ones
    fig_u2_rank       F2 — the reachable set of alpha4 D^T D is a 2-D surface
                      of rank-one semidefinite matrices; the box of Eq. (24) is
                      4-D and contains non-symmetric, rank-two, indefinite ones
    fig_u3_experiment F3 — the paper's own measured frequency rise leaves the
                      paper's own box
    fig_u4_onesided   T1/T2 — material is removed, never added: the reachable
                      set is a one-sided CURVE, not a symmetric box
    fig_u5_coupling   T4 — the mid-pass state creates modal coupling that a
                      DIAGONAL Delta cannot represent at any magnitude
    fig_u6_amplitude  the two descriptions side by side, per matrix entry

    python analysis/figures_uncertainty.py  -> results/figures_uncertainty/*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_plate, RPM_S, AP_S, AE_S, RESULTS
from chebyshev_plate import ChebyshevPlate
from milling_dynamics import alpha4_average, dtd_paper_gauge

OUT = os.path.join(RESULTS, 'figures_uncertainty')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 1.8, 'figure.constrained_layout.use': True,
})
C_IMPL, C_PAPER, C_BAD, C_OK, C_GREY = '#0E7490', '#C2410C', '#9F1239', '#15803D', '0.55'
BOX = dict(fc='white', ec='0.86', lw=.7, pad=3.5)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print('  ', p)
    return p


def _gauge():
    bare = ChebyshevPlate(PX=14, PZ=14)
    xs, DtD, DD0, LDD, _ = dtd_paper_gauge(bare, 401)
    abar4 = alpha4_average(RPM_S, AP_S, bare.hp, AE_S)
    return bare, xs, DtD, DD0, LDD, abar4


# ---------------------------------------------------------------------------
def fig_cross():
    """F1: the three terms of an interval product, and which one is kept."""
    _, xs, DtD, DD0, LDD, abar4 = _gauge()
    a40, LPa = 1.6 * abar4, 1.3 * abar4
    t1 = np.abs(a40) * LDD                     # |a0| L_d   -- dropped
    t2 = np.abs(DD0) * abs(LPa)                # |d0| L_a   -- dropped
    t3 = abs(LPa) * LDD                        # L_a L_d    -- kept, Eq. (25)
    exact = t1 + t2 + t3
    lab = ['(1,1)', '(1,2)', '(2,1)', '(2,2)']
    ij = [(0, 0), (0, 1), (1, 0), (1, 1)]
    T1 = np.array([t1[i] for i in ij]); T2 = np.array([t2[i] for i in ij])
    T3 = np.array([t3[i] for i in ij]); EX = np.array([exact[i] for i in ij])
    idx = np.arange(4)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    w = 0.2
    a.bar(idx - 1.5 * w, T3, w, color=C_OK,
          label=r'$L_{P\alpha} L_{DD}$  — KEPT by Eq. (25)')
    a.bar(idx - 0.5 * w, T1, w, color=C_BAD,
          label=r'$|\alpha_{40}|\,L_{DD}$  — dropped')
    a.bar(idx + 0.5 * w, T2, w, color='#F59E0B',
          label=r'$|DD_0|\,L_{P\alpha}$  — dropped')
    a.bar(idx + 1.5 * w, EX, w, color='0.35', label='exact envelope = sum')
    a.set_xticks(idx, lab); a.set_yscale('log')
    a.set_ylim(1e2, 4e5)
    a.set_xlabel('entry of $\\Delta_{PrD}$'); a.set_ylabel('term magnitude (N/m)')
    a.set_title('F1 — the envelope has three terms; Eq. (25) keeps the smallest')
    a.legend(loc='lower left', fontsize=7.8, ncol=2)
    for k in range(4):
        a.annotate(f'{100*T3[k]/EX[k]:.1f} %', (k - 1.5 * w, T3[k]),
                   textcoords='offset points', xytext=(0, 4), ha='center',
                   fontsize=7.5, color=C_OK, fontweight='bold')

    ratio = EX / np.maximum(np.array([t3[i] for i in ij]), 1e-30)
    b.bar(idx, ratio, color=C_BAD)
    b.axhline(1.0, color='0.35', lw=1.1, ls='--')
    b.set_xticks(idx, lab); b.set_yscale('log')
    b.set_xlabel('entry of $\\Delta_{PrD}$')
    b.set_ylabel('exact envelope / Eq. (25)')
    b.set_title('Under-covering factor, entry by entry')
    for k in range(4):
        b.annotate(f'{ratio[k]:.1f}×', (k, ratio[k]),
                   textcoords='offset points', xytext=(0, 5), ha='center',
                   fontsize=9, fontweight='bold', color=C_BAD)
    b.text(0.5, 0.06,
           'the (1,1) entry — the regenerative stiffness of mode 1, the one\n'
           'the paper calls "nearly constant" — is the worst covered of the four',
           transform=b.transAxes, fontsize=8, color='0.3', ha='center', bbox=BOX)
    return save(fig, 'fig_u1_cross.png')


# ---------------------------------------------------------------------------
def fig_rank():
    """F2: reachable 2-D surface against the 4-D box."""
    _, xs, DtD, DD0, LDD, abar4 = _gauge()
    a40, LPa = 1.6 * abar4, 1.3 * abar4
    L = abs(LPa) * LDD
    lo, hi = 0.3 * abar4, 2.9 * abar4

    # reachable: a * D^T D over a in [lo, hi] and x along the edge
    A = np.linspace(lo, hi, 41)
    R = np.array([[a * m for m in DtD] for a in A]).reshape(-1, 2, 2)
    # box corners of Eq. (24)-(25)
    corners = []
    for s in np.ndindex(2, 2, 2, 2):
        sg = np.array(s) * 2 - 1
        corners.append(a40 * DD0 + np.array([[sg[0] * L[0, 0], sg[1] * L[0, 1]],
                                             [sg[2] * L[1, 0], sg[3] * L[1, 1]]]))
    corners = np.array(corners)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # (1) the (11, 22) plane, with BOTH boxes
    a.scatter(R[:, 0, 0], R[:, 1, 1], s=3, color=C_IMPL, alpha=.35,
              label='reachable  $\\alpha_4 D^TD$')
    bx = [corners[:, 0, 0].min(), corners[:, 0, 0].max()]
    by = [corners[:, 1, 1].min(), corners[:, 1, 1].max()]
    a.add_patch(plt.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                              fill=False, ec=C_BAD, lw=2.0, ls='--',
                              label='box of Eq. (25) — TOO SMALL'))
    Lx = (np.abs(a40) * LDD + np.abs(DD0) * abs(LPa) + abs(LPa) * LDD)
    ex = [a40 * DD0[0, 0] - Lx[0, 0], a40 * DD0[0, 0] + Lx[0, 0]]
    ey = [a40 * DD0[1, 1] - Lx[1, 1], a40 * DD0[1, 1] + Lx[1, 1]]
    a.add_patch(plt.Rectangle((min(ex), min(ey)), abs(ex[1] - ex[0]),
                              abs(ey[1] - ey[0]), fill=False, ec='#F59E0B',
                              lw=1.8, ls='-',
                              label='exact envelope — contains, but is a BOX'))
    a.set_xlabel('entry (1,1)'); a.set_ylabel('entry (2,2)')
    a.set_title('Both failures in one plane')
    a.legend(loc='lower left', fontsize=7.6)
    a.text(0.97, 0.95,
           'F1: the published box misses\nmost of the reachable set\n\n'
           'F2: the box that does contain it\nis mostly empty — the set is a\n'
           'thin 2-D surface, not a rectangle',
           transform=a.transAxes, fontsize=7.8, color='0.2', ha='right',
           va='top', bbox=BOX)

    # (2) symmetry: the box lets (1,2) and (2,1) differ
    d_off = corners[:, 0, 1] - corners[:, 1, 0]
    a2 = np.abs(R[:, 0, 1] - R[:, 1, 0]).max()
    b.hist(d_off, bins=9, color=C_BAD, alpha=.85)
    b.axvline(0, color=C_OK, lw=2.4,
              label=f'reachable: max |asymmetry| = {a2:.1e}')
    b.set_xlabel('$(\\Delta)_{12} - (\\Delta)_{21}$ at the box corners')
    b.set_ylabel('count')
    b.set_title('Symmetry: broken by construction')
    b.legend(loc='upper center', fontsize=8)

    # (3) definiteness: eigenvalues of the symmetric part
    ec = np.array([np.linalg.eigvalsh(0.5 * (m + m.T)) for m in corners])
    er = np.array([np.sort(np.linalg.eigvalsh(0.5 * (m + m.T))) for m in R])
    c.scatter(ec[:, 0], ec[:, 1], s=42, color=C_BAD, marker='s',
              label='box corners', zorder=3)
    c.plot(er[:, 0], er[:, 1], color=C_IMPL, lw=3, alpha=.8,
           label='reachable (one eigenvalue $\\equiv$ 0)')
    c.axhline(0, color='0.4', lw=.9); c.axvline(0, color='0.4', lw=.9)
    c.set_xlabel('smaller eigenvalue'); c.set_ylabel('larger eigenvalue')
    c.set_title('Definiteness: the box changes sign')
    c.legend(loc='upper left', fontsize=8)
    c.set_ylim(min(ec[:, 1].min(), -2e3) * 1.15, ec[:, 1].max() * 1.35)
    n_ind = int(np.sum(ec[:, 0] * ec[:, 1] < 0))
    c.text(0.5, 0.06,
           f'{n_ind} of 16 corners are INDEFINITE;\n'
           'every reachable matrix has one eigenvalue\n'
           'exactly zero and the other of one fixed sign',
           transform=c.transAxes, fontsize=8, color='0.3', ha='center', bbox=BOX)
    return save(fig, 'fig_u2_rank.png')


# ---------------------------------------------------------------------------
def fig_experiment():
    """F3: the paper's own measurement leaves the paper's own box."""
    m = np.linspace(-0.10, 0.10, 401)          # mass perturbation
    k = np.linspace(-0.10, 0.10, 401)          # stiffness perturbation
    MM, KK = np.meshgrid(m, k)
    rise = 100.0 * (np.sqrt((1 + KK) / (1 + MM)) - 1.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cs = ax.contourf(100 * MM, 100 * KK, rise, levels=18, cmap='RdYlBu_r')
    fig.colorbar(cs, ax=ax, label='frequency change (%)')
    ax.contour(100 * MM, 100 * KK, rise, levels=[10.55], colors='k',
               linewidths=1.6, linestyles='--')
    ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color='k', lw=2)
    ax.plot(-10, 10, 'k*', ms=15)
    ax.annotate('best corner of the box:\n+10.55 %', (-10, 10),
                textcoords='offset points', xytext=(14, -34), fontsize=8.5,
                bbox=BOX)
    ax.set_xlabel('mass perturbation (%)'); ax.set_ylabel('stiffness perturbation (%)')
    ax.set_title('F3 — the box of Section 3.2 against the paper\'s own measurement')
    ax.text(0.5, 0.055,
            'Section 5 of the paper MEASURES a +17 % rise (540 → 632 Hz).\n'
            'The most the ±10 % box can produce is +10.55 %.\n'
            'A symmetric box would need ±15.6 % to contain it.',
            transform=ax.transAxes, fontsize=8.5, color='0.15', ha='center',
            bbox=BOX)
    return save(fig, 'fig_u3_experiment.png')


# ---------------------------------------------------------------------------
def fig_onesided():
    """T1/T2: the reachable set is a one-sided curve, not a symmetric box."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'phase2'))
    from uncertainty import RemovalFamily
    import config as C
    fam = RemovalFamily(n=2)
    d_eta = AP_S / build_plate().hp
    g = np.linspace(0.0, C.ETA_MAX, 25)
    M11, M22, K11, K22 = [], [], [], []
    for e in g:
        M, _, K = fam.matrices(e, xi=1.0, z=1.0, d_eta=d_eta)
        M11.append(M[0, 0]); M22.append(M[1, 1])
        K11.append(K[0, 0]); K22.append(K[1, 1])
    M11, M22 = np.array(M11), np.array(M22)
    K11 = np.array(K11) / K11[0]; K22 = np.array(K22) / K22[0]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, (y1, y2, ttl, ylab) in zip(
            (a, b),
            [((M11 - 1) * 100, (M22 - 1) * 100, 'modal mass', r'$\Delta M_{ii}/M_{ii}$ (%)'),
             ((K11 - 1) * 100, (K22 - 1) * 100, 'modal stiffness', r'$\Delta K_{ii}/K_{ii}$ (%)')]):
        ax.axhspan(-10, 10, color=C_BAD, alpha=.10,
                   label='box of Section 3.2 ($\\pm$10 %)')
        ax.axhline(0, color='0.4', lw=.9)
        ax.plot(g * 100, y1, 'o-', color=C_IMPL, ms=3.5, label='mode 1')
        ax.plot(g * 100, y2, 's-', color=C_PAPER, ms=3.5, label='mode 2')
        ax.set_xlabel('removed volume fraction $\\eta$ (%)')
        ax.set_ylabel(ylab)
        ax.set_title(f'T1/T2 — {ttl}: one-sided and monotone')
        ax.legend(loc='lower left', fontsize=8)
    a.text(0.5, 0.30,
           f'mode 1 reaches {(M11[-1]-1)*100:.1f} % — the box allows 10 %\n'
           'and spends its whole POSITIVE half on adding material,\nwhich milling cannot do',
           transform=a.transAxes, fontsize=8, color='0.2', ha='center', bbox=BOX)
    b.text(0.5, 0.55,
           f'mode 1 moves {(K11[-1]-1)*100:.3f} % over the WHOLE range —\n'
           'the box gives it $\\pm$10 %, i.e. 266× more than needed,\n'
           'while giving the mass 2.7× less than it needs',
           transform=b.transAxes, fontsize=8, color='0.2', ha='center', bbox=BOX)
    return save(fig, 'fig_u4_onesided.png')


# ---------------------------------------------------------------------------
def fig_coupling():
    """T4: the mid-pass state creates coupling a DIAGONAL Delta cannot express."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'phase2'))
    from uncertainty import RemovalFamily
    import config as C
    fam = RemovalFamily(n=2)
    hp = build_plate().hp
    xi = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for eta, col in ((C.ETA_MAX / 2, C_IMPL), (C.ETA_MAX, C_PAPER)):
        off, dia = [], []
        for x in xi:
            M, _, _ = fam.matrices(eta, xi=x, z=1.0, d_eta=eta)
            off.append(abs(M[0, 1])); dia.append(abs(M[0, 0] - 1.0))
        ax.plot(xi, off, 'o-', color=col, ms=3.5,
                label=f'$|\\Delta M_{{12}}|$ at $\\eta$ = {eta:.4f}')
    ax.axhline(0.10, color=C_BAD, ls='--', lw=1.4,
               label="the box's own diagonal half-width (10 % of $M_{11}$ = 1)")
    ax.set_xlabel('fraction of the pass already machined  $\\xi$')
    ax.set_ylabel('off-diagonal modal mass  $|\\Delta M_{12}|$')
    ax.set_title('T4 — mid-pass coupling, which a diagonal $\\Delta$ cannot represent')
    ax.legend(loc='lower center', fontsize=8)
    ax.set_ylim(-0.008, 0.145)
    ax.text(0.985, 0.96,
            'Eq. (22) makes $\\Delta_{PrM}$ DIAGONAL by construction, so this\n'
            'entry is identically zero in the published description — at ANY\n'
            'magnitude of the ten parameters.\n\n'
            'It is not under-sized. It is ABSENT — and it reaches the same\n'
            'order as the diagonal half-width the box does carry.',
            transform=ax.transAxes, fontsize=8, color='0.2', ha='right',
            va='top', bbox=BOX)
    return save(fig, 'fig_u5_coupling.png')


# ---------------------------------------------------------------------------
def fig_amplitude():
    """The two descriptions side by side, per matrix entry."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'phase2'))
    from uncertainty import RemovalFamily
    import config as C
    fam = RemovalFamily(n=2)
    d_eta = AP_S / build_plate().hp
    M0, _, K0 = fam.matrices(0.0, 1.0, 1.0, d_eta)
    Me, Ce, Ke = fam.matrices(C.ETA_MAX, 1.0, 1.0, d_eta)
    C0 = fam.matrices(0.0, 1.0, 1.0, d_eta)[1]
    need = [abs(Me[0, 0] / M0[0, 0] - 1), abs(Me[1, 1] / M0[1, 1] - 1),
            abs(Ke[0, 0] / K0[0, 0] - 1), abs(Ke[1, 1] / K0[1, 1] - 1),
            abs(Ce[0, 0] / C0[0, 0] - 1), abs(Ce[1, 1] / C0[1, 1] - 1)]
    need = 100 * np.array(need)
    given = np.array([10., 10., 10., 10., 20., 20.])
    lab = ['$M_{11}$', '$M_{22}$', '$K_{11}$', '$K_{22}$', '$C_{11}$', '$C_{22}$']
    i = np.arange(6); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(i - w / 2, given, w, color=C_BAD, label='given by Section 3.2')
    ax.bar(i + w / 2, need, w, color=C_IMPL, label='needed by the physics')
    ax.set_xticks(i, lab); ax.set_ylabel('half-width (%)')
    ax.set_yscale('log')
    ax.set_title('The same "10 %" is 2.7× too small on one entry '
                 'and 266× too large on another')
    ax.legend(loc='lower left', ncol=2, fontsize=8)
    ax.set_ylim(1.2e-2, 1.2e2)
    for k in range(6):
        r = need[k] / given[k]
        txt = (f'{r:.2f}× under' if r > 1 else f'{1/r:.0f}× over')
        ax.annotate(txt, (k + w / 2, need[k]),
                    textcoords='offset points', xytext=(0, 5), ha='center',
                    fontsize=8, color=(C_BAD if r > 1 else '#B45309'),
                    fontweight='bold')
    return save(fig, 'fig_u6_amplitude.png')


def main():
    print('writing uncertainty-audit figures ->', OUT)
    fig_cross(); fig_rank(); fig_experiment()
    fig_onesided(); fig_coupling(); fig_amplitude()


if __name__ == '__main__':
    main()
