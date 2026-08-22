"""
figures_model.py — the verification figures for Phases 0-1.
============================================================
The reproduction verdict rests on four levels that validate DIFFERENT things,
and on one defect.  The numbers live in docs/01_REPRODUCTION.md; these are the
pictures, so the claims can be checked by eye and not only by table.

    fig_v1_frequencies   natural frequencies against Table 4, and the residual
                         pattern -- uniform sign is the whole argument that the
                         3 % gap is one global stiffness offset, not a
                         mode-shape error
    fig_v2_antiresonance the transfer function's ZEROS, which no calibration can
                         move: validation of the mode shapes independent of the
                         frequency fit
    fig_v3_dtd           Fig. 7 of the paper -- the spatial dependence D^T D
                         along the top edge, reproduced element by element
    fig_v4_alpha         the non-smooth milling coefficients over one tooth
                         period, with the averages of Eq. (23)
    fig_v5_sign          the sign-convention defect: the same plant, the two
                         conventions, and which one matches the paper's own
                         statement that cutting at the operating point diverges
    fig_v6_lobes         stability lobes against spindle speed at three tool
                         positions -- the position dependence the whole
                         contribution turns on

    python analysis/figures_model.py     -> results/figures_model/*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (build_plate, F_MEASURED, F_THEORETICAL, RPM_S, AP_S,
                     AE_S, FZ_S, RESULTS)
from chebyshev_plate import ChebyshevPlate
from milling_dynamics import (alpha4_series, alpha4_average, dtd_paper_gauge,
                              N_TEETH)
from stability_fdm import stability_limit
from time_domain import TimeSim

OUT = os.path.join(RESULTS, 'figures_model')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'semibold',
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 1.8, 'figure.constrained_layout.use': True,
})
C_IMPL, C_PAPER, C_BAD, C_OK = '#0E7490', '#C2410C', '#9F1239', '#15803D'


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print('  ', p)
    return p


# ---------------------------------------------------------------------------
def fig_frequencies():
    """Natural frequencies, and why the residual is one stiffness offset."""
    bare = ChebyshevPlate(PX=14, PZ=14)
    f_impl = np.asarray(bare.freq_n[:5], float)
    f_th = np.asarray(F_THEORETICAL[:5], float)
    f_ms = np.asarray(F_MEASURED[:5], float)
    err = 100.0 * (f_impl / f_th - 1.0)
    idx = np.arange(5)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    w = 0.27
    a.bar(idx - w, f_impl, w, label='implementation', color=C_IMPL)
    a.bar(idx, f_th, w, label='paper, Table 4 (model)', color=C_PAPER)
    a.bar(idx + w, f_ms, w, label='paper, Table 4 (measured)', color='0.62')
    a.set_xticks(idx, [f'mode {i+1}' for i in idx])
    a.set_ylabel('frequency (Hz)')
    a.set_title('Natural frequencies, bare plate')
    a.legend(loc='upper left')
    for i, (v, e) in enumerate(zip(f_impl, err)):
        a.text(i - w, v + 90, f'{v:.0f}', ha='center', fontsize=7.5,
               color=C_IMPL)

    b.axhline(0, color='0.4', lw=.9)
    b.plot(idx, err, 'o-', color=C_IMPL, ms=6)
    b.fill_between(idx, err, 0, color=C_IMPL, alpha=.12)
    b.set_xticks(idx, [f'{i+1}' for i in idx])
    b.set_xlabel('mode'); b.set_ylabel('error vs Table 4 (%)')
    b.set_title('Residual: same sign on all five modes')
    b.set_ylim(min(err) - 1.2, 0.6)
    for i, e in enumerate(err):
        b.annotate(f'{e:+.2f} %', (i, e), textcoords='offset points',
                   xytext=(0, -13), ha='center', fontsize=7.5)
    b.text(0.5, 0.06,
           'a mode-shape error would change sign across modes;\n'
           'a uniform offset is one missing stiffness term',
           transform=b.transAxes, ha='center', fontsize=8, color='0.35')
    return save(fig, 'fig_v1_frequencies.png')


# ---------------------------------------------------------------------------
def fig_antiresonance():
    """Zeros of the collocated force response: mode shapes, calibration-free."""
    p = build_plate()                       # calibrated, as in Phase 1
    n = 5
    wn = np.asarray(p.omega_n[:n], float)
    z = np.asarray(p.zeta_modes[:n], float)
    Do = p.D_row(p.lp, p.hp)[:n]
    f = np.linspace(1.0, 5000.0, 40001)
    w = 2 * np.pi * f
    G = np.zeros_like(w, dtype=complex)
    for k in range(n):
        G += Do[k] ** 2 / (wn[k] ** 2 - w ** 2 + 2j * z[k] * wn[k] * w)
    mag = np.abs(G)
    db = 20 * np.log10(mag)
    az = [f[i] for i in range(1, len(f) - 1)
          if mag[i] < mag[i - 1] and mag[i] < mag[i + 1]]

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.semilogx(f, db, color=C_IMPL, lw=1.6)
    for k in range(n):
        ax.axvline(wn[k] / (2 * np.pi), color=C_PAPER, lw=1.0, ls=':', alpha=.85)
    for v in az:
        ax.axvline(v, color=C_OK, lw=1.2, ls='--', alpha=.9)
        ax.annotate(f'{v:.0f}', (v, db.max()), textcoords='offset points',
                    xytext=(3, -10), fontsize=7.5, color=C_OK)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|y_{\rm obs}/F|$  (dB re 1 m/N)')
    ax.set_title('Collocated force response — poles (dotted), '
                 'antiresonances (dashed)')
    ax.set_xlim(200, 5000)
    ax.text(0.015, 0.05,
            'poles at ' + ', '.join(f'{v/(2*np.pi):.0f}' for v in wn) + ' Hz'
            '  (calibrated onto Table 4)\n'
            'zeros at ' + ', '.join(f'{v:.1f}' for v in az) + ' Hz\n\n'
            'the zeros are fixed by the mode SHAPES alone: no frequency\n'
            'calibration can move them, so they validate the shapes on\n'
            'their own, independently of the 3 % stiffness offset',
            transform=ax.transAxes, fontsize=8, color='0.3', va='bottom',
            bbox=dict(fc='white', ec='0.85', lw=.7, pad=4))
    return save(fig, 'fig_v2_antiresonance.png')


# ---------------------------------------------------------------------------
def fig_dtd():
    """Fig. 7 of the paper: D^T D along the top edge, element by element."""
    bare = ChebyshevPlate(PX=14, PZ=14)
    xs, DtD, DD0, LDD, _ = dtd_paper_gauge(bare, n_pos=401)
    xn = xs / xs.max()
    # the paper's own read-out of Fig. 7, mean and amplitude of each element
    target = [('$DD_{11}$ mean', DD0[0, 0], 3.705),
              ('$DD_{11}$ ampl', LDD[0, 0], 0.105),
              ('$DD_{12}$ ampl', LDD[0, 1], 3.45),
              ('$DD_{22}$ mean', DD0[1, 1], 1.675),
              ('$DD_{22}$ ampl', LDD[1, 1], 1.675)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    a.plot(xn, DtD[:, 0, 0], color=C_IMPL, label=r'$(D^TD)_{11}$')
    a.plot(xn, DtD[:, 1, 1], color=C_PAPER, label=r'$(D^TD)_{22}$')
    a.plot(xn, DtD[:, 0, 1], color='0.5', label=r'$(D^TD)_{12}$')
    a.set_xlabel('tool position  $x_P / l_P$'); a.set_ylabel('element value')
    a.set_title("Fig. 7 reproduced — spatial dependence of $D^TD$")
    a.legend(ncol=3, loc='upper center')
    a.set_xlim(0, 1)
    # mode 2 node
    k = int(np.argmin(np.abs(DtD[:, 1, 1])))
    a.axvline(xn[k], color='0.35', ls=':', lw=1.1)
    a.annotate('mode-2 node', (xn[k], 1.15), fontsize=8, color='0.35',
               ha='center', backgroundcolor='white')
    det = DtD[:, 0, 0] * DtD[:, 1, 1] - DtD[:, 0, 1] ** 2
    a.text(0.02, 0.05,
           f'max |det| over {len(xs)} positions = {np.abs(det).max():.1e}\n'
           f'against an element scale of {np.abs(DtD).max():.2f}\n'
           r'$D^TD$ is an outer product: rank one at EVERY position',
           transform=a.transAxes, fontsize=8, color='0.35', va='bottom')

    lab = [t[0] for t in target]
    mine = np.array([t[1] for t in target])
    theirs = np.array([t[2] for t in target])
    i = np.arange(len(target)); w = 0.36
    b.bar(i - w / 2, mine, w, color=C_IMPL, label='implementation')
    b.bar(i + w / 2, theirs, w, color=C_PAPER, label="paper, Fig. 7 read-out")
    b.set_xticks(i, lab, fontsize=8)
    b.set_ylabel('value')
    b.set_title('Element by element, against the published curve')
    b.legend(loc='upper right')
    for k in range(len(target)):
        e = 100 * (mine[k] / theirs[k] - 1)
        b.annotate(f'{e:+.2f} %', (k, max(mine[k], theirs[k])),
                   textcoords='offset points', xytext=(0, 5), ha='center',
                   fontsize=7.5,
                   color=(C_OK if abs(e) < 5 else C_BAD))
    b.set_ylim(0, 4.4)
    return save(fig, 'fig_v3_dtd.png')


# ---------------------------------------------------------------------------
def fig_alpha():
    """The non-smooth milling coefficients over one tooth period."""
    p = build_plate()
    n = 4000
    tau = 60.0 / (N_TEETH * RPM_S)
    t = (np.arange(n) + 0.5) * tau / n
    a3, a4 = alpha4_series(RPM_S, AP_S, p.hp, n, AE_S)
    a4b = alpha4_average(RPM_S, AP_S, p.hp, AE_S)
    cut = a4 != 0.0
    t0, t1 = t[cut].min(), t[cut].max()

    fig, (ax, cx, bx) = plt.subplots(1, 3, figsize=(12.4, 3.6))

    # --- full period, with the Eq. (23) band shaded ------------------------
    lo, hi = sorted([0.3 * a4b, 2.9 * a4b])
    ax.axhspan(lo, hi, color=C_PAPER, alpha=.10,
               label=r'Eq. (23) band $[0.3,\,2.9]\,\bar\alpha_4$')
    ax.plot(t * 1e3, a4, color=C_IMPL, label=r'$\alpha_4(t)$')
    ax.axhline(a4b, color=C_PAPER, ls='--', lw=1.4,
               label=r'$\bar\alpha_4$ = %.0f N/m' % a4b)
    ax.set_xlabel('time (ms)'); ax.set_ylabel(r'$\alpha_4$  (N/m)')
    ax.set_title('Regenerative coefficient, one tooth period')
    ax.legend(loc='lower left', fontsize=7.6)
    ax.text(0.97, 0.06,
            f'peak / mean = {np.abs(a4).max()/abs(a4b):.1f}',
            transform=ax.transAxes, fontsize=8.5, color='0.3', ha='right')

    # --- zoom on the in-cut window ----------------------------------------
    m = (t >= t0 - 0.06e-3) & (t <= t1 + 0.06e-3)
    cx.axhspan(lo, hi, color=C_PAPER, alpha=.10)
    cx.plot(t[m] * 1e3, a4[m], color=C_IMPL)
    cx.axhline(a4b, color=C_PAPER, ls='--', lw=1.4)
    cx.axhline(0, color='0.6', lw=.9)
    cx.set_xlabel('time (ms)')
    cx.set_title('Zoom: the tooth in cut')
    cx.text(0.5, 0.06,
            f'the tooth cuts for {100*cut.mean():.1f} % of the period\n'
            f'({(t1-t0)*1e3:.3f} ms of {tau*1e3:.3f} ms)\n'
            r'and $\alpha_4$ is exactly zero the rest of the time',
            transform=cx.transAxes, fontsize=8, color='0.3', ha='center')

    # --- the feed coefficient ---------------------------------------------
    bx.plot(t * 1e3, a3, color=C_PAPER, label=r'$\alpha_3(t)$')
    bx.axhline(float(np.mean(a3)), color='0.45', ls='--',
               label=r'mean = %.0f' % float(np.mean(a3)))
    bx.set_xlabel('time (ms)')
    bx.set_ylabel(r'$\alpha_3$  (N/m per unit feed)')
    bx.set_title(f'Feed coefficient — $\\tau$ = {tau*1e3:.3f} ms '
                 f'at {RPM_S} rpm')
    bx.legend(loc='upper left', fontsize=8)
    bx.text(0.97, 0.5,
            f'radial immersion\n$a_e$ = {AE_S*1e3:.2f} mm\n'
            f'$a_p$ = {AP_S*1e3:.1f} mm',
            transform=bx.transAxes, fontsize=8, color='0.3', ha='right')
    return save(fig, 'fig_v4_alpha.png')


# ---------------------------------------------------------------------------
def fig_sign():
    """The sign-convention defect, in the time domain."""
    p = build_plate()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    txt = []
    for sgn, col, lab in ((+1.0, C_OK, 'sign = +1 (Eq. 13 as published)'),
                          (-1.0, C_BAD, 'sign = −1')):
        sim = TimeSim(p, RPM_S, AP_S, sign=sgn, ae=AE_S, fz=FZ_S, n_modes=5)
        r = sim.run(T=0.16, moving=False, x0=0.0)
        y = np.asarray(r['y_obs'], float) * 1e6
        t = np.asarray(r['t'], float)
        ax.plot(t * 1e3, y, color=col, label=lab, lw=1.6)
        lim = stability_limit(p, RPM_S, 0.0, coeff_scale=sgn)
        txt.append(f'{lab}:  a_p,lim = {lim*1e3:.4f} mm')
    ax.set_xlabel('time (ms)'); ax.set_ylabel('sensor displacement (μm)')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_title(f'Same plant, two sign conventions — {RPM_S} rpm, '
                 f'$a_p$ = {AP_S*1e3:.1f} mm')
    ax.legend(loc='lower left', ncol=2)
    ax.text(0.985, 0.97, '\n'.join(txt) +
            '\n\nthe paper states the cut at this point DIVERGES and measures'
            '\na limit near 0.1 mm — only +1 reproduces that',
            transform=ax.transAxes, fontsize=8, color='0.3', va='top',
            ha='right', bbox=dict(fc='white', ec='0.85', lw=.7, pad=4))
    ax.set_ylim(-3e4, 3e4)
    return save(fig, 'fig_v5_sign.png')


# ---------------------------------------------------------------------------
def fig_lobes(n_rpm=60):
    """Stability lobes at three tool positions."""
    p = build_plate()
    rpms = np.linspace(2000, 9000, n_rpm)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for frac, col, lab in ((0.0, C_IMPL, '$x_P = 0$'),
                           (0.5, C_PAPER, '$x_P = l_P/2$'),
                           (1.0, '0.45', '$x_P = l_P$')):
        lim = [stability_limit(p, r, frac * p.lp, hi=2.0e-3, tol=5e-6, m=40)
               for r in rpms]
        ax.plot(rpms, np.array(lim) * 1e3, color=col, label=lab)
    ax.axvline(RPM_S, color='0.3', ls=':', lw=1.1)
    ax.annotate(f'{RPM_S} rpm', (RPM_S, ax.get_ylim()[1]),
                textcoords='offset points', xytext=(4, -12), fontsize=8,
                color='0.3')
    ax.set_xlabel('spindle speed (rpm)')
    ax.set_ylabel('$a_{p,\\mathrm{lim}}$ (mm)')
    ax.set_title('Stability lobes — and how much they move with tool position')
    ax.legend()
    return save(fig, 'fig_v6_lobes.png')


def main():
    print('writing verification figures ->', OUT)
    fig_frequencies()
    fig_antiresonance()
    fig_dtd()
    fig_alpha()
    fig_sign()
    fig_lobes()


if __name__ == '__main__':
    main()
