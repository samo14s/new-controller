"""
The normalised design chart suggested by the prior-art check (docs/04 sec. 1):
t5 * sigma_n against g / sigma_n, where g is the quasi-static adaptation rate
(slow-branch rate, proportional to mu L A^2 |Gc||Gs| cos dphi) and sigma_n the
decay rate of the adjacent mode.  In these units every case should fall on a
family of curves indexed only by |Delta| / sigma_n.

Curves: closed-form two-pole model (analysis_edges.py) for the three
detunings.  Points: (i) the testbed (averaged-LTI loop on the truth plant,
three damping ratios); (ii) the paper's EXPERIMENTAL mu-sweeps (Figs. 16c,
19c, 21c, pixel data) with g = ln20 * muL / C (C = slow-branch constant fitted
by data/digitize_curves.py) and sigma_n from the inversion of the paper's own
maps (results/edges.txt) -- no testbed parameter enters the paper points.
Writes figures/fig_design_chart.png and results/design_chart.txt.
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import Tee, Testbed, FIGS, DIGIT
import theory

log = Tee('design_chart.txt')
LN20 = np.log(20.0)
F_N = {120.0: 117.07, 280.0: 276.07, 85.0: 86.96}
# zeta of the adjacent mode inverted from the paper's experimental maps (results/edges.txt)
ZETA_PAPER = {120.0: 0.00228, 280.0: 0.00297, 85.0: 0.00199}
FIGMU = {120.0: ('16c', 128), 280.0: ('19c', 128), 85.0: ('21c', 64)}


def t5_two_pole(x, d):
    """t5*sigma as a function of x = g/sigma and d = |Delta|/sigma."""
    a = LN20 / x
    b = np.log(20 * x / d) / (1 - x) if (20 * x > d and x < 1) else 0.0
    return np.inf if x >= 1 else max(a, b)


with open(os.path.join(DIGIT, 'analysis.json')) as fh:
    ana = json.load(fh)


def slow_constant(fig, L):
    """C = t * mu * L on the slow branch (median of t*mu over the slow points
    listed by data/digitize_curves.py in analysis.json)."""
    tx = ana[f'fig{fig}']['series']['Experiment']['t_times_x']
    return float(np.median(tx)) * L


fig, ax = plt.subplots(figsize=(8.5, 6))
xs = np.linspace(0.01, 0.995, 400)
cols = {120.0: 'C0', 280.0: 'C3', 85.0: 'C2'}
for f, col in cols.items():
    d = 2 * np.pi * abs(f - F_N[f]) / (ZETA_PAPER[f] * 2 * np.pi * F_N[f])
    ax.plot(xs, [t5_two_pole(x, d) for x in xs], '-', color=col, lw=1.2,
            label=f'two-pole formula, {f:g} Hz (|Delta|/sigma = {d:.1f})')

# paper experimental mu-sweeps
log("paper experimental points (pixel data), sigma_n from the map inversion:")
for f, col in cols.items():
    fg, L = FIGMU[f]
    C = slow_constant(fg, L)
    sig = ZETA_PAPER[f] * 2 * np.pi * F_N[f]
    pts = []
    with open(os.path.join(DIGIT, f'fig{fg}_best.csv')) as fh:
        for r in csv.DictReader(fh):
            if r['series'] == 'Experiment':
                mu, t = float(r['x_nominal']), float(r['y'])
                if t < 19.5:
                    pts.append((LN20 * mu * L / C / sig, t * sig))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], 'o', mfc='none', color=col, ms=6,
            label=f'paper experiment, Fig. {fg}')
    log(f"  {f:g} Hz: C = {C:.3e} s, sigma = {sig:.3f} 1/s, {len(pts)} points,"
        f" x range {pts[:, 0].min():.2f}..{pts[:, 0].max():.2f}, min t5*sigma = {pts[:, 1].min():.2f}")

# testbed, averaged LTI on the truth plant, three damping ratios
log("testbed (truth plant, averaged LTI):")
for zeta, mk in ((0.0017, 's'), (0.0024, '^'), (0.0035, 'v')):
    T = Testbed(zeta=zeta)
    fs = T.fs
    for f, s, a, k in ((120.0, 1, 1, 3), (280.0, 0, 0, 5)):
        w0 = 2 * np.pi * f / fs
        H = T.frf([f], out_rows=T.iv)[0]
        g, gp = H[s, 1 + a], H[s, 0]
        sig = zeta * T.P.omega[k]
        Ag, Bg, Cg = T.Ad, T.Bd[:, [1 + a]], T.Cc[[T.iv[s]]]

        def run(beta):
            Ah, Bh, Ch = theory.averaged_controller(beta * abs(g), w0, np.angle(g))
            Acl, Bd, Ccl = theory.closed_loop(Ag, Bg, Cg, Ah, Bh, Ch)
            return theory.lti_t5(Acl, Bd, Ccl, gp * (-5j), w0, fs, f)
        lo, hi = 1e-8, 1.0
        while run(hi)[1] <= 1 + 1e-9:
            hi *= 2
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if run(mid)[1] <= 1 + 1e-9 else (lo, mid)
        # quasi-static adaptation rate per unit beta: decay rate of the
        # resonator pole (nearest e^{i w0}) at a small gain
        b0 = 0.02 * lo
        Ah, Bh, Ch = theory.averaged_controller(b0 * abs(g), w0, np.angle(g))
        lam = np.linalg.eigvals(theory.closed_loop(Ag, Bg, Cg, Ah, Bh, Ch)[0])
        ia = np.argmin(np.abs(lam - np.exp(1j * w0)))
        rate_per_beta = -np.log(abs(lam[ia])) * fs / b0
        bet = np.linspace(0.08, 0.99, 30) * lo
        t = np.array([run(b)[0] for b in bet])
        ok = t < 19.5
        ax.plot(rate_per_beta * bet[ok] / sig, t[ok] * sig, mk, color=cols[f], ms=4, alpha=0.6,
                label=f'testbed {f:g} Hz, zeta = {100 * zeta:.2f} %' if True else None)
        log(f"  {f:g} Hz zeta {100 * zeta:.2f} %: divergence at g/sigma = {rate_per_beta * lo / sig:.3f}")

ax.set_xlabel(r'normalised adaptation rate  $g/\sigma_n$')
ax.set_ylabel(r'normalised convergence time  $t_5\,\sigma_n$')
ax.set_ylim(0, 25)
ax.set_xlim(0, 1.1)
ax.axvline(1.0, color='k', lw=0.8, ls='--')
ax.text(1.005, 22, 'damping budget\nexhausted', fontsize=8)
ax.grid(alpha=0.3)
ax.legend(fontsize=6.5, ncol=2, loc='upper center')
ax.set_title('One chart for every case: tonal FxLMS on a lightly damped plate', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_design_chart.png'), dpi=140)
log("saved figures/fig_design_chart.png")
