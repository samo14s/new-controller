"""
G4 -- the damping budget of tonal FxLMS on a lightly damped structure.
=======================================================================
Observation in theory_probe.txt: at the optimum, the decay rates of the
adaptation pole (near w0) and of the adjacent structural pole add up to the
open-loop decay rate sigma_n = zeta_n omega_n of that mode.

First-order argument (baseband, modal-dominant secondary path
G(s) ~ R / (s - s_n), s_n = -sigma_n + i Delta, |Delta| >> sigma_n, Gc = G):
the adaptation integrator h' = -a G* e closes a 2nd-order loop whose
characteristic polynomial has trace -(sigma_n - i Delta) independent of a, so
    Re(s_adapt) + Re(s_mode) = -sigma_n          for every gain a,
and the loop can only REDISTRIBUTE the mode's damping:
    adaptation rate a|G|^2 -> sigma_n  means  mode rate -> 0  (divergence),
    fastest t5 when the two rates are comparable, t_min ~ c / sigma_n.
Pre-declared check: along beta (mu swept, L fixed) on the TRUTH loop,
|sum of the two decay rates / sigma_n - 1| <= 10 % wherever the loop is stable.
Writes results/budget.txt.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import Tee, identify, Testbed, FIGS
from control import Plants
import theory
import calib

log = Tee('budget.txt')
cal = calib.load()
T, M, _, _ = identify(Testbed(zeta=cal['zeta']))
fs, A = T.fs, 5.0
CASES = [('120 Hz / S2 / A2', 120.0, 1, [1], 3), ('280 Hz / S1 / A1', 280.0, 0, [0], 5),
         ('85 Hz / S2 / A1+A2', 85.0, 1, [0, 1], 2)]
worst = 0.0
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (label, f, s, acts, kmode) in zip(axes, CASES):
    rows = []
    w0 = 2 * np.pi * f / fs
    fn = T.P.freq[kmode]
    sig_n = T.zeta[kmode] * T.P.omega[kmode]
    gc = M.frf([f])[0][s, [1 + a for a in acts]]
    cols = [1 + a for a in acts]
    log(f"\n=== {label}: adjacent mode {kmode + 1} at {fn:.2f} Hz, sigma_n = {sig_n:.3f} 1/s,"
        f" detuning {f - fn:+.2f} Hz (Delta/sigma = {abs(2 * np.pi * (f - fn)) / sig_n:.1f}) ===")
    log("   beta/beta_div | rate adapt | rate mode | sum/sigma_n")
    # find divergence beta by bisection on spectral radius
    def poles(scale):
        betas = scale * np.abs(gc)
        Ah, Bh, Ch = theory.averaged_controller(betas, w0, np.angle(gc))
        Acl, _, _ = theory.closed_loop(T.Ad, T.Bd[:, cols], T.Cc[[T.iv[s]]], Ah, Bh, Ch)
        return np.linalg.eigvals(Acl)
    lo, hi = 1e-6, 1.0
    while np.max(np.abs(poles(hi))) <= 1 + 1e-9:
        hi *= 2
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if np.max(np.abs(poles(mid))) <= 1 + 1e-9 else (lo, mid)
    bdiv = lo
    for frac in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        lam = poles(frac * bdiv)
        fz = np.angle(lam) * fs / (2 * np.pi)
        rate = -np.log(np.abs(lam)) * fs
        # pole nearest w0 (adaptation) and nearest the structural mode, positive freq
        pos = fz > 0
        ia = np.argmin(np.where(pos, np.abs(fz - f) + 1e3 * (np.abs(fz - fn) < np.abs(fz - f)), np.inf))
        im = np.argmin(np.where(pos & (np.arange(lam.size) != ia), np.abs(fz - fn), np.inf))
        tot = rate[ia] + rate[im]
        if len(acts) == 2:
            # two actuators: one extra marginal resonator pole sits at w0 (null space);
            # take the slowest NON-marginal pole near w0 as the adaptation pole
            near = np.nonzero(pos & (np.abs(fz - f) < 1.0) & (rate > 1e-6))[0]
            if near.size:
                ia = near[np.argmin(np.abs(fz[near] - f))]
                tot = rate[ia] + rate[im]
        dev = abs(tot / sig_n - 1)
        worst = max(worst, dev)
        log(f"   {frac:13.2f} | {rate[ia]:10.3f} | {rate[im]:9.3f} | {tot / sig_n:10.3f}")
        rows.append((frac, rate[ia] / sig_n, rate[im] / sig_n, tot / sig_n))
    rows = np.array(rows)
    ax.plot(rows[:, 0], rows[:, 1], 'o-', label='adaptation pole')
    ax.plot(rows[:, 0], rows[:, 2], 's-', label=f'adjacent mode ({fn:.2f} Hz)')
    ax.plot(rows[:, 0], rows[:, 3], 'k--', label='sum')
    ax.axhline(1.0, color='0.6', lw=0.8)
    ax.set_xlabel(r'$\beta/\beta_{\rm div}$')
    ax.set_ylabel(r'decay rate / $\sigma_n$')
    ax.set_title(label, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle('Damping budget: the adaptation only redistributes the adjacent mode\'s damping', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_budget.png'), dpi=140)
log(f"\nG4: max |sum/sigma_n - 1| = {100 * worst:.1f} %  -> {'PASS' if worst <= 0.10 else 'FAIL'}")
