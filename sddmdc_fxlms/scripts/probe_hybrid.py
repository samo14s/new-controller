"""
Preview of paper 3 (docs/05 sec. 4) -- the hybrid law t_min'/t_min = sigma_n/sigma_n'.
=====================================================================================
A positive-position-feedback (PPF) loop from S2 to A2, tuned on mode 4
(117.07 Hz), adds damping to the mode adjacent to the 120 Hz disturbance;
tonal FxLMS then runs on the CLOSED-LOOP plant through the same actuator.
The damping budget predicts that the fastest convergence scales with the
closed-loop decay rate of that mode.

Pre-declared: for PPF gains giving sigma_4'/sigma_4 between 1 and ~5,
|(t_min / t_min') / (sigma_4' / sigma_4) - 1| <= 20 %.
FxLMS times from the averaged-LTI predictor (validated in docs/03, G2) with a
perfect compensation path (the closed-loop path itself), so only the plant
damping changes.  Writes results/hybrid_probe.txt.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

from _common import Tee, Testbed, FIGS
import theory
import calib

log = Tee('hybrid_probe.txt')
cal = calib.load()
T = Testbed(zeta=cal['zeta'])
fs = T.fs
f0, s_idx, a_idx, k_mode = 120.0, 1, 1, 3          # 120 Hz, S2, A2, mode 4
w0 = 2 * np.pi * f0 / fs
wn = T.P.omega[k_mode]
Ad, Bd, Cc = T.Ad, T.Bd, T.Cc
ys = T.iv[s_idx]


def ppf(gain, zf=0.3, wf=wn):
    from scipy.signal import tf2ss
    num = [gain * wf**2]
    den = [1.0, 2 * zf * wf, wf**2]
    Ak, Bk, Ck, Dk, _ = cont2discrete(tf2ss(num, den), 1 / fs, method='bilinear')
    return Ak, Bk, Ck, Dk


def closed(gain, sign):
    """Augmented discrete plant: inputs [E, A2_ff], output S2; PPF S2 -> A2."""
    n = Ad.shape[0]
    if gain == 0:
        return Ad, Bd[:, [0, 1 + a_idx]], Cc[[ys]]
    Ak, Bk, Ck, Dk = ppf(gain)
    nk = Ak.shape[0]
    b_a = Bd[:, [1 + a_idx]]
    c_s = Cc[[ys]]
    # u_fb(k) = sign * (Ck xk + Dk y(k)),  y(k) = c_s x(k)  (strictly proper plant)
    A = np.block([[Ad + sign * b_a @ Dk @ c_s, sign * b_a @ Ck],
                  [Bk @ c_s, Ak]])
    B = np.vstack([Bd[:, [0, 1 + a_idx]], np.zeros((nk, 2))])
    C = np.hstack([c_s, np.zeros((1, nk))])
    return A, B, C


def mode_rate(A):
    lam = np.linalg.eigvals(A)
    fz = np.angle(lam) * fs / (2 * np.pi)
    cand = np.nonzero((fz > 100) & (fz < 135))[0]
    i = cand[np.argmin(np.abs(np.abs(lam[cand]) - 1))]      # least damped near mode 4
    return -np.log(abs(lam[i])) * fs, fz[i], np.max(np.abs(lam))


def frf(A, B, C, f):
    z = np.exp(1j * 2 * np.pi * f / fs)
    return (C @ np.linalg.solve(z * np.eye(A.shape[0]) - A, B))[0]


def t_min_of(A, B, C):
    H = frf(A, B, C, f0)
    gp, gs = H[0], H[1]

    def run(beta):
        Ah, Bh, Ch = theory.averaged_controller(beta * abs(gs), w0, np.angle(gs))
        Acl, Bdd, Ccl = theory.closed_loop(A, B[:, [1]], C, Ah, Bh, Ch)
        return theory.lti_t5(Acl, Bdd, Ccl, gp * (-5j), w0, fs, f0)
    lo, hi = 1e-9, 1.0
    while run(hi)[1] <= 1 + 1e-9:
        hi *= 2
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if run(mid)[1] <= 1 + 1e-9 else (lo, mid)
    fr = np.linspace(0.05, 0.99, 50)
    ts = np.array([run(x * lo)[0] for x in fr])
    return ts.min(), abs(gs), abs(gp)


A0, B0, C0 = closed(0, 1)
sig0, f40, _ = mode_rate(A0)
# choose the sign that ADDS damping to mode 4
sign = max((+1, -1), key=lambda sg: mode_rate(closed(0.05, sg)[0])[0])
log(f"open loop: mode 4 at {f40:.2f} Hz, sigma_4 = {sig0:.3f} 1/s; PPF sign = {sign:+d} (adds damping)")
tm0, gs0, gp0 = t_min_of(A0, B0, C0)
log(f"open loop: t_min = {tm0:.2f} s, |Gs(120)| = {gs0:.4f}, |Gp(120)| = {gp0:.4f}")
log("  PPF gain | mode-4 freq | sigma_4'/sigma_4 | stable | t_min' [s] | t_min/t_min' | ratio/pred - 1")
worst = 0.0
pts = [(1.0, 1.0)]
for gain in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
    A, B, C = closed(gain, sign)
    sg, fz, rho = mode_rate(A)
    if rho >= 1:
        log(f"  {gain:8.2f} | closed loop unstable (rho = {rho:.5f})")
        continue
    tm, gs, gp = t_min_of(A, B, C)
    pred = sg / sig0
    meas = tm0 / tm
    dev = meas / pred - 1
    if 1.0 <= pred <= 5.5:
        worst = max(worst, abs(dev))
    pts.append((pred, meas))
    log(f"  {gain:8.2f} | {fz:9.2f}   | {pred:16.2f} | yes    | {tm:10.2f} | {meas:12.2f} | {100 * dev:+6.1f} %"
        f"   (|Gs| {gs:.4f}, |Gp| {gp:.4f})")
log(f"\nPre-declared check (sigma ratio 1..5, |dev| <= 20 %): max |dev| = {100 * worst:.1f} % -> "
    f"{'PASS' if worst <= 0.20 else 'FAIL'}")

pts = np.array(pts)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([1, 4], [1, 4], 'k--', label=r'law: $t_{\min}/t_{\min}\prime=\sigma_4\prime/\sigma_4$')
ax.fill_between([1, 4], [0.8, 3.2], [1.2, 4.8], color='0.9', label='pre-declared +-20 %')
ax.plot(pts[:, 0], pts[:, 1], 'o', color='C3', label='PPF (S2->A2) + FxLMS, testbed')
ax.set_xlabel(r'added damping  $\sigma_4\prime/\sigma_4$')
ax.set_ylabel(r'FxLMS speed-up  $t_{\min}/t_{\min}\prime$')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title('Hybrid design law, 120 Hz', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_hybrid.png'), dpi=140)
log("saved figures/fig_hybrid.png")
