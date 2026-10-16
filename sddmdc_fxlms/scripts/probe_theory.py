"""
M0 gate of research line S1 -- is the convergence map predictable?
==================================================================
Three pre-declared checks on the calibrated virtual testbed:

  G1  exactness: the two-weight reduction (theory.reduced_fxlms) reproduces the
      full L-tap FxLMS to round-off.             pass: max |e_full - e_2w| < 1e-9 V
  G2  the averaged-LTI predictor (beta = mu L A^2 |Gc|) reproduces the full
      FxLMS convergence time along mu (L = 128) for the three cases.
                                                  pass: median error <= 10 %
  G3  it locates the divergence boundary: first mu where the full FxLMS stays
      above 5 % for 20 s vs first mu where the LTI spectral radius exceeds 1.
                                                  pass: within one grid step
Plus a diagnosis of WHAT limits the fastest convergence (closed-loop modes
dominating e(t) at t5) and the cost of a full map by both routes.
Writes results/theory_probe.txt.
"""
import time

import numpy as np

from _common import Tee, identify, Testbed
from control import Plants, run_case, lti_t5
import sim
import theory
import calib

log = Tee('theory_probe.txt')
cal = calib.load()
ZETA, KAPPA = cal['zeta'], cal['mu_scale']
T, M, _, _ = identify(Testbed(zeta=ZETA))
P = Plants(T, M)
fs, A = T.fs, 5.0
log(f"testbed zeta = {ZETA:.5f}, mu_scale = {KAPPA:.4f}; ROM r = {M.r}")

# ---------------------------------------------------------------- G1
log("\n=== G1: exact two-weight reduction ===")
worst = 0.0
for f, s, a, mu, L in [(120.0, 1, 1, 2e-6, 100), (280.0, 0, 0, 3e-6, 37), (85.0, 1, 1, 1e-6, 16)]:
    w = 2 * np.pi * f / fs
    gc = M.frf([f])[0][s, 1 + a]
    lam_p, Bz_p, Cz_p = P.plant('rom', [s], [a])
    ua = np.zeros(2, complex)
    ua[0] = -1j * A
    Zp, Zm = sim.steady_state_modal(lam_p, Bz_p, ua, w)
    k_on, ns = int(fs), int(fs) + int(21 * fs)
    G, _ = theory.gram(L, w)
    e2 = theory.reduced_fxlms(lam_p, Bz_p, Cz_p, Zp + Zm, A, w, -np.pi / 2, abs(gc), np.angle(gc),
                              G, mu, np.zeros(ns), k_on, ns)
    ef = run_case(P, 'rom', f, s, a, mu, L, return_signals=True)['e'][:, 0]
    d = np.abs(ef - e2).max()
    worst = max(worst, d)
    log(f"  {f:g} Hz, L = {L:3d}, mu = {mu:.0e}: max|e_full - e_2w| = {d:.2e} V (max|e| = {np.abs(ef).max():.3f})")
log(f"  G1: {'PASS' if worst < 1e-9 else 'FAIL'}")

# ---------------------------------------------------------------- G2, G3
CASES = [('120 Hz / S2 / A2', 120.0, 1, [1], 128), ('280 Hz / S1 / A1', 280.0, 0, [0], 128),
         ('85 Hz / S2 / A1+A2', 85.0, 1, [0, 1], 64)]
mus_p = np.r_[np.arange(0.5, 10, 0.5), np.arange(10, 26, 1.0)] * 1e-6
g2 = []
for label, f, s, acts, L in CASES:
    log(f"\n=== G2/G3: {label}, L = {L}, mu swept (paper units) ===")
    t0 = time.time()
    tf = np.array([run_case(P, 'truth', f, s, acts, KAPPA * m, L)['t_c'] for m in mus_p])
    t_full = time.time() - t0
    t0 = time.time()
    res = [lti_t5(P, 'truth', f, s, acts, KAPPA * m, L) for m in mus_p]
    t_lti = time.time() - t0
    tl = np.array([r[0] for r in res])
    rho = np.array([r[1] for r in res])
    ok = (tf < 19.5) & (tl < 19.5)
    err = np.abs(tl[ok] / tf[ok] - 1)
    g2.append(np.median(err))
    log("  mu_paper : " + " ".join(f"{m * 1e6:5.1f}" for m in mus_p))
    log("  full     : " + " ".join(f"{t:5.2f}" for t in tf))
    log("  LTI      : " + " ".join(f"{t:5.2f}" for t in tl))
    log(f"  median |LTI/full - 1| = {100 * np.median(err):.1f} %, max = {100 * err.max():.1f} %"
        f" over {ok.sum()} converging points")
    mask = (np.arange(tf.size) > np.argmin(tf)) & (tf >= 19.5)
    i_full = int(np.argmax(mask)) if mask.any() else -1
    i_lti = int(np.argmax(rho > 1 + 1e-9)) if (rho > 1 + 1e-9).any() else -1
    log(f"  divergence: full FxLMS first non-convergent mu = {mus_p[i_full] * 1e6 if i_full >= 0 else float('nan'):.1f}e-6, "
        f"LTI spectral radius > 1 from mu = {mus_p[i_lti] * 1e6 if i_lti >= 0 else float('nan'):.1f}e-6"
        f"  -> G3 {'PASS' if i_lti >= 0 and i_full >= 0 and abs(i_lti - i_full) <= 1 else 'FAIL'}")
    log(f"  cost: full sweep {t_full:.1f} s, LTI sweep {t_lti:.1f} s")
    # what limits the optimum?
    k = int(np.argmin(tf))
    gc = M.frf([f])[0]
    cols = [1 + a for a in acts]
    betas = KAPPA * mus_p[k] * L * A**2 * np.abs(gc[s, cols])
    Ah, Bh, Ch = theory.averaged_controller(betas, 2 * np.pi * f / fs, np.angle(gc[s, cols]))
    Acl, Bd, Ccl = theory.closed_loop(T.Ad, T.Bd[:, cols], T.Cc[[T.iv[s]]], Ah, Bh, Ch)
    lam, V = np.linalg.eig(Acl)
    D = T.frf([f], out_rows=T.iv)[0][s, 0] * (-1j * A)
    Xp = np.linalg.solve(np.exp(1j * 2 * np.pi * f / fs) * np.eye(len(lam)) - Acl, Bd[:, 0] * D / 2)
    amp = (Ccl[0] @ V) * np.linalg.solve(V, -2 * Xp.real)
    kk = int(tf[k] * fs)
    c = np.abs(amp) * np.abs(lam)**kk
    top = [i for i in np.argsort(c)[::-1] if lam[i].imag >= 0][:2]
    fz = np.abs(np.log(lam[top].astype(complex))) * fs / (2 * np.pi)
    sig = -np.log(np.abs(lam[top])) * fs
    log(f"  at the optimum (mu = {mus_p[k] * 1e6:.1f}e-6, t5 = {tf[k]:.2f} s) e(t5) is carried by "
        f"closed-loop poles at {np.round(fz, 2)} Hz with decay rates {np.round(sig, 2)} 1/s;"
        f" open-loop structural modes: {np.round(T.P.freq[:6], 2)} Hz, sigma_n = zeta*omega_n = "
        f"{np.round(ZETA * T.P.omega[:6], 2)} 1/s")

log(f"\nG2: median errors {np.round(100 * np.array(g2), 1)} % -> "
    f"{'PASS' if max(g2) <= 0.10 else 'FAIL'}")
