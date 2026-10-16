"""
M1 preview -- closed-form edges of the effective-convergence region.
====================================================================
Baseband two-pole model (theory.py, G4): adjacent mode s_n = -sigma + i Delta,
perfect compensation path, adaptation rate g = a |G(w0)|^2 [1/s].  With
x(0) = h(0) = 0 the error is

    e(t)/D = A_ad e^{lam_ad t} + A_m e^{lam_m t},
    lam_ad ~ -g,  lam_m ~ -(sigma - g)  (trace conservation, G4),
    A_ad ~ 1,     |A_m| ~ g / |Delta|,

so the 5 % time is
    t5(g) = max( ln20 / g ,  ln(20 g/|Delta|) / (sigma - g) ),   g < sigma.

Hence, for a threshold T (5 s in the paper):
    lower edge  g_lo = ln20 / T                        (slow branch)
    upper edge  ln(20 g_up/|Delta|) = T (sigma - g_up)  (mode ringing)
    rho = g_up / g_lo = (mu L)_up / (mu L)_lo

rho depends ONLY on (sigma_n, Delta, T): not on gains, mu, L, A, the
step-size convention, or mu_scale.  So (i) it is predicted from the
identified model's modal data alone, and (ii) a published map can be
INVERTED to estimate the damping of the adjacent mode.
Pre-declared check: on the testbed (known zeta), |rho_formula / rho_testbed - 1|
<= 10 % for 120 and 280 Hz (results/map_probe.txt).
Writes results/edges.txt.
"""
import json
import os

import numpy as np
from scipy.optimize import brentq

from _common import Tee, ROOT
from plate import F_EXP
import calib

log = Tee('edges.txt')
LN20, T = np.log(20.0), 5.0
cal = calib.load()
ZETA = cal['zeta']


def t5(g, sig, dlt):
    a = LN20 / g
    b = np.log(20 * g / dlt) / (sig - g) if (20 * g > dlt and g < sig) else 0.0
    return np.inf if g >= sig else max(a, b)


def g_up(sig, dlt, T=T):
    f = lambda g: np.log(20 * g / dlt) - T * (sig - g)
    lo = max(dlt / 20 * (1 + 1e-9), 1e-9)
    if f(lo) > 0:          # mode term never binds before sigma -> edge at g where ln20/g ... (degenerate)
        return lo
    return brentq(f, lo, sig * (1 - 1e-12))


def rho(sig, dlt, T=T):
    return g_up(sig, dlt, T) / (LN20 / T)


def t_min(sig, dlt):
    gs = np.linspace(1e-3, sig * 0.999, 20000)
    v = np.array([t5(g, sig, dlt) for g in gs])
    return v.min(), gs[v.argmin()] / sig


CASES = [('120 Hz (mode 4)', 120.0, 3), ('280 Hz (mode 6)', 280.0, 5), ('85 Hz (mode 3)', 85.0, 2)]
# testbed edges from results/map_probe.txt (full FxLMS, truth plant)
TESTBED = {120.0: (510.0, 1474.0), 280.0: (221.0, 1452.0)}
# published maps (data/digitized/maps_muL_analysis.txt): lower, upper mu*L (x1e-6), exp / sim
PAPER = {120.0: dict(exp=(497, 1303), sim=(516, 1342)), 280.0: dict(exp=(246, 2002), sim=(323, 2078)),
         85.0: dict(exp=(675, 1115), sim=(604, 935))}
# published t_min (docs/01 H4)
TMIN = {120.0: (2.90, 2.52), 280.0: (1.19, 1.07), 85.0: (3.80, 3.75)}

log(f"T = {T} s, ln20 = {LN20:.3f}; testbed zeta = {ZETA:.4f}")
ok = True
for label, f, k in CASES:
    fn = F_EXP[k]
    wn = 2 * np.pi * fn
    dlt = abs(2 * np.pi * (f - fn))
    sig = ZETA * wn
    r = rho(sig, dlt)
    tm, xs = t_min(sig, dlt)
    log(f"\n=== {label}: f_n = {fn} Hz, |Delta| = {dlt:.2f} rad/s ===")
    log(f"  testbed zeta {ZETA:.4f}: sigma = {sig:.3f} 1/s, Delta/sigma = {dlt / sig:.1f}")
    log(f"  formula: rho = (muL)_up/(muL)_lo = {r:.2f};  t_min = {tm:.2f} s at g/sigma = {xs:.2f}")
    if f in TESTBED:
        lo, up = TESTBED[f]
        rt = up / lo
        dev = r / rt - 1
        ok &= abs(dev) <= 0.10
        log(f"  testbed (full FxLMS): rho = {up:.0f}/{lo:.0f} = {rt:.2f}  -> formula error {100 * dev:+.1f} %")
    for kind in ('exp', 'sim'):
        lo, up = PAPER[f][kind]
        rp = up / lo
        try:
            sig_hat = brentq(lambda s: rho(s, dlt) - rp, dlt / 200, dlt * 2)
            z_hat = sig_hat / wn
            tm_hat, _ = t_min(sig_hat, dlt)
            log(f"  paper {kind}: rho = {up}/{lo} = {rp:.2f}  -> inverted zeta_{k + 1} = {100 * z_hat:.3f} %"
                f" (sigma = {sig_hat:.2f} 1/s); formula t_min = {tm_hat:.2f} s vs published "
                f"{TMIN[f][0 if kind == 'exp' else 1]:.2f} s")
        except ValueError:
            log(f"  paper {kind}: rho = {rp:.2f} -> no solution in range")
log(f"\nPre-declared check (testbed, 120 and 280 Hz, |error| <= 10 %): {'PASS' if ok else 'FAIL'}")
log("Independent damping estimates for comparison: zeta_3 ~ 0.165 % from the beat decay of Fig. 14(a)"
    " (docs/01 H4); zeta = 0.24 % fitted on Fig. 16(c) (results/calibration.txt).")
