"""What the 8.4 Hz proof-mass actuator mode does to the stability limit.

1. Ga(iw)/3 at the chatter band (magnitude, phase).
2. Closed-loop poles without cutting, and the gain at which the DVF loop
   itself goes unstable.
3. Low-frequency Re Gcl: a zero-order root needs Re Gcl < 0, so the actuator
   mode can only create its own (low-frequency) lobe family if Re Gcl < 0
   somewhere below the structural mode.  Threshold gain for that.
4. Brute-force count of ZOA roots below 60 Hz.
5. a_lim with the full Eq. (3) actuator vs a flat 3 N/V actuator, 5-rpm grid
   (lobe sweep, polished) -> actuator_effect.csv.
6. SDM: the multiplier of the actuator mode at the stability boundary.
Writes actuator_mode.json and actuator_effect.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import json
import numpy as np
from scipy.optimize import brentq
import model as M
import zoa_lobes as L
import sdm

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# 1 ----------------------------------------------------------------------
fs = np.array([8.4, 20, 50, 100, 129.3, 150, 200, 250])
Gr = M.Ga(2j * np.pi * fs) / M.KA
out["Ga_over_3"] = [dict(f_Hz=float(f), mag=float(abs(x)), phase_deg=float(np.degrees(np.angle(x))))
                    for f, x in zip(fs, Gr)]

# 2 ----------------------------------------------------------------------
poles = {}
for z in (0.0134, 0.0145):
    for g in (0, 300, 800, 1500):
        ev = np.linalg.eigvals(M.A_closed(g, z))
        ev = ev[np.argsort(-np.abs(ev.imag))]
        poles[f"z{z}_g{g}"] = [dict(re=float(e.real), im=float(e.imag), f_Hz=float(abs(e) / 2 / np.pi),
                                    zeta=float(-e.real / abs(e))) for e in ev[::2]]
out["closed_loop_poles"] = poles


def max_re(g, z):
    return np.max(np.linalg.eigvals(M.A_closed(g, z)).real)


gs = np.logspace(2, 6, 400)
mr = np.array([max_re(g, 0.0134) for g in gs])
j = np.flatnonzero(mr > 0)
g_unst = brentq(lambda g: max_re(g, 0.0134), gs[j[0] - 1], gs[j[0]]) if j.size else np.inf
out["dvf_loop_unstable_above_g"] = float(g_unst)

# 3 ----------------------------------------------------------------------
# Re(1/Gcl) = K - M w^2 - g w Im Ga(iw) and Im Ga > 0, so Re Gcl < 0 first
# appears where (K - M w^2) / (w Im Ga) is smallest.  Restricted to the
# actuator band (< 60 Hz); near the structural mode the threshold is trivially
# small because K - M w^2 -> 0 there (that is the ordinary structural lobe).
w = 2 * np.pi * np.linspace(0.01, 60.0, 200000)
imGa = M.Ga(1j * w).imag
gstar = (M.KS - M.MS * w**2) / (w * imGa)
i = np.argmin(gstar)
out["ReGcl_negative_below_60Hz_needs_g_above"] = dict(g=float(gstar[i]), at_f_Hz=float(w[i] / 2 / np.pi))
lf = 2 * np.pi * np.linspace(0.05, 60, 60000)
out["min_ReGcl_times_k_below_60Hz"] = {
    str(g): float(np.min(M.Gcl(1j * lf, g, 0.0134).real) * M.KS) for g in (0, 300, 800, 1500, 3000, 6000)}

# 4 ----------------------------------------------------------------------
def roots_below(n, g, z, fhi=60.0):
    f = np.arange(0.05, fhi, 1e-3)
    ww = 2 * np.pi * f
    T = 60.0 / (M.NT * n)
    a = 1 / (L.C_CUT * (1 - np.exp(-1j * ww * T)) * M.Gcl(1j * ww, g, z))
    im = a.imag
    idx = np.flatnonzero(np.sign(im[:-1]) * np.sign(im[1:]) < 0)
    k = np.floor(ww * T / (2 * np.pi))
    idx = idx[k[idx] == k[idx + 1]]                 # drop tooth-passing poles
    return int(np.sum(a.real[idx] > 0))


cnt = {}
for g in (0, 300, 800, 1500):
    cnt[str(g)] = int(sum(roots_below(n, g, 0.0134) for n in np.arange(500, 3001, 50)))
out["n_positive_zoa_roots_below_60Hz_over_51_speeds"] = cnt

# 5 ----------------------------------------------------------------------
rp = np.arange(500, 3001, 5.0)
rows = []
summary5 = {}
for z in (0.0134, 0.0145):
    for g in (300, 800, 1500):
        af, ff, _ = L.envelope(rp, g, z)
        ai, fi, _ = L.envelope(rp, g, z, ideal=True)
        rel = af / ai - 1
        summary5[f"z{z}_g{g}"] = dict(min_rel=float(rel.min()), rpm_min=float(rp[np.argmin(rel)]),
                                      max_rel=float(rel.max()), rpm_max=float(rp[np.argmax(rel)]),
                                      mean_abs_rel=float(np.mean(np.abs(rel))))
        for n, x, y, r, f1, f2 in zip(rp, af, ai, rel, ff, fi):
            rows.append(dict(zeta=z, g=g, rpm=n, a_full_mm=f"{x*1e3:.6f}", a_ideal_mm=f"{y*1e3:.6f}",
                             rel=f"{r:.4e}", fc_full_Hz=f"{f1:.3f}", fc_ideal_Hz=f"{f2:.3f}"))
out["zoa_full_vs_ideal_actuator_5rpm"] = summary5
with open(os.path.join(HERE, "actuator_effect.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

# 6 ----------------------------------------------------------------------
act = []
# boundary of the SDM with the time-averaged force (identical model to the
# ZOA), so the largest multiplier sits on the unit circle
for n, g in ((500, 1500), (1300, 1500), (3000, 1500), (3000, 300)):
    a = sdm.boundary(n, g, 0.0134, helix="mean", rtol=1e-7)[0]
    T = 60.0 / (M.NT * n)
    ev = np.linalg.eigvals(M.A_closed(g, 0.0134))
    lam_a = ev[np.argmin(np.abs(ev))]                       # actuator pole
    mu0 = np.exp(lam_a * T)
    Phi = sdm.monodromy(n, a, g, 0.0134, helix="mean")
    mus = np.linalg.eigvals(Phi)
    k = np.argmin(np.abs(mus - mu0))
    act.append(dict(rpm=n, g=g, a_mm=float(a * 1e3), mu_act_no_cut=float(abs(mu0)),
                    mu_act_at_alim=float(abs(mus[k])), mu_max_at_alim=float(np.max(np.abs(mus)))))
out["sdm_actuator_multiplier"] = act

with open(os.path.join(HERE, "actuator_mode.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("closed_loop_poles",)}, indent=1))
print(json.dumps(out["closed_loop_poles"]["z0.0134_g1500"], indent=1))
