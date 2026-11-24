"""Global attractivity of the forced orbit of the clipped DVF loop below the
circle-criterion depth (second external review, point 5: the claim that a cut
shallower than the uncontrolled limit stays stable with the command saturated).

Model.  The averaged (zero-order) linear-force delay model of the cut,
    m x'' + c x' + k x = kappa a [x(t-T) - x(t)] + Ga(D) V + forcing,
with the clipped DVF law V = sat_Vmax(-g x').  On the forced orbit x_p the
command is b(t) = -g x_p'(t); if |b(t)| <= Vmax (the orbit is not clipped),
the deviation v = x' - x_p' sees the command deviation
    V - b = -phi(t, g v),   phi(t, y) = b(t) - sat(b(t) - y),
and, sat being monotone and 1-Lipschitz, 0 <= phi(t, y) y <= y^2: a
time-varying nonlinearity in the sector [0, 1].  The linear part from the
command deviation to g v is
    L(s) = g s Ga(s) G(s) / (1 - kappa a (1 - e^{-sT}) G(s)),
the uncontrolled cut with the actuator, which is stable iff a < a_lim(n, 0)
(the zero-order limit is exact for this averaged model).  The circle criterion
for the sector [0, 1] (Zames 1966; Desoer and Vidyasagar 1975 for systems with
delay) then gives finite-gain L2 stability of the deviation, and with it the
decay of every deviation from the forced orbit, if
    mu(a) = inf_w Re[1 + L(jw)] > 0.
a_circ(n, g) is the largest depth below a_lim(n, 0) at which this holds.

Protocol.  mu is minimised on a frequency grid (0.1-60 Hz by 0.01 Hz,
60-600 Hz by 0.002 Hz, 600-5000 Hz by 0.5 Hz) and refined by three nested
zooms (2001 points each) around the 8 lowest local minima of Re[1 + L] and
the 8 highest local maxima of |L|, since near a_lim(0) the dip is as narrow
as the decay rate of the critical pole.  a_circ: 40 depths in (0, a_lim(0)),
then bisection (relative tolerance 1e-6) between the last depth that passes
and the first that fails; a pass after a fail is reported (non-monotone).
Done for every speed of the 251-speed grid, at the gain g_best of
results/feas_g1500.npz and at g = 1500 V s/m, and at the four basin speeds
for g in {250, 500, 1000, 1500, 2000, 3000}.  The closed-form forced
command must stay below Vmax at every depth up to a_circ (checked on 400
depths; where it does not, the orbit itself is clipped and the result does
not apply).  The band edge of
results/c1_band.npz is compared.

    OMP_NUM_THREADS=1 python3 verification/paper/circle_criterion.py          # full run
    OMP_NUM_THREADS=1 python3 verification/paper/circle_criterion.py orbit    # only the orbit limit

writes verification/paper/circle_criterion.json and .npz (about 15 minutes on a
loaded machine).  a_circ_eff = min(a_circ, the first depth at which the forced
orbit is clipped) is the depth up to which Proposition 1 applies.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")

import params as P  # noqa: E402

P.set_damping(float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"]))

import c4_robust as R  # noqa: E402
import feasibility as F  # noqa: E402
import zoa  # noqa: E402
from structure import Ga  # noqa: E402

F_HZ = np.unique(np.concatenate([np.arange(0.1, 60.0, 0.01), np.arange(60.0, 600.0, 0.002),
                                 np.arange(600.0, 5000.0 + 0.25, 0.5)]))
W = 2 * np.pi * F_HZ
S = 1j * W
GS = R.G_struct(S)
GA = Ga(S)
BASIN_RPM = (1090.0, 1420.0, 2500.0, 2870.0)
G_SWEEP = (250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0)


def L_of(w, rpm, a, g):
    s = 1j * np.asarray(w, dtype=float)
    T = 60.0 / (rpm * P.N_TEETH)
    Gs = R.G_struct(s)
    return g * s * Ga(s) * Gs / (1.0 - Gs * R.KAPPA * a * (1.0 - np.exp(-s * T)))


def _zoom(rpm, a, g, lo, hi, levels=3, n=2001):
    best_w, best_v = None, np.inf
    for _ in range(levels):
        w = np.linspace(lo, hi, n)
        v = np.real(1.0 + L_of(w, rpm, a, g))
        k = int(np.argmin(v))
        if v[k] < best_v:
            best_v, best_w = float(v[k]), float(w[k])
        dw = w[1] - w[0]
        lo, hi = w[k] - 2 * dw, w[k] + 2 * dw
    return best_v, best_w


def mu(rpm, a, g, refine=True):
    """inf_w Re[1 + L(jw)] and the frequency (Hz) where it is reached."""
    T = 60.0 / (rpm * P.N_TEETH)
    L = g * S * GA * GS / (1.0 - GS * R.KAPPA * a * (1.0 - np.exp(-S * T)))
    re = np.real(1.0 + L)
    k0 = int(np.argmin(re))
    best_v, best_w = float(re[k0]), float(W[k0])
    if not refine:
        return best_v, best_w / (2 * np.pi)
    # candidates: lowest local minima of Re(1 + L) and highest local maxima of |L|
    absL = np.abs(L)
    i = np.arange(1, len(W) - 1)
    mins = i[(re[i] <= re[i - 1]) & (re[i] <= re[i + 1])]
    maxs = i[(absL[i] >= absL[i - 1]) & (absL[i] >= absL[i + 1])]
    cand = set(mins[np.argsort(re[mins])[:8]].tolist()) | set(maxs[np.argsort(-absL[maxs])[:8]].tolist())
    for k in cand:
        v, w = _zoom(rpm, a, g, W[k - 1], W[k + 1])
        if v < best_v:
            best_v, best_w = v, w
    return best_v, best_w / (2 * np.pi)


def a_circ(rpm, g, n_scan=40, rtol=1e-6):
    a0 = zoa.a_lim(rpm, 0.0)
    depths = a0 * np.arange(1, n_scan) / n_scan
    ok = np.array([mu(rpm, a, g)[0] > 0 for a in depths])
    nonmono = bool(np.any(~ok[:-1] & ok[1:]))
    if ok.all():
        lo, hi = depths[-1], a0 * (1 - 1e-9)
        if mu(rpm, hi, g)[0] > 0:
            return dict(a0=a0, a_circ=hi, nonmonotone=nonmono, reached_a0=True)
    else:
        j = int(np.argmin(ok))           # first failing depth
        if j == 0:
            lo, hi = 0.0, depths[0]
        else:
            lo, hi = depths[j - 1], depths[j]
    while hi - lo > rtol * hi:
        m = 0.5 * (lo + hi)
        if mu(rpm, m, g)[0] > 0:
            lo = m
        else:
            hi = m
    return dict(a0=a0, a_circ=lo, nonmonotone=nonmono, reached_a0=False)


def orbit_peak_V(rpm, g, a):
    return float(F.forced_demand(rpm, g, np.array([a]))[0] / P.GA_GAIN)


def orbit_peak_V_up_to(rpm, g, a, n=400):
    """Largest peak forced command over n depths in (0, a]: the sector [0, 1]
    needs the forced orbit unclipped at every depth up to a_circ."""
    d = a * np.arange(1, n + 1) / n
    return float(F.forced_demand(rpm, g, d).max() / P.GA_GAIN)


def first_clipped_depth(rpm, g, a_top, n=2000):
    """Smallest of n depths in (0, a_top] at which the peak forced command exceeds
    Vmax (the orbit itself is clipped there), or None."""
    d = a_top * np.arange(1, n + 1) / n
    v = F.forced_demand(rpm, g, d) / P.GA_GAIN
    k = np.nonzero(v > R.VMAX)[0]
    return None if k.size == 0 else float(d[k[0]])


def add_orbit_limit(out):
    """a_circ_eff = min(a_circ, first depth at which the forced orbit is clipped):
    Proposition 1 needs |b(t)| <= Vmax at the depth considered."""
    rpm = np.array(out["rpm"])
    for label in ("g_best", "g1500"):
        rows = out[label]["rows"]
        for q in rows:
            a_c = q["a_circ_mm"] * 1e-3
            a_k = first_clipped_depth(q["rpm"], q["g"], a_c)
            q["first_clipped_depth_mm"] = None if a_k is None else a_k * 1e3
            q["a_circ_eff_mm"] = q["a_circ_mm"] if a_k is None else min(q["a_circ_mm"], a_k * 1e3)
            q["ratio_eff"] = q["a_circ_eff_mm"] / q["a0_mm"]
        r = np.array([q["ratio_eff"] for q in rows])
        lim = [q for q in rows if q["first_clipped_depth_mm"] is not None]
        out[label].update(ratio_eff_min=float(r.min()), ratio_eff_median=float(np.median(r)),
                          ratio_eff_max=float(r.max()), rpm_ratio_eff_min=float(rpm[int(np.argmin(r))]),
                          n_ratio_eff_below_0p9=int(np.sum(r < 0.9)),
                          n_orbit_limited=len(lim), rpm_orbit_limited=[q["rpm"] for q in lim],
                          orbit_limited_ratio_eff=[float(min(q["ratio_eff"] for q in lim)),
                                                   float(np.median([q["ratio_eff"] for q in lim])),
                                                   float(max(q["ratio_eff"] for q in lim))] if lim else None)
        print(label, "with the orbit unclipped: a_circ_eff / a_lim(0) min %.4f (%.0f rpm), median %.4f; orbit-limited at %d speeds" % (
            r.min(), rpm[int(np.argmin(r))], np.median(r), len(lim)), flush=True)
    return out


def main():
    if sys.argv[1:] == ["orbit"]:            # post-process a stored run
        path = os.path.join(HERE, "circle_criterion.json")
        with open(path) as f:
            out = json.load(f)
        out = add_orbit_limit(out)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        return
    t0 = time.time()
    fg = np.load(os.path.join(RES, "feas_g1500.npz"))
    rpm, g_best = fg["rpm"], fg["g_best"]
    band = np.load(os.path.join(RES, "c1_band.npz"))
    out = dict(rpm=rpm.tolist())
    arrays = {}
    for label, gains in (("g_best", g_best), ("g1500", np.full_like(rpm, 1500.0))):
        rows = []
        for n, g in zip(rpm, gains):
            r = a_circ(float(n), float(g))
            m_half, f_half = mu(float(n), 0.5 * r["a0"], float(g))
            m_c, f_c = mu(float(n), r["a_circ"], float(g))
            m_fail, f_fail = mu(float(n), min(r["a_circ"] * (1 + 1e-4), r["a0"] * (1 - 1e-9)), float(g))
            rows.append(dict(rpm=float(n), g=float(g), a0_mm=r["a0"] * 1e3, a_circ_mm=r["a_circ"] * 1e3,
                             ratio=r["a_circ"] / r["a0"], nonmonotone=r["nonmonotone"], reached_a0=r["reached_a0"],
                             mu_at_half_a0=m_half, f_mu_half_hz=f_half, mu_at_a_circ=m_c,
                             f_fail_hz=f_fail, mu_just_above=m_fail,
                             orbit_peak_V_at_a_circ=orbit_peak_V(float(n), float(g), r["a_circ"]),
                             orbit_peak_V_up_to_a_circ=orbit_peak_V_up_to(float(n), float(g), r["a_circ"])))
        ratio = np.array([q["ratio"] for q in rows])
        arrays[label + "_a_circ_mm"] = np.array([q["a_circ_mm"] for q in rows])
        arrays[label + "_a0_mm"] = np.array([q["a0_mm"] for q in rows])
        out[label] = dict(rows=rows, ratio_min=float(ratio.min()), ratio_median=float(np.median(ratio)),
                          ratio_max=float(ratio.max()), rpm_ratio_min=float(rpm[int(np.argmin(ratio))]),
                          n_reached_a0=int(sum(q["reached_a0"] for q in rows)),
                          n_ratio_below_0p95=int(np.sum(ratio < 0.95)), n_ratio_below_0p9=int(np.sum(ratio < 0.9)),
                          n_nonmonotone=int(sum(q["nonmonotone"] for q in rows)),
                          max_orbit_peak_V_at_a_circ=float(max(q["orbit_peak_V_at_a_circ"] for q in rows)),
                          max_orbit_peak_V_up_to_a_circ=float(max(q["orbit_peak_V_up_to_a_circ"] for q in rows)),
                          n_orbit_clipped_below_a_circ=int(sum(q["orbit_peak_V_up_to_a_circ"] > R.VMAX for q in rows)),
                          rpm_orbit_clipped_below_a_circ=[q["rpm"] for q in rows if q["orbit_peak_V_up_to_a_circ"] > R.VMAX])
        print(label, "a_circ / a_lim(0): min %.4f (%.0f rpm), median %.4f, max %.4f; reached a0 at %d; nonmonotone %d" % (
            ratio.min(), rpm[int(np.argmin(ratio))], np.median(ratio), ratio.max(), out[label]["n_reached_a0"],
            out[label]["n_nonmonotone"]), flush=True)
    # band edge (every 20 rpm) against a_circ and a_lim(0) at the same speeds and gains
    idx = np.searchsorted(rpm, band["rpm"])
    assert np.allclose(rpm[idx], band["rpm"])
    gb = band["g"]
    assert np.allclose(gb, g_best[idx])
    edge = band["a_band_low"]                                  # mm
    found = np.isfinite(edge)
    ac = arrays["g_best_a_circ_mm"][idx]
    a0 = arrays["g_best_a0_mm"][idx]
    out["band"] = dict(n_found=int(found.sum()), edge_over_a_circ_min=float(np.min(edge[found] / ac[found])),
                       edge_over_a0_min=float(np.min(edge[found] / a0[found])),
                       all_edges_above_a_circ=bool(np.all(edge[found] > ac[found])))
    # gain sweep at the four basin speeds
    sweep = []
    for n in BASIN_RPM:
        for g in G_SWEEP:
            r = a_circ(n, g)
            sweep.append(dict(rpm=n, g=g, a0_mm=r["a0"] * 1e3, a_circ_mm=r["a_circ"] * 1e3,
                              ratio=r["a_circ"] / r["a0"], nonmonotone=r["nonmonotone"]))
    out["gain_sweep"] = sweep
    # resolution check: the plain grid minimum against the refined one at the basin speeds, g = 1500
    chk = []
    for n in BASIN_RPM:
        r = a_circ(n, 1500.0)
        for frac in (0.5, 0.9, 0.99):
            a = min(frac * r["a0"], r["a_circ"]) if frac < 0.99 else 0.99 * r["a0"]
            chk.append(dict(rpm=n, a_mm=a * 1e3, mu_grid=mu(n, a, 1500.0, refine=False)[0],
                            mu_refined=mu(n, a, 1500.0)[0]))
    out["refinement_check"] = chk
    out["notes"] = ("averaged zero-order model, linear force law, no contact loss; sector [0, 1] holds while the "
                    "forced orbit is unclipped (orbit_peak_V_at_a_circ << 9 V); circle criterion for the "
                    "time-varying sector (Zames 1966; Desoer & Vidyasagar 1975)")
    out = add_orbit_limit(out)
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(HERE, "circle_criterion.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(HERE, "circle_criterion.npz"), rpm=rpm, g_best=g_best, **arrays)
    print("done in %.0f s" % out["runtime_s"])


if __name__ == "__main__":
    main()
