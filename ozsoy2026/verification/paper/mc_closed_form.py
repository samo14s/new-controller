"""Closed-form Monte Carlo: robustness of the free voltage and of linear stability
of the g = 1500 V s/m DVF design to parameter uncertainty.

Purpose.  The manuscript quotes the free voltage u0 = Vclip - bmax that the
forced command bmax leaves under the clip, for nominal parameters only (it
lists the sensitivity to 10-20 % errors as not quantified).  Here the cutting
coefficients, the modal parameters, the actuator constant, eccentric runout,
the clip level and a loop delay are sampled, and at the four C1 basin speeds
and four depths fixed by the NOMINAL design the following are computed per
sample, in the closed-form zero-order (averaged) model of src/c4_robust.py:
  a_lim   linear stability limit of the delayed DVF loop (D-subdivision on the
          zoa grid; checked against c4_robust.a_lim_cf, argument principle)
  stable  whether the cut at the fixed depth is linearly stable (no root of
          Delta in Re s > 0; the DVF loop itself is checked at a = 0)
  bmax    peak of the forced command over one revolution, worst over 8 runout
          phases in [0, 2 pi / N) (c4_robust.worst_orbit_command at one depth)
  u0      Vclip - bmax; set to -inf where the cut is linearly unstable (no
          stable forced orbit, so no free voltage).  u0 <= 0 at a stable depth
          means the forced orbit itself saturates the command.

Protocol.
  parameters (mc_common.py): K_tc, K_rc +-10 % (level pm10) or +-20 % (level
    pm20); natural frequency +-5 % with K fixed; damping +-20 %; K_a +-10 %;
    runout U(0, 5 um); clip U(8.1, 9 V); loop delay U(0, 300 us); uniform and
    independent; the same uniform draws serve both levels (common random
    numbers); seed mc_common.SEED.
  speeds 1090, 1420, 2500, 2870 rpm; g = 1500 V s/m; depths 0.5, 0.7, 0.9,
    0.98 of the NOMINAL a_lim (zoa.a_lim at nominal parameters, 1.34 % damping).
  A case is (level, speed, depth); every case has n samples (default 4000).
  Statistics per case: median, 5th and 95th percentiles of u0 (all samples,
    -inf for unstable ones, and over the linearly stable samples only),
    P(linearly unstable), P(forced orbit saturates: stable and bmax >= Vclip),
    P(u0 <= 0) = their sum, with Wilson 95 % intervals.
  No-runout variant: the same samples with the runout set to 0 (stability
    unchanged), to separate the runout from the parametric uncertainty
    ("no_runout" in every case).
  Sensitivity: Spearman rank correlation of u0 (all samples, and stable only),
    of bmax (stable) and of a_lim with each parameter; one-at-a-time ranges
    (each parameter over its range on 11 points, the others nominal: runout 0,
    delay 0, clip 9 V).
  Checks before the campaign (all stored in "checks"):
    * the nominal sample reproduces the stored a_lim (results/c1_basin.npz),
      the stored free voltage at a_lim (verification/paper/revision_runs.json,
      from c1_df, 128 samples per tooth period) and the stored runout that
      brings the worst forced command to 9 V (revision_runs.json);
    * the vectorised model (mc_common.Model) against the src functions with the
      sample's parameters set in every module (mc_common.apply_params):
      a_lim_cf, unstable_zeros at the four depths, worst_orbit_command, zoa.a_lim;
    * every cached copy returns to nominal after the campaign;
    * 8 runout phases against 32 on a subset (the peak over the phases).

Run (about 4 min on one core):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
        python3 verification/paper/mc_closed_form.py [--n 4000]
Writes verification/paper/mc_closed_form.json and mc_closed_form.npz
(npz: the samples, a_lim, bmax, stable flags and u0 of both levels, and the
one-at-a-time grids, for figures).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_common as MC                       # noqa: E402  (puts src on the path, sets 1.34 %)
from mc_common import P, R, zoa              # noqa: E402

OUT_JSON = os.path.join(HERE, "mc_closed_form.json")
OUT_NPZ = os.path.join(HERE, "mc_closed_form.npz")
N_OAT = 11
LEVEL_KEY = {0.1: "pm10", 0.2: "pm20"}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def q(x, p):
    """Quantile without interpolation (an actual sample value); None for -inf."""
    v = float(np.quantile(x, p, method="inverted_cdf"))
    return None if not np.isfinite(v) else v


def spearman(x, y):
    m = np.isfinite(x) | np.isneginf(x)
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------ evaluation ----
def evaluate(p, depths_by_speed, n_psi=8, a0_stable=None):
    """One sample: a_lim (m) per speed, stable flags, bmax (V) and u0 (V) per
    speed and depth, the number of D-subdivision crossings below each depth,
    and whether the exact zero count was needed (two or more crossings)."""
    m = MC.Model(p)
    ns, nd = len(MC.SPEEDS), len(MC.FRACS)
    al = np.empty(ns)
    stable = np.zeros((ns, nd), bool)
    ncr = np.zeros((ns, nd), int)
    exact = np.zeros((ns, nd), bool)
    bmax = np.empty((ns, nd))
    if a0_stable is None:
        a0_stable = m.zeros(MC.SPEEDS[0], 0.0) == 0          # DVF loop without cutting (speed-free)
    for i, rpm in enumerate(MC.SPEEDS):
        d = depths_by_speed[i]
        c = m.crossings(rpm)
        al[i] = (c[0] if c.size else np.inf) if a0_stable else 0.0
        for j, a in enumerate(d):
            ncr[i, j] = int(np.sum(c < a))
            if not a0_stable:
                stable[i, j] = False
            elif ncr[i, j] == 0:
                stable[i, j] = True
            elif ncr[i, j] == 1:
                stable[i, j] = False
            else:
                exact[i, j] = True
                stable[i, j] = m.zeros(rpm, a) == 0
        bmax[i] = m.bmax(rpm, d, n_psi=n_psi)
    u0 = np.where(stable, m.vclip - bmax, -np.inf)
    return dict(a_lim=al, stable=stable, bmax=bmax, u0=u0, ncross=ncr, exact=exact, a0_stable=bool(a0_stable))


def nominal_depths():
    MC.restore_nominal()
    al = np.array([zoa.a_lim(n, MC.G_DESIGN) for n in MC.SPEEDS])
    return al, np.array([[f * a for f in MC.FRACS] for a in al])


# ---------------------------------------------------------------- checks ----
def check_nominal(al_nom, depths):
    """The nominal sample against the stored results."""
    out = {}
    basin = np.load(os.path.join(MC.RES, "c1_basin.npz"))
    rev = json.load(open(os.path.join(HERE, "revision_runs.json")))
    m = MC.Model(MC.NOMINAL)
    from feasibility import forced_velocity
    for i, rpm in enumerate(MC.SPEEDS):
        k = str(int(rpm))
        sel = (basin["rpm"] == rpm) & (basin["vmax"] > 0)
        stored_al = float(basin["a_lim"][sel][0])
        b_cf = m.bmax(rpm, depths[i])
        b_src = [R.orbit_command(rpm, a, MC.G_DESIGN)[0] for a in depths[i]]
        # free voltage at a_lim as stored (c1_df: forced_velocity with 128 samples per tooth period)
        b_al_128 = float(np.abs(-MC.G_DESIGN * forced_velocity(rpm, MC.G_DESIGN, [al_nom[i]], n_t=P.STEPS_PER_REV // P.N_TEETH)[0]).max())
        b_al_cf = float(m.bmax(rpm, [al_nom[i]])[0])
        st = rev["stored"]["free_voltage_at_alim"][k]
        out[k] = dict(
            a_lim_mm=al_nom[i] * 1e3, a_lim_stored_mm=stored_al, a_lim_model_mm=m.a_lim(rpm) * 1e3,
            a_lim_cf_mm=R.a_lim_cf(rpm, MC.G_DESIGN) * 1e3,
            depths_mm=[float(a * 1e3) for a in depths[i]],
            bmax_V=[float(b) for b in b_cf], bmax_src_orbit_command_V=[float(b) for b in b_src],
            u0_9V=[float(9.0 - b) for b in b_cf], u0_8p1V=[float(8.1 - b) for b in b_cf],
            bmax_at_alim_128_per_tooth_V=b_al_128, bmax_at_alim_stored_V=st["bmax_V"],
            bmax_at_alim_model_V=b_al_cf, free_at_alim_9V_model=9.0 - b_al_cf)
    # runout that brings the worst forced command over 40 depths up to 0.9 a_lim to 9 V
    rr = {}
    for i, rpm in enumerate(MC.SPEEDS):
        dep = np.linspace(0.9 * al_nom[i] / 40, 0.9 * al_nom[i], 40)

        def f(r):
            return float(MC.Model(dict(MC.NOMINAL, runout=r)).bmax(rpm, dep).max())
        k = str(int(rpm))
        stored = rev["closed_form"]["runout_to_clip_g1500"][k]
        c0, hi = f(0.0), 10e-6
        if f(hi) < 9.0:
            rr[k] = dict(cmd0_V=c0, runout_um=None, stored=stored)
            continue
        lo = 0.0
        for _ in range(30):
            mm = 0.5 * (lo + hi)
            lo, hi = (mm, hi) if f(mm) < 9.0 else (lo, mm)
        rr[k] = dict(cmd0_V=c0, runout_um=hi * 1e6, stored=stored)
    out["runout_to_clip_g1500"] = rr
    return out


def check_against_src(S, depths, rng, n_alim=6, n_zero=40, n_orbit=6):
    """Model against src with the sample set in every module (apply_params)."""
    out = dict(alim=[], zeros=[], orbit=[], zoa=[])
    n = len(S["K_TC"])
    for kind, cnt in (("alim", n_alim), ("zeros", n_zero), ("orbit", n_orbit)):
        for _ in range(cnt):
            i, s = int(rng.integers(n)), int(rng.integers(len(MC.SPEEDS)))
            p = MC.sample(S, i)
            rpm = MC.SPEEDS[s]
            MC.apply_params(p)
            m = MC.Model(p)
            if kind == "alim":
                a_src = R.a_lim_cf(rpm, MC.G_DESIGN, tau=p["tau"])
                a_mc = m.a_lim(rpm)
                out["alim"].append(dict(i=i, rpm=rpm, a_lim_cf_mm=a_src * 1e3, model_mm=a_mc * 1e3,
                                        rel=float(a_mc / a_src - 1)))
                m0 = MC.Model(dict(p, tau=0.0))
                out["zoa"].append(dict(i=i, rpm=rpm, zoa_mm=zoa.a_lim(rpm, MC.G_DESIGN) * 1e3,
                                       model_tau0_mm=m0.a_lim(rpm) * 1e3))
            elif kind == "zeros":
                ev = evaluate(p, depths)
                z = [R.unstable_zeros(rpm, a, MC.G_DESIGN, tau=p["tau"]) for a in depths[s]]
                al = ev["a_lim"][s]
                zz = ([R.unstable_zeros(rpm, al * f, MC.G_DESIGN, tau=p["tau"]) for f in (0.999, 1.001)]
                      if np.isfinite(al) and al > 0 else None)
                out["zeros"].append(dict(i=i, rpm=rpm, src_zeros=z, model_stable=ev["stable"][s].tolist(),
                                         agree=bool(all((zi == 0) == st for zi, st in zip(z, ev["stable"][s]))),
                                         zeros_at_alim_0999_1001=zz))
            else:
                b_src = [R.worst_orbit_command(rpm, [a], MC.G_DESIGN, runout=p["runout"], tau=p["tau"])
                         for a in depths[s]]
                b_mc = m.bmax(rpm, depths[s])
                out["orbit"].append(dict(i=i, rpm=rpm, src_V=b_src, model_V=b_mc.tolist(),
                                         max_abs_diff_V=float(np.max(np.abs(np.array(b_src) - b_mc)))))
    MC.restore_nominal()
    out["summary"] = dict(
        alim_max_abs_rel=max(abs(r["rel"]) for r in out["alim"]),
        zoa_max_abs_rel=max(abs(r["zoa_mm"] / r["model_tau0_mm"] - 1) for r in out["zoa"]),
        zeros_all_agree=all(r["agree"] for r in out["zeros"]),
        zeros_at_alim_ok=all(r["zeros_at_alim_0999_1001"][0] == 0 and r["zeros_at_alim_0999_1001"][1] > 0
                             for r in out["zeros"] if r["zeros_at_alim_0999_1001"] is not None),
        orbit_max_abs_diff_V=max(r["max_abs_diff_V"] for r in out["orbit"]))
    return out


# ------------------------------------------------------------ statistics ----
def case_stats(u0, bmax, stable, vclip, params):
    n = len(u0)
    nu = int(np.sum(~stable))
    sat = stable & (bmax >= vclip)
    ns = int(np.sum(sat))
    le0 = nu + ns
    margin = stable & (u0 <= 0.1 * vclip)
    us = u0[stable]
    out = dict(n=n, p_unstable=nu / n, p_unstable_ci95=wilson(nu, n),
               p_saturated=ns / n, p_saturated_ci95=wilson(ns, n),
               p_u0_le_0=le0 / n, p_u0_le_0_ci95=wilson(le0, n),
               p_u0_le_0_given_stable=(ns / (n - nu)) if n > nu else None,
               p_u0_le_10pct_clip=(nu + int(np.sum(margin))) / n,
               u0_p5_V=q(u0, 0.05), u0_median_V=q(u0, 0.5), u0_p95_V=q(u0, 0.95),
               u0_p5_is_unstable=bool(q(u0, 0.05) is None), u0_median_is_unstable=bool(q(u0, 0.5) is None),
               u0_stable_p5_V=q(us, 0.05) if us.size else None,
               u0_stable_median_V=q(us, 0.5) if us.size else None,
               u0_stable_p95_V=q(us, 0.95) if us.size else None,
               u0_stable_min_V=float(us.min()) if us.size else None,
               bmax_stable_median_V=q(bmax[stable], 0.5) if us.size else None,
               bmax_stable_p95_V=q(bmax[stable], 0.95) if us.size else None,
               bmax_all_median_V=q(bmax, 0.5), bmax_all_p95_V=q(bmax, 0.95))
    rho_all = {k: spearman(u0, params[k]) for k in MC.NAMES}
    rho_st = {k: spearman(us, params[k][stable]) for k in MC.NAMES} if us.size > 10 else {}
    rho_b = {k: spearman(bmax[stable], params[k][stable]) for k in MC.NAMES if k != "vclip"} if us.size > 10 else {}

    def dom(r):
        r = {k: v for k, v in r.items() if v is not None}
        if not r:
            return None
        k = max(r, key=lambda z: abs(r[z]))
        return dict(param=k, rho=r[k], ranking=sorted(r, key=lambda z: -abs(r[z])))
    out.update(spearman_u0_all=rho_all, spearman_u0_stable=rho_st, spearman_bmax_stable=rho_b,
               dominant_u0_all=dom(rho_all), dominant_u0_stable=dom(rho_st), dominant_bmax_stable=dom(rho_b))
    return out


def run_level(level, n, al_nom, depths, a0_cache):
    S = MC.draw(n, level)
    ns, nd = len(MC.SPEEDS), len(MC.FRACS)
    AL = np.empty((n, ns))
    ST = np.zeros((n, ns, nd), bool)
    BM = np.empty((n, ns, nd))
    U0 = np.empty((n, ns, nd))
    NC = np.zeros((n, ns, nd), int)
    EX = np.zeros((n, ns, nd), bool)
    A0 = np.zeros(n, bool)
    BM0 = np.empty((n, ns, nd))                 # same samples without runout
    for i in range(n):
        p = MC.sample(S, i)
        ev = evaluate(p, depths, a0_stable=a0_cache.get(i))
        a0_cache[i] = ev["a0_stable"]
        AL[i], ST[i], BM[i], U0[i], NC[i], EX[i], A0[i] = (ev["a_lim"], ev["stable"], ev["bmax"], ev["u0"],
                                                         ev["ncross"], ev["exact"], ev["a0_stable"])
        m0 = MC.Model(dict(p, runout=0.0))
        BM0[i] = [m0.bmax(rpm, depths[s]) for s, rpm in enumerate(MC.SPEEDS)]
    U00 = np.where(ST, S["vclip"][:, None, None] - BM0, -np.inf)
    res = dict(n=n, ranges={k: list(v) for k, v in MC.ranges(level).items()},
               loop_unstable_without_cut=int(np.sum(~A0)), exact_zero_counts_needed=int(EX.sum()),
               speeds={})
    for s, rpm in enumerate(MC.SPEEDS):
        ratio = AL[:, s] / al_nom[s]
        d = dict(a_lim_nominal_mm=al_nom[s] * 1e3,
                 a_lim_median_mm=float(np.median(AL[:, s]) * 1e3),
                 a_lim_p5_mm=float(np.quantile(AL[:, s], 0.05) * 1e3),
                 a_lim_p95_mm=float(np.quantile(AL[:, s], 0.95) * 1e3),
                 a_lim_ratio_p5=float(np.quantile(ratio, 0.05)), a_lim_ratio_median=float(np.median(ratio)),
                 a_lim_ratio_p95=float(np.quantile(ratio, 0.95)), a_lim_ratio_min=float(ratio.min()),
                 a_lim_ratio_max=float(ratio.max()),
                 spearman_a_lim={k: spearman(AL[:, s], S[k]) for k in MC.NAMES if k not in ("runout", "vclip")},
                 fracs={})
        for j, f in enumerate(MC.FRACS):
            cs = case_stats(U0[:, s, j], BM[:, s, j], ST[:, s, j], S["vclip"], S)
            c0 = case_stats(U00[:, s, j], BM0[:, s, j], ST[:, s, j], S["vclip"], S)
            cs.update(depth_mm=float(depths[s, j] * 1e3),
                      no_runout={k: c0[k] for k in ("p_saturated", "p_saturated_ci95", "p_u0_le_0", "p_u0_le_0_ci95",
                                                    "p_u0_le_10pct_clip", "u0_p5_V", "u0_median_V", "u0_p95_V",
                                                    "u0_stable_p5_V", "u0_stable_median_V", "u0_stable_p95_V",
                                                    "bmax_stable_median_V", "bmax_stable_p95_V",
                                                    "dominant_u0_stable", "spearman_u0_stable")})
            d["fracs"][str(f)] = cs
        res["speeds"][str(int(rpm))] = d
    arrays = dict(params=np.column_stack([S[k] for k in MC.NAMES]), psi=S["psi"], a_lim=AL, stable=ST,
                  bmax=BM, u0=U0, ncross=NC, exact=EX, a0_stable=A0, bmax_no_runout=BM0, u0_no_runout=U00)
    return res, arrays


def oat(level, al_nom, depths):
    """One parameter at a time over its range (11 points), the others nominal."""
    rg = MC.ranges(level)
    out, grids = {}, {}
    for k in MC.NAMES:
        vals = np.linspace(rg[k][0], rg[k][1], N_OAT)
        evs = [evaluate(dict(MC.NOMINAL, **{k: float(v)}), depths) for v in vals]
        U = np.array([e["u0"] for e in evs])            # (N_OAT, ns, nd)
        AL = np.array([e["a_lim"] for e in evs])        # (N_OAT, ns)
        BM = np.array([e["bmax"] for e in evs])
        grids[k] = dict(values=vals, u0=U, a_lim=AL, bmax=BM)
        per = {}
        for s, rpm in enumerate(MC.SPEEDS):
            e = dict(a_lim_ratio_lo=float(AL[0, s] / al_nom[s]), a_lim_ratio_hi=float(AL[-1, s] / al_nom[s]),
                     a_lim_ratio_min=float(AL[:, s].min() / al_nom[s]), a_lim_ratio_max=float(AL[:, s].max() / al_nom[s]),
                     fracs={})
            for j, f in enumerate(MC.FRACS):
                u = U[:, s, j]
                fin = np.isfinite(u)
                e["fracs"][str(f)] = dict(
                    u0_lo_V=float(u[0]) if fin[0] else None, u0_hi_V=float(u[-1]) if fin[-1] else None,
                    u0_min_V=float(u[fin].min()) if fin.any() else None,
                    u0_max_V=float(u[fin].max()) if fin.any() else None,
                    u0_range_V=(float(u.max() - u.min()) if fin.all() else None),
                    unstable_points=int(np.sum(~fin)), bmax_range_V=float(BM[:, s, j].max() - BM[:, s, j].min()))
            per[str(int(rpm))] = e
        out[k] = dict(values=vals.tolist(), unit=MC.UNITS[k], speeds=per)
    # ranking by the u0 range at each case (an unstable point counts as an infinite range)
    rank = {}
    for s, rpm in enumerate(MC.SPEEDS):
        for f in MC.FRACS:
            r = {}
            for k in MC.NAMES:
                e = out[k]["speeds"][str(int(rpm))]["fracs"][str(f)]
                r[k] = np.inf if e["u0_range_V"] is None else e["u0_range_V"]
            order = sorted(r, key=lambda z: -r[z])
            rank[f"{int(rpm)}_{f}"] = [(z, None if not np.isfinite(r[z]) else float(r[z])) for z in order]
    out["ranking_by_u0_range"] = rank
    return out, grids


def check_npsi(S, depths, n_sub=50):
    """Worst forced command over 8 runout phases against 32 phases."""
    d = []
    for i in range(n_sub):
        m = MC.Model(MC.sample(S, i))
        for s, rpm in enumerate(MC.SPEEDS):
            b8 = m.bmax(rpm, depths[s], n_psi=8)
            m._W = None
            b32 = m.bmax(rpm, depths[s], n_psi=32)
            m._W = None
            d.append(b32 - b8)
    d = np.array(d)
    return dict(n_pairs=int(d.shape[0]), max_underestimate_V=float(d.max()), median_V=float(np.median(d)),
                p99_V=float(np.quantile(d, 0.99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="samples per level (>= 2000)")
    ap.add_argument("--quick", action="store_true", help="functional test: skip the src checks, write to a scratch name")
    a = ap.parse_args()
    global OUT_JSON, OUT_NPZ
    if a.quick:
        OUT_JSON, OUT_NPZ = OUT_JSON.replace(".json", "_quick.json"), OUT_NPZ.replace(".npz", "_quick.npz")
    t0 = time.time()
    state0 = MC.cache_state()
    al_nom, depths = nominal_depths()
    rng = np.random.default_rng(MC.SEED + 1)
    checks = dict(nominal=check_nominal(al_nom, depths))
    S10 = MC.draw(a.n, 0.1)
    S20 = MC.draw(a.n, 0.2)
    if not a.quick:
        checks["model_vs_src_pm10"] = check_against_src(S10, depths, rng)
        checks["model_vs_src_pm20"] = check_against_src(S20, depths, rng, n_alim=4, n_zero=20, n_orbit=4)
        checks["n_psi_8_vs_32"] = check_npsi(S10, depths)
        print("checks:", json.dumps(checks["model_vs_src_pm10"]["summary"]),
              json.dumps(checks["model_vs_src_pm20"]["summary"]), json.dumps(checks["n_psi_8_vs_32"]), flush=True)
    t_checks = time.time() - t0
    print(f"checks done in {t_checks:.0f} s", flush=True)
    levels, arrays, oats, ogrids = {}, {}, {}, {}
    a0_cache = {}
    for lev in MC.LEVELS:
        key = LEVEL_KEY[lev]
        t1 = time.time()
        levels[key], arrays[key] = run_level(lev, a.n, al_nom, depths, a0_cache)
        oats[key], ogrids[key] = oat(lev, al_nom, depths)
        levels[key]["runtime_s"] = time.time() - t1
        print(f"level {key}: {time.time() - t1:.0f} s", flush=True)
    MC.restore_nominal()
    checks["caches_restored"] = MC.cache_state() == state0
    nominal = {str(int(rpm)): dict(a_lim_mm=float(al_nom[s] * 1e3),
                                   depths_mm=[float(x * 1e3) for x in depths[s]],
                                   bmax_V=checks["nominal"][str(int(rpm))]["bmax_V"],
                                   u0_9V=checks["nominal"][str(int(rpm))]["u0_9V"],
                                   u0_8p1V=checks["nominal"][str(int(rpm))]["u0_8p1V"])
               for s, rpm in enumerate(MC.SPEEDS)}
    out = dict(
        note=("Closed-form (zero-order, averaged) Monte Carlo of the free voltage u0 = Vclip - bmax and of the "
              "linear stability of the g = 1500 V s/m DVF design at fixed depths (fractions of the NOMINAL "
              "a_lim). u0 = -inf (null in quantiles) where the cut is linearly unstable. See the docstring of "
              "mc_closed_form.py for the protocol; mc_common.py for the parameters and caches."),
        g=MC.G_DESIGN, speeds=list(MC.SPEEDS), fracs=list(MC.FRACS), seed=MC.SEED, zeta_nominal=MC.ZETA0,
        nominal_params=MC.NOMINAL, units=MC.UNITS, n_per_case=a.n,
        nominal=nominal, levels=levels, oat=oats, checks=checks,
        runtime_s=dict(total=time.time() - t0, checks=t_checks))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    with open(OUT_JSON, "w") as fh:
        json.dump(clean(out), fh, indent=1)
    npz = dict(names=np.array(MC.NAMES), speeds=np.array(MC.SPEEDS), fracs=np.array(MC.FRACS),
               a_lim_nominal=al_nom, depths=depths)
    for key in arrays:
        for k, v in arrays[key].items():
            npz[f"{key}_{k}"] = v
        for k, gr in ogrids[key].items():
            for kk, vv in gr.items():
                npz[f"{key}_oat_{k}_{kk}"] = vv
    np.savez_compressed(OUT_NPZ, **npz)
    print(f"done in {time.time() - t0:.0f} s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
