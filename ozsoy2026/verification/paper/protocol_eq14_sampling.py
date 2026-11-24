"""Sampling of Eq. (14): the steady demand maximised over 80 depths against a dense search.

Reviewer's question: the authority condition of Eq. (14) of the manuscript,
F_ss(Omega, g) = max over 0 < a <= kappa a_lim(Omega, g) of K_a b_max(a) <= F_max,
is evaluated in src/feasibility.demand_envelope on 80 evenly spaced depths
(and the peak over time on 256 samples per tooth period, 40 harmonics).  The
forced response oscillates in the depth (helix factor of Eq. Fm), so the
sampled maximum can miss the true one.  How large is the underestimate, and
would any stored gain change?

For all 251 speeds of results/feas_g1500.npz, at the stored gain g_best
(G = [0, 1500] V s/m, kappa = 0.9, guard 1, F_max = 27 N):
  d80     demand_envelope as used (80 depths, 256 time samples);
  d2000   the same on 2000 depths, which contain the 80 (isolates the depth
          sampling; d2000 >= d80 by construction);
  dref    every local maximum of the 2000-depth profile within 0.5 % of its
          top, and the end depth kappa a_lim, refined by a bounded 1-D
          maximisation in the depth (scipy minimize_scalar over one dense step
          on either side, xatol 1e-10 m) with 4096 time samples per tooth
          period, which contain the 256 (depth and time refined; dref >= d2000);
  dref_h80  dref's depth recomputed with 80 harmonics instead of 40
          (truncation, for information).
Underestimates: (d2000 - d80) / d2000 (depth sampling) and (dref - d80) / dref
(total), in N and in V (divided by K_a = 3 N/V).

Gains.  Where dref(g_best) > 27 N the stored gain is infeasible under the
dense condition.  The upper edge of the feasible interval is then
  * estimated at every such speed by linearising the demand in g at g_best
    (slope from the 80-depth demand at g_best and g_best - 0.5 V s/m):
    g_lin = g_best - (dref(g_best) - 27 N) / slope;
  * found without the bisection grid (Brent's method on demand(g) = 27 N,
    xtol 1e-7 V s/m) for both the 80-depth and the dense demand, and
  * recomputed by the bisection of feasibility.edge_gain (g_hi = 1500,
    tol 0.25 V s/m) with the dense demand in place of the 80-depth one,
at the N_TOP = 10 speeds with the largest relative underestimate, the 10
with the largest absolute underestimate and the 10 with the largest
linearised gain drop (the protocol answer).  The bisection walks
the same midpoints as edge_gain, so it differs from the stored bisection
result g_ref by whole grid steps (1500/2^13 = 0.18 V s/m) or not at all; it
assumes, as edge_gain does, that the feasible set is one interval [0, g_b].

Closed form only (about 2 min on one core).  Re-running skips the work if the
JSON exists (--fresh recomputes).  Run:
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 verification/paper/protocol_eq14_sampling.py
Writes verification/paper/protocol_eq14_sampling.json and .npz.
"""
import argparse
import os
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import protocol_common as PC
import feasibility as F

OUT = os.path.join(PC.HERE, "protocol_eq14_sampling.json")
NPZ = os.path.join(PC.HERE, "protocol_eq14_sampling.npz")
N_DENSE = 2000
N_T_FINE = 4096
TOP_REL = 0.005                     # refine every local maximum within 0.5 % of the dense top
FMAX = PC.P.F_MAX
GUARD = F.GUARD
KAPPA = F.KAPPA
G_HI = 1500.0
TOL = 0.25
N_TOP = 10


def demand_at(rpm, g, a, n_t=N_T_FINE, n_harm=F.N_HARM):
    return float(F.forced_demand(rpm, g, [a], n_harm=n_harm, n_t=n_t)[0])


def dense_envelope(rpm, g):
    """(a_lim, d2000, dref, depth of dref) at gain g."""
    al = PC.a_lim(rpm, g)
    if not np.isfinite(al) or g == 0:
        return al, 0.0, 0.0, np.nan
    top = KAPPA * al
    depths = np.linspace(top / N_DENSE, top, N_DENSE)
    d = F.forced_demand(rpm, g, depths)
    d2000 = float(d.max())
    h = depths[1] - depths[0]
    loc = [i for i in range(N_DENSE) if (i == 0 or d[i] >= d[i - 1]) and (i == N_DENSE - 1 or d[i] >= d[i + 1])]
    cand = [i for i in loc if d[i] >= (1 - TOP_REL) * d2000]
    best, a_best = -np.inf, np.nan
    for i in cand:
        lo, hi = max(depths[i] - h, 1e-9), min(depths[i] + h, top)
        r = minimize_scalar(lambda a: -demand_at(rpm, g, a), bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-10))
        for a_try, v in ((r.x, -r.fun), (depths[i], demand_at(rpm, g, depths[i]))):
            if v > best:
                best, a_best = v, a_try
    vend = demand_at(rpm, g, top)
    if vend > best:
        best, a_best = vend, top
    return al, d2000, float(best), float(a_best)


def edge_gain_dense(rpm, g_hi=G_HI, tol=TOL):
    """feasibility.edge_gain with the dense demand (same midpoints)."""
    ok = lambda g: np.isfinite(PC.a_lim(rpm, g)) and dense_envelope(rpm, g)[2] <= GUARD * FMAX
    if ok(g_hi):
        return g_hi
    lo, hi = 0.0, g_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def edge_continuous(demand, g0, step=0.5):
    """Gain at which demand(g) = guard F_max (Brent), in a bracket grown around g0."""
    f = lambda g: demand(g) - GUARD * FMAX
    lo, hi = g0 - step, g0 + step
    while f(lo) > 0:
        lo -= 2 * step
    while f(hi) <= 0:
        hi += 2 * step
    return float(brentq(f, lo, hi, xtol=1e-7, rtol=1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.fresh:
        print(f"{OUT} exists; --fresh recomputes")
        return
    t0 = time.time()
    D = np.load(os.path.join(PC.RES, "feas_g1500.npz"))
    assert float(D["zeta"]) == PC.ZETA and float(D["kappa"]) == KAPPA and float(D["gmax"]) == G_HI
    rpm, gb, gref = D["rpm"], D["g_best"], D["g_ref"]
    n = len(rpm)
    cols = {k: np.full(n, np.nan) for k in ("a_lim_mm", "d80", "d2000", "dref", "dref_h80", "a_at_dref_frac",
                                             "a_at_d80_frac", "slope_N_per_Vspm", "g_lin")}
    for i, (N, g) in enumerate(zip(rpm, gb)):
        N, g = float(N), float(g)
        al, d80 = F.demand_envelope(N, g)
        depths80 = np.linspace(KAPPA * al / F.N_DEPTH, KAPPA * al, F.N_DEPTH)
        cols["a_at_d80_frac"][i] = depths80[int(np.argmax(F.forced_demand(N, g, depths80)))] / al
        _, d2000, dref, a_ref = dense_envelope(N, g)
        cols["a_lim_mm"][i] = al * 1e3
        cols["d80"][i], cols["d2000"][i], cols["dref"][i] = d80, d2000, dref
        cols["a_at_dref_frac"][i] = a_ref / al
        cols["dref_h80"][i] = demand_at(N, g, a_ref, n_harm=80)
        cols["slope_N_per_Vspm"][i] = (d80 - F.demand_envelope(N, g - 0.5)[1]) / 0.5
    infeasible = cols["dref"] > GUARD * FMAX
    bind = gb < G_HI
    cols["g_lin"] = np.where(infeasible, gb - (cols["dref"] - GUARD * FMAX) / cols["slope_N_per_Vspm"], gb)
    pos = cols["d80"] > 0
    rel_depth = np.where(pos, (cols["d2000"] - cols["d80"]) / cols["d2000"], np.nan)
    rel_tot = np.where(pos, (cols["dref"] - cols["d80"]) / cols["dref"], np.nan)
    rel_time = np.where(pos, (cols["dref"] - cols["d2000"]) / cols["dref"], np.nan)
    abs_tot = cols["dref"] - cols["d80"]

    # the speeds with the largest relative and absolute underestimates (all speeds)
    top_rel = [int(i) for i in np.argsort(-np.nan_to_num(rel_tot, nan=-1))[:N_TOP]]
    top_abs = [int(i) for i in np.argsort(-abs_tot)[:N_TOP]]
    dg_lin0 = np.where(infeasible, gb - cols["g_lin"], -np.inf)
    top_dg = [int(i) for i in np.argsort(-dg_lin0)[:N_TOP] if infeasible[i]]
    top = sorted(set(top_rel) | set(top_abs) | set(top_dg))
    detail = []
    for i in top:
        N, g = float(rpm[i]), float(gb[i])
        row = dict(rpm=N, g_best=g, g_ref=float(gref[i]), at_bound=bool(g >= G_HI), d80_N=float(cols["d80"][i]),
                   d2000_N=float(cols["d2000"][i]), dref_N=float(cols["dref"][i]), rel_underestimate=float(rel_tot[i]),
                   abs_underestimate_N=float(abs_tot[i]), infeasible_under_dense=bool(infeasible[i]),
                   a_at_dref_frac=float(cols["a_at_dref_frac"][i]), a_at_d80_frac=float(cols["a_at_d80_frac"][i]))
        if infeasible[i]:
            gd = edge_gain_dense(N)
            # capped at the bound of G (an edge above it is not the gain)
            e80 = min(edge_continuous(lambda x: F.demand_envelope(N, x)[1], g), G_HI)
            ed = min(edge_continuous(lambda x: dense_envelope(N, x)[2], g), G_HI)
            row.update(g_bisect_dense=gd, bisect_change_from_g_ref_Vspm=float(gref[i] - gd),
                       bisect_change_from_g_best_Vspm=float(g - gd),
                       g_edge_80=e80, g_edge_dense=ed, edge_drop_Vspm=e80 - ed, edge_drop_rel=(e80 - ed) / e80,
                       a_lim_drop_mm=(PC.a_lim(N, e80) - PC.a_lim(N, ed)) * 1e3,
                       g_lin=float(cols["g_lin"][i]), lin_drop_Vspm=float(g - cols["g_lin"][i]))
        detail.append(row)
    inf_rows = [r for r in detail if r["infeasible_under_dense"]]
    dg_lin = gb - cols["g_lin"]
    da_lin = np.array([(cols["a_lim_mm"][i] - PC.a_lim(float(rpm[i]), float(cols["g_lin"][i])) * 1e3)
                       if infeasible[i] else 0.0 for i in range(n)])
    i_rel, i_abs = int(np.nanargmax(rel_tot)), int(np.nanargmax(abs_tot))
    res = dict(
        PC.header(__file__, __doc__),
        protocol=dict(n_depth_used=F.N_DEPTH, n_t_used=F.N_T, n_harm_used=F.N_HARM, n_depth_dense=N_DENSE,
                      n_t_fine=N_T_FINE, top_rel=TOP_REL, kappa=KAPPA, guard=GUARD, fmax_N=FMAX, g_hi=G_HI,
                      tol=TOL, n_top=N_TOP),
        n_speeds=n, n_binding=int(bind.sum()), n_at_bound=int((~bind).sum()),
        underestimate=dict(
            total_rel_max=float(rel_tot[i_rel]), total_rel_max_rpm=float(rpm[i_rel]),
            total_rel_max_d80_dref_N=[float(cols["d80"][i_rel]), float(cols["dref"][i_rel])],
            total_abs_max_N=float(abs_tot[i_abs]), total_abs_max_V=float(abs_tot[i_abs] / PC.P.GA_GAIN),
            total_abs_max_rpm=float(rpm[i_abs]),
            total_abs_max_d80_dref_N=[float(cols["d80"][i_abs]), float(cols["dref"][i_abs])],
            total_rel_median=float(np.nanmedian(rel_tot)), total_rel_p95=float(np.nanpercentile(rel_tot, 95)),
            total_rel_min=float(np.nanmin(rel_tot)),
            depth_sampling_rel_max=float(np.nanmax(rel_depth)),
            depth_sampling_rel_max_rpm=float(rpm[int(np.nanargmax(rel_depth))]),
            time_sampling_rel_max=float(np.nanmax(rel_time)),
            binding_rel_max=float(np.nanmax(rel_tot[bind])), binding_abs_max_N=float(np.nanmax(abs_tot[bind])),
            at_bound_rel_max=float(np.nanmax(rel_tot[~bind])), at_bound_abs_max_N=float(np.nanmax(abs_tot[~bind])),
            at_bound_max_dref_N=float(np.nanmax(cols["dref"][~bind])),
            harmonics_80_vs_40_rel_max=float(np.nanmax(np.abs(cols["dref_h80"][pos] / cols["dref"][pos] - 1))),
            n_speeds_with_d80_zero=int((~pos).sum())),
        gains=dict(
            n_infeasible_under_dense=int(infeasible.sum()),
            n_infeasible_at_bound=int((infeasible & ~bind).sum()),
            n_infeasible_binding=int((infeasible & bind).sum()),
            linearised_drop_max_Vspm=float(np.nanmax(dg_lin)),
            linearised_drop_max_rpm=float(rpm[int(np.nanargmax(dg_lin))]),
            linearised_drop_max_rel=float(np.nanmax(dg_lin / gb)),
            linearised_drop_median_where_infeasible=float(np.median(dg_lin[infeasible])) if infeasible.any() else 0.0,
            linearised_alim_drop_max_mm=float(np.max(da_lin)),
            top_speeds_bisection_changes_from_g_ref_Vspm=[r["bisect_change_from_g_ref_Vspm"] for r in inf_rows],
            top_speeds_bisection_n_changed=int(sum(r["bisect_change_from_g_ref_Vspm"] > 0 for r in inf_rows)),
            top_speeds_continuous_edge_drop_max_Vspm=max((r["edge_drop_Vspm"] for r in inf_rows), default=None),
            top_speeds_continuous_edge_drop_max_rel=max((r["edge_drop_rel"] for r in inf_rows), default=None),
            top_speeds_alim_drop_max_mm=max((r["a_lim_drop_mm"] for r in inf_rows), default=None),
            top_speeds_lin_vs_continuous=[(r["lin_drop_Vspm"], r["edge_drop_Vspm"]) for r in inf_rows]),
        top_speeds=detail,
        top_sets=dict(by_rel=[float(rpm[i]) for i in top_rel], by_abs=[float(rpm[i]) for i in top_abs],
                      by_linearised_gain_drop=[float(rpm[i]) for i in top_dg]),
        runtime_s=round(time.time() - t0, 1))
    PC.save_json(OUT, res)
    np.savez_compressed(NPZ, rpm=rpm, g_best=gb, g_ref=gref, infeasible=infeasible, **cols)
    print({k: v for k, v in res.items() if k in ("underestimate", "gains", "runtime_s")})


if __name__ == "__main__":
    main()
