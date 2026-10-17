"""Which feasibility test reproduces the Fig. 8 gain map?

For a set of speeds, scan g on [0, 3000] and evaluate the Eq. (2) fitness under
several definitions of the demanded actuator force (all at depth a_lim(ZOA)):
  kind   force = 3 g|xdot|,  volt = g|xdot|,  actual = |f_act|
  spr    512 (paper, Table 2) or 2048 (removes most of the half-step force lag)
  window 'full' = all 50 revolutions, 'ss' = revolutions 25-50 only
The optimum per definition (largest feasible a_lim, refined by bisection on the
feasibility edge) is compared with Fig. 8 (vector-digitised) at the same speed.
Reads src/ only; writes gainmap_identification.{csv,json}.
"""
import csv
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)
import params as P               # noqa: E402
from zoa import a_lim, _gcl_grid  # noqa: E402
from tdsim import simulate       # noqa: E402

SPEEDS = [500, 740, 800, 1000, 1300, 1940, 2300, 2500, 2800, 3000]
GS = np.arange(0.0, 3000.0 + 1e-9, 25.0)
FMAX = P.F_MAX
CONFIGS = [(k, s, w) for s in (512, 2048) for k in ("force", "volt", "actual") for w in ("full", "ss")]

_gcl = {}


def alim(rpm, g):
    if g not in _gcl:
        if len(_gcl) > 400:
            _gcl.clear()
        _gcl[g] = _gcl_grid(g)
    return a_lim(rpm, g, gcl=_gcl[g])


def demands(rpm, g, spr):
    """All demand variants at a = a_lim(rpm, g) for one simulation."""
    a = alim(rpm, g)
    if not np.isfinite(a):
        return a, None
    if g == 0:
        return a, {c: 0.0 for c in CONFIGS if c[1] == spr}
    r = simulate(rpm, a, g, store=True, spr=spr)
    v, fa = np.abs(r["v"]), np.abs(r["fact"])
    i0 = (P.N_REVS // 2) * spr
    out = {}
    for k in ("force", "volt", "actual"):
        for w in ("full", "ss"):
            sl = slice(i0, None) if w == "ss" else slice(None)
            if k == "actual":
                d = fa[sl].max()
            else:
                d = (P.GA_GAIN if k == "force" else 1.0) * g * v[sl].max()
            out[(k, spr, w)] = float(d) if not r["diverged"] else np.inf
    return a, out


def main():
    fig8 = np.loadtxt(os.path.join(HERE, "..", "targets", "fig8_gain_map.csv"), delimiter=",", skiprows=1)
    rows = []
    t0 = time.time()
    for n in SPEEDS:
        scan = {c: [] for c in CONFIGS}
        al = []
        for g in GS:
            a512, d512 = demands(n, g, 512)
            _, d2048 = demands(n, g, 2048)
            al.append(a512)
            for c in CONFIGS:
                dd = d512 if c[1] == 512 else d2048
                scan[c].append(np.inf if dd is None else dd[c])
        al = np.array(al)
        g8 = float(np.interp(n, fig8[:, 0], fig8[:, 1]))
        for c in CONFIGS:
            d = np.array(scan[c])
            feas = (d <= FMAX) & np.isfinite(al)
            fit = np.where(feas, al, 0.0)
            j = int(np.argmax(fit))
            g_best, a_best = GS[j], fit[j]
            # bisection on the edge above the best grid point
            if j + 1 < len(GS) and not feas[j + 1]:
                lo, hi = GS[j], GS[j + 1]
                while hi - lo > 0.5:
                    mid = 0.5 * (lo + hi)
                    am, dm = demands(n, mid, c[1])
                    if dm is not None and dm[c] <= FMAX:
                        lo = mid
                        if am > a_best:
                            g_best, a_best = mid, am
                    else:
                        hi = mid
            n_int = int(np.sum(np.diff(feas.astype(int)) == -1) + (1 if feas[-1] else 0))
            rows.append(dict(rpm=n, kind=c[0], spr=c[1], window=c[2], g_opt=round(float(g_best), 2),
                             a_opt_mm=round(float(a_best) * 1e3, 4), g_fig8=round(g8, 1),
                             g_ratio_to_fig8=round(float(g_best) / g8, 3),
                             a_zoa_at_g_fig8_mm=round(float(alim(n, g8)) * 1e3, 4),
                             n_feasible_intervals=n_int,
                             upper_bound_hit=bool(feas[-1])))
        print(f"{n} rpm done ({time.time()-t0:.0f}s)", flush=True)
        for r in rows[-len(CONFIGS):]:
            print("   ", r["kind"], r["spr"], r["window"], r["g_opt"], r["a_opt_mm"], "fig8", r["g_fig8"], r["g_ratio_to_fig8"])
    with open(os.path.join(HERE, "gainmap_identification.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    summ = {}
    for c in CONFIGS:
        rr = [r for r in rows if (r["kind"], r["spr"], r["window"]) == c]
        lr = np.log([max(r["g_ratio_to_fig8"], 1e-3) for r in rr])
        summ["%s_spr%d_%s" % c] = dict(median_ratio=float(np.exp(np.median(lr))),
                                        rms_log_ratio=float(np.sqrt(np.mean(lr**2))),
                                        ratios={r["rpm"]: r["g_ratio_to_fig8"] for r in rr})
    with open(os.path.join(HERE, "gainmap_identification.json"), "w") as fh:
        json.dump(dict(zeta=P.ZETA, speeds=SPEEDS, g_grid_step=25.0, fmax=FMAX, summary=summ), fh, indent=1)
    for k, v in sorted(summ.items(), key=lambda kv: kv[1]["rms_log_ratio"]):
        print(f"{k:24s} median g/g_fig8 {v['median_ratio']:.3f}  rms log {v['rms_log_ratio']:.3f}")


if __name__ == "__main__":
    main()
