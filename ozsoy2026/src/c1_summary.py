"""Headline numbers of C1, computed from the stored results.

    python c1_summary.py      -> ../results/c1_summary.json
The report quotes these numbers; tests/test_c1_results.py pins them.
"""
import json
import os
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VMAX = P.F_MAX / P.GA_GAIN


def load(name):
    return np.load(os.path.join(RES, name))


def r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def map_summary():
    d = load("c1_map.npz")
    al = d["a_lim"]
    out = dict(n_speeds=int(len(al)), build=str(d["build"]))
    for k in ("step_clip", "step_free", "edge_clip", "feed_clip"):
        q = d[f"a_{k}"] / al
        out[k] = dict(mean_mm=r(np.nanmean(d[f"a_{k}"])), ratio_min=r(q.min(), 3),
                      ratio_median=r(np.median(q), 3), below_097=int(np.sum(q < 0.97)))
    hit = d["a_step_clip"] < 0.97 * al
    out["a_lim_mean_mm"] = r(al.mean())
    out["step_zone_width_mean_mm"] = r(np.mean((al - d["a_step_clip"])[hit]))
    out["step_zone_width_max_mm"] = r(np.max((al - d["a_step_clip"])[hit]))
    out["step_zone_ratio_min_rpm"] = float(d["rpm"][np.argmin(d["a_step_clip"] / al)])
    soft = np.minimum(d["a_edge_clip"], d["a_feed_clip"]) / al
    out["soft_both_ge_097"] = int(np.sum(soft >= 0.97))
    gain = (np.maximum(d["a_edge_clip"], d["a_feed_clip"]) - d["a_step_clip"])[hit]
    out["soft_minus_step_mean_mm"] = r(gain.mean())
    i = int(np.argmin(np.abs(d["rpm"] - 2500)))
    out["at_2500"] = dict(a_lim=r(al[i]), step_clip=r(d["a_step_clip"][i]), edge_clip=r(d["a_edge_clip"][i]),
                          feed_clip=r(d["a_feed_clip"][i]), step_free=r(d["a_step_free"][i]))
    return out


def band_summary():
    from zoa import a_lim
    b = load("c1_band.npz")
    al = np.array([a_lim(n, g) for n, g in zip(b["rpm"], b["g"])]) * 1e3
    a0 = np.array([a_lim(n, 0.0) for n in b["rpm"]]) * 1e3
    low = b["a_band_low"]
    ok = np.isfinite(low)
    q = low[ok] / al[ok]
    return dict(n_speeds=int(len(low)), found=int(ok.sum()), none_at_rpm=b["rpm"][~ok].tolist(),
                g_where_none=[r(x, 0) for x in b["g"][~ok]],
                ratio_min=r(q.min()), ratio_median=r(np.median(q)), ratio_max=r(q.max()),
                over_alim0_min=r((low[ok] / a0[ok]).min()), kick_mps=float(b["dv_hi"]))


def basin_summary():
    b, d = load("c1_basin.npz"), load("c1_df.npz")
    out = {}
    clip, free = b["vmax"] > 0, b["vmax"] == 0
    out["free_all_inf"] = bool(np.all(np.isinf(b["dv"][free])))
    out["free_fracs"] = sorted(set(b["frac"][free].round(2).tolist()))
    ratios = []
    for n in sorted(set(b["rpm"].tolist())):
        m = clip & (b["rpm"] == n)
        f, dv = b["frac"][m], b["dv"][m]
        med = np.median(dv, axis=1)
        fin = np.all(np.isfinite(dv), axis=1)
        j = int(np.argmin(np.abs(d["rpm"] - n)))
        p = b["p_mode"][m]
        ok = np.isfinite(p) & fin
        dvdf = np.interp(f[ok], d["frac"], d["A_dual"][j]) / (d["g"][j] * p[ok])
        ratios += list(med[ok] / dvdf)
        imp = b["mass"] * med / b["tooth_impulse"][m]
        out[f"{n:.0f}"] = dict(first_finite_frac=r(f[fin].min()) if fin.any() else None,
                               dv_090_mmps=r(med[f == 0.9][0] * 1e3), dv_098_mmps=r(med[f == 0.98][0] * 1e3),
                               impulse_098_pct=r(imp[f == 0.98][0] * 100, 1),
                               phase_spread_098=r(dv[f == 0.98].max() / dv[f == 0.98].min()))
    out["td_over_df_min"], out["td_over_df_max"] = r(min(ratios)), r(max(ratios))
    return out


def gainplane_summary():
    d = load("c1_gainplane.npz")
    out = {}
    for i, n in enumerate(d["rpm"]):
        g = d["gains"]
        sat = g[d["orbit_peak_V"][i] > VMAX]
        hi = g >= 1500
        out[f"{n:.0f}"] = dict(step_clip_plateau_mm=r(np.median(d["a_step_clip"][i][hi])),
                               a_lim_at_3000=r(d["a_lim"][i][-1]),
                               edge_at_3000=r(d["a_edge_clip"][i][-1]),
                               orbit_saturates_from_g=r(sat.min(), 0) if sat.size else None)
    return out


def main():
    out = {}
    for k, f in (("map", map_summary), ("band", band_summary), ("basin", basin_summary),
                 ("gainplane", gainplane_summary)):
        try:
            out[k] = f()
        except FileNotFoundError:
            out[k] = None
    json.dump(out, open(os.path.join(RES, "c1_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
