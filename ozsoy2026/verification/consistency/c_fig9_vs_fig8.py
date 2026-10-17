"""(c) Is the Fig. 9 controlled depth the zero-order a_lim at the Fig. 8 gain?

For every vertex of the Fig. 9 controlled curve (10-rpm grid):
  g8      Fig. 8 gain at that speed (linear interpolation of the Fig. 8
          polyline; flagged when the speed is not a Fig. 8 vertex)
  a_plot  plotted controlled depth (Fig. 9, solid)
  a_zoa   zero-order a_lim(rpm, g8) from src/zoa.py (zeta 1.34 %, and 1.45 %)
  a_unc   plotted uncontrolled depth (Fig. 9, dashed) and model a_lim(rpm, 0)
  a_best  max over g in [0, 2500] of a_lim(rpm, g) (no actuator limit)
  g_req   smallest gain on that grid whose a_lim reaches a_plot (if any)
Then the speed bands where the plotted controlled curve lies below the
plotted uncontrolled curve, and where the model says a_lim(g8) < a_lim(0).
Hypothesis test H1 (labelled as such in the outputs): the plotted depth is
the speed-independent lobe floor of the closed loop at the Fig. 8 gain,
a_floor(g) = 2 pi / (N Kt |alpha_yy| max_w(-Re Gcl(iw, g))).

Outputs: c_fig9_vs_fig8.csv, c_fig9_vs_fig8.json, c_fig9_vs_fig8.png
"""
import json
import os

import numpy as np

from common import fig8, fig9_controlled, fig9_uncontrolled, save_csv, save_json, HERE
import params as P
import structure as S
import zoa

G_SCAN = np.arange(0.0, 2501.0, 25.0)


def bands(rpm, mask, extra=None):
    out = []
    i = 0
    n = len(rpm)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            b = dict(rpm_from=rpm[i], rpm_to=rpm[j], n_points=j - i + 1)
            if extra is not None:
                b.update(extra(i, j))
            out.append(b)
            i = j + 1
        else:
            i += 1
    return out


F_FLOOR = np.concatenate([np.linspace(0.2, 60, 20000), np.linspace(60, 400, 200001)])


def a_floor(g):
    """Minimum over spindle speed of the ZOA limit (mm): 2 pi / (N Kt alpha Re Gcl)."""
    re = S.Gcl(2j * np.pi * F_FLOOR, g).real
    return 2 * np.pi / (P.N_TEETH * P.K_TC * abs(zoa.ALPHA) * (-re.min())) * 1e3


def crossings(x, d):
    """rpm where d changes sign (linear interpolation)."""
    out = []
    for i in np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
        out.append(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))
    return out


def main():
    x8, y8 = fig8()
    r9, a9 = fig9_controlled()
    xu, au = fig9_uncontrolled()
    o = np.argsort(xu, kind="stable")
    xu, au = xu[o], au[o]
    rpm = np.round(r9 / 10) * 10                  # nominal 10-rpm grid
    g8 = np.interp(r9, x8, y8)
    is_vertex = np.array([np.min(np.abs(x8 - r)) < 1.0 for r in r9])
    a_unc_plot = np.interp(r9, xu, au)

    res = {}
    for z in (0.0134, 0.0145):
        P.set_damping(z)
        res[z] = dict(a_zoa=np.array([zoa.a_lim(r, g) for r, g in zip(rpm, g8)]) * 1e3,
                      a0=zoa.sld(rpm, 0.0) * 1e3)
    P.set_damping(0.0134)

    # gain scan (zeta 1.34 %)
    cache = os.path.join(HERE, "c_gain_scan_cache.npz")
    if os.path.exists(cache):
        A = np.load(cache)["A"]
    else:
        A = np.empty((len(rpm), len(G_SCAN)))
        for j, g in enumerate(G_SCAN):
            gcl = zoa._gcl_grid(g)
            for i, r in enumerate(rpm):
                A[i, j] = zoa.a_lim(r, g, gcl=gcl)
        A *= 1e3
        np.savez_compressed(cache, A=A, g=G_SCAN, rpm=rpm)
    a_best = A.max(axis=1)
    g_best = G_SCAN[A.argmax(axis=1)]
    g_req = np.array([G_SCAN[np.argmax(A[i] >= a9[i])] if np.any(A[i] >= a9[i]) else np.nan
                      for i in range(len(rpm))])

    z = res[0.0134]
    afl = np.array([a_floor(g) for g in g8])
    rel_floor = a9 / afl - 1
    a_min_g = A.min(axis=1)
    rel = a9 / z["a_zoa"] - 1
    rel145 = a9 / res[0.0145]["a_zoa"] - 1
    rows = [(rpm[i], g8[i], int(is_vertex[i]), a9[i], z["a_zoa"][i], rel[i] * 100,
             res[0.0145]["a_zoa"][i], rel145[i] * 100, a_unc_plot[i], z["a0"][i], a_best[i], g_best[i],
             g_req[i], afl[i], rel_floor[i] * 100, a_min_g[i]) for i in range(len(rpm))]
    save_csv("c_fig9_vs_fig8.csv",
             ["rpm", "g_fig8", "g_is_fig8_vertex", "a_plot_mm", "a_zoa_g8_z134_mm", "rel_diff_pct_z134",
              "a_zoa_g8_z145_mm", "rel_diff_pct_z145", "a_uncontrolled_plot_mm", "a_zoa_g0_z134_mm",
              "a_best_over_g_mm", "g_at_best", "g_required_for_a_plot",
              "H1_a_floor_g8_mm", "H1_rel_diff_pct", "a_min_over_g_mm"], rows)

    absr = np.abs(rel)
    summ = dict(n_points=len(rpm),
                n_fig8_vertex=int(is_vertex.sum()),
                within_1pct=int((absr <= 0.01).sum()), within_2pct=int((absr <= 0.02).sum()),
                within_5pct=int((absr <= 0.05).sum()), above_5pct=int((absr > 0.05).sum()),
                median_abs_rel_pct=np.median(absr) * 100,
                median_abs_rel_pct_z145=np.median(np.abs(rel145)) * 100,
                within_2pct_z145=int((np.abs(rel145) <= 0.02).sum()))
    # bands where the plotted depth differs from a_zoa(g8) by more than 2 %
    def ex(i, j):
        k = slice(i, j + 1)
        m = np.argmax(np.abs(rel[k])) + i
        return dict(max_rel_pct=rel[m] * 100, at_rpm=rpm[m], a_plot=a9[m], a_zoa=z["a_zoa"][m],
                    g8=g8[m], g8_interpolated_points=int((~is_vertex[k]).sum()),
                    plot_minus_zoa_mm_range=[float((a9[k] - z["a_zoa"][k]).min()),
                                             float((a9[k] - z["a_zoa"][k]).max())])
    summ["bands_rel_gt_2pct"] = bands(rpm, absr > 0.02, ex)

    # controlled below uncontrolled
    def ex2(i, j):
        k = slice(i, j + 1)
        d = a9[k] - a_unc_plot[k]
        m = np.argmin(d) + i
        return dict(max_deficit_mm=float(-d.min()), at_rpm=rpm[m], a_plot=a9[m], a_unc_plot=a_unc_plot[m],
                    a_zoa_g8=z["a_zoa"][m], a_zoa_g0=z["a0"][m])
    below_plot = a9 < a_unc_plot
    summ["bands_controlled_below_uncontrolled_plotted"] = bands(rpm, below_plot, ex2)
    below_model = z["a_zoa"] < z["a0"]
    summ["bands_model_azoa_g8_below_g0"] = bands(rpm, below_model, lambda i, j: dict(
        max_deficit_mm=float((z["a0"][i:j + 1] - z["a_zoa"][i:j + 1]).max())))
    # where can no gain in [0, 2500] reach the plotted depth?
    summ["bands_plot_above_best_any_gain"] = bands(rpm, a9 > a_best * 1.01, lambda i, j: dict(
        max_excess_pct=float(((a9[i:j + 1] / a_best[i:j + 1]) - 1).max() * 100)))
    summ["bands_plot_below_min_any_gain"] = bands(rpm, a9 < a_min_g * 0.99, lambda i, j: dict(
        max_deficit_pct=float((1 - a9[i:j + 1] / a_min_g[i:j + 1]).max() * 100)))
    summ["crossings_controlled_vs_uncontrolled_plotted_rpm"] = crossings(r9, a9 - a_unc_plot)
    # dense crossing estimate on the uncontrolled polyline itself (controlled interpolated onto it)
    m = (xu >= r9.min()) & (xu <= r9.max())
    summ["crossings_dense_rpm"] = crossings(xu[m], np.interp(xu[m], r9, a9) - au[m])
    ar = np.abs(rel_floor)
    summ["H1_lobe_floor"] = dict(
        note="hypothesis test, not a claim about the authors' method",
        median_abs_rel_pct=np.median(ar) * 100, within_2pct=int((ar <= 0.02).sum()),
        within_5pct=int((ar <= 0.05).sum()),
        by_band={f"{lo}-{hi}": dict(median_rel_pct=float(np.median(rel_floor[(rpm >= lo) & (rpm <= hi)]) * 100),
                                    min_rel_pct=float(rel_floor[(rpm >= lo) & (rpm <= hi)].min() * 100),
                                    max_rel_pct=float(rel_floor[(rpm >= lo) & (rpm <= hi)].max() * 100))
                 for lo, hi in ((500, 1150), (1160, 1350), (1360, 1490), (1500, 2050), (2060, 3000))})
    summ["by_band_rel_to_azoa_g8"] = {
        f"{lo}-{hi}": dict(median_rel_pct=float(np.median(rel[(rpm >= lo) & (rpm <= hi)]) * 100),
                           min_rel_pct=float(rel[(rpm >= lo) & (rpm <= hi)].min() * 100),
                           max_rel_pct=float(rel[(rpm >= lo) & (rpm <= hi)].max() * 100))
        for lo, hi in ((500, 1150), (1160, 1350), (1360, 1490), (1500, 2050), (2060, 3000))}
    # the uncontrolled plotted curve against the model (sanity of the comparison basis)
    ru = np.abs(a_unc_plot / z["a0"] - 1)
    summ["uncontrolled_plot_vs_model_g0_median_abs_rel_pct"] = np.median(ru) * 100
    save_json("c_fig9_vs_fig8.json", summ)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(rpm, a9, "C0-", lw=1.5, label="Fig 9 controlled (plotted, incl. covered part)")
    ax[0].plot(rpm, z["a_zoa"], "C3--", lw=1, label="ZOA a_lim at Fig 8 gain (zeta 1.34 %)")
    ax[0].plot(xu, au, "C0:", lw=1, label="Fig 9 uncontrolled (plotted)")
    ax[0].plot(rpm, a_best, color="0.6", lw=0.8, label="max over g of a_lim (no force limit)")
    ax[0].plot(rpm, afl, "C2-.", lw=0.8, label="H1: lobe floor a_floor(g8) (hypothesis)")
    ax[0].set_ylim(0, 32)
    ax[0].set_ylabel("depth (mm)")
    ax[0].axvspan(1495, 2051, color="0.9", zorder=0, label="covered by white shapes in PDF")
    ax[0].legend(fontsize=7)
    ax[1].plot(rpm, rel * 100, "k.-", ms=3, lw=0.6)
    ax[1].plot(rpm[~is_vertex], rel[~is_vertex] * 100, "o", mfc="none", color="C1", ms=5,
               label="Fig 8 value interpolated (not a vertex)")
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_ylabel("a_plot / a_zoa(g8) - 1 (%)")
    ax[1].legend(fontsize=7)
    ax[2].plot(x8, y8, "C0-", lw=1, label="Fig 8 gain")
    ax[2].plot(rpm, g_req, "C3.", ms=3, label="smallest g with a_lim >= plotted depth")
    ax[2].set_ylabel("gain (V s/m)")
    ax[2].set_xlabel("spindle speed (rpm)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{HERE}/c_fig9_vs_fig8.png", dpi=150)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
