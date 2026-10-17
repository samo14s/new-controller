"""(e) Fig. 8: speed intervals where the plotted gain is an exact straight line.

The Fig. 8 polyline is read from the PDF (214 vertices; the optimisation grid
is 500:10:3000 rpm = 251 speeds).  A run is a maximal set of consecutive
vertices that all lie within TOL of the chord joining its first and last
vertex.  For every run longer than 30 rpm the script reports its end points,
slope, number of 10-rpm grid speeds inside it, number of those speeds that
carry a polyline vertex, and the largest deviation from the chord.

Each run is then related to
  * the tooth-passing resonance speeds  rpm_k = 60 fn / (N k), fn = 129.3 Hz;
  * the lobe peaks of the plotted uncontrolled SLD (Fig. 9, dashed);
  * the white overlay rectangles of Fig. 9 (see d_white_shapes.json).
The same run detection is applied to the Fig. 9 controlled curve.

Outputs: e_fig8_linear_runs.json, e_fig8_linear_runs.csv, e_fig8_linear_runs.png
"""
import json
import os

import numpy as np

from common import fig8, fig9_controlled, fig9_uncontrolled, save_csv, save_json, HERE
import params as P
import extract_vector_figs as evf

TOL = 0.05            # V s/m: exact collinearity (tick-fit error of the axis map is 0.065)
MIN_SPAN = 20.0       # rpm


def runs(x, y, tol):
    out = []
    i = 0
    n = len(x)
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            k = j + 1
            xs, ys = x[i:k + 1], y[i:k + 1]
            chord = y[i] + (y[k] - y[i]) * (xs - x[i]) / (x[k] - x[i])
            if np.abs(ys - chord).max() > tol:
                break
            j = k
        xs, ys = x[i:j + 1], y[i:j + 1]
        chord = y[i] + (y[j] - y[i]) * (xs - x[i]) / (x[j] - x[i])
        out.append(dict(i0=i, i1=j, rpm_from=x[i], rpm_to=x[j], g_from=y[i], g_to=y[j],
                        span=x[j] - x[i], slope=(y[j] - y[i]) / (x[j] - x[i]),
                        n_vertices=j - i + 1, max_dev=float(np.abs(ys - chord).max())))
        i = j
    return out


def local_maxima(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    k = np.argmax(ys)
    return xs[k], ys[k]


def main():
    x8, y8 = fig8()
    r8 = np.round(x8 / 10) * 10
    tooth = {k: 60 * P.FN / (P.N_TEETH * k) for k in range(1, 6)}
    xu, au = fig9_uncontrolled()
    peaks = [local_maxima(xu, au, lo, hi) for lo, hi in ((450, 520), (600, 700), (900, 1050), (1850, 2050))]

    dpath = os.path.join(HERE, "d_white_shapes.json")
    covers = []
    if os.path.exists(dpath):
        with open(dpath) as fh:
            dd = json.load(fh)
        covers = [(e["seqno"], e["rpm"]) for e in dd["white_fills"] if e["role"] == "overlay rectangle"
                  and e["rpm"][0] < 2300]

    R = runs(r8, y8, TOL)
    long_ = []
    for r in R:
        if r["span"] < MIN_SPAN:
            continue
        grid = np.arange(r["rpm_from"], r["rpm_to"] + 1, 10)
        r["grid_speeds"] = len(grid)
        r["grid_speeds_with_vertex"] = int(sum(np.any(np.abs(r8 - g) < 1) for g in grid))
        r["missing_grid_speeds"] = [float(g) for g in grid if not np.any(np.abs(r8 - g) < 1)]
        if not r["missing_grid_speeds"] and r["span"] < 100:
            continue
        r["kind"] = "constant" if abs(r["slope"]) * r["span"] < 2 * TOL else "sloped straight line"
        kn = min(tooth, key=lambda k: min(abs(tooth[k] - r["rpm_from"]), abs(tooth[k] - r["rpm_to"]))
                 if not (r["rpm_from"] <= tooth[k] <= r["rpm_to"]) else -1)
        r["nearest_tooth_passing"] = dict(k=kn, rpm=tooth[kn],
                                          inside=bool(r["rpm_from"] <= tooth[kn] <= r["rpm_to"]))
        mu = (xu >= r["rpm_from"]) & (xu <= r["rpm_to"])
        r["uncontrolled_plot_depth_range_mm"] = [float(au[mu].min()), float(au[mu].max())]
        near = []
        for k, rk in tooth.items():
            if r["rpm_from"] - 30 <= rk <= r["rpm_to"] + 30:
                near.append(f"tooth-passing k={k} at {rk:.1f} rpm")
        for px, py in peaks:
            if r["rpm_from"] - 30 <= px <= r["rpm_to"] + 30:
                near.append(f"uncontrolled lobe peak {py:.1f} mm at {px:.0f} rpm (Fig 9 dashed)")
        for sq, rr in covers:
            if rr[1] >= r["rpm_from"] and rr[0] <= r["rpm_to"]:
                near.append(f"overlaps Fig 9 white rectangle seqno {sq} ({rr[0]:.0f}-{rr[1]:.0f} rpm)")
        r["related"] = near
        long_.append(r)

    # Fig 9 controlled: same detection, for the speeds of the long sloped Fig 8 runs
    x9, y9 = fig9_controlled()
    R9 = runs(np.round(x9 / 10) * 10, y9, 0.02)   # mm; axis fit error 0.0009 mm
    for r in long_:
        inside = [q for q in R9 if q["rpm_to"] > r["rpm_from"] and q["rpm_from"] < r["rpm_to"]]
        r["fig9_runs_inside"] = len(inside)
        r["fig9_longest_straight_inside_rpm"] = max(q["span"] for q in inside) if inside else 0

    allgrid = np.arange(500, 3001, 10)
    rep = dict(tolerance_Vs_per_m=TOL, n_vertices=len(x8), grid_speeds=251,
               speeds_without_vertex=int(251 - len(np.unique(r8))),
               missing_speed_list=[float(g) for g in allgrid if not np.any(np.abs(r8 - g) < 1)],
               tooth_passing_speeds=tooth, uncontrolled_lobe_peaks=peaks,
               fig8_min=dict(rpm=float(r8[np.argmin(y8)]), g=float(y8.min())),
               fig8_max=dict(rpm=float(r8[np.argmax(y8)]), g=float(y8.max())),
               long_runs=long_)
    save_json("e_fig8_linear_runs.json", rep)
    save_csv("e_fig8_linear_runs.csv",
             ["rpm_from", "rpm_to", "span_rpm", "g_from", "g_to", "slope_per_rpm", "kind", "n_vertices",
              "grid_speeds", "grid_speeds_with_vertex", "max_dev_from_chord", "nearest_tooth_k",
              "nearest_tooth_rpm", "unc_depth_min_mm", "unc_depth_max_mm", "related"],
             [(r["rpm_from"], r["rpm_to"], r["span"], r["g_from"], r["g_to"], r["slope"], r["kind"],
               r["n_vertices"], r["grid_speeds"], r["grid_speeds_with_vertex"], r["max_dev"],
               r["nearest_tooth_passing"]["k"], r["nearest_tooth_passing"]["rpm"],
               r["uncontrolled_plot_depth_range_mm"][0], r["uncontrolled_plot_depth_range_mm"][1],
               " | ".join(r["related"])) for r in long_])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(x8, y8, "C0.-", ms=3, lw=0.8, label="Fig 8 vertices")
    for r in long_:
        col = "C3" if r["kind"] != "constant" else "C2"
        ax[0].plot([r["rpm_from"], r["rpm_to"]], [r["g_from"], r["g_to"]], "-", color=col, lw=3, alpha=0.4)
    for k, rk in tooth.items():
        if 480 <= rk <= 3000:
            for a in ax:
                a.axvline(rk, color="0.4", ls=":", lw=0.8)
            ax[0].text(rk, 1550, f"k={k}", fontsize=7, ha="center")
    ax[0].set_ylabel("gain (V s/m)")
    ax[0].set_title("red: straight sloped runs, green: constant runs (within %.2f V s/m)" % TOL, fontsize=9)
    ax[1].plot(x9, y9, "C0-", lw=1, label="Fig 9 controlled")
    ax[1].plot(xu, au, "C0:", lw=1, label="Fig 9 uncontrolled")
    for sq, rr in covers:
        ax[1].axvspan(rr[0], rr[1], color="0.85", zorder=0)
    ax[1].set_ylim(0, 31)
    ax[1].set_xlabel("spindle speed (rpm); dotted verticals: tooth-passing speeds 60 fn/(N k)")
    ax[1].set_ylabel("depth (mm)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{HERE}/e_fig8_linear_runs.png", dpi=150)

    print("missing", rep["missing_speed_list"])
    print("speeds without vertex", rep["speeds_without_vertex"], "min", rep["fig8_min"], "max", rep["fig8_max"])
    print("tooth", {k: round(v, 1) for k, v in tooth.items()}, "peaks", [(round(a), round(b, 2)) for a, b in peaks])
    for r in long_:
        print("%6.0f-%6.0f (%4.0f rpm) g %7.1f -> %7.1f slope %+.3f %-21s vtx %2d/%2d grid  maxdev %.3f  fig9 runs %d (longest %d) unc %.1f-%.1f mm k%d %s | %s" % (
            r["rpm_from"], r["rpm_to"], r["span"], r["g_from"], r["g_to"], r["slope"], r["kind"],
            r["grid_speeds_with_vertex"], r["grid_speeds"], r["max_dev"], r["fig9_runs_inside"],
            r["fig9_longest_straight_inside_rpm"], *r["uncontrolled_plot_depth_range_mm"],
            r["nearest_tooth_passing"]["k"], r["nearest_tooth_passing"]["inside"], "; ".join(r["related"])))


if __name__ == "__main__":
    main()
