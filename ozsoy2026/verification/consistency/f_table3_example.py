"""(f) Table 3 'Proposed method' and the 1300 rpm worked example against Figs. 8/9.

Also (g) a few numeric statements of the text checked against the figures.

Numbers typed from the paper (Table 3; Sec. 2.1 Step 3; Sec. 3; Sec. 4):
  Table 3 speeds / clipped DVF / proposed method (mm)
  worked example: 1300 rpm, g = 1240 -> a_lim = 11.8 mm, max|F_d| = 29 N > 27 N

Outputs: f_table3_example.json, f_table3.csv
"""
import json

import numpy as np

from common import fig8, fig9_controlled, fig9_uncontrolled, save_csv, save_json
import params as P
import structure as S
import zoa
from objective import demand
from c_fig9_vs_fig8 import a_floor

T3_RPM = [500, 560, 600, 800, 1300, 2300, 2500, 2800, 3000]
T3_CLIPPED = [15.2, 8.7, 12.2, 11.2, 12.1, 4.5, 4.7, 5.8, 6.2]
T3_PROPOSED = [14.9, 8.0, 11.8, 10.8, 12.0, 4.9, 5.3, 6.6, 7.1]
HIDDEN = [(958.48, 975.42), (1495.4, 2050.8)]   # d_white_shapes.json, rpm covered by overlays


def at(x, y, r):
    """Value of a polyline at speed r; flags whether a vertex sits within 1 rpm."""
    return float(np.interp(r, x, y)), bool(np.min(np.abs(x - r)) < 1.0)


def main():
    x8, y8 = fig8()
    x9, y9 = fig9_controlled()
    xu, au = fig9_uncontrolled()
    o = np.argsort(xu, kind="stable")
    xu, au = xu[o], au[o]
    P.set_damping(0.0134)

    rows = []
    t3 = []
    for r, c, p in zip(T3_RPM, T3_CLIPPED, T3_PROPOSED):
        a9, v9 = at(x9, y9, r)
        g8, v8 = at(x8, y8, r)
        az = zoa.a_lim(r, g8) * 1e3
        fl = a_floor(g8)
        d = dict(rpm=r, table3_clipped=c, table3_proposed=p, fig9_controlled=a9, fig9_vertex=v9,
                 diff_table_minus_fig9=p - a9, rel_table_vs_fig9_pct=(p / a9 - 1) * 100,
                 fig8_gain=g8, fig8_vertex=v8, zoa_at_fig8_gain=az, H1_floor_at_fig8_gain=fl,
                 fig9_uncontrolled=float(np.interp(r, xu, au)))
        t3.append(d)
        rows.append((r, c, p, a9, p - a9, (p / a9 - 1) * 100, g8, az, fl, d["fig9_uncontrolled"]))
    save_csv("f_table3.csv", ["rpm", "table3_clipped_mm", "table3_proposed_mm", "fig9_controlled_mm",
                              "table_minus_fig9_mm", "rel_pct", "fig8_gain", "zoa_a_lim_at_fig8_gain_mm",
                              "H1_floor_at_fig8_gain_mm", "fig9_uncontrolled_mm"], rows)

    # ---- worked example ----------------------------------------------------
    ex = dict(paper=dict(rpm=1300, g=1240, a_lim_mm=11.8, Fd_N=29.0))
    g8_1300, _ = at(x8, y8, 1300)
    a9_1300, _ = at(x9, y9, 1300)
    ex["fig8_gain_1300"] = g8_1300
    ex["fig6_label_gain_1300"] = 1162
    ex["fig9_controlled_1300"] = a9_1300
    ex["table3_proposed_1300"] = 12.0
    ex["text_sec4_1300_mm"] = 10.0
    ex["experiment_1300"] = "Fig 10 / Fig 9 markers: 8 mm stable, 10 mm chatter"
    for z in (0.0134, 0.0145):
        P.set_damping(z)
        key = f"model_zeta_{z*100:.2f}%"
        m = {}
        for g in (1240.0, g8_1300, 1162.0):
            al = zoa.a_lim(1300, g)
            m[f"g{g:.0f}"] = dict(a_lim_mm=al * 1e3, H1_floor_mm=a_floor(g),
                                  demand_at_a_lim=dict(
                                      force_3g_xdot=demand(1300, al, g, "force"),
                                      volt_g_xdot=demand(1300, al, g, "volt"),
                                      actual_fact=demand(1300, al, g, "actual")),
                                  demand_at_11p8mm=dict(
                                      force_3g_xdot=demand(1300, 11.8e-3, g, "force"),
                                      volt_g_xdot=demand(1300, 11.8e-3, g, "volt"),
                                      actual_fact=demand(1300, 11.8e-3, g, "actual")))
        gs = np.arange(0, 2001, 50.0)
        al = np.array([zoa.a_lim(1300, g) for g in gs]) * 1e3
        m["a_lim_vs_g"] = dict(g=gs, a_mm=al, monotone_increasing=bool(np.all(np.diff(al) >= -1e-9)))
        # gain at which the model a_lim equals the paper's numbers
        m["g_for_11p8mm"] = float(np.interp(11.8, al, gs)) if np.all(np.diff(al) > 0) else None
        m["g_for_12p0mm"] = float(np.interp(12.0, al, gs)) if np.all(np.diff(al) > 0) else None
        m["g_for_fig9_value"] = float(np.interp(a9_1300, al, gs)) if np.all(np.diff(al) > 0) else None
        ex[key] = m
    P.set_damping(0.0134)

    # ---- (g) statements in the text --------------------------------------
    st = {}
    a730c, _ = at(x9, y9, 730)
    a730u = float(np.interp(730, xu, au))
    st["ratio_730rpm"] = dict(text="13 times near 730 rpm", controlled=a730c, uncontrolled=a730u,
                              ratio=a730c / a730u)
    unc9 = np.interp(x9, xu, au)
    ratio = y9 / unc9
    st["max_ratio_controlled_over_uncontrolled"] = dict(ratio=float(ratio.max()), rpm=float(x9[np.argmax(ratio)]),
                                                        controlled=float(y9[np.argmax(ratio)]),
                                                        uncontrolled=float(unc9[np.argmax(ratio)]))
    w = (xu >= 500) & (xu <= 3000)
    st["uncontrolled_min_500_3000"] = dict(mm=float(au[w].min()), rpm=float(xu[w][np.argmin(au[w])]))
    vis = np.ones_like(x9, bool)
    for lo, hi in HIDDEN:
        vis &= ~((x9 >= lo) & (x9 <= hi))
    st["controlled_min_all"] = dict(mm=float(y9.min()), rpm=float(x9[np.argmin(y9)]))
    st["controlled_min_visible_vertices"] = dict(mm=float(y9[vis].min()), rpm=float(x9[vis][np.argmin(y9[vis])]))
    # value of the controlled path where it emerges from the last overlay
    st["controlled_at_2050.8rpm"] = float(np.interp(2050.8, x9, y9))
    st["text_worst_case"] = "1.2 mm (uncontrolled) to 3.6 mm (DVF controlled)"
    st["gain_range_text"] = "237 to 1487 V s/m"
    st["fig8_gain_range"] = dict(min=float(y8.min()), at=float(x8[np.argmin(y8)]),
                                 max=float(y8.max()), at_max=float(x8[np.argmax(y8)]))
    st["fixed_gain_alternative_text"] = 237
    out = dict(table3=t3, worked_example=ex, text_statements=st)
    save_json("f_table3_example.json", out)

    for d in t3:
        print("%4d  T3 %.1f  Fig9 %.2f (vertex %s)  diff %+.2f mm (%+.1f %%)  g8 %.0f  zoa %.2f  floor %.2f" % (
            d["rpm"], d["table3_proposed"], d["fig9_controlled"], d["fig9_vertex"], d["diff_table_minus_fig9"],
            d["rel_table_vs_fig9_pct"], d["fig8_gain"], d["zoa_at_fig8_gain"], d["H1_floor_at_fig8_gain"]))
    print(json.dumps(ex, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))[:4000])
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
