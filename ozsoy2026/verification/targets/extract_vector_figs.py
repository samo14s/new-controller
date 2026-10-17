"""Recover numerical targets from the vector figures of Ozsoy, Sims, Ozturk (2026).

Figures handled (all stored as vector paths in the published PDF):
  Fig. 6(b)  pole map            -> fig6b_poles.csv
  Fig. 8     optimised gain map  -> fig8_gain_map.csv, fig8_segments.csv
  Fig. 9     SLD + test points   -> fig9_markers.csv, fig9_curves.csv, fig9_masks.csv,
                                    fig9_features.json

Axis maps are linear fits through the tick-mark positions (PDF points).  The tick
positions are re-detected from the drawing list and compared with the nominal
values passed by the orchestrator; the fit residuals are written to
vector_axis_check.json.

Usage:  python3 extract_vector_figs.py [paper.pdf]
"""
import csv
import json
import os
import sys

import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-0/-home-user-new-controller/d42dfb76-3818-53d8-8bcc-e62ff0a55916/scratchpad/paper.pdf")


# ----------------------------------------------------------------- helpers
def near(c, ref, tol=0.02):
    return c is not None and max(abs(a - b) for a, b in zip(c, ref)) <= tol


def is_grey(c, tol=0.01):
    return c is not None and max(c) - min(c) < tol


def linfit(pos, val):
    pos, val = np.asarray(pos, float), np.asarray(val, float)
    A = np.vstack([np.ones_like(pos), pos]).T
    p = np.linalg.lstsq(A, val, rcond=None)[0]
    return p, float(np.abs(A @ p - val).max())


def ticks_on_edge(drawings, clip, edge_x=None, edge_y=None, maxlen=3.0):
    """Short grey/black tick strokes touching the left edge (edge_x) or bottom edge (edge_y)."""
    xs, ys = set(), set()
    for d in drawings:
        if d["type"] != "s" or len(d["items"]) != 1 or d["items"][0][0] != "l":
            continue
        c = d.get("color")
        if c is None or not is_grey(c) or c[0] > 0.3:
            continue
        p0, p1 = d["items"][0][1], d["items"][0][2]
        if not (clip[0] <= min(p0.x, p1.x) <= clip[2] and clip[1] <= min(p0.y, p1.y) <= clip[3]):
            continue
        if edge_y is not None and abs(p0.x - p1.x) < 1e-3 and abs(p0.y - p1.y) < maxlen \
                and abs(max(p0.y, p1.y) - edge_y) < 0.05:
            xs.add(round(p0.x, 2))
        if edge_x is not None and abs(p0.y - p1.y) < 1e-3 and abs(p0.x - p1.x) < maxlen \
                and abs(min(p0.x, p1.x) - edge_x) < 0.05:
            ys.add(round(p0.y, 2))
    return sorted(xs), sorted(ys)


def path_points(d):
    pts = [(d["items"][0][1].x, d["items"][0][1].y)]
    for it in d["items"]:
        if it[0] == "l":
            pts.append((it[2].x, it[2].y))
        elif it[0] == "c":
            pts.append((it[4].x, it[4].y))
    return np.array(pts)


def wcsv(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


doc = pymupdf.open(PDF)
axis_report = {}

# ================================================================ Fig. 9
page = doc[7]                      # journal p.253
dr = page.get_drawings()
CLIP9 = (50, 430, 300, 600)
X_NOM = [66.34, 109.29, 152.25, 195.21, 238.16, 281.12]
Y_NOM = [573.7, 551.61, 529.53, 507.45, 485.36, 463.28, 441.2]
xt, yt = ticks_on_edge(dr, CLIP9, edge_x=66.34 - 12.89, edge_y=573.7)
# the axes box of Fig. 9 starts at x=53.45 (white background rect); ticks hang off y=573.7
px9, ex9 = linfit(X_NOM, [500, 1000, 1500, 2000, 2500, 3000])
py9, ey9 = linfit(Y_NOM, [0, 5, 10, 15, 20, 25, 30])
axis_report["fig9"] = dict(x_ticks_nominal=X_NOM, x_ticks_detected=xt, y_ticks_nominal=Y_NOM,
                           y_ticks_detected=yt, x_fit_resid_rpm=ex9, y_fit_resid_mm=ey9,
                           rpm_per_pt=px9[1], mm_per_pt=-py9[1])


def to9(x, y):
    return px9[0] + px9[1] * x, py9[0] + py9[1] * y


in9 = lambda r: CLIP9[0] <= r.x0 and r.x1 <= CLIP9[2] and CLIP9[1] <= r.y0 and r.y1 <= CLIP9[3]
LEGEND_Y = 462.0                   # anything above this (smaller y) is in the legend box

markers = []
# controlled stable: green rings (filled annulus, 8 Bezier items)
for i, d in enumerate(dr):
    r = d["rect"]
    if not in9(r) or r.y1 < LEGEND_Y:
        continue
    if d["type"] == "f" and near(d.get("fill"), (0, 1, 0)) and len(d["items"]) == 8:
        markers.append(("controlled_stable", "green circle", i, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2,
                        r.x1 - r.x0))
# controlled chatter: red crosses (two diagonal strokes with the same bbox)
red = {}
for i, d in enumerate(dr):
    r = d["rect"]
    if in9(r) and r.y1 > LEGEND_Y and d["type"] == "s" and near(d.get("color"), (1, 0, 0)):
        red.setdefault((round(r.x0, 2), round(r.y0, 2)), []).append((i, r))
for k, v in red.items():
    i, r = v[0]
    markers.append(("controlled_chatter", "red x (%d strokes)" % len(v), i, (r.x0 + r.x1) / 2,
                    (r.y0 + r.y1) / 2, r.x1 - r.x0))
# uncontrolled stable: black '+' = one vertical + one horizontal stroke sharing a centre
blk = [(i, d) for i, d in enumerate(dr) if in9(d["rect"]) and d["rect"].y1 > LEGEND_Y and d["type"] == "s"
       and d.get("color") is not None and max(d["color"]) < 0.01 and d["items"][0][0] == "l"]
vert = [(i, d["rect"]) for i, d in blk if d["rect"].width < 0.01]
hori = [(i, d["rect"]) for i, d in blk if d["rect"].height < 0.01]
for iv, rv in vert:
    for ih, rh in hori:
        if abs(rv.x0 - (rh.x0 + rh.x1) / 2) < 0.1 and abs(rh.y0 - (rv.y0 + rv.y1) / 2) < 0.1:
            markers.append(("uncontrolled_stable", "black +", iv, rv.x0, rh.y0, rh.x1 - rh.x0))
# uncontrolled chatter: diamonds ('qu' items; one is black, one is dark grey 0.149)
for i, d in enumerate(dr):
    r = d["rect"]
    if in9(r) and r.y1 > LEGEND_Y and d["items"][0][0] == "qu":
        markers.append(("uncontrolled_chatter", "diamond (stroke %s)" % str(tuple(round(c, 3) for c in d["color"])),
                        i, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.x1 - r.x0))

# text labels A / B: one filled glyph path (black) -> split into letters by vertical gap
labels = []
for i, d in enumerate(dr):
    r = d["rect"]
    if in9(r) and r.y1 > LEGEND_Y and d["type"] == "f" and near(d.get("fill"), (0, 0, 0)) and r.width < 6:
        pts = path_points(d)
        ys = np.sort(pts[:, 1])
        gap = np.argmax(np.diff(ys))
        cut = (ys[gap] + ys[gap + 1]) / 2
        for sel, name in ((pts[:, 1] < cut, "upper"), (pts[:, 1] >= cut, "lower")):
            q = pts[sel]
            cx, cy = (q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2
            labels.append(dict(path=i, glyph_block=name, x_pt=cx, y_pt=cy,
                               n_vertices=int(sel.sum()), rpm=to9(cx, cy)[0], depth_mm=to9(cx, cy)[1]))
# the upper glyph has Bezier bowls -> 'B', the lower one is straight strokes -> 'A' (checked visually)
for lb in labels:
    lb["letter"] = "B" if lb["glyph_block"] == "upper" else "A"

rows = []
for kind, desc, idx, cx, cy, size in sorted(markers, key=lambda m: (m[0], m[3], m[4])):
    rpm, a = to9(cx, cy)
    lab = ""
    for lb in labels:   # a label sits just left of the marker it names
        if abs(lb["y_pt"] - cy) < 2.5 and 0 < cx - lb["x_pt"] < 8:
            lab = lb["letter"]
    rows.append([kind, desc, lab, round(rpm, 2), round(a, 3), round(abs(px9[1]) * size / 2, 1),
                 round(abs(py9[1]) * size / 2, 2), round(cx, 3), round(cy, 3), idx])
wcsv("fig9_markers.csv",
     ["kind", "glyph", "label", "rpm", "depth_mm", "marker_halfwidth_rpm", "marker_halfheight_mm",
      "x_pt", "y_pt", "drawing_index"], rows)

# white rectangles painted over the plot after the curves
masks = []
for i, d in enumerate(dr):
    r = d["rect"]
    if in9(r) and d["type"] == "fs" and near(d.get("fill"), (1, 1, 1)) and d["items"][0][0] == "re":
        r0, a1 = to9(r.x0, r.y1)
        r1, a0 = to9(r.x1, r.y0)
        masks.append([i, round(r0, 1), round(r1, 1), round(a1, 2), round(a0, 2),
                      "legend" if r.y0 < LEGEND_Y else "plot"])
wcsv("fig9_masks.csv", ["drawing_index", "rpm_from", "rpm_to", "depth_from_mm", "depth_to_mm", "where"], masks)


def masked(rpm, a):
    return any(m[5] == "plot" and m[1] <= rpm <= m[2] and m[3] <= a <= m[4] for m in masks)


curves = {}
for i, d in enumerate(dr):
    r = d["rect"]
    if in9(r) and d["type"] == "s" and near(d.get("color"), (0, 0.447, 0.74)) and len(d["items"]) > 100:
        dashed = d.get("dashes", "[] 0").strip() not in ("[] 0", "")
        P = path_points(d)
        R, A = to9(P[:, 0], P[:, 1])
        curves["uncontrolled" if dashed else "controlled"] = (R, A)
crow = []
for name in ("controlled", "uncontrolled"):
    R, A = curves[name]
    for x, y in zip(R, A):
        crow.append([name, round(x, 2), round(y, 4), int(not masked(x, y))])
wcsv("fig9_curves.csv", ["curve", "rpm", "depth_mm", "visible_in_print"], crow)

# ---------------------------------------------------------- Fig. 9 features
Rc, Ac = curves["controlled"]
Ru, Au = curves["uncontrolled"]


def interp(R, A, x):
    o = np.argsort(R)
    return float(np.interp(x, R[o], A[o]))


def extrema(R, A, lo, hi, kind):
    m = (R >= lo) & (R <= hi)
    j = np.argmin(A[m]) if kind == "min" else np.argmax(A[m])
    return dict(rpm=round(float(R[m][j]), 1), depth_mm=round(float(A[m][j]), 3))


feat = {}
feat["uncontrolled_minima"] = [extrema(Ru, Au, lo, hi, "min") for lo, hi in
                               ((389, 470), (470, 600), (600, 900), (1000, 1500), (2100, 3050))]
feat["uncontrolled_peaks"] = [extrema(Ru, Au, lo, hi, "max") for lo, hi in
                              ((420, 520), (520, 700), (900, 1050), (1500, 2050))]
# where the dashed curve leaves the top of the axes (30 mm)
top = Ru[(Au >= 29.0)]
feat["uncontrolled_clipped_at_top_rpm_range"] = [round(float(top.min()), 1), round(float(top.max()), 1)] if len(top) else None
feat["controlled_rpm_range"] = [round(float(Rc.min()), 1), round(float(Rc.max()), 1)]
feat["controlled_rpm_step"] = round(float(np.median(np.diff(Rc))), 3)
feat["controlled_max"] = extrema(Rc, Ac, 500, 3000, "max")
feat["controlled_min_all"] = extrema(Rc, Ac, 500, 3000, "min")
vis = np.array([not masked(x, y) for x, y in zip(Rc, Ac)])
feat["controlled_min_visible"] = extrema(Rc[vis], Ac[vis], 500, 3000, "min")
hid = Rc[~vis]
runs, start = [], None
for k in range(len(Rc)):
    if not vis[k] and start is None:
        start = k
    if start is not None and (vis[k] or k == len(Rc) - 1):
        end = k - 1 if vis[k] else k
        runs.append([round(float(Rc[start]), 1), round(float(Rc[end]), 1)])
        start = None
feat["controlled_hidden_rpm_ranges"] = runs
feat["controlled_hidden_min"] = extrema(Rc[~vis], Ac[~vis], 0, 4000, "min") if len(hid) else None
# controlled / uncontrolled at the speeds the paper discusses
speeds = [500, 560, 600, 700, 730, 760, 800, 1200, 1300, 1950, 2300, 2500, 2700, 2800, 3000]
feat["at_speeds"] = [dict(rpm=s, controlled_mm=round(interp(Rc, Ac, s), 3),
                          uncontrolled_mm=round(interp(Ru, Au, s), 3),
                          ratio=round(interp(Rc, Ac, s) / interp(Ru, Au, s), 2),
                          controlled_visible=bool(not masked(s, interp(Rc, Ac, s)))) for s in speeds]
grid = np.arange(500, 3001, 10.0)
ratio = np.array([interp(Rc, Ac, s) / interp(Ru, Au, s) for s in grid])
j = int(np.argmax(ratio))
feat["max_ratio"] = dict(rpm=float(grid[j]), ratio=round(float(ratio[j]), 2),
                         controlled_mm=round(interp(Rc, Ac, grid[j]), 3), uncontrolled_mm=round(interp(Ru, Au, grid[j]), 3))
below = grid[ratio < 1]
feat["controlled_below_uncontrolled_rpm"] = [float(s) for s in below]
with open(os.path.join(HERE, "fig9_features.json"), "w") as fh:
    json.dump(dict(labels=labels, features=feat), fh, indent=1)

# ================================================================ Fig. 6(b)
page = doc[6]                      # journal p.252
dr6 = page.get_drawings()
pxr, exr = linfit([333.8, 384.3, 434.8, 485.3], [-150, -100, -50, 0])
pyi, eyi = linfit([246.63, 217.39, 188.15, 158.9, 129.66, 100.42, 71.18],
                  [-1500, -1000, -500, 0, 500, 1000, 1500])
axis_report["fig6b"] = dict(x_fit_resid=exr, y_fit_resid=eyi)
COLS = {"open_loop": (0, 0, 0), "500rpm_g1487": (0.071, 0.439, 0.748), "1300rpm_g1162": (0.231, 0.67, 0.2),
        "2500rpm_g585": (0.93, 0.69, 0.129), "3000rpm_g797": (0.518, 0.09, 0.82)}
prow = []
for name, col in COLS.items():
    cents = {}
    for d in dr6:
        r = d["rect"]
        if not (333 < r.x0 and r.x1 < 486 and 60 < r.y0 and r.y1 < 250):
            continue
        c = d.get("color") if d["type"] == "s" else d.get("fill")
        if not near(c, col, 0.01):
            continue
        if name == "open_loop" and d["items"][0][0] != "c":
            continue
        if name != "open_loop" and not (d["type"] == "s" and d["items"][0][0] == "l"):
            continue
        if r.width > 3 or r.height > 3:
            continue
        key = (round((r.x0 + r.x1) / 2, 1), round((r.y0 + r.y1) / 2, 1))
        cents[key] = cents.get(key, 0) + 1
    for (cx, cy), n in sorted(cents.items()):
        sig = pxr[0] + pxr[1] * cx
        om = pyi[0] + pyi[1] * cy
        # legend glyphs: one column at x ~ 451 pt inside the upper-right legend box (y < 127 pt)
        if abs(cx - 451.2) < 1.5 and cy < 127:
            continue
        zeta = -sig / np.hypot(sig, om)
        prow.append([name, round(sig, 2), round(om, 1), round(np.hypot(sig, om) / 2 / np.pi, 2),
                     round(100 * zeta, 3), n])
wcsv("fig6b_poles.csv", ["case", "real_rad_s", "imag_rad_s", "wn_Hz", "zeta_percent", "n_copies"], prow)

# ================================================================ Fig. 8
dr8 = dr  # same page as Fig. 9
page = doc[7]
CLIP8 = (30, 255, 300, 400)
xt8, yt8 = ticks_on_edge(dr, CLIP8, edge_x=None, edge_y=None)
gain_path = None
for d in dr:
    r = d["rect"]
    if CLIP8[0] <= r.x0 and r.x1 <= CLIP8[2] and CLIP8[1] <= r.y0 and r.y1 <= CLIP8[3] \
            and d["type"] == "s" and near(d.get("color"), (0, 0.447, 0.74)) and len(d["items"]) > 50:
        gain_path = d
# axes of Fig. 8: detect frame ticks generically (grey strokes, length < 3 pt) and fit
segs = []
for d in dr:
    r = d["rect"]
    if not (CLIP8[0] <= r.x0 and r.x1 <= CLIP8[2] and CLIP8[1] <= r.y0 and r.y1 <= CLIP8[3]):
        continue
    if d["type"] == "s" and len(d["items"]) == 1 and d["items"][0][0] == "l" and is_grey(d.get("color")) \
            and d["color"][0] < 0.3:
        p0, p1 = d["items"][0][1], d["items"][0][2]
        segs.append((p0.x, p0.y, p1.x, p1.y))
hor = [s for s in segs if abs(s[1] - s[3]) < 1e-3]
ver = [s for s in segs if abs(s[0] - s[2]) < 1e-3]
frame = max(hor, key=lambda s: abs(s[2] - s[0]))
ybot = max(s[1] for s in hor if abs(s[2] - s[0]) > 0.9 * abs(frame[2] - frame[0]))
xleft = min(min(s[0], s[2]) for s in hor if abs(s[2] - s[0]) > 0.9 * abs(frame[2] - frame[0]))
xt8 = sorted({round(s[0], 2) for s in ver if abs(s[1] - s[3]) < 6 and abs(max(s[1], s[3]) - ybot) < 1e-2})
yt8 = sorted({round(s[1], 2) for s in hor if abs(s[2] - s[0]) < 6 and abs(min(s[0], s[2]) - xleft) < 1e-2})[::-1]
px8, ex8 = linfit(xt8, [500, 1000, 1500, 2000, 2500, 3000])
py8, ey8 = linfit(yt8, list(range(200, 1601, 200)))
axis_report["fig8"] = dict(x_ticks=xt8, y_ticks=yt8, x_fit_resid_rpm=ex8, y_fit_resid=ey8)
P = path_points(gain_path)
G_r = px8[0] + px8[1] * P[:, 0]
G_g = py8[0] + py8[1] * P[:, 1]
wcsv("fig8_gain_map.csv", ["rpm", "gain_Vs_per_m"], [[round(a, 2), round(b, 2)] for a, b in zip(G_r, G_g)])
# long straight segments = speeds with no plotted sample (MATLAB joins neighbours by a line)
dR = np.diff(G_r)
seg_rows = []
for k in np.where(dR > 15)[0]:
    seg_rows.append([round(G_r[k], 1), round(G_g[k], 1), round(G_r[k + 1], 1), round(G_g[k + 1], 1),
                     round(dR[k], 1)])
wcsv("fig8_segments.csv", ["rpm_from", "gain_from", "rpm_to", "gain_to", "span_rpm"], seg_rows)
slope = np.diff(G_g) / dR
stretches, k = [], 0
while k < len(slope):
    m = k
    while m + 1 < len(slope) and abs(slope[m + 1] - slope[k]) < 0.02 * max(1.0, abs(slope[k])):
        m += 1
    if G_r[m + 1] - G_r[k] >= 40:
        stretches.append(dict(rpm_from=round(float(G_r[k]), 1), gain_from=round(float(G_g[k]), 1),
                              rpm_to=round(float(G_r[m + 1]), 1), gain_to=round(float(G_g[m + 1]), 1),
                              slope_per_rpm=round(float(slope[k]), 3), n_vertices=int(m + 2 - k)))
    k = m + 1
f8 = dict(straight_stretches=stretches, n_vertices=len(G_r), median_step_rpm=round(float(np.median(dR)), 2),
          max=dict(rpm=round(float(G_r[np.argmax(G_g)]), 1), gain=round(float(G_g.max()), 1)),
          min=dict(rpm=round(float(G_r[np.argmin(G_g)]), 1), gain=round(float(G_g.min()), 1)),
          at_speeds={str(s): round(float(np.interp(s, G_r, G_g)), 1)
                     for s in (500, 560, 600, 700, 730, 760, 800, 1000, 1200, 1300, 1950, 2300, 2500, 2700, 2800, 3000)},
          long_segments=seg_rows)
with open(os.path.join(HERE, "fig8_features.json"), "w") as fh:
    json.dump(f8, fh, indent=1)

with open(os.path.join(HERE, "vector_axis_check.json"), "w") as fh:
    json.dump(axis_report, fh, indent=1, default=float)
print(json.dumps(axis_report, indent=1, default=float))
print(open(os.path.join(HERE, "fig9_markers.csv")).read())
print(open(os.path.join(HERE, "fig9_masks.csv")).read())
print(open(os.path.join(HERE, "fig6b_poles.csv")).read())
print(json.dumps(f8, indent=1))
print(json.dumps(feat, indent=1))
print(json.dumps(labels, indent=1))


# ================================================================ Figs. 5 and 6(a): FRF peaks
def frame_axes(drawings, clip):
    segs = []
    for d in drawings:
        r = d["rect"]
        if not (clip[0] <= r.x0 and r.x1 <= clip[2] and clip[1] <= r.y0 and r.y1 <= clip[3]):
            continue
        if d["type"] == "s" and len(d["items"]) == 1 and d["items"][0][0] == "l" and is_grey(d.get("color")) \
                and d["color"][0] < 0.3:
            p0, p1 = d["items"][0][1], d["items"][0][2]
            segs.append((p0.x, p0.y, p1.x, p1.y))
    hor = [s for s in segs if abs(s[1] - s[3]) < 1e-3]
    ver = [s for s in segs if abs(s[0] - s[2]) < 1e-3]
    fr = max(hor, key=lambda s: abs(s[2] - s[0]))
    wid = abs(fr[2] - fr[0])
    ybot = max(s[1] for s in hor if abs(s[2] - s[0]) > 0.9 * wid)
    xleft = min(min(s[0], s[2]) for s in hor if abs(s[2] - s[0]) > 0.9 * wid)
    xt = sorted({round(s[0], 2) for s in ver if abs(s[1] - s[3]) < 6 and abs(max(s[1], s[3]) - ybot) < 1e-2})
    yt = sorted({round(s[1], 2) for s in hor if abs(s[2] - s[0]) < 6 and abs(min(s[0], s[2]) - xleft) < 1e-2})[::-1]
    return xt, yt


FRF = {
    "fig5": dict(page=5, clip=(30, 205, 300, 340), xv=[50, 100, 150, 200],
                 yv=[0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6, 2.5e-6],
                 curves={"measured": (0, 0, 1), "identified": (1, 0, 0)}),
    "fig6a": dict(page=6, clip=(80, 60, 300, 250), xv=list(range(0, 301, 50)), yv=list(range(-155, -109, 5)),
                  curves={"uncontrolled": (0, 0, 0), "500rpm_g1487": (0.0706, 0.4392, 0.7480),
                          "1300rpm_g1162": (0.2314, 0.6699, 0.2000), "2500rpm_g585": (0.9297, 0.6895, 0.1294),
                          "3000rpm_g797": (0.5176, 0.0901, 0.8203)}),
}
frf_out = {}
for name, f in FRF.items():
    drf = doc[f["page"]].get_drawings()
    xt_, yt_ = frame_axes(drf, f["clip"])
    pxf, exf = linfit(xt_, f["xv"])
    pyf, eyf = linfit(yt_, f["yv"])
    axis_report[name] = dict(x_ticks=xt_, y_ticks=yt_, x_fit_resid=exf, y_fit_resid=eyf)
    res = {}
    for cname, col in f["curves"].items():
        pts = []
        for d in drf:
            r = d["rect"]
            cl = f["clip"]
            if not (cl[0] <= r.x0 and r.x1 <= cl[2] and cl[1] <= r.y0 and r.y1 <= cl[3]):
                continue
            if d["type"] != "s" or not near(d.get("color"), col, 0.02) or len(d["items"]) < 20:
                continue
            P = path_points(d)
            pts.append(np.c_[pxf[0] + pxf[1] * P[:, 0], pyf[0] + pyf[1] * P[:, 1]])
        if not pts:
            continue
        P = np.vstack(pts)
        P = P[np.argsort(P[:, 0])]
        j = int(np.argmax(P[:, 1]))
        e = dict(peak_freq_Hz=round(float(P[j, 0]), 2), peak_value=float(P[j, 1]), n_points=len(P))
        if name == "fig6a":
            e["value_at_129.3Hz_dB"] = round(float(np.interp(129.3, P[:, 0], P[:, 1])), 2)
            m = (P[:, 0] > 2) & (P[:, 0] < 20)
            if m.any():
                k = int(np.argmax(P[m, 1]))
                Q = P[m]
                if 0 < k < len(Q) - 1 and Q[k, 1] - Q[:, 1].min() > 0.1:   # interior local maximum only
                    e["actuator_bump"] = dict(freq_Hz=round(float(Q[k, 0]), 2), dB=round(float(Q[k, 1]), 2),
                                              rise_dB=round(float(Q[k, 1] - Q[-1, 1]), 2))
            e["peak_value"] = round(e["peak_value"], 2)
            # equivalent damping of a 1-DOF with k = 1.34e7 N/m having this peak: |G|max = 1/(2 zeta k)
            e["zeta_equiv_percent"] = round(100 / (2 * 1.34e7 * 10 ** (e["peak_value"] / 20)), 3)
        else:
            e["zeta_equiv_percent"] = round(100 / (2 * 1.34e7 * e["peak_value"]), 3)
        res[cname] = e
    frf_out[name] = res
with open(os.path.join(HERE, "frf_features.json"), "w") as fh:
    json.dump(frf_out, fh, indent=1)
with open(os.path.join(HERE, "vector_axis_check.json"), "w") as fh:
    json.dump(axis_report, fh, indent=1, default=float)
print(json.dumps(frf_out, indent=1))
