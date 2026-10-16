#!/usr/bin/env python3
"""Digitize the effective-convergence-region maps (2-D colour panels) of
Ouyang et al. (2026) and test the "constant mu*L band" hypothesis.

Source article: Q. Ouyang, L. Dong, J. Liu, J. Zhou, "From data-driven DMDc
modeling to control parameter optimization: A framework for adaptive vibration
control of thin-walled composites", Thin-Walled Structures 228 (2026) 115090,
doi:10.1016/j.tws.2026.115090.

Panels digitized (raster images extracted from the PDF, copied to
data/source_figs/):
    Fig. 18 (c) Exp, (d) Sim   120 Hz, sensor S2, single channel  p15_img1.png
    Fig. 20 (c) Exp, (d) Sim   280 Hz, sensor S1, single channel  p16_img1.png
    Fig. 23 (c) Exp, (d) Sim    85 Hz, sensor S2, two channels    p18_img1.png
Colour = convergence time t (decay of the sensing-voltage amplitude to 5 %),
white = t > 5 s or divergent (Sec. 5 of the article).

What the panels are (established from the pixels, see checkmap_*.png):
  * a MATLAB surface on the tested (mu, L) grid with interpolated shading and
    the quad edges drawn -> the dark "mesh lines" ARE the tested grid
    (mu step 1e-6 in Figs. 18/20; 0.5e-6 for mu <= 10e-6 and 1e-6 above in
    Fig. 23; L in {..., 40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140,
    144, 150, 160});
  * the white/colour boundary does NOT pass through grid nodes: it cuts cells
    as straight segments and is lined by the darkest red (t -> 5 s).  It is the
    t = 5 s iso-line of the renderer's piecewise-linear interpolant of the node
    data (colours interpolated in data space, t > 5 mapped to white).  Along a
    grid row the interpolant is linear between two tested nodes, so the
    boundary position on a node row is the linear-interpolation crossing
    between the last node with t <= 5 and the first node with t > 5; from it
    the (hidden) value of the outside node can even be recovered:
        t_out = t_in + (5 - t_in) / s,  s = (mu_5 - mu_in)/(mu_out - mu_in).
    The authors clipped divergent/slow cases to 20 s, so t_out <= 20 is a
    consistency check.

Method (automatic once the per-figure configuration below is fixed):
  1. Axes frame: inside a rough box, the column/row with the most dark
     achromatic pixels (chroma < 40, grey < 160) is each frame line; sub-pixel
     centre = darkness-weighted centroid over +/-2 px.
  2. Ticks: MATLAB ticks point inward; on the bottom (left) axis a tick is a
     column (row) with >= 3 dark achromatic pixels 3..6 px inside the frame.
     The printed tick labels (typed in the configuration) are matched to the
     detected ticks by the contiguous label subset with the lowest linear-fit
     residual, constrained so that every printed label maps inside the axes;
     pixel -> data is the least-squares line.
  3. Colour bar: interior = columns with chroma > 80; colour vs row = median
     over the central columns; colour-bar ticks = dark marks on the inner right
     edge; the subset ambiguity is resolved with the vertical centres of the
     label text blobs.  Row -> value by least squares.
  4. Pixels: white (Delta-E76 to white < 20) = background; achromatic
     (chroma < 35) = text/ticks; the rest = region.  Region pixels get the value
     of the nearest colour-bar entry in CIE-Lab (k-d tree).  Mesh-line pixels
     (local brightness minimum: V < 0.9 x both region neighbours 3 px away,
     horizontally or vertically) and pixels with Delta-E > 12 are invalid and
     are filled with the nearest valid pixel's value.
  5. Mesh lines: columns (rows) where >= 30 % of the region pixels are
     vertical (horizontal) mesh-line pixels and >= 12 such pixels; positions
     rounded to the 0.5e-6 (mu) / 1-tap (L) lattice (raw values in maps_meta).
     The grid is extended to the axis limits with the end step (flagged).
  6. Nodes: value = mean of valid pixels within 4 px of the intersection;
     "in region" if the majority of the 3x3 pixels at the node are region.
  7. Boundary on every node row: first/last region pixel of that pixel row
     (censored if it touches the frame), plus node brackets and t_out.

Outputs (mu in units of 1e-6, L in taps, t in s), in data/digitized/:
  map_<fig>_<exp|sim>.csv           regular grid, mu step 0.1, L step 1,
                                    columns mu,L,t (empty t = NaN = white)
  map_<fig>_<exp|sim>_nodes.csv     tested grid nodes: mu,L,t,in_region,
                                    extended (1 = grid point not seen as a
                                    mesh line, inferred)
  map_<fig>_<exp|sim>_boundary.csv  per node row: pixel boundary mu_lo/mu_hi,
                                    censoring, node brackets, t_out, mu_opt,
                                    t_min, products mu*L
  map_<fig>_<exp|sim>_contour.csv   per pixel row: first/last region pixel
  checkmap_<fig>_<exp|sim>.png      overlay + re-render + colour residual
  checkmap_muL_summary.png          mu*L vs L for all panels
  maps_meta.json                    calibration & QA numbers
  maps_muL_analysis.json / .txt     hypothesis-test results

Hypothesis (H-muL): the effective region is a band of constant mu*L, i.e.
mu_lo(L)*L, mu_hi(L)*L and mu_opt(L)*L do not depend on L.  Measures: CV of
the products, Pearson r(L, mu*L), log-log slope b of mu vs L with 95 % CI
(b = -1 <=> mu*L const), and the alternative "delay law" mu*(L + D) = C
(classical FxLMS step bound mu < ~2/(P_x' (L + D)) with D = secondary-path
delay in samples), D fitted by least squares with a profile-F 95 % interval.

Run:  python3 digitize_maps.py      (paths are relative to this file)
Dependencies: numpy, scipy, matplotlib, Pillow.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage, stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "source_figs")
OUT_DIR = os.path.join(HERE, "digitized")

# ---------------------------------------------------------------------------
# Configuration: rough boxes (x0, y0, x1, y1) in image pixels (refined
# automatically), and the tick labels as printed in the figure.
# ---------------------------------------------------------------------------
CFG = {
    "fig18": dict(
        img="p15_img1.png", title="Fig. 18 (120 Hz, S2, single channel)",
        xt=[2, 4, 6, 8, 10, 12, 14], yt=[40, 60, 80, 100, 120, 140, 160],
        ct=[2, 2.5, 3, 3.5, 4, 4.5, 5],
        panels={
            "exp": dict(ax=(110, 640, 688, 1098), cb=(718, 630, 756, 1110)),
            "sim": dict(ax=(911, 642, 1491, 1103), cb=(1520, 632, 1558, 1112)),
        }),
    "fig20": dict(
        img="p16_img1.png", title="Fig. 20 (280 Hz, S1, single channel)",
        xt=[5, 10, 15], yt=[20, 40, 60, 80, 100, 120, 140, 160],
        ct=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        panels={
            "exp": dict(ax=(99, 634, 687, 1097), cb=(712, 624, 750, 1106)),
            "sim": dict(ax=(906, 634, 1490, 1097), cb=(1518, 624, 1556, 1106)),
        }),
    "fig23": dict(
        img="p18_img1.png", title="Fig. 23 (85 Hz, S2, two channels, mu1=mu2, L1=L2)",
        xt=[5, 10, 15, 20], yt=[40, 60, 80, 100, 120, 140, 160],
        ct=[2.5, 3, 3.5, 4, 4.5, 5],
        panels={
            "exp": dict(ax=(107, 623, 669, 1096), cb=(696, 614, 734, 1106)),
            "sim": dict(ax=(927, 628, 1489, 1101), cb=(1515, 618, 1554, 1110)),
        }),
}
T_THRESHOLD = 5.0          # s, "effective convergence" criterion of the paper
T_CLIP = 20.0              # s, value the authors assigned to divergent cases
FINE_DMU = 0.1             # regular output grid step in mu (units 1e-6)
FINE_DL = 1.0              # regular output grid step in L


# ---------------------------------------------------------------------------
# Colour utilities and geometry detection
# ---------------------------------------------------------------------------
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255, ... x 3) -> CIE-Lab (D65)."""
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def jet_rgb(v: np.ndarray) -> np.ndarray:
    """MATLAB jet colormap, v in [0,1] -> RGB 0..1 (for re-rendering only)."""
    v = np.clip(np.asarray(v, float), 0, 1)
    r = np.clip(np.minimum(4 * v - 1.5, -4 * v + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * v - 0.5, -4 * v + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * v + 0.5, -4 * v + 2.5), 0, 1)
    return np.stack([r, g, b], -1)


# ---------------------------------------------------------------------------
# Geometry detection
# ---------------------------------------------------------------------------
def dark_achromatic(a: np.ndarray) -> np.ndarray:
    chroma = a.max(2) - a.min(2)
    return (chroma < 40) & (a.mean(2) < 160)


def refine_line(profile: np.ndarray, darkness: np.ndarray, lo: int) -> float:
    """argmax of a count profile, then darkness-weighted centroid +/-2 px."""
    k = int(np.argmax(profile))
    idx = np.arange(max(k - 2, 0), min(k + 3, len(darkness)))
    w = darkness[idx]
    w = np.clip(w - w.min(), 0, None)
    if w.sum() == 0:
        return float(lo + k)
    return float(lo + (idx * w).sum() / w.sum())


def find_frame(a: np.ndarray, box, search: int = 12):
    x0, y0, x1, y1 = box
    D = dark_achromatic(a)
    dk = 255.0 - a.mean(2)
    ys = slice(y0 + 25, y1 - 25)
    xs = slice(x0 + 25, x1 - 25)
    out = {}
    for name, c in (("xl", x0), ("xr", x1)):
        lo, hi = c - search, c + search + 1
        prof = D[ys, lo:hi].sum(0)
        dprof = dk[ys, lo:hi].mean(0)
        out[name] = refine_line(prof, dprof, lo)
    for name, c in (("yt", y0), ("yb", y1)):
        lo, hi = c - search, c + search + 1
        prof = D[lo:hi, xs].sum(1)
        dprof = dk[lo:hi, xs].mean(1)
        out[name] = refine_line(prof, dprof, lo)
    return out


def group_positions(mask1d: np.ndarray, weights: np.ndarray, offset: int):
    """Group consecutive True entries; return weighted centroids."""
    pos = []
    i = 0
    n = len(mask1d)
    while i < n:
        if mask1d[i]:
            j = i
            while j + 1 < n and mask1d[j + 1]:
                j += 1
            idx = np.arange(i, j + 1)
            w = weights[idx] + 1e-9
            pos.append(offset + float((idx * w).sum() / w.sum()))
            i = j + 1
        else:
            i += 1
    return pos


def find_ticks(a, fr, axis: str):
    """Inward tick marks on the bottom (axis='x') or left (axis='y') frame."""
    D = dark_achromatic(a).astype(int)
    dk = 255.0 - a.mean(2)
    xl, xr, yt, yb = (int(round(fr[k])) for k in ("xl", "xr", "yt", "yb"))
    if axis == "x":
        strip = D[yb - 6: yb - 2, :]
        cnt = strip.sum(0)
        w = dk[yb - 6: yb - 2, :].sum(0)
        m = cnt >= 3
        m[: xl + 4] = False
        m[xr - 3:] = False
        return group_positions(m, w, 0)
    strip = D[:, xl + 3: xl + 7]
    cnt = strip.sum(1)
    w = dk[:, xl + 3: xl + 7].sum(1)
    m = cnt >= 3
    m[: yt + 4] = False
    m[yb - 3:] = False
    return group_positions(m, w, 0)


def label_centres_right_of(a, cx1, cy0, cy1, width=55):
    """Vertical centres of the tick-label text blobs right of a colour bar."""
    D = dark_achromatic(a)[max(cy0 - 15, 0):cy1 + 16, cx1 + 4:cx1 + 4 + width]
    rows = D.any(1)
    cents = []
    i = 0
    while i < len(rows):
        if rows[i]:
            j = i
            while j + 1 < len(rows) and (rows[j + 1] or (j + 3 < len(rows) and rows[j + 3])):
                j += 1
            if j - i >= 6:                      # ignore specks
                cents.append(max(cy0 - 15, 0) + (i + j) / 2.0)
            i = j + 1
        else:
            i += 1
    return cents


def match_ticks(pos, labels, descending: bool, span, label_px=None):
    """Contiguous label subset with minimum linear-fit residual.

    Uniformly spaced ticks fit any contiguous label subset equally well, so
    the subset is anchored by the requirement that EVERY printed label must
    map inside the axis span (frame-to-frame, +/-3 px): all labels of these
    figures are printed inside the axes (some coincide with the frame)."""
    pos = np.sort(np.asarray(pos, float))
    labs = sorted(labels, reverse=descending)
    n = len(pos)
    if n < 2:
        raise RuntimeError(f"only {n} ticks found")
    if n > len(labs):
        raise RuntimeError(f"{n} ticks found for {len(labs)} labels: {pos}")
    best = None
    lo, hi = min(span) - 3, max(span) + 3
    for k in range(len(labs) - n + 1):
        v = np.asarray(labs[k:k + n], float)
        A = np.vstack([pos, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        res = v - A @ coef
        slope = coef[0]
        rms_px = float(np.sqrt(np.mean((res / slope) ** 2)))
        allpx = (np.asarray(labs, float) - coef[1]) / slope
        inside = bool(np.all((allpx >= lo) & (allpx <= hi)))
        lab_err = 0.0
        if label_px is not None and len(label_px) == len(labs):
            lab_err = float(np.mean(np.abs(np.sort(allpx) - np.sort(label_px))))
        score = (0 if inside else 1, round(lab_err / 10.0), rms_px)
        if best is None or score < best[0]:
            best = (score, coef, v, rms_px)
    score, coef, v, rms_px = best
    if score[0]:
        raise RuntimeError("no label subset maps all labels inside the axis")
    return dict(slope=float(coef[0]), intercept=float(coef[1]),
                rms_px=rms_px, ticks_px=pos.tolist(), ticks_val=v.tolist())


def find_colorbar(a, box):
    x0, y0, x1, y1 = box
    sub = a[y0:y1, x0:x1].astype(float)
    chroma = sub.max(2) - sub.min(2)
    colfrac = (chroma > 80).mean(0)
    cols = np.where(colfrac > 0.8)[0]
    cx0, cx1 = x0 + cols.min(), x0 + cols.max()
    cmid = (cx0 + cx1) // 2
    rowc = (a[y0:y1, cmid - 3: cmid + 4].astype(float))
    rchroma = (rowc.max(2) - rowc.min(2)).min(1)
    rows = np.where(rchroma > 60)[0]
    cy0, cy1 = y0 + rows.min(), y0 + rows.max()
    # colour vs row: median over central columns (avoid edge ticks)
    c_lo, c_hi = cx0 + 3, cx1 - 6
    ys = np.arange(cy0 + 1, cy1)
    colors = np.median(a[cy0 + 1:cy1, c_lo:c_hi + 1].astype(float), axis=1)
    # ticks: inner right edge darker than the centre
    Vc = a[cy0:cy1 + 1, c_lo:c_hi + 1].astype(float).max(2).mean(1)
    # clean interior columns (exclude blended edge columns: lower median V)
    Vcols = np.median(a[cy0:cy1 + 1, cx0:cx1 + 1].astype(float).max(2), axis=0)
    good = np.where(Vcols >= 0.93 * np.median(Vcols))[0]
    ix1 = cx0 + good.max()
    Ve = a[cy0:cy1 + 1, ix1 - 4:ix1 + 1].astype(float).max(2).min(1)
    e = Vc - Ve
    drop = e - ndimage.median_filter(e, size=31, mode="nearest")
    m = drop > 0.2 * Vc
    m[:3] = False
    m[-3:] = False
    ticks = group_positions(m, drop, cy0)
    return dict(cx0=int(cx0), cx1=int(cx1), cy0=int(cy0), cy1=int(cy1),
                ys=ys, colors=colors, ticks=ticks)


# ---------------------------------------------------------------------------
# Panel digitization


# ---------------------------------------------------------------------------
# Panel digitization
# ---------------------------------------------------------------------------
def snap_lines(raw, lattice):
    snapped = np.round(np.asarray(raw, float) / lattice) * lattice
    return np.unique(snapped), np.asarray(raw, float) - snapped


def extend_grid(g, lo, hi, tol=0.3):
    """Extend a sorted grid to [lo, hi] with its end steps; returns grid and
    a boolean 'extended' flag per point."""
    g = sorted(set(float(v) for v in g))
    ext = {v: False for v in g}
    if len(g) >= 2:
        s0 = g[1] - g[0]
        v = g[0] - s0
        while v >= lo - tol * s0:
            ext[round(v, 6)] = True
            v -= s0
        s1 = g[-1] - g[-2]
        v = g[-1] + s1
        while v <= hi + tol * s1:
            ext[round(v, 6)] = True
            v += s1
    keys = sorted(ext)
    return np.array(keys), np.array([ext[k] for k in keys])


def digitize_panel(a, pcfg, cfg):
    fr = find_frame(a, pcfg["ax"])
    xt = find_ticks(a, fr, "x")
    yt = find_ticks(a, fr, "y")
    calx = match_ticks(xt, cfg["xt"], descending=False, span=(fr["xl"], fr["xr"]))
    caly = match_ticks(yt, cfg["yt"], descending=True, span=(fr["yt"], fr["yb"]))
    cb = find_colorbar(a, pcfg["cb"])
    cb_labels = label_centres_right_of(a, cb["cx1"], cb["cy0"], cb["cy1"])
    calc = match_ticks(cb["ticks"], cfg["ct"], descending=True,
                       span=(cb["cy0"], cb["cy1"]), label_px=cb_labels)
    calc["label_text_centres_px"] = cb_labels

    def px2mu(x):
        return calx["slope"] * np.asarray(x, float) + calx["intercept"]

    def px2L(y):
        return caly["slope"] * np.asarray(y, float) + caly["intercept"]

    def mu2px(m):
        return (np.asarray(m, float) - calx["intercept"]) / calx["slope"]

    def L2px(L):
        return (np.asarray(L, float) - caly["intercept"]) / caly["slope"]

    cb_vals = calc["slope"] * cb["ys"] + calc["intercept"]
    tree = cKDTree(rgb2lab(cb["colors"]))

    # interior crop (2 px inside the frame lines)
    X0 = int(math.ceil(fr["xl"])) + 2
    X1 = int(math.floor(fr["xr"])) - 2
    Y0 = int(math.ceil(fr["yt"])) + 2
    Y1 = int(math.floor(fr["yb"])) - 2
    sub = a[Y0:Y1 + 1, X0:X1 + 1].astype(float)
    lab = rgb2lab(sub)
    dE_white = np.linalg.norm(lab - rgb2lab(np.array([255.0, 255.0, 255.0])), axis=-1)
    chroma = sub.max(2) - sub.min(2)
    white = dE_white < 20
    achrom = (~white) & (chroma < 35)
    region = (~white) & (~achrom)
    lbl, nl = ndimage.label(region)
    if nl:
        sizes = ndimage.sum(region, lbl, index=np.arange(1, nl + 1))
        keep = np.zeros(nl + 1, bool)
        keep[1:] = sizes >= 30
        region = keep[lbl]
    dist, idx = tree.query(lab.reshape(-1, 3))
    dist = dist.reshape(region.shape)
    tval = cb_vals[idx].reshape(region.shape)

    # mesh-line pixels = local brightness minima across the line
    V = sub.max(2)
    Vr = np.where(region, V, np.nan)
    s = 3

    def shifted(axis, k):
        out = np.full_like(Vr, np.nan)
        if axis == 1:
            if k > 0:
                out[:, k:] = Vr[:, :-k]
            else:
                out[:, :k] = Vr[:, -k:]
        else:
            if k > 0:
                out[k:, :] = Vr[:-k, :]
            else:
                out[:k, :] = Vr[-k:, :]
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        l_, r_ = shifted(1, s), shifted(1, -s)
        u_, d_ = shifted(0, s), shifted(0, -s)
        vline = region & np.isfinite(l_) & np.isfinite(r_) & (V < 0.9 * np.fmin(l_, r_))
        hline = region & np.isfinite(u_) & np.isfinite(d_) & (V < 0.9 * np.fmin(u_, d_))
    valid = region & ~vline & ~hline & (dist < 12)
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    tmap = np.where(region, tval[iy, ix], np.nan)

    # iso-level actually used for the white boundary: values of valid pixels
    # 2..3 px inside the region from a white pixel
    dw = ndimage.distance_transform_edt(~white)
    rim = valid & (dw >= 2) & (dw <= 3)
    rim_level = (float(np.median(tval[rim])), float(np.percentile(tval[rim], 10)),
                 float(np.percentile(tval[rim], 90))) if rim.any() else None

    # mesh-line positions
    rc_col, vl_col = region.sum(0), vline.sum(0)
    rc_row, hl_row = region.sum(1), hline.sum(1)
    fcol = vl_col / np.maximum(rc_col, 1)
    frow = hl_row / np.maximum(rc_row, 1)
    mcol = (fcol >= 0.30) & (vl_col >= 12)
    mrow = (frow >= 0.30) & (hl_row >= 12)
    vx = np.array(group_positions(mcol, vl_col.astype(float), X0))
    hy = np.array(group_positions(mrow, hl_row.astype(float), Y0))
    mu_lines_raw = px2mu(vx)
    L_lines_raw = px2L(hy)
    mu_axis = (float(px2mu(fr["xl"])), float(px2mu(fr["xr"])))
    L_axis = (float(px2L(fr["yb"])), float(px2L(fr["yt"])))
    mu_seen, mu_res = snap_lines(mu_lines_raw, 0.5)
    L_seen, L_res = snap_lines(L_lines_raw, 1.0)
    mu_grid, mu_ext = extend_grid(mu_seen, mu_axis[0], mu_axis[1])
    # L: irregular steps -> do not extrapolate the step; add the frame rows
    # (MATLAB 'axis tight' puts the outermost tested rows on the frame)
    Lset = {float(v): False for v in L_seen}
    for Lf in (round(L_axis[0]), round(L_axis[1])):
        if all(abs(Lf - v) > 2 for v in Lset):
            Lset[float(Lf)] = True
    L_grid = np.array(sorted(Lset))
    L_ext = np.array([Lset[v] for v in L_grid])

    # node sampling
    H, W = region.shape
    yy, xx = np.mgrid[-4:5, -4:5]
    r2 = xx ** 2 + yy ** 2
    nodes = []
    for L, le in zip(L_grid, L_ext):
        py = min(max(float(L2px(L)) - Y0, 2.0), H - 3.0)
        for mu, me in zip(mu_grid, mu_ext):
            px = min(max(float(mu2px(mu)) - X0, 2.0), W - 3.0)
            iy0, ix0 = int(round(py)), int(round(px))
            ys_ = np.clip(iy0 + yy, 0, H - 1)
            xs_ = np.clip(ix0 + xx, 0, W - 1)
            inreg = region[ys_, xs_][r2 <= 2].mean() >= 0.5   # 3x3 majority
            vv = valid[ys_, xs_] & (r2 <= 16)
            t = float(np.mean(tval[ys_, xs_][vv])) if (inreg and vv.sum() >= 3) else np.nan
            if inreg and not np.isfinite(t):
                t = float(np.nanmean(tmap[ys_, xs_][r2 <= 9]))
            nodes.append((float(mu), float(L), t, bool(inreg), bool(me or le)))

    # fine regular grid
    mu_f = np.round(np.arange(math.ceil(mu_axis[0] * 10) / 10,
                              mu_axis[1] + 1e-9, FINE_DMU), 4)
    L_f = np.arange(math.ceil(L_axis[0]), math.floor(L_axis[1]) + 1e-9, FINE_DL)
    PX = np.clip(np.round(mu2px(mu_f) - X0).astype(int), 0, W - 1)
    PY = np.clip(np.round(L2px(L_f) - Y0).astype(int), 0, H - 1)
    tfine = tmap[np.ix_(PY, PX)]          # rows: L, cols: mu

    # pixel-level outline (per pixel row)
    contour = []
    for r in range(H):
        cols = np.where(region[r])[0]
        if cols.size:
            contour.append((float(px2L(r + Y0)), float(px2mu(cols[0] + X0)),
                            float(px2mu(cols[-1] + X0)),
                            int(cols[0] <= 1), int(cols[-1] >= W - 2),
                            int(np.sum(np.diff(cols) > 1))))

    qa = dict(
        frame_px=fr, mu_axis=mu_axis, L_axis=L_axis,
        cal_mu=calx, cal_L=caly, cal_cbar=calc,
        cbar_px=dict(cx0=cb["cx0"], cx1=cb["cx1"], cy0=cb["cy0"], cy1=cb["cy1"]),
        cbar_limits=(float(calc["slope"] * cb["cy1"] + calc["intercept"]),
                     float(calc["slope"] * cb["cy0"] + calc["intercept"])),
        px_per_mu=float(1 / calx["slope"]), px_per_L=float(-1 / caly["slope"]),
        n_region_px=int(region.sum()),
        frac_region_valid=float(valid.sum() / max(region.sum(), 1)),
        frac_region_meshline=float((vline | hline).sum() / max(region.sum(), 1)),
        median_dE_valid=float(np.median(dist[valid])) if valid.any() else None,
        p95_dE_valid=float(np.percentile(dist[valid], 95)) if valid.any() else None,
        boundary_rim_level_median_p10_p90=rim_level,
        mu_lines_raw=np.round(mu_lines_raw, 3).tolist(),
        mu_lines_snap_residual_max=float(np.max(np.abs(mu_res))) if len(mu_res) else None,
        L_lines_raw=np.round(L_lines_raw, 2).tolist(),
        L_lines_snap_residual_max=float(np.max(np.abs(L_res))) if len(L_res) else None,
        mu_grid=mu_grid.tolist(), mu_grid_extended=mu_ext.tolist(),
        L_grid=L_grid.tolist(), L_grid_extended=L_ext.tolist(),
    )
    return dict(qa=qa, nodes=nodes, mu_f=mu_f, L_f=L_f, tfine=tfine,
                contour=contour, crop=(X0, Y0, X1, Y1), region=region,
                tmap=tmap, valid=valid, vline=vline, hline=hline,
                px2mu=px2mu, px2L=px2L, mu2px=mu2px, L2px=L2px)


# ---------------------------------------------------------------------------
# Boundary on node rows
# ---------------------------------------------------------------------------
def boundary_rows(res):
    qa = res["qa"]
    region = res["region"]
    X0, Y0, X1, Y1 = res["crop"]
    H, W = region.shape
    mu_grid = list(qa["mu_grid"])
    T = {(mu, L): (t, ok) for mu, L, t, ok, _ in res["nodes"]}
    rows = []
    for L in qa["L_grid"]:
        py = int(round(min(max(float(res["L2px"](L)) - Y0, 1.0), H - 2.0)))
        cols = np.where(region[py])[0]
        ok = np.array([T[(m, L)][1] for m in mu_grid])
        tt = np.array([T[(m, L)][0] for m in mu_grid])
        if cols.size == 0 or not ok.any():
            continue
        segs = int(np.sum(np.diff(cols) > 1)) + 1
        mu_lo = float(res["px2mu"](cols[0] + X0 - 0.5))     # outer pixel edge
        mu_hi = float(res["px2mu"](cols[-1] + X0 + 0.5))
        lo_c = int(cols[0] <= 1)
        hi_c = int(cols[-1] >= W - 2)
        iv = np.where(ok)[0]
        i_lo, i_hi = iv[0], iv[-1]

        def recover(i_in, i_out, mu5):
            if i_out < 0 or i_out >= len(mu_grid) or not np.isfinite(tt[i_in]):
                return np.nan, np.nan
            m_in, m_out = mu_grid[i_in], mu_grid[i_out]
            sfrac = (mu5 - m_in) / (m_out - m_in)
            # t_out is ill-conditioned when the boundary sits on the node
            # (s small) or the inside node is itself at ~5 s
            if sfrac < 0.08 or (T_THRESHOLD - tt[i_in]) < 0.15:
                return float(sfrac), np.nan
            return float(sfrac), float(tt[i_in] + (T_THRESHOLD - tt[i_in]) / sfrac)

        s_lo, tout_lo = (np.nan, np.nan) if lo_c else recover(i_lo, i_lo - 1, mu_lo)
        s_hi, tout_hi = (np.nan, np.nan) if hi_c else recover(i_hi, i_hi + 1, mu_hi)
        tv = np.where(ok, tt, np.nan)
        j = int(np.nanargmin(tv))
        mu_opt = mu_grid[j]
        mu_opt_p = mu_opt
        opt_c = int(j == 0 or j == len(mu_grid) - 1 or not ok[j - 1] or not ok[j + 1])
        if not opt_c:
            y0, y1, y2 = tv[j - 1], tv[j], tv[j + 1]
            h_l, h_r = mu_grid[j] - mu_grid[j - 1], mu_grid[j + 1] - mu_grid[j]
            # parabola through 3 (possibly unequally spaced) points
            A = np.array([[(-h_l) ** 2, -h_l, 1], [0, 0, 1], [h_r ** 2, h_r, 1]])
            c2, c1, _ = np.linalg.solve(A, np.array([y0, y1, y2]))
            if c2 > 0:
                mu_opt_p = float(mu_opt + np.clip(-c1 / (2 * c2), -h_l, h_r))
        rows.append(dict(
            L=float(L), n_segments=segs, n_valid_nodes=int(ok.sum()),
            mu_lo=mu_lo, lo_censored=lo_c, mu_hi=mu_hi, hi_censored=hi_c,
            mu_lo_node_in=mu_grid[i_lo],
            mu_lo_node_out=mu_grid[i_lo - 1] if i_lo > 0 else np.nan,
            mu_hi_node_in=mu_grid[i_hi],
            mu_hi_node_out=mu_grid[i_hi + 1] if i_hi < len(mu_grid) - 1 else np.nan,
            s_lo=s_lo, t_out_lo=tout_lo, s_hi=s_hi, t_out_hi=tout_hi,
            mu_opt=float(mu_opt), mu_opt_par=float(mu_opt_p), opt_censored=opt_c,
            t_min=float(tv[j]),
            P_lo=np.nan if lo_c else mu_lo * L,
            P_hi=np.nan if hi_c else mu_hi * L,
            P_opt=np.nan if opt_c else mu_opt_p * L,
            band_ratio=(mu_hi / mu_lo) if not (lo_c or hi_c) else np.nan,
        ))
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stats_product(L, mu):
    L = np.asarray(L, float)
    mu = np.asarray(mu, float)
    m = np.isfinite(mu) & np.isfinite(L)
    L, mu = L[m], mu[m]
    n = len(L)
    out = dict(n=int(n))
    if n < 3:
        return out
    P = mu * L
    out.update(L_range=[float(L.min()), float(L.max())],
               P_mean=float(P.mean()), P_std=float(P.std(ddof=1)),
               P_cv=float(P.std(ddof=1) / P.mean()),
               P_min=float(P.min()), P_max=float(P.max()),
               P_max_over_min=float(P.max() / P.min()),
               r_L_P=float(np.corrcoef(L, P)[0, 1]))
    x, y = np.log(L), np.log(mu)
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = np.sum((y - A @ coef) ** 2) / max(n - 2, 1)
    se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    tq = stats.t.ppf(0.975, max(n - 2, 1))
    out.update(loglog_slope=float(coef[0]), loglog_se=float(se),
               loglog_ci95=[float(coef[0] - tq * se), float(coef[0] + tq * se)],
               minus1_in_ci95=bool(coef[0] - tq * se <= -1 <= coef[0] + tq * se))
    c = np.polyfit(L, P, 1)
    out.update(P_slope_per_tap=float(c[0]),
               P_trend_rel_over_range=float(c[0] * (L.max() - L.min()) / P.mean()))
    return out


def fit_delay_law(L, mu, cens_L=(), cens_mu_min=None, tol=0.25):
    """mu = C/(L + D), least squares in mu; D scanned, C closed form.
    95 % profile interval for D: RSS(D) <= RSS_min (1 + F(1,n-2)/(n-2)).

    Censoring: rows whose region reaches the right frame (mu_max) only say
    mu_boundary(L) >= mu_max.  The unconstrained fit is reported together with
    the number of such rows it violates (prediction < mu_max - tol), and a
    constrained fit (best D among those violating none) is reported too."""
    L = np.asarray(L, float)
    mu = np.asarray(mu, float)
    m = np.isfinite(mu) & np.isfinite(L)
    L, mu = L[m], mu[m]
    n = len(L)
    if n < 4:
        return None
    cL = np.asarray(cens_L, float)
    Ds = np.arange(-0.9 * L.min(), 600.01, 0.5)
    rss = np.empty_like(Ds)
    Cs = np.empty_like(Ds)
    nviol = np.zeros(len(Ds), int)
    for k, D in enumerate(Ds):
        g = 1.0 / (L + D)
        Cs[k] = np.sum(g * mu) / np.sum(g * g)
        rss[k] = np.sum((mu - Cs[k] * g) ** 2)
        if cL.size and cens_mu_min is not None:
            nviol[k] = int(np.sum(Cs[k] / (cL + D) < cens_mu_min - tol))
    feas = np.where(nviol == 0)[0]
    cons = None
    if cL.size and feas.size:
        kc = feas[int(np.argmin(rss[feas]))]
        cons = dict(D=float(Ds[kc]), C=float(Cs[kc]), rms=float(math.sqrt(rss[kc] / n)))
    k = int(np.argmin(rss))
    thr = rss[k] * (1 + stats.f.ppf(0.95, 1, n - 2) / (n - 2))
    inside = Ds[rss <= thr]
    k0 = int(np.argmin(np.abs(Ds - 0.0)))
    F = (rss[k0] - rss[k]) / (rss[k] / (n - 2)) if rss[k] > 0 else float("inf")
    return dict(n=int(n), D=float(Ds[k]), C=float(Cs[k]), rms=float(math.sqrt(rss[k] / n)),
                D_ci95=[float(inside.min()), float(inside.max())],
                D_ci_hits_scan_edge=bool(inside.max() >= Ds[-1] - 0.1 or
                                         inside.min() <= Ds[0] + 0.1),
                C_D0=float(Cs[k0]), rms_D0=float(math.sqrt(rss[k0] / n)),
                F_vs_D0=float(F), p_vs_D0=float(1 - stats.f.cdf(F, 1, n - 2)),
                D0_in_ci95=bool(inside.min() <= 0 <= inside.max()),
                n_censored_rows=int(cL.size),
                censored_violations_unconstrained=int(nviol[k]),
                censored_violations_D0=int(nviol[k0]),
                censored_worst_shortfall=(float(np.max(cens_mu_min - Cs[k] / (cL + Ds[k])))
                                          if cL.size else None),
                constrained_fit=cons)


def near_opt_width(res, rows, dt=0.1):
    """Width (in mu) of the set t <= t_min + dt along each node row (pixel map)."""
    X0, Y0, X1, Y1 = res["crop"]
    H, W = res["region"].shape
    for r in rows:
        py = int(round(min(max(float(res["L2px"](r["L"])) - Y0, 1.0), H - 2.0)))
        tr = res["tmap"][py]
        okp = np.isfinite(tr)
        if not okp.any():
            r["near_opt_width"] = np.nan
            continue
        tmin = np.nanmin(tr)
        sel = np.where(okp & (tr <= tmin + dt))[0]
        r["near_opt_width"] = float((sel.max() - sel.min() + 1) / res["qa"]["px_per_mu"])
        r["near_opt_rel_width"] = r["near_opt_width"] / max(r["mu_opt_par"], 1e-9)


def analyse(rows, qa=None):
    Lr = [r["L"] for r in rows]
    A = {}
    A["lower"] = stats_product(Lr, [r["mu_lo"] if not r["lo_censored"] else np.nan
                                    for r in rows])
    A["upper"] = stats_product(Lr, [r["mu_hi"] if not r["hi_censored"] else np.nan
                                    for r in rows])
    A["opt"] = stats_product(Lr, [r["mu_opt_par"] if not r["opt_censored"] else np.nan
                                  for r in rows])
    A["delay_lower"] = fit_delay_law(Lr, [r["mu_lo"] if not r["lo_censored"] else np.nan
                                          for r in rows])
    mu_max = qa["mu_axis"][1] if qa else None
    cens_hi = [r["L"] for r in rows if r["hi_censored"]]
    cens_opt = [r["L"] for r in rows if r["opt_censored"] and r["mu_opt"] >= (mu_max or 1e9) - 0.6]
    A["delay_upper"] = fit_delay_law(Lr, [r["mu_hi"] if not r["hi_censored"] else np.nan
                                          for r in rows], cens_hi, mu_max)
    A["delay_opt"] = fit_delay_law(Lr, [r["mu_opt_par"] if not r["opt_censored"] else np.nan
                                        for r in rows], cens_opt, mu_max)
    # constant-product law vs censored rows
    for key, cl in (("upper", cens_hi), ("opt", cens_opt)):
        st = A[key]
        if st.get("n", 0) >= 3 and cl and mu_max:
            pred = st["P_mean"] / np.asarray(cl, float)
            st["censored_rows_L"] = list(map(float, cl))
            st["censored_violations"] = int(np.sum(pred < mu_max - 0.25))
            st["censored_worst_shortfall"] = float(np.max(mu_max - pred))
    w = [r.get("near_opt_rel_width", np.nan) for r in rows if not r["opt_censored"]]
    w = np.asarray(w, float)
    A["near_opt_rel_width_median"] = float(np.nanmedian(w)) if np.isfinite(w).any() else None
    br = [r["band_ratio"] for r in rows if np.isfinite(r["band_ratio"])]
    A["band_ratio_hi_over_lo"] = dict(n=len(br), values=br,
                                      mean=float(np.mean(br)) if br else None,
                                      min=float(np.min(br)) if br else None,
                                      max=float(np.max(br)) if br else None)
    tl = np.array([r["t_out_lo"] for r in rows if np.isfinite(r["s_lo"])], float)
    th = np.array([r["t_out_hi"] for r in rows if np.isfinite(r["s_hi"])], float)
    sl = [r["s_lo"] for r in rows if np.isfinite(r["s_lo"])]
    sh = [r["s_hi"] for r in rows if np.isfinite(r["s_hi"])]
    A["outside_node_recovery"] = dict(
        lower_s_median=float(np.median(sl)) if sl else None,
        upper_s_median=float(np.median(sh)) if sh else None,
        lower_t_out_median=float(np.nanmedian(tl)) if np.isfinite(tl).any() else None,
        upper_t_out_median=float(np.nanmedian(th)) if np.isfinite(th).any() else None,
        lower_n_t_out=int(np.isfinite(tl).sum()), lower_frac_t_out_ge_10=float(np.mean(np.array(tl)[np.isfinite(tl)] >= 10)) if np.isfinite(tl).any() else None,
        upper_n_t_out=int(np.isfinite(th).sum()), upper_frac_t_out_ge_10=float(np.mean(np.array(th)[np.isfinite(th)] >= 10)) if np.isfinite(th).any() else None)
    A["rows"] = rows
    return A


def contour_stats(contour, L_nodes_min):
    """Robustness check on every pixel row (interpolated rows included)."""
    c = np.array(contour, float)
    if c.size == 0:
        return {}
    L = c[:, 0]
    keep = L >= L_nodes_min
    lo = np.where(c[:, 3] == 0, c[:, 1], np.nan)
    hi = np.where(c[:, 4] == 0, c[:, 2], np.nan)
    return dict(lower=stats_product(L[keep], lo[keep]),
                upper=stats_product(L[keep], hi[keep]))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if not np.isfinite(v):
        return "inf" if v == np.inf else ""
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if v != 0 else "0"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(v) for v in r])


def check_image(a, res, rows, A, title, clim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X0, Y0, X1, Y1 = res["crop"]
    qa = res["qa"]
    pad = 30
    oy, ox = max(Y0 - pad, 0), max(X0 - pad, 0)
    crop = a[oy:Y1 + pad, ox:X1 + pad]
    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.4))
    ax = axs[0]
    ax.imshow(crop)
    fr = qa["frame_px"]
    ax.plot([fr["xl"] - ox, fr["xr"] - ox, fr["xr"] - ox, fr["xl"] - ox, fr["xl"] - ox],
            [fr["yt"] - oy, fr["yt"] - oy, fr["yb"] - oy, fr["yb"] - oy, fr["yt"] - oy],
            "m-", lw=0.6)
    for x in qa["cal_mu"]["ticks_px"]:
        ax.plot([x - ox], [fr["yb"] - oy + 12], "m^", ms=5)
    for y in qa["cal_L"]["ticks_px"]:
        ax.plot([fr["xl"] - ox - 12], [y - oy], "m>", ms=5)
    for mu, L, t, ok, ext in res["nodes"]:
        px, py = res["mu2px"](mu) - ox, res["L2px"](L) - oy
        if ok:
            ax.plot(px, py, "o", mfc="none", mec="k", ms=3.2, mew=0.6)
        else:
            ax.plot(px, py, "x", color=("0.75" if ext else "0.45"), ms=2.4, mew=0.5)
    for r in rows:
        py = res["L2px"](r["L"]) - oy
        ax.plot(res["mu2px"](r["mu_lo"]) - ox, py, "s", mfc="none",
                mec=("c" if r["lo_censored"] else "b"), ms=6, mew=1.0)
        ax.plot(res["mu2px"](r["mu_hi"]) - ox, py, "s", mfc="none",
                mec=("orange" if r["hi_censored"] else "r"), ms=6, mew=1.0)
    ax.set_title("source pixels; frame/ticks (magenta); grid nodes in region (o) / out (x);\n"
                 "boundary on node rows lo (blue) hi (red), censored at frame (cyan/orange)",
                 fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    # re-render
    ax = axs[1]
    tf = res["tfine"]
    rgb = jet_rgb((tf - clim[0]) / (clim[1] - clim[0]))
    rgb[~np.isfinite(tf)] = 1.0
    ext = [res["mu_f"][0] - 0.05, res["mu_f"][-1] + 0.05,
           res["L_f"][0] - 0.5, res["L_f"][-1] + 0.5]
    ax.imshow(rgb, origin="lower", aspect="auto", extent=ext)
    nod = np.array([(m, L) for m, L, t, ok, e in res["nodes"] if ok])
    if len(nod):
        ax.scatter(nod[:, 0], nod[:, 1], s=4, c="k", lw=0)
    Ls = np.linspace(qa["L_axis"][0], qa["L_axis"][1], 300)
    for key, c, lab in (("lower", "b", "lower"), ("upper", "r", "upper"),
                        ("opt", "k", "t_min ridge")):
        st = A[key]
        if st.get("n", 0) >= 3:
            ax.plot(st["P_mean"] / Ls, Ls, c + "--", lw=1.0,
                    label=f"{lab}: mu*L={st['P_mean']:.0f} (CV {100 * st['P_cv']:.0f}%)")
    d = A["delay_upper"]
    if d:
        ax.plot(d["C"] / (Ls + d["D"]), Ls, "r:", lw=1.4,
                label=f"upper: mu*(L+{d['D']:.0f})={d['C']:.0f}")
    for r in rows:
        ax.plot(r["mu_lo"], r["L"], "b|" if not r["lo_censored"] else "c|", ms=7)
        ax.plot(r["mu_hi"], r["L"], "r|" if not r["hi_censored"] else "y|", ms=7)
        if not r["opt_censored"]:
            ax.plot(r["mu_opt_par"], r["L"], "k+", ms=5)
    ax.set_xlim(qa["mu_axis"]); ax.set_ylim(qa["L_axis"])
    ax.set_xlabel("mu (x1e-6)"); ax.set_ylabel("L")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("re-rendered from digitized regular grid (jet, same colour limits);\n"
                 "ticks = boundary on node rows, + = t_min (parabolic), curves = fitted laws",
                 fontsize=8.5)
    # colour residual
    ax = axs[2]
    tm = res["tmap"]
    src = a[Y0:Y1 + 1, X0:X1 + 1].astype(float)
    rr = jet_rgb((tm - clim[0]) / (clim[1] - clim[0])) * 255
    dE = np.linalg.norm(rgb2lab(src) - rgb2lab(np.nan_to_num(rr, nan=255)), axis=-1)
    dE[~res["region"]] = np.nan
    im = ax.imshow(dE, cmap="magma", vmin=0, vmax=30)
    plt.colorbar(im, ax=ax, fraction=0.04, label="Delta-E76: source vs jet(t digitized)")
    ax.set_title(f"colour residual on region pixels: median {np.nanmedian(dE):.1f}\n"
                 "(mesh lines show bright by construction: they were masked and filled)",
                 fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return float(np.nanmedian(dE))


def summary_plot(analysis, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = list(CFG)
    fig, axs = plt.subplots(1, len(figs), figsize=(16, 4.8), sharey=False)
    for ax, fg in zip(axs, figs):
        for key, mk in (("exp", "o"), ("sim", "s")):
            tag = f"{fg}_{key}"
            if tag not in analysis:
                continue
            rows = analysis[tag]["rows"]
            for fld, c in (("P_lo", "tab:blue"), ("P_hi", "tab:red"), ("P_opt", "k")):
                L = [r["L"] for r in rows if np.isfinite(r[fld])]
                P = [r[fld] for r in rows if np.isfinite(r[fld])]
                ax.plot(L, P, mk + ("-" if key == "exp" else "--"), color=c, ms=4,
                        mfc=(c if key == "exp" else "none"), lw=0.8,
                        label=f"{key} {fld[2:]}")
        ax.set_title(CFG[fg]["title"], fontsize=9)
        ax.set_xlabel("L (taps)")
        ax.set_ylabel("mu*L  (1e-6 x taps)")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("H-muL: a constant mu*L band would give horizontal lines "
                 "(blue = lower boundary, red = upper boundary, black = t_min ridge; "
                 "rows censored by the axis limits omitted)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        if o == np.inf:
            return "inf"
        return o if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"source": "Ouyang et al., Thin-Walled Struct. 228 (2026) 115090, "
                      "doi:10.1016/j.tws.2026.115090; Figs. 18(c,d), 20(c,d), 23(c,d)",
            "units": {"mu": "1e-6 (multiply by 1e-6 for the FxLMS step size)",
                      "L": "FIR taps", "t": "s (convergence time, decay to 5 %)"},
            "script": "data/digitize_maps.py", "panels": {}}
    analysis = {}
    all_nodes = {}
    bkeys = ["L", "n_segments", "n_valid_nodes", "mu_lo", "lo_censored", "mu_hi",
             "hi_censored", "mu_lo_node_out", "mu_lo_node_in", "mu_hi_node_in",
             "mu_hi_node_out", "s_lo", "t_out_lo", "s_hi", "t_out_hi", "mu_opt",
             "mu_opt_par", "opt_censored", "t_min", "near_opt_width", "P_lo", "P_hi",
             "P_opt", "band_ratio"]
    for fg, cfg in CFG.items():
        a = np.asarray(Image.open(os.path.join(SRC_DIR, cfg["img"])).convert("RGB"))
        a = a.astype(np.uint8)
        for key, pcfg in cfg["panels"].items():
            tag = f"{fg}_{key}"
            res = digitize_panel(a, pcfg, cfg)
            qa = res["qa"]
            clim = qa["cbar_limits"]
            rows_out = []
            for i, L in enumerate(res["L_f"]):
                for j, mu in enumerate(res["mu_f"]):
                    rows_out.append((float(mu), float(L), float(res["tfine"][i, j])))
            write_csv(os.path.join(OUT_DIR, f"map_{tag}.csv"), ["mu", "L", "t"], rows_out)
            write_csv(os.path.join(OUT_DIR, f"map_{tag}_nodes.csv"),
                      ["mu", "L", "t", "in_region", "extended"],
                      [(m, L, t, int(ok), int(e)) for m, L, t, ok, e in res["nodes"]])
            write_csv(os.path.join(OUT_DIR, f"map_{tag}_contour.csv"),
                      ["L", "mu_first", "mu_last", "first_at_frame", "last_at_frame",
                       "n_gaps"], res["contour"])
            rows = boundary_rows(res)
            near_opt_width(res, rows)
            write_csv(os.path.join(OUT_DIR, f"map_{tag}_boundary.csv"), bkeys,
                      [[r[k] for k in bkeys] for r in rows])
            A = analyse(rows, qa)
            L_nodes_min = min(r["L"] for r in rows)
            A["pixel_rows_check"] = contour_stats(res["contour"], L_nodes_min)
            med_dE = check_image(
                a, res, rows, A,
                f"{cfg['title']} - panel {'(c) Exp' if key == 'exp' else '(d) Sim'}",
                clim, os.path.join(OUT_DIR, f"checkmap_{tag}.png"))
            qa["median_dE_rerender"] = med_dE
            tn = np.array([t for m, L, t, ok, e in res["nodes"] if ok and np.isfinite(t)])
            qa["node_t_min"] = float(tn.min()) if tn.size else None
            qa["node_t_max"] = float(tn.max()) if tn.size else None
            qa["n_nodes_in_region"] = int(tn.size)
            qa["n_nodes_total"] = len(res["nodes"])
            meta["panels"][tag] = qa
            analysis[tag] = A
            all_nodes[tag] = res["nodes"]
            print(f"[{tag}] mu {qa['mu_axis'][0]:.3f}..{qa['mu_axis'][1]:.3f}  "
                  f"L {qa['L_axis'][0]:.2f}..{qa['L_axis'][1]:.2f}  cbar "
                  f"{clim[0]:.3f}..{clim[1]:.3f}  tick-rms px {qa['cal_mu']['rms_px']:.2f}/"
                  f"{qa['cal_L']['rms_px']:.2f}/{qa['cal_cbar']['rms_px']:.2f}  nodes "
                  f"{qa['n_nodes_in_region']}/{qa['n_nodes_total']}  rim level "
                  f"{qa['boundary_rim_level_median_p10_p90'][0]:.2f}  dE {med_dE:.1f}")
    meta["paper_crosschecks"] = paper_crosschecks(all_nodes)
    summary_plot(analysis, os.path.join(OUT_DIR, "checkmap_muL_summary.png"))
    with open(os.path.join(OUT_DIR, "maps_meta.json"), "w") as f:
        json.dump(clean(meta), f, indent=1)
    with open(os.path.join(OUT_DIR, "maps_muL_analysis.json"), "w") as f:
        json.dump(clean(analysis), f, indent=1)
    write_text_report(analysis, meta)


# Statements of the article's text about the 1-D sweeps (Sec. 5, Figs. 16,
# 19, 21), used as an independent check of the digitized node values.
PAPER_CHECKS = [
    # (figure, line kind, fixed value, statement)
    ("fig18", "row", 128, "Fig.16(c): eff. mu in [4,10] (exp & sim); t_min at mu=9"),
    ("fig18", "col", 10, "Fig.16(d): eff. L in [48,128]; t_min at L=100 (exp), 112 (sim)"),
    ("fig20", "row", 128, "Fig.19(c): t_min at mu=11 (exp & sim); eff. mu ranges differ by 1"),
    ("fig20", "col", 12, "Fig.19(d): t_min at L=112 / 120; eff. L ranges identical"),
    ("fig23", "row", 64, "Fig.21(c): t_min at mu=16 (exp), 13 (sim)"),
    ("fig23", "col", 6, "Fig.21(d): t_min at L=150 (exp), 128 (sim)"),
]


def paper_crosschecks(all_nodes):
    out = []
    for fg, kind, val, txt in PAPER_CHECKS:
        for key in ("exp", "sim"):
            nodes = all_nodes.get(f"{fg}_{key}")
            if nodes is None:
                continue
            if kind == "row":
                line = [(m, t) for m, L, t, ok, e in nodes if L == val and ok]
            else:
                line = [(L, t) for m, L, t, ok, e in nodes if m == val and ok]
            if not line:
                out.append(dict(check=txt, panel=f"{fg}_{key}", eff_range=None))
                continue
            xs = [x for x, _ in line]
            ts = np.array([t for _, t in line])
            j = int(np.nanargmin(ts))
            near = [x for x, t in line if t <= ts[j] + 0.05]
            out.append(dict(check=txt, panel=f"{fg}_{key}",
                            line=f"{'L' if kind == 'row' else 'mu'}={val}",
                            eff_range=[min(xs), max(xs)], argmin=xs[j],
                            t_min=float(ts[j]), within_0p05s_of_min=near))
    return out


def write_text_report(analysis, meta):
    out = []
    out.append("H-muL test: is the effective-convergence region (t <= 5 s) a band of")
    out.append("constant mu*L?  mu in 1e-6, P = mu*L in 1e-6 x taps.  Boundary = white/colour")
    out.append("edge on each tested L row (t = 5 s iso-line of the authors' interpolant).")
    out.append("c = censored by the axis limit (not used).  s = fractional position of the")
    out.append("boundary between last inside node and first outside node; t_out = recovered")
    out.append("value of that outside node (authors clip divergent cases at 20 s).")
    out.append("")
    for tag, A in analysis.items():
        qa = meta["panels"][tag]
        out.append(f"=== {tag}: rows {len(A['rows'])}, node t {qa['node_t_min']:.2f}.."
                   f"{qa['node_t_max']:.2f} s, rim level {qa['boundary_rim_level_median_p10_p90'][0]:.2f} s")
        out.append("   L |  mu_lo   P_lo  s_lo t_out |  mu_hi   P_hi  s_hi t_out | mu_opt t_min  P_opt")
        for r in A["rows"]:
            def f(v, w=6, d=2):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return " " * (w - 1) + "-"
                if v == np.inf:
                    return " " * (w - 3) + "inf"
                return f"{v:{w}.{d}f}"
            out.append(
                f"{r['L']:4.0f} | {f(r['mu_lo'])}{'c' if r['lo_censored'] else ' '}"
                f"{f(r['P_lo'], 6, 0)} {f(r['s_lo'], 5)} {f(r['t_out_lo'], 5, 1)} |"
                f" {f(r['mu_hi'])}{'c' if r['hi_censored'] else ' '}{f(r['P_hi'], 6, 0)}"
                f" {f(r['s_hi'], 5)} {f(r['t_out_hi'], 5, 1)} | {f(r['mu_opt_par'], 6, 2)}"
                f"{'c' if r['opt_censored'] else ' '}{f(r['t_min'], 5)} {f(r['P_opt'], 6, 0)}")
        for k in ("lower", "upper", "opt"):
            s = A[k]
            if s.get("n", 0) >= 3:
                out.append(
                    f"  {k:6s} n={s['n']:2d} L={s['L_range'][0]:.0f}-{s['L_range'][1]:.0f}"
                    f" P={s['P_mean']:.0f}+-{s['P_std']:.0f} (CV {100 * s['P_cv']:.1f}%,"
                    f" max/min {s['P_max_over_min']:.2f}) r(L,P)={s['r_L_P']:+.2f}"
                    f" b={s['loglog_slope']:+.2f} [{s['loglog_ci95'][0]:+.2f},"
                    f"{s['loglog_ci95'][1]:+.2f}] trend {100 * s['P_trend_rel_over_range']:+.0f}%")
            else:
                out.append(f"  {k:6s} n={s.get('n', 0)} (too few uncensored rows)")
        for k in ("delay_lower", "delay_upper", "delay_opt"):
            d = A[k]
            if d:
                out.append(
                    f"  {k:11s} D={d['D']:.1f} [{d['D_ci95'][0]:.0f},{d['D_ci95'][1]:.0f}]"
                    f"{'*' if d['D_ci_hits_scan_edge'] else ''} C={d['C']:.0f}"
                    f" rms={d['rms']:.3f} | D=0: rms={d['rms_D0']:.3f}, F={d['F_vs_D0']:.1f},"
                    f" p={d['p_vs_D0']:.3g}")
        for k in ("upper", "opt"):
            st = A[k]
            if "censored_violations" in st:
                out.append(f"  const-P {k}: censored rows {st['censored_rows_L']} -> violations"
                           f" {st['censored_violations']} (worst shortfall"
                           f" {st['censored_worst_shortfall']:+.2f}e-6)")
        for k in ("delay_upper", "delay_opt"):
            d = A[k]
            if d and d["n_censored_rows"]:
                cf = d["constrained_fit"]
                out.append(f"  {k}: unconstrained fit violates {d['censored_violations_unconstrained']}"
                           f"/{d['n_censored_rows']} censored rows (worst shortfall"
                           f" {d['censored_worst_shortfall']:+.2f}e-6); constrained fit: "
                           + (f"D={cf['D']:.1f} C={cf['C']:.0f} rms={cf['rms']:.3f}" if cf else "none"))
        out.append(f"  near-optimal (t<=t_min+0.1 s) width / mu_opt, median: "
                   f"{A['near_opt_rel_width_median']}")
        o = A["outside_node_recovery"]
        out.append(f"  outside nodes: s_lo med {o['lower_s_median']}, t_out_lo med "
                   f"{o['lower_t_out_median']}; s_hi med {o['upper_s_median']}, t_out_hi med "
                   f"{o['upper_t_out_median']}")
        b = A["band_ratio_hi_over_lo"]
        if b["n"]:
            out.append(f"  band ratio mu_hi/mu_lo: n={b['n']} mean {b['mean']:.2f}"
                       f" (min {b['min']:.2f}, max {b['max']:.2f})")
        pr = A["pixel_rows_check"]
        for k in ("lower", "upper"):
            s = pr.get(k, {})
            if s.get("n", 0) >= 3:
                out.append(f"  pixel-rows {k}: n={s['n']} CV {100 * s['P_cv']:.1f}%"
                           f" r={s['r_L_P']:+.2f} b={s['loglog_slope']:+.2f}")
        out.append("")
    out.append("=== Cross-check of digitized nodes against the article's text (1-D sweeps)")
    for c in meta.get("paper_crosschecks", []):
        if c.get("eff_range") is None:
            out.append(f"  {c['panel']}: {c['check']} -> no node in region")
            continue
        out.append(f"  {c['panel']:9s} {c['line']:7s} eff.range {c['eff_range'][0]:g}-"
                   f"{c['eff_range'][1]:g}, argmin {c['argmin']:g} (t={c['t_min']:.2f} s; "
                   f"within 0.05 s: {c['within_0p05s_of_min']})  | paper: {c['check']}")
    with open(os.path.join(OUT_DIR, "maps_muL_analysis.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
