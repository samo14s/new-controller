#!/usr/bin/env python3
"""Digitize the convergence-time curves of Ouyang et al. (2026).

Source: Q. Ouyang, L. Dong, J. Liu, J. Zhou, "From data-driven DMDc modeling to
control parameter optimization: A framework for adaptive vibration control of
thin-walled composites", Thin-Walled Structures 228 (2026) 115090,
doi:10.1016/j.tws.2026.115090.

Panels: Fig. 16(c,d), 17(b,c), 19(c,d), 21(c,d), 22(a,b) (main axes) and the
zoom insets of 16(c,d), 19(c,d), 21(c,d), 22(a,b).

Method (fully automatic once the per-panel configuration below is fixed):
  1. Axes frame: inside a rough region of interest (ROI) the longest dark
     (achromatic) horizontal / vertical pixel runs are found; the outermost ones
     are the box.  Line centres are used.
  2. Tick marks: MATLAB draws ticks inward from the box; columns (rows) that are
     dark 2-6 px inside the bottom/top (left/right) box line are tick candidates.
     The frame edges are candidates too (ticks that coincide with the box).
  3. Calibration: the labelled tick values of the configuration are matched to
     the candidates by exhaustive pair search (largest number of matched labels,
     then smallest residual) and a least-squares line pixel->data is fitted.
     The RMS residual (px) of every axis is reported.
  4. Series masks: a pixel belongs to colour c if, written as a blend
     p = a*c + (1-a)*white, a >= 0.45 and the blend residual < 45 (0-255 scale).
     Other series' colours and dark achromatic pixels are "occluders".
  5. Markers: an annulus (circle markers) or square-frame (square markers)
     template is correlated with the series mask; the score is the fraction of
     the non-occluded template pixels that carry the series colour, minus a
     penalty for series colour just outside the marker (rejects thick lines and
     blobs).  Local maxima above 0.55 are marker candidates.  The marker size
     (radius r, outline width w) is chosen per axes and series as the size that
     gives the best median score of the 3 best markers (the clean ones; sizes
     fitted to partly hidden markers come out too small and bias the centres).
     Each centre is then refined on the raw matched fraction (not normalised by
     occlusion) within +-3 px, with a 3-point parabola for sub-pixel accuracy.
  6. Data: pixel -> data with the calibration; x is snapped to a nominal grid
     (mu on multiples of 0.5e-6; L on the set used in the paper) and the snap
     residual (fraction of the local grid step) is reported.
  7. Expected grid points (configuration "expect") that were not found:
       a) the other series has a marker at the same x and this series' colour
          is present around it -> the marker is hidden under the other one
          (drawn on top): relaxed score (visible fraction >= 12 %) within
          +-(r+3) px of that marker, accepted if >= 0.4 (method
          "occluded_fit"); otherwise the covering marker's centre is used
          (method "coincident", e.g. both curves at the 20 s cap);
       b) else a weaker maximum (>= 0.35) within +-4 px of the nominal column,
          between the neighbouring markers' heights (method "local"), or the
          relaxed score >= 0.5 (method "local_relaxed").
     Every point carries its method in diagnostics.csv; check_<panel>.png
     shows peak points in black, local in orange, occluded_fit in magenta,
     coincident in cyan ("+" main axes, "x" inset).
  8. Analysis (on <panel>_best.csv): optimum and near-optimal set, mu*L at the
     optimum, effective range (t < 5 s), divergence onset, slow-branch test of
     t*mu (t*L) = const and fit t = C/(mu L), sim-vs-exp discrepancies, the
     paper's textual statements against the digitized data, cross-figure and
     inset-vs-main consistency.

Outputs (in --out):
  <panel>.csv          main-axes points           columns series,x,y,x_nominal
  <panel>_inset.csv    inset points (if any)      same columns
  <panel>_best.csv     inset value where available (higher resolution), else
                       main-axes value            same columns
  diagnostics.csv      every point with pixel coordinates, score, method
  calibration.csv      fitted pixel->data maps and residuals per axes
  check_<panel>.png    extracted points drawn on the original image
  analysis.json / analysis.txt   optima, mu*L products, slow-branch fits,
                       checks (analysis.json also holds the best curves)

Usage:
  python digitize_curves.py --fig16 p13_img1.png --fig17 p15_img0.png \
      --fig19 p16_img0.png --fig21 p17_img0.png --fig22 p18_img0.png \
      --out digitized
  (the five PNGs are the raster images embedded in PDF pages 13, 15, 16, 17,
  18 of the article; copies are kept in data/source_figs/.)

Requires numpy, scipy, matplotlib, Pillow.
"""
import argparse
import csv
import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# colours (MATLAB defaults used in the article)
# ----------------------------------------------------------------------------
PALETTE = {
    'blue': (0, 0, 255),
    'red': (255, 0, 0),
    'purple': (126, 47, 142),
    'green': (119, 172, 48),
    'orange': (217, 83, 25),
    'yellow': (237, 177, 32),
}

MU_GRID = list(np.round(np.arange(0.5, 20.01, 0.5), 2))           # x 1e-6
L_GRID = [16, 20, 24, 30, 32, 40, 48, 56, 60, 64, 72, 80, 90, 96, 100,
          104, 112, 120, 128, 136, 140, 144, 150, 160]

# ----------------------------------------------------------------------------
# panel configuration
#   roi      : rough box (x0, y0, x1, y1) in image pixels containing the axes
#   xticks / yticks : labelled tick values (in plotted units)
#   xmul     : multiplier from plotted units to data (1e-6 for mu axes)
#   exclude  : boxes (x0, y0, x1, y1) whose markers are ignored (legend,
#              inset incl. its tick labels)
#   series   : (name, colour, marker)
#   grid     : 'mu' or 'L'  (nominal x grid for snapping)
#   expect   : nominal x values expected for every series (visual reading of
#              the figure); missing ones trigger the local re-search
# ----------------------------------------------------------------------------
MU_A = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
        9.5, 10]
PANELS = {
    'fig16c': dict(
        fig='fig16', roi=(60, 650, 800, 1130),
        xticks=list(range(1, 13)), yticks=[5, 10, 15, 20], xmul=1e-6,
        exclude=[(290, 760, 670, 1010), (555, 690, 770, 760)],
        series=[('Experiment', 'blue', 'o'), ('Simulation', 'red', 's')],
        grid='mu', expect=MU_A + [11, 12],
        inset=dict(roi=(325, 765, 665, 960), xticks=[5, 6, 7, 8, 9, 10],
                   yticks=[2.5, 3, 3.5, 4],
                   expect=[5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])),
    'fig16d': dict(
        fig='fig16', roi=(870, 660, 1600, 1140),
        xticks=[32, 40, 60, 80, 100, 120, 140],
        yticks=[2, 4, 6, 8, 10, 12, 14], xmul=1.0,
        exclude=[(1050, 755, 1455, 975), (1360, 695, 1580, 765)],
        series=[('Experiment', 'purple', 'o'), ('Simulation', 'green', 's')],
        grid='L',
        expect=[32, 40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140, 144,
                150],
        inset=dict(roi=(1095, 755, 1450, 930), xticks=[64, 80, 100, 120],
                   yticks=[3, 4], yapprox=[862, 784],
                   expect=[64, 80, 90, 96, 100, 112, 120, 128])),
    'fig17b': dict(
        fig='fig17', roi=(940, 40, 1600, 300),
        xticks=[2, 4, 6, 8, 10], yticks=[5, 10, 15, 20], xmul=1e-6,
        exclude=[(1275, 85, 1560, 155)],
        series=[('Exp_sdDMDc', 'blue', 'o'), ('Exp_FEM', 'orange', 's')],
        grid='mu', expect=MU_A),
    'fig17c': dict(
        fig='fig17', roi=(940, 350, 1600, 600),
        xticks=[20, 40, 60, 80, 100], yticks=[5, 10, 15, 20], xmul=1.0,
        exclude=[(1315, 395, 1565, 465)],
        series=[('Exp_sdDMDc', 'purple', 'o'), ('Exp_FEM', 'yellow', 's')],
        grid='L', expect=[16, 20, 30, 32, 40, 48, 60, 64, 80, 90, 96]),
    'fig19c': dict(
        fig='fig19', roi=(60, 640, 800, 1120),
        xticks=[2, 4, 6, 8, 10, 12, 14, 16], yticks=[0, 5, 10, 15, 20],
        xmul=1e-6,
        exclude=[(240, 745, 675, 1030), (550, 680, 765, 750)],
        series=[('Experiment', 'blue', 'o'), ('Simulation', 'red', 's')],
        grid='mu',
        expect=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17],
        inset=dict(roi=(285, 745, 670, 955), xticks=[8, 10, 12, 14],
                   yticks=[1, 1.5, 2, 2.5],
                   expect=[7, 8, 9, 10, 11, 12, 13, 14])),
    'fig19d': dict(
        fig='fig19', roi=(870, 640, 1600, 1130),
        xticks=[20, 40, 60, 80, 100, 120, 140], yticks=[2, 4, 6, 8, 10],
        xmul=1.0,
        exclude=[(1095, 755, 1510, 980), (1360, 680, 1580, 750)],
        series=[('Experiment', 'purple', 'o'), ('Simulation', 'green', 's')],
        grid='L',
        expect=[16, 32, 40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140,
                144, 150, 160],
        inset=dict(roi=(1135, 755, 1505, 940), xticks=[80, 100, 120, 140],
                   yticks=[1, 1.5, 2],
                   expect=[80, 90, 96, 100, 112, 120, 128, 140])),
    'fig21c': dict(
        fig='fig21', roi=(60, 620, 790, 1100),
        xticks=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], yticks=[5, 10, 15, 20],
        xmul=1e-6,
        exclude=[(285, 705, 655, 935), (575, 935, 660, 972),
                 (85, 1003, 300, 1068)],
        series=[('Experiment', 'blue', 'o'), ('Simulation', 'red', 's')],
        grid='mu',
        expect=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
                8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        inset=dict(roi=(305, 710, 650, 900), xticks=[10, 12, 14, 16],
                   yticks=[4, 5, 6, 7],
                   expect=[9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17])),
    'fig21d': dict(
        fig='fig21', roi=(870, 620, 1600, 1100),
        xticks=[32, 40, 60, 80, 100, 120, 140, 160], yticks=[5, 10, 15, 20],
        xmul=1.0,
        exclude=[(1110, 745, 1550, 975), (1345, 665, 1560, 735)],
        series=[('Experiment', 'purple', 'o'), ('Simulation', 'green', 's')],
        grid='L',
        expect=[32, 40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140, 144,
                150, 160],
        inset=dict(roi=(1155, 745, 1550, 935), xticks=[100, 120, 140],
                   yticks=[4, 5, 6],
                   expect=[100, 112, 120, 128, 140, 144, 150])),
    'fig22a': dict(
        fig='fig22', roi=(80, 30, 800, 470),
        xticks=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], yticks=[5, 10, 15, 20],
        xmul=1e-6,
        exclude=[(290, 125, 655, 330), (110, 60, 350, 140)],
        series=[('Experiment', 'blue', 'o'), ('Simulation', 'red', 's')],
        grid='mu',
        expect=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
                8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        inset=dict(roi=(335, 125, 650, 295), xticks=[10, 12, 14, 16],
                   yticks=[4, 5, 6],
                   expect=[10, 11, 12, 13, 14, 15, 16, 17])),
    'fig22b': dict(
        fig='fig22', roi=(890, 30, 1600, 480),
        xticks=[32, 40, 60, 80, 100, 120, 140, 160],
        yticks=[3, 4, 5, 6, 7, 8, 9, 10], xmul=1.0,
        exclude=[(1130, 70, 1560, 300), (925, 65, 1150, 140)],
        series=[('Experiment', 'purple', 'o'), ('Simulation', 'green', 's')],
        grid='L',
        expect=[32, 40, 48, 60, 64, 80, 90, 96, 100, 112, 120, 128, 140, 144,
                150, 160],
        inset=dict(roi=(1170, 75, 1555, 265), xticks=[100, 120, 140, 160],
                   yticks=[3.5, 4, 4.5, 5, 5.5],
                   expect=[100, 112, 120, 128, 140, 144, 150, 160])),
}


# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------
def load(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(float)


def dark_mask(a, vmax=120, chroma=45):
    mx = a.max(2)
    mn = a.min(2)
    return (mx < vmax) & ((mx - mn) < chroma)


def colour_mask(a, rgb, amin=0.45, rmax=45.0):
    w = np.array([255.0, 255.0, 255.0])
    v = w - np.array(rgb, float)
    d = w - a
    alpha = d @ v / (v @ v)
    resid = np.linalg.norm(d - alpha[..., None] * v, axis=2)
    return (alpha >= amin) & (resid < rmax)


def longest_run(v, gap=3):
    """Longest run of True allowing interior gaps of <= gap pixels."""
    idx = np.flatnonzero(v)
    if idx.size == 0:
        return 0
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return int((ends - starts + 1).max())


def _groups(idx):
    out = []
    for i in idx:
        if out and i - out[-1][1] <= 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def detect_frame(dark, roi, frac=0.4, colour=None):
    """Return (xl, xr, yt, yb, thick) of the outermost long dark lines.

    Runs are measured on dark|colour (data markers drawn over the box do not
    break it) but a line must also be at least 25 % dark.
    """
    x0, y0, x1, y1 = roi
    m = dark[y0:y1, x0:x1]
    mc = m if colour is None else (m | colour[y0:y1, x0:x1])
    H, W = m.shape
    rows = [y for y in range(H) if longest_run(mc[y]) >= frac * W
            and m[y].sum() >= 0.25 * W]
    cols = [x for x in range(W) if longest_run(mc[:, x]) >= frac * H
            and m[:, x].sum() >= 0.25 * H]
    if not rows or not cols:
        raise RuntimeError('frame not found in roi %s' % (roi,))
    gr, gc = _groups(rows), _groups(cols)
    yt = y0 + 0.5 * (gr[0][0] + gr[0][1])
    yb = y0 + 0.5 * (gr[-1][0] + gr[-1][1])
    xl = x0 + 0.5 * (gc[0][0] + gc[0][1])
    xr = x0 + 0.5 * (gc[-1][0] + gc[-1][1])
    if len(gc) == 1:          # right edge broken by data: search lower frac
        cols2 = [x for x in range(W) if longest_run(m[:, x]) >= 0.25 * H]
        gc2 = _groups(cols2)
        xr = x0 + 0.5 * (gc2[-1][0] + gc2[-1][1])
    th = max(gr[0][1] - gr[0][0], gr[-1][1] - gr[-1][0],
             gc[0][1] - gc[0][0]) + 1
    return xl, xr, yt, yb, th


def tick_candidates(dark, frame, depth=(2, 6)):
    """Tick-mark pixel positions along x (bottom/top) and y (left/right)."""
    xl, xr, yt, yb, th = frame
    h = th / 2.0
    d0, d1 = depth
    xs, ys = [], []
    for ybase, sgn in ((yb, -1), (yt, +1)):
        rows = [int(round(ybase + sgn * (h + k))) for k in range(d0, d1 + 1)]
        c0, c1 = int(xl + h + 3), int(xr - h - 3)
        hit = [x for x in range(c0, c1)
               if sum(dark[r, x] for r in rows) >= len(rows) - 1]
        xs += [0.5 * (g[0] + g[1]) for g in _groups(hit) if g[1] - g[0] <= 5]
    for xbase, sgn in ((xl, +1), (xr, -1)):
        cols = [int(round(xbase + sgn * (h + k))) for k in range(d0, d1 + 1)]
        r0, r1 = int(yt + h + 3), int(yb - h - 3)
        hit = [y for y in range(r0, r1)
               if sum(dark[y, c] for c in cols) >= len(cols) - 1]
        ys += [0.5 * (g[0] + g[1]) for g in _groups(hit) if g[1] - g[0] <= 5]
    # merge duplicates from the two opposite edges
    def merge(v):
        v = sorted(v)
        out = []
        for p in v:
            if out and p - out[-1][-1] <= 2.5:
                out[-1].append(p)
            else:
                out.append([p])
        return [float(np.mean(g)) for g in out]
    xs = merge(xs) + [xl, xr]
    ys = merge(ys) + [yt, yb]
    return sorted(xs), sorted(ys)


def calibrate(values, cands, sign, tol=2.5, approx=None):
    """Match labelled values to candidate pixels; LS fit p = a + b*v.

    approx: optional approximate pixel position of every labelled tick (read
    off the image); needed when only two ticks are labelled, because any pair
    of candidates then fits exactly.  Each value is then matched to the
    nearest candidate within 12 px of its approximate position.
    """
    values = np.asarray(values, float)
    cands = np.asarray(cands, float)
    if approx is not None:
        v, p = [], []
        for val, ap in zip(values, approx):
            d = np.abs(cands - ap)
            if d.min() <= 12:
                v.append(val)
                p.append(cands[d.argmin()])
        v, p = np.array(v), np.array(p)
        B = np.vstack([np.ones_like(v), v]).T
        (a, b), *_ = np.linalg.lstsq(B, p, rcond=None)
        rms = float(np.sqrt(np.mean((a + b * v - p) ** 2)))
        return dict(a=float(a), b=float(b), rms=rms, n=len(v),
                    ntot=len(values), matched=list(zip(v.tolist(),
                                                       p.tolist())))
    best = None
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(len(cands)):
                for m in range(len(cands)):
                    if k == m:
                        continue
                    b = (cands[m] - cands[k]) / (values[j] - values[i])
                    if b * sign <= 0:
                        continue
                    a = cands[k] - b * values[i]
                    pred = a + b * values
                    dist = np.abs(pred[:, None] - cands[None, :])
                    jj = dist.argmin(1)
                    ok = dist[np.arange(n), jj] < tol
                    if ok.sum() < 2:
                        continue
                    res = dist[np.arange(n), jj][ok]
                    key = (ok.sum(), -res.mean())
                    if best is None or key > best[0]:
                        best = (key, ok.copy(), jj.copy())
    if best is None:
        raise RuntimeError('tick calibration failed')
    _, ok, jj = best
    v = values[ok]
    p = cands[jj[ok]]
    B = np.vstack([np.ones_like(v), v]).T
    (a, b), *_ = np.linalg.lstsq(B, p, rcond=None)
    rms = float(np.sqrt(np.mean((a + b * v - p) ** 2)))
    return dict(a=float(a), b=float(b), rms=rms, n=int(ok.sum()),
                ntot=n, matched=list(zip(v.tolist(), p.tolist())))


# ----------------------------------------------------------------------------
# marker templates and detection
# ----------------------------------------------------------------------------
def template(kind, r, w, ss=4):
    R = int(math.ceil(r + w / 2 + 4))
    g = (np.arange(-R * ss, (R + 1) * ss) + 0.5) / ss - 0.5 / ss
    X, Y = np.meshgrid(g, g)
    if kind == 'o':
        d = np.hypot(X, Y)
    else:
        d = np.maximum(np.abs(X), np.abs(Y))
    ring = (np.abs(d - r) <= w / 2).astype(float)
    outer = ((d >= r + w / 2 + 1.0) & (d <= r + w / 2 + 3.0)).astype(float)
    n = 2 * R + 1

    def down(t):
        t = t[:n * ss, :n * ss]
        return t.reshape(n, ss, n, ss).mean(axis=(1, 3))
    return down(ring), down(outer)


def score_parts(S, O, kind, r, w):
    """fvis: series fraction of the visible (non-occluded) template pixels;
    ftot: series fraction of all template pixels; visf: visible fraction;
    fout: series fraction in a ring just outside the marker."""
    T, To = template(kind, r, w)
    num = fftconvolve(S, T, mode='same')
    occ = fftconvolve(O, T, mode='same')
    out = fftconvolve(S, To, mode='same')
    tot = T.sum()
    vis = np.maximum(tot - occ, 1e-6)
    return num / vis, num / tot, vis / tot, out / To.sum()


def score_map(S, O, kind, r, w, relaxed=False):
    fvis, ftot, visf, fout = score_parts(S, O, kind, r, w)
    sc = fvis - 0.8 * fout
    if relaxed:     # for markers mostly hidden under the other series
        sc[(ftot < 0.10) | (visf < 0.12)] = 0.0
    else:
        sc[(ftot < 0.30) | (visf < 0.35)] = 0.0
    return sc


def peaks(sc, r, thr):
    size = max(3, int(round(1.6 * r)))
    mx = ndimage.maximum_filter(sc, size=size)
    ys, xs = np.nonzero((sc == mx) & (sc >= thr))
    out = []
    for y, x in zip(ys, xs):
        if 1 <= y < sc.shape[0] - 1 and 1 <= x < sc.shape[1] - 1:
            def para(m1, c, p1):
                den = m1 - 2 * c + p1
                return 0.0 if den >= 0 else 0.5 * (m1 - p1) / den
            dx = para(sc[y, x - 1], sc[y, x], sc[y, x + 1])
            dy = para(sc[y - 1, x], sc[y, x], sc[y + 1, x])
        else:
            dx = dy = 0.0
        out.append((x + dx, y + dy, float(sc[y, x])))
    # suppress near-duplicates (plateaus)
    out.sort(key=lambda t: -t[2])
    keep = []
    for p in out:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.8 * r for q in keep):
            keep.append(p)
    return keep


def refine(m, p, rad=3):
    """Move peak p=(x, y, score) to the max of map m within +-rad px."""
    x0, y0 = int(round(p[0])), int(round(p[1]))
    a, b = max(1, x0 - rad), min(m.shape[1] - 2, x0 + rad)
    c, d = max(1, y0 - rad), min(m.shape[0] - 2, y0 + rad)
    if a > b or c > d:
        return p
    win = m[c:d + 1, a:b + 1]
    k = np.unravel_index(win.argmax(), win.shape)
    y, x = k[0] + c, k[1] + a

    def para(m1, c_, p1):
        den = m1 - 2 * c_ + p1
        return 0.0 if den >= 0 else 0.5 * (m1 - p1) / den
    dx = para(m[y, x - 1], m[y, x], m[y, x + 1])
    dy = para(m[y - 1, x], m[y, x], m[y + 1, x])
    return (x + dx, y + dy, p[2])


SIZES = {k: [(r, w) for r in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
             for w in (2.0, 2.5, 3.0)] for k in ('o', 's')}


# ----------------------------------------------------------------------------
# digitization of one axes
# ----------------------------------------------------------------------------
def snap(xval, grid):
    g = np.asarray(grid, float)
    i = int(np.abs(g - xval).argmin())
    sp = np.diff(g).min() if len(g) > 1 else 1.0
    loc = []
    if i > 0:
        loc.append(g[i] - g[i - 1])
    if i < len(g) - 1:
        loc.append(g[i + 1] - g[i])
    sp = min(loc) if loc else sp
    return float(g[i]), float((xval - g[i]) / sp)


def digitize_axes(img, dark, name, roi, xticks, yticks, xmul, series,
                  gridname, expect, exclude, thr=0.55, xapprox=None,
                  yapprox=None):
    anycol = np.zeros(dark.shape, bool)
    for _, c, _ in series:
        anycol |= colour_mask(img, PALETTE[c])
    frame = detect_frame(dark, roi, colour=anycol)
    xl, xr, yt, yb, th = frame
    # tick length differs between main axes and insets: try several depths
    # and keep, per axis, the calibration matching the most labelled ticks
    cx = cy = None
    for depth in ((2, 6), (1, 4), (1, 3), (0, 2)):
        xc, yc = tick_candidates(dark, frame, depth)
        c1 = calibrate(xticks, xc, +1, approx=xapprox)
        c2 = calibrate(yticks, yc, -1, approx=yapprox)
        if cx is None or (c1['n'], -c1['rms']) > (cx['n'], -cx['rms']):
            cx = c1
        if cy is None or (c2['n'], -c2['rms']) > (cy['n'], -cy['rms']):
            cy = c2
    for c, lab in ((cx, 'x'), (cy, 'y')):
        if c['n'] < c['ntot']:
            print('  WARNING %s %s-axis: only %d/%d labelled ticks matched'
                  % (name, lab, c['n'], c['ntot']))
    grid = MU_GRID if gridname == 'mu' else L_GRID

    def px2data(px, py):          # -> (x in data units, y)
        return xmul * (px - cx['a']) / cx['b'], (py - cy['a']) / cy['b']

    def data2px(x, y):            # x in data units
        return cx['a'] + cx['b'] * x / xmul, cy['a'] + cy['b'] * y

    H, W = dark.shape
    # search window: frame expanded by a marker size
    pad = 10
    X0, X1 = int(max(0, xl - pad)), int(min(W, xr + pad + 1))
    Y0, Y1 = int(max(0, yt - pad)), int(min(H, yb + pad + 1))
    sub = img[Y0:Y1, X0:X1]
    masks = {c: colour_mask(sub, PALETTE[c]) for _, c, _ in series}
    dk = dark_mask(sub, vmax=170, chroma=35)
    sizes_used = {}
    founds, maps = {}, {}
    # ---- pass 1: template peaks for every series
    for sname, col, kind in series:
        S = masks[col].astype(float)
        O = dk.copy()
        for _, c2, _ in series:
            if c2 != col:
                O |= masks[c2]
        O = O.astype(float)
        # choose marker size: best median score of the strongest peaks
        best = None
        nexp = len(expect)
        for r, w in SIZES[kind]:
            sc = score_map(S, O, kind, r, w)
            pk = peaks(sc, r, thr)
            pk = [p for p in pk if not _excluded(p[0] + X0, p[1] + Y0,
                                                 exclude, frame)]
            # size criterion: median score of the 3 best markers (the
            # clean, unoccluded ones); a size fitted to partly hidden
            # markers would be too small and bias the centres
            top = sorted([p[2] for p in pk], reverse=True)[:3]
            if len(top) < 3:
                continue
            q = float(np.median(top))
            if best is None or q > best[0]:
                best = (q, r, w, sc, pk)
        _, r, w, sc, pk = best
        sizes_used[sname] = (r, w)
        maps[sname] = (sc, score_map(S, O, kind, r, w, relaxed=True), r)
        # localisation: the occlusion-normalised score favours positions
        # where part of the ring lies on the other series (counted as
        # unknown), which biases partly hidden markers away from the
        # occluder.  The centre is therefore refined on the raw matched
        # fraction ftot (unbiased by occlusion) within +-3 px.
        ftot = score_parts(S, O, kind, r, w)[1]
        pk = [refine(ftot, p_) for p_ in pk]
        found = {}
        for (px, py, sco) in pk:
            gx, gy = px + X0, py + Y0
            x, y = px2data(gx, gy)
            xn, res = snap(x / xmul, grid)
            if abs(res) > 0.35:
                continue
            if xn in found and found[xn]['score'] >= sco:
                continue
            found[xn] = dict(series=sname, px=gx, py=gy, x=x, y=y,
                             x_nominal=xn * xmul, snap_res=res, score=sco,
                             method='peak')
        founds[sname] = found

    def record(sname, xn, gx, gy, sco, method):
        x, y = px2data(gx, gy)
        return dict(series=sname, px=float(gx), py=float(gy), x=x, y=y,
                    x_nominal=xn * xmul, snap_res=snap(x / xmul, grid)[1],
                    score=float(sco), method=method)

    def best_in(m, xa, xb, ya, yb_):
        """max of map m inside the pixel box (global coords)."""
        a, b = int(max(0, xa - X0)), int(min(m.shape[1], xb - X0 + 1))
        c, d = int(max(0, ya - Y0)), int(min(m.shape[0], yb_ - Y0 + 1))
        if a >= b or c >= d:
            return None
        win = m[c:d, a:b].copy()
        for yy in range(win.shape[0]):
            for xx in range(win.shape[1]):
                if _excluded(xx + a + X0, yy + c + Y0, exclude, frame):
                    win[yy, xx] = 0.0
        k = np.unravel_index(win.argmax(), win.shape)
        return float(win[k]), k[1] + a + X0, k[0] + c + Y0

    # ---- pass 2: expected but missing points
    for sname, col, kind in series:
        found = founds[sname]
        sc, scr, r = maps[sname]
        for xn in expect:
            if xn in found:
                continue
            ex_px, _ = data2px(xn * xmul, 0)
            others = [founds[o][xn] for o, _, _ in series
                      if o != sname and xn in founds[o]
                      and founds[o][xn]['method'] == 'peak']
            # y-span for the local searches: between the neighbouring markers of
            # the same series (whole column at the ends of a curve)
            ks = sorted(found)
            lo = [found[k]['py'] for k in ks if k < xn][-1:]
            hi = [found[k]['py'] for k in ks if k > xn][:1]
            ys = lo + hi
            if len(ys) == 2:          # bracketed: stay near the polyline
                ya, yb_ = min(ys) - 3 * r, max(ys) + 3 * r
            else:                     # end of a curve: whole column
                ya, yb_ = Y0, Y1
            # (a) hidden under the other series' marker (drawn on top): only
            #     if this series' colour is present right around that marker
            if others:
                ox, oy = others[0]['px'], others[0]['py']
                rr = int(r + 4)
                yy0, xx0 = int(oy - Y0), int(ox - X0)
                near = masks[col][max(0, yy0 - rr):yy0 + rr + 1,
                                  max(0, xx0 - rr):xx0 + rr + 1].sum()
                if near >= 8:
                    b = best_in(scr, ox - r - 3, ox + r + 3, oy - r - 3,
                                oy + r + 3)
                    if b is not None and b[0] >= 0.4:
                        found[xn] = record(sname, xn, b[1], b[2], b[0],
                                           'occluded_fit')
                    else:
                        found[xn] = record(sname, xn, ox, oy, 0.0,
                                           'coincident')
                    continue
            # (b) weaker marker near the expected column
            b = best_in(sc, ex_px - 4, ex_px + 4, ya, yb_)
            if b is not None and b[0] >= 0.35:
                found[xn] = record(sname, xn, b[1], b[2], b[0], 'local')
                continue
            b = best_in(scr, ex_px - 4, ex_px + 4, ya, yb_)
            if b is not None and b[0] >= 0.5:
                found[xn] = record(sname, xn, b[1], b[2], b[0],
                                   'local_relaxed')
                continue
            print('  MISSING %s %s x=%g' % (name, sname, xn))
    pts = []
    for sname, _, _ in series:
        found = founds[sname]
        for xn in sorted(set(found) - set(expect)):
            found[xn]['method'] += '_unexpected'
        pts += [found[k] for k in sorted(found)]
    cal = dict(axes=name, frame=[xl, xr, yt, yb], thick=th,
               x_a=cx['a'], x_b=cx['b'], x_rms_px=cx['rms'],
               x_ticks_matched='%d/%d' % (cx['n'], cx['ntot']),
               y_a=cy['a'], y_b=cy['b'], y_rms_px=cy['rms'],
               y_ticks_matched='%d/%d' % (cy['n'], cy['ntot']),
               x_px_per_unit=cx['b'], y_px_per_unit=cy['b'],
               marker_sizes=json.dumps(sizes_used))
    return pts, cal, (cx, cy, frame)


def _excluded(gx, gy, exclude, frame):
    for (a, b, c, d) in exclude:
        if a <= gx <= c and b <= gy <= d:
            return True
    return False


# ----------------------------------------------------------------------------
# overlay
# ----------------------------------------------------------------------------
def overlay(img, panel, cfg, main_pts, inset_pts, cals, path):
    x0, y0, x1, y1 = cfg['roi']
    crop = img[y0:y1, x0:x1].astype(np.uint8)
    h, w = crop.shape[:2]
    fig = plt.figure(figsize=(w / 50.0, h / 50.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop, extent=(x0, x1, y1, y0), interpolation='nearest')
    cols = {s: c for s, c, _ in cfg['series']}
    for (cx, cy, fr) in cals:
        for v, p in cx['matched']:
            ax.plot([p, p], [fr[3] + 6, fr[3] + 16], color='magenta', lw=1.5)
        for v, p in cy['matched']:
            ax.plot([fr[0] - 16, fr[0] - 6], [p, p], color='magenta', lw=1.5)
        ax.plot([fr[0], fr[1], fr[1], fr[0], fr[0]],
                [fr[2], fr[2], fr[3], fr[3], fr[2]], color='cyan', lw=0.8,
                ls=':')
    for pts, mk in ((main_pts, '+'), (inset_pts, 'x')):
        for p in pts:
            c = {'peak': 'black', 'local': 'darkorange',
                 'occluded_fit': 'magenta', 'coincident': 'cyan'}.get(
                     p['method'], 'red')
            ax.plot(p['px'], p['py'], mk, color=c, ms=9, mew=1.6)
    for (a, b, c, d) in cfg['exclude']:
        ax.plot([a, c, c, a, a], [b, b, d, d, b], color='grey', lw=0.6,
                ls='--')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis('off')
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['series', 'x', 'y', 'x_nominal'])
        for p in rows:
            wr.writerow([p['series'], '%.6g' % p['x'], '%.4f' % p['y'],
                         '%.6g' % p['x_nominal']])


def best_rows(main, inset):
    out = []
    ins = {(p['series'], round(p['x_nominal'], 12)): p for p in inset}
    for p in main:
        key = (p['series'], round(p['x_nominal'], 12))
        q = ins.get(key)
        out.append(dict(p, y=q['y'], src='inset') if q is not None
                   else dict(p, src='main'))
    return out


SWEEPS = {
    # panel: fixed parameter, swept parameter, excitation frequency [Hz], note
    'fig16c': dict(fix=('L', 128), sw='mu', f=120.0, note='1-ch S2'),
    'fig16d': dict(fix=('mu', 1e-5), sw='L', f=120.0, note='1-ch S2'),
    'fig17b': dict(fix=('L', 128), sw='mu', f=120.0, note='1-ch S2 exp'),
    'fig17c': dict(fix=('mu', 1e-5), sw='L', f=120.0, note='1-ch S2 exp'),
    'fig19c': dict(fix=('L', 128), sw='mu', f=280.0, note='1-ch S1'),
    'fig19d': dict(fix=('mu', 1.2e-5), sw='L', f=280.0, note='1-ch S1'),
    'fig21c': dict(fix=('L', 64), sw='mu', f=85.0,
                   note='2-ch S2, mu1=mu2, L1=L2'),
    'fig21d': dict(fix=('mu', 6e-6), sw='L', f=85.0,
                   note='2-ch S2, mu1=mu2, L1=L2'),
    'fig22a': dict(fix=('L', 64), sw='mu', f=85.0,
                   note='2-ch, sweep mu2; mu1=1.4e-5, L1=L2=64'),
    'fig22b': dict(fix=('mu', 6e-6), sw='L', f=85.0,
                   note='2-ch, sweep L2; L1=128, mu1=mu2=6e-6'),
}
FS = 10000.0     # dSPACE sampling rate [Hz]
CAP = 19.8       # points at the 20 s cap (divergent / >20 s) are excluded
TOL_OPT = 0.03   # [s] points within this of the minimum are "near-optimal"
SLOW_K = 1.5     # slow branch: left of the optimum and t >= SLOW_K * t_min


def lam_min(L, f):
    """Smaller eigenvalue (per unit filtered-tone power/2) of the L x L
    autocorrelation matrix of a pure tone, R_ij = cos(w(i-j)):
    lambda_min = (L - |sin(L w)/sin(w)|)/2,  w = 2 pi f / fs."""
    w = 2 * math.pi * f / FS
    L = np.asarray(L, float)
    return 0.5 * (L - np.abs(np.sin(L * w) / math.sin(w)))


def series_arrays(rows, s):
    P = sorted([p for p in rows if p['series'] == s],
               key=lambda p: p['x_nominal'])
    return (np.array([p['x_nominal'] for p in P]),
            np.array([p['y'] for p in P]))


def eff_range(x, t, thr=5.0):
    """x-range of the contiguous run of t < thr containing the minimum."""
    ok = t < thr
    if not ok.any():
        return None
    i0 = int(np.argmin(np.where(t < CAP, t, np.inf)))
    if not ok[i0]:
        return None
    i, j = i0, i0
    while i - 1 >= 0 and ok[i - 1]:
        i -= 1
    while j + 1 < len(t) and ok[j + 1]:
        j += 1
    return [float(x[i]), float(x[j])]


def analyse(panel, rows, dash=None):
    sw_cfg = SWEEPS[panel]
    fixname, fixval = sw_cfg['fix']
    sw, f = sw_cfg['sw'], sw_cfg['f']
    res = {}
    for s in sorted(set(p['series'] for p in rows)):
        x, t = series_arrays(rows, s)
        mu = x if sw == 'mu' else np.full_like(x, fixval)
        L = x if sw == 'L' else np.full_like(x, fixval)
        valid = t < CAP
        if not valid.any():
            continue
        i0 = int(np.flatnonzero(valid)[np.argmin(t[valid])])
        tmin = float(t[i0])
        d = dict(n_points=int(len(x)), x_opt=float(x[i0]), t_min=tmin,
                 muL_opt=float(mu[i0] * L[i0]),
                 near_opt=[float(v) for v in x[(t <= tmin + TOL_OPT)
                                               & valid]],
                 eff_range_5s=eff_range(x, t, 5.0))
        if dash is not None:
            d['eff_range_drawn_line'] = eff_range(x, t, dash)
        # divergence onset: last uncapped / first capped point right of opt
        cap_r = [i for i in range(i0 + 1, len(x)) if not valid[i]]
        if cap_r:
            j = cap_r[0]
            d['divergence_bracket_x'] = [float(x[j - 1]), float(x[j])]
            d['divergence_bracket_muL'] = [float(mu[j - 1] * L[j - 1]),
                                           float(mu[j] * L[j])]
        slow = (np.arange(len(x)) < i0) & (t >= SLOW_K * tmin) & valid
        if slow.sum() >= 3:
            ts, ms, Ls, xs = t[slow], mu[slow], L[slow], x[slow]
            prod = ts * xs                     # t*mu or t*L
            g = 1.0 / (ms * Ls)
            C = float((ts * g).sum() / (g * g).sum())
            rel = (ts - C * g) / ts
            p_exp, _ = np.polyfit(np.log(xs), np.log(ts), 1)
            A = np.vstack([g, np.ones_like(g)]).T
            (C2, t0), *_ = np.linalg.lstsq(A, ts, rcond=None)
            d.update(slow_x=[float(v) for v in xs],
                     slow_t=[round(float(v), 4) for v in ts],
                     t_times_x=[float(v) for v in prod],
                     t_times_x_cv=float(prod.std(ddof=1) / prod.mean()),
                     loglog_exponent=float(-p_exp),
                     C=C, C_rel_rms=float(np.sqrt(np.mean(rel ** 2))),
                     C_rel_max=float(np.abs(rel).max()),
                     C_with_offset=float(C2), t0_offset=float(t0),
                     kappa=float(math.log(20.0) / (C * FS)))
            if sw == 'L':     # eigenvalue-spread variant for L sweeps
                lm = lam_min(Ls, f)
                q = ts * lm
                pl, _ = np.polyfit(np.log(lm), np.log(ts), 1)
                d.update(t_times_lammin_cv=float(q.std(ddof=1) / q.mean()),
                         loglog_exponent_vs_lammin=float(-pl))
        res[s] = d
    return res


def compare(rows, a='Experiment', b='Simulation'):
    """sim-vs-exp discrepancy on common, uncapped grid points."""
    xa, ta = series_arrays(rows, a)
    xb, tb = series_arrays(rows, b)
    if not len(xa) or not len(xb):
        return None
    da = dict(zip(np.round(xa, 12), ta))
    db = dict(zip(np.round(xb, 12), tb))
    out = {}
    for name, cond in (('stable', lambda u, v: u < CAP and v < CAP),
                       ('effective', lambda u, v: u < 5.0 and v < 5.0)):
        ks = [k for k in sorted(set(da) & set(db)) if cond(da[k], db[k])]
        if not ks:
            continue
        rel = np.array([abs(db[k] - da[k]) / da[k] for k in ks])
        ab = np.array([abs(db[k] - da[k]) for k in ks])
        out[name] = dict(n=len(ks), x_range=[float(ks[0]), float(ks[-1])],
                         max_rel=float(rel.max()),
                         x_at_max_rel=float(ks[int(rel.argmax())]),
                         max_abs=float(ab.max()),
                         x_at_max_abs=float(ks[int(ab.argmax())]))
    return out


def inset_consistency(main_pts, inset_pts):
    m = {(p['series'], round(p['x_nominal'], 12)): p['y'] for p in main_pts}
    d = [q['y'] - m[(q['series'], round(q['x_nominal'], 12))]
         for q in inset_pts
         if (q['series'], round(q['x_nominal'], 12)) in m]
    if not d:
        return None
    d = np.array(d)
    return dict(n=len(d), mean=float(d.mean()), max_abs=float(np.abs(d).max()))


def dashed_level(dark, frame, cy, exclude):
    """y value of the dashed threshold line (many short dark runs)."""
    xl, xr, yt, yb, th = frame
    best = None
    for y in range(int(yt + th + 2), int(yb - th - 1)):
        if abs((y - cy['a']) / cy['b'] - 5.0) > 1.5:
            continue
        row = dark[y, int(xl + th + 2):int(xr - th - 1)]
        idx = np.flatnonzero(row)
        if idx.size < 0.3 * row.size:
            continue
        runs = 1 + int((np.diff(idx) > 1).sum())
        if runs >= 8 and (best is None or idx.size > best[1]):
            best = (y, idx.size)
    if best is None:
        return None
    # centre of the line: average over adjacent rows with similar coverage
    y0 = best[0]
    ys = [y for y in range(y0 - 3, y0 + 4)
          if dark[y, int(xl + th + 2):int(xr - th - 1)].sum() >
          0.6 * best[1]]
    yc = float(np.mean(ys))
    return (yc - cy['a']) / cy['b']


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    for f in ('fig16', 'fig17', 'fig19', 'fig21', 'fig22'):
        ap.add_argument('--' + f, help='raster image of ' + f)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'digitized'))
    ap.add_argument('--panels', default=','.join(PANELS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    diag, cal_rows, analysis = [], [], {}
    for panel in args.panels.split(','):
        cfg = PANELS[panel]
        path = getattr(args, cfg['fig'])
        if path is None:
            print('skip %s (no --%s)' % (panel, cfg['fig']))
            continue
        img = load(path)
        dark = dark_mask(img)
        excl = list(cfg['exclude'])
        main_pts, cal, cxy = digitize_axes(
            img, dark, panel, cfg['roi'], cfg['xticks'], cfg['yticks'],
            cfg['xmul'], cfg['series'], cfg['grid'], cfg['expect'], excl)
        cals = [cxy]
        cal_rows.append(cal)
        inset_pts = []
        if 'inset' in cfg:
            ic = cfg['inset']
            inset_pts, cal_i, cxy_i = digitize_axes(
                img, dark, panel + '_inset', ic['roi'], ic['xticks'],
                ic['yticks'], cfg['xmul'], cfg['series'], cfg['grid'],
                ic['expect'], [], thr=0.55, xapprox=ic.get('xapprox'),
                yapprox=ic.get('yapprox'))
            # inset frame edges clip the curves: keep only expected x values
            inset_pts = [p for p in inset_pts
                         if any(abs(p['x_nominal'] - e * cfg['xmul']) < 1e-12
                                for e in ic['expect'])]
            cals.append(cxy_i)
            cal_rows.append(cal_i)
        write_csv(os.path.join(args.out, panel + '.csv'), main_pts)
        if inset_pts:
            write_csv(os.path.join(args.out, panel + '_inset.csv'), inset_pts)
        best = best_rows(main_pts, inset_pts)
        write_csv(os.path.join(args.out, panel + '_best.csv'), best)
        for src, pts in (('main', main_pts), ('inset', inset_pts)):
            for p in pts:
                diag.append(dict(panel=panel, axes=src, **p))
        overlay(img, panel, cfg, main_pts, inset_pts, cals,
                os.path.join(args.out, 'check_%s.png' % panel))
        dash = dashed_level(dark, cxy[2], cxy[1], excl)
        curves = {s_: [(p_['x_nominal'], p_['y']) for p_ in best
                       if p_['series'] == s_] for s_, _, _ in cfg['series']}
        analysis[panel] = dict(
            sweep=SWEEPS[panel], drawn_threshold_line_s=dash,
            series=analyse(panel, best, dash),
            sim_vs_exp=compare(best), _curve=curves,
            inset_minus_main=(inset_consistency(main_pts, inset_pts)
                              if inset_pts else None),
            resolution_s_per_px=dict(
                main=-1.0 / cal['y_b'],
                inset=(-1.0 / cal_rows[-1]['y_b'] if inset_pts else None)))
        n = {s: sum(p['series'] == s for p in main_pts)
             for s, _, _ in cfg['series']}
        nfb = sum(not p['method'].startswith('peak')
                  for p in main_pts + inset_pts)
        print('%-7s main %s  inset %d  fallback %d  x-rms %.2f px  '
              'y-rms %.2f px' % (panel, n, len(inset_pts), nfb,
                                 cal['x_rms_px'], cal['y_rms_px']))
    with open(os.path.join(args.out, 'diagnostics.csv'), 'w',
              newline='') as f:
        keys = ['panel', 'axes', 'series', 'x', 'y', 'x_nominal', 'snap_res',
                'px', 'py', 'score', 'method']
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        for d in diag:
            wr.writerow({k: (('%.6g' % d[k]) if isinstance(d[k], float)
                             else d[k]) for k in keys})
    with open(os.path.join(args.out, 'calibration.csv'), 'w',
              newline='') as f:
        keys = list(cal_rows[0].keys())
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for c in cal_rows:
            wr.writerow(c)
    with open(os.path.join(args.out, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=1)    # includes the best curves
    txt = report(analysis)
    print(txt)
    with open(os.path.join(args.out, 'analysis.txt'), 'w') as f:
        f.write(txt + '\n')


def _curve_val(a, s_, x):
    for k, v in a['_curve'][s_]:
        if abs(k - x) <= 1e-12 * max(1.0, abs(x)):
            return v
    return float('nan')


def _maxdiff(a, xmax, s1='Experiment', s2='Simulation'):
    c1 = {round(k, 12): v for k, v in a['_curve'][s1]}
    c2 = {round(k, 12): v for k, v in a['_curve'][s2]}
    ks = [k for k in sorted(set(c1) & set(c2)) if k <= xmax * (1 + 1e-9)
          and c1[k] < CAP and c2[k] < CAP]
    d = np.array([abs(c2[k] - c1[k]) for k in ks])
    r = np.array([abs(c2[k] - c1[k]) / c1[k] for k in ks])
    return float(d.max()), float(r.max()), len(ks)


def claims(an):
    """Statements of the paper text (Sec. 5) against the digitized data."""
    out = ['', 'PAPER STATEMENTS vs DIGITIZED DATA']

    def S(p_, s_):
        return an[p_]['series'][s_]

    def rng(p_, s_, key='eff_range_5s'):
        r = S(p_, s_)[key]
        sw = an[p_]['sweep']['sw']
        return 'None' if r is None else '[%s, %s]' % (_fx(sw, r[0]),
                                                      _fx(sw, r[1]))
    try:
        e, m = S('fig16c', 'Experiment'), S('fig16c', 'Simulation')
        out.append('  16c "min at mu=9e-6": exp opt %s near %s; sim opt %s '
                   'near %s' % (_fx('mu', e['x_opt']),
                                [_fx('mu', v) for v in e['near_opt']],
                                _fx('mu', m['x_opt']),
                                [_fx('mu', v) for v in m['near_opt']]))
        c = an['fig16c']['sim_vs_exp']
        out.append('  16c "max rel err <12%%, abs <0.4 s across the entire '
                   'stable range": stable range %.1f%% / %.2f s; effective '
                   'range %.1f%% / %.2f s' % (
                       100 * c['stable']['max_rel'], c['stable']['max_abs'],
                       100 * c['effective']['max_rel'],
                       c['effective']['max_abs']))
        out.append('  16c "effective conv. for mu in [4e-6,1e-5] (both)": exp '
                   '%s, sim %s (sim t(1.1e-5)=%.2f s)' % (
                       rng('fig16c', 'Experiment'),
                       rng('fig16c', 'Simulation'),
                       _curve_val(an['fig16c'], 'Simulation', 1.1e-5)))
        out.append('  16d "effective for L in [48,128]": exp %s, sim %s at '
                   '5 s; with the drawn line (%.2f s) sim %s (t_sim(48)=%.2f)'
                   % (rng('fig16d', 'Experiment'), rng('fig16d', 'Simulation'),
                      an['fig16d']['drawn_threshold_line_s'],
                      rng('fig16d', 'Simulation', 'eff_range_drawn_line'),
                      _curve_val(an['fig16d'], 'Simulation', 48)))
        e, m = S('fig16d', 'Experiment'), S('fig16d', 'Simulation')
        out.append('  16d "optimal L 112 (sim) / 100 (exp)": exp opt %g near '
                   '%s; sim opt %g near %s' % (e['x_opt'], e['near_opt'],
                                               m['x_opt'], m['near_opt']))
        out.append('  16d "relative error within 13%%": max %.1f%%' % (
            100 * an['fig16d']['sim_vs_exp']['stable']['max_rel']))
        out.append('  17a/b "FEM 17.64 s at mu=5e-6, L=128": 17b FEM(5e-6) = '
                   '%.2f s' % _curve_val(an['fig17b'], 'Exp_FEM', 5e-6))
        out.append('  17b/c "FEM minima 4.48 s and 3.83 s": %.2f s (mu=%s), '
                   '%.2f s (L=%g)' % (
                       S('fig17b', 'Exp_FEM')['t_min'],
                       _fx('mu', S('fig17b', 'Exp_FEM')['x_opt']),
                       S('fig17c', 'Exp_FEM')['t_min'],
                       S('fig17c', 'Exp_FEM')['x_opt']))
        out.append('  17c "FEM diverges for L>60": FEM t(60)=%.2f s, t(64)='
                   '%.2f s' % (_curve_val(an['fig17c'], 'Exp_FEM', 60),
                               _curve_val(an['fig17c'], 'Exp_FEM', 64)))
        for p_, sw in (('fig19c', 'mu'), ('fig19d', 'L'), ('fig21c', 'mu'),
                       ('fig21d', 'L')):
            e, m = S(p_, 'Experiment'), S(p_, 'Simulation')
            out.append('  %s optima: exp %s (near %s), sim %s (near %s); '
                       'effective exp %s sim %s' % (
                           p_[3:], _fx(sw, e['x_opt']),
                           [_fx(sw, v) for v in e['near_opt']],
                           _fx(sw, m['x_opt']),
                           [_fx(sw, v) for v in m['near_opt']],
                           rng(p_, 'Experiment'), rng(p_, 'Simulation')))
        out.append('  19 "abs discrepancy < 1.3 s in effective regions": '
                   '19c %.2f s, 19d %.2f s' % (
                       an['fig19c']['sim_vs_exp']['effective']['max_abs'],
                       an['fig19d']['sim_vs_exp']['effective']['max_abs']))
        d1 = _maxdiff(an['fig21c'], 1.4e-5)
        d2 = _maxdiff(an['fig21d'], 140)
        out.append('  21 "curves match closely for mu<=1.4e-5 and L<=140": '
                   'max |diff| %.2f s (%.0f%%) for mu<=1.4e-5, %.2f s (%.0f%%)'
                   ' for L<=140' % (d1[0], 100 * d1[1], d2[0], 100 * d2[1]))
        out.append('  22a "minima 3.48 s (sim) and 3.81 s (exp)": sim %.2f s '
                   '(mu2=%s), exp %.2f s (mu2=%s)' % (
                       S('fig22a', 'Simulation')['t_min'],
                       _fx('mu', S('fig22a', 'Simulation')['x_opt']),
                       S('fig22a', 'Experiment')['t_min'],
                       _fx('mu', S('fig22a', 'Experiment')['x_opt'])))
        out.append('  22 "discrepancies within 1.5 s across effective regions"'
                   ': 22a %.2f s, 22b %.2f s' % (
                       an['fig22a']['sim_vs_exp']['effective']['max_abs'],
                       an['fig22b']['sim_vs_exp']['effective']['max_abs']))
    except KeyError as exc:      # some panels not processed
        out.append('  (incomplete: %s)' % exc)
    return out


def _fx(sw, v):
    return ('%.1fe-6' % (v * 1e6)) if sw == 'mu' else ('%g' % v)


def report(analysis):
    L = ['Digitized convergence-time curves, Ouyang et al. TWS 228 (2026) '
         '115090', '(best = inset value where available, else main axes; '
         'capped points t>=%.1f s excluded from optima/fits)' % CAP, '',
         'OPTIMA  (near = grid points within %.2f s of the minimum)'
         % TOL_OPT]
    for panel, a in analysis.items():
        sw = a['sweep']['sw']
        fixname, fixval = a['sweep']['fix']
        for s, d in a['series'].items():
            L.append('  %-7s %-11s %s=%-7g opt %s=%-7s t_min=%.3f s  '
                     'mu*L=%.3e  near=[%s]  eff(<5s)=%s' % (
                         panel, s, fixname, fixval, sw, _fx(sw, d['x_opt']),
                         d['t_min'], d['muL_opt'],
                         ','.join(_fx(sw, v) for v in d['near_opt']),
                         None if d['eff_range_5s'] is None else
                         '[%s,%s]' % tuple(_fx(sw, v)
                                           for v in d['eff_range_5s'])))
    L += ['', 'SLOW BRANCH  (left of optimum, t >= %.1f t_min): '
          't = C/(mu L) LS fit; CV of t*x; log-log slope p (t ~ x^-p);'
          ' offset model t = C2/(mu L) + t0' % SLOW_K]
    for panel, a in analysis.items():
        sw = a['sweep']['sw']
        for s, d in a['series'].items():
            if 'C' not in d:
                L.append('  %-7s %-11s  (fewer than 3 slow-branch points)'
                         % (panel, s))
                continue
            extra = ''
            if 't_times_lammin_cv' in d:
                extra = '  [lam_min model: CV=%.2f p=%.2f]' % (
                    d['t_times_lammin_cv'], d['loglog_exponent_vs_lammin'])
            L.append('  %-7s %-11s n=%d %s=[%s..%s]  CV(t*%s)=%.3f  p=%.2f  '
                     'C=%.3e s (rel rms %.1f%%, max %.1f%%)  C2=%.3e t0=%.2f s'
                     '  kappa=ln20/(C fs)=%.3g%s' % (
                         panel, s, len(d['slow_x']), sw,
                         _fx(sw, d['slow_x'][0]), _fx(sw, d['slow_x'][-1]),
                         sw, d['t_times_x_cv'], d['loglog_exponent'], d['C'],
                         100 * d['C_rel_rms'], 100 * d['C_rel_max'],
                         d['C_with_offset'], d['t0_offset'], d['kappa'],
                         extra))
    L += ['', 'SIM vs EXP (common uncapped points; effective = both < 5 s)']
    for panel, a in analysis.items():
        c = a['sim_vs_exp']
        if not c:
            continue
        sw = a['sweep']['sw']
        for k, v in c.items():
            L.append('  %-7s %-9s n=%2d  max rel %.1f%% (at %s)  max abs '
                     '%.2f s (at %s)' % (panel, k, v['n'], 100 * v['max_rel'],
                                        _fx(sw, v['x_at_max_rel']),
                                        v['max_abs'],
                                        _fx(sw, v['x_at_max_abs'])))
    L += ['', 'DIVERGENCE ONSET (last converged / first capped point right '
          'of the optimum)']
    for panel, a in analysis.items():
        sw = a['sweep']['sw']
        for s_, d in a['series'].items():
            if 'divergence_bracket_x' in d:
                bx, bm = d['divergence_bracket_x'], d['divergence_bracket_muL']
                L.append('  %-7s %-11s %s in (%s, %s]  mu*L in (%.3e, %.3e]'
                         % (panel, s_, sw, _fx(sw, bx[0]), _fx(sw, bx[1]),
                            bm[0], bm[1]))
    L += ['', 'C CONSISTENCY  (same case: mu-sweep vs L-sweep; exp vs sim)']
    for c1, c2, lab in (('fig16c', 'fig16d', '120 Hz 1-ch'),
                        ('fig19c', 'fig19d', '280 Hz 1-ch'),
                        ('fig21c', 'fig21d', '85 Hz 2-ch')):
        if c1 not in analysis or c2 not in analysis:
            continue
        g = {}
        for pn in (c1, c2):
            for s_ in ('Experiment', 'Simulation'):
                g[(pn, s_)] = analysis[pn]['series'].get(s_, {}).get('C')
        if None in g.values():
            continue
        L.append('  %-11s C_mu exp %.3e sim %.3e | C_L exp %.3e sim %.3e | '
                 'C_L/C_mu exp %.2f sim %.2f | C_exp/C_sim mu %.2f L %.2f' % (
                     lab, g[(c1, 'Experiment')], g[(c1, 'Simulation')],
                     g[(c2, 'Experiment')], g[(c2, 'Simulation')],
                     g[(c2, 'Experiment')] / g[(c1, 'Experiment')],
                     g[(c2, 'Simulation')] / g[(c1, 'Simulation')],
                     g[(c1, 'Experiment')] / g[(c1, 'Simulation')],
                     g[(c2, 'Experiment')] / g[(c2, 'Simulation')]))
    L += ['', 'CROSS-FIGURE CHECK (same experimental sd-DMDc data plotted '
          'twice)']
    for pa, sa, pb, sb in (('fig16c', 'Experiment', 'fig17b', 'Exp_sdDMDc'),
                           ('fig16d', 'Experiment', 'fig17c', 'Exp_sdDMDc')):
        if pa in analysis and pb in analysis:
            ra = {round(k, 12): v for k, v in analysis[pa]['_curve'][sa]}
            rb = {round(k, 12): v for k, v in analysis[pb]['_curve'][sb]}
            ks = sorted(set(ra) & set(rb))
            dd = np.array([rb[k] - ra[k] for k in ks])
            L.append('  %s vs %s: n=%d  mean(%s-%s)=%+.3f s  max |diff|=%.3f s'
                     % (pa, pb, len(ks), pb, pa, dd.mean(),
                        np.abs(dd).max()))
    L += claims(analysis)
    L += ['', 'QUALITY: resolution (s/px), inset-minus-main offset, drawn '
          'threshold line']
    for panel, a in analysis.items():
        r = a['resolution_s_per_px']
        im = a['inset_minus_main']
        L.append('  %-7s main %.4f s/px  inset %s  inset-main mean %s max %s'
                 '  dashed line at %s s' % (
                     panel, r['main'],
                     '-' if r['inset'] is None else '%.4f s/px' % r['inset'],
                     '-' if im is None else '%+.3f' % im['mean'],
                     '-' if im is None else '%.3f' % im['max_abs'],
                     '-' if a['drawn_threshold_line_s'] is None else
                     '%.2f' % a['drawn_threshold_line_s']))
    return '\n'.join(L)


if __name__ == '__main__':
    main()
