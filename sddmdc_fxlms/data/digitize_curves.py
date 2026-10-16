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
     blobs).  Local maxima above a threshold are marker centres (sub-pixel by a
     3-point parabola).  Marker size is chosen automatically per axes among a
     few candidates (largest median peak score).
  6. Data: pixel -> data with the calibration; x is snapped to a nominal grid
     (mu on multiples of 0.5e-6; L on the set used in the paper) and the snap
     residual is reported.  Expected grid points that were not found get a
     local re-search with a lower threshold (method="local") and, failing that,
     the centre of the series-coloured run in the nominal column
     (method="column").

Outputs (in --out):
  <panel>.csv          main-axes points           columns series,x,y,x_nominal
  <panel>_inset.csv    inset points (if any)      same columns
  <panel>_best.csv     inset value where available (higher resolution), else
                       main-axes value            same columns
  diagnostics.csv      every point with pixel coordinates, score, method
  calibration.csv      fitted pixel->data maps and residuals per axes
  check_<panel>.png    extracted points drawn on the original image
  analysis.json / analysis.txt   optima, mu*L products, slow-branch fits

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


SIZES = {'o': [(4.5, 2.5), (5.0, 2.5), (5.5, 3.0), (6.0, 3.0), (6.5, 3.0),
               (3.5, 2.0), (4.0, 2.0)],
         's': [(4.5, 2.5), (5.0, 2.5), (5.5, 3.0), (6.0, 3.0), (6.5, 3.0),
               (3.5, 2.0), (4.0, 2.0)]}


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
            top = sorted([p[2] for p in pk], reverse=True)[:nexp]
            if not top:
                continue
            q = float(np.median(top)) * min(1.0, len(top) / nexp)
            if best is None or q > best[0]:
                best = (q, r, w, sc, pk)
        _, r, w, sc, pk = best
        sizes_used[sname] = (r, w)
        maps[sname] = (sc, score_map(S, O, kind, r, w, relaxed=True), r)
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
                    if b is not None and b[0] >= 0.5:
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
MARK = {'o': 'o', 's': 's'}


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
    # panel: (fixed parameter name, fixed value, swept parameter)
    'fig16c': ('L', 128, 'mu'), 'fig16d': ('mu', 1e-5, 'L'),
    'fig17b': ('L', 128, 'mu'), 'fig17c': ('mu', 1e-5, 'L'),
    'fig19c': ('L', 128, 'mu'), 'fig19d': ('mu', 1.2e-5, 'L'),
    'fig21c': ('L', 64, 'mu'), 'fig21d': ('mu', 6e-6, 'L'),
    'fig22a': ('L', 64, 'mu'), 'fig22b': ('mu', 6e-6, 'L'),
}
FS = 10000.0     # dSPACE sampling rate [Hz]
CAP = 19.5       # points at the 20 s cap (divergent / >20 s) are excluded


def analyse(panel, rows):
    fixname, fixval, sw = SWEEPS[panel]
    res = {}
    for s in sorted(set(p['series'] for p in rows)):
        P = sorted([p for p in rows if p['series'] == s],
                   key=lambda p: p['x_nominal'])
        x = np.array([p['x_nominal'] for p in P])
        t = np.array([p['y'] for p in P])
        mu = x if sw == 'mu' else np.full_like(x, fixval)
        L = x if sw == 'L' else np.full_like(x, fixval)
        valid = t < CAP
        if not valid.any():
            continue
        i0 = int(np.flatnonzero(valid)[np.argmin(t[valid])])
        tmin = float(t[i0])
        # slow branch: left of the optimum, t >= 2 tmin, not capped
        slow = (np.arange(len(x)) < i0) & (t >= 2.0 * tmin) & valid
        d = dict(n_points=int(len(x)), x_opt=float(x[i0]), t_min=tmin,
                 muL_opt=float(mu[i0] * L[i0]),
                 near_opt=[float(v) for v in x[(t <= tmin + 0.05) & valid]])
        if slow.sum() >= 3:
            ts, ms, Ls, xs = t[slow], mu[slow], L[slow], x[slow]
            prod = ts * xs                     # t*mu or t*L
            g = 1.0 / (ms * Ls)
            C = float((ts * g).sum() / (g * g).sum())
            rel = (ts - C * g) / ts
            # power law t = K x^-p (log-log LS)
            p_exp, lnK = np.polyfit(np.log(xs), np.log(ts), 1)
            # offset model t = C2/(mu L) + t0
            A = np.vstack([g, np.ones_like(g)]).T
            (C2, t0), *_ = np.linalg.lstsq(A, ts, rcond=None)
            d.update(slow_x=[float(v) for v in xs],
                     slow_t=[float(v) for v in ts],
                     t_times_x_mean=float(prod.mean()),
                     t_times_x_cv=float(prod.std(ddof=1) / prod.mean()),
                     C=C, C_rel_rms=float(np.sqrt(np.mean(rel ** 2))),
                     C_rel_max=float(np.abs(rel).max()),
                     loglog_exponent=float(-p_exp),
                     C_with_offset=float(C2), t0_offset=float(t0),
                     kappa=float(math.log(20.0) / (C * FS)))
        res[s] = d
    return res


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
        analysis[panel] = analyse(panel, best)
        n = {s: sum(p['series'] == s for p in main_pts)
             for s, _, _ in cfg['series']}
        print('%-7s main %s  inset %d  x-rms %.2f px  y-rms %.2f px' % (
            panel, n, len(inset_pts), cal['x_rms_px'], cal['y_rms_px']))
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
        json.dump(analysis, f, indent=1)
    lines = []
    for panel, a in analysis.items():
        fixname, fixval, sw = SWEEPS[panel]
        for s, d in a.items():
            line = ('%-7s %-11s %s=%g  opt %s=%-8g t_min=%.2f s  muL_opt=%.3e'
                    % (panel, s, fixname, fixval, sw, d['x_opt'], d['t_min'],
                       d['muL_opt']))
            if 'C' in d:
                line += ('  slow n=%d CV(t*%s)=%.2f p=%.2f C=%.3e s '
                         '(rel rms %.2f)' % (len(d['slow_x']), sw,
                                            d['t_times_x_cv'], sw and
                                            d['loglog_exponent'], d['C'],
                                            d['C_rel_rms']))
            lines.append(line)
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out, 'analysis.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
