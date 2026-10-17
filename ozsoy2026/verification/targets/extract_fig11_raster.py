"""Digitise Fig. 11 of Ozsoy, Sims, Ozturk (2026) (single-gain 'alternative optimisation' SLD).

Fig. 11 is embedded as a JPEG (1035 x 653 px) on PDF page index 8 (journal p.254).
Steps
  1. extract the embedded image with PyMuPDF (kept in a scratch folder, not here);
  2. axis calibration from the light-grey grid lines (x: 500..3000 rpm, y: 0..30 mm),
     located with sub-pixel centroids, then a least-squares linear fit;
  3. markers by colour thresholding + connected components
       red  -> controlled chatter (x), green -> controlled stable (o),
       black -> uncontrolled '+', diamond and the letters A-D;
  4. the two blue curves: per image column, the topmost blue run is the solid
     (controlled) line, a lower, separate blue run is the dashed (uncontrolled) line;
  5. coarse features (lobe peaks and minima) of both curves.

Outputs (this folder): fig11_markers.csv, fig11_curves.csv, fig11_features.json,
                       fig11_axis_check.json
Usage:  python3 extract_fig11_raster.py [paper.pdf] [scratch_dir]
"""
import csv
import json
import os
import sys

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/tmp/claude-0/-home-user-new-controller/d42dfb76-3818-53d8-8bcc-e62ff0a55916/scratchpad"
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "paper.pdf")
OUT_IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCR, "tg")
os.makedirs(OUT_IMG, exist_ok=True)

doc = pymupdf.open(PDF)
page = doc[8]
imgs = page.get_images(full=True)
# Fig. 11 is the image placed in the upper-left column of the page
xref = None
for im in imgs:
    r = page.get_image_rects(im[0])[0]
    if r.x0 < 100 and r.y1 < 300:
        xref, place = im[0], r
pix = pymupdf.Pixmap(doc, xref)
if pix.n - pix.alpha > 3:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
png = os.path.join(OUT_IMG, "fig11.png")
pix.save(png)
im = np.asarray(Image.open(png).convert("RGB")).astype(int)
H, W, _ = im.shape
R, G, B = im[..., 0], im[..., 1], im[..., 2]
gray = im.mean(axis=2)

# ------------------------------------------------------------ 1. grid lines
light = (gray > 200) & (gray < 245) & (np.abs(R - B) < 12)


def line_centres(profile, thr, nexp):
    idx = np.where(profile > thr)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    cents = []
    for g in groups:
        lo, hi = max(g[0] - 1, 0), g[-1] + 2
        w = profile[lo:hi].astype(float)
        cents.append(float((np.arange(lo, hi) * w).sum() / w.sum()))
    return cents


# vertical grid lines: count light pixels per column inside the plot rows
colprof = light[30:580, :].sum(axis=0)
xc = [c for c in line_centres(colprof, 250, 6) if 100 < c < 1020]
rowprof = light[:, 60:1020].sum(axis=1)
yc = [c for c in line_centres(rowprof, 400, 7) if 60 < c < 600]
XV = [500, 1000, 1500, 2000, 2500, 3000]
YV = [30, 25, 20, 15, 10, 5, 0]
assert len(xc) == 6, xc
assert len(yc) == 7, yc


def linfit(p, v):
    A = np.vstack([np.ones(len(p)), p]).T
    c = np.linalg.lstsq(A, np.asarray(v, float), rcond=None)[0]
    return c, float(np.abs(A @ c - v).max())


cx_, ex = linfit(xc, XV)
cy_, ey = linfit(yc, YV)
to_rpm = lambda x: cx_[0] + cx_[1] * x
to_mm = lambda y: cy_[0] + cy_[1] * y
axis = dict(image_px=[W, H], placed_on_pdf_page_rect=[place.x0, place.y0, place.x1, place.y1],
            x_grid_px=xc, y_grid_px=yc, rpm_per_px=cx_[1], mm_per_px=-cy_[1],
            x_fit_resid_rpm=ex, y_fit_resid_mm=ey)

# ------------------------------------------------------------ 2. markers
red = (R > 180) & (G < 110) & (B < 110)
green = (G > 180) & (R < 150) & (B < 150)
black = (R < 90) & (G < 90) & (B < 90)
plot_box = np.zeros_like(red)
plot_box[int(yc[0]) + 2:int(yc[-1]) - 1, int(xc[0]) - 120:W - 6] = True   # inside axes
legend = np.zeros_like(red)
legend[20:125, 760:W] = True
st = np.ones((3, 3))

rows = []
lab, n = ndi.label(red & plot_box, st)
for i, o in enumerate(ndi.find_objects(lab)):
    m = lab[o] == i + 1
    if m.sum() < 30:
        continue
    yy, xx = np.nonzero(m)
    cy = (o[0].start + o[0].stop - 1) / 2
    cx = (o[1].start + o[1].stop - 1) / 2
    rows.append(dict(kind="controlled_chatter", glyph="red x", x_px=cx, y_px=cy,
                     w_px=o[1].stop - o[1].start, h_px=o[0].stop - o[0].start, npx=int(m.sum())))
# green rings are cut by the red crosses: merge green blobs whose centres are close
lab, n = ndi.label(green & plot_box, st)
blobs = []
for i, o in enumerate(ndi.find_objects(lab)):
    m = lab[o] == i + 1
    if m.sum() < 10:
        continue
    blobs.append([o[0].start, o[0].stop, o[1].start, o[1].stop, int(m.sum())])
merged = []
for b in sorted(blobs, key=lambda b: b[2]):
    for mb in merged:
        if abs((b[2] + b[3]) / 2 - (mb[2] + mb[3]) / 2) < 15 and abs(b[0] - mb[0]) < 40:
            mb[0], mb[1] = min(mb[0], b[0]), max(mb[1], b[1])
            mb[2], mb[3] = min(mb[2], b[2]), max(mb[3], b[3])
            mb[4] += b[4]
            break
    else:
        merged.append(list(b))
for y0, y1, x0, x1, npx in merged:
    # the ring's left/right extremes are intact, so its horizontal span gives the diameter; the
    # vertical centre is taken as bottom - radius because the top arc is partly hidden by the red x
    rad = (x1 - x0) / 2
    rows.append(dict(kind="controlled_stable", glyph="green circle", x_px=(x0 + x1 - 1) / 2,
                     y_px=(y1 - 1) - rad + 0.5, w_px=x1 - x0, h_px=y1 - y0, npx=npx))
# black objects inside the plot (not legend, not tick labels)
lab, n = ndi.label(black & plot_box & ~legend, st)
blk = []
for i, o in enumerate(ndi.find_objects(lab)):
    m = lab[o] == i + 1
    if m.sum() < 30 or o[1].start < xc[0]:
        continue
    blk.append((o, m))
letters = []
for o, m in blk:
    h, w = m.shape
    if h > 30:      # diamond and '+' touch each other: split them
        ext = np.array([(np.nonzero(r)[0].max() - np.nonzero(r)[0].min() + 1) if r.any() else 0 for r in m])
        # '+' horizontal bar: widest row among the last 18 rows
        lowrows = np.arange(h)[h - 18:]
        y_p = int(lowrows[np.argmax(ext[h - 18:])])
        cols = np.nonzero(m[y_p])[0]
        vcol = np.nonzero(m[h - 3])[0]           # bottom of the vertical bar
        # diamond (outline): its middle row has the largest left-right extent above the '+' bar
        up = ext[:y_p - 3]
        rows_max = np.where(up >= up.max() - 1)[0]
        y_d = float(rows_max.mean())
        xd = np.nonzero(m[int(round(y_d))])[0]
        rows.append(dict(kind="uncontrolled_chatter", glyph="diamond", x_px=o[1].start + (xd.min() + xd.max()) / 2,
                         y_px=o[0].start + y_d, w_px=int(xd.max() - xd.min() + 1), h_px=None, npx=None))
        rows.append(dict(kind="uncontrolled_stable", glyph="black +", x_px=o[1].start + vcol.mean(),
                         y_px=o[0].start + y_p, w_px=int(cols.max() - cols.min() + 1), h_px=None, npx=None))
    else:
        letters.append(dict(x_px=(o[1].start + o[1].stop - 1) / 2, y_px=(o[0].start + o[0].stop - 1) / 2,
                            w=w, h=h, npx=int(m.sum())))
# letters are placed left of their marker; order top->bottom on the image is D, C, B, A
letters.sort(key=lambda d: d["y_px"])
for L, ch in zip(letters, "DCBA"):
    L["letter"] = ch
for r in rows:
    r["label"] = ""
    for L in letters:
        if abs(L["y_px"] - r["y_px"]) < 9 and 0 < r["x_px"] - L["x_px"] < 45:
            r["label"] = L["letter"]
# pixel uncertainty: half a pixel for the centre plus the calibration residual
for r in rows:
    r["rpm"] = to_rpm(r["x_px"])
    r["depth_mm"] = to_mm(r["y_px"])
    r["u_rpm"] = 0.5 * abs(cx_[1]) * 2 + ex       # +-1 px + fit residual
    r["u_mm"] = 0.5 * abs(cy_[1]) * 2 + ey
    r["nominal_rpm"] = int(round(r["rpm"] / 50.0) * 50)
    r["nominal_depth_mm"] = round(r["depth_mm"] * 2) / 2

with open(os.path.join(HERE, "fig11_markers.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["kind", "glyph", "label", "rpm", "depth_mm", "u_rpm", "u_mm", "nominal_rpm",
                "nominal_depth_mm", "x_px", "y_px"])
    for r in sorted(rows, key=lambda r: (r["kind"], r["rpm"])):
        w.writerow([r["kind"], r["glyph"], r["label"], round(r["rpm"], 1), round(r["depth_mm"], 3),
                    round(r["u_rpm"], 1), round(r["u_mm"], 3), r["nominal_rpm"], r["nominal_depth_mm"],
                    round(r["x_px"], 1), round(r["y_px"], 1)])

# ------------------------------------------------------------ 3. curves
blue = (B > 130) & (R < 110) & (G > 60) & (G < 170) & (B - R > 60)
blue &= ~legend
blue[:int(yc[0]) - 60, :] = False           # title area
cur_rows = []
# columns covered by the red/green markers: the solid line is hidden behind them there
occl = np.zeros(W, bool)
for r in rows:
    if r["kind"].startswith("controlled"):
        hw = r["w_px"] / 2 + 2
        occl[int(r["x_px"] - hw):int(r["x_px"] + hw) + 1] = True
for x in range(int(xc[0]) - 50, W - 4):
    ys = np.nonzero(blue[:int(yc[-1]) + 1, x])[0]
    if len(ys) == 0:
        continue
    runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
    top = runs[0]                      # solid (controlled) line is the upper curve everywhere in Fig. 11
    if not occl[x]:
        cur_rows.append(("controlled_top_run", x, top[0], top[-1]))
    if len(runs) > 1:
        bot = runs[-1]                 # dashed (uncontrolled) line, when a separate run exists
        cur_rows.append(("uncontrolled_bottom_run", x, bot[0], bot[-1]))
with open(os.path.join(HERE, "fig11_curves.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["curve", "rpm", "depth_mm_centre", "depth_mm_run_top", "depth_mm_run_bottom", "x_px"])
    for name, x, y0, y1 in cur_rows:
        w.writerow([name.split("_")[0], round(to_rpm(x), 1), round(to_mm((y0 + y1) / 2), 3),
                    round(to_mm(y0 - 0.5), 3), round(to_mm(y1 + 0.5), 3), x])

# ------------------------------------------------------------ 4. features
LW = 1.5        # half line width in px of the solid curve (3 px thick)
ctrl = np.array([(x, y0, y1) for n_, x, y0, y1 in cur_rows if n_.startswith("controlled")], float)
unc = np.array([(x, y0, y1) for n_, x, y0, y1 in cur_rows if n_.startswith("uncontrolled")], float)


def lobe_feats(arr, ranges, kind):
    out = []
    for lo, hi in ranges:
        rr = to_rpm(arr[:, 0])
        m = (rr >= lo) & (rr <= hi)
        if not m.any():
            continue
        if kind == "peak":      # cusp: use the topmost pixel of the run, corrected by half the line width
            j = np.argmin(arr[m, 1])
            y = arr[m, 1][j] + LW - 0.5
        else:                   # minimum: centre of the run
            j = np.argmax((arr[m, 1] + arr[m, 2]) / 2)
            y = (arr[m, 1][j] + arr[m, 2][j]) / 2
        out.append(dict(rpm=round(float(rr[m][j]), 1), depth_mm=round(float(to_mm(y)), 2)))
    return out


feat = dict(
    controlled_peaks=lobe_feats(ctrl, [(450, 560), (600, 720), (900, 1100), (1800, 2100)], "peak"),
    controlled_minima=lobe_feats(ctrl, [(400, 480), (500, 620), (680, 900), (1050, 1300), (2300, 3000)], "min"),
    uncontrolled_peaks=lobe_feats(unc, [(450, 560), (600, 720), (900, 1100), (1800, 2100)], "peak"),
    uncontrolled_minima=lobe_feats(unc, [(395, 470), (500, 620), (680, 900), (1050, 1300), (2300, 3000)], "min"),
)


def value_at(arr, s, interp_ok=True):
    rr = to_rpm(arr[:, 0])
    yc_ = to_mm((arr[:, 1] + arr[:, 2]) / 2)
    j = np.argmin(np.abs(rr - s))
    if abs(rr[j] - s) <= 4:
        return round(float(yc_[j]), 2), "direct"
    if not interp_ok:
        return None, "missing"
    lo, hi = rr < s, rr > s
    if lo.any() and hi.any():
        jl = np.where(lo)[0][np.argmax(rr[lo])]
        jh = np.where(hi)[0][np.argmin(rr[hi])]
        if rr[jh] - rr[jl] < 120:
            return round(float(np.interp(s, [rr[jl], rr[jh]], [yc_[jl], yc_[jh]])), 2), \
                "interpolated %.0f-%.0f rpm" % (rr[jl], rr[jh])
    return None, "missing"


feat["at_speeds"] = []
for s_ in (500, 560, 600, 700, 730, 800, 1000, 1200, 1300, 1500, 2300, 2500, 2700, 2800, 3000):
    c, cm = value_at(ctrl, s_)
    u, um = value_at(unc, s_, interp_ok=False)
    feat["at_speeds"].append(dict(rpm=s_, controlled_mm=c, controlled_how=cm, uncontrolled_mm=u))
feat["uncontrolled_peaks_note"] = ("raster dash gaps miss the cusps: use the Fig. 9 vector peaks "
                                   "(6.78 mm @488, 8.99 @651, 13.64 @977, 29.27 @1952 rpm) instead")
feat["notes"] = ("depth values are coarse (raster, ~0.06 mm/px, ~2.9 rpm/px); peaks use the top pixel "
                 "of the 3-px line plus half a line width, minima use the run centre; dashed-curve "
                 "values are missing in dash gaps.")

with open(os.path.join(HERE, "fig11_features.json"), "w") as fh:
    json.dump(dict(axis=axis, letters=letters, features=feat), fh, indent=1, default=float)
print(json.dumps(axis, indent=1, default=float))
print(open(os.path.join(HERE, "fig11_markers.csv")).read())
print(json.dumps(letters, indent=1, default=float))
print(json.dumps(feat, indent=1, default=float))
