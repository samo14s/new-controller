"""Peak frequencies of the FFT panels of Figs. 10 and 12 (raster JPEGs in the PDF).

Each FFT panel spans 0..400 Hz between its left and right frame lines.  For every
image column inside a panel the height of the blue spectrum above the bottom frame
is measured; spectral lines are local maxima of that profile.  Each peak is then
labelled as a spindle harmonic (multiple of the spindle frequency, which includes the
tooth-passing harmonics) or as a non-synchronous line (chatter candidate).

Frequency resolution of the digitisation: 400 Hz / (panel width in px) ~ 1.2-1.3 Hz/px;
quoted uncertainty is +-1.5 px.

Outputs: fft_peaks.csv, fft_panels.json
Usage: python3 extract_fft_peaks.py [paper.pdf] [scratch_dir]
"""
import csv
import json
import os
import sys

import numpy as np
import pymupdf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/tmp/claude-0/-home-user-new-controller/d42dfb76-3818-53d8-8bcc-e62ff0a55916/scratchpad"
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "paper.pdf")
OUT_IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCR, "tg")
os.makedirs(OUT_IMG, exist_ok=True)
doc = pymupdf.open(PDF)


def image(page_idx, want):
    page = doc[page_idx]
    for im in page.get_images(full=True):
        r = page.get_image_rects(im[0])[0]
        if want(r):
            pix = pymupdf.Pixmap(doc, im[0])
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            path = os.path.join(OUT_IMG, "fft_src_p%d_x%d.png" % (page_idx, im[0]))
            pix.save(path)
            return np.asarray(Image.open(path).convert("RGB")).astype(int)
    raise RuntimeError("image not found")


# (figure, page index, placement test, panel captions top->bottom from the figure captions)
FIGS = [
    ("Fig10", 7, lambda r: r.x0 > 300 and r.y0 > 200,
     [("A", 1300, 8.0, "stable"), ("B", 1300, 10.0, "chatter")]),
    ("Fig12", 8, lambda r: r.x0 > 300 and r.y1 > 400,
     [("A", 2800, 0.5, "stable"), ("B", 2800, 1.5, "chatter"),
      ("C", 2700, 3.0, "stable"), ("D", 2700, 4.0, "chatter")]),
]


def frame_lines(im):
    g = im.mean(axis=2)
    H, W = g.shape
    line = (g < 215) & (np.abs(im[..., 0] - im[..., 2]) < 30)
    vcols = []
    for x in range(W):
        col = line[:, x]
        idx = np.nonzero(col)[0]
        if len(idx) < 150:
            continue
        sp = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
        for s in sp:
            if s[-1] - s[0] > 150:
                vcols.append((x, s[0], s[-1]))
    return vcols


rows = []
panels_out = []
for fig, pidx, want, caps in FIGS:
    im = image(pidx, want)
    H, W, _ = im.shape
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    blue = (B > 120) & (R < 120) & (B - R > 50)
    v = frame_lines(im)
    # right-hand panels: frame columns with x > W/2; group them per vertical span
    right = [t for t in v if t[0] > W * 0.6]
    spans = {}
    for x, y0, y1 in right:
        key = None
        for k in spans:
            if abs(k[0] - y0) < 15:
                key = k
        if key is not None and y1 > key[1]:
            spans[(key[0], y1)] = spans.pop(key)
            key = (key[0], y1)
        spans.setdefault(key or (y0, y1), []).append(x)
    panels = sorted([(k, sorted(xs)) for k, xs in spans.items()], key=lambda p: p[0][0])
    # keep, per span, the leftmost and rightmost frame columns (merge 1-2 px thick lines)
    clean = []
    for (y0, y1), xs in panels:
        groups = np.split(np.array(xs), np.where(np.diff(xs) > 2)[0] + 1)
        xs_c = [float(np.mean(gp)) for gp in groups]
        clean.append(dict(y_top=int(y0), y_bot=int(y1), x_left=xs_c[0],
                          x_right=xs_c[-1] if len(xs_c) > 1 else None))
    # a panel drawn without a right frame line (top panel of Fig. 12): give it the median
    # width of the other panels of the same figure
    wid = np.median([c["x_right"] - c["x_left"] for c in clean if c["x_right"] is not None])
    for c in clean:
        c["right_frame"] = "detected" if c["x_right"] is not None else "assumed (median width)"
        if c["x_right"] is None:
            c["x_right"] = c["x_left"] + float(wid)
    assert len(clean) == len(caps), (fig, clean)
    for pan, (lab, rpm, depth, state) in zip(clean, caps):
        xl, xr, yt, yb = pan["x_left"], pan["x_right"], pan["y_top"], pan["y_bot"]
        hz_px = 400.0 / (xr - xl)
        prof = np.zeros(W)
        for x in range(int(np.ceil(xl)) + 2, int(np.floor(xr)) - 1):
            ys = np.nonzero(blue[yt + 2:yb - 1, x])[0]
            if len(ys):
                prof[x] = (yb - (yt + 2 + ys[0])) / (yb - yt)     # fraction of panel height
        # spectral lines are drawn ~3 px wide: a line is a run of adjacent columns of (nearly)
        # equal height; a step of more than 25 % in height starts a new line
        segs, cur = [], []
        for x in range(W):
            h = prof[x]
            if h < 0.04:
                if cur:
                    segs.append(cur)
                cur = []
                continue
            if cur and abs(h - prof[cur[-1]]) > 0.25 * max(h, prof[cur[-1]]):
                segs.append(cur)
                cur = []
            cur.append(x)
        if cur:
            segs.append(cur)
        # a 1-px segment glued to a neighbouring segment is an anti-aliasing shoulder, not a line
        keep = []
        for k, sg in enumerate(segs):
            touch = (k > 0 and segs[k - 1][-1] == sg[0] - 1) or (k + 1 < len(segs) and segs[k + 1][0] == sg[-1] + 1)
            if len(sg) == 1 and touch:
                continue
            keep.append(sg)
        segs = keep
        fs = rpm / 60.0
        lines = [((sg[0] + sg[-1]) / 2.0, max(prof[sg])) for sg in segs]
        f_raw = [(xm - xl) * hz_px for xm, _ in lines]
        is_sync = [abs(f / fs - round(f / fs)) * fs < 2.0 * hz_px for f in f_raw]
        # per-panel offset from the synchronous lines (they must sit on k*fs exactly)
        err = [f - round(f / fs) * fs for f, sy in zip(f_raw, is_sync) if sy]
        off = float(np.mean(err)) if len(err) >= 2 else 0.0
        top = max(h for _, h in lines)
        for (xm, h), f, sy in zip(lines, f_raw, is_sync):
            fc = f - off
            n = fc / fs
            sync = abs(n - round(n)) * fs < 1.5 * hz_px
            cls = (("spindle harmonic %d" % round(n)) + (" (tooth-passing x%d)" % (round(n) // 4)
                                                         if round(n) % 4 == 0 else "")) if sync else \
                "non-synchronous (chatter candidate)"
            rows.append([fig, lab, rpm, depth, state, round(f, 1), round(fc, 1), round(1.5 * hz_px, 1),
                         round(h, 3), round(h / top, 3), round(n, 2), cls, len(segs[[l[0] for l in lines].index(xm)])])
        pan.update(fig=fig, label=lab, rpm=rpm, depth_mm=depth, caption_state=state, hz_per_px=hz_px,
                   sync_line_offset_Hz=dict(n=len(err), mean=round(off, 2),
                                            max_abs=round(float(np.max(np.abs(err))), 2) if err else None))
        panels_out.append(pan)

with open(os.path.join(HERE, "fft_peaks.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["figure", "cut", "rpm", "depth_mm", "caption_state", "freq_Hz_raw", "freq_Hz_offset_corrected", "u_freq_Hz",
                "height_frac_of_panel", "height_rel_to_tallest", "freq_over_spindle_freq", "class", "width_px"])
    w.writerows(rows)
with open(os.path.join(HERE, "fft_panels.json"), "w") as fh:
    json.dump(panels_out, fh, indent=1)
print(open(os.path.join(HERE, "fft_peaks.csv")).read())
