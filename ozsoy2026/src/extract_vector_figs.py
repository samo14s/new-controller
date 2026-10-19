"""Recover the plotted data of the paper's vector figures (Figs. 5, 6a, 8, 9).

The published PDF stores these plots as vector paths, so the curves can be read
back exactly instead of being digitised by eye.  For each figure the axis map
(PDF points -> data units) is fitted by least squares on the tick labels that
PyMuPDF reports, then every polyline of the wanted colour is converted.

Usage:  python extract_vector_figs.py <paper.pdf> <out_dir>
"""
import json
import os
import sys
import numpy as np
import pymupdf


def frame_ticks(drawings, clip):
    """Tick-mark positions of a MATLAB-style axes box inside clip.

    The tick labels are glyph outlines (no text), so the axis map is fitted on
    the tick marks: short grey strokes that sit on the bottom and left edges.
    Returns sorted unique x positions (bottom edge) and y positions (left edge).
    """
    segs = []
    for d in drawings:
        c = d.get("color")
        if c is None or max(c) > 0.3 or len(d["items"]) != 1 or d["items"][0][0] != "l":
            continue
        p0, p1 = d["items"][0][1], d["items"][0][2]
        if not (clip[0] <= min(p0.x, p1.x) and max(p0.x, p1.x) <= clip[2]
                and clip[1] <= min(p0.y, p1.y) and max(p0.y, p1.y) <= clip[3]):
            continue
        segs.append((p0.x, p0.y, p1.x, p1.y))
    horiz = [s for s in segs if abs(s[1] - s[3]) < 1e-3]
    vert = [s for s in segs if abs(s[0] - s[2]) < 1e-3]
    long_h = max(horiz, key=lambda s: abs(s[2] - s[0]))
    width = abs(long_h[2] - long_h[0])
    frame_h = [s for s in horiz if abs(s[2] - s[0]) > 0.9 * width]
    y_bot = max(s[1] for s in frame_h)
    x_left = min(min(s[0], s[2]) for s in frame_h)
    xt = sorted({round(s[0], 2) for s in vert
                 if abs(s[1] - s[3]) < 6 and abs(max(s[1], s[3]) - y_bot) < 1e-2})
    yt = sorted({round(s[1], 2) for s in horiz
                 if abs(s[2] - s[0]) < 6 and abs(min(s[0], s[2]) - x_left) < 1e-2})
    return xt, yt[::-1]


def fit_axis(pos, values):
    pos, values = np.asarray(pos, float), np.asarray(values, float)
    if len(pos) != len(values):
        raise ValueError(f"{len(pos)} tick marks for {len(values)} tick values: {pos}")
    A = np.vstack([np.ones_like(pos), pos]).T
    p = np.linalg.lstsq(A, values, rcond=None)[0]
    return p, float(np.abs(values - A @ p).max())


def polylines(drawings, color, rect, tol=0.02):
    """All line items of a given stroke colour within rect, as point lists."""
    out = []
    for d in drawings:
        col = d.get("color")
        if col is None or np.max(np.abs(np.array(col) - np.array(color))) > tol:
            continue
        r = d["rect"]
        if not (r.x0 >= rect[0] - 1 and r.x1 <= rect[2] + 1 and r.y0 >= rect[1] - 1 and r.y1 <= rect[3] + 1):
            continue
        seg = []
        for it in d["items"]:
            if it[0] == "l":
                p0, p1 = it[1], it[2]
                if seg and (abs(seg[-1][0] - p0.x) > 1e-3 or abs(seg[-1][1] - p0.y) > 1e-3):
                    out.append(seg)
                    seg = []
                if not seg:
                    seg.append((p0.x, p0.y))
                seg.append((p1.x, p1.y))
            elif it[0] == "c":
                p0, p3 = it[1], it[4]
                if not seg:
                    seg.append((p0.x, p0.y))
                seg.append((p3.x, p3.y))
        if seg:
            out.append(seg)
    return out


def convert(segs, px, py):
    return [[(px[0] + px[1] * x, py[0] + py[1] * y) for x, y in s] for s in segs]


FIGS = {
    # page index, clip of one axes box, tick values read off the rendered figure,
    # curves by RGB stroke colour
    "fig5_frf": dict(page=5, clip=(30, 205, 300, 340),
                     xticks=[50, 100, 150, 200], yticks=[0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6, 2.5e-6],
                     curves={"measured": (0.0, 0.0, 1.0), "identified": (1.0, 0.0, 0.0)}),
    "fig6a_closed_loop_frf": dict(page=6, clip=(80, 60, 300, 250),
                                  xticks=list(range(0, 301, 50)), yticks=list(range(-155, -109, 5)),
                                  curves={"uncontrolled": (0.0, 0.0, 0.0),
                                          "500rpm_g1487": (0.0706, 0.4392, 0.7480),
                                          "1300rpm_g1162": (0.2314, 0.6699, 0.2000),
                                          "2500rpm_g585": (0.9297, 0.6895, 0.1294),
                                          "3000rpm_g797": (0.5176, 0.0901, 0.8203)}),
    "fig8_gains": dict(page=7, clip=(30, 255, 300, 400),
                       xticks=list(range(500, 3001, 500)), yticks=list(range(200, 1601, 200)),
                       curves={"gain": (0.0, 0.4470, 0.7402)}),
    "fig9_sld": dict(page=7, clip=(30, 425, 300, 600),
                     xticks=list(range(500, 3001, 500)), yticks=list(range(0, 31, 5)),
                     curves={"blue": (0.0, 0.4470, 0.7402)}),
}


def main(pdf, outdir, probe=False):
    os.makedirs(outdir, exist_ok=True)
    doc = pymupdf.open(pdf)
    report = {}
    for name, f in FIGS.items():
        page = doc[f["page"]]
        drawings = page.get_drawings()
        xt, yt = frame_ticks(drawings, f["clip"])
        if probe:
            print(name, "x ticks", xt, "\n   y ticks", yt)
            continue
        px, ex = fit_axis(xt, f["xticks"])
        if f["yticks"] is None:
            report[name] = dict(note="y ticks not resolved", x_ticks=xt, y_ticks=yt)
            continue
        py, ey = fit_axis(yt, f["yticks"])
        data = {}
        for cname, col in f["curves"].items():
            for d in drawings:
                c = d.get("color")
                if c is None or np.max(np.abs(np.array(c) - np.array(col))) > 0.02:
                    continue
                r = d["rect"]
                cl = f["clip"]
                if not (r.x0 >= cl[0] and r.x1 <= cl[2] and r.y0 >= cl[1] and r.y1 <= cl[3]):
                    continue
                if len(d["items"]) < 20:
                    continue          # legend swatches and markers
                dashed = d.get("dashes", "[] 0").strip() not in ("[] 0", "")
                key = cname + ("_dashed" if dashed else "")
                data.setdefault(key, []).extend(convert(polylines([d], col, cl), px, py))
        report[name] = dict(x_fit_err=ex, y_fit_err=ey,
                            curves={k: sum(len(s) for s in v) for k, v in data.items()})
        with open(f"{outdir}/{name}.json", "w") as fh:
            json.dump(data, fh)
    if not probe:
        with open(f"{outdir}/axis_fits.json", "w") as fh:
            json.dump(report, fh, indent=1)
        # flat, rpm-sorted point lists used by the comparison scripts
        for out, src, key, head in [("fig8_gain_points", "fig8_gains", "gain", "rpm,gain_Vs_per_m"),
                                    ("fig9_controlled_points", "fig9_sld", "blue", "rpm,depth_mm"),
                                    ("fig9_uncontrolled_points", "fig9_sld", "blue_dashed", "rpm,depth_mm")]:
            with open(f"{outdir}/{src}.json") as fh:
                pts = np.array([p for seg in json.load(fh)[key] for p in seg])
            np.savetxt(f"{outdir}/{out}.csv", pts[np.argsort(pts[:, 0])], delimiter=",",
                       header=head, comments="")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], probe=len(sys.argv) > 3)
