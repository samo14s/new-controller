"""(d) White filled shapes drawn over Fig. 9 in the published PDF.

Reads the vector content of PDF page 8 (journal p. 253) with PyMuPDF and
reports, for every white-filled path inside the Fig. 9 axes:
  * fill / stroke colour, PDF drawing order (seqno), rectangle in PDF points
    and in data units (rpm, mm);
  * whether it is painted after the blue curves;
  * which parts of the solid (controlled) and dashed (uncontrolled) curves,
    grid lines and experiment markers fall inside it.
Then scans every page of the PDF for the same pattern (white fill painted
after a data curve that it overlaps).

Outputs: d_white_shapes.json, d_hidden_controlled_points.csv,
         d_fig9_published_crop.png, d_fig9_vector_reconstruction.png
"""
import numpy as np
import pymupdf

from common import PDF, save_json, save_csv, HERE
import extract_vector_figs as evf

PAGE = 7                         # 0-based page index (journal p. 253)
CLIP = evf.FIGS["fig9_sld"]["clip"]
BLUE = (0.0, 0.4470, 0.7402)
WHITE = (1.0, 1.0, 1.0)


def close(c, ref, tol=0.02):
    return c is not None and max(abs(a - b) for a, b in zip(c, ref)) <= tol


def inside(r, clip):
    return r.x0 >= clip[0] - 1 and r.x1 <= clip[2] + 1 and r.y0 >= clip[1] - 1 and r.y1 <= clip[3] + 1


def seg_in_rect(p0, p1, R):
    """Liang-Barsky: parameter interval [t0, t1] of segment p0->p1 inside rect R."""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - R[0]), (dx, R[2] - x0), (-dy, y0 - R[1]), (dy, R[3] - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return t0, t1


def line_items(d):
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts.append(((it[1].x, it[1].y), (it[2].x, it[2].y)))
        elif it[0] == "c":
            pts.append(((it[1].x, it[1].y), (it[4].x, it[4].y)))
    return pts


def main():
    doc = pymupdf.open(PDF)
    page = doc[PAGE]
    drs = page.get_drawings()
    xt, yt = evf.frame_ticks(drs, CLIP)
    px, ex = evf.fit_axis(xt, evf.FIGS["fig9_sld"]["xticks"])
    py, ey = evf.fit_axis(yt, evf.FIGS["fig9_sld"]["yticks"])
    X = lambda x: px[0] + px[1] * x
    Y = lambda y: py[0] + py[1] * y

    fig = [d for d in drs if inside(d["rect"], CLIP)]
    solid = [d for d in fig if close(d.get("color"), BLUE) and len(d["items"]) > 20
             and d.get("dashes", "[] 0").strip() in ("[] 0", "")]
    dashed = [d for d in fig if close(d.get("color"), BLUE) and len(d["items"]) > 20
              and d.get("dashes", "[] 0").strip() not in ("[] 0", "")]
    assert len(solid) == 1 and len(dashed) == 1
    solid, dashed = solid[0], dashed[0]

    whites = [d for d in fig if close(d.get("fill"), WHITE)]
    axes_box = [min(xt), min(yt), max(xt), max(yt)]
    rep = dict(page_index=PAGE, journal_page=253, axis_fit_err=dict(x_rpm=ex, y_mm=ey),
               solid_curve=dict(seqno=solid["seqno"], n_items=len(solid["items"]),
                                width_pt=solid["width"]),
               dashed_curve=dict(seqno=dashed["seqno"], n_items=len(dashed["items"]),
                                 dashes=dashed["dashes"]),
               white_fills=[])
    covers = []
    for d in whites:
        r = d["rect"]
        role = "?"
        if abs(r.width - (CLIP[2] - CLIP[0])) < 60 and r.height > 100:
            role = "axes background"
        elif any(abs(o["rect"].x0 - r.x0) < 0.2 and abs(o["rect"].y1 - r.y1) < 0.2
                 and o["seqno"] > d["seqno"] and o["type"] == "s" and close(o.get("color"), (0.149,) * 3)
                 for o in fig):
            role = "legend box"
        else:
            role = "overlay rectangle"
        e = dict(seqno=d["seqno"], type=d["type"], fill=d.get("fill"), stroke=d.get("color"),
                 stroke_width=d.get("width"), rect_pt=[round(v, 2) for v in r],
                 rpm=[X(r.x0), X(r.x1)], depth_mm=[Y(r.y1), Y(r.y0)],
                 after_solid_curve=d["seqno"] > solid["seqno"],
                 after_dashed_curve=d["seqno"] > dashed["seqno"], role=role)
        rep["white_fills"].append(e)
        if role == "overlay rectangle":
            covers.append((d, e))

    # what is painted after the overlays
    last_cover = max(d["seqno"] for d, _ in covers)
    rep["painted_after_overlays"] = [
        dict(seqno=d["seqno"], type=d["type"], color=d.get("color"), dashes=d.get("dashes"),
             rect_pt=[round(v, 2) for v in d["rect"]]) for d in fig if d["seqno"] > last_cover]

    # ---- what each overlay hides -------------------------------------------
    def hidden_parts(d_curve, rects):
        segs = line_items(d_curve)
        parts = []
        for p0, p1 in segs:
            for R in rects:
                t = seg_in_rect(p0, p1, R)
                if t is None or t[1] - t[0] < 1e-9:
                    continue
                a = (p0[0] + t[0] * (p1[0] - p0[0]), p0[1] + t[0] * (p1[1] - p0[1]))
                b = (p0[0] + t[1] * (p1[0] - p0[0]), p0[1] + t[1] * (p1[1] - p0[1]))
                parts.append((X(a[0]), Y(a[1]), X(b[0]), Y(b[1]), np.hypot(b[0] - a[0], b[1] - a[1])))
        return parts, sum(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) for p0, p1 in segs)

    rects = [[d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1] for d, _ in covers]
    for d, e in covers:
        R = [d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1]
        for name, cv in (("controlled", solid), ("uncontrolled", dashed)):
            parts, _ = hidden_parts(cv, [R])
            if parts:
                arr = np.array(parts)
                e[f"hides_{name}"] = dict(rpm=[arr[:, [0, 2]].min(), arr[:, [0, 2]].max()],
                                          depth_mm=[arr[:, [1, 3]].min(), arr[:, [1, 3]].max()],
                                          length_pt=arr[:, 4].sum())
            else:
                e[f"hides_{name}"] = None
        # grid lines / markers / other paths painted before and overlapping
        e["other_paths_overlapped"] = []
        for o in fig:
            if o["seqno"] >= d["seqno"] or o in (solid, dashed) or close(o.get("fill"), WHITE):
                continue
            if o["rect"].intersects(d["rect"]) or (o["rect"].width == 0 or o["rect"].height == 0) and \
                    seg_in_rect((o["rect"].x0, o["rect"].y0), (o["rect"].x1, o["rect"].y1), R):
                desc = "grid line" if close(o.get("color"), (0.875,) * 3) else (
                    "marker/glyph" if len(o["items"]) < 40 else "path")
                e["other_paths_overlapped"].append(dict(seqno=o["seqno"], kind=desc, type=o["type"],
                                                        color=o.get("color"), fill=o.get("fill"),
                                                        rect_pt=[round(v, 2) for v in o["rect"]]))

    # union over the overlay rectangles
    summary = {}
    for name, cv in (("controlled", solid), ("uncontrolled", dashed)):
        parts, total = hidden_parts(cv, rects)
        arr = np.array(parts)
        # merge rpm intervals
        iv = sorted((min(a, c), max(a, c)) for a, _, c, _, _ in parts)
        merged = []
        for lo, hi in iv:
            if merged and lo <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        summary[name] = dict(hidden_length_fraction=arr[:, 4].sum() / total,
                             hidden_rpm_intervals=merged,
                             hidden_depth_range_mm=[arr[:, [1, 3]].min(), arr[:, [1, 3]].max()])
    rep["hidden_summary"] = summary

    # vertices of the controlled curve inside the overlays (the plotted data)
    verts = []
    for p0, p1 in line_items(solid):
        verts.append(p0)
    verts.append(line_items(solid)[-1][1])
    rows = []
    for vx, vy in verts:
        ins = [i for i, R in enumerate(rects) if R[0] <= vx <= R[2] and R[1] <= vy <= R[3]]
        rows.append((X(vx), Y(vy), ";".join(str(covers[i][0]["seqno"]) for i in ins)))
    hid = [r for r in rows if r[2]]
    save_csv("d_hidden_controlled_points.csv", ["rpm", "depth_mm", "covered_by_seqno"], hid)
    rep["controlled_vertices_total"] = len(rows)
    rep["controlled_vertices_hidden"] = len(hid)
    if hid:
        h = np.array([(a, b) for a, b, _ in hid])
        rep["controlled_vertices_hidden_rpm"] = [h[:, 0].min(), h[:, 0].max()]
        rep["controlled_vertices_hidden_depth_mm"] = [h[:, 1].min(), h[:, 1].max()]
        # the visible curve just outside the hidden band
        allv = np.array([(a, b) for a, b, _ in rows])
        lo = allv[allv[:, 0] < h[:, 0].min() - 1e-6]
        hi_ = allv[allv[:, 0] > h[:, 0].max() + 1e-6]
        rep["visible_neighbours"] = dict(before=lo[-3:].tolist(), after=hi_[:3].tolist())

    # ---- whole-document scan for white shapes painted over data curves -------
    scan = []
    for pi in range(len(doc)):
        dl = doc[pi].get_drawings()
        curves = [d for d in dl if d["type"] in ("s", "fs") and len(d["items"]) > 20
                  and d.get("color") is not None and not close(d.get("color"), WHITE)
                  and max(d["color"]) - min(d["color"]) > 0.05 or (d["type"] in ("s",) and len(d["items"]) > 100)]
        for w in dl:
            if not close(w.get("fill"), WHITE):
                continue
            under = [c["seqno"] for c in curves if c["seqno"] < w["seqno"] and c["rect"].intersects(w["rect"])
                     and any(seg_in_rect(p0, p1, [w["rect"].x0, w["rect"].y0, w["rect"].x1, w["rect"].y1])
                             for p0, p1 in line_items(c))]
            if under:
                scan.append(dict(page_index=pi, seqno=w["seqno"], type=w["type"], stroke=w.get("color"),
                                 rect_pt=[round(v, 2) for v in w["rect"]], curves_under=under))
    rep["document_scan_white_over_curves"] = scan
    save_json("d_white_shapes.json", rep)

    # ---- pictures ----------------------------------------------------------
    clip = pymupdf.Rect(CLIP[0], CLIP[1], CLIP[2] + 10, CLIP[3] - 5)
    page.get_pixmap(dpi=600, clip=clip).save(f"{HERE}/d_fig9_published_crop.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig_, ax = plt.subplots(figsize=(8, 5))
    for cv, ls, lab in ((solid, "-", f"solid blue path (seqno {solid['seqno']})"),
                        (dashed, "--", f"dashed blue path (seqno {dashed['seqno']})")):
        first = True
        for p0, p1 in line_items(cv):
            ax.plot([X(p0[0]), X(p1[0])], [Y(p0[1]), Y(p1[1])], ls, color="C0", lw=1,
                    label=lab if first else None)
            first = False
    for d, e in covers:
        if e["rpm"][0] > 2200:   # legend swatch overlay, outside the data region of interest
            continue
        ax.add_patch(Rectangle((e["rpm"][0], e["depth_mm"][0]), e["rpm"][1] - e["rpm"][0],
                               e["depth_mm"][1] - e["depth_mm"][0], fill=False, ec="red", lw=1))
        ax.text(e["rpm"][0], e["depth_mm"][1] + 0.2, str(d["seqno"]), color="red", fontsize=7)
    ax.set_xlim(400, 3050)
    ax.set_ylim(0, 30)
    ax.set_xlabel("spindle speed (rpm)")
    ax.set_ylabel("depth (mm)")
    ax.set_title("Fig. 9 vector paths with the outlines of the white overlay rectangles (red)")
    ax.legend(loc="upper left", fontsize=8)
    fig_.tight_layout()
    fig_.savefig(f"{HERE}/d_fig9_vector_reconstruction.png", dpi=150)

    # console summary
    for e in rep["white_fills"]:
        print(e["seqno"], e["type"], e["role"], "rpm %.1f-%.1f" % tuple(e["rpm"]),
              "mm %.2f-%.2f" % tuple(e["depth_mm"]), "after curves:", e["after_solid_curve"],
              "| hides ctrl:", None if not e.get("hides_controlled") else
              ("%.1f-%.1f rpm, %.2f-%.2f mm" % (*e["hides_controlled"]["rpm"], *e["hides_controlled"]["depth_mm"])),
              "| hides unc:", None if not e.get("hides_uncontrolled") else
              ("%.1f-%.1f rpm, %.2f-%.2f mm" % (*e["hides_uncontrolled"]["rpm"], *e["hides_uncontrolled"]["depth_mm"])))
    print("summary", summary)
    print("hidden controlled vertices", rep["controlled_vertices_hidden"], "of", rep["controlled_vertices_total"],
          rep.get("controlled_vertices_hidden_rpm"), rep.get("controlled_vertices_hidden_depth_mm"))
    print("neighbours", rep.get("visible_neighbours"))
    print("painted after overlays", rep["painted_after_overlays"])
    print("document scan", scan)


if __name__ == "__main__":
    main()
