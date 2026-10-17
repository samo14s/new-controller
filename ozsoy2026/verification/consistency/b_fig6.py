"""(b) Fig. 6a closed-loop FRFs and Fig. 6b pole markers against their labels.

For each labelled pair (speed, gain) = (500, 1487), (1300, 1162), (2500, 585),
(3000, 797):
  * best-fit DVF gain of the plotted Fig. 6a curve (least squares in dB,
    0-250 Hz, plant fixed; zeta = 1.34 % and, for comparison, 1.45 %);
  * dominant closed-loop pole of the model at the labelled and fitted gain;
  * pole positions plotted in Fig. 6b (vector markers read from the PDF) and
    the gain that the plotted real part implies;
  * Fig. 8 gain at the same speed.

Outputs: b_fig6.json, b_fig6.csv, b_fig6_overlay.png
"""
import numpy as np
import pymupdf
from scipy.optimize import least_squares, brentq

from common import PDF, curve, dedup, fig8, save_json, save_csv, HERE
import params as P
import structure as S
import extract_vector_figs as evf

LABELS = [  # Fig 6a key in digitised json, speed, labelled gain, Fig 6b colour
    ("500rpm_g1487", 500, 1487, (0.0706, 0.4392, 0.7480)),
    ("1300rpm_g1162_dashed", 1300, 1162, (0.2314, 0.6699, 0.2000)),
    ("2500rpm_g585_dashed", 2500, 585, (0.9297, 0.6895, 0.1294)),
    ("3000rpm_g797_dashed", 3000, 797, (0.5176, 0.0901, 0.8203)),
]
FIG6B_CLIP = (325, 60, 495, 255)


def fig6b_markers():
    """Centres of the Fig. 6b pole markers in (sigma, omega) rad/s, per colour."""
    doc = pymupdf.open(PDF)
    page = doc[6]
    drs = page.get_drawings()
    xt, yt = evf.frame_ticks(drs, FIG6B_CLIP)
    px, _ = evf.fit_axis(xt, [-150, -100, -50, 0])
    py, _ = evf.fit_axis(yt, [-1500, -1000, -500, 0, 500, 1000, 1500])
    # legend box: white fill followed by a dark stroked rectangle of the same size
    legend = None
    for d in drs:
        r = d["rect"]
        if d["type"] == "s" and d["items"][0][0] == "re" and FIG6B_CLIP[0] < r.x0 and r.x1 < FIG6B_CLIP[2] \
                and r.width < 60 and FIG6B_CLIP[1] < r.y0 < FIG6B_CLIP[3]:
            legend = r
    out = {}

    def add(name, kind, cx, cy):
        s, w = px[0] + px[1] * cx, py[0] + py[1] * cy
        out.setdefault(name, {}).setdefault(kind, set()).add((round(s, 2), round(w, 2)))

    for d in drs:
        r = d["rect"]
        if not (FIG6B_CLIP[0] < r.x0 and r.x1 < FIG6B_CLIP[2] and FIG6B_CLIP[1] < r.y0 < FIG6B_CLIP[3]):
            continue
        if legend is not None and legend.contains(r):
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if d["type"] == "f" and d.get("fill") == (0.0, 0.0, 0.0) and len(d["items"]) == 4 and r.width < 4:
            add("open-loop", "all", cx, cy)
        elif d["type"] == "s" and len(d["items"]) == 1 and d["items"][0][0] == "l" and r.width < 3 and r.height < 3:
            for key, rpm, g, col in LABELS:
                if max(abs(a - b) for a, b in zip(d["color"], col)) < 0.02:
                    add(f"{rpm}rpm", "all", cx, cy)
    res = {}
    for name, v in out.items():
        pts = sorted(v["all"])
        struct = [p for p in pts if abs(p[1]) > 400]
        act = [p for p in pts if abs(p[1]) <= 400]
        res[name] = dict(
            structural=(np.mean([p[0] for p in struct]), np.mean([abs(p[1]) for p in struct])),
            structural_points=struct,
            actuator=(np.mean([p[0] for p in act]), np.mean([abs(p[1]) for p in act])) if act else None,
            actuator_points=act)
    return res


def dominant(g):
    p = S.closed_loop_poles(g)
    s = p[np.argmax(np.abs(p.imag))]
    a = p[np.argmin(np.where(p.imag > 0, np.abs(p.imag), np.inf))]
    return (s.real, abs(s.imag)), (a.real, abs(a.imag))


def g_from_sigma(sigma):
    try:
        return brentq(lambda g: dominant(g)[0][0] - sigma, 0.0, 5000.0)
    except ValueError:
        return np.nan


def fit_fscale(f, ydb):
    """Uniform frequency-axis scale of the plotted uncontrolled curve (fn, k fixed)."""
    def res(p):
        P.set_damping(p[1])
        return 20 * np.log10(np.abs(S.G(2j * np.pi * f * p[0]))) - ydb
    r = least_squares(res, [1.0, 0.0134], xtol=1e-14, ftol=1e-14)
    return r.x[0], r.x[1], np.sqrt(np.mean(r.fun ** 2))


def fit_gain(f, ydb, fscale=1.0):
    w = 2 * np.pi * f * fscale
    s = 1j * w

    def res(g):
        return 20 * np.log10(np.abs(S.Gcl(s, g[0]))) - ydb
    r = least_squares(res, [800.0], bounds=([0.0], [5000.0]), xtol=1e-12, ftol=1e-12)
    # 1-sigma-like spread: gain change that raises rms by 0.01 dB
    return r.x[0], np.sqrt(np.mean(r.fun ** 2)), np.abs(r.fun).max()


def main():
    g8x, g8y = fig8()
    mk = fig6b_markers()
    rep = dict(fig6b_markers=mk, cases=[])
    rows = []
    curves = {}
    f0, y0 = dedup(*curve("fig6a_closed_loop_frf.json", "uncontrolled"))
    m0 = f0 > 0.5
    fsc, zsc, rms_sc = fit_fscale(f0[m0], y0[m0])
    rep["fig6a_uncontrolled_fscale_fit"] = dict(fscale=fsc, zeta=zsc, rms_dB=rms_sc)
    for zeta, fscale in ((0.0134, 1.0), (0.0145, 1.0), (zsc, fsc)):
        P.set_damping(zeta)
        for key, rpm, g_lab, col in LABELS:
            f, y = dedup(*curve("fig6a_closed_loop_frf.json", key))
            m = f > 0.5
            f, y = f[m], y[m]
            curves[key] = (f, y)
            g_fit, rms, mx = fit_gain(f, y, fscale)
            rms_lab = np.sqrt(np.mean((20 * np.log10(np.abs(S.Gcl(2j * np.pi * f * fscale, g_lab))) - y) ** 2))
            ipk = np.argmax(y)
            st_lab, ac_lab = dominant(g_lab)
            st_fit, _ = dominant(g_fit)
            plotted = mk[f"{rpm}rpm"]
            g_plot = g_from_sigma(plotted["structural"][0])
            g8 = float(np.interp(rpm, g8x, g8y))
            case = dict(zeta=zeta, fscale=fscale, rpm=rpm, label_gain=g_lab, fig8_gain_at_rpm=g8,
                        fig6a_fit_gain=g_fit, fig6a_rms_dB_fit=rms, fig6a_maxabs_dB_fit=mx,
                        fig6a_rms_dB_at_label_gain=rms_lab,
                        fig6a_peak=dict(f=f[ipk], dB=y[ipk]),
                        model_peak_dB_at_label=20 * np.log10(np.abs(S.Gcl(2j * np.pi * np.linspace(100, 160, 6001), g_lab))).max(),
                        model_pole_at_label=st_lab, model_actuator_pole_at_label=ac_lab,
                        model_pole_at_fit=st_fit,
                        fig6b_plotted_pole=plotted["structural"], fig6b_plotted_actuator_pole=plotted["actuator"],
                        gain_implied_by_fig6b_sigma=g_plot)
            rep["cases"].append(case)
            rows.append((zeta * 100, fscale, rpm, g_lab, g8, g_fit, rms, rms_lab, st_lab[0], st_fit[0],
                         plotted["structural"][0], plotted["structural"][1], g_plot,
                         ac_lab[0], ac_lab[1],
                         plotted["actuator"][0] if plotted["actuator"] else np.nan,
                         plotted["actuator"][1] if plotted["actuator"] else np.nan))
    ol = mk["open-loop"]["structural"]
    rep["open_loop_plotted"] = ol
    P.set_damping(0.0134)
    rep["open_loop_model_1p34"] = dominant(0.0)
    P.set_damping(0.0145)
    rep["open_loop_model_1p45"] = dominant(0.0)
    P.set_damping(0.0134)
    save_json("b_fig6.json", rep)
    save_csv("b_fig6.csv", ["zeta_pct", "fig6a_freq_scale", "rpm", "label_gain", "fig8_gain_at_rpm", "fig6a_fit_gain",
                            "fig6a_rms_dB_at_fit", "fig6a_rms_dB_at_label", "model_sigma_at_label",
                            "model_sigma_at_fit", "fig6b_sigma_plotted", "fig6b_omega_plotted",
                            "gain_implied_by_fig6b_sigma", "model_act_sigma_at_label",
                            "model_act_omega_at_label", "fig6b_act_sigma_plotted",
                            "fig6b_act_omega_plotted"], rows)

    # overlay picture (zeta = 1.34 %)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    f0, y0 = dedup(*curve("fig6a_closed_loop_frf.json", "uncontrolled"))
    ax[0].plot(f0, y0, "k", lw=1, label="Fig 6a uncontrolled (vector)")
    ff = np.linspace(0.5, 250, 5000)
    ax[0].plot(ff, 20 * np.log10(np.abs(S.G(2j * np.pi * ff))), "k:", lw=1, label="model, zeta 1.34 %")
    for (key, rpm, g_lab, col), c in zip(LABELS, rep["cases"][:4]):
        f, y = curves[key]
        ax[0].plot(f, y, color=col, lw=1.2, label=f"{rpm} rpm curve (label g={g_lab})")
        ax[0].plot(ff, 20 * np.log10(np.abs(S.Gcl(2j * np.pi * ff, g_lab))), "--", color=col, lw=0.8)
        ax[0].plot(ff, 20 * np.log10(np.abs(S.Gcl(2j * np.pi * ff, c["fig6a_fit_gain"]))), ":", color=col,
                   lw=1.5)
        ps = c["fig6b_plotted_pole"]
        ax[1].plot(ps[0], ps[1], "x", color=col, ms=8, label=f"Fig 6b {rpm} rpm (plotted)")
        ml = c["model_pole_at_label"]
        ax[1].plot(ml[0], ml[1], "o", mfc="none", color=col, ms=8, label=f"model at g={g_lab}")
    ax[1].plot(ol[0], ol[1], "kD", ms=5, label="Fig 6b open loop")
    ax[0].set_xlim(100, 160)
    ax[0].set_ylim(-140, -108)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("dB re 1 m/N")
    ax[0].set_title("solid: Fig 6a vector data; dashed: model at label g; dotted: model at fitted g", fontsize=8)
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("real part (rad/s)")
    ax[1].set_ylabel("imag part (rad/s)")
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].set_title("structural poles: x = plotted in Fig 6b, o = model at labelled gain", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{HERE}/b_fig6_overlay.png", dpi=150)

    for c in rep["cases"]:
        print("fscale %.5f zeta %.3f%% %4d rpm label %4d fig8 %7.1f | 6a fit g %7.1f rms %.3f dB (label rms %.3f) peak %.2f dB"
              " | sigma model(label) %7.2f fit %7.2f plotted %7.2f -> g %7.1f | act model %s plotted %s" % (
                  c["fscale"], c["zeta"] * 100, c["rpm"], c["label_gain"], c["fig8_gain_at_rpm"], c["fig6a_fit_gain"],
                  c["fig6a_rms_dB_fit"], c["fig6a_rms_dB_at_label_gain"], c["fig6a_peak"]["dB"],
                  c["model_pole_at_label"][0], c["model_pole_at_fit"][0], c["fig6b_plotted_pole"][0],
                  c["gain_implied_by_fig6b_sigma"],
                  np.round(c["model_actuator_pole_at_label"], 2), np.round(c["fig6b_plotted_actuator_pole"], 2)))
    print("fig6a fscale fit", rep["fig6a_uncontrolled_fscale_fit"])
    print("open loop plotted", ol, "model 1.34", rep["open_loop_model_1p34"][0], "1.45", rep["open_loop_model_1p45"][0])


if __name__ == "__main__":
    main()
