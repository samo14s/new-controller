"""Every published figure and table of the paper against the reproduction.

One figure per published result: the paper's data (extracted from the PDF's
vector paths; Fig. 11 by colour from its raster image; Table 3 and the text as
printed), the reproduction, and the difference.  The numbers quoted on the
figures are written to ../results/paper_vs_ours.json.

    python figures_paper_vs_ours.py    -> ../figures/p01_*.png ... p12_*.png
"""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from structure import G, Gcl, closed_loop_poles
from zoa import sld, a_lim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
TGT = os.path.join(ROOT, "verification", "targets")
PAPER, OURS, OURS2, GREY = "#d1495b", "#1f5fa8", "#7b5ea7", "#6c757d"
GOOD, BAD = "#2e7d32", "#c62828"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
SRC_NOTE = "paper data: extracted from the PDF's vector paths"
STATS = {}
F6A_AXIS = 129.3 / 129.02            # Fig. 6a is drawn with fn = 129.02 Hz on its frequency axis (README 5.1)
STRAIGHT = ((500, 550), (560, 610), (620, 740), (1410, 1860))   # Fig. 8 straight segments (README 4.4)
HIDDEN = ((958.5, 975.4), (1498.5, 2050.8))                      # Fig. 9 controlled curve under white fills
BANDS = ((500, 1150), (1160, 1490), (1500, 2050), (2060, 2310), (2320, 3000))


def dig(name, key):
    d = json.load(open(os.path.join(ROOT, "digitized", name)))
    p = np.array([q for s in d[key] for q in s])
    return p[np.argsort(p[:, 0])]


def dcsv(name):
    return np.loadtxt(os.path.join(ROOT, "digitized", name), delimiter=",", skiprows=1)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def in_any(x, spans):
    x = np.asarray(x, float)
    return np.any([(x >= a) & (x <= b) for a, b in spans], axis=0)


def shade(ax, spans, color, label=None, alpha=0.12):
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, label=label if k == 0 else None)


def r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- Fig. 5 ----
def p01_fig5():
    i = dig("fig5_frf.json", "identified")
    m = dig("fig5_frf.json", "measured_dashed")
    f = np.linspace(50, 210, 4000)
    curves = {}
    for z in (0.0145, 0.0134):
        P.set_damping(z)
        curves[z] = (np.abs(G(2j * np.pi * f)), np.abs(G(2j * np.pi * i[:, 0])))
    P.set_damping(P.ZETA_PAPER_CURVES)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(i[:, 0], i[:, 1] * 1e6, color=PAPER, lw=4, alpha=0.35, label="paper Fig. 5: identified")
    ax[0].plot(m[:, 0], m[:, 1] * 1e6, ":", color=PAPER, lw=1.1, label="paper Fig. 5: measured")
    ax[0].plot(f, curves[0.0145][0] * 1e6, color=OURS, lw=1.1, label="reproduction: Table 1 (129.3 Hz, 1.45 %, 1.34e7 N/m)")
    ax[0].plot(f, curves[0.0134][0] * 1e6, "--", color=GREY, lw=0.9, label="reproduction with 1.34 % (the paper's other curves)")
    e145 = 100 * (curves[0.0145][1] / i[:, 1] - 1)
    e134 = 100 * (curves[0.0134][1] / i[:, 1] - 1)
    ax[1].plot(i[:, 0], e145, color=OURS, lw=0.9, label=f"Table 1 model: median |diff| {np.median(np.abs(e145)):.2f} %")
    ax[1].plot(i[:, 0], e134, "--", color=GREY, lw=0.9, label=f"1.34 % model: peak +{e134.max():.1f} %")
    ax[0].set(ylabel="|G| (um/N)", title="P01  Paper Fig. 5 vs reproduction: structure FRF")
    ax[0].legend(fontsize=7.3)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Frequency (Hz)", ylabel="reproduction / paper - 1 (%)", xlim=(50, 210), ylim=(-3, 10))
    ax[1].legend(fontsize=7.3, loc="upper right")
    save(fig, "p01_fig5_structure_frf.png")
    STATS["p01"] = dict(median_abs_pct_table1=r(np.median(np.abs(e145))), max_abs_pct_table1=r(np.abs(e145).max()),
                        peak_pct_134=r(e134.max(), 2))


# --------------------------------------------------------------- Fig. 6a ----
F6_CASES = [("uncontrolled", "uncontrolled", 0.0, 0.0, "k"),
            ("500rpm_g1487", "500 rpm, g = 1487", 1487.0, 1542.9, "C0"),
            ("1300rpm_g1162_dashed", "1300 rpm, g = 1162", 1162.0, 1162.2, "C2"),
            ("2500rpm_g585_dashed", "2500 rpm, g = 585", 585.0, 685.4, "C1"),
            ("3000rpm_g797_dashed", "3000 rpm, g = 797", 797.0, 830.0, "C4")]


def p02_fig6a():
    d = json.load(open(os.path.join(ROOT, "digitized", "fig6a_closed_loop_frf.json")))
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[1, 1], sharey=a1)
    st = {}
    for key, lab, g_leg, g_fit, col in F6_CASES:
        p = np.array([q for s in d[key] for q in s])
        p = p[(p[:, 0] > 2) & (p[:, 0] < 250)]
        p = p[np.argsort(p[:, 0])]
        fp = p[:, 0] * F6A_AXIS                       # the paper's axis rescaled to fn = 129.3 Hz
        leg = 20 * np.log10(np.abs(Gcl(2j * np.pi * fp, g_leg)))
        fit = 20 * np.log10(np.abs(Gcl(2j * np.pi * fp, g_fit)))
        a0.plot(fp, p[:, 1], color=col, lw=4, alpha=0.3)
        ff = np.linspace(2, 250, 3000)
        a0.plot(ff, 20 * np.log10(np.abs(Gcl(2j * np.pi * ff, g_leg))), color=col, lw=1, label=f"{lab}")
        if abs(g_fit - g_leg) > 1:
            a0.plot(ff, 20 * np.log10(np.abs(Gcl(2j * np.pi * ff, g_fit))), "--", color=col, lw=0.9)
        rl, rf = p[:, 1] - leg, p[:, 1] - fit
        a1.plot(fp, rl, color=col, lw=0.9, label=f"{lab}: rms {np.sqrt(np.mean(rl ** 2)):.3f} dB")
        a2.plot(fp, rf, color=col, lw=0.9, label=f"g = {g_fit:.0f}: rms {np.sqrt(np.mean(rf ** 2)):.3f} dB")
        st[lab] = dict(g_legend=g_leg, g_fit=g_fit, rms_db_legend=r(np.sqrt(np.mean(rl ** 2))),
                       rms_db_fit=r(np.sqrt(np.mean(rf ** 2))), peak_db_paper=r(p[:, 1].max(), 2))
    a0.plot([], [], color="0.3", lw=4, alpha=0.3, label="thick: paper Fig. 6a")
    a0.plot([], [], color="0.3", lw=1, label="thin: reproduction, legend gain")
    a0.plot([], [], "--", color="0.3", lw=0.9, label="dashed: reproduction, gain that fits the drawn peak")
    a0.set(xlabel="Frequency (Hz)", ylabel="|Gcl| (dB re 1 m/N)", xlim=(0, 250), ylim=(-155, -108),
           title="P02  Paper Fig. 6a vs reproduction: closed-loop FRF (zeta = 1.34 %)")
    a0.legend(fontsize=7, ncol=2, loc="upper left")
    for a, t in ((a1, "difference with the gains of the legend"), (a2, "difference with 1543 / 1162 / 685 / 830")):
        a.axhline(0, color="k", lw=0.6)
        a.set(xlabel="Frequency (Hz)", xlim=(0, 250), ylim=(-1.6, 1.6), title=t)
        a.legend(fontsize=6.6, loc="lower left")
    a1.set_ylabel("paper - reproduction (dB)")
    fig.text(0.5, -0.01, "The paper's frequency axis is rescaled by 129.3/129.02 (its curves peak at 129.02 Hz, its own fn "
             f"marker is at 129.3 Hz). {SRC_NOTE}.", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "p02_fig6a_closed_loop_frf.png")
    STATS["p02"] = st


# --------------------------------------------------------------- Fig. 6b ----
def p03_fig6b():
    """The paper's poles: marker centres from the vector paths, unrounded
    (verification/consistency/b_fig6.json; fig6b_poles.csv rounds them to 0.1 pt, about 1.7 rad/s)."""
    mk = json.load(open(os.path.join(ROOT, "verification", "consistency", "b_fig6.json")))["fig6b_markers"]
    cases = [("open-loop", "open loop", 0.0, 0.0, "k"), ("500rpm", "500 rpm, g = 1487", 1487.0, 1542.9, "C0"),
             ("1300rpm", "1300 rpm, g = 1162", 1162.0, 1162.2, "C2"),
             ("2500rpm", "2500 rpm, g = 585", 585.0, 685.4, "C1"),
             ("3000rpm", "3000 rpm, g = 797", 797.0, 830.0, "C4")]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    st = {}
    offs = {"open loop": (0, 14), "500 rpm, g = 1487": (0, 14), "1300 rpm, g = 1162": (0, 14),
            "2500 rpm, g = 585": (0, -30), "3000 rpm, g = 797": (0, 14)}
    for key, lab, g_leg, g_fit, col in cases:
        s_leg, s_fit = closed_loop_poles(g_leg), closed_loop_poles(g_fit)
        s_leg, s_fit = s_leg[s_leg.imag > 0], s_fit[s_fit.imag > 0]
        st[lab] = {}
        for k, mode in enumerate(("structural", "actuator")):
            q = mk[key][mode]
            if q is None:
                continue
            ps = complex(q[0], q[1])
            ml = s_leg[np.argmin(np.abs(s_leg - ps))]
            mf = s_fit[np.argmin(np.abs(s_fit - ps))]
            ax[k].plot(ps.real, ps.imag, "o", ms=12, mfc="none", mec=col, mew=1.6)
            ax[k].plot(ml.real, ml.imag, "x", color=col, ms=8, mew=1.4)
            if abs(g_fit - g_leg) > 1:
                ax[k].plot(mf.real, mf.imag, "+", color=col, ms=11, mew=1.4)
            st[lab][mode] = dict(paper=[r(ps.real, 2), r(ps.imag, 2)], legend_gain=[r(ml.real, 2), r(ml.imag, 2)],
                                 fit_gain=[r(mf.real, 2), r(mf.imag, 2)], fit_minus_paper=r(abs(mf - ps), 3))
            if k == 0:
                txt = f"{lab}\npaper {ps.real:.2f}, legend gain {ml.real:.2f}"
                if abs(g_fit - g_leg) > 1:
                    txt += f"\ng = {g_fit:.0f}: {mf.real:.2f}"
                ax[k].annotate(txt, (ps.real, ps.imag), textcoords="offset points", xytext=offs[lab],
                               fontsize=6.5, color=col, ha="center", va="bottom" if offs[lab][1] > 0 else "top")
    for k in range(2):
        ax[k].plot([], [], "o", mfc="none", mec="0.3", ms=9, label="paper Fig. 6b")
        ax[k].plot([], [], "x", color="0.3", label="reproduction, legend gain")
        ax[k].plot([], [], "+", color="0.3", ms=9, label="reproduction, gain that fits Fig. 6a")
    ax[0].set(xlabel="Real part (rad/s)", ylabel="Imaginary part (rad/s)", xlim=(-150, 5), ylim=(785, 825),
              title="P03  Paper Fig. 6b vs reproduction: structural poles (zeta = 1.34 %)")
    ax[1].set(xlabel="Real part (rad/s)", xlim=(-7.85, -7.35), ylim=(52.2, 52.5), title="actuator poles (8.4 Hz), zoomed")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[1].legend(fontsize=7, loc="lower left")
    worst = max(v["fit_minus_paper"] for c in st.values() for v in c.values())
    fig.text(0.5, -0.02, f"{SRC_NOTE} (marker centres, unrounded); upper half plane only. With the gains that fit "
             f"Fig. 6a every pole matches within {worst:.2f} rad/s.", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "p03_fig6b_poles.png")
    STATS["p03"] = dict(cases=st, worst_fit_gain_distance_rad_s=r(worst, 3))


# ---------------------------------------------------------------- Fig. 8 ----
def band_medians(x, ratio, mask=None):
    out = {}
    for a, b in BANDS:
        m = (x >= a) & (x <= b) & np.isfinite(ratio)
        if mask is not None:
            m &= mask
        out[f"{a}-{b}"] = r(np.median(ratio[m]), 2) if m.sum() >= 3 else None
    return out


def p04_fig8():
    d = np.load(os.path.join(RES, "gainmap_main.npz"))
    gp = dcsv("fig8_gain_points.csv")
    rpm = d["rpm"]
    g_paper = np.interp(rpm, gp[:, 0], gp[:, 1])
    ratio = d["g_best"] / g_paper
    keep = ~in_any(rpm, STRAIGHT)
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
    shade(ax[0], STRAIGHT, "0.4", "paper: straight segments (collinear vertices, probably interpolated)")
    ax[0].plot(gp[:, 0], gp[:, 1], color=PAPER, lw=1.6, label="paper Fig. 8")
    ax[0].plot(rpm, d["g_best"], color=OURS, lw=1.4, label="reproduction (Eq. 2, force reading)")
    ax[0].set(ylabel="g_dvf (V s/m)", ylim=(0, 1600), title="P04  Paper Fig. 8 vs reproduction: gain map")
    ax[0].legend(fontsize=7.3, loc="upper right")
    ax[1].plot(rpm, np.where(keep, ratio, np.nan), color=OURS, lw=1.2)
    ax[1].plot(rpm, np.where(~keep, ratio, np.nan), color=OURS, lw=0.8, alpha=0.35)
    ax[1].axhline(1, color="k", lw=0.6)
    med = band_medians(rpm, ratio, keep)
    for a, b in BANDS:
        v = med[f"{a}-{b}"]
        if v is not None and (a, b) != (1500, 2050):
            m = keep & (rpm >= a) & (rpm <= b)
            span = "" if m.sum() >= 0.8 * np.sum((rpm >= a) & (rpm <= b)) else f"\n({rpm[m].min():.0f}-{rpm[m].max():.0f} only)"
            ax[1].text((a + b) / 2, 2.75, f"median {v:.2f}{span}", ha="center", va="top", fontsize=7)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="reproduction / paper", xlim=(500, 3000), ylim=(0, 2.9))
    ax[1].annotate("2060-2310: an isolated feasible pocket of Eq. (2);\non the interval from g = 0 the ratio is 0.72",
                   xy=(2250, 1.86), xytext=(2380, 1.75), textcoords="data", fontsize=6.6, ha="left", va="center",
                   color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    fig.text(0.5, -0.03, f"{SRC_NOTE}. Ratio drawn faint on the paper's straight segments (excluded from the medians).\n"
             "Reproduction: sharp entry from rest, demand read as force (3 N/V), force evaluated at every RK4 stage "
             "(README 4.6, 6); below 1500 rpm the force scheme alone moves the gain by up to 3x.", ha="center", fontsize=7)
    save(fig, "p04_fig8_gain_map.png")
    m15 = keep & (rpm >= 1500) & (rpm <= 2050)
    med["1500-2050"] = None
    STATS["p04"] = dict(band_median_ratio=med, band_1870_2050_range=[r(ratio[m15].min(), 2), r(ratio[m15].max(), 2)],
                        band_note="paper's straight segments excluded; 500-1150 is 750-1150 and 1160-1490 is 1160-1400",
                        paper_range=[r(gp[:, 1].min(), 1), r(gp[:, 1].max(), 1)],
                        ours_range=[r(d["g_best"].min(), 1), r(d["g_best"].max(), 1)],
                        ours_min_rpm=float(rpm[np.argmin(d["g_best"])]),
                        g_2500=[r(np.interp(2500, gp[:, 0], gp[:, 1]), 1), r(np.interp(2500, rpm, d["g_best"]), 1)],
                        g_2300=[r(np.interp(2300, gp[:, 0], gp[:, 1]), 1), r(np.interp(2300, rpm, d["g_best"]), 1)])


# -------------------------------------------------------- Fig. 9 (no control)
def p05_fig9_uncontrolled():
    u = dcsv("fig9_uncontrolled_points.csv")
    u = u[(u[:, 0] >= 400) & (u[:, 0] <= 3000)]
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(u[:, 0], u[:, 1], color=PAPER, lw=4, alpha=0.35, label="paper Fig. 9: uncontrolled")
    st = {}
    for z, col, ls, lab in [(0.0134, OURS, "-", "reproduction, zeta = 1.34 %"),
                            (0.0145, GREY, "--", "reproduction, zeta = 1.45 % (Table 1)")]:
        P.set_damping(z)
        a = sld(u[:, 0], 0.0) * 1e3
        ax[0].plot(u[:, 0], a, ls, color=col, lw=1, label=lab)
        rel = 100 * (a / u[:, 1] - 1)
        ax[1].plot(u[:, 0], rel, ls, color=col, lw=0.9, label=f"{lab}: median |diff| {np.median(np.abs(rel)):.2f} %")
        st[f"zeta_{z}"] = dict(median_abs_pct=r(np.median(np.abs(rel))), min_mm=r(a.min()))
    P.set_damping(P.ZETA_PAPER_CURVES)
    st["paper_min_mm"] = r(u[:, 1].min())
    ax[0].set(ylabel="a_lim (mm)", ylim=(0, 31), title="P05  Paper Fig. 9 vs reproduction: uncontrolled stability lobes")
    ax[0].legend(fontsize=7.5)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="reproduction / paper - 1 (%)", ylim=(-2, 17), xlim=(400, 3000))
    ax[1].legend(fontsize=7.3, loc="upper right", ncol=2)
    fig.text(0.5, -0.01, f"{SRC_NOTE} (400-3000 rpm; the paper's first points, 389-399 rpm, jump to 16 mm).",
             ha="center", fontsize=7)
    save(fig, "p05_fig9_uncontrolled_sld.png")
    STATS["p05"] = st


def markers(ax, fig_name):
    rows = list(csv.DictReader(open(os.path.join(TGT, "experimental_points.csv"))))
    style = {("none", "stable"): ("+", "k", "uncontrolled, stable"), ("none", "chatter"): ("D", "k", "uncontrolled, chatter"),
             ("ctrl", "stable"): ("o", GOOD, "controlled, stable"), ("ctrl", "chatter"): ("x", BAD, "controlled, chatter")}
    seen = set()
    for q in rows:
        if q["figure"] != fig_name:
            continue
        kind = "none" if q["control"] == "none" else "ctrl"
        mk, col, lab = style[(kind, q["state"])]
        ax.plot(float(q["rpm"]), float(q["depth_mm"]), mk, ms=7 if mk != "+" else 9, mfc="none", mec=col, color=col,
                mew=1.4, label=f"paper test: {lab}" if (kind, q["state"]) not in seen else None)
        seen.add((kind, q["state"]))


# ----------------------------------------------------------- Fig. 9 (control)
def p06_fig9_controlled():
    d = np.load(os.path.join(RES, "gainmap_main.npz"))
    cp = dcsv("fig9_controlled_points.csv")
    up = dcsv("fig9_uncontrolled_points.csv")
    gp = dcsv("fig8_gain_points.csv")
    rpm = d["rpm"]
    P.set_damping(float(d["zeta"]))
    a_paper = np.interp(rpm, cp[:, 0], cp[:, 1])
    a_pg = sld(rpm, np.interp(rpm, gp[:, 0], gp[:, 1])) * 1e3          # our model at the paper's Fig. 8 gains
    hid = in_any(rpm, HIDDEN)
    straight = in_any(rpm, STRAIGHT)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    shade(ax[0], HIDDEN, PAPER, "paper: covered by white fills in the PDF", alpha=0.1)
    ax[0].plot(up[:, 0], up[:, 1], ":", color=PAPER, lw=1.1, label="paper Fig. 9: uncontrolled")
    ax[0].plot(cp[:, 0], cp[:, 1], color=PAPER, lw=1.7, label="paper Fig. 9: controlled (incl. covered part)")
    ax[0].plot(rpm, d["a_best"], color=OURS, lw=1.6, label="reproduction: controlled, our gain map")
    ax[0].plot(rpm, a_pg, "--", color=OURS2, lw=1.1, label="reproduction: our model at the paper's Fig. 8 gains")
    markers(ax[0], "Fig9")
    ax[0].set(ylabel="Depth of cut (mm)", ylim=(0, 32), title="P06  Paper Fig. 9 vs reproduction: controlled stability limit")
    ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ra, rg = d["a_best"] / a_paper, a_pg / a_paper
    full = ~hid & ~straight
    ax[1].plot(rpm, np.where(full, ra, np.nan), color=OURS, lw=1.2, label="our gain map / paper")
    ax[1].plot(rpm, np.where(full, rg, np.nan), "--", color=OURS2, lw=1, label="our model at the paper's gains / paper")
    ax[1].plot(rpm, np.where(~full, ra, np.nan), color=OURS, lw=0.8, alpha=0.3)
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="ratio to the paper", ylim=(0, 4), xlim=(500, 3000))
    ax[1].legend(fontsize=7, loc="upper left")
    shade(ax[1], STRAIGHT, "0.4", None, alpha=0.08)
    ax[1].annotate("2060-2310: our optimum is an isolated feasible\npocket; from g = 0 the depth would be\n5.8-7.5 mm (1.4x the paper)",
                   xy=(2250, 2.9), xytext=(2400, 2.75), textcoords="data", fontsize=6.4, ha="left", va="center",
                   color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    # medians as in README 5.2: the paper's straight Fig. 8 segments excluded, the covered speeds kept
    med = band_medians(rpm, ra, ~straight)
    medg = band_medians(rpm, rg, ~hid & ~straight)
    med["1500-2050"] = medg["1500-2050"] = None       # the paper's curve there is the covered part (README 5.2)
    for a, b in BANDS:
        v = med[f"{a}-{b}"]
        if v is not None and (a, b) != (1500, 2050):
            m = ~straight & (rpm >= a) & (rpm <= b)
            span = "" if m.sum() >= 0.8 * np.sum((rpm >= a) & (rpm <= b)) else f" ({rpm[m].min():.0f}-{rpm[m].max():.0f})"
            ax[1].text((a + b) / 2, 0.12, f"median {v:.2f}{span}", ha="center", va="bottom", fontsize=7, color=OURS)
    hr = ra[hid]
    fig.text(0.5, -0.01, f"{SRC_NOTE}. Ratios drawn faint where the paper's curve is covered ({hr.min():.1f}-{hr.max():.1f}x "
             f"there, {hr.max():.1f}x at {rpm[hid][np.argmax(hr)]:.0f} rpm) and on Fig. 8's straight segments (grey, "
             "excluded from the medians).", ha="center", fontsize=7)
    save(fig, "p06_fig9_controlled_sld.png")
    # README 4.3: the paper's curve relative to our model at the paper's Fig. 8 gains, at its visible vertices
    vv = cp[~in_any(cp[:, 0], HIDDEN)]
    dev = np.abs(vv[:, 1] / (sld(vv[:, 0], np.interp(vv[:, 0], gp[:, 0], gp[:, 1])) * 1e3) - 1)
    edge = HIDDEN[1][1]                             # the visible line starts at the white fill's edge
    vis_line = np.concatenate([[[edge, np.interp(edge, cp[:, 0], cp[:, 1])]], vv[vv[:, 0] > edge]])
    vis_line = np.concatenate([vv[vv[:, 0] < HIDDEN[0][0]], vis_line])
    j = int(np.argmin(vis_line[:, 1]))
    i730 = int(np.argmin(np.abs(rpm - 730)))
    k = int(np.argmax(d["a_best"] / d["a_unc"]))
    STATS["p06"] = dict(band_median_ours_over_paper=med, band_median_model_at_paper_gains_over_paper=medg,
                        band_note="straight Fig. 8 segments excluded; the ours/paper medians keep the covered speeds (as README 5.2)",
                        covered_ratio_range=[r(hr.min(), 2), r(hr.max(), 2)],
                        paper_vs_model_at_paper_gains=dict(visible_vertices=int(len(vv)), median_dev_pct=r(100 * np.median(dev), 1),
                                                           within_2pct=int(np.sum(dev < 0.02)), within_5pct=int(np.sum(dev < 0.05))),
                        worst_ours=[r(d["a_best"].min(), 2), float(rpm[np.argmin(d["a_best"])])],
                        paper_visible_min=[r(vis_line[j, 1], 2), r(vis_line[j, 0], 1)],
                        paper_covered_min=[r(cp[in_any(cp[:, 0], HIDDEN), 1].min(), 2)],
                        ratio_730=[r(np.interp(730, cp[:, 0], cp[:, 1]) / np.interp(730, up[:, 0], up[:, 1]), 2),
                                   r(d["a_best"][i730] / d["a_unc"][i730], 2)],
                        max_ratio_ours=[r((d["a_best"] / d["a_unc"])[k], 2), float(rpm[k])],
                        mean_paper=r(a_paper.mean(), 2), mean_ours=r(d["a_best"].mean(), 2),
                        a_1300=[r(np.interp(1300, cp[:, 0], cp[:, 1]), 2), r(np.interp(1300, rpm, d["a_best"]), 2)])


# --------------------------------------------------------------- Fig. 11 ----
def p07_fig11():
    rows = list(csv.DictReader(open(os.path.join(TGT, "fig11_curves.csv"))))
    cc = np.array([(float(q["rpm"]), float(q["depth_mm_centre"]), float(q["depth_mm_run_top"]) - float(q["depth_mm_run_bottom"]))
                   for q in rows if q["curve"] == "controlled"])
    u = np.array([(float(q["rpm"]), float(q["depth_mm_centre"])) for q in rows if q["curve"] == "uncontrolled"])
    cc, u = cc[np.argsort(cc[:, 0])], u[np.argsort(u[:, 0])]
    c = cc[:, :2]
    sg = np.load(os.path.join(RES, "single_gain_main.npz"))
    g_ours = float(sg["g_ref"])
    fine = np.arange(390, 3101, 2.0)
    m = (c[:, 0] >= 400) & (c[:, 0] <= 3000)
    sel = (cc[:, 0] >= 500) & (cc[:, 0] <= 3000) & (cc[:, 2] < 0.6) & (cc[:, 1] > 0.5) & (cc[:, 1] < 30)
    #     the columns of verification/critic/fig11_gain_fit.py (README 4.5)
    gl = {237.0: "our model at g = 237", 270.0: "our model at g = 270 (best fit)", g_ours: f"our single gain g = {g_ours:.1f}"}
    mod = {g: sld(c[m, 0], g) * 1e3 for g in gl}
    fig, ax = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.15]})
    ax[0].plot(u[:, 0], u[:, 1], ":", color=PAPER, lw=1.2, label="paper Fig. 11: uncontrolled")
    ax[0].plot(c[:, 0], c[:, 1], color=PAPER, lw=3, alpha=0.45, label="paper Fig. 11: single gain 237")
    ax[0].plot(fine, sld(fine, 237.0) * 1e3, "--", color=OURS2, lw=1, label="reproduction: our model at g = 237")
    ax[0].plot(fine, sld(fine, 270.0) * 1e3, color=GREY, lw=0.9, label="reproduction: our model at g = 270 (best fit)")
    ax[0].plot(fine, sld(fine, g_ours) * 1e3, color=OURS, lw=1.6,
               label=f"reproduction: our single gain g = {g_ours:.1f}\n(force reading of Eq. 2; about 290 with the volt reading)")
    markers(ax[0], "Fig11")
    ax[0].set(ylabel="Depth of cut (mm)", ylim=(0, 34), title="P07  Paper Fig. 11 vs reproduction: single fixed gain (zeta = 1.34 %)")
    ax[0].legend(fontsize=6.6, loc="upper left", ncol=2)
    st = {}
    for g, col, ls in ((237.0, OURS2, "--"), (270.0, GREY, "-"), (g_ours, OURS, "-")):
        rel = 100 * (mod[g] / c[m, 1] - 1)
        relsel = 100 * (sld(cc[sel, 0], g) * 1e3 / cc[sel, 1] - 1)
        ax[1].plot(c[m, 0], rel, ls, color=col, lw=0.9 if g != g_ours else 1.2,
                   label=f"{gl[g]}: median {np.median(relsel):+.1f} %")
        st[f"g{g:.1f}_median_pct"] = r(np.median(relsel), 2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="reproduction / paper - 1 (%)", ylim=(-68, 18), xlim=(390, 3100))
    ax[1].legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.43))
    fig.text(0.5, -0.01, "paper data: Fig. 11 is a raster image, its curves extracted by colour (about +-0.07 mm, +-4 rpm; "
             "more on the steep lobe flanks, where the spikes come from). Medians over 500-3000 rpm, flank columns excluded.",
             ha="center", fontsize=7)
    save(fig, "p07_fig11_single_gain.png")
    st.update(g_paper=237.0, g_ours=r(g_ours, 1), mean_ours=r(float(np.mean(sg["alim_ref"])), 2),
              min_ours=r(sld(fine, g_ours).min() * 1e3, 2), paper_min=r(c[m, 1].min(), 2),
              median_selection="500-3000 rpm, 0.5-30 mm, columns whose run is under 0.6 mm (as verification/critic/fig11_gain_fit.py)",
              median_columns=int(sel.sum()))
    STATS["p07"] = st


# --------------------------------------------------------------- Table 3 ----
def p08_table3():
    t = np.genfromtxt(os.path.join(RES, "table3_main.csv"), delimiter=",", names=True)
    x = np.arange(len(t))
    w = 0.16
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax[0].bar(x - 2 * w, t["paper_clip"], w, color=PAPER, alpha=0.45, label="paper: clipped DVF")
    ax[0].bar(x - w, t["paper_prop"], w, color=PAPER, label="paper: proposed (saturation-aware)")
    ax[0].bar(x, t["a_clip1500_td"], w, color="#7aa6d6", label="reproduction: clipped DVF, g = 1500")
    ax[0].bar(x + w, t["a_clip3000_td"], w, color="#b5d3ef", label="reproduction: clipped DVF, g = 3000")
    ax[0].bar(x + 2 * w, t["a_prop_td"], w, color=OURS, label="reproduction: proposed")
    ax[0].set(ylabel="Limiting depth (mm)", title="P08  Paper Table 3 vs reproduction: clipped DVF against the proposed gain")
    ax[0].legend(fontsize=6.8, ncol=2, loc="upper right")
    dp = t["paper_prop"] - t["paper_clip"]
    do = t["a_prop_td"] - t["a_clip1500_td"]
    ax[1].bar(x - 0.18, dp, 0.36, color=PAPER, label="paper: proposed - clipped")
    ax[1].bar(x + 0.18, do, 0.36, color=OURS, label="reproduction: proposed - clipped (g = 1500)")
    ax[1].axhline(0, color="k", lw=0.7)
    for k in range(len(t)):
        ax[1].text(x[k] - 0.18, dp[k] + (0.25 if dp[k] >= 0 else -0.25), f"{dp[k]:+.1f}", ha="center",
                   va="bottom" if dp[k] >= 0 else "top", fontsize=6.5, color=PAPER)
        ax[1].text(x[k] + 0.18, do[k] - 0.25, f"{do[k]:+.1f}", ha="center", va="top", fontsize=6.5, color=OURS)
    ax[1].set_xticks(x, [f"{int(q)}" for q in t["rpm"]])
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="difference (mm)", ylim=(-8.5, 2.5))
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.text(0.5, -0.11, "Reproduction: time-domain limits with the 9 V command limit and the full surface memory, sharp entry "
             "from rest (README 4.6); paper: Table 3 as printed.\nThe proposed gain is Eq. (2) evaluated at a_lim, which "
             "rejects any gain that saturates the actuator at entry; at 2300 rpm it is an isolated feasible pocket (g = 849).",
             ha="center", fontsize=7)
    save(fig, "p08_table3.png")
    STATS["p08"] = dict(rpm=[int(q) for q in t["rpm"]], paper_prop_minus_clip=[r(v, 2) for v in dp],
                        ours_prop_minus_clip1500=[r(v, 2) for v in do])


# ------------------------------------------------------------ test points ---
def p09_test_points():
    rows = json.load(open(os.path.join(RES, "cuts_main.json")))
    seen, cuts = set(), []
    for q in rows:                                 # the 2800 rpm uncontrolled cuts are drawn in Figs. 9 and 11
        key = (q["rpm"], q["depth_mm"], q["controller"])
        if key not in seen:
            seen.add(key)
            cuts.append(q)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    mk = {"none": "s", "map": "o", "single": "^"}
    st = {}
    for k, (key, lab) in enumerate((("paper", "the paper's gains (Fig. 8, 237)"), ("ours", "our gains (map, 60.3)"))):
        agree = 0
        for q in cuts:
            x = q["depth_mm"] / q[f"alim_{key}"]
            pred = q[f"pred_{key}"] if key == "paper" else q["pred_ours"]
            ok = pred == q["measured"]
            agree += ok
            col = GOOD if ok else BAD
            ax[k].plot(x, q["rpm"], mk[q["controller"]], ms=8, mfc=col if q["measured"] == "chatter" else "none",
                       mec=col, mew=1.5)
        ax[k].axvline(1, color="k", lw=0.8)
        ax[k].set_xscale("log")
        ax[k].set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3], ["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
        ax[k].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[k].set(xlabel=f"measured depth / model a_lim at {lab}", xlim=(0.25, 4),
                  title=f"{'P09  Published test points: ' if k == 0 else ''}model with {lab}: {agree}/{len(cuts)} agree")
        ax[k].text(0.02, 0.02, "left: predicted stable", transform=ax[k].transAxes, fontsize=7)
        ax[k].text(0.98, 0.02, "right: predicted chatter", transform=ax[k].transAxes, fontsize=7, ha="right")
        st[key] = agree
    for name, m_ in (("uncontrolled", "s"), ("mapped gain", "o"), ("single gain", "^")):
        ax[1].plot([], [], m_, mfc="none", mec="0.3", ms=7, label=name)
    ax[1].plot([], [], "o", mfc="0.3", mec="0.3", label="filled: measured chatter")
    ax[1].plot([], [], "o", mfc="none", mec="0.3", label="open: measured stable")
    ax[1].plot([], [], "o", mfc="none", mec=GOOD, mew=1.5, label="green: model agrees")
    ax[1].plot([], [], "o", mfc="none", mec=BAD, mew=1.5, label="red: model disagrees")
    ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax[0].set_ylabel("Spindle speed (rpm)")
    fig.text(0.5, -0.02, "19 distinct cuts (21 markers in Figs. 9 and 11); paper data: marker positions from the PDF.",
             ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "p09_test_points.png")
    STATS["p09"] = dict(n=len(cuts), agree_paper_gains=st["paper"], agree_our_gains=st["ours"],
                        agree_our_gains_td=sum(q["pred_ours_td"] == q["measured"] for q in cuts))


# ---------------------------------------------------- chatter frequencies ---
def p10_chatter_frequencies():
    rows = {q["cut"]: q for q in json.load(open(os.path.join(RES, "cuts_main.json"))) if q["cut"]}
    meas = {"10B": (149.5, "Fig. 10 B: 1300 rpm, 10 mm, mapped gain"), "12B": (133.3, "Fig. 12 B: 2800 rpm, 1.5 mm, uncontrolled"),
            "12D": (138.1, "Fig. 12 D: 2700 rpm, 4 mm, single gain")}
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    st = {}
    for k, (cut, (fm, lab)) in enumerate(meas.items()):
        q = rows[cut]
        y = len(meas) - 1 - k
        ax.errorbar(fm, y, xerr=1.9, fmt="o", color=PAPER, ms=8, capsize=4, label="paper: measured chatter line (+-1.9 Hz)" if k == 0 else None)
        ax.plot(q["fc_paper"], y + 0.12, "s", mfc="none", mec=OURS2, ms=8, mew=1.4,
                label="reproduction: onset frequency at the paper's gain" if k == 0 else None)
        ax.plot(q["fc_ours"], y - 0.12, "D", mfc="none", mec=OURS, ms=7, mew=1.4,
                label="reproduction: onset frequency at our gain" if k == 0 else None)
        sim = q["f_nonsync"] if q["pred_ours_td"] == "chatter" else None
        if sim is not None:
            ax.plot(sim, y - 0.12, "o", color=OURS, ms=6, label="reproduction: simulated chatter line (our gain)" if k == 0 else None)
        else:
            ax.annotate(f"predicted stable at both gains (a_lim {q['alim_paper']:.2f} / {q['alim_ours']:.2f} mm > "
                        f"{q['depth_mm']:.0f} mm);\nmarkers: frequency at the boundary", (q["fc_ours"], y - 0.12),
                        textcoords="offset points", xytext=(-6, -10), fontsize=6.6, color=OURS, ha="right", va="top")
        st[cut] = dict(measured=fm, onset_paper_gain=r(q["fc_paper"], 2), onset_our_gain=r(q["fc_ours"], 2),
                       simulated=r(sim, 2) if sim else None)
    ax.set_yticks(range(len(meas)), [v[1] for v in list(meas.values())[::-1]], fontsize=7.5)
    ax.set(xlabel="Frequency (Hz)", xlim=(125, 156), ylim=(-0.6, len(meas) - 0.4),
           title="P10  Paper Figs. 10 and 12 vs reproduction: chatter frequency")
    h = [ax.errorbar([], [], xerr=[], fmt="o", color=PAPER, ms=8, capsize=4)[0],
         ax.plot([], [], "s", mfc="none", mec=OURS2, ms=8, mew=1.4)[0],
         ax.plot([], [], "D", mfc="none", mec=OURS, ms=7, mew=1.4)[0],
         ax.plot([], [], "o", color=OURS, ms=6)[0]]
    ax.legend(h, ["paper: measured chatter line (+-1.9 Hz)", "reproduction: onset frequency at the paper's gain",
                  "reproduction: onset frequency at our gain", "reproduction: simulated chatter line (our gain)"],
              fontsize=6.8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.5, -0.06, "paper data: spectral lines read from the FFT panels of Figs. 10 and 12 (raster). The 10B line is 2 Hz "
             "below spindle harmonic 7 (151.7 Hz at 1300 rpm).", ha="center", fontsize=7)
    save(fig, "p10_chatter_frequencies.png")
    STATS["p10"] = st


# -------------------------------------------------------- headline numbers --
def p11_headline():
    s6, s4, s7 = STATS["p06"], STATS["p04"], STATS["p07"]
    from objective import fitness
    _, al, fd = fitness(1300.0, 1240.0, detail=True)
    items = [
        ("largest gain on the map (V s/m)", 1487.0, s4["ours_range"][1], "text and Fig. 8"),
        ("smallest gain on the map (V s/m)", 256.9, s4["ours_range"][0], "Fig. 8; the text prints 237"),
        ("single fixed gain (V s/m)", 237.0, s7["g_ours"], "text; ours with the force reading (about 290 with the volt reading)"),
        ("improvement over no control near 730 rpm (x)", 13.0, s6["ratio_730"][1], "text; Fig. 9 gives 12.15"),
        ("worst-case controlled depth (mm)", 3.6, s6["worst_ours"][0], "text; Fig. 9: 3.80 visible, 3.22 covered"),
        ("uncontrolled minimum depth (mm)", 1.2, STATS["p05"]["zeta_0.0134"]["min_mm"], "text; Fig. 9 gives 1.212"),
        ("controlled limit at 1300 rpm (mm)", 11.49, s6["a_1300"][1], "Fig. 9; the text says 10, Table 3 12.0"),
        ("worked example: a_lim, 1300 rpm, g = 1240 (mm)", 11.8, al * 1e3, "text"),
        ("worked example: peak demand (N)", 29.0, fd, "text; ours with the force reading"),
        ("test points on the predicted side (of 19)", 17.0, float(STATS["p09"]["agree_paper_gains"]),
         "counted on Figs. 9 and 11; ours: model at the paper's gains"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for k, (lab, pv, ov, src) in enumerate(items):
        y = len(items) - 1 - k
        ax.plot([pv, ov], [y, y], color="0.75", lw=2, zorder=1)
        ax.plot(pv, y, "o", color=PAPER, ms=8, zorder=2)
        ax.plot(ov, y, "D", color=OURS, ms=7, zorder=3)
        ax.text(max(pv, ov) * 1.12, y + 0.13, f"paper {pv:g}  /  ours {ov:.4g}   ({ov / pv:.2f}x)", va="center", fontsize=7.5)
        ax.text(max(pv, ov) * 1.12, y - 0.22, src, va="center", fontsize=6.3, color=GREY)
    ax.set_xscale("log")
    ax.set_yticks(range(len(items)), [i[0] for i in items[::-1]], fontsize=8)
    ax.set(xlim=(0.8, 3e5), xlabel="value (log scale)", title="P11  Headline numbers: paper vs reproduction")
    ax.plot([], [], "o", color=PAPER, label="paper (printed, or read from its figures: see grey notes)")
    ax.plot([], [], "D", color=OURS, label="reproduction")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "p11_headline_numbers.png")
    STATS["p11"] = [dict(quantity=lab, paper=pv, ours=r(ov, 3), source=src) for lab, pv, ov, src in items]


# ------------------------------------------- section 8: beyond the paper ---
def p12_reformulated():
    f = np.load(os.path.join(RES, "feas_g1500.npz"))
    c = np.load(os.path.join(RES, "feas_cert_g1500.npz"))
    d = np.load(os.path.join(RES, "gainmap_main.npz"))
    gp = dcsv("fig8_gain_points.csv")
    cp = dcsv("fig9_controlled_points.csv")
    up = dcsv("fig9_uncontrolled_points.csv")
    rpm = f["rpm"]
    hid = in_any(rpm, HIDDEN)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
    ax[0, 0].plot(gp[:, 0], gp[:, 1], color=PAPER, lw=1.6, label="paper Fig. 8")
    ax[0, 0].plot(d["rpm"], d["g_best"], "--", color=GREY, lw=1, label="Eq. (2) as written")
    ax[0, 0].plot(rpm, f["g_best"], color=OURS, lw=1.6, label="reformulated (search capped at 1500)")
    ax[0, 0].set(ylabel="g_dvf (V s/m)", ylim=(0, 1650), title="P12  Beyond the reproduction: gain map")
    h, lab = ax[0, 0].get_legend_handles_labels()
    ax[1, 0].legend(h, lab, fontsize=6.6, loc="upper left", title="upper panel", title_fontsize=6.6)
    for a in (ax[0, 1], ax[1, 1]):
        shade(a, HIDDEN, PAPER, "paper: covered by white fills in the PDF" if a is ax[0, 1] else None, alpha=0.1)
    ax[0, 1].plot(up[:, 0], up[:, 1], ":", color=PAPER, lw=1.1, label="paper Fig. 9: uncontrolled")
    ax[0, 1].plot(cp[:, 0], cp[:, 1], color=PAPER, lw=1.6, label="paper Fig. 9: controlled (incl. covered part)")
    ax[0, 1].plot(d["rpm"], d["a_best"], "--", color=GREY, lw=1, label="reproduction of Eq. (2) as written")
    ax[0, 1].plot(rpm, c["a_cert"], color=OURS, lw=1.6, label="reformulated: checked depth (0.9 a_lim margin, TD, 9 V)")
    ax[0, 1].set(ylabel="Depth of cut (mm)", ylim=(0, 32), title="controlled depth")
    ax[0, 1].legend(fontsize=6.8, loc="upper left")
    gpi = np.interp(rpm, gp[:, 0], gp[:, 1])
    api = np.interp(rpm, cp[:, 0], cp[:, 1])
    rr = c["a_cert"] / api
    ax[1, 0].plot(rpm, f["g_best"] / gpi, color=OURS, lw=1.1)
    ax[1, 0].axhline(1, color="k", lw=0.6)
    ax[1, 0].set(xlabel="Spindle speed (rpm)", ylabel="reformulated / paper", ylim=(0, 4.4), xlim=(500, 3000))
    ax[1, 1].plot(rpm, np.where(~hid, rr, np.nan), color=OURS, lw=1.1)
    ax[1, 1].plot(rpm, np.where(hid, rr, np.nan), color=OURS, lw=0.8, alpha=0.3)
    ax[1, 1].axhline(1, color="k", lw=0.6)
    ax[1, 1].set(xlabel="Spindle speed (rpm)", ylabel="checked depth / paper", ylim=(0, 9), xlim=(500, 3000))
    fig.text(0.5, -0.03, "The checked depth keeps a 10 % margin below a_lim, which the paper's curve and Eq. (2) do not. "
             "Ratio drawn faint where the paper's curve is covered.\nThe reformulated gain sits on its 1500 V s/m search cap at 157 "
             "of 251 speeds. Reproduction: sharp entry from rest, demand read as force, full surface memory (README 4.6, 8).",
             ha="center", fontsize=7)
    save(fig, "p12_reformulated_vs_paper.png")
    STATS["p12"] = dict(mean_checked_mm=r(c["a_cert"].mean(), 2), mean_paper_mm=r(api.mean(), 2),
                        mean_paper_visible_mm=r(api[~hid].mean(), 2), median_ratio=r(np.median(rr), 2),
                        median_ratio_visible=r(np.median(rr[~hid]), 2), max_ratio_visible=r(rr[~hid].max(), 2),
                        min_checked=[r(c["a_cert"].min(), 2), float(rpm[np.argmin(c["a_cert"])])])


def main():
    os.makedirs(FIG, exist_ok=True)
    P.set_damping(P.ZETA_PAPER_CURVES)
    for fn in (p01_fig5, p02_fig6a, p03_fig6b, p04_fig8, p05_fig9_uncontrolled, p06_fig9_controlled, p07_fig11,
               p08_table3, p09_test_points, p10_chatter_frequencies, p11_headline, p12_reformulated):
        P.set_damping(P.ZETA_PAPER_CURVES)
        fn()
        print("ok", fn.__name__, flush=True)
    json.dump(STATS, open(os.path.join(RES, "paper_vs_ours.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
