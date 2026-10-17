"""Figures of the reproduction, each overlaid on the paper's own plotted data.

Paper curves come from ../digitized (recovered from the PDF's vector paths by
extract_vector_figs.py); everything else is recomputed here.
    python figures.py --tag main
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from structure import G, Ga, Gcl, closed_loop_poles
from zoa import sld

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DIG = os.path.join(HERE, "..", "digitized")
FIG = os.path.join(HERE, "..", "figures")

OURS = "#1f5fa8"
PAPER = "#d1495b"
UNC = "#6c757d"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _pts(name, key):
    with open(os.path.join(DIG, name)) as fh:
        d = json.load(fh)
    p = np.array([q for s in d[key] for q in s])
    return p[np.argsort(p[:, 0])]


def fig_actuator_and_frf():
    f = np.linspace(0.5, 220, 4000)
    s = 2j * np.pi * f
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(f, np.abs(Ga(s)), color=OURS)
    ax[0].set(xlabel="Frequency (Hz)", ylabel="|Ga| (N/V)", xlim=(0, 200),
              title="Actuator force per volt, Eq. (3)")
    for z, ls, lab in [(P.ZETA_TABLE1, "-", "model, zeta = 1.45 % (Table 1)"),
                       (P.ZETA_PAPER_CURVES, "--", "model, zeta = 1.34 %")]:
        P.set_damping(z)
        ax[1].plot(f, np.abs(G(s)), ls, color=OURS, lw=1, label=lab)
    P.set_damping(P.ZETA_PAPER_CURVES)
    m = _pts("fig5_frf.json", "measured_dashed")
    i = _pts("fig5_frf.json", "identified")
    ax[1].plot(m[:, 0], m[:, 1], ":", color=PAPER, lw=1.2, label="paper Fig. 5, measured")
    ax[1].plot(i[:, 0], i[:, 1], color=PAPER, lw=0.6, alpha=0.6, label="paper Fig. 5, identified")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="|G| (m/N)", xlim=(50, 210),
              title="Preloaded structure")
    ax[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "r01_actuator_structure_frf.png"))
    plt.close(fig)


def fig6(gmap=None):
    f = np.linspace(0.5, 250, 5000)
    s = 2j * np.pi * f
    cases = [("Uncontrolled", 0, "uncontrolled", "k"), ("500 rpm, g = 1487", 1487, "500rpm_g1487", "C0"),
             ("1300 rpm, g = 1162", 1162, "1300rpm_g1162_dashed", "C2"),
             ("2500 rpm, g = 585", 585, "2500rpm_g585_dashed", "C1"),
             ("3000 rpm, g = 797", 797, "3000rpm_g797_dashed", "C4")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for lab, g, key, col in cases:
        ax[0].plot(f, 20 * np.log10(np.abs(Gcl(s, g))), color=col, lw=1.2, label=f"ours: {lab}")
        p = _pts("fig6a_closed_loop_frf.json", key)
        ax[0].plot(p[:, 0], p[:, 1], ":", color=col, lw=1.6, alpha=0.8)
        poles = closed_loop_poles(g)
        ax[1].plot(poles.real, poles.imag, "x" if g else "o", color=col, ms=6, label=lab)
    ax[0].plot([], [], "k:", label="dotted: paper Fig. 6a")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="|Gcl| (dB re 1 m/N)", ylim=(-155, -108),
              xlim=(0, 260), title="Closed-loop FRF (paper gains)")
    ax[0].legend(fontsize=6.5)
    ax[1].set(xlabel="Real part (rad/s)", ylabel="Imag part (rad/s)", xlim=(-150, 5),
              ylim=(-1000, 1000), title="Closed-loop poles (Eq. 4)")
    ax[1].legend(fontsize=7, loc="center left")
    fig.savefig(os.path.join(FIG, "r02_fig6_frf_poles.png"))
    plt.close(fig)


def fig8_9(tag):
    d = np.load(os.path.join(RES, f"gainmap_{tag}.npz"))
    rpm = d["rpm"]
    gp = np.loadtxt(os.path.join(DIG, "fig8_gain_points.csv"), delimiter=",", skiprows=1)
    cp = np.loadtxt(os.path.join(DIG, "fig9_controlled_points.csv"), delimiter=",", skiprows=1)
    up = np.loadtxt(os.path.join(DIG, "fig9_uncontrolled_points.csv"), delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(gp[:, 0], gp[:, 1], color=PAPER, lw=1.4, label="paper Fig. 8")
    ax.plot(rpm, d["g_ref"], color=OURS, lw=1.4, label="reproduction (reference optimum)")
    if "g_sade" in d:
        ax.plot(rpm, d["g_sade"], ".", color="k", ms=2.5, label="reproduction (SADE, best of 3 seeds)")
    ax.set(xlabel="Spindle speed (rpm)", ylabel="g_dvf (V s/m)", xlim=(500, 3000),
           title="Spindle-speed dependent optimised gain")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, f"r03_fig8_gain_map_{tag}.png"))
    plt.close(fig)

    fine = np.arange(380, 3101, 2.0)
    a0 = sld(fine, 0.0) * 1e3
    a_map = d["a_sade"] if "a_sade" in d else d["a_ref"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(fine, a0, "--", color=UNC, lw=1, label="reproduction: uncontrolled")
    ax.plot(up[:, 0], up[:, 1], ":", color=PAPER, lw=1.2, label="paper Fig. 9: uncontrolled")
    ax.plot(rpm, a_map, color=OURS, lw=1.6, label="reproduction: controlled (gain map)")
    ax.plot(cp[:, 0], cp[:, 1], color=PAPER, lw=1.4, label="paper Fig. 9: controlled (incl. covered part)")
    hid = (cp[:, 0] > 1495) & (cp[:, 0] < 2051)
    ax.plot(cp[hid, 0], cp[hid, 1], color=PAPER, lw=4, alpha=0.25,
            label="covered by white fills in the PDF")
    _markers(ax)
    ax.set(xlabel="Spindle speed (rpm)", ylabel="Depth of cut (mm)", xlim=(380, 3050),
           ylim=(0, 32), title="Stability lobes, saturation-aware gain map")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.savefig(os.path.join(FIG, f"r04_fig9_sld_{tag}.png"))
    plt.close(fig)


def _markers(ax, fig_key="fig9"):
    path = os.path.join(HERE, "..", "verification", "targets", f"{fig_key}_markers.csv")
    if not os.path.exists(path):
        return
    try:
        m = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        return
    style = {"uncontrolled_stable": ("+", "k"), "uncontrolled_chatter": ("D", "k"),
             "controlled_stable": ("o", "g"), "controlled_chatter": ("x", "r")}
    names = m.dtype.names
    kcol = [n for n in names if "type" in n or "kind" in n or "class" in n]
    if not kcol:
        return
    for k, (mk, col) in style.items():
        sel = m[m[kcol[0]] == k]
        if len(sel):
            ax.plot(sel[names[[i for i, n in enumerate(names) if "rpm" in n][0]]],
                    sel[names[[i for i, n in enumerate(names) if "mm" in n or "depth" in n][0]]],
                    mk, mfc="none", color=col, ms=6, label=f"paper test: {k.replace('_', ' ')}")


def fig11(tag):
    sg = np.load(os.path.join(RES, f"single_gain_{tag}.npz"))
    fine = np.arange(380, 3101, 2.0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(fine, sld(fine, 0.0) * 1e3, "--", color=UNC, lw=1, label="uncontrolled")
    ax.plot(fine, sld(fine, 237.0) * 1e3, color=PAPER, lw=1.2,
            label="paper's single gain g = 237 V s/m (our model)")
    g = float(sg["g_ref"])
    ax.plot(fine, sld(fine, g) * 1e3, color=OURS, lw=1.6,
            label=f"reproduced single gain g = {g:.0f} V s/m")
    _markers(ax, "fig11")
    ax.set(xlabel="Spindle speed (rpm)", ylabel="Depth of cut (mm)", xlim=(380, 3100),
           ylim=(0, 33), title="Single fixed gain (alternative strategy)")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(os.path.join(FIG, f"r05_fig11_single_gain_{tag}.png"))
    plt.close(fig)


def table3(tag):
    path = os.path.join(RES, f"table3_{tag}.csv")
    if not os.path.exists(path):
        return
    t = np.genfromtxt(path, delimiter=",", names=True)
    paper_clip = [15.2, 8.7, 12.2, 11.2, 12.1, 4.5, 4.7, 5.8, 6.2]
    paper_prop = [14.9, 8.0, 11.8, 10.8, 12.0, 4.9, 5.3, 6.6, 7.1]
    x = np.arange(len(t))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - 1.5 * w, paper_clip, w, color=PAPER, alpha=0.45, label="paper: clipped DVF")
    ax.bar(x - 0.5 * w, paper_prop, w, color=PAPER, label="paper: proposed")
    ax.bar(x + 0.5 * w, t["a_clip_td"], w, color=OURS, alpha=0.45, label="reproduction: clipped DVF (TD)")
    ax.bar(x + 1.5 * w, t["a_prop_td"], w, color=OURS, label="reproduction: proposed (TD)")
    ax.set_xticks(x, [f"{int(r)}" for r in t["rpm"]])
    ax.set(xlabel="Spindle speed (rpm)", ylabel="Limiting depth (mm)",
           title="Table 3: clipped DVF vs saturation-aware gain")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, f"r06_table3_{tag}.png"))
    plt.close(fig)


def cuts(tag):
    path = os.path.join(RES, f"cuts_{tag}.npz")
    if not os.path.exists(path):
        return
    c = np.load(path)
    with open(os.path.join(RES, f"cuts_{tag}.json")) as fh:
        summ = {r["label"]: r for r in json.load(fh)}
    labs = list(summ)
    fig, ax = plt.subplots(len(labs), 2, figsize=(9, 1.7 * len(labs)),
                           gridspec_kw={"width_ratios": [2.2, 1]})
    for k, lab in enumerate(labs):
        r = summ[lab]
        ax[k, 0].plot(c[f"{lab}_t"], c[f"{lab}_acc"], color="#4a90c2", lw=0.4)
        ax[k, 0].plot(c[f"{lab}_opr_t"], c[f"{lab}_opr"], "k.", ms=2)
        ax[k, 0].set_ylabel("acc (m/s2)", fontsize=7)
        ax[k, 0].set_title(f"{lab}: {r['rpm']} rpm, {r['depth_mm']} mm, {r['controller']} "
                           f"(g = {r['gain']:.0f}) -> simulated {r['predicted']}, paper {r['measured']}",
                           fontsize=7.5)
        ax[k, 1].plot(c[f"{lab}_f"], c[f"{lab}_S"], color="#4a90c2", lw=0.8)
        ftp = float(c[f"{lab}_ftp"])
        for h in range(1, 5):
            ax[k, 1].axvline(h * ftp, color="k", lw=0.5, ls=":")
        ax[k, 1].set_xlim(0, 420)
    ax[-1, 0].set_xlabel("Time (s)")
    ax[-1, 1].set_xlabel("Frequency (Hz); dotted: tooth-passing harmonics")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"r07_cuts_{tag}.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    P.set_damping(P.ZETA_PAPER_CURVES)
    fig_actuator_and_frf()
    fig6()
    if os.path.exists(os.path.join(RES, f"gainmap_{a.tag}.npz")):
        d = np.load(os.path.join(RES, f"gainmap_{a.tag}.npz"))
        P.set_damping(float(d["zeta"]))
        fig8_9(a.tag)
        if os.path.exists(os.path.join(RES, f"single_gain_{a.tag}.npz")):
            fig11(a.tag)
        table3(a.tag)
        cuts(a.tag)


if __name__ == "__main__":
    main()
