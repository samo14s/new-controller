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
    ax.plot(rpm, d["g_best"], color=OURS, lw=1.4, label="reproduction (best of SADE and reference)")
    if "g_sade" in d:
        ax.plot(rpm, d["g_sade"], ".", color="k", ms=2.5, label="SADE, best of 3 seeds")
    ax.plot(rpm, d["g_ref"], "x", color="0.5", ms=2.5, label="reference scan + edge bisection")
    ax.set(xlabel="Spindle speed (rpm)", ylabel="g_dvf (V s/m)", xlim=(500, 3000),
           title="Spindle-speed dependent optimised gain")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, f"r03_fig8_gain_map_{tag}.png"))
    plt.close(fig)

    fine = np.arange(380, 3101, 2.0)
    a0 = sld(fine, 0.0) * 1e3
    a_map = d["a_best"] if "a_best" in d else d["a_ref"]
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
    x = np.arange(len(t))
    w = 0.16
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - 2 * w, t["paper_clip"], w, color=PAPER, alpha=0.45, label="paper: clipped DVF")
    ax.bar(x - w, t["paper_prop"], w, color=PAPER, label="paper: proposed")
    ax.bar(x, t["a_clip1500_td"], w, color="#7aa6d6", label="reproduction: clipped DVF, g = 1500 (TD, 9 V limit)")
    ax.bar(x + w, t["a_clip3000_td"], w, color="#9ec5ea", label="reproduction: clipped DVF, g = 3000 (TD, 9 V limit)")
    ax.bar(x + 2 * w, t["a_prop_td"], w, color=OURS, label="reproduction: proposed gain (TD, 9 V limit)")
    ax.set_xticks(x, [f"{int(r)}" for r in t["rpm"]])
    ax.set(xlabel="Spindle speed (rpm)", ylabel="Limiting depth (mm)",
           title="Table 3: clipped DVF vs saturation-aware gain")
    ax.legend(fontsize=6.5, ncol=2)
    fig.savefig(os.path.join(FIG, f"r06_table3_{tag}.png"))
    plt.close(fig)


def cuts(tag):
    path = os.path.join(RES, f"cuts_{tag}.npz")
    if not os.path.exists(path):
        return
    c = np.load(path)
    with open(os.path.join(RES, f"cuts_{tag}.json")) as fh:
        rows = sorted([r for r in json.load(fh) if r["cut"]], key=lambda r: r["cut"])
    fig, ax = plt.subplots(len(rows), 2, figsize=(9, 1.75 * len(rows)),
                           gridspec_kw={"width_ratios": [2.2, 1]})
    for k, r in enumerate(rows):
        lab = r["cut"]
        ax[k, 0].plot(c[f"{lab}_t"], c[f"{lab}_acc"], color="#4a90c2", lw=0.4)
        ax[k, 0].plot(c[f"{lab}_opr_t"], c[f"{lab}_opr"], "k.", ms=2)
        ax[k, 0].set_ylabel("acc (m/s2)", fontsize=7)
        tag_d = " (diverges)" if r.get("diverged_td") else ""
        ax[k, 0].set_title(f"Cut {lab}: {r['rpm']:.0f} rpm, {r['depth_mm']} mm, {r['controller']} "
                           f"(g = {r['g_ours']:.0f}) -> simulated {r['pred_ours_td']}{tag_d}; "
                           f"measured {r['measured']}", fontsize=7.5)
        ax[k, 1].semilogy(c[f"{lab}_f"], c[f"{lab}_S"] + 1e-12, color="#4a90c2", lw=0.8)
        ftp = 4 * r["rpm"] / 60.0
        for h in range(1, 4):
            ax[k, 1].axvline(h * ftp, color="k", lw=0.5, ls=":")
        ax[k, 1].set_xlim(0, 420)
    ax[-1, 0].set_xlabel("Time (s); dots: once per revolution")
    ax[-1, 1].set_xlabel("Frequency (Hz); dotted: tooth passing")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"r07_cuts_{tag}.png"))
    plt.close(fig)


def points(tag):
    path = os.path.join(RES, f"cuts_{tag}.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        rows = json.load(fh)
    d = np.load(os.path.join(RES, f"gainmap_{tag}.npz"))
    sg = np.load(os.path.join(RES, f"single_gain_{tag}.npz"))
    fine = np.arange(400, 3051, 5.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ax[0].plot(fine, sld(fine, 0.0) * 1e3, "--", color=UNC, lw=1, label="uncontrolled")
    ax[0].plot(d["rpm"], d["a_best"], color=OURS, lw=1.4, label="reproduced gain map")
    cp = np.loadtxt(os.path.join(DIG, "fig9_controlled_points.csv"), delimiter=",", skiprows=1)
    ax[0].plot(cp[:, 0], cp[:, 1], color=PAPER, lw=1, alpha=0.7, label="paper Fig. 9 controlled")
    ax[1].plot(fine, sld(fine, 0.0) * 1e3, "--", color=UNC, lw=1)
    ax[1].plot(fine, sld(fine, float(sg["g_ref"])) * 1e3, color=OURS, lw=1.4,
               label=f"reproduced single gain {float(sg['g_ref']):.0f}")
    ax[1].plot(fine, sld(fine, 237.0) * 1e3, color=PAPER, lw=1, alpha=0.7, label="paper single gain 237 (our model)")
    for r in rows:
        k = 0 if r["figure"] == "Fig9" else 1
        mk = {"none": "s", "map": "o", "single": "o"}[r["controller"]]
        ok = r["pred_ours_td"] == r["measured"]
        col = "g" if r["measured"] == "stable" else "r"
        ax[k].plot(r["rpm"], r["depth_mm"], mk, mfc=col if ok else "none", mec=col, ms=6)
    for k in range(2):
        ax[k].plot([], [], "o", mfc="g", mec="g", label="measured stable, predicted stable")
        ax[k].plot([], [], "o", mfc="r", mec="r", label="measured chatter, predicted chatter")
        ax[k].plot([], [], "o", mfc="none", mec="k", label="open: prediction disagrees")
        ax[k].set(xlabel="Spindle speed (rpm)", xlim=(400, 3050), ylim=(0, 20))
        ax[k].legend(fontsize=6.3, loc="upper right")
    ax[0].set(ylabel="Depth of cut (mm)", title="Mapped gain: published test points")
    ax[1].set(title="Single gain: published test points")
    fig.savefig(os.path.join(FIG, f"r09_test_points_{tag}.png"))
    plt.close(fig)


def sensitivity(tag="main"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = [("main", "reproduction (zeta 1.34 %, force, stage-wise force)", OURS, "-"),
              ("zeta145", "zeta 1.45 % (Table 1)", "C2", "--"),
              ("volt", "demand read as g|xdot| (volts) <= 27", "C1", ":"),
              ("hold512", "force held per step (512 steps/rev)", "C4", "-.")]
    gp = np.loadtxt(os.path.join(DIG, "fig8_gain_points.csv"), delimiter=",", skiprows=1)
    cp = np.loadtxt(os.path.join(DIG, "fig9_controlled_points.csv"), delimiter=",", skiprows=1)
    ax[0].plot(gp[:, 0], gp[:, 1], color=PAPER, lw=1.6, label="paper Fig. 8")
    ax[1].plot(cp[:, 0], cp[:, 1], color=PAPER, lw=1.6, label="paper Fig. 9 (incl. covered part)")
    for t, lab, col, ls in styles:
        p = os.path.join(RES, f"gainmap_{t}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        g = d["g_best"] if "g_best" in d else d["g_ref"]
        a = d["a_best"] if "a_best" in d else d["a_ref"]
        ax[0].plot(d["rpm"], g, ls, color=col, lw=1.1, label=lab)
        ax[1].plot(d["rpm"], a, ls, color=col, lw=1.1, label=lab)
    ax[0].set(xlabel="Spindle speed (rpm)", ylabel="g_dvf (V s/m)", xlim=(500, 3000),
              title="Gain map: sensitivity to unstated choices")
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="a_lim (mm)", xlim=(500, 3000), ylim=(0, 40),
              title="Controlled limit: same cases")
    ax[0].legend(fontsize=6.5)
    ax[1].legend(fontsize=6.5)
    fig.savefig(os.path.join(FIG, "r08_sensitivity.png"))
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
        points(a.tag)
        sensitivity(a.tag)


if __name__ == "__main__":
    main()
