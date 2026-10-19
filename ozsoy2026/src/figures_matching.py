"""Evidence figures: one claim per figure, each with its residual.

    python figures_matching.py      -> ../figures/m01_*.png ... m08_*.png
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
from zoa import sld

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
OURS, PAPER, GREY, GOOD, BAD = "#1f5fa8", "#d1495b", "#6c757d", "#2e7d32", "#c62828"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def pts(name, key):
    d = json.load(open(os.path.join(ROOT, "digitized", name)))
    p = np.array([q for s in d[key] for q in s])
    return p[np.argsort(p[:, 0])]


def csvpts(name):
    return np.loadtxt(os.path.join(ROOT, "digitized", name), delimiter=",", skiprows=1)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def m01_uncontrolled_sld():
    u = csvpts("fig9_uncontrolled_points.csv")
    u = u[(u[:, 0] > 400) & (u[:, 0] < 3000)]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(u[:, 0], u[:, 1], color=PAPER, lw=3.5, alpha=0.35, label="paper Fig. 9 (vector data)")
    out = {}
    for z, col, ls, lab in [(0.0134, OURS, "-", "model, zeta = 1.34 %"),
                            (0.0145, GREY, "--", "model, zeta = 1.45 % (Table 1)")]:
        P.set_damping(z)
        a = sld(u[:, 0], 0.0) * 1e3
        ax[0].plot(u[:, 0], a, ls, color=col, lw=1, label=lab)
        rel = 100 * (a / u[:, 1] - 1)
        ax[1].plot(u[:, 0], rel, ls, color=col, lw=0.8)
        out[z] = np.median(np.abs(rel))
    P.set_damping(P.ZETA_PAPER_CURVES)
    ax[0].set(ylabel="a_lim (mm)", ylim=(0, 31),
              title="M1  Uncontrolled stability lobes: model vs paper")
    ax[0].legend(fontsize=7.5)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="error (%)", ylim=(-2, 14))
    ax[1].text(0.01, 0.92, f"median |error|: 1.34 % -> {out[0.0134]:.2f} %,   1.45 % -> {out[0.0145]:.2f} %",
               transform=ax[1].transAxes, va="top", fontsize=8)
    save(fig, "m01_uncontrolled_sld_match.png")


def m02_structure_frf():
    i = pts("fig5_frf.json", "identified")
    m = pts("fig5_frf.json", "measured_dashed")
    f = np.linspace(50, 210, 3000)
    P.set_damping(0.0145)
    g145 = np.abs(G(2j * np.pi * f))
    gi = np.abs(G(2j * np.pi * i[:, 0]))
    P.set_damping(P.ZETA_PAPER_CURVES)
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(i[:, 0], i[:, 1] * 1e6, color=PAPER, lw=3.5, alpha=0.35, label="paper Fig. 5, identified")
    ax[0].plot(m[:, 0], m[:, 1] * 1e6, ":", color=PAPER, lw=1, label="paper Fig. 5, measured")
    ax[0].plot(f, g145 * 1e6, color=OURS, lw=1, label="model, Table 1 (129.3 Hz, 1.45 %, 1.34e7 N/m)")
    rel = 100 * (gi / i[:, 1] - 1)
    ax[1].plot(i[:, 0], rel, color=OURS, lw=0.8)
    ax[0].set(ylabel="|G| (um/N)", title="M2  Structure FRF: Table 1 model vs paper Fig. 5")
    ax[0].legend(fontsize=7.5)
    ax[1].set(xlabel="Frequency (Hz)", ylabel="error (%)", ylim=(-3, 3), xlim=(50, 210))
    ax[1].text(0.01, 0.9, f"median |error| {np.median(np.abs(rel)):.2f} %, max {np.abs(rel).max():.2f} %",
               transform=ax[1].transAxes, va="top", fontsize=8)
    save(fig, "m02_structure_frf_match.png")


def m03_closed_loop():
    d = json.load(open(os.path.join(ROOT, "digitized", "fig6a_closed_loop_frf.json")))
    poles = list(csv.DictReader(open(os.path.join(ROOT, "verification", "targets", "fig6b_poles.csv"))))
    cases = [("uncontrolled", 0.0, "k", "uncontrolled"), ("1300rpm_g1162_dashed", 1162.0, "C2", "1300 rpm, g = 1162")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for key, g, col, lab in cases:
        p = np.array([q for s in d[key] for q in s])
        p = p[(p[:, 0] > 1) & (p[:, 0] < 250)]
        p = p[np.argsort(p[:, 0])]
        mdl = 20 * np.log10(np.abs(Gcl(2j * np.pi * p[:, 0], g)))
        ax[0].plot(p[:, 0], p[:, 1], color=col, lw=3.5, alpha=0.3)
        ax[0].plot(p[:, 0], mdl, color=col, lw=1, label=f"{lab} (model thin, paper thick)")
        ax[1].plot(p[:, 0], p[:, 1] - mdl, color=col, lw=0.8,
                   label=f"{lab}: rms {np.sqrt(np.mean((p[:, 1]-mdl)**2)):.3f} dB")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="|Gcl| (dB re 1 m/N)", xlim=(0, 250), ylim=(-155, -108),
              title="M3  Closed loop vs paper Fig. 6a")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="Frequency (Hz)", ylabel="paper - model (dB)", xlim=(0, 250), ylim=(-0.6, 0.6),
              title="Residual")
    ax[1].legend(fontsize=7)
    ax[1].text(0.02, 0.04, "spike at the peak: the published panel's frequency axis\nis offset by 0.21 % (its own fn marker sits at 129.29 Hz,\nits peak at 129.01 Hz)", transform=ax[1].transAxes, fontsize=6.5)
    for case, g, col in [("open_loop", 0.0, "k"), ("1300rpm_g1162", 1162.0, "C2")]:
        s = closed_loop_poles(g)
        s = s[s.imag > 100]
        pp = [r for r in poles if r["case"] == case and float(r["imag_rad_s"]) > 100][0]
        ax[2].plot(float(pp["real_rad_s"]), float(pp["imag_rad_s"]), "o", ms=11, mfc="none", mec=col, mew=1.5)
        ax[2].plot(s.real, s.imag, "x", color=col, ms=8)
        ax[2].annotate(f"paper {float(pp['real_rad_s']):.2f}\nmodel {s.real[0]:.2f}",
                       (s.real[0], s.imag[0]), textcoords="offset points", xytext=(6, -26), fontsize=7)
    ax[2].plot([], [], "o", mfc="none", mec="k", label="paper Fig. 6b")
    ax[2].plot([], [], "x", color="k", label="model")
    ax[2].set(xlabel="Real part (rad/s)", ylabel="Imag part (rad/s)", xlim=(-115, 5), ylim=(790, 820),
              title="Dominant poles vs Fig. 6b")
    ax[2].legend(fontsize=7, loc="lower left")
    save(fig, "m03_closed_loop_match.png")


def m04_zoa_vs_sdm():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "verification", "zoa_sdm", "compare_12speeds.csv"))))
    zoa = np.array([float(r["a_zoa_src_mm"]) for r in rows])
    sdm = np.array([float(r["a_sdm_meanH_mm"]) for r in rows])
    hel = np.array([float(r["a_sdm_helix_mm"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    lim = [0, max(zoa.max(), hel.max()) * 1.05]
    ax[0].plot(lim, lim, color=GREY, lw=0.8)
    ax[0].plot(sdm, zoa, "o", ms=4, mfc="none", color=OURS, label="SDM, averaged force (same model)")
    ax[0].plot(hel, zoa, "+", ms=5, color=PAPER, label="SDM, time-periodic force with helix")
    ax[0].set(xlabel="independent SDM a_lim (mm)", ylabel="our ZOA a_lim (mm)", xlim=lim, ylim=lim,
              title="M4  Stability solver vs independent SDM (96 cases)")
    ax[0].legend(fontsize=7.5)
    r1 = np.abs(zoa / sdm - 1)
    r2 = zoa / hel - 1
    ax[1].semilogy(np.arange(len(r1)), np.maximum(r1, 1e-9), "o", ms=3, mfc="none", color=OURS,
                   label=f"vs averaged SDM: max {r1.max():.1e}")
    ax[1].semilogy(np.arange(len(r2)), np.maximum(np.abs(r2), 1e-9), "+", ms=4, color=PAPER,
                   label=f"vs helical SDM: {100*r2.min():.2f} % to {100*r2.max():.2f} %")
    ax[1].set(xlabel="case (zeta x gain x speed)", ylabel="|relative difference|", ylim=(1e-9, 1e-1),
              title="Relative difference")
    ax[1].legend(fontsize=7.5)
    save(fig, "m04_zoa_vs_sdm.png")


def m05_simulator():
    c = json.load(open(os.path.join(ROOT, "verification", "tdsim", "check7_final_scheme.json")))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    tr = c["trajectory"]
    lab = [f"{t['rpm']}/{t['a_mm']:g}/{t['g']:.0f}" for t in tr]
    x = np.arange(len(tr))
    ax[0].bar(x - 0.2, [100 * t["rel_err_fmode0"] for t in tr], 0.4, color=GREY, label="force held per step")
    ax[0].bar(x + 0.2, [100 * t["rel_err_fmode1"] for t in tr], 0.4, color=OURS, label="force per RK4 stage (used)")
    ax[0].set_xticks(x, lab, rotation=30, fontsize=7)
    ax[0].set(ylabel="max error in x (%)", title="M5a  512 steps/rev vs exact (8192)")
    ax[0].legend(fontsize=7)
    dm = c["demand"]
    ax[1].bar(np.arange(len(dm)), [100 * d["rel"] for d in dm], color=OURS)
    ax[1].set_xticks(np.arange(len(dm)), [f"{d['rpm']}/{d['g']:.0f}" for d in dm], fontsize=7)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(ylabel="demand 512 vs 8192 (%)", ylim=(-0.5, 0.5), title="M5b  Peak demand at a_lim")
    bd = c["boundary"]
    ax[2].plot([b["zoa_mm"] for b in bd], [b["td_mm"] for b in bd], "o", color=OURS)
    lim = [0, 20]
    ax[2].plot(lim, lim, color=GREY, lw=0.8)
    ax[2].set(xlabel="ZOA a_lim (mm)", ylabel="time-domain limit (mm)", xlim=lim, ylim=lim,
              title=f"M5c  Time-domain vs ZOA: +{100*min(b['rel'] for b in bd):.1f} to +{100*max(b['rel'] for b in bd):.1f} %")
    save(fig, "m05_simulator_match.png")


def m06_optimiser():
    d = np.load(os.path.join(ROOT, "results", "gainmap_main.npz"))
    rpm = d["rpm"]
    rel = 100 * (d["a_sade"] / d["a_ref"] - 1)
    runs = d["sade_runs"]
    spread = runs[:, :, 1].max(1) - runs[:, :, 1].min(1)
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    ax[0].plot(rpm, rel, ".", color=OURS, ms=3)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(ylabel="SADE vs reference (%)", ylim=(-5, 5),
              title=f"M6  Optimiser: SADE reaches the reference optimum ({np.sum(np.abs(rel) < 1)}/251 within 1 %)")
    off = [f"{n}: {rel[int(np.argmin(np.abs(rpm - n)))]:+.0f} %" for n in (2280, 2300, 2310)]
    ax[0].text(0.01, 0.95, "off the scale (isolated feasible pocket, map takes the better):  " + ",  ".join(off),
               transform=ax[0].transAxes, va="top", fontsize=7.5,
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax[1].semilogy(rpm, np.maximum(spread, 1e-6), ".", color=GREY, ms=3)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="spread of 3 seeds (mm)", ylim=(1e-6, 10))
    ax[1].text(0.01, 0.9, f"identical (< 0.01 mm) at {np.sum(spread < 0.01)}/251 speeds; the outliers sit on a "
               "fragmented feasible set (2280-2310 rpm)", transform=ax[1].transAxes, va="top", fontsize=7.5)
    save(fig, "m06_optimiser_match.png")


def m07_high_speed_gain():
    d = np.load(os.path.join(ROOT, "results", "gainmap_main.npz"))
    g8 = csvpts("fig8_gain_points.csv")
    rpm, g = d["rpm"], d["g_best"]
    gp = np.interp(rpm, g8[:, 0], g8[:, 1])
    m = rpm >= 2060
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(g8[:, 0], g8[:, 1], color=PAPER, lw=3.5, alpha=0.35, label="paper Fig. 8")
    ax[0].plot(rpm[m], g[m], color=OURS, lw=1.2, label="reproduction")
    ax[0].set(ylabel="g_dvf (V s/m)", xlim=(2060, 3000), ylim=(300, 900),
              title="M7  Gain map at high speed vs paper Fig. 8")
    ax[0].legend(fontsize=7.5)
    r = g[m] / gp[m]
    hi = rpm[m] >= 2320
    ax[1].plot(rpm[m][hi], r[hi], ".", color=GOOD, ms=4, label=f"2320-3000: median {np.median(r[hi]):.2f}")
    ax[1].plot(rpm[m][~hi], r[~hi], ".", color=BAD, ms=4, label=f"2060-2310: median {np.median(r[~hi]):.2f}")
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="ours / paper", ylim=(0.8, 2.0))
    ax[1].legend(fontsize=7.5, loc="upper right")
    save(fig, "m07_high_speed_gain_match.png")


def m08_test_points():
    rows = json.load(open(os.path.join(ROOT, "results", "cuts_main.json")))
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ok = 0
    for r in rows:
        good = r["pred_paper"] == r["measured"]
        ok += good
        mk = {"none": "s", "map": "o", "single": "^"}[r["controller"]]
        col = GOOD if good else BAD
        ax.plot(r["depth_mm"] / r["alim_paper"], r["rpm"], mk, ms=7, mfc=col if r["measured"] == "chatter" else "none",
                mec=col, mew=1.4)
    ax.axvline(1, color="k", lw=0.8)
    ax.text(0.98, 0.04, "right of the line: predicted chatter", transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.text(0.02, 0.04, "left: predicted stable", transform=ax.transAxes, fontsize=7.5)
    for mk, lab in [("s", "uncontrolled"), ("o", "paper's mapped gain"), ("^", "paper's single gain 237")]:
        ax.plot([], [], mk, mfc="none", mec="k", label=lab)
    ax.plot([], [], "o", mfc="k", mec="k", label="filled: measured chatter; open: measured stable")
    ax.plot([], [], "o", mfc="none", mec=GOOD, label="green: model agrees")
    ax.plot([], [], "o", mfc="none", mec=BAD, label="red: model disagrees")
    ax.set(xscale="log", xlabel="measured depth / model a_lim at the paper's gain", ylabel="Spindle speed (rpm)",
           ylim=(500, 3200), title=f"M8  Published test points with the paper's own gains: {ok}/{len(rows)} agree")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    save(fig, "m08_test_points_match.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for f in (m01_uncontrolled_sld, m02_structure_frf, m03_closed_loop, m04_zoa_vs_sdm,
              m05_simulator, m06_optimiser, m07_high_speed_gain, m08_test_points):
        f()
        print("done", f.__name__)
