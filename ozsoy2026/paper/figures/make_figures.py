"""Publication figures of the manuscript (Figs. 2-10), drawn from the stored results.

    cd paper/figures && OMP_NUM_THREADS=1 python make_figures.py            # all nine
    cd paper/figures && OMP_NUM_THREADS=1 python make_figures.py fig03 fig07

Writes figNN_<name>.pdf (vector, TrueType fonts embedded) and figNN_<name>.png
(300 dpi preview) beside this file, and figure_numbers.json with every number
that the captions (captions.tex) quote, computed from the same arrays.

Data: ../../results/*.npz (see README.md in this folder for the file of each
figure); model code is imported from ../../src.  Two things are computed here
rather than read:
  * Fig. 3 runs two short simulations of the time-domain simulator (the
    protocol of src/c1_unsafe_zone.py and src/figures_c1.py: face entry, 9 V
    clip, full surface memory, kick at revolution 100);
  * Figs. 6, 7 and 9 rescale the stored C2/C4 certificates to the closed-form
    command bound of C3 (c3_periodic.orbit), exactly as src/figures_c2.py and
    src/c3_summary.py do (README 14.9).  The factors are 0.86-1 except where
    the margin is almost nil.

Journal style: no titles, panel letters top left, 8 pt Liberation Sans
(Helvetica/Arial metrics), 0.8-1.2 pt lines, 90 or 190 mm wide, Okabe-Ito colours:
DVF black; reformulated condition and delayed term blue; Eq. (2) of the
published method grey; uncontrolled dashed grey; certificates C2 orange and C3
green (open markers); simulations filled markers.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker, transforms
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True                        # write nothing into ../../src
import params as P                                    # noqa: E402

P.set_damping(P.ZETA_PAPER_CURVES)                    # 1.34 %, as every stored campaign
VMAX = P.F_MAX / P.GA_GAIN                            # 9 V
SPEEDS = (1090, 1420, 2500, 2870)                     # the four basin speeds of C1-C4

# ------------------------------------------------------------------ style ----
MM = 1.0 / 25.4
W1, W2 = 90 * MM, 190 * MM
K = "#000000"           # DVF
BLUE = "#0072B2"        # reformulated condition; delayed term (g2 = 2000, map design)
BLUE_L = "#56B4E9"      # delayed term g2 = 1000; same-g1 design
BLUE_D = "#002D5A"      # delayed term g2 = 4000
GREY = "#7A7A7A"        # Eq. (2) of the published method (solid); uncontrolled (dashed)
C2C = "#E69F00"         # certificate C2 (averaged model)
C3C = "#009E73"         # certificate C3 (periodic model)
STEP = "#D55E00"        # simulated step entry
FACE = "#CC79A7"        # simulated face entry; describing-function estimate (Fig. 5)
LIGHT = "#E3E3E3"       # bistable band (fill)
STEP_FILL = "#F6D3BD"   # step entry ends in chatter (fill; STEP lightened, opaque)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans",
    "mathtext.fallback": "stixsans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.8, "patch.linewidth": 0.8, "hatch.linewidth": 0.6,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.columnspacing": 1.2, "legend.labelspacing": 0.3,
    "axes.labelpad": 2.0, "figure.dpi": 100, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
})

NUM = {}               # numbers quoted in the captions -> figure_numbers.json


def load(name):
    return np.load(os.path.join(RES, name))


def save(fig, stem):
    fig.savefig(os.path.join(HERE, stem + ".pdf"))
    fig.savefig(os.path.join(HERE, stem + ".png"))
    plt.close(fig)
    print("wrote", stem, flush=True)


def letter(ax, s, text=None, out=False):
    """Panel letter '(s)' (and an optional label) in the upper-left corner of the panel:
    inside the axes on a white ground, or (out=True) just above the axes, where a
    row of 'no chatter up to 3 m/s' markers runs along the top of the axes."""
    lab = f"({s})" + (f" {text}" if text else "")
    if out:
        return ax.annotate(lab, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, 2.5),
                           textcoords="offset points", ha="left", va="bottom", fontsize=8)
    t = ax.annotate(lab, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(3.0, -3.0), textcoords="offset points",
                    ha="left", va="top", fontsize=8, zorder=20)
    t.set_bbox(dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9))
    return t


def top_marks(ax, x, color, marker="^", y=0.94, **kw):
    """Markers at a fixed height in axes units (for 'no chatter up to the search limit')."""
    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    return ax.plot(x, np.full(len(x), y), marker, color=color, transform=tr, ls="none", clip_on=False, **kw)


def r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def rng(x, nd=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(min=r(x.min(), nd), median=r(np.median(x), nd), max=r(x.max(), nd), n=int(x.size))


# ---------------------------------------- C2/C4 certificates, corrected bound ----
def c2_factor(n, g, a):
    """C2 certificate size factor (9 V - b_C3) / (9 V - b_C2) (src/figures_c2.bound_factor, README 14.9)."""
    import c2_certificate as c2
    import c3_periodic as c3
    return (VMAX - c3.orbit(n, g, a)[2].max()) / (VMAX - c2.orbit(n, g, a)[2])


def surface_factor(S):
    """c2_surface rescaled with its stored u0 = 9 V - b_C2 (src/c3_summary.margin_errata)."""
    import c3_periodic as c3
    fac = np.ones_like(S["cert"])
    for i, (n, g, al) in enumerate(zip(S["rpm"], S["g"], S["a_lim"])):
        for j, fr in enumerate(S["frac"]):
            if S["cert"][i, j] > 0:
                bk = c3.orbit(n, g, fr * al * 1e-3)[2].max()
                fac[i, j] = max(0.0, (VMAX - bk) / S["u0"][i, j])
    return fac


def gainmap_factors(G):
    """c4_gainmap certificates rescaled with the stored b_max (src/c3_summary.margin_errata)."""
    import c3_periodic as c3
    fd = np.ones_like(G["kick_dvf"])
    fc = np.ones_like(G["kick"])
    for i, n in enumerate(G["rpm"]):
        for j, dep in enumerate(G["depth"]):
            for k, g1 in enumerate(G["g1_dvf"]):
                if G["kick_dvf"][i, j, k] > 0:
                    bk = c3.orbit(n, g1, dep * 1e-3)[2].max()
                    fd[i, j, k] = max(0.0, (VMAX - bk) / (VMAX - G["bmax_dvf"][i, j, k]))
            for k, g1 in enumerate(G["g1"]):
                if np.any(G["kick"][i, j, k] > 0):
                    bk = c3.orbit(n, g1, dep * 1e-3)[2].max()
                    fc[i, j, k] = np.maximum(0.0, (VMAX - bk) / (VMAX - G["bmax"][i, j, k]))
    return fd, fc


# =============================================================== Fig. 2 ====
def fig02():
    """Gain maps and depths: Eq. (2) as written (reproduced) vs the reformulated condition."""
    f, c, e, m = load("feas_g1500.npz"), load("feas_cert_g1500.npz"), load("gainmap_main.npz"), load("c1_map.npz")
    fig, ax = plt.subplots(1, 2, figsize=(W2, 72 * MM), layout="constrained")
    a = ax[0]
    a.plot(e["rpm"], e["g_best"], color=GREY, lw=1.0, label="Published condition (reproduced)")
    a.plot(f["rpm"], f["g_best"], color=BLUE, lw=1.2, label=r"Reformulated condition, $g\leq1500$ V s/m")
    a.set(xlabel="Spindle speed $n$ (rpm)", ylabel="Gain $g$ (V s/m)", xlim=(500, 3000), ylim=(0, 1700))
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0.0)
    b = ax[1]
    b.plot(m["rpm"], m["a_lim0"], "--", color=GREY, lw=1.0, label=r"$a_{\mathrm{lim}}(0)$, uncontrolled")
    b.plot(e["rpm"], e["a_best"], color=GREY, lw=1.0, label=r"$a_{\mathrm{lim}}$ at the published-condition gains")
    b.plot(c["rpm"], c["a_op_linear"], ":", color=BLUE, lw=1.2,
           label=r"$\kappa\,a_{\mathrm{lim}}$, reformulated ($\kappa=0.9$)")
    b.plot(c["rpm"], c["a_cert"], color=BLUE, lw=1.2, label=r"$a_{\mathrm{chk}}$, reformulated")
    b.set(xlabel="Spindle speed $n$ (rpm)", ylabel="Axial depth of cut $a$ (mm)", xlim=(500, 3000), ylim=(0, 32))
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0.0)
    letter(a, "a")
    letter(b, "b")
    save(fig, "fig02_feasibility")
    gov = c["a_td_clipped"] < c["a_op_linear"]
    NUM["fig02"] = dict(
        speeds=int(len(f["rpm"])), gain_at_1500=int(np.sum(f["g_best"] >= 1500 - 1e-6)),
        gain_min=[r(f["g_best"].min(), 1), float(f["rpm"][np.argmin(f["g_best"])])],
        eq2_gain_min=[r(e["g_best"].min(), 1), float(e["rpm"][np.argmin(e["g_best"])])],
        mean_a_chk_mm=r(c["a_cert"].mean()), mean_a_lim_eq2_mm=r(e["a_best"].mean()),
        mean_a_lim0_mm=r(m["a_lim0"].mean()), td_clipped_governs=int(gov.sum()),
        min_a_chk=[r(c["a_cert"].min()), float(c["rpm"][np.argmin(c["a_cert"])])],
        a_chk_over_a_lim=[r((c["a_cert"] / c["a_lim"]).min()), r((c["a_cert"] / c["a_lim"]).max())],
        a_chk_2500=r(float(c["a_cert"][c["rpm"] == 2500][0])), a_lim_2500=r(float(c["a_lim"][c["rpm"] == 2500][0])))


# =============================================================== Fig. 3 ====
FIG3 = dict(rpm=2500.0, g=1500.0, frac=0.9, kicks=(7.0e-3, 7.6e-3), phase=0)


def fig03():
    """The same cut after a small and a slightly larger kick (two short simulations)."""
    from tdsim import simulate
    from tdboundary import classify_run
    from zoa import a_lim
    from c1_unsafe_zone import NREG, KICK_REV, WINDOW, N_PHASE, SPR, ND
    rpm, g, frac = FIG3["rpm"], FIG3["g"], FIG3["frac"]
    al = a_lim(rpm, g)
    a = frac * al
    ks = KICK_REV * SPR + (FIG3["phase"] * ND) // N_PHASE       # phase 0: start of a tooth period
    B = load("c1_basin.npz")
    row = np.nonzero((B["rpm"] == rpm) & np.isclose(B["frac"], frac) & (B["vmax"] > 0))[0][0]
    crit = float(B["dv"][row, FIG3["phase"]])
    runs = []
    for dv in FIG3["kicks"]:
        out = simulate(rpm, a, g, vmax=VMAX, store=True, nrev=KICK_REV + WINDOW, edge=True,
                       kick_step=ks, kick_dv=dv, nreg=NREG)
        st = (ks // ND + 1) * ND
        stable = classify_run(out["x"][st:], SPR, out["diverged"])["stable"]
        runs.append((dv, out, stable))
    assert runs[0][2] and not runs[1][2], "expected: small kick settles, larger kick chatters"
    T = 60.0 / rpm                                     # one revolution
    t0, t1 = -1 * T, 30 * T
    fig, ax = plt.subplots(2, 2, figsize=(W2, 92 * MM), layout="constrained", sharex=True)
    for j, (dv, out, stable) in enumerate(runs):
        t = out["t"] - out["t"][ks]
        w = (t >= t0) & (t <= t1)
        ax[0, j].plot(t[w], out["x"][w] * 1e6, color=K, lw=0.8)
        ax[1, j].plot(t[w], out["vcmd"][w], color=K, lw=0.8)
        for s in (1, -1):
            ax[1, j].axhline(s * VMAX, color=STEP, lw=0.9, ls="--")
        for q in range(2):
            ax[q, j].axvline(0.0, color=GREY, lw=0.8, ls=":")
        ax[1, j].set(xlabel="Time after the kick $t-t_k$ (s)", ylim=(-11, 13.5), xlim=(t0, t1),
                     yticks=(-10, -5, 0, 5, 10))
    ax[0, 0].set_ylabel(r"Displacement $x$ (µm)")
    ax[1, 0].set_ylabel("Command $V$ (V)")
    ax[0, 0].set_ylim(-36, -8)
    ax[0, 1].set_ylim(-520, 520)
    for j in range(2):
        ax[1, j].text(t1, VMAX, r"$V_{\max}=9$ V ", color=STEP, ha="right", va="bottom", fontsize=8)
    lab0 = rf"$\Delta v={runs[0][0]*1e3:.1f}$ mm/s: returns to the forced orbit"
    lab1 = rf"$\Delta v={runs[1][0]*1e3:.1f}$ mm/s: clipped chatter"
    letter(ax[0, 0], "a", lab0)
    letter(ax[0, 1], "b", lab1)
    letter(ax[1, 0], "c")
    letter(ax[1, 1], "d")
    save(fig, "fig03_two_outcomes")
    dv0, o0, _ = runs[0]
    dv1, o1, _ = runs[1]
    end = slice(-SPR, None)
    NUM["fig03"] = dict(
        rpm=rpm, g=g, frac=frac, a_lim_mm=r(al * 1e3), depth_mm=r(a * 1e3), kick_revolution=KICK_REV,
        kick_phase="0 of 16 (start of a tooth period)", kicks_mm_s=[dv0 * 1e3, dv1 * 1e3],
        critical_kick_this_phase_mm_s=r(crit * 1e3), critical_kick_min_16_phases_mm_s=r(np.nanmin(B["dv"][row]) * 1e3),
        critical_kick_median_16_phases_mm_s=r(np.median(B["dv"][row]) * 1e3),
        orbit_x_um=[r(o0["x"][end].min() * 1e6, 1), r(o0["x"][end].max() * 1e6, 1)],
        orbit_V=[r(o0["vcmd"][end].min()), r(o0["vcmd"][end].max())],
        chatter_x_um=[r(o1["x"][end].min() * 1e6, 0), r(o1["x"][end].max() * 1e6, 0)],
        chatter_V=[r(o1["vcmd"][end].min()), r(o1["vcmd"][end].max())],
        run="face entry, |V| <= 9 V, full surface memory (nreg 0), 400 revolutions, judged over the 300 after the kick")


# =============================================================== Fig. 4 ====
def fig04():
    """Entry limits and the bistable band over 251 speeds (C1)."""
    d, bd = load("c1_map.npz"), load("c1_band.npz")
    n, al = d["rpm"], d["a_lim"]
    lowb = bd["a_band_low"]
    al_b = np.interp(bd["rpm"], n, al)
    fig, ax = plt.subplots(2, 1, figsize=(W2, 118 * MM), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [2.3, 1]})
    a = ax[0]
    a.fill_between(bd["rpm"], lowb, al_b, where=np.isfinite(lowb), color=LIGHT, lw=0, interpolate=False,
                   label="Bistable band")
    a.fill_between(n, d["a_step_clip"], al, where=d["a_step_clip"] < al, color=STEP_FILL, lw=0,
                   label="Step entry ends in chatter")
    a.plot(n, d["a_lim0"], "--", color=GREY, lw=1.0, label=r"$a_{\mathrm{lim}}(0)$, uncontrolled")
    a.plot(n, al, color=K, lw=1.2, label=r"$a_{\mathrm{lim}}(g)$, linear")
    a.plot(n, d["a_edge_clip"], "--", color=FACE, lw=1.0, label=r"Face entry, $|V|\leq9$ V")
    a.plot(n, d["a_step_clip"], color=STEP, lw=1.0, label=r"Step entry, $|V|\leq9$ V")
    a.plot(bd["rpm"], lowb, "o", color=K, ms=2.2, mew=0, label="Lower edge of the bistable band")
    a.set(ylabel="Axial depth of cut $a$ (mm)", ylim=(0, 33), xlim=(480, 3020))
    h, lab = a.get_legend_handles_labels()
    order = [3, 2, 5, 4, 6, 0, 1]
    a.legend([h[i] for i in order], [lab[i] for i in order], loc="lower left", bbox_to_anchor=(0.0, 1.01),
             ncol=3, borderaxespad=0.0)
    b = ax[1]
    b.axhline(1.0, color=K, lw=0.8)
    b.plot(n, d["a_edge_clip"] / al, "--", color=FACE, lw=1.0)
    b.plot(n, d["a_step_clip"] / al, color=STEP, lw=1.0)
    b.plot(bd["rpm"], lowb / al_b, "o", color=K, ms=2.2, mew=0)
    b.set(xlabel="Spindle speed $n$ (rpm)", ylabel=r"$a\,/\,a_{\mathrm{lim}}(g)$ (–)", ylim=(0, 1.12))
    letter(a, "a")
    letter(b, "b")
    save(fig, "fig04_entry_map")
    rs = d["a_step_clip"] / al
    re_ = d["a_edge_clip"] / al
    rb = lowb / al_b
    NUM["fig04"] = dict(
        speeds=int(len(n)), step_below_097=int(np.sum(rs < 0.97)), step_ratio_min=[r(rs.min(), 3), float(n[np.argmin(rs)])],
        step_ratio_median=r(np.median(rs), 3), face_ratio_min=r(re_.min(), 3),
        band_speeds=int(len(lowb)), band_found=int(np.isfinite(lowb).sum()),
        band_ratio=rng(rb, 2), band_over_a_lim0_min=r(np.nanmin(lowb / np.interp(bd["rpm"], n, d["a_lim0"])), 2),
        mean_step_mm=r(np.mean(d["a_step_clip"])), mean_face_mm=r(np.mean(d["a_edge_clip"])), mean_a_lim_mm=r(al.mean()),
        at_2500=dict(a_lim=r(float(al[n == 2500][0])), step=r(float(d["a_step_clip"][n == 2500][0])),
                     face=r(float(d["a_edge_clip"][n == 2500][0]))))


# =============================================================== Fig. 5 ====
def fig05():
    """Critical kick vs depth with and without the clip; describing-function estimate (C1)."""
    b, d = load("c1_basin.npz"), load("c1_df.npz")
    top = float(b["dv_hi"]) * 1e3                       # the 3 m/s search limit
    fig, ax = plt.subplots(1, 4, figsize=(W2, 70 * MM), sharey=True, layout="constrained")
    ratios, shrink = [], []
    for i, n in enumerate(SPEEDS):
        a = ax[i]
        j = int(np.argmin(np.abs(d["rpm"] - n)))
        g = float(d["g"][j])
        m = (b["rpm"] == n) & (b["vmax"] > 0)
        f, dv = b["frac"][m], b["dv"][m] * 1e3
        fin = np.all(np.isfinite(dv), axis=1)
        med, lo, hi = np.median(dv, axis=1), dv.min(axis=1), dv.max(axis=1)
        a.axhline(top, color=GREY, lw=0.8, ls=":")
        a.plot(f[fin], med[fin], "o-", color=K, ms=3.2, lw=1.0, label=r"$|V|\leq9$ V: median over 16 phases")
        a.errorbar(f[fin], med[fin], yerr=[med[fin] - lo[fin], hi[fin] - med[fin]], fmt="none", ecolor=K,
                   elinewidth=0.8, capsize=2.2, capthick=0.8, label=r"$|V|\leq9$ V: range over 16 phases (lower end: minimum)")
        none = ~np.any(np.isfinite(dv), axis=1)
        a.plot(f[none] - 0.006, np.full(none.sum(), top), "^", color=K, ms=4,
               label=r"$|V|\leq9$ V: no chatter up to 3 m/s")
        mf = (b["rpm"] == n) & (b["vmax"] == 0)
        dvf = b["dv"][mf] * 1e3
        fmin = np.array([x[np.isfinite(x)].min() if np.isfinite(x).any() else np.inf for x in dvf])
        nf = ~np.isfinite(fmin)
        a.plot(b["frac"][mf][nf] + 0.006, np.full(nf.sum(), top), "^", color=GREY, ms=4,
               label="No voltage limit: no chatter up to 3 m/s")
        if (~nf).any():
            a.plot(b["frac"][mf][~nf], fmin[~nf], "D", color=GREY, ms=3.5, mew=0,
                   label="No voltage limit: smallest kick ending in chatter")
            for fr, v in zip(b["frac"][mf][~nf], fmin[~nf]):
                k = np.nonzero(np.isclose(f, fr))[0][0]
                shrink.append(v / lo[k])
        p = b["p_mode"][m]
        ok = np.isfinite(p) & fin
        A2 = np.interp(f[ok], d["frac"], d["A_dual"][j])
        dvdf = A2 / (g * p[ok]) * 1e3
        a.plot(f[ok], dvdf, "x-.", color=FACE, ms=4, lw=1.0, label="Describing-function estimate")
        ratios += list(med[ok] / dvdf)
        a.set(xlabel=r"$a\,/\,a_{\mathrm{lim}}$ (–)", xlim=(0.1, 1.02), yscale="log", ylim=(1, 1e4),
              xticks=(0.2, 0.4, 0.6, 0.8, 1.0))
        letter(a, "abcd"[i], f"{n} rpm", out=True)
    ax[0].set_ylabel(r"Critical kick $\Delta v^{*}$ (mm/s)")
    h, lab = [], []
    for a_ in ax:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    order = [lab.index(s) for s in (r"$|V|\leq9$ V: median over 16 phases",
                                    r"$|V|\leq9$ V: range over 16 phases (lower end: minimum)",
                                    r"$|V|\leq9$ V: no chatter up to 3 m/s",
                                    "No voltage limit: smallest kick ending in chatter",
                                    "No voltage limit: no chatter up to 3 m/s", "Describing-function estimate")]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="outside lower center", ncol=2)
    save(fig, "fig05_basin")
    stats = {}
    for n in SPEEDS:
        m = (b["rpm"] == n) & (b["vmax"] > 0)
        dv = b["dv"][m] * 1e3
        fr = b["frac"][m]
        k9, k98 = np.nonzero(np.isclose(fr, 0.9))[0][0], np.nonzero(np.isclose(fr, 0.98))[0][0]
        fin = np.all(np.isfinite(dv), axis=1)
        stats[str(n)] = dict(first_finite_frac=float(fr[fin].min()),
                             at_090_median_min=[r(np.median(dv[k9]), 1), r(dv[k9].min(), 2)],
                             at_098_median_min=[r(np.median(dv[k98]), 2), r(dv[k98].min(), 2)])
    NUM["fig05"] = dict(by_rpm=stats, td_median_over_df=[r(min(ratios)), r(max(ratios))],
                        clip_shrinks_basin_by=[r(min(shrink), 0), r(max(shrink), 0)],
                        free_points=16, free_points_with_chatter=len(shrink),
                        phase_spread_max=r(np.nanmax([np.max(x) / np.min(x) for x in b["dv"][(b["vmax"] > 0)]
                                                      if np.all(np.isfinite(x))]), 2))


# =============================================================== Fig. 6 ====
def fig06():
    """C2 and C3 certified kicks against the simulated critical kick (min over 16 phases)."""
    from zoa import a_lim
    v, c3b = load("c2_validate.npz"), load("c3_basin.npz")
    fac = np.array([c2_factor(n, g, fr * a_lim(n, g)) if c > 0 else 1.0
                    for n, g, fr, c in zip(v["rpm"], v["g"], v["frac"], v["cert"])])
    c2c = v["cert"] * fac * 1e3
    td = v["td_min"] * 1e3
    fig, ax = plt.subplots(1, 4, figsize=(W2, 66 * MM), sharey=True, layout="constrained")
    for i, n in enumerate(SPEEDS):
        a = ax[i]
        m = v["rpm"] == n
        f = v["frac"][m]
        k = np.nonzero(c3b["rpm"] == n)[0]
        k = k[np.argsort(c3b["frac"][k])]
        assert np.allclose(c3b["frac"][k], f)
        c3c = np.where(c3b["cert"][k] > 0, c3b["cert"][k], np.nan) * 1e3
        fin = np.isfinite(td[m])
        a.plot(f[fin], td[m][fin], "o-", color=K, ms=3.5, lw=1.0, label="Simulated critical kick (minimum over 16 phases)")
        if (~fin).any():
            top_marks(a, f[~fin], K, ms=4, label="Simulated: no chatter up to 3 m/s")
        a.plot(f, c2c[m], "s--", color=C2C, ms=3.8, mfc="white", lw=1.0, label="C2 certificate (averaged model)")
        a.plot(f, c3c, "D--", color=C3C, ms=3.6, mfc="white", lw=1.0, label="C3 certificate (periodic model)")
        a.set(xlabel=r"$a\,/\,a_{\mathrm{lim}}$ (–)", xlim=(0.1, 1.02), yscale="log", ylim=(0.5, 250),
              xticks=(0.2, 0.4, 0.6, 0.8, 1.0))
        letter(a, "abcd"[i], f"{n} rpm", out=True)
    ax[0].set_ylabel(r"Kick $\Delta v$ (mm/s)")
    h, lab = [], []
    for a_ in ax:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    save(fig, "fig06_certificates")
    ok2 = np.isfinite(td) & (c2c > 0)
    near = ok2 & (v["frac"] >= 0.7)
    c3all = c3b["cert"] * 1e3
    tdc3 = np.array([np.nanmin(x) if np.isfinite(x).any() else np.inf for x in c3b["td_kick"]]) * 1e3
    ok3 = np.isfinite(tdc3) & (c3all > 0)
    near3 = ok3 & (c3b["frac"] >= 0.7)
    q = c3all[c3all > 0] / (c3b["c2_cert"][c3all > 0] * 1e3)
    NUM["fig06"] = dict(
        c2_points=int(len(c2c)), c2_certified=int(np.sum(c2c > 0)),
        c2_below_td=[int(np.sum(c2c[ok2] <= td[ok2])), int(ok2.sum())],
        td_over_c2=rng(td[ok2] / c2c[ok2], 2), td_over_c2_median_from_07=r(np.median(td[near] / c2c[near])),
        c3_certified=[int(np.sum(c3all > 0)), int(len(c3all))],
        c3_not_certified=[f"{x:.0f} rpm {y:.2f}" for x, y in zip(c3b["rpm"][c3all <= 0], c3b["frac"][c3all <= 0])],
        c3_below_td=[int(np.sum(c3all[ok3] <= tdc3[ok3])), int(ok3.sum())],
        td_over_c3_from_07=rng(tdc3[near3] / c3all[near3], 2),
        c3_over_c2=rng(q, 3), c2_correction_factor=rng(fac[v["cert"] > 0], 4),
        td_at_098=[r(x) for x in td[np.isclose(v["frac"], 0.98)]])


# =============================================================== Fig. 7 ====
def fig07():
    """C2 certified kick over speed and depth; step-entry certificate beta at the four speeds."""
    from zoa import a_lim
    S, St, mp, d1 = load("c2_surface.npz"), load("c2_step.npz"), load("c1_map.npz"), load("feas_g1500.npz")
    sfac = surface_factor(S)
    cert = S["cert"] * sfac * 1e3                                    # (speed, depth), mm/s
    n, fr = S["rpm"], S["frac"]
    fig = plt.figure(figsize=(W2, 128 * MM), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.45, 1], hspace=0.02)
    a = sf[0].subplots(1, 1)
    edges = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    cols = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(edges)))
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(edges, cmap.N, extend="max")
    xe = np.concatenate([[n[0] - 30], 0.5 * (n[1:] + n[:-1]), [n[-1] + 30]])
    ye = np.concatenate([[fr[0] - 0.05], 0.5 * (fr[1:] + fr[:-1]), [fr[-1] + 0.01]])
    Z = np.ma.masked_less_equal(cert.T, 0.0)
    im = a.pcolormesh(xe, ye, Z, cmap=cmap, norm=norm, shading="flat", edgecolors="face", linewidth=0.3)
    #   (cell edges in the cell colour: no hairline seams between cells in PDF viewers)
    none = cert.T <= 0
    for (jj, ii) in zip(*np.nonzero(none)):
        a.add_patch(plt.Rectangle((xe[ii], ye[jj]), xe[ii + 1] - xe[ii], ye[jj + 1] - ye[jj], fc="white",
                                  ec=GREY, lw=0.8, hatch="////", zorder=3))
    ratio = mp["a_step_clip"] / mp["a_lim"]
    a.plot(mp["rpm"], ratio, color="white", lw=2.4, zorder=4)
    a.plot(mp["rpm"], ratio, color=STEP, lw=1.1, zorder=5, label="Simulated step-entry limit")
    a.set(xlabel="Spindle speed $n$ (rpm)", ylabel=r"$a\,/\,a_{\mathrm{lim}}$ (–)", xlim=(xe[0], xe[-1]),
          ylim=(ye[0], ye[-1]), yticks=fr)
    a.set_yticklabels([f"{x:g}" for x in fr])
    cb = sf[0].colorbar(im, ax=a, pad=0.01, aspect=18, ticks=edges, spacing="uniform", extend="max", extendfrac=0.12)
    cb.set_label("C2 certified kick (mm/s)")
    cb.ax.set_yticklabels([f"{x:g}" for x in edges])
    cb.ax.tick_params(direction="out", right=True, left=False)
    hatch = Patch(fc="white", ec=GREY, hatch="////", lw=0.8, label="Not certified")
    a.legend(handles=[a.get_legend_handles_labels()[0][0], hatch], loc="lower left", bbox_to_anchor=(0.0, 1.01),
             ncol=2, borderaxespad=0.0)
    letter(a, "a")
    ax = sf[1].subplots(1, 4, sharey=True)
    betas, bfac = [], []
    for i, nn in enumerate(St["rpm"]):
        g = float(np.interp(nn, d1["rpm"], d1["g_best"]))
        fb = np.array([c2_factor(nn, g, x * a_lim(nn, g)) for x in St["frac"]])
        beta = St["beta"][i] * fb
        betas.append(beta)
        bfac.append(fb)
        b = ax[i]
        b.axhspan(1.0, 10.0, color=C2C, alpha=0.12, lw=0)
        b.axhline(1.0, color=K, lw=0.8)
        j = int(np.argmin(np.abs(mp["rpm"] - nn)))
        b.axvline(mp["a_step_clip"][j] / mp["a_lim"][j], color=STEP, lw=1.0, ls="--",
                  label="Simulated step-entry limit")
        b.plot(St["frac"], beta, "o--", color=C2C, mfc="white", ms=3.8, lw=1.0, label=r"Step-entry certificate $\beta$ (C2)")
        b.set(xlabel=r"$a\,/\,a_{\mathrm{lim}}$ (–)", xlim=(0.25, 0.95), yscale="log", ylim=(0.1, 4),
              xticks=(0.3, 0.5, 0.7, 0.9))
        letter(b, "bcde"[i], f"{nn:.0f} rpm")
    ax[0].set_ylabel(r"$\beta$ (–)")
    ax[0].yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2]))
    ax[0].yaxis.set_major_formatter(ticker.FixedFormatter(["0.1", "0.2", "0.5", "1", "2"]))
    ax[0].yaxis.set_minor_formatter(ticker.NullFormatter())
    h, lab = ax[0].get_legend_handles_labels()
    h.append(Patch(color=C2C, alpha=0.25, lw=0))
    lab.append(r"$\beta\geq1$: entry certified")
    sf[1].legend(h, lab, loc="outside lower center", ncol=3)
    save(fig, "fig07_certified_surface")
    k9 = list(np.round(fr, 6)).index(0.9)
    betas = np.array(betas)
    NUM["fig07"] = dict(
        grid=[int(len(n)), int(len(fr))], certified=[int(np.sum(cert > 0)), int(cert.size)],
        not_certified=[f"{x:.0f} rpm {y:.2f}" for (x, y) in zip(np.repeat(n, len(fr)).reshape(cert.shape)[cert <= 0],
                                                                np.tile(fr, len(n)).reshape(cert.shape)[cert <= 0])],
        kick_090=rng(cert[:, k9], 2), kick_range=rng(cert[cert > 0], 2),
        beta_by_rpm={f"{x:.0f}": [r(y) for y in row] for x, row in zip(St["rpm"], betas)},
        beta_fracs=[float(x) for x in St["frac"]], beta_certified=int(np.sum(betas >= 1)),
        surface_factor=rng(sfac[S["cert"] > 0], 4), beta_factor=rng(np.array(bfac), 4),
        step_limit_ratio={f"{x:.0f}": r(float(mp["a_step_clip"][np.argmin(np.abs(mp["rpm"] - x))] /
                                              mp["a_lim"][np.argmin(np.abs(mp["rpm"] - x))]), 3) for x in St["rpm"]})


# =============================================================== Fig. 8 ====
def fig08():
    """Linear limit under the same authority rule (DVF vs DVF + D) and simulated entries (C4)."""
    L, E = load("c4_limits.npz"), load("c4_entry.npz")
    names, des, ent = list(L["feas_names"]), list(E["designs"]), list(E["names"])
    fig, ax = plt.subplots(3, 1, figsize=(W2, 140 * MM), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [2.0, 1.15, 0.75]})
    a = ax[0]
    a.plot(L["rpm"], L["feas_a_lim"][:, names.index("dvf")], color=K, lw=1.2, label="DVF, linear limit")
    for nm, col, ls in (("d1000", BLUE_L, "-"), ("d2000", BLUE, "-"), ("d4000", BLUE_D, "--")):
        g2 = float(L["feas_g2"][names.index(nm)])
        a.plot(L["rpm"], L["feas_a_lim"][:, names.index(nm)], ls, color=col, lw=1.2,
               label=rf"DVF + D, $g_2={g2:.0f}$, linear limit")
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_dvf"), ent.index("edge_clip")], "s", color=K, ms=3.5,
           label="DVF, face entry")
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_d2000"), ent.index("edge_clip")], "s", color=BLUE, ms=3.5,
           label=r"DVF + D, $g_2=2000$, face entry")
    a.set(ylabel="Axial depth of cut $a$ (mm)", ylim=(0, 85), xlim=(480, 3020))
    h, lab = a.get_legend_handles_labels()
    order = [0, 4, 1, 5, 2, 3]
    a.legend([h[k] for k in order], [lab[k] for k in order], loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
             borderaxespad=0.0)
    b = ax[1]
    b.plot(E["rpm"], E["a_entry"][:, des.index("feas_dvf"), ent.index("step_clip")], "^-", color=K, ms=3.8, lw=1.0,
           label="DVF, step entry")
    b.plot(E["rpm"], E["a_entry"][:, des.index("feas_d2000"), ent.index("step_clip")], "^-", color=BLUE, ms=3.8,
           lw=1.0, label=r"DVF + D, $g_2=2000$, step entry")
    b.set(ylabel="Axial depth of cut $a$ (mm)", ylim=(0, 40))
    b.legend(loc="upper right", ncol=1)
    c = ax[2]
    c.plot(L["rpm"], L["feas_g1"][:, names.index("dvf")], color=K, lw=1.2, label="DVF")
    c.plot(L["rpm"], L["feas_g1"][:, names.index("d2000")], "--", color=BLUE, lw=1.2, label=r"DVF + D, $g_2=2000$")
    c.set(xlabel="Spindle speed $n$ (rpm)", ylabel="Gain $g_1$ (V s/m)", ylim=(0, 3950), yticks=(0, 1000, 2000, 3000))
    c.legend(loc="lower right", ncol=1)
    letter(a, "a")
    letter(b, "b")
    letter(c, "c")
    save(fig, "fig08_delayed_limits")
    dvf = L["feas_a_lim"][:, names.index("dvf")]
    st = {nm: rng(L["feas_a_lim"][:, names.index(nm)] / dvf, 2) for nm in ("d1000", "d2000", "d4000")}
    face = E["a_entry"][:, des.index("feas_d2000"), ent.index("edge_clip")] / E["a_entry"][:, des.index("feas_dvf"), ent.index("edge_clip")]
    step = E["a_entry"][:, des.index("feas_d2000"), ent.index("step_clip")] / E["a_entry"][:, des.index("feas_dvf"), ent.index("step_clip")]
    NUM["fig08"] = dict(
        speeds=int(len(L["rpm"])), entry_speeds=int(len(E["rpm"])), g1_max=float(L["g1_max"]),
        dvf_authority_limited=int(np.sum(L["feas_g1"][:, names.index("dvf")] < float(L["g1_max"]) - 1e-6)),
        a_lim_over_dvf=st, dvf_a_lim_mm=rng(dvf, 2), d2000_a_lim_mm=rng(L["feas_a_lim"][:, names.index("d2000")], 2),
        face_d2000_over_dvf=rng(face, 2), step_d2000_over_dvf=rng(step, 2),
        step_dvf_mm=rng(E["a_entry"][:, des.index("feas_dvf"), ent.index("step_clip")], 2),
        step_d2000_mm=rng(E["a_entry"][:, des.index("feas_d2000"), ent.index("step_clip")], 2))


# =============================================================== Fig. 9 ====
def _td_series(V, V2, n, kind):
    def rows(D, kd):
        mm = (D["rpm"] == n) & (D["kind"] == kd)
        return D["depth"][mm], D["td_min"][mm] * 1e3
    if kind == "dvf":                                  # fine-grid best where simulated, else the first validation's
        d1, t1 = rows(V, "dvf")
        d2, t2 = rows(V2, "dvf_fine")
        keep = ~np.isin(d1, d2)
        d, t = np.concatenate([d1[keep], d2]), np.concatenate([t1[keep], t2])
    elif kind == "comb":
        d1, t1 = rows(V, "comb")
        d2, t2 = rows(V2, "comb")
        d, t = np.concatenate([d1, d2]), np.concatenate([t1, t2])
    else:
        d, t = rows(V2, "same_g1")
    o = np.argsort(d)
    return d[o], t[o]


def fig09():
    """Certified kick (best DVF, best DVF + D) and simulated critical kicks at the four speeds (C4)."""
    G, V, V2 = load("c4_gainmap.npz"), load("c4_validate.npz"), load("c4_validate2.npz")
    fd, fc = gainmap_factors(G)
    kick = G["kick"] * fc * 1e3
    kdvf = G["kick_dvf"] * fd * 1e3
    dep = G["depth"]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 72 * MM), sharey=True, layout="constrained")
    gap = {}
    for i, n in enumerate(G["rpm"]):
        a = ax[i]
        dbest = np.maximum(kick[i, :, :, 0].max(axis=1), kdvf[i].max(axis=1))
        cbest = kick[i, :, :, 1:].reshape(len(dep), -1).max(axis=1)
        a.plot(dep, np.where(dbest > 0, dbest, np.nan), "o--", color=C2C, mfc="white", ms=3.5, lw=1.0,
               label="Certificate, best DVF (26 gains)")
        a.plot(dep, np.where(cbest > 0, cbest, np.nan), "D--", color=BLUE, mfc="white", ms=3.4, lw=1.0,
               label=r"Certificate, best DVF + D (20 pairs $g_1, g_2$)")
        for kind, col, mk, lab in (("dvf", K, "o", "Simulated, best DVF"),
                                   ("comb", BLUE, "D", "Simulated, best DVF + D (map design)"),
                                   ("same", BLUE_L, "s", r"Simulated, DVF + D with the $g_1$ of the best DVF")):
            d, t = _td_series(V, V2, n, kind)
            fin = np.isfinite(t)
            a.plot(d[fin], t[fin], mk + "-", color=col, ms=3.6, lw=1.0, label=lab, zorder=4)
            if (~fin).any():
                off = {"dvf": -0.7, "comb": 0.0, "same": 0.7}[kind]
                top_marks(a, d[~fin] + off, col, ms=4, label="Simulated: no chatter up to 3 m/s" if kind == "dvf" else None)
        a.set(xlabel="Axial depth of cut $a$ (mm)", xlim=(0, 36), yscale="log", ylim=(0.3, 300), xticks=(0, 10, 20, 30))
        letter(a, "abcd"[i], f"{n:.0f} rpm", out=True)
        row = {}
        for D in (1.0, 2.0, 4.0):
            for lab, y in (("dvf", dbest), ("comb", cbest)):
                bad = np.where(y < D)[0]
                last = (bad[0] - 1) if bad.size else len(y) - 1
                row[f"{lab}_{D:g}"] = float(dep[last]) if last >= 0 else None
        gap[f"{n:.0f}"] = row
    ax[0].set_ylabel("Kick (mm/s)")
    h, lab = [], []
    for a_ in ax:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    order = [lab.index(s) for s in ("Simulated, best DVF", "Certificate, best DVF (26 gains)",
                                    "Simulated, best DVF + D (map design)",
                                    r"Certificate, best DVF + D (20 pairs $g_1, g_2$)",
                                    r"Simulated, DVF + D with the $g_1$ of the best DVF",
                                    "Simulated: no chatter up to 3 m/s")]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="outside lower center", ncol=3)
    save(fig, "fig09_delayed_basin")
    # validation designs: strict-solver certificates, corrected as in src/c3_summary.margin_errata
    import c2_certificate as c2
    import c3_periodic as c3
    allc, allt = [], []
    for X in (V, V2):
        for k in range(len(X["cert"])):
            c = X["cert"][k]
            if c > 0:
                n, a, g1 = X["rpm"][k], X["depth"][k] * 1e-3, X["g1"][k]
                c = c * (VMAX - c3.orbit(n, g1, a)[2].max()) / (VMAX - c2.orbit(n, g1, a)[2])
            allc.append(c * 1e3)
            allt.append(X["td_min"][k] * 1e3)
    allc, allt = np.array(allc), np.array(allt)
    ok = np.isfinite(allt) & (allc > 0)
    # simulated pairs at the same depth (src/c4_summary.py): DVF on the fine g1 grid where simulated, else coarse
    pairs = []
    for n in SPEEDS:
        dd, td_d = _td_series(V, V2, n, "dvf")
        dc, td_c = _td_series(V, V2, n, "comb")
        ds, td_s = _td_series(V, V2, n, "same")
        for dep_, t_d in zip(dd, td_d):
            cert_d = None
            m2 = (V2["rpm"] == n) & (V2["depth"] == dep_) & (V2["kind"] == "dvf_fine")
            m1 = (V["rpm"] == n) & (V["depth"] == dep_) & (V["kind"] == "dvf")
            cert_d = (V2["cert"][m2][0] if m2.any() else V["cert"][m1][0]) * 1e3
            t_c = td_c[dc == dep_][0] if np.any(dc == dep_) else np.nan
            t_s = td_s[ds == dep_][0] if np.any(ds == dep_) else np.nan
            pairs.append((n, dep_, t_d, t_c, t_s, cert_d))
    pr = np.array(pairs)
    fs = np.isfinite(pr[:, 2]) & np.isfinite(pr[:, 4]) & (pr[:, 2] > 0)
    fcb = np.isfinite(pr[:, 2]) & np.isfinite(pr[:, 3]) & (pr[:, 2] > 0)
    near = fs & (pr[:, 5] > 0) & (pr[:, 5] <= 2.1)
    NUM["fig09"] = dict(
        certified_without_gap_mm=gap,
        validation_points=int(len(allc)),
        validation_cert_below_td=[int(np.sum(allc[ok] <= allt[ok])), int(ok.sum())],
        validation_td_over_cert=rng(allt[ok] / allc[ok], 2),
        same_g1_over_dvf=rng(pr[fs, 4] / pr[fs, 2], 2), same_g1_better=[int(np.sum(pr[fs, 4] > pr[fs, 2])), int(fs.sum())],
        same_g1_where_dvf_cert_le_2p1=dict(pairs=[[float(x), float(y)] for x, y in pr[near, :2]],
                                           ratio=rng(pr[near, 4] / pr[near, 2], 2)),
        map_design_over_dvf=rng(pr[fcb, 3] / pr[fcb, 2], 2),
        map_design_better=[int(np.sum(pr[fcb, 3] > pr[fcb, 2])), int(fcb.sum())],
        gainmap_factor=dict(dvf=rng(fd[G["kick_dvf"] > 0], 4), comb=rng(fc[G["kick"] > 0], 4)))


# ============================================================== Fig. 10 ====
def fig10():
    """Robustness of the delayed term: comb-delay error, runout, loop delay with an assumed second mode."""
    R = load("c4_robust.npz")
    lab = list(R["labels"])
    designs = [("DVF", K, "o", "DVF"), ("tooth comb 1000", BLUE_L, "s", r"DVF + D, $g_2=1000$"),
               ("tooth comb", BLUE, "D", r"DVF + D, $g_2=2000$"), ("tooth comb 4000", BLUE_D, "^", r"DVF + D, $g_2=4000$")]
    spd = [(0, "o", "-"), (1, "s", "--"), (2, "^", "-."), (3, "D", ":")]
    fig, ax = plt.subplots(2, 3, figsize=(W2, 122 * MM), layout="constrained")
    j2 = lab.index("tooth comb")
    eps = R["eps"] * 100
    # (a) comb-delay error: limit relative to the exact comb; (d) worst orbit command over the design depths
    for i, mk, ls in spd:
        n = R["rpm"][i]
        ax[0, 0].plot(eps, R["eps_a_lim"][i, j2] / R["a_lim"][i, j2], mk + ls, color=BLUE, ms=3.2, lw=1.0,
                      label=f"{n:.0f} rpm")
        ax[1, 0].plot(eps, R["eps_cmd"][i, j2], mk + ls, color=BLUE, ms=3.2, lw=1.0)
    ax[0, 0].axhline(1.0, color=K, lw=0.8)
    ax[0, 0].axhline(0.9, color=K, lw=0.8, ls=":")
    ax[0, 0].text(-3.25, 0.895, r"$\kappa=0.9$", va="top", ha="left", fontsize=8)
    ax[0, 0].set(xlabel=r"Comb-delay error $T_c/T-1$ (%)", ylabel=r"$a_{\mathrm{lim}}(T_c)\,/\,a_{\mathrm{lim}}(T)$ (–)",
                 xlim=(-3.4, 3.4), ylim=(0.76, 1.42), xticks=np.arange(-3, 4))
    ax[0, 0].legend(loc="upper right", ncol=1, handlelength=2.8, title=r"$g_2=2000$", title_fontsize=8)
    ax[1, 0].axhline(VMAX, color=STEP, lw=0.9, ls="--")
    ax[1, 0].text(0.0, VMAX + 0.04, r"$V_{\max}=9$ V", color=STEP, va="bottom", ha="center", fontsize=8)
    ax[1, 0].axhline(8.1, color=K, lw=0.8, ls=":")
    ax[1, 0].text(-3.12, 8.07, "8.1 V", va="top", ha="left", fontsize=8)
    ax[1, 0].set(xlabel=r"Comb-delay error $T_c/T-1$ (%)", ylabel="Worst forced-orbit command (V)", xlim=(-3.4, 3.4),
                 ylim=(7.0, 10.2), xticks=np.arange(-3, 4))
    # (b) runout at 2500 rpm; (e) runout tolerated over the speeds
    i25 = int(np.argmin(np.abs(R["rpm"] - 2500)))
    tol = {}
    for key, col, mk, name in designs:
        jj = lab.index(key)
        ax[0, 1].plot(R["runout"] * 1e6, R["runout_cmd"][i25, jj], mk + "-", color=col, ms=3.2, lw=1.0, label=name)
        t = []
        for i in range(len(R["rpm"])):
            rc = R["runout_cmd"][i, jj]
            t.append(np.interp(VMAX, rc, R["runout"]) * 1e6 if rc[0] < VMAX <= rc[-1] else
                     (R["runout"][-1] * 1e6 if rc[-1] < VMAX else 0.0))
        tol[name] = t
        ax[1, 1].plot(R["rpm"], t, mk + "-", color=col, ms=3.4, lw=1.0)
    ax[0, 1].axhline(VMAX, color=STEP, lw=0.9, ls="--")
    ax[0, 1].text(5.1, VMAX - 0.12, r"$V_{\max}=9$ V", color=STEP, va="top", ha="right", fontsize=8)
    ax[0, 1].set(xlabel="Eccentric runout (µm)", ylabel="Worst forced-orbit command (V)", xlim=(-0.2, 5.2),
                 ylim=(7.5, 14.5))
    ax[1, 1].set(xlabel="Spindle speed $n$ (rpm)", ylabel="Runout tolerated (µm)", ylim=(0, 5.2),
                 xlim=(950, 3010), xticks=(1000, 1500, 2000, 2500, 3000))
    # (c) loop delay: worst case over the four speeds and the three assumed second modes (computed delays only)
    tau = R["tau"] * 1e6
    modes = [k for k in range(R["mode2"].shape[0]) if np.isfinite(R["mode2"][k, 0])]
    worst = {}
    for (key, col, mk, name), off in zip(designs, (-9.0, -3.0, 3.0, 9.0)):
        jj = lab.index(key)
        A = R["tau_a_lim"][:, jj][:, modes]                          # (rpm, mode, tau)
        w = (A / A[:, :, :1]).min(axis=(0, 1))
        worst[name] = w
        ax[0, 2].plot(tau + off, w, mk, color=col, ms=3.6, ls="none")
    ax[0, 2].axhline(0.5, color=K, lw=0.8, ls=":")
    ax[0, 2].set(xlabel=r"Loop delay $\tau$ (µs)", ylabel=r"Worst $a_{\mathrm{lim}}(\tau)\,/\,a_{\mathrm{lim}}(0)$ (–)",
                 xlim=(-25, 425), ylim=(-0.06, 1.22), xticks=(0, 100, 200, 300, 400))
    # (f) a_lim over the speeds for two delay / second-mode cases
    km = {float(R["mode2"][k, 0]): k for k in modes}
    it = {t: int(np.argmin(np.abs(R["tau"] - t))) for t in (200e-6, 300e-6)}
    for key, col, mk, name in designs[:3]:
        jj = lab.index(key)
        ax[1, 2].plot(R["rpm"], R["tau_a_lim"][:, jj, km[400.0], it[300e-6]], mk + "-", color=col, ms=3.4, lw=1.0)
        ax[1, 2].plot(R["rpm"], R["tau_a_lim"][:, jj, km[600.0], it[200e-6]], mk + "--", color=col, ms=3.4, lw=1.0)
    case = [Line2D([], [], color=K, lw=1.0, ls="-"), Line2D([], [], color=K, lw=1.0, ls="--")]
    ax[1, 2].legend(case, ["300 µs, 400 Hz mode", "200 µs, 600 Hz mode"], loc="upper right", ncol=1)
    ax[1, 2].set(xlabel="Spindle speed $n$ (rpm)", ylabel=r"$a_{\mathrm{lim}}$ (mm)", ylim=(0, 64), xlim=(950, 3010),
                 xticks=(1000, 1500, 2000, 2500, 3000))
    for a_, s, txt in zip(ax.ravel(), "abcdef", (None, "2500 rpm", None, None, None, None)):
        letter(a_, s, txt)
    hd = [Line2D([], [], color=col, marker=mk, ms=3.4, lw=1.0) for _, col, mk, _ in designs]
    fig.legend(hd, [d[3] for d in designs], loc="outside upper center", ncol=4)
    save(fig, "fig10_robustness")
    rel = R["eps_a_lim"][:, j2] / R["a_lim"][:, j2][:, None]
    ke = list(np.round(R["eps"], 4))
    k2p, k2m, k3p, k3m, k1p, k1m = (ke.index(x) for x in (0.02, -0.02, 0.03, -0.03, 0.01, -0.01))
    cmd = R["eps_cmd"][:, j2]
    thr = {}
    for name, w in worst.items():
        below = np.nonzero(w < 0.5)[0]
        thr[name] = float(tau[below[0] - 1]) if below.size else float(tau[-1])
    NUM["fig10"] = dict(
        designs="g1 the largest <= 3000 V s/m with the forced command <= 8.1 V up to 0.9 a_lim (10 % guard band); "
                "closed-form ZOA model (src/c4_robust.py)",
        g1_by_rpm={name: [r(x, 0) for x in R["g1"][:, lab.index(key)]] for key, _, _, name in designs},
        eps_plus2_rel=[r(rel[:, k2p].min()), r(rel[:, k2p].max())], eps_minus2_rel=[r(rel[:, k2m].min()), r(rel[:, k2m].max())],
        eps_pm1_rel=[r(rel[:, [k1m, k1p]].min()), r(rel[:, [k1m, k1p]].max())],
        eps_pm2_worst_cmd=[r(cmd[:, [k2m, k2p]].min()), r(cmd[:, [k2m, k2p]].max())],
        eps_pm3_cases_above_9V=[int(np.sum(cmd[:, [k3m, k3p]] > VMAX)), int(cmd[:, [k3m, k3p]].size)],
        runout_tolerated_um={k: [r(min(v)), r(max(v))] for k, v in tol.items()},
        loop_delay_largest_with_ratio_ge_05_us=thr,
        at_300us_400Hz={name: [r(x, 1) for x in R["tau_a_lim"][:, lab.index(key), km[400.0], it[300e-6]]]
                        for key, _, _, name in designs[:3]},
        at_200us_600Hz={name: [r(x, 1) for x in R["tau_a_lim"][:, lab.index(key), km[600.0], it[200e-6]]]
                        for key, _, _, name in designs[:3]},
        second_modes_f_kratio_zeta=[[float(x) for x in R["mode2"][k]] for k in modes],
        a_lim_nominal_mm={name: [r(x, 1) for x in R["a_lim"][:, lab.index(key)]] for key, _, _, name in designs})


FIGS = dict(fig02=fig02, fig03=fig03, fig04=fig04, fig05=fig05, fig06=fig06, fig07=fig07, fig08=fig08,
            fig09=fig09, fig10=fig10)


def main(argv):
    todo = argv or list(FIGS)
    path = os.path.join(HERE, "figure_numbers.json")
    if argv and os.path.exists(path):
        NUM.update(json.load(open(path)))
    for k in todo:
        FIGS[k]()
    with open(path, "w") as fh:
        json.dump(dict(sorted(NUM.items())), fh, indent=1)
    print("wrote figure_numbers.json")


if __name__ == "__main__":
    main(sys.argv[1:])
