"""C1 figures: saturation-induced unsafe zones.

    python figures_c1.py      -> ../figures/c1_01_*.png ... c1_06_*.png
Needs ../results/c1_map.npz, c1_gainplane.npz, c1_basin.npz, c1_df.npz
(c1_unsafe_zone.py); c1_02 and c1_03 run their own simulations.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from tdsim import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OURS, PAPER, GREY, GOOD, BAD, SOFT = "#1f5fa8", "#d1495b", "#6c757d", "#2e7d32", "#c62828", "#e09f3e"
VMAX = P.F_MAX / P.GA_GAIN
SPR = P.STEPS_PER_REV
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    return np.load(os.path.join(RES, name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def c1_01_map():
    d = load("c1_map.npz")
    n = d["rpm"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a = ax[0]
    a.fill_between(n, d["a_step_clip"], d["a_step_free"], color=BAD, alpha=0.18, lw=0,
                   label="saturation-induced unsafe zone (step entry)")
    a.plot(n, d["a_lim"], color="k", lw=1.2, label="a_lim(g), linear (ZOA)")
    a.plot(n, d["a_step_free"], color=GREY, lw=0.9, ls="--", label="step entry, no limit (TD)")
    a.plot(n, d["a_step_clip"], color=BAD, lw=1.2, label="step entry, |V| <= 9 V (TD)")
    a.plot(n, d["a_edge_clip"], color=GOOD, lw=1.0, label="face entry, |V| <= 9 V (TD)")
    a.plot(n, d["a_ramp_clip"], color=SOFT, lw=0.9, ls="-.", label="depth ramp 10 rev, |V| <= 9 V")
    a.plot(n, d["a_feed_clip"], color=OURS, lw=0.9, ls=":", label="feed ramp 10 rev, |V| <= 9 V")
    a.plot(n, d["a_lim0"], color=GREY, lw=0.8, label="a_lim, no control")
    a.set(ylabel="depth limit (mm)", ylim=(0, None),
          title="C1-1  Depth limit by entry type, gain map of the reformulated condition")
    a.legend(fontsize=7, ncol=2, loc="upper left")
    b = ax[1]
    for k, col, ls in [("step_clip", BAD, "-"), ("edge_clip", GOOD, "-"), ("ramp_clip", SOFT, "-."),
                       ("feed_clip", OURS, ":")]:
        b.plot(n, d[f"a_{k}"] / d["a_step_free"], color=col, ls=ls, lw=1)
    gap = d["a_step_clip"] < 0.99 * d["a_step_free"]
    b.set(xlabel="Spindle speed (rpm)", ylabel="limit / unclipped", ylim=(0.5, 1.05))
    b.text(0.01, 0.06, f"step entry, clipped: >1 % below the unclipped limit at {gap.sum()}/{len(n)} speeds; "
           f"min ratio {np.nanmin(d['a_step_clip'] / d['a_step_free']):.2f}",
           transform=b.transAxes, fontsize=7.5)
    save(fig, "c1_01_unsafe_zone_map.png")


def _valid(r):
    """Number of stored samples before a run diverged (|x| > 1 cm)."""
    if not r["diverged"]:
        return len(r["x"])
    nz = np.nonzero(r["x"])[0]
    return int(nz[-1]) + 1


def _deviation(r, k):
    """Peak |x(t) - x(t - T_tooth)| per tooth period: distance from the
    tooth-periodic orbit, zero once the cut has settled."""
    nd = SPR // P.N_TEETH
    e = np.abs(r["x"][nd:k] - r["x"][:k - nd])
    m = len(e) // nd
    return (np.arange(m) + 1) / P.N_TEETH, e[: m * nd].reshape(m, nd).max(axis=1)


def c1_02_mechanism(rpm=2500, a=15e-3, g=1500.0):
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
    for j, (lab, kw) in enumerate([("step entry", {}), ("face entry", {"edge": True})]):
        r = simulate(rpm, a, g, vmax=VMAX, store=True, nrev=120, **kw)
        k = _valid(r)
        t = r["t"][:k] * rpm / 60
        col = BAD if r["diverged"] else GOOD
        ax[0, j].plot(t, r["x"][:k] * 1e6, color=col, lw=0.5)
        ax[1, j].plot(t, r["vcmd"][:k], color=col, lw=0.5)
        for s in (1, -1):
            ax[1, j].axhline(s * VMAX, color=GREY, lw=0.8, ls="--")
        ax[0, j].set(title=f"{lab}: {'chatter' if r['diverged'] else 'stable'}", ylim=(-120, 60))
        ax[1, j].set(xlabel="revolutions", ylim=(-10, 10), xlim=(0, 45))
        if r["diverged"]:
            ax[0, j].annotate(f"grows past 1 cm by rev {t[-1]:.0f}", (t[-1], 40), (15, 40), fontsize=7.5,
                              arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0, 0].set(ylabel="x (um)")
    ax[1, 0].set(ylabel="command V (V)")
    fig.suptitle(f"C1-2  Same cut, two outcomes: {rpm} rpm, {a*1e3:.0f} mm, g = {g:.0f} V s/m, "
                 f"|V| <= 9 V (a_lim = 17.2 mm)", fontsize=9)
    save(fig, "c1_02_same_cut_two_outcomes.png")


def c1_03_kicks(rpm=2500, a=15e-3, g=1500.0):
    ks = 100 * SPR + 37
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for j, (vmax, lab) in enumerate([(VMAX, "|V| <= 9 V"), (0.0, "no limit")]):
        for dv, col in [(3e-3, GOOD), (10e-3, SOFT), (30e-3, BAD)]:
            r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=180, edge=True, kick_step=ks, kick_dv=dv)
            t, e = _deviation(r, _valid(r))
            ax[j].semilogy(t, e * 1e6, color=col, lw=1, label=f"kick {dv*1e3:.0f} mm/s")
        ax[j].set(title=f"settled cut kicked at rev 100, {lab}", xlabel="revolutions", xlim=(95, 180),
                  ylim=(1e-6, 1e4))
    ax[0].set(ylabel="distance from the periodic orbit (um)")
    ax[1].legend(fontsize=7.5)
    fig.suptitle(f"C1-3  Bistability: the stable orbit and chatter coexist only with the limit "
                 f"({rpm} rpm, {a*1e3:.0f} mm)", fontsize=9, y=1.02)
    save(fig, "c1_03_bistability_kicks.png")


def c1_04_basin():
    b, d = load("c1_basin.npz"), load("c1_df.npz")
    speeds = sorted(set(b["rpm"].tolist()))
    top = 700.0                                    # where "no chatter up to 1 m/s" is drawn
    fig, ax = plt.subplots(1, len(speeds), figsize=(11, 3.4), sharey=True)
    for i, n in enumerate(speeds):
        j = int(np.argmin(np.abs(d["rpm"] - n)))
        g = d["g"][j]
        m = (b["rpm"] == n) & (b["vmax"] > 0)
        f, dv = b["frac"][m], b["dv"][m] * 1e3
        fin = np.all(np.isfinite(dv), axis=1)
        med, lo, hi = np.median(dv, axis=1), dv.min(axis=1), dv.max(axis=1)
        ax[i].fill_between(f[fin], lo[fin], hi[fin], color=BAD, alpha=0.2, lw=0)
        ax[i].plot(f[fin], med[fin], "o-", color=BAD, ms=3, lw=1, label="critical kick, |V| <= 9 V (TD, 16 phases)")
        if (~fin).any():
            ax[i].plot(f[~fin], np.full((~fin).sum(), top), "^", color=BAD, mfc="none", ms=5,
                       label="|V| <= 9 V: no chatter up to 1 m/s")
        mf = (b["rpm"] == n) & (b["vmax"] == 0)
        ax[i].plot(b["frac"][mf], np.full(mf.sum(), top), "^", color=GREY, ms=5,
                   label="no limit: no chatter up to 1 m/s")
        A2 = d["A_dual"][j] / g * 1e3
        ax[i].plot(d["frac"], d["A_single"][j] / g * 1e3, color=GREY, ls="--", lw=1, label="DF, single input")
        ax[i].plot(d["frac"], A2, color=OURS, lw=1, label="DF, with the forced command")
        r = med[fin] / np.interp(f[fin], d["frac"], A2)
        ax[i].text(0.04, 0.04, f"TD / DF: {r.min():.1f}-{r.max():.1f}", transform=ax[i].transAxes, fontsize=7.5)
        ax[i].set(yscale="log", ylim=(0.3, 1500), xlim=(0.45, 1.0), title=f"{n:.0f} rpm, g = {g:.0f}",
                  xlabel="depth / a_lim(g)")
    ax[0].set(ylabel="velocity amplitude (mm/s)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("C1-4  The basin of the stable cut shrinks to zero at a_lim; the describing function "
                 "is a conservative estimate of it", fontsize=9, y=1.03)
    save(fig, "c1_04_basin_vs_depth.png")


def c1_05_gainplane():
    d = load("c1_gainplane.npz")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for i, n in enumerate(d["rpm"]):
        g = d["gains"]
        sat = d["orbit_peak_V"][i] > VMAX
        if sat.any():
            ax[i].axvspan(g[sat].min() - 75, g.max() + 40, color=GREY, alpha=0.12, lw=0,
                          label="forced orbit itself > 9 V")
        ax[i].fill_between(g, d["a_step_clip"][i], d["a_lim"][i], color=BAD, alpha=0.15, lw=0,
                           label="unsafe zone (step entry)")
        ax[i].plot(g, d["a_lim"][i], color="k", lw=1.1, label="a_lim(g), linear")
        ax[i].plot(g, d["a_step_clip"][i], "o-", color=BAD, ms=2.5, lw=1.1, label="step entry, |V| <= 9 V")
        ax[i].plot(g, d["a_edge_clip"][i], "s-", color=GOOD, ms=2.5, lw=1.0, label="face entry, |V| <= 9 V")
        ax[i].axvline(d["g_map"][i], color=OURS, lw=0.8, ls=":", label="gain of the map")
        ax[i].set(title=f"{n:.0f} rpm", xlabel="g (V s/m)", ylim=(0, 45), xlim=(0, 3040))
    ax[0].set(ylabel="depth limit (mm)")
    ax[0].legend(fontsize=6.8, loc="upper left")
    fig.suptitle("C1-5  A hard entry caps the depth whatever the gain; a soft entry follows a_lim "
                 "until the forced orbit saturates", fontsize=9, y=1.03)
    save(fig, "c1_05_gain_plane.png")


def c1_06_df_prediction():
    d, m = load("c1_df.npz"), load("c1_map.npz")
    n = d["rpm"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(n, m["a_step_clip"], color=BAD, lw=1.2, label="step entry, |V| <= 9 V (TD)")
    ax[0].plot(n, d["a_pred_dual"], color=OURS, lw=1, label="DF prediction (entry transient vs threshold)")
    ax[0].plot(n, d["a_pred_single"], color=GREY, lw=0.9, ls="--", label="DF, single input")
    ax[0].plot(n, m["a_lim"], color="k", lw=0.8, label="a_lim(g)")
    ax[0].set(ylabel="depth (mm)", ylim=(0, None), title="C1-6  Predicting the step-entry limit without a clipped simulation")
    ax[0].legend(fontsize=7)
    for p, col, ls in [(d["a_pred_dual"], OURS, "-"), (d["a_pred_single"], GREY, "--")]:
        ax[1].plot(n, 100 * (p / m["a_step_clip"] - 1), color=col, ls=ls, lw=0.9)
    e = 100 * (d["a_pred_dual"] / m["a_step_clip"] - 1)
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="error (%)")
    ax[1].text(0.01, 0.9, f"dual-input: median {np.nanmedian(e):+.1f} %, within 10 % at "
               f"{np.sum(np.abs(e) < 10)}/{len(e)} speeds", transform=ax[1].transAxes, fontsize=7.5, va="top")
    save(fig, "c1_06_df_prediction.png")


if __name__ == "__main__":
    for f in (c1_01_map, c1_02_mechanism, c1_03_kicks, c1_04_basin, c1_05_gainplane, c1_06_df_prediction):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, "(missing", os.path.basename(e.filename), ")")
