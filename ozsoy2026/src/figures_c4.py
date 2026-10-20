"""C4 figures: authority allocation with the orbit-invisible comb term.

    python figures_c4.py      -> ../figures/c4_01_*.png ... c4_04_*.png
Needs ../results/c4_limits.npz, c4_gainmap.npz, c4_entry.npz, c4_validate.npz
(c4_comb.py) and the C1 and C2 results.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OURS, GREY, GOOD, BAD, SOFT = "#1f5fa8", "#6c757d", "#2e7d32", "#c62828", "#e09f3e"
COMB = ["#000000", "#9ecae1", "#4292c6", "#08519c", "#08306b"]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    return np.load(os.path.join(RES, name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def c4_01_limits():
    L, E, M = load("c4_limits.npz"), load("c4_entry.npz"), load("c1_map.npz")
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    for j, g2 in enumerate(L["g2"]):
        lab = "DVF (paper gain), linear" if g2 == 0 else f"+ comb g2 = {g2:.0f} V s/m, linear"
        ax[0].plot(L["rpm"], L["a_lim"][:, j], color=COMB[j], lw=1.6 if g2 == 0 else 1.1, label=lab)
    sat = L["a_sat"]
    if np.isfinite(sat).any():
        ax[0].plot(L["rpm"], sat, ":", color=BAD, lw=1, label="forced orbit alone reaches 9 V")
    ax[0].plot(M["rpm"], M["a_edge_clip"], "-", color=SOFT, lw=1, label="simulated face entry, 9 V, DVF (C1)")
    names = list(E["names"])
    for j, (g2, mk) in enumerate(zip(E["g2"], ("s", "D"))):
        col = COMB[list(L["g2"]).index(g2)] if g2 in L["g2"] else OURS
        ax[0].plot(E["rpm"], E["a_entry"][:, j, names.index("edge_clip")], mk, color=col, ms=4, mfc="none",
                   label=f"simulated face entry, 9 V, comb g2 = {g2:.0f}")
    ax[0].set(ylabel="depth (mm)", ylim=(0, None),
              title="C4-1  The comb term raises the chatter-free depth without spending authority on the forced orbit")
    ax[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    ax[1].plot(M["rpm"], M["a_step_clip"], "-", color=BAD, lw=1, label="simulated step entry, 9 V, DVF (C1)")
    for j, (g2, mk) in enumerate(zip(E["g2"], ("s", "D"))):
        col = COMB[list(L["g2"]).index(g2)] if g2 in L["g2"] else OURS
        ax[1].plot(E["rpm"], E["a_entry"][:, j, names.index("step_clip")], mk + "-", color=col, ms=3.5, lw=0.8,
                   mfc="none", label=f"simulated step entry, 9 V, comb g2 = {g2:.0f}")
    ax[1].plot(L["rpm"], L["a_lim"][:, 0], color="k", lw=0.8, label="DVF linear limit")
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="depth (mm)", ylim=(0, None))
    ax[1].legend(fontsize=6.5, ncol=2, loc="upper left")
    save(fig, "c4_01_limits.png")


def c4_02_gainmap():
    G = load("c4_gainmap.npz")
    rpm, dep, g1s, g2s = G["rpm"], G["depth"], G["g1"], G["g2"]
    kick = G["kick"] * 1e3
    fig, ax = plt.subplots(2, len(rpm), figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    i_ref = int(np.argmin(np.abs(g1s - 1500.0)))
    for i, n in enumerate(rpm):
        k = kick[i].reshape(len(dep), -1)
        best = k.max(axis=1)
        arg = k.argmax(axis=1)
        p, q = np.unravel_index(arg, (len(g1s), len(g2s)))
        dvf_best = kick[i, :, :, 0].max(axis=1)
        ref = kick[i, :, i_ref, 0]
        a = ax[0, i]
        for y, col, mk, lab in [(ref, GREY, "s", "DVF g = 1500 (C2)"),
                                (dvf_best, SOFT, "^", "DVF, best g1 of the grid"),
                                (best, OURS, "D", "DVF + comb, best (g1, g2) of the grid")]:
            fin = y > 0
            a.plot(dep[fin], y[fin], mk + "-", color=col, ms=3.5, lw=1, label=lab)
        a.axvline(G["a_lim_ref"][i], color="k", lw=0.8, ls="--")
        a.set(yscale="log", ylim=(0.3, 300), title=f"{n:.0f} rpm")
        ax[1, i].plot(dep[best > 0], g1s[p][best > 0], "o-", color=SOFT, ms=3, lw=1, label="g1 (DVF)")
        ax[1, i].plot(dep[best > 0], g2s[q][best > 0], "s-", color=OURS, ms=3, lw=1, label="g2 (comb)")
        ax[1, i].axvline(G["a_lim_ref"][i], color="k", lw=0.8, ls="--")
        ax[1, i].set(xlabel="depth (mm)", ylim=(-200, 4400))
    ax[0, 0].set(ylabel="certified kick (mm/s)")
    ax[1, 0].set(ylabel="chosen gain (V s/m)")
    ax[0, 0].legend(fontsize=6.5, loc="lower left")
    ax[1, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("C4-2  Certified gain map: the gains that maximise the certified kick at each depth "
                 "(dashed: DVF linear limit at g = 1500)", fontsize=9, y=1.0)
    save(fig, "c4_02_gainmap.png")


def c4_03_validation():
    V = load("c4_validate.npz")
    speeds = sorted(set(V["rpm"].tolist()))
    fig, ax = plt.subplots(1, len(speeds), figsize=(12, 3.6), sharey=True)
    top = 2000.0
    for i, n in enumerate(speeds):
        b = (V["rpm"] == n) & (V["kind"] == "best")
        r = (V["rpm"] == n) & (V["kind"] == "dvf")
        for m, key, col, mk, lab in [(b, "td_min", BAD, "o", "simulated, best design (min over 16 phases)"),
                                     (r, "td_min", GREY, "s", "simulated, DVF g = 1500"),
                                     (b, "model_kick", SOFT, "^", "averaged model, best design")]:
            y = V[key][m] * 1e3
            d = V["depth"][m]
            fin = np.isfinite(y) & (y > 0)
            ax[i].plot(d[fin], y[fin], mk + "-", color=col, ms=3.5, lw=1, label=lab)
            if (~fin).any():
                ax[i].plot(d[~fin], np.full((~fin).sum(), top), mk, color=col, mfc="none", ms=5)
        c = V["cert"][V["rpm"][V["kind"] == "best"] == n] * 1e3      # cert holds the best-design points only
        ax[i].plot(V["depth"][b], c, "D-", color=OURS, ms=3.5, lw=1, label="certificate, best design")
        ax[i].set(yscale="log", ylim=(0.5, 4000), title=f"{n:.0f} rpm", xlabel="depth (mm)")
    ax[0].set(ylabel="critical kick (mm/s)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    off = f"off-axis boundary points chattering in the simulator: {int(V['offaxis_bad'].sum())}/{int(V['offaxis_runs'].sum())}"
    fig.suptitle(f"C4-3  Best certified design in the simulator (open markers at the top: no chatter up to 3 m/s); "
                 f"{off}", fontsize=8.5, y=1.02)
    save(fig, "c4_03_validation.png")


def c4_04_timeseries():
    """2500 rpm, 25 mm, face entry, |V| <= 9 V: DVF alone and DVF + comb."""
    import sys
    sys.path.insert(0, HERE)
    import params as P
    from tdsim import simulate
    rpm, a, g1, g2 = 2500.0, 25e-3, 1500.0, 2000.0
    nrev = 140
    runs = [("DVF g = 1500", dict()), (f"DVF g1 = 1500 + comb g2 = {g2:.0f}", dict(g2=g2))]
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    for j, (lab, kw) in enumerate(runs):
        r = simulate(rpm, a, g1, vmax=9.0, store=True, nrev=nrev, edge=True, nreg=4, **kw)
        t = r["t"] * rpm / 60.0
        ax[0, j].plot(t, r["x"] * 1e6, lw=0.4, color=OURS)
        ax[0, j].set(title=f"{lab}: {'diverged' if r['diverged'] else 'face entry'}", ylabel="x (um)")
        ax[1, j].plot(t, r["vcmd"], lw=0.4, color=SOFT)
        ax[1, j].axhline(9, color=BAD, lw=0.7, ls="--")
        ax[1, j].axhline(-9, color=BAD, lw=0.7, ls="--")
        ax[1, j].set(xlabel="revolutions", ylabel="command V (V)", ylim=(-10, 10))
    fig.suptitle(f"C4-4  {rpm:.0f} rpm, a = {a*1e3:.0f} mm (1.45 x the DVF limit 17.2 mm), face entry, |V| <= 9 V",
                 fontsize=9, y=1.0)
    save(fig, "c4_04_timeseries.png")


if __name__ == "__main__":
    for fn in (c4_01_limits, c4_02_gainmap, c4_03_validation, c4_04_timeseries):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, "(missing", os.path.basename(e.filename), ")")
