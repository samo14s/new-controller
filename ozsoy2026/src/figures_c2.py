"""C2 figures: the certified basin.

    python figures_c2.py      -> ../figures/c2_01_*.png ... c2_03_*.png
Needs ../results/c2_validate.npz, c2_step.npz, c2_surface.npz
(c2_certificate.py) and the C1 results.
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
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    return np.load(os.path.join(RES, name))


def bound_factor(n, g, a):
    """Factor that rescales a C2 certificate (size or beta) from C2's sampled
    command bound to the closed-form bound of C3 (README 14.9):
    (9 V - b_C3) / (9 V - b_C2)."""
    import params as P
    import c2_certificate as c2
    import c3_periodic as c3
    vmax = P.F_MAX / P.GA_GAIN
    return (vmax - c3.orbit(n, g, a)[2].max()) / (vmax - c2.orbit(n, g, a)[2])


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def c2_01_validation():
    from zoa import a_lim
    v = dict(load("c2_validate.npz"))
    v["cert"] = np.array([c * bound_factor(n, g, fr * a_lim(n, g)) if c > 0 else c
                          for n, g, fr, c in zip(v["rpm"], v["g"], v["frac"], v["cert"])])
    speeds = sorted(set(v["rpm"].tolist()))
    top = 2000.0
    fig, ax = plt.subplots(1, len(speeds), figsize=(11.5, 3.6), sharey=True)
    for i, n in enumerate(speeds):
        m = v["rpm"] == n
        f = v["frac"][m]
        for key, col, mk, lab in [("td_min", BAD, "o", "simulated cut (TD, min over 16 phases)"),
                                  ("model_kick", GREY, "s", "averaged model (min over 4 phases)"),
                                  ("cert", OURS, "D", "certificate (checked LMI, command bound of 14.9)")]:
            y = v[key][m] * 1e3
            fin = np.isfinite(y) & (y > 0)
            ax[i].plot(f[fin], y[fin], mk + "-", color=col, ms=3.5, lw=1, label=lab)
            if (~fin).any() and key != "cert":
                ax[i].plot(f[~fin], np.full((~fin).sum(), top), "^", color=col, mfc="none", ms=5)
        ax[i].set(yscale="log", ylim=(0.5, 4000), xlim=(0.1, 1.0), title=f"{n:.0f} rpm",
                  xlabel="depth / a_lim(g)")
        ok = np.isfinite(v["td_min"][m]) & (v["cert"][m] > 0)
        r = v["td_min"][m][ok] / v["cert"][m][ok]
        ax[i].text(0.96, 0.62, f"TD / cert {r.min():.1f}-{r.max():.1f}", transform=ax[i].transAxes, fontsize=7.5,
                   ha="right", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    ax[0].set(ylabel="critical kick (mm/s)")
    ax[0].plot([], [], "^", color="0.3", mfc="none", ms=5, label="no chatter for any kick up to 3 m/s (drawn at 2 m/s)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    off = ""
    if "offaxis_bad" in v:
        off = (f"; off-axis boundary points in the simulator: {int(v['offaxis_bad'].sum())}/"
               f"{int(v['offaxis_runs'].sum())} chatter")
    fig.suptitle(f"C2-1  Certified kick vs the averaged model and the simulated cut (certificate below both at every "
                 f"point; sampled check max V = {np.nanmax(v['proof_vmax']):.3f}{off})", fontsize=8.5, y=1.02)
    save(fig, "c2_01_validation.png")


def c2_02_step():
    from zoa import a_lim
    s, m = load("c2_step.npz"), load("c1_map.npz")
    d1 = load("feas_g1500.npz")
    fig, ax = plt.subplots(1, len(s["rpm"]), figsize=(11.5, 3.2), sharey=True)
    for i, n in enumerate(s["rpm"]):
        j = int(np.argmin(np.abs(m["rpm"] - n)))
        g = float(np.interp(n, d1["rpm"], d1["g_best"]))
        f = np.array([bound_factor(n, g, fr * a_lim(n, g)) for fr in s["frac"]])
        beta = np.minimum(s["beta"][i] * f, 10.0)
        ax[i].semilogy(s["frac"], beta, "D-", color=OURS, ms=4, lw=1, label="step-entry certificate beta")
        ax[i].axhline(1.0, color="k", lw=0.8)
        ax[i].fill_between([0, 1], 1, 10, color=GOOD, alpha=0.08, lw=0)
        ax[i].axvline(m["a_step_clip"][j] / m["a_lim"][j], color=BAD, lw=1, ls="--",
                      label="simulated step-entry limit (C1)")
        ax[i].set(title=f"{n:.0f} rpm", xlabel="depth / a_lim(g)", xlim=(0.25, 0.95), ylim=(0.1, 10))
    ax[0].set(ylabel="beta (>= 1: entry certified)")
    ax[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("C2-2  Hard entry at phase 0 (tooth angle 0 at t = 0) certified where beta >= 1 (green): on the grid "
                 "only at 0.3 a_lim with the chip-validity bound; never past the simulated limit", fontsize=9, y=1.03)
    save(fig, "c2_02_step_entry.png")


def c2_03_surface():
    d, mp, bd = load("c2_surface.npz"), load("c1_map.npz"), load("c1_band.npz")
    n, f, al = d["rpm"], d["frac"], d["a_lim"]
    cert = d["cert"] * 1e3
    for k in range(len(n)):
        for j in range(len(f)):
            if cert[k, j] > 0:
                cert[k, j] *= bound_factor(n[k], d["g"][k], f[j] * al[k] * 1e-3)
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.8")
    im = ax[0].pcolormesh(n, f, np.ma.masked_less_equal(np.log10(np.maximum(cert.T, 1e-12)), -12), cmap=cmap,
                          shading="nearest", vmin=-0.5, vmax=1.5)
    cb = fig.colorbar(im, ax=ax[0], pad=0.01, extend="both")
    cb.set_label("certified kick, log10 (mm/s); grey: none")
    ax[0].plot(mp["rpm"], mp["a_step_clip"] / mp["a_lim"], color=BAD, lw=1, label="simulated step-entry limit (C1)")
    ax[0].plot(bd["rpm"], bd["a_band_low"] / np.interp(bd["rpm"], mp["rpm"], mp["a_lim"]), ".", color=SOFT, ms=3,
               label="lower edge of the bistable band (C1)")
    ax[0].set(ylabel="depth / a_lim(g)", ylim=(0.45, 1.0),
              title="C2-3  Certified kick over speed and depth (averaged model, checked LMI)")
    ax[0].legend(fontsize=7, loc="lower right")
    # certified depth for a disturbance level D: largest grid depth whose certified kick >= D, all shallower ones too
    for D, col, ymark in [(1.0, GOOD, 2.4), (2.0, OURS, 1.4), (4.0, GREY, 0.4)]:
        dep = np.full(len(n), np.nan)
        for k in range(len(n)):
            okk = cert[k] >= D
            if okk[0]:
                last = np.argmin(okk) - 1 if not okk.all() else len(f) - 1
                dep[k] = f[last] * al[k]
        ax[1].plot(n, dep, "o-", color=col, ms=2.5, lw=1, label=f"certified for kicks up to {D:g} mm/s")
        # speeds where already the shallowest grid depth (0.5 a_lim) is not certified for D
        miss = np.isnan(dep)
        if miss.any():
            ax[1].plot(n[miss], np.full(miss.sum(), ymark), "v", color=col, ms=4, clip_on=False,
                       label=f"< 0.5 a_lim for {D:g} mm/s (marker at the bottom)")
    ax[1].plot(n, al, color="k", lw=1, label="a_lim(g)")
    ax[1].plot(mp["rpm"], mp["a_step_clip"], color=BAD, lw=1, ls="--", label="simulated step-entry limit")
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="depth (mm): every grid depth up to it certified",
              ylim=(0, None))
    ax[1].legend(fontsize=7, ncol=2, loc="upper left")
    save(fig, "c2_03_certified_surface.png")


if __name__ == "__main__":
    for fn in (c2_01_validation, c2_02_step, c2_03_surface):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, "(missing", os.path.basename(e.filename), ")")
