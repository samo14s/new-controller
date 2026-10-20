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


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def c2_01_validation():
    v = load("c2_validate.npz")
    speeds = sorted(set(v["rpm"].tolist()))
    top = 2000.0
    fig, ax = plt.subplots(1, len(speeds), figsize=(11.5, 3.6), sharey=True)
    for i, n in enumerate(speeds):
        m = v["rpm"] == n
        f = v["frac"][m]
        for key, col, mk, lab in [("td_min", BAD, "o", "simulated cut (TD, min over 16 phases)"),
                                  ("model_kick", GREY, "s", "averaged model (min over 4 phases)"),
                                  ("cert", OURS, "D", "certificate (checked LMI)")]:
            y = v[key][m] * 1e3
            fin = np.isfinite(y) & (y > 0)
            ax[i].plot(f[fin], y[fin], mk + "-", color=col, ms=3.5, lw=1, label=lab)
            if (~fin).any() and key != "cert":
                ax[i].plot(f[~fin], np.full((~fin).sum(), top), "^", color=col, mfc="none", ms=5)
        ax[i].set(yscale="log", ylim=(0.5, 4000), xlim=(0.1, 1.0), title=f"{n:.0f} rpm",
                  xlabel="depth / a_lim(g)")
        ok = np.isfinite(v["td_min"][m]) & (v["cert"][m] > 0)
        r = v["td_min"][m][ok] / v["cert"][m][ok]
        ax[i].text(0.04, 0.04, f"TD / cert {r.min():.1f}-{r.max():.1f}", transform=ax[i].transAxes, fontsize=7.5)
    ax[0].set(ylabel="critical kick (mm/s)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    sound = np.sum(v["cert"][np.isfinite(v["td_min"])] <= v["td_min"][np.isfinite(v["td_min"])])
    fig.suptitle(f"C2-1  Certified kick vs the averaged model and the simulated cut "
                 f"(certificate below both at every point; proof check max V = {np.nanmax(v['proof_vmax']):.3f})",
                 fontsize=9, y=1.02)
    save(fig, "c2_01_validation.png")


def c2_02_step():
    s, m = load("c2_step.npz"), load("c1_map.npz")
    fig, ax = plt.subplots(1, len(s["rpm"]), figsize=(11.5, 3.2), sharey=True)
    for i, n in enumerate(s["rpm"]):
        j = int(np.argmin(np.abs(m["rpm"] - n)))
        beta = np.minimum(s["beta"][i], 1e3)
        ax[i].semilogy(s["frac"], beta, "D-", color=OURS, ms=4, lw=1, label="step-entry certificate beta")
        ax[i].axhline(1.0, color="k", lw=0.8)
        ax[i].fill_between([0, 1], 1, 1e3, color=GOOD, alpha=0.08, lw=0)
        ax[i].axvline(m["a_step_clip"][j] / m["a_lim"][j], color=BAD, lw=1, ls="--",
                      label="simulated step-entry limit (C1)")
        ax[i].set(title=f"{n:.0f} rpm", xlabel="depth / a_lim(g)", xlim=(0.25, 0.95), ylim=(0.1, 1e3))
    ax[0].set(ylabel="beta (>= 1: entry certified)")
    ax[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("C2-2  Certifying the hard entry without a clipped simulation: certified where beta >= 1 "
                 "(green), always left of the simulated limit", fontsize=9, y=1.03)
    save(fig, "c2_02_step_entry.png")


def c2_03_surface():
    d, mp, bd = load("c2_surface.npz"), load("c1_map.npz"), load("c1_band.npz")
    n, f, al = d["rpm"], d["frac"], d["a_lim"]
    cert = d["cert"] * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    im = ax[0].pcolormesh(n, f, np.log10(np.maximum(cert.T, 1e-2)), cmap="viridis", shading="nearest",
                          vmin=-0.5, vmax=1.5)
    cb = fig.colorbar(im, ax=ax[0], pad=0.01)
    cb.set_label("certified kick, log10 (mm/s)")
    ax[0].plot(mp["rpm"], mp["a_step_clip"] / mp["a_lim"], color=BAD, lw=1, label="simulated step-entry limit (C1)")
    ax[0].plot(bd["rpm"], bd["a_band_low"] / np.interp(bd["rpm"], mp["rpm"], mp["a_lim"]), ".", color=SOFT, ms=3,
               label="lower edge of the bistable band (C1)")
    ax[0].set(ylabel="depth / a_lim(g)", ylim=(0.45, 1.0),
              title="C2-3  Certified kick over speed and depth (averaged model, checked LMI)")
    ax[0].legend(fontsize=7, loc="lower right")
    # certified depth for a disturbance level D: largest grid depth whose certified kick >= D, all shallower ones too
    for D, col in [(1.0, GOOD), (2.0, OURS), (4.0, GREY)]:
        dep = np.full(len(n), np.nan)
        for k in range(len(n)):
            okk = cert[k] >= D
            if okk[0]:
                last = np.argmin(okk) - 1 if not okk.all() else len(f) - 1
                dep[k] = f[last] * al[k]
        ax[1].plot(n, dep, "o-", color=col, ms=2.5, lw=1, label=f"certified for kicks up to {D:g} mm/s")
    ax[1].plot(n, al, color="k", lw=1, label="a_lim(g)")
    ax[1].plot(mp["rpm"], mp["a_step_clip"], color=BAD, lw=1, ls="--", label="simulated step-entry limit")
    ax[1].set(xlabel="Spindle speed (rpm)", ylabel="depth (mm)", ylim=(0, None))
    ax[1].legend(fontsize=7, ncol=2, loc="upper left")
    save(fig, "c2_03_certified_surface.png")


if __name__ == "__main__":
    for fn in (c2_01_validation, c2_02_step, c2_03_surface):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, "(missing", os.path.basename(e.filename), ")")
