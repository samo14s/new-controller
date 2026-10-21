"""C3 figures: the certified basin on the time-periodic model.

    python figures_c3.py      -> ../figures/c3_01_*.png ... c3_05_*.png
Needs ../results/c3_floquet.npz, c3_basin.npz, c3_lowimm.npz, c3_decomp.npz
(c3_periodic.py) and the C1/C2 results.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import c3_periodic as c3
from zoa import a_lim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OURS, GREY, GOOD, BAD, SOFT = "#1f5fa8", "#6c757d", "#2e7d32", "#c62828", "#e09f3e"
SPEEDS = (1090, 1420, 2500, 2870)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    return np.load(os.path.join(RES, name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def _fmin(x):
    x = np.asarray(x, float)
    f = x[np.isfinite(x)]
    return f.min() if f.size else np.nan


def c3_01_coefficient():
    """a H(t) over a tooth period, the command margin, and the Floquet limit against the ZOA."""
    F = load("c3_floquet.npz")
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4), gridspec_kw={"height_ratios": [1, 1]})
    rpm, g = 2500.0, 1500.0
    Tp = 60.0 / (rpm * P.N_TEETH)
    t = np.linspace(0, Tp, 1001)
    a = ax[0, 0]
    for frac, ls in ((0.15, ":"), (0.5, "--"), (0.9, "-")):
        dep = frac * a_lim(rpm, g)
        a.plot(t / Tp, c3.aH(t, rpm, dep) / (c3.HALF.kappa * dep), ls, color=OURS, lw=1.1,
               label=f"half immersion, {frac:.2f} a_lim ({dep*1e3:.1f} mm)")
    low = c3.Cut(c3.LOW_AE)
    dep = 0.9 * low.a_lim(rpm, g)
    a.plot(t / Tp, c3.aH(t, rpm, dep, low) / (low.kappa * dep), color=SOFT, lw=1.1,
           label=f"ae/D = {c3.LOW_AE}, 0.9 a_lim ({dep*1e3:.1f} mm)")
    a.set(xlabel="time / tooth period", ylabel="a H(t) / its mean", title="C3-1a  Periodic coefficient, 2500 rpm")
    a.legend(fontsize=6.5, loc="lower right")
    a = ax[0, 1]
    for frac, col in ((0.5, GREY), (0.9, OURS), (0.98, BAD)):
        dep = frac * a_lim(rpm, g)
        _, b, bk = c3.orbit(rpm, g, dep, n_t=512)
        a.plot(np.arange(512) / 512, np.abs(b), color=col, lw=1, label=f"|b(t)|, {frac:.2f} a_lim")
        a.step(np.arange(17) / 16, np.append(bk, bk[-1]), where="post", color=col, lw=0.6, ls="--")
    a.axhline(9.0, color="k", lw=0.8)
    a.set(xlabel="time / tooth period", ylabel="V", ylim=(0, 9.5),
          title="C3-1b  Forced command |b(t)|; dashed: C3's bound per subinterval")
    a.legend(fontsize=6.5, loc="lower right")
    a = ax[1, 0]
    for i, ae in enumerate(F["ae"]):
        a.plot(F["rpm"], F["a_floquet"][i] * 1e3, color=OURS if ae == 0.5 else SOFT, lw=1.1,
               label=f"periodic collocated model (Floquet), ae/D = {ae:g}")
        a.plot(F["rpm"], F["a_zoa"][i] * 1e3, color="k", lw=0.6, ls="--")
    a.plot([], [], color="k", lw=0.6, ls="--", label="ZOA (averaged)")
    a.set(xlabel="Spindle speed (rpm)", ylabel="a_lim (mm)", ylim=(0, None),
          title="C3-1c  Stability limit on C2's gain map")
    a.legend(fontsize=6.5)
    a = ax[1, 1]
    for i, ae in enumerate(F["ae"]):
        a.plot(F["rpm"], F["a_floquet"][i] / F["a_zoa"][i], color=OURS if ae == 0.5 else SOFT, lw=1,
               label=f"ae/D = {ae:g}")
    a.axhline(1.0, color="k", lw=0.6)
    a.set(xlabel="Spindle speed (rpm)", ylabel="Floquet / ZOA",
          title="C3-1d  Periodic limit / ZOA: never below 1")
    a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "c3_01_coefficient.png")


def _basin_panels(B, title, td_label, fname, phases_td=None):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
    for a, n in zip(ax, SPEEDS):
        m = B["rpm"] == n
        if not m.any():
            continue
        f = B["frac"][m]
        o = np.argsort(f)
        f = f[o]
        sel = lambda k: B[k][m][o]
        tdm = np.array([_fmin(r) for r in sel("td_kick")]) * 1e3
        a.semilogy(f, sel("c2_cert") * 1e3, "s--", color=GREY, ms=3.5, mfc="none", lw=0.9, label="C2 certificate (averaged)")
        a.semilogy(f, np.where(sel("cert") > 0, sel("cert"), np.nan) * 1e3, "o-", color=OURS, ms=3.5, lw=1.2,
                   label="C3 certificate (periodic)")
        a.semilogy(f, np.array([_fmin(r) for r in sel("model_kick")]) * 1e3, "-", color=GOOD, lw=1,
                   label="periodic model, critical kick")
        a.semilogy(f, np.array([_fmin(r) for r in sel("avg_kick")]) * 1e3, ":", color="k", lw=1,
                   label="averaged model, critical kick")
        a.semilogy(f, tdm, "^", color=BAD, ms=4, mfc="none", label=td_label)
        a.set(title=f"{n} rpm", xlabel="depth / a_lim")
    ax[0].set_ylabel("kick (mm/s), smallest over the phases")
    ax[0].legend(fontsize=6.3, loc="lower left")
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    save(fig, fname)


def c3_02_basin():
    B = load("c3_basin.npz")
    _basin_panels(B, "C3-2  Certified kick at the C1 basin points: C3 on the periodic model against C2, "
                     "the two models' critical kicks and the simulation", "simulated (C1)", "c3_02_basin.png")


def c3_03_phase():
    """Phase-resolved: certificate, models and simulation at the kick phases of C1."""
    B = load("c3_basin.npz")
    pts = [(2500, 0.4), (2500, 0.9), (1420, 0.95), (1090, 0.7)]
    fig, ax = plt.subplots(1, len(pts), figsize=(12, 3.6))
    for a, (n, f) in zip(ax, pts):
        i = np.nonzero((B["rpm"] == n) & np.isclose(B["frac"], f))[0]
        if not i.size:
            continue
        i = i[0]
        ph = np.arange(c3.N_PHASE) / c3.N_PHASE
        a.plot(ph, np.where(B["phase_cert"][i] > 0, B["phase_cert"][i], np.nan) * 1e3, "o-", color=OURS, ms=3, lw=1.1,
               label="C3 at the kick phase")
        a.axhline(B["c2_cert"][i] * 1e3, color=GREY, ls="--", lw=0.9, label="C2 (any phase)")
        a.plot(ph, B["model_kick"][i] * 1e3, "-", color=GOOD, lw=1, label="periodic model")
        a.plot(ph, B["avg_kick"][i] * 1e3, ":", color="k", lw=1, label="averaged model")
        a.plot(ph, B["td_kick"][i] * 1e3, "^", color=BAD, ms=3.5, mfc="none", label="simulated (C1)")
        a.set(title=f"{n} rpm, {f:.2f} a_lim", xlabel="kick phase / tooth period", ylim=(0, None))
    ax[0].set_ylabel("kick (mm/s)")
    ax[0].legend(fontsize=6.3)
    fig.suptitle("C3-3  Phase by phase: the any-phase certificate evaluated at each kick phase of C1, "
                 "the two models' critical kicks and the simulation", fontsize=9.5)
    fig.tight_layout()
    save(fig, "c3_03_phase.png")


def c3_04_decomp():
    D, B = load("c3_decomp.npz"), load("c3_basin.npz")
    variants = list(D["variants"])
    labels = ["C2"] + variants[:2] + ["periodic, resolved, m_c 16"] + variants[2:]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(D["rpm"]))
    w = 0.8 / len(labels)
    full = np.array([B["cert"][(B["rpm"] == r) & np.isclose(B["frac"], f)][0] for r, f in zip(D["rpm"], D["frac"])])
    vals = [D["c2_size"]] + [D["size"][:, 0], D["size"][:, 1], full] + [D["size"][:, j] for j in range(2, len(variants))]
    cols = [GREY, "#9ecae1", SOFT, OURS, "#c6dbef", "#08306b"]
    for j, (lab, v) in enumerate(zip(labels, vals)):
        ax.bar(x + (j - len(labels) / 2 + 0.5) * w, v / D["c2_size"], w, color=cols[j % len(cols)], label=lab)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{r:.0f}\n{f:.2f}" for r, f in zip(D["rpm"], D["frac"])], fontsize=7)
    ax.set(ylabel="certified kick / C2", title="C3-4  What each ingredient adds (rpm, depth / a_lim)")
    ax.legend(fontsize=6.5, ncol=3)
    save(fig, "c3_04_decomp.png")


def c3_05_lowimm():
    L = load("c3_lowimm.npz")
    _basin_panels(L, f"C3-5  Low radial immersion (ae/D = {float(L['ae']):g}): certificates, models and simulation",
                  "simulated (C1 protocol, 4 phases)", "c3_05_lowimm.png")


def main():
    for f, need in ((c3_01_coefficient, "c3_floquet.npz"), (c3_02_basin, "c3_basin.npz"), (c3_03_phase, "c3_basin.npz"),
                    (c3_04_decomp, "c3_decomp.npz"), (c3_05_lowimm, "c3_lowimm.npz")):
        if os.path.exists(os.path.join(RES, need)):
            f()


if __name__ == "__main__":
    main()
