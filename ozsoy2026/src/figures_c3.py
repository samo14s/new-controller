"""C3 figures: the certified basin on the time-periodic model.

    python figures_c3.py      -> ../figures/c3_01_*.png ... c3_05_*.png
Needs ../results/c3_floquet.npz, c3_basin.npz, c3_lowimm.npz, c3_decomp.npz,
c3_edge.npz, c3_straight.npz, c3_counter.npz (c3_periodic.py).
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
    """What each ingredient adds (0.9 a_lim), the edge (0.98 a_lim), and low immersion."""
    D, B = load("c3_decomp.npz"), load("c3_basin.npz")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.6, 1, 1.4]})
    a = ax[0]
    variants = [str(v) for v in D["variants"]]
    full = np.array([B["cert"][(B["rpm"] == r) & np.isclose(B["frac"], f)][0] for r, f in zip(D["rpm"], D["frac"])])
    cols = {"averaged, resolved": "#9ecae1", "periodic, single level": SOFT, "periodic, resolved, m_c 8": "#c6dbef",
            "periodic, resolved, m_c 32": "#08306b"}
    col = {v: D["size"][:, j] for j, v in enumerate(variants)}
    bars = [(v, col[v]) for v in variants if "m_c" not in v] + [("periodic, resolved, m_c 8", col["periodic, resolved, m_c 8"]),
                                                                  ("periodic, resolved, m_c 16 (C3)", full),
                                                                  ("periodic, resolved, m_c 32", col["periodic, resolved, m_c 32"])]
    x = np.arange(len(D["rpm"]))
    w = 0.8 / len(bars)
    for j, (lab, v) in enumerate(bars):
        a.bar(x + (j - len(bars) / 2 + 0.5) * w, v / D["c2_size"], w, color=cols.get(lab, OURS), label=lab)
    a.axhline(1.0, color="k", lw=0.6)
    a.set_xticks(x, [f"{r:.0f} rpm" for r in D["rpm"]])
    a.set(ylabel="certified kick / C2", ylim=(0.8, 1.27), title=f"C3-4a  Ingredients at {float(D['frac'][0]):.2f} a_lim")
    a.legend(fontsize=6, loc="lower left")
    a = ax[1]
    if os.path.exists(os.path.join(RES, "c3_edge.npz")):
        G = load("c3_edge.npz")
        x = np.arange(len(G["rpm"]))
        for j, mc in enumerate(G["m_c"]):
            a.plot(x + (j - 0.5) * 0.18, np.where(G["size"][:, j] > 0, G["size"][:, j], np.nan) * 1e3, "o", ms=6,
                   color=OURS if mc == 16 else "#08306b", label=f"C3, {mc} subintervals")
        a.plot(x, G["c2_size"] * 1e3, "s", ms=6, mfc="none", color=GREY, label="C2")
        for i in np.nonzero(np.all(G["size"] <= 0, axis=1))[0]:
            a.annotate("not certified", (x[i], G["c2_size"][i] * 1e3), textcoords="offset points", xytext=(0, -16),
                       ha="center", fontsize=6.5, color=BAD)
        a.set_xticks(x, [f"{r:.0f} rpm" for r in G["rpm"]])
        a.set(ylabel="certified kick (mm/s)", ylim=(0, None), title=f"C3-4b  At {float(G['frac']):.2f} a_lim")
        a.legend(fontsize=6.5)
    a = ax[2]
    L = load("c3_lowimm.npz")
    o = np.lexsort((L["frac"], L["rpm"]))
    x = np.arange(len(o))
    a.semilogy(x, L["c2_cert"][o] * 1e3, "s", color=GREY, mfc="none", label="C2 (averaged)")
    a.semilogy(x, np.where(L["cert"][o] > 0, L["cert"][o], np.nan) * 1e3, "o", color=OURS, label="C3 (periodic)")
    a.semilogy(x, np.array([_fmin(r) for r in L["model_kick"][o]]) * 1e3, "_", ms=12, mew=2, color=GOOD,
               label="periodic model, critical kick")
    a.semilogy(x, np.array([_fmin(r) for r in L["avg_kick"][o]]) * 1e3, "_", ms=8, mew=1, color="k",
               label="averaged model, critical kick")
    a.semilogy(x, np.array([_fmin(r) for r in L["td_kick"][o]]) * 1e3, "^", color=BAD, mfc="none", label="simulated")
    a.set_xticks(x, [f"{r:.0f}\n{f:.1f}" for r, f in zip(L["rpm"][o], L["frac"][o])], fontsize=7)
    a.set(ylabel="kick (mm/s), smallest over the phases",
          title=f"C3-4c  ae/D = {float(L['ae']):g} (rpm, depth / a_lim)")
    a.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    save(fig, "c3_04_decomp.png")


def c3_05_averaging():
    """Where the averaged model is wrong: straight teeth."""
    S, C = load("c3_straight.npz"), load("c3_counter.npz")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    labs = list(S["gain_labels"])
    style = {"0": ("k", "g = 0 (no control)"), "100": (SOFT, "g = 100"), "250": (GOOD, "g = 250"),
             "500": ("#9ecae1", "g = 500"), "map": (OURS, "C2's gain map")}
    a = ax[0]
    for i, lab in enumerate(labs):
        f = np.where(S["first_unstable"][i] < 1, S["first_unstable"][i], np.nan)
        a.plot(S["rpm"], f, "o-", ms=3, lw=1, color=style[lab][0], label=style[lab][1])
    a.axhline(1.0, color="k", lw=0.6)
    a.set(xlabel="Spindle speed (rpm)", ylabel="first unstable depth / ZOA limit", ylim=(0.9, 1.01),
          title="C3-5a  Straight teeth: flip lobes below the ZOA limit")
    a.legend(fontsize=6.5, loc="lower left")
    a = ax[1]
    for i, lab in enumerate(labs):
        if lab == "0":
            continue
        a.semilogy(S["rpm"], S["bmax_098"][i], "-", lw=1, color=style[lab][0], label=style[lab][1])
        un = S["first_unstable"][i] < 1
        a.semilogy(S["rpm"][un], S["bmax_098"][i][un], "o", ms=5, color=style[lab][0])
    a.axhline(9.0, color=BAD, lw=1, label="9 V")
    a.set(xlabel="Spindle speed (rpm)", ylabel="forced command at 0.98 a_lim (V)",
          title="C3-5b  ... where the forced orbit alone needs > 9 V (dots)")
    a.legend(fontsize=6.5)
    a = ax[2]
    kt = int(C["kick_tooth"])
    for i, lab in enumerate(C["labels"]):
        xt = C["x_tooth"][i]
        dev = np.diff(xt)[kt:]                                   # x - x(t - T) once per tooth
        a.semilogy(np.arange(len(dev)) / P.N_TEETH, np.abs(dev) * 1e6 + 1e-9, lw=0.8,
                   color=BAD if lab == "straight" else OURS,
                   label=f"{lab}: C2's method {C['c2_size'][i]*1e3:.1f} mm/s, C3 "
                         + (f"{C['c3_size'][i]*1e3:.1f} mm/s" if C["c3_size"][i] > 0 else "none (|mu| > 1)"))
    a.set(xlabel="revolutions after a 0.1 mm/s kick", ylabel="|x(t) - x(t - T)| at the teeth (um)",
          title=f"C3-5c  {float(C['rpm']):.0f} rpm, no control, {float(C['frac']):.2f} a_lim")
    a.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout()
    save(fig, "c3_05_averaging.png")


def main():
    for f, need in ((c3_01_coefficient, "c3_floquet.npz"), (c3_02_basin, "c3_basin.npz"), (c3_03_phase, "c3_basin.npz"),
                    (c3_04_decomp, "c3_decomp.npz"), (c3_05_averaging, "c3_counter.npz")):
        if os.path.exists(os.path.join(RES, need)):
            f()


if __name__ == "__main__":
    main()
