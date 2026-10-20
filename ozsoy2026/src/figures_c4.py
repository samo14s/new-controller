"""C4 figures: actuator authority and the certified basin with a delayed-difference term.

    python figures_c4.py      -> ../figures/c4_01_*.png ... c4_05_*.png
Needs ../results/c4_limits.npz, c4_gainmap.npz, c4_entry.npz, c4_validate.npz,
c4_robust.npz (c4_comb.py, c4_robust.py) and the C1 results.
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
BLUES = {"d1000": "#6baed6", "d2000": "#2171b5", "d4000": "#08306b"}
PCOL = {"p3e5": "#fdae6b", "p1e6": "#d94801"}
VMAX = 9.0
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    return np.load(os.path.join(RES, name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def c4_01_limits():
    L, E, M = load("c4_limits.npz"), load("c4_entry.npz"), load("c1_map.npz")
    names = list(L["feas_names"])
    des, ent = list(E["designs"]), list(E["names"])
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 0.6]})
    a = ax[0]
    a.plot(L["rpm"], L["feas_a_lim"][:, names.index("dvf")], color="k", lw=1.6, label="DVF, best feasible g <= 3000")
    for n in ("d1000", "d2000", "d4000"):
        g2 = float(L["feas_g2"][names.index(n)])
        a.plot(L["rpm"], L["feas_a_lim"][:, names.index(n)], color=BLUES[n], lw=1.1,
               label=f"+ D term g2 = {g2:.0f} V s/m (feasible g1)")
    for n in ("p3e5", "p1e6"):
        kd = float(L["feas_kd"][names.index(n)])
        a.plot(L["rpm"], L["feas_a_lim"][:, names.index(n)], "--", color=PCOL[n], lw=1,
               label=f"+ P term kd = {kd:.0e} V/m (feasible g1)")
    a.plot(L["rpm"], L["a_lim"][:, 0], color=GREY, lw=0.8, ls=":", label="DVF at the paper gain (C1, C2)")
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_dvf"), ent.index("edge_clip")], "s", color="k", ms=3.5,
           mfc="none", label="simulated face entry, 9 V: DVF")
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_d2000"), ent.index("edge_clip")], "D", color=BLUES["d2000"],
           ms=3.5, mfc="none", label="simulated face entry, 9 V: g2 = 2000")
    a.set(ylabel="depth (mm)", ylim=(0, None),
          title="C4-1  Linear limit under the same authority rule (forced demand <= 9 V up to 0.9 a_lim), "
                "and simulated entries")
    a.legend(fontsize=6.3, ncol=2, loc="upper left")
    a = ax[1]
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_dvf"), ent.index("step_clip")], "s-", color="k", ms=3,
           lw=0.8, mfc="none", label="simulated step entry, 9 V: DVF (best feasible g)")
    a.plot(E["rpm"], E["a_entry"][:, des.index("feas_d2000"), ent.index("step_clip")], "D-", color=BLUES["d2000"],
           ms=3, lw=0.8, mfc="none", label="simulated step entry, 9 V: + g2 = 2000")
    a.plot(M["rpm"], M["a_step_clip"], color=GREY, lw=0.8, label="simulated step entry, 9 V: DVF at the paper gain (C1)")
    a.set(ylabel="depth (mm)", ylim=(0, None))
    a.legend(fontsize=6.5, loc="upper right")
    a = ax[2]
    a.plot(L["rpm"], L["feas_g1"][:, names.index("dvf")], color="k", lw=1, label="feasible g1, DVF")
    a.plot(L["rpm"], L["feas_g1"][:, names.index("d2000")], color=BLUES["d2000"], lw=1, ls="--",
           label="feasible g1 with g2 = 2000")
    a.plot(L["rpm"], L["g1"], color=GREY, lw=0.8, ls=":", label="paper gain")
    a.set(xlabel="Spindle speed (rpm)", ylabel="g1 (V s/m)", ylim=(0, 3300))
    a.legend(fontsize=6.5, ncol=3, loc="lower left")
    save(fig, "c4_01_limits.png")


def c4_02_gainmap():
    G = load("c4_gainmap.npz")
    rpm, dep, g1s, g2s = G["rpm"], G["depth"], G["g1"], G["g2"]
    kick = G["kick"] * 1e3
    i_ref = int(np.argmin(np.abs(g1s - 1500.0)))
    fig, ax = plt.subplots(2, len(rpm), figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for i, n in enumerate(rpm):
        dvf = kick[i, :, :, 0].max(axis=1)
        comb = kick[i, :, :, 1:].reshape(len(dep), -1).max(axis=1)
        ref = kick[i, :, i_ref, 0]
        a = ax[0, i]
        for y, col, mk, lab in [(ref, GREY, "s", "DVF g = 1500 (C2)"), (dvf, "k", "^", "DVF, best g1 of the grid"),
                                (comb, OURS, "D", "DVF + D term, best (g1, g2 > 0)")]:
            a.plot(dep, np.where(y > 0, y, np.nan), mk + "-", color=col, ms=3.5, lw=1, label=lab)   # gaps: nothing certified
        a.axvline(G["a_lim_ref"][i], color="k", lw=0.7, ls="--")
        a.set(yscale="log", ylim=(0.3, 100), title=f"{n:.0f} rpm")
        k = kick[i].reshape(len(dep), -1)
        p, q = np.unravel_index(k.argmax(axis=1), (len(g1s), len(g2s)))
        ok = k.max(axis=1) > 0
        ax[1, i].plot(dep[ok], g1s[p][ok], "o-", color="k", ms=3, lw=1, label="g1 of the best design")
        ax[1, i].plot(dep[ok], g2s[q][ok], "s-", color=OURS, ms=3, lw=1, label="g2 of the best design")
        ax[1, i].axvline(G["a_lim_ref"][i], color="k", lw=0.7, ls="--")
        ax[1, i].set(xlabel="depth (mm)", ylim=(-200, 4400))
    ax[0, 0].set(ylabel="certified kick (mm/s)")
    ax[1, 0].set(ylabel="gain (V s/m)")
    ax[0, 0].legend(fontsize=6.5, loc="lower left")
    ax[1, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("C4-2  Certified gain map: gains that maximise the certified kick at each depth "
                 f"(g1 in {{{', '.join(f'{g:.0f}' for g in g1s)}}}, g2 in {{{', '.join(f'{g:.0f}' for g in g2s)}}}; "
                 "dashed: DVF limit at g = 1500)", fontsize=8.5, y=1.0)
    save(fig, "c4_02_gainmap.png")


def c4_03_validation():
    V = load("c4_validate.npz")
    speeds = sorted(set(V["rpm"].tolist()))
    top = 2000.0
    fig, ax = plt.subplots(1, len(speeds), figsize=(12, 3.8), sharey=True)
    for i, n in enumerate(speeds):
        for kind, col, lab in (("dvf", "k", "best DVF"), ("comb", OURS, "best DVF + D term")):
            m = (V["rpm"] == n) & (V["kind"] == kind)
            d = V["depth"][m]
            t, u = V["td_min"][m] * 1e3, V["unsettled"][m]
            fin = np.isfinite(t)
            ax[i].plot(d[fin], t[fin], "o-", color=col, ms=4, lw=1.2, label=f"simulated, {lab} (min over 16 phases)")
            inf = np.isinf(t)
            ax[i].plot(d[inf], np.full(inf.sum(), top), "^", color=col, mfc="none", ms=5)
            ax[i].plot(d[u], np.full(u.sum(), 0.7), "x", color=col, ms=6)
            ax[i].plot(d, V["cert"][m] * 1e3, "D--", color=col, ms=3, lw=0.8, mfc="none", label=f"certificate, {lab}")
            mk = V["model_kick"][m] * 1e3
            ax[i].plot(d, mk, ":", color=col, lw=0.8, label=f"averaged model, {lab}")
        ax[i].set(yscale="log", ylim=(0.5, 4000), title=f"{n:.0f} rpm", xlabel="depth (mm)")
    ax[0].set(ylabel="critical kick (mm/s)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    m = V["kind"] == "comb"
    fig.suptitle("C4-3  Best designs of the gain map in the simulator (triangles at the top: no chatter up to 3 m/s; "
                 f"x: the unkicked cut chatters); off-axis boundary points chattering: "
                 f"{int(V['offaxis_bad'][m].sum())}/{int(V['offaxis_runs'][m].sum())}", fontsize=8.3, y=1.02)
    save(fig, "c4_03_validation.png")


def c4_04_robust():
    R = load("c4_robust.npz")
    labels = list(R["labels"])
    cols = {"DVF": "k", "tooth comb": BLUES["d2000"], "spindle comb": GOOD}
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.6))
    j = labels.index("tooth comb")
    for i, n in enumerate(R["rpm"]):
        c = plt.cm.viridis(i / max(1, len(R["rpm"]) - 1))
        ax[0, 0].plot(R["eps"] * 100, R["eps_a_lim"][i, j] / R["a_lim"][i, labels.index("DVF")], "o-", color=c, ms=3,
                      lw=1, label=f"{n:.0f} rpm")
        ax[1, 0].plot(R["eps"] * 100, R["eps_cmd"][i, j], "o-", color=c, ms=3, lw=1, label=f"{n:.0f} rpm")
    ax[0, 0].axhline(1, color="k", lw=0.7)
    ax[0, 0].set(xlabel="comb-delay error Tc / T - 1 (%)", ylabel="a_lim / a_lim(DVF)",
                 title="tooth comb (g2 = 2000): delay error")
    ax[1, 0].axhline(VMAX, color=BAD, lw=0.8, ls="--")
    ax[1, 0].set(xlabel="comb-delay error Tc / T - 1 (%)", ylabel="worst orbit command (V)", ylim=(0, 12))
    ax[0, 0].legend(fontsize=6.5)
    for lab in labels:
        jj = labels.index(lab)
        for i, n in enumerate(R["rpm"]):
            ax[0, 1].plot(R["runout"] * 1e6, R["runout_cmd"][i, jj], "o-", color=cols[lab], ms=2.5, lw=0.8,
                          alpha=0.4 + 0.6 * i / max(1, len(R["rpm"]) - 1), label=lab if i == len(R["rpm"]) - 1 else None)
    ax[0, 1].axhline(VMAX, color=BAD, lw=0.8, ls="--")
    ax[0, 1].set(xlabel="eccentric runout (um)", ylabel="worst orbit command (V)", ylim=(0, 16),
                 title="runout (darker: higher speed)")
    ax[0, 1].legend(fontsize=6.5)
    for lab in labels:
        jj = labels.index(lab)
        tol = []
        for i in range(len(R["rpm"])):
            rc = R["runout_cmd"][i, jj]
            tol.append(np.interp(VMAX, rc, R["runout"]) * 1e6 if (rc[0] < VMAX <= rc[-1]) else
                       (R["runout"][-1] * 1e6 if rc[-1] < VMAX else 0.0))
        ax[1, 1].plot(R["rpm"], tol, "o-", color=cols[lab], ms=4, lw=1, label=lab)
    ax[1, 1].set(xlabel="Spindle speed (rpm)", ylabel="runout tolerated (um)",
                 title=f"largest runout with orbit <= 9 V (capped at {R['runout'][-1] * 1e6:.0f} um)")
    ax[1, 1].legend(fontsize=6.5)
    i25 = int(np.argmin(np.abs(R["rpm"] - 2500)))
    for lab in labels:
        jj = labels.index(lab)
        for k, (ls, mlab) in enumerate(zip(("-", "--", ":"), ("one mode", "+ 400 Hz mode", "+ 800 Hz mode"))):
            ax[0, 2].plot(R["tau"] * 1e6, R["tau_a_lim"][i25, jj, k], ls, color=cols[lab], lw=1.1,
                          label=f"{lab}, {mlab}")
    ax[0, 2].set(xlabel="loop delay (us)", ylabel="a_lim (mm)", ylim=(0, 50),
                 title=f"loop delay, {R['rpm'][i25]:.0f} rpm (spindle comb partly off scale)")
    ax[0, 2].legend(fontsize=5.8)
    it = [int(np.argmin(np.abs(R["tau"] - t))) for t in (200e-6, 300e-6)]
    for lab, mk in (("DVF", "s"), ("tooth comb", "D")):
        jj = labels.index(lab)
        ax[1, 2].plot(R["rpm"], R["tau_a_lim"][:, jj, 0, 0], mk + ":", color=cols[lab], ms=3, lw=0.8,
                      label=f"{lab}: ideal loop")
        for k, ls in zip(it, ("-", "--")):
            ax[1, 2].plot(R["rpm"], R["tau_a_lim"][:, jj, 1, k], mk + ls, color=cols[lab], ms=4, lw=1,
                          label=f"{lab}: tau = {R['tau'][k] * 1e6:.0f} us, + 400 Hz mode")
    ax[1, 2].set(xlabel="Spindle speed (rpm)", ylabel="a_lim (mm)", ylim=(0, None),
                 title="loop delay with an unmodelled 400 Hz mode (0: unstable)")
    ax[1, 2].legend(fontsize=5.8)
    fig.suptitle("C4-4  Robustness of the delayed design (closed form, ZOA; feasible g1 at each speed; "
                 "checked against the simulator, see README 13)", fontsize=9, y=1.0)
    fig.tight_layout()
    save(fig, "c4_04_robustness.png")


def c4_05_timeseries():
    """2500 rpm, 29 mm, face entry, |V| <= 9 V: DVF at its best feasible gain,
    and the same g1 with g2 = 2000."""
    import sys
    sys.path.insert(0, HERE)
    from tdsim import simulate
    L = load("c4_limits.npz")
    names = list(L["feas_names"])
    rpm, a, g2 = 2500.0, 29e-3, 2000.0
    g1 = float(np.interp(rpm, L["rpm"], L["feas_g1"][:, names.index("dvf")]))
    nrev = 160
    runs = [(f"DVF g = {g1:.0f}", dict()), (f"DVF g1 = {g1:.0f} + D term g2 = {g2:.0f}", dict(g2=g2))]
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    for j, (lab, kw) in enumerate(runs):
        r = simulate(rpm, a, g1, vmax=VMAX, store=True, nrev=nrev, edge=True, nreg=4, **kw)
        t = r["t"] * rpm / 60.0
        end = int(np.flatnonzero(r["x"])[-1]) + 1 if r["diverged"] else len(t)      # the run stops at |x| = 1 cm
        ax[0, j].plot(t[:end], r["x"][:end] * 1e6, lw=0.4, color=OURS)
        ax[0, j].set(title=lab + (": chatter, run stopped at |x| = 1 cm" if r["diverged"] else ": settles"),
                     ylabel="x (um)", xlim=(0, nrev))
        ax[1, j].plot(t[:end], r["vcmd"][:end], lw=0.4, color=SOFT)
        for s in (1, -1):
            ax[1, j].axhline(s * VMAX, color=BAD, lw=0.7, ls="--")
        ax[1, j].set(xlabel="revolutions", ylabel="command V (V)", ylim=(-10, 10))
    fig.suptitle(f"C4-5  {rpm:.0f} rpm, a = {a*1e3:.0f} mm, face entry, |V| <= 9 V (DVF linear limit at this gain "
                 f"{float(np.interp(rpm, L['rpm'], L['feas_a_lim'][:, names.index('dvf')])):.1f} mm)", fontsize=9, y=1.0)
    save(fig, "c4_05_timeseries.png")


if __name__ == "__main__":
    for fn in (c4_01_limits, c4_02_gainmap, c4_03_validation, c4_04_robust, c4_05_timeseries):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, "(missing", os.path.basename(e.filename), ")")
