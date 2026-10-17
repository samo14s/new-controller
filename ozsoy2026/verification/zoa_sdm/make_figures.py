"""Figures: ZOA (src) vs SDM stability boundaries, and relative differences."""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
dz = json.load(open(os.path.join(HERE, "dense_sld_z0134.json")))
rp = np.array(dz["rpm"])
sd = list(csv.DictReader(open(os.path.join(HERE, "dense_sdm.csv"))))
cmp12 = list(csv.DictReader(open(os.path.join(HERE, "compare_12speeds.csv"))))

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
for col, g in enumerate((0, 1500)):
    ax, axr = axs[0, col], axs[1, col]
    zo = np.array(dz[str(g)]["a_src_mm"])
    ax.plot(rp, zo, "k-", lw=1.2, label="ZOA, src/zoa.py (1-rpm grid)")
    for mode, mk, c in (("straight", "o", "C3"), ("helix", "s", "C0")):
        rr = [r for r in sd if float(r["g"]) == g and r["mode"] == mode]
        n = np.array([float(r["rpm"]) for r in rr])
        a = np.array([float(r["a_mm"]) for r in rr])
        flip = np.array([r["type"] == "flip" for r in rr])
        lab = "SDM, straight flute" if mode == "straight" else "SDM, 45 deg helix"
        ax.plot(n[~flip], a[~flip], mk, ms=4, mfc="none", color=c, label=lab + " (Hopf)")
        if flip.any():
            ax.plot(n[flip], a[flip], mk, ms=5, color=c, label=lab + " (flip)")
        zi = np.interp(n, rp, zo)
        axr.plot(n, 100 * (zi / a - 1), mk + "-", ms=3, lw=0.7, color=c, mfc="none", label=lab)
    ax.set_ylabel("a_lim [mm]")
    ax.set_title(f"g = {g} V s/m, zeta = 1.34 %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    axr.axhline(0, color="k", lw=0.6)
    axr.set_ylabel("ZOA/SDM - 1 [%]")
    axr.set_xlabel("spindle speed [rpm]")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_zoa_vs_sdm.png"), dpi=130)

# 12-speed summary: relative difference vs depth
fig, ax = plt.subplots(figsize=(7, 4.5))
for key, mk, c, lab in (("rel_src_vs_sdm_meanH", "^", "k", "SDM, time-averaged force (= ZOA model)"),
                        ("rel_src_vs_sdm", "o", "C3", "SDM, straight flute"),
                        ("rel_src_vs_sdm_helix", "s", "C0", "SDM, 45 deg helix")):
    x = np.array([float(r["a_sdm_mm"]) for r in cmp12])
    y = np.array([100 * float(r[key]) for r in cmp12])
    ax.plot(x, y, mk, mfc="none", color=c, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel("SDM a_lim [mm] (straight flute)")
ax.set_ylabel("ZOA / SDM - 1 [%]")
ax.set_title("96 cases: 12 speeds x g in {0,300,800,1500} x zeta in {1.34,1.45} %")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_rel_diff_12speeds.png"), dpi=130)
print("figures written")
