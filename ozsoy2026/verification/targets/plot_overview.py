"""Own rendering of the digitised targets (not a copy of the paper's figures):
Fig. 9 curves + test points, Fig. 11 curves + test points, Fig. 8 gain map."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda n: list(csv.DictReader(open(os.path.join(HERE, n))))
c9, c11, g8, ex = rd("fig9_curves.csv"), rd("fig11_curves.csv"), rd("fig8_gain_map.csv"), rd("experimental_points.csv")
fig, ax = plt.subplots(3, 1, figsize=(9, 11))
for name, ls in (("controlled", "-"), ("uncontrolled", "--")):
    x = np.array([float(r["rpm"]) for r in c9 if r["curve"] == name])
    y = np.array([float(r["depth_mm"]) for r in c9 if r["curve"] == name])
    v = np.array([r["visible_in_print"] == "1" for r in c9 if r["curve"] == name])
    ax[0].plot(x, y, ls, color="C0", lw=1.2, label="Fig. 9 %s" % name)
    if name == "controlled":
        ax[0].plot(np.where(v, np.nan, x), np.where(v, np.nan, y), "-", color="C3", lw=2, label="hidden in print")
for name, ls in (("controlled", "-"), ("uncontrolled", "--")):
    x = [float(r["rpm"]) for r in c11 if r["curve"] == name]
    y = [float(r["depth_mm_centre"]) for r in c11 if r["curve"] == name]
    ax[1].plot(x, y, ".", ms=1.5, color="C0" if name == "controlled" else "C7", label="Fig. 11 %s (raster)" % name)
sty = {("none", "stable"): ("k+", None), ("none", "chatter"): ("kD", "none"),
       ("stable",): ("o",), }
for e in ex:
    a = ax[0] if e["figure"] == "Fig9" else ax[1]
    rpm, d = float(e["rpm"]), float(e["depth_mm"])
    if e["control"] == "none":
        a.plot(rpm, d, "k+" if e["state"] == "stable" else "kD", mfc="none", ms=7)
    else:
        a.plot(rpm, d, "o" if e["state"] == "stable" else "x", color="g" if e["state"] == "stable" else "r",
               mfc="none", ms=8, mew=1.5)
    if e["label"]:
        a.annotate(e["label"], (rpm, d), xytext=(-12, 0), textcoords="offset points", fontsize=8)
for a, t in zip(ax[:2], ("Fig. 9 (vector read-out): mapped gains", "Fig. 11 (raster read-out): single gain 237 V s/m")):
    a.set_xlim(350, 3100); a.set_ylim(0, 33); a.grid(alpha=.3); a.set_ylabel("depth of cut (mm)"); a.set_title(t)
    a.legend(fontsize=8, loc="upper right")
ax[2].plot([float(r["rpm"]) for r in g8], [float(r["gain_Vs_per_m"]) for r in g8], "-o", ms=2)
for rpm, g in ((500, 1487), (1300, 1162), (2500, 585), (3000, 797)):
    ax[2].plot(rpm, g, "s", mfc="none", color="C3")
ax[2].axhline(237, color="C1", ls=":", label="single gain 237 / text minimum")
ax[2].set_title("Fig. 8 gain map (vector read-out); squares = Fig. 6 legend gains")
ax[2].set_xlabel("spindle speed (rpm)"); ax[2].set_ylabel("g (V s/m)"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "targets_overview.png"), dpi=110)
