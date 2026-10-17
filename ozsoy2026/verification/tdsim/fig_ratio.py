"""Figure check3_ratio.png: every boundary of check3 divided by zoa.a_lim."""
import json
import common
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(common.out("check3_stability.json")))
rows = d["cases"]
series = [("a_sd", "semi-discretisation (exact linear, helix)", "ks", dict(mfc="none", ms=8)),
          ("a_sd_nohelix", "semi-discretisation, helix 0", "k+", {}),
          ("a_zoh512", "ZOA + dt/2 hold delay (spr 512)", "C1--", {}),
          ("a_td512_n1000", "tdsim spr 512, D1", "C1o", {}),
          ("a_tdD2_512_n300", "tdsim spr 512, D2 (growth)", "C3x", {}),
          ("a_td2048_n1000", "tdsim spr 2048, D1", "C0^", {}),
          ("a_td8192_n300", "tdsim spr 8192, D1", "C2v", {})]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, g in zip(axs, (0.0, 600.0, 1200.0)):
    rr = sorted([r for r in rows if r["g"] == g], key=lambda r: r["rpm"])
    rpm = np.array([r["rpm"] for r in rr])
    for key, lab, sty, kw in series:
        y = np.array([np.nan if r[key] is None else r[key] / r["a_zoa"] for r in rr], float)
        ax.plot(rpm, y, sty, label=lab, **kw)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_title(f"g = {g:.0f} V s/m", fontsize=10)
    ax.set_xlabel("spindle speed (rpm)")
    ax.grid(alpha=0.3)
axs[0].set_ylabel("a_lim / a_lim(ZOA)")
axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(common.out("check3_ratio.png"), dpi=130)
