"""(c, hypothesis H2) Is the Fig. 9 controlled depth the force-limited depth?

H2 (a hypothesis, not a claim about the authors' method): the plotted depth is
min(a_lim(g8), a_F), where a_F is the depth at which the time-domain peak
demand at the Fig. 8 gain reaches 27 N.  Below a_lim the forced response is
linear in depth (checked: D(0.5 mm)/D(1 mm) is reported), so
a_F = 27 N * 1 mm / D(1 mm).  Three demand definitions from src/objective.py.

Output: c2_hypothesis_force_limited.csv / .json
"""
import numpy as np

from common import fig8, fig9_controlled, save_csv, save_json
import params as P
import zoa
from objective import demand


def main():
    P.set_damping(0.0134)
    x8, y8 = fig8()
    r9, a9 = fig9_controlled()
    rpm = np.round(r9 / 10) * 10
    g8 = np.interp(r9, x8, y8)
    rows = []
    for r, g, ap in zip(rpm, g8, a9):
        d = {k: demand(r, 1e-3, g, k) for k in ("force", "volt", "actual")}
        lin = demand(r, 0.5e-3, g, "force") / d["force"]
        al = zoa.a_lim(r, g) * 1e3
        rows.append((r, g, ap, al, 27 / d["force"], 27 / d["volt"], 27 / d["actual"], lin))
    A = np.array(rows)
    save_csv("c2_hypothesis_force_limited.csv",
             ["rpm", "g_fig8", "a_plot_mm", "a_zoa_mm", "aF_force_3g_xdot_mm", "aF_volt_g_xdot_mm",
              "aF_actual_fact_mm", "linearity_D05_over_D1"], rows)
    summ = dict(note="hypothesis test", linearity_range=[A[:, 7].min(), A[:, 7].max()])
    for j, n in ((4, "force"), (5, "volt"), (6, "actual")):
        h = np.minimum(A[:, 3], A[:, j])
        rel = A[:, 2] / h - 1
        band = (A[:, 0] >= 1880) & (A[:, 0] <= 2100)
        summ[n] = dict(median_abs_rel_pct=np.median(np.abs(rel)) * 100,
                       within_5pct=int((np.abs(rel) < 0.05).sum()), n=len(rel),
                       median_abs_rel_pct_1880_2100=np.median(np.abs(rel[band])) * 100)
    save_json("c2_hypothesis_force_limited.json", summ)
    print(summ)


if __name__ == "__main__":
    main()
