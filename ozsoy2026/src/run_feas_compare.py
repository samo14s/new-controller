"""What changes under the reformulated feasibility condition.

Uses the map of run_feasibility_map.py (tag g1500 by default) and recomputes,
with the same definitions as for the paper's condition:
  * the single fixed gain: with a feasible set that is one interval [0, g_b]
    at every speed, the joint feasible set is [0, min g_b];
  * Table 3: the new per-speed gain against the clipped baselines, both as
    time-domain limits with the 9 V command limit;
  * the 21 published test points under the new gains.
    python run_feas_compare.py --tag g1500
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool
import numpy as np
import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
POINTS = os.path.join(HERE, "..", "verification", "targets", "experimental_points.csv")
SPEEDS = [500, 560, 600, 800, 1300, 2300, 2500, 2800, 3000]
VMAX = P.F_MAX / P.GA_GAIN


def _td_limit(args):
    rpm, g, zeta = args
    P.set_damping(zeta)
    from tdboundary import boundary
    from zoa import a_lim
    return rpm, g, boundary(rpm, g, vmax=VMAX) * 1e3, a_lim(rpm, g) * 1e3


def _point(args):
    rpm, depth, g, zeta = args
    P.set_damping(zeta)
    from zoa import a_lim
    from tdboundary import classify
    al = a_lim(rpm, g) * 1e3
    c = classify(rpm, depth * 1e-3, g, vmax=VMAX if g > 0 else 0.0)
    return al, "stable" if depth < al else "chatter", "stable" if c["stable"] else "chatter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1500")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"feas_{a.tag}.npz"))
    zeta, kappa = float(d["zeta"]), float(d["kappa"])
    rpm, g = d["rpm"], d["g_best"]
    P.set_damping(zeta)
    from zoa import sld
    out = {}

    # single fixed gain
    g_single = float(g.min())
    a_single = sld(rpm, g_single) * 1e3
    out["single_gain"] = dict(g=g_single, binding_rpm=float(rpm[np.argmin(g)]),
                              mean_alim=float(a_single.mean()), min_alim=float(a_single.min()),
                              mean_alim_map=float(d["a_best"].mean()))

    # Table 3 with the new gains (clipped baselines are in table3_main.csv)
    jobs = [(n, float(np.interp(n, rpm, g)), zeta) for n in SPEEDS]
    with Pool(a.workers) as pool:
        t3 = pool.map(_td_limit, jobs)
    old = np.genfromtxt(os.path.join(RES, "table3_main.csv"), delimiter=",", names=True)
    cert = np.load(os.path.join(RES, f"feas_cert_{a.tag}.npz"))
    rows = []
    for (n, gn, td, zo), o in zip(t3, old):
        ac = float(np.interp(n, cert["rpm"], cert["a_cert"]))
        rows.append(dict(rpm=n, g_new=gn, a_new_td=td, a_new_zoa=zo, a_new_operating=kappa * zo,
                         a_new_certified=ac,
                         a_paper_criterion_td=float(o["a_prop_td"]),
                         a_clip1500_td=float(o["a_clip1500_td"]),
                         a_clip3000_td=float(o["a_clip3000_td"]),
                         paper_prop=float(o["paper_prop"]), paper_clip=float(o["paper_clip"])))
    out["table3"] = rows

    # test points
    g8 = None
    pts, jobs = [], []
    with open(POINTS) as fh:
        for r in csv.DictReader(fh):
            n, depth, ctrl = float(r["rpm"]), float(r["depth_mm"]), r["control"]
            if ctrl.startswith("mapped"):
                gg = float(np.interp(n, rpm, g))
            elif ctrl.startswith("single"):
                gg = g_single
            else:
                gg = 0.0
            pts.append(dict(figure=r["figure"], label=r["label"], rpm=n, depth_mm=depth,
                            controller=ctrl, measured=r["state"], g=gg))
            jobs.append((n, depth, gg, zeta))
    with Pool(a.workers) as pool:
        res = pool.map(_point, jobs)
    for p, (al, pz, pt) in zip(pts, res):
        p.update(alim=al, pred_zoa=pz, pred_td=pt)
    out["points"] = pts
    out["agreement"] = dict(zoa=sum(p["pred_zoa"] == p["measured"] for p in pts),
                            td=sum(p["pred_td"] == p["measured"] for p in pts), n=len(pts))

    with open(os.path.join(RES, f"feas_compare_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    s = out["single_gain"]
    print(f"single gain {s['g']:.1f} V s/m (binding {s['binding_rpm']:.0f} rpm), mean a_lim "
          f"{s['mean_alim']:.2f} mm vs map {s['mean_alim_map']:.2f} mm, min {s['min_alim']:.2f}")
    print(" rpm | g_new  new TD  new ZOA  cert | paper-crit TD | clip1500 clip3000 | paper prop clip")
    for r in rows:
        print(f"{r['rpm']:4d} | {r['g_new']:6.0f} {r['a_new_td']:6.2f} {r['a_new_zoa']:7.2f} "
              f"{r['a_new_certified']:6.2f} | {r['a_paper_criterion_td']:12.2f} | "
              f"{r['a_clip1500_td']:8.2f} {r['a_clip3000_td']:8.2f} | {r['paper_prop']:9.1f} {r['paper_clip']:5.1f}")
    print(f"test points: ZOA {out['agreement']['zoa']}/{len(pts)}, TD {out['agreement']['td']}/{len(pts)}")
    for p in pts:
        if p["pred_td"] != p["measured"] or p["pred_zoa"] != p["measured"]:
            print(f"  miss: {p['figure']} {p['rpm']:.0f} rpm {p['depth_mm']} mm {p['controller']} "
                  f"g={p['g']:.0f} a_lim={p['alim']:.2f} measured {p['measured']} "
                  f"zoa {p['pred_zoa']} td {p['pred_td']}")


if __name__ == "__main__":
    main()
