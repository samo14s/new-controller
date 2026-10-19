"""Spindle-speed gain map over 500-3000 rpm (Fig. 8) and the matching SLD (Fig. 9).

    python run_gain_map.py --tag main --zeta 0.0134 --kind force --sade
    python run_gain_map.py --tag volt --kind volt            (reference only)
    python run_gain_map.py --tag zeta145 --zeta 0.0145       (reference only)
    python run_gain_map.py --tag hold512 --fmode 0           (reference only)
Writes ../results/gainmap_<tag>.npz and .csv
"""
import argparse
import os
import time
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def work(args):
    rpm, zeta, kind, do_sade, fmode = args
    P.set_damping(zeta)
    import tdsim
    tdsim.FMODE = fmode
    from gainmap import reference, run_sade
    from zoa import a_lim
    t0 = time.time()
    ref = reference(rpm, kind)
    row = dict(rpm=rpm, g_ref=ref["g"], a_ref=ref["a"], a_unc=a_lim(rpm, 0.0) * 1e3,
               scan_f=ref["scan_f"], scan_alim=ref["scan_alim"], scan_fd=ref["scan_fd"],
               scan_g=ref["scan_g"])
    if do_sade:
        runs = run_sade(rpm, kind)
        best = max(runs, key=lambda r: r[1])
        row.update(g_sade=best[0], a_sade=best[1],
                   sade_runs=np.array(runs, dtype=float))
    row["seconds"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zeta", type=float, default=P.ZETA_PAPER_CURVES)
    ap.add_argument("--kind", default="force")
    ap.add_argument("--sade", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fmode", type=int, default=1,
                    help="1: force re-evaluated per RK4 stage; 0: held per step")
    ap.add_argument("--rpm", type=float, nargs="*")
    a = ap.parse_args()
    rpms = a.rpm if a.rpm else P.RPM_GRID
    jobs = [(float(n), a.zeta, a.kind, a.sade, a.fmode) for n in rpms]
    rows = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(work, jobs)):
            rows.append(r)
            msg = f"[{i+1}/{len(jobs)}] {r['rpm']:.0f} rpm  ref g={r['g_ref']:.1f} a={r['a_ref']:.2f}"
            if a.sade:
                msg += f"  sade g={r['g_sade']:.1f} a={r['a_sade']:.2f}"
            print(msg + f"  ({r['seconds']:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["rpm"])
    keys = ["rpm", "g_ref", "a_ref", "a_unc"] + (["g_sade", "a_sade"] if a.sade else [])
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data["scan_g"] = rows[0]["scan_g"]
    for k in ("scan_f", "scan_alim", "scan_fd"):
        data[k] = np.array([r[k] for r in rows])
    if a.sade:
        data["sade_runs"] = np.array([r["sade_runs"] for r in rows])
        # both answers optimise the same objective: keep the better one per speed
        use_sade = data["a_sade"] > data["a_ref"]
        data["g_best"] = np.where(use_sade, data["g_sade"], data["g_ref"])
        data["a_best"] = np.where(use_sade, data["a_sade"], data["a_ref"])
    data["zeta"] = a.zeta
    data["kind"] = a.kind
    data["fmode"] = a.fmode
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"gainmap_{a.tag}.npz"), **data)
    np.savetxt(os.path.join(RES, f"gainmap_{a.tag}.csv"),
               np.column_stack([data[k] for k in keys]), delimiter=",",
               header=",".join(keys), comments="", fmt="%.4f")


if __name__ == "__main__":
    main()
