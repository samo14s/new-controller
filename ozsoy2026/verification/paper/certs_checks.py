"""Certificate checks asked by the second external review.

(1) nconv: collocation convergence of the averaged certificate C2 at more points.
    The certified kick of c2_certificate.certify_kick (e = unit velocity, the
    command bound of the C2 solve, as for verification/paper/revision_runs.json
    beta_vs_N) at N = nodes_for(rpm), round(1.5 N) and 2 N, at 1090, 1420, 2500
    and 2870 rpm, g = 1500 V s/m and a = 0.5, 0.9 and 0.98 a_lim(rpm, 1500).  The
    two points already in revision_runs.json (2500 and 1090 rpm at 0.9 a_lim)
    are reused, not recomputed.
(2) offaxis: the off-axis validation of the periodic certificate C3 with the
    full surface memory.  The stored C3 campaign (src/c3_periodic.td_offaxis)
    applied the boundary points of the certified tube in the simulator with the
    last four passes remembered (nreg = 4); the C3 ellipsoids are not stored, so
    C3 is solved again (c3_periodic.certify with its stored settings, M = 16
    segments) at 0.9 a_lim at the four speeds and at 0.98 a_lim at 1090, 1420
    and 2870 rpm (2500 rpm at 0.98 a_lim has no certificate), its size compared
    with results/c3_basin.npz, and the same boundary points (c2.boundary_directions
    at the nodes 0, 4, 8 and 12 of 16, kick at revolution 100 of a face entry,
    300 revolutions classified as in C1) applied with nreg = 0 (full memory) and,
    for comparison, nreg = 4.

Damping of every stored result (params.set_damping(gainmap_main.npz zeta), 1.34 %).
Nothing in src/ is modified; td_offaxis is copied here with the memory as a parameter.

    cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/certs_checks.py nconv
    cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/certs_checks.py offaxis

Each part writes its own JSON after every unit (certs_nconv.json, certs_offaxis.json)
and skips the units already there, so an interrupted run resumes.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")

import params as P  # noqa: E402

ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
P.set_damping(ZETA)

import c2_certificate as c2  # noqa: E402
import c3_periodic as C3  # noqa: E402
from zoa import a_lim  # noqa: E402

G = 1500.0
SPEEDS = (1090.0, 1420.0, 2500.0, 2870.0)
NCONV_FRACS = (0.5, 0.9, 0.98)
OFFAXIS = [(n, 0.9) for n in SPEEDS] + [(1090.0, 0.98), (1420.0, 0.98), (2870.0, 0.98)]


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)


def run_nconv():
    path = os.path.join(HERE, "certs_nconv.json")
    out = _load(path)
    out.setdefault("note", "C2 certified kick (mm/s, command bound of the C2 solve) at N, round(1.5 N), 2 N; "
                           "g = 1500 V s/m; N = c2.nodes_for(rpm)")
    pts = out.setdefault("points", {})
    rr = _load(os.path.join(HERE, "revision_runs.json")).get("beta_vs_N", {})
    for n in SPEEDS:
        al = a_lim(n, G)
        N0 = c2.nodes_for(n)
        Ns = [N0, int(round(1.5 * N0)), 2 * N0]
        for fr in NCONV_FRACS:
            key = f"{n:.0f}_{fr}"
            if key in pts and len(pts[key].get("beta_mmps", [])) == 3:
                continue
            stored = {}
            for v in (rr if isinstance(rr, list) else []):
                if v.get("kind") == "nconv" and v["args"][0] == n and abs(v["args"][1] - fr) < 1e-9:
                    stored[int(v["args"][2])] = v["beta_mmps"]
            if all(N in stored for N in Ns):
                pts[key] = dict(rpm=n, frac=fr, a_mm=fr * al * 1e3, N=Ns, beta_mmps=[stored[N] for N in Ns],
                                source="revision_runs.json beta_vs_N")
                _save(path, out)
                continue
            rec = dict(rpm=n, frac=fr, a_mm=fr * al * 1e3, N=Ns, beta_mmps=[], notes=[], seconds=[])
            for N in Ns:
                t0 = time.time()
                beta, note = c2.certify_kick(n, fr * al, G, N=N)
                rec["beta_mmps"].append(beta * 1e3)
                rec["notes"].append(note)
                rec["seconds"].append(round(time.time() - t0, 1))
                print(f"nconv {n:.0f} rpm {fr} a_lim N {N}: {beta * 1e3:.4f} mm/s ({time.time() - t0:.0f} s) {note}", flush=True)
            pts[key] = rec
            _save(path, out)
    ch = []
    for v in pts.values():
        b = v["beta_mmps"]
        if b[0] > 0:
            ch.append(dict(rpm=v["rpm"], frac=v["frac"], rel_2N=(b[2] - b[0]) / b[0], rel_15N=(b[1] - b[0]) / b[0],
                           max_abs_rel=max(abs(b[1] - b[0]), abs(b[2] - b[0])) / b[0]))
    out["summary"] = dict(n_points=len(pts), max_abs_rel_change=max(c["max_abs_rel"] for c in ch) if ch else None,
                          rel_2N_range=[min(c["rel_2N"] for c in ch), max(c["rel_2N"] for c in ch)] if ch else None,
                          per_point=ch)
    _save(path, out)
    print("nconv summary", json.dumps(out["summary"])[:600], flush=True)


def td_offaxis_nreg(rpm, a, g, cert, nreg, kick_rev=100, window=300):
    """c3_periodic.td_offaxis with the surface memory as a parameter (nreg = 0: full memory)."""
    from tdsim import simulate
    from tdboundary import classify_run
    Tm, W = cert[0], cert[1]
    mc, n = len(W), Tm.shape[0]
    spr = P.STEPS_PER_REV
    nd = spr // P.N_TEETH
    assert nd % mc == 0
    bad = runs = 0
    peaks = []
    for k in range(0, mc, max(1, mc // 4)):
        ks = kick_rev * spr + k * nd // mc
        for X in c2.boundary_directions((Tm, np.linalg.inv(W[k])), n):
            r = simulate(rpm, a, g, vmax=C3.VMAX, store=True, nrev=kick_rev + window, edge=True, nreg=nreg,
                         kick_step=ks, pert=c2.deviation_to_pert(rpm, X))
            st = (ks // nd + 1) * nd
            bad += not classify_run(r["x"][st:], spr, r["diverged"])["stable"]
            peaks.append(float(np.max(np.abs(r["x"][st:]))))
            runs += 1
    return bad, runs, max(peaks) if peaks else None


def run_offaxis():
    path = os.path.join(HERE, "certs_offaxis.json")
    out = _load(path)
    out.setdefault("note", "C3 re-solved (stored settings, M = 16) and its off-axis test in the simulator with "
                           "nreg = 0 (full memory) and nreg = 4 (stored campaign); g = 1500 V s/m, 9 V clip")
    pts = out.setdefault("points", {})
    B = np.load(os.path.join(RES, "c3_basin.npz"), allow_pickle=True)
    C3.set_immersion(C3.HALF)
    for n, fr in OFFAXIS:
        key = f"{n:.0f}_{fr}"
        if key in pts and "nreg0" in pts[key]:
            continue
        al = a_lim(n, G)
        a = fr * al
        t0 = time.time()
        r = C3.certify(n, a, G, C3.HALF, full=True)
        t1 = time.time()
        i = int(np.nonzero((B["rpm"] == n) & np.isclose(B["frac"], fr))[0][0])
        stored = float(B["cert"][i])
        rec = dict(rpm=n, frac=fr, a_mm=a * 1e3, size_mmps=r["size"] * 1e3, stored_mmps=stored * 1e3,
                   rel_to_stored=(r["size"] / stored - 1.0) if stored > 0 else None, note=r["note"],
                   solve_s=round(t1 - t0, 1), stored_offaxis=[int(B["offaxis_bad"][i]), int(B["offaxis_runs"][i])])
        print(f"offaxis {n:.0f} rpm {fr}: C3 {r['size'] * 1e3:.4f} mm/s (stored {stored * 1e3:.4f}) in {t1 - t0:.0f} s; {r['note']}",
              flush=True)
        if r["cert"] is not None:
            C3.set_immersion(C3.HALF)
            for nreg in (0, 4):
                t2 = time.time()
                bad, runs, pk = td_offaxis_nreg(n, a, G, r["cert"], nreg)
                rec[f"nreg{nreg}"] = dict(chatter=bad, runs=runs, max_abs_x_mm=pk * 1e3 if pk is not None else None,
                                          seconds=round(time.time() - t2, 1))
                print(f"   nreg {nreg}: {bad} of {runs} chattered ({time.time() - t2:.0f} s)", flush=True)
        else:
            rec["nreg0"] = None
        pts[key] = rec
        _save(path, out)
    done = [v for v in pts.values() if v.get("nreg0")]
    out["summary"] = dict(n_points=len(pts), runs_nreg0=sum(v["nreg0"]["runs"] for v in done),
                          chatter_nreg0=sum(v["nreg0"]["chatter"] for v in done),
                          runs_nreg4=sum(v["nreg4"]["runs"] for v in done if v.get("nreg4")),
                          chatter_nreg4=sum(v["nreg4"]["chatter"] for v in done if v.get("nreg4")),
                          max_abs_rel_to_stored=max(abs(v["rel_to_stored"]) for v in pts.values() if v["rel_to_stored"] is not None)
                          if pts else None)
    _save(path, out)
    print("offaxis summary", out["summary"], flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("nconv", "all"):
        run_nconv()
    if what in ("offaxis", "all"):
        run_offaxis()
