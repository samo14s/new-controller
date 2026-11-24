"""Are the 'diverged' runs unbounded, or excursions beyond the 1 cm abort that come back?

Purpose.  The simulator stops a run as divergent when |x| exceeds 1 cm
(src/tdsim.c).  attractors_band_spot.py finds such divergences at 700 and
1020-1060 rpm, while runs at neighbouring depths come back to the forced orbit
after excursions of up to 9.8 mm, just below the abort.  So 'diverged' may mean
an unbounded escape or only an excursion beyond 1 cm.  This script reruns every
diverged run of attractors_band_spot.json with the abort raised to 10 cm and
classifies where it ends.  Either way the model is far outside its validity
(1 cm is 200 feeds per tooth); the point is only what 'divergence' means.

Protocol.  A private build of src/tdsim.c in a temporary directory with only
the threshold of the abort changed (1e-2 -> 1e-1 m; same flags as tdsim.py, plain
DVF variant), swapped for tdsim._lib inside the workers, so tdsim.simulate is
used unchanged and no file of the repository is modified.  Same runs as
attractors_band_spot.py (1600 revolutions, 9 V clip, full surface memory, gain
of the gain map, step entry or face entry plus a kick at revolution 100),
classified as there: forced / chatter / transient / diverged (now |x| > 10 cm),
with the largest |x| and the revolution of the largest |x|; also the revolution
after the disturbance (t = 0 for the step entry, the kick otherwise) from which
the run stays on the forced orbit (per-revolution max |x(t) - x(t - T)| below
0.01 fz to the end), and the D1 verdict over the 300 revolutions after the
disturbance, as the band and map campaigns judge it (step entry: revolutions
0-300; kick: x[st : 400 * 512 + 1], st the first tooth boundary after the kick).

Run (2 worker processes, under 2 min):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_abort.py --workers 2
Writes verification/paper/attractors_abort.json.
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import time
from multiprocessing import Pool

import numpy as np

import attractors_common as AC

HERE = AC.HERE
ABORT_OLD, ABORT_NEW = "if (!(fabs(z[0]) < 1e-2)) { diverged = 1; break; }", \
    "if (!(fabs(z[0]) < 1e-1)) { diverged = 1; break; }"
_SO = None


def build(tmp):
    src = open(os.path.join(AC.SRC, "tdsim.c")).read()
    assert src.count(ABORT_OLD) == 1
    c = os.path.join(tmp, "tdsim_abort10cm.c")
    open(c, "w").write(src.replace(ABORT_OLD, ABORT_NEW))
    so = os.path.join(tmp, "_tdsim_abort10cm.so")
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-DCOMB=0", "-o", so, c, "-lm"])
    return so


def _init(so):
    global _SO
    _SO = so


def run(args):
    rpm, g, a, ic = args
    AC.setup()
    import tdsim
    lib = ctypes.CDLL(_SO)
    lib.simulate.argtypes = tdsim._lib.simulate.argtypes
    lib.simulate.restype = tdsim._lib.simulate.restype
    tdsim._lib = lib                          # this worker only: the 10 cm abort
    from attractors_band_spot import NREV, KICK_REV, STEADY, STAT_TOL
    t0 = time.time()
    opts = {} if ic == "step" else dict(edge=True, kick_step=KICK_REV * AC.SPR, kick_dv=float(ic[4:]))
    r = tdsim.simulate(rpm, a, g, vmax=AC.VMAX, store=True, nrev=NREV, nreg=0, **opts)
    dstep = AC.divergence_step(r)
    x = r["x"]
    n_ok = len(x) if dstep is None else dstep + 1
    k = int(np.argmax(np.abs(x[:n_ok])))
    out = dict(rpm=rpm, g=g, a_mm=a * 1e3, ic=ic, diverged_10cm=dstep is not None,
               diverged_rev=None if dstep is None else dstep / AC.SPR, max_x_mm=float(np.abs(x[k]) * 1e3),
               rev_of_max=k / AC.SPR, exceeded_1cm=bool(np.abs(x[:n_ok]).max() > 1e-2))
    from tdboundary import classify_run
    if ic == "step":
        w0, w1, d0 = 0, 300 * AC.SPR + 1, 0
    else:
        ks = opts["kick_step"]
        w0, w1, d0 = (ks // AC.ND + 1) * AC.ND, (KICK_REV + 300) * AC.SPR + 1, ks
    out["d1_chatter_300rev"] = not classify_run(x[w0:w1], AC.SPR, dstep is not None and dstep < w1)["stable"]
    if dstep is None:
        _, res_rev = AC.per_rev(r, w0, (len(x) - 1 - w0) // AC.SPR)
        rr = AC.return_rev(res_rev[1:]) if ic == "step" else AC.return_rev(res_rev)
        out["back_on_forced_orbit_rev_after_disturbance"] = (
            None if rr is None else (rr + (1 if ic == "step" else 0)) + (w0 - d0) / AC.SPR)
    if dstep is not None:
        out["state"] = "diverged"
    else:
        i0, i1 = (NREV - STEADY) * AC.SPR, NREV * AC.SPR
        s = AC.window_stats(r, i0, i1, rpm)
        out.update({kk: s[kk] for kk in ("pp_x", "rms_x", "clip_frac", "res_tooth", "f_chatter")})
        if s["res_tooth"] < AC.RET_TOL:
            out["state"] = "forced"
        else:
            pa, pb = np.ptp(x[i0:i0 + 100 * AC.SPR]), np.ptp(x[i1 - 100 * AC.SPR:i1])
            h = (i0 + i1) // 2
            ra, rb = np.std(x[i0:h]), np.std(x[h:i1])
            out["state"] = "chatter" if abs(pb / pa - 1) <= STAT_TOL and abs(rb / ra - 1) <= STAT_TOL else "transient"
    out["runtime_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    spot = json.load(open(os.path.join(HERE, "attractors_band_spot.json")))
    jobs = [(r["rpm"], r["g"], r["a_mm"] * 1e-3, r["ic"]) for p in spot["speeds"] for r in p["runs"]
            if r["state"] == "diverged"]
    tmp = tempfile.mkdtemp()
    try:
        so = build(tmp)
        with Pool(a.workers, initializer=_init, initargs=(so,)) as pool:
            res = pool.map(run, jobs, chunksize=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    counts = {}
    for r in res:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    out = dict(note=__doc__, zeta=AC.ZETA, build_of_unmodified_source=AC.build_id(), abort_m=0.1,
               summary=dict(n_runs=len(res), state_counts=counts,
                            max_x_mm_range=[min(r["max_x_mm"] for r in res), max(r["max_x_mm"] for r in res)],
                            back_on_forced_orbit_rev_range=[
                                min(r["back_on_forced_orbit_rev_after_disturbance"] for r in res
                                    if r.get("back_on_forced_orbit_rev_after_disturbance") is not None),
                                max(r["back_on_forced_orbit_rev_after_disturbance"] for r in res
                                    if r.get("back_on_forced_orbit_rev_after_disturbance") is not None)]
                            if any(r.get("back_on_forced_orbit_rev_after_disturbance") is not None for r in res) else None,
                            d1_chatter_300rev_count=sum(1 for r in res if r["d1_chatter_300rev"])),
               runs=res, runtime_s=round(time.time() - t0, 1), workers=a.workers)
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_abort.json"), "w"), indent=1)
    for r in res:
        print(f"{r['rpm']:.0f} rpm {r['a_mm']:.2f} mm {r['ic']}: {r['state']} (max |x| {r['max_x_mm']:.1f} mm "
              f"at rev {r['rev_of_max']:.1f}; back on the forced orbit "
              f"{r.get('back_on_forced_orbit_rev_after_disturbance')} rev after the disturbance; "
              f"D1 chatter over 300 rev: {r['d1_chatter_300rev']})")
    print(json.dumps(AC.jsonable(out["summary"])))
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
