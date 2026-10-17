"""Demonstrates a guard against tooth-passing poles for the per-speed root
scan of src/zoa.py (not applied to src/; shown here so it can be adopted).

Guard: a sign change of Im a(w) on [w_i, w_{i+1}] is discarded when the
bracket contains or touches a tooth-passing harmonic, i.e. when
floor(w T / 2 pi) differs between the end points or either end point lies
within 1e-6 (in units of the harmonic number) of an integer.  The rest of the
scan is the same as in src/zoa.py.
Checks: the cases flagged by spurious_scan.py and the four check gains on a
1-rpm grid for both damping values.  Writes zoa_fix_demo.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
import json
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
import params as P
import zoa
import zoa_lobes as L


def a_lim_guarded(rpm, g, gcl=None, w=zoa.W_GRID):
    if gcl is None:
        gcl = zoa._gcl_grid(g, w)
    T = 60.0 / (P.N_TEETH * rpm)
    a = 4 * np.pi / (P.N_TEETH * P.K_TC * zoa.ALPHA * (1 - np.exp(-1j * w * T)) * gcl)
    im = a.imag
    idx = np.nonzero(np.sign(im[:-1]) * np.sign(im[1:]) < 0)[0]
    ph = w * T / (2 * np.pi)
    near = np.abs(ph - np.round(ph)) < 1e-6
    keep = (np.floor(ph[idx]) == np.floor(ph[idx + 1])) & ~near[idx] & ~near[idx + 1]
    idx = idx[keep]
    if idx.size == 0:
        return np.inf, np.nan
    t = im[idx] / (im[idx] - im[idx + 1])
    are = a.real[idx] + t * (a.real[idx + 1] - a.real[idx])
    wc = w[idx] + t * (w[idx + 1] - w[idx])
    ok = are > 0
    if not np.any(ok):
        return np.inf, np.nan
    j = np.argmin(np.where(ok, are, np.inf))
    return are[j], wc[j] / (2 * np.pi)


out = {"flagged_cases": []}
fl = os.path.join(HERE, "spurious_scan.csv")
extra = [dict(zeta=0.0145, rpm=1983.0, g=300.0), dict(zeta=0.0134, rpm=2181.0, g=3000.0),
         dict(zeta=0.0134, rpm=2418.0, g=6000.0)]
cases = (list(csv.DictReader(open(fl))) if os.path.exists(fl) else []) + extra
for c in cases:
    z, n, g = float(c["zeta"]), float(c["rpm"]), float(c["g"])
    P.set_damping(z)
    a_src = zoa.a_lim(n, g)
    a_fix, _ = a_lim_guarded(n, g)
    a_ref = L.envelope([n], g, z)[0][0]
    out["flagged_cases"].append(dict(zeta=z, rpm=n, g=g, a_src_mm=a_src * 1e3, a_guarded_mm=a_fix * 1e3,
                                     a_lobes_mm=a_ref * 1e3, rel_guarded=a_fix / a_ref - 1,
                                     rel_src=a_src / a_ref - 1))
    print(out["flagged_cases"][-1], flush=True)

rp = np.arange(500, 3001, 1.0)
dense = {}
for z in (0.0134, 0.0145):
    P.set_damping(z)
    for g in (0.0, 300.0, 800.0, 1500.0):
        gcl = zoa._gcl_grid(g)
        a_fix = np.array([a_lim_guarded(n, g, gcl=gcl)[0] for n in rp])
        a_src = zoa.sld(rp, g)
        a_ref = L.envelope(rp, g, z)[0]
        dense[f"{z}_{int(g)}"] = dict(max_abs_rel_guarded=float(np.max(np.abs(a_fix / a_ref - 1))),
                                      max_abs_rel_src=float(np.max(np.abs(a_src / a_ref - 1))))
        print(z, g, dense[f"{z}_{int(g)}"], flush=True)
out["dense_1rpm"] = dense
with open(os.path.join(HERE, "zoa_fix_demo.json"), "w") as fh:
    json.dump(out, fh, indent=1)
