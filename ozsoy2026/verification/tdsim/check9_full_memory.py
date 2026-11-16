"""The full surface memory (tdsim nreg = 0) against a brute force (audit check).

nreg = 0 cuts the chip against the deepest earlier pass at each angular slot,
tracked as the last pass that cut there.  The brute force is the same
simulator built with NREG_MAX = 600: the chip is the least over the last 600
passes, which is the full memory for runs of up to 600 passes (200
revolutions here).  Cases: a stable cut (where only the tiny chips at the
entry angle can differ from single regeneration), and four clipped cuts with
large-amplitude chatter, where 4 passes are not enough.
Writes check9_full_memory.json.  About 10 min on one core.

Result: the stable cut differs from single regeneration by 4.9 % (the tool
leaves the cut briefly during the entry transient) and from the brute force
by 0.3 %.  Clipped chatter: the amplitude agrees with the brute force to
0.4 % and the run diverges (1050 rpm) or stays bounded in the same cases;
pointwise the chaotic trajectories separate.  With 4 passes, 2 of the 4
chatter cases diverge where the full memory stays bounded, and a third is
27 % too large.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)
import tdsim                                  # noqa: E402
from zoa import a_lim                         # noqa: E402


def brute_library(tmp):
    """tdsim.c built with NREG_MAX = 600 (plain-DVF variant)."""
    src = open(os.path.join(SRC, "tdsim.c")).read()
    assert src.count("#define NREG_MAX 8") == 1
    c = os.path.join(tmp, "tdsim_brute.c")
    open(c, "w").write(src.replace("#define NREG_MAX 8", "#define NREG_MAX 600"))
    so = os.path.join(tmp, "_tdsim_brute.so")
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-DCOMB=0", "-o", so, c, "-lm"])
    lib = ctypes.CDLL(so)
    lib.simulate.argtypes = tdsim._lib.simulate.argtypes
    lib.simulate.restype = tdsim._lib.simulate.restype
    return lib


def summary(r):
    x = r["x"]
    k = (np.nonzero(x)[0][-1] + 1) if r["diverged"] else len(x)
    return dict(diverged=bool(r["diverged"]), max_x_mm=float(np.abs(x[:k]).max() * 1e3))


tmp = tempfile.mkdtemp()
try:
    brute = brute_library(tmp)
    out = dict()
    c = dict(rpm=2500.0, a=6e-3, g=536.0)
    r1 = tdsim.simulate(store=True, nrev=80, **c)
    r0 = tdsim.simulate(store=True, nrev=80, nreg=0, **c)
    out["stable_full_vs_single"] = dict(case=c, max_rel_dev=float(np.abs(r0["x"] - r1["x"]).max() / np.abs(r1["x"]).max()))
    print(out["stable_full_vs_single"], flush=True)
    rows = []
    for n, g, f, kw in ((2350.0, 1217.0, 0.9, {}), (1050.0, 1500.0, 0.9, {}), (2500.0, 1500.0, None, dict(a=15e-3)),
                        (1090.0, 1500.0, 0.98, dict(edge=True, kick_step=100 * 512 + 37, kick_dv=12e-3))):
        kw = dict(kw)
        a = kw.pop("a", None) or f * a_lim(n, g)
        args = dict(vmax=9.0, store=True, nrev=160, **kw)
        rf = tdsim.simulate(n, a, g, nreg=0, **args)
        r4 = tdsim.simulate(n, a, g, nreg=4, **args)
        lib, tdsim._lib = tdsim._lib, brute
        try:
            rb = tdsim.simulate(n, a, g, nreg=600, **args)
        finally:
            tdsim._lib = lib
        m = min(len(rf["x"]), len(rb["x"]))
        dev = float(np.abs(rf["x"][:m] - rb["x"][:m]).max() / max(np.abs(rb["x"][:m]).max(), 1e-12))
        row = dict(rpm=n, g=g, a_mm=a * 1e3, options={k: v for k, v in kw.items()}, full_vs_brute_max_rel_dev=dev,
                   full=summary(rf), brute600=summary(rb), nreg4=summary(r4))
        rows.append(row)
        print(row, flush=True)
    out["clipped_chatter"] = rows
    json.dump(out, open(os.path.join(HERE, "check9_full_memory.json"), "w"), indent=1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
