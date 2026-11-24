"""Shared helpers of the numerical-protocol checks (verification/paper/protocol_*.py).

The checks re-run parts of the C1 kick protocol of src/c1_unsafe_zone.py with
one protocol constant changed at a time (kick revolution, number of phases,
clip level, classification window).  c1_unsafe_zone reads KICK_REV, WINDOW and
N_PHASE as module constants, so instead of patching them this module repeats
its two kick functions with those constants as arguments:

  kick_run        = c1_unsafe_zone._kick_chatters (face entry, full surface
                    memory nreg = 0, kick dv added to x' at step ks, the run
                    judged by tdboundary.classify_run from the first tooth
                    boundary after the kick to the end of the run)
  critical_kick   = c1_unsafe_zone._critical_kick (upward search on DV_GRID
                    from DV_LO, nan if the DV_LO kick already chatters, inf
                    if no kick up to 3 m/s does, else N_BISECT bisections in
                    log dv of the first interval that ends in chatter; the
                    returned value is the upper end of the last interval)

With kick_rev = 100, window = 300, n_phase = 16 and vmax = 9 V they are the
stored protocol of results/c1_basin.npz; protocol_kick_timing.py checks that
they reproduce the stored values exactly.  Each tested kick is kept in a trace
(dv, verdict, the classifier's growth factor mu and envelope ratio q, and the
end regenerative displacement), so that the closeness of each verdict can be
read afterwards.

The damping is set at import to the value of results/gainmap_main.npz
(1.34 %), before any other src module computes anything.  No module of src
used here caches damping-dependent quantities: zoa's frequency grid and
ALPHA, feasibility's slice-force harmonics and c1_unsafe_zone's constants
do not depend on it, and structure.G reads params.C at call time.

Not a script; imported by the protocol_*.py scripts.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")

import params as P                                              # noqa: E402

ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
P.set_damping(ZETA)

import c1_unsafe_zone as C1                                     # noqa: E402
from tdsim import simulate, SOURCE_SHA                          # noqa: E402
from tdboundary import classify_run                             # noqa: E402
from zoa import a_lim                                           # noqa: E402

SPR, ND, NREG = C1.SPR, C1.ND, C1.NREG
DV_GRID, DV_LO, N_BISECT = C1.DV_GRID, C1.DV_LO, C1.N_BISECT
VMAX = C1.VMAX                                                  # 9 V
G_DESIGN = 1500.0                                               # V s/m, the gain of the C1 basin points
SPEEDS = tuple(float(n) for n in C1.BASIN_RPM)                  # 1090, 1420, 2500, 2870 rpm
BUILD = SOURCE_SHA[:12]


def basin():
    """results/c1_basin.npz as a dict of arrays."""
    return dict(np.load(os.path.join(RES, "c1_basin.npz")))


def stored_row(rpm, frac, vmax=VMAX):
    """Stored critical kicks (m/s, 16 phases) and the row's metadata."""
    B = basin()
    m = (B["rpm"] == rpm) & np.isclose(B["frac"], frac) & np.isclose(B["vmax"], vmax)
    i = np.nonzero(m)[0]
    assert len(i) == 1, (rpm, frac, vmax)
    i = int(i[0])
    return dict(dv=B["dv"][i].astype(float), g=float(B["g"][i]), a_lim_mm=float(B["a_lim"][i]),
                decay=float(B["decay"][i]), p_mode=float(B["p_mode"][i]),
                tooth_impulse=float(B["tooth_impulse"][i]), mass=float(B["mass"]))


def depth(rpm, frac, g=G_DESIGN):
    """Depth (m) of a basin point, as run_basin computes it (frac * a_lim(rpm, g))."""
    return frac * a_lim(rpm, g)


def kick_step(phase, n_phase, kick_rev):
    """Step of the kick: phase of n_phase per tooth period, in revolution kick_rev."""
    return kick_rev * SPR + (phase * ND) // n_phase


def kick_run(rpm, a, g, vmax, ks, dv, kick_rev, window, extra_windows=()):
    """One kicked run (as c1_unsafe_zone._kick_chatters with KICK_REV = kick_rev,
    WINDOW = window).  Returns the verdict and the classifier's numbers.
    extra_windows: shorter windows (revolutions after kick_rev) also judged on
    the same run, from the same start, ending at revolution kick_rev + w."""
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=kick_rev + window, edge=True,
                 kick_step=ks, kick_dv=dv, nreg=NREG)
    st = (ks // ND + 1) * ND
    out = dict(dv=float(dv), **_verdict(r["x"][st:], r["diverged"]),
               diverged=bool(r["diverged"]), max_x_mm=float(r["max_x"] * 1e3))
    if extra_windows:
        # tdsim stops a run at |x| >= 1 cm and leaves zeros after the abort step; a shorter
        # window is diverged only if the abort came before its end.  The run up to the end of
        # a shorter window is the same as a run of that length (the integration is causal).
        if r["diverged"]:
            nz = np.flatnonzero(r["x"])
            n_abort = int(nz[-1]) + 1 if nz.size else 0
        else:
            n_abort = len(r["x"])
        for w in extra_windows:
            end = (kick_rev + w) * SPR + 1
            out[f"w{w}"] = _verdict(r["x"][st:end], n_abort < end)
    return out


def _verdict(x, diverged):
    c = classify_run(x, SPR, diverged)
    return dict(chatter=not c["stable"], mu=float(c["mu"]), q=float(c["q"]),
                dx_end_over_fz=float(c["dx_end_over_fz"]))


def critical_kick(rpm, a, g, vmax, phase, n_phase=16, kick_rev=C1.KICK_REV, window=C1.WINDOW,
                  extra_windows=()):
    """Critical kick (m/s) as c1_unsafe_zone._critical_kick with the protocol constants as
    arguments.  Returns (value, trace) with one trace entry per run."""
    ks = kick_step(phase, n_phase, kick_rev)
    trace = []

    def chatters(dv):
        o = kick_run(rpm, a, g, vmax, ks, dv, kick_rev, window, extra_windows)
        trace.append(o)
        return o["chatter"]

    if chatters(DV_LO):
        return float("nan"), trace
    for k in range(1, len(DV_GRID)):
        if chatters(DV_GRID[k]):
            break
    else:
        return float("inf"), trace
    lo, hi = np.log(DV_GRID[k - 1]), np.log(DV_GRID[k])
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if chatters(np.exp(mid)):
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi)), trace


def bracket(trace, value):
    """(largest kick below value that did not chatter, value) from a trace."""
    below = [t["dv"] for t in trace if not t["chatter"] and t["dv"] < value]
    return (max(below) if below else float("nan")), value


# ------------------------------------------------------------- output ----
def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(path, obj):
    """Write atomically (tmp file + rename), so a stop never leaves a half file.
    inf and nan are written as Infinity and NaN (read back by Python's json)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    os.replace(tmp, path)


def load_json(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def header(script, doc):
    """Common provenance block."""
    return dict(script=os.path.relpath(script, ROOT), purpose=doc.strip().splitlines()[0],
                zeta=ZETA, g=G_DESIGN, vmax_default=VMAX, nreg=NREG, spr=SPR, nd=ND,
                dv_grid=DV_GRID.tolist(), n_bisect=N_BISECT, build=BUILD,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
