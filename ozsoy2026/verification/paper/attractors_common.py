"""Shared helpers of the attractor analyses (attractors_kick.py, attractors_bifurcation.py).

Not an analysis by itself: it sets up the model as every C1 campaign does
(src/ on the path, damping of results/gainmap_main.npz, 1.34 %), and holds the
statistics of a stored run that both analyses report.

Statistics of a window [i0, i1) of a stored run (tdsim.simulate, store=True):
  pp_x, rms_x        peak-to-peak and RMS (about the mean) of x, m
  pp_V, rms_V        peak-to-peak and RMS of the applied (clipped) command V, V
  rms_v              RMS of x', m/s
  clip_frac          fraction of steps with |V| at the clip (|V| >= vmax (1 - 1e-12))
  res_tooth          max |x(t) - x(t - T)| over the window, m (0 on the forced orbit,
                     which is T-periodic, T the tooth period)
  nonsync_frac       variance of x not explained by its tooth-synchronous average,
                     over the variance of x (0: T-periodic motion; near 1: chatter)
  f_dom              frequency of the largest peak of the spectrum of x, Hz
  f_chatter          frequency of the largest peak of the spectrum of the non-synchronous
                     part of x (x minus its tooth-synchronous average), Hz: the chatter
                     frequency; both peaks with Hann window and parabolic interpolation
  f_tooth            tooth-passing frequency, Hz
The window must hold a whole number of tooth periods.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import params as P                          # noqa: E402

ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
SPR = P.STEPS_PER_REV                       # 512 steps per revolution
ND = SPR // P.N_TEETH                       # 128 steps per tooth period
VMAX = P.F_MAX / P.GA_GAIN                  # 9 V
RET_TOL = 0.01 * P.FZ                       # 0.5 um: tooth-periodic residual of a run back on the forced orbit


def setup():
    """Damping of every C1 campaign (read at call time by src; nothing in
    tdsim, zoa, structure or feasibility caches it).  Call in every worker."""
    P.set_damping(ZETA)
    return P


def build_id():
    from tdsim import SOURCE_SHA
    return SOURCE_SHA[:12]


def _peak(f, S, fmin=1.0):
    """Frequency of the largest peak of the amplitude spectrum S (parabolic
    interpolation of log S around the largest bin above fmin)."""
    m = f > fmin
    idx = np.nonzero(m)[0]
    k = idx[int(np.argmax(S[m]))]
    if 0 < k < len(S) - 1 and S[k - 1] > 0 and S[k + 1] > 0:
        a, b, c = np.log(S[k - 1]), np.log(S[k]), np.log(S[k + 1])
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(f[k] + d * (f[1] - f[0]))
    return float(f[k])


def window_stats(r, i0, i1, rpm, vmax=VMAX):
    """Statistics of steps [i0, i1) of the stored run r (see the module docstring)."""
    assert (i1 - i0) % ND == 0 and i0 >= ND
    x, v, V = r["x"][i0:i1], r["v"][i0:i1], r["vcmd"][i0:i1]
    xd = r["x"][i0 - ND:i1 - ND]
    dt = 60.0 / (rpm * SPR)
    K = (i1 - i0) // ND
    sync = x.reshape(K, ND).mean(axis=0)
    resid = x - np.tile(sync, K)
    var = float(np.var(x))
    w = np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), dt)
    S = np.abs(np.fft.rfft((x - x.mean()) * w))
    Sr = np.abs(np.fft.rfft((resid - resid.mean()) * w))
    return dict(
        pp_x=float(np.ptp(x)), rms_x=float(np.std(x)), mean_x=float(np.mean(x)),
        pp_V=float(np.ptp(V)), rms_V=float(np.sqrt(np.mean(V ** 2))),
        rms_v=float(np.sqrt(np.mean(v ** 2))),
        clip_frac=float(np.mean(np.abs(V) >= vmax * (1 - 1e-12))) if vmax > 0 else 0.0,
        res_tooth=float(np.max(np.abs(x - xd))),
        nonsync_frac=float(np.var(resid) / var) if var > 0 else 0.0,
        f_dom=_peak(f, S), f_chatter=_peak(f, Sr) if np.any(Sr > 0) else float("nan"),
        f_tooth=float(P.N_TEETH * rpm / 60.0))


def per_rev(r, i0, nrev):
    """Per revolution from step i0: peak-to-peak of x and the largest tooth-periodic
    residual max |x(t) - x(t - T)| (NaN after a divergence, and the residual NaN
    for a revolution that starts less than one tooth period after t = 0)."""
    x = r["x"]
    n_ok = len(x)
    if r["diverged"]:
        nz = np.nonzero(x)[0]
        n_ok = int(nz[-1]) + 1 if nz.size else 0
    pp = np.full(nrev, np.nan)
    res = np.full(nrev, np.nan)
    for k in range(nrev):
        a, b = i0 + k * SPR, i0 + (k + 1) * SPR
        if b > n_ok:
            break
        seg = x[a:b]
        pp[k] = np.ptp(seg)
        if a >= ND:
            res[k] = np.max(np.abs(seg - x[a - ND:b - ND]))
    return pp, res


def divergence_step(r):
    """Step at which a diverged run stopped (None if it did not diverge)."""
    if not r["diverged"]:
        return None
    nz = np.nonzero(r["x"])[0]
    return int(nz[-1]) if nz.size else 0


def return_rev(res_rev, tol=RET_TOL):
    """First revolution after which the per-revolution tooth-periodic residual stays
    below tol to the end of the record (None if it never does, or if it diverged)."""
    if np.any(~np.isfinite(res_rev)):
        return None
    above = np.nonzero(res_rev >= tol)[0]
    if above.size == 0:
        return 0
    k = int(above[-1]) + 1
    return k if k < len(res_rev) else None


def jsonable(o):
    """Recursively turn numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o
