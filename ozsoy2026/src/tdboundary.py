"""Time-domain stability classification and boundary search.

The forced response of a stable cut repeats every revolution, so the
revolution-to-revolution difference r(t) = x(t) - x(t - T_rev) isolates the
non-periodic (chatter) part.  Its per-revolution peak e_k decays geometrically
in a stable cut and grows in an unstable one; the sign of the fitted log-slope
over the second half of the run decides.  Once the tool starts leaving the cut
the chatter amplitude saturates; a non-decaying envelope that is a sizeable
fraction of the motion is then also called chatter.
"""
import numpy as np
import params as P
from tdsim import simulate


def classify(rpm, a, g, vmax=0.0, nrev=120, slope_tol=0.0, level_tol=0.25):
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=nrev)
    if r["diverged"]:
        return dict(chatter=True, slope=np.inf, level=1.0, run=r)
    spr = P.STEPS_PER_REV
    x = r["x"][: nrev * spr].reshape(nrev, spr)
    e = np.abs(np.diff(x, axis=0)).max(axis=1) + 1e-18      # per-rev envelope
    k = np.arange(e.size)
    half = k >= e.size // 2
    slope = np.polyfit(k[half], np.log(e[half]), 1)[0]       # per revolution
    level = e[-5:].mean() / (np.ptp(x[-5:]) + 1e-18)
    chatter = bool(slope > slope_tol or (level > level_tol and slope > -0.01))
    return dict(chatter=chatter, slope=slope, level=level, run=r)


def boundary(rpm, g, vmax=0.0, a_hi=40e-3, tol=0.02e-3, nrev=120):
    """Largest stable depth by bisection (m); assumes one stable->chatter switch."""
    lo, hi = 0.0, a_hi
    if not classify(rpm, hi, g, vmax, nrev)["chatter"]:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if classify(rpm, mid, g, vmax, nrev)["chatter"]:
            hi = mid
        else:
            lo = mid
    return lo
