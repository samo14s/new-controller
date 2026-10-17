"""Fitness of Eq. (2): a_lim(Omega, g) if the demanded force stays within F_max.

Demand definitions (the paper writes F_d = g_dvf * xdot with g in V s/m and
compares it with 27 N; the dimensionally consistent force is K_a g xdot):
  "force"   F_d = K_a * g * |xdot|          (default; N)
  "volt"    F_d = g * |xdot|                (Eq. as printed, numerically)
  "actual"  |f_act| after the actuator dynamics Ga(s)
"""
import numpy as np
import params as P
from zoa import a_lim
from tdsim import simulate

DEMANDS = ("force", "volt", "actual")


def demand(rpm, a, g, kind="force", skip_revs=0, nrev=P.N_REVS):
    if skip_revs:
        r = simulate(rpm, a, g, store=True, nrev=nrev)
        i0 = skip_revs * P.STEPS_PER_REV
        v = np.abs(r["v"][i0:]).max()
        if kind == "force":
            return P.GA_GAIN * g * v
        if kind == "volt":
            return g * v
        return np.abs(r["fact"][i0:]).max()
    r = simulate(rpm, a, g, nrev=nrev)
    if r["diverged"]:
        return np.inf
    if kind == "force":
        return P.GA_GAIN * r["max_demand"]
    if kind == "volt":
        return r["max_demand"]
    return r["max_fact"]


def fitness(rpm, g, kind="force", fmax=P.F_MAX, skip_revs=0, detail=False):
    """Eq. (2) in mm; 0 for infeasible candidates."""
    g = float(np.atleast_1d(g)[0])
    al = a_lim(rpm, g)
    if not np.isfinite(al):
        return (0.0, al, np.nan) if detail else 0.0
    fd = demand(rpm, al, g, kind, skip_revs) if g > 0 else 0.0
    f = al * 1e3 if fd <= fmax else 0.0
    return (f, al, fd) if detail else f
