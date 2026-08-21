"""
design.py — one optimiser, one objective, five structures.
==========================================================
The decision vector lives in [0, 1]^n so every parameter is explored at the
same scale whatever its physical magnitude.  This is a fairness requirement:
without it the structure with the widest bounds would be penalised.
"""
import numpy as np

import config as C
import controllers as K
from objective import evaluate
from pso import pso


class Design:
    """Normalised vector -> (ss, pd)."""

    def __init__(self, kind, plate, alpha40=None, removal=None, eta_hat=0.0):
        self.kind = kind
        self.plate = plate
        self.alpha40 = alpha40
        self.removal = removal
        self.eta_hat = eta_hat
        bd = C.BOUNDS[kind]
        self.names = list(bd.keys())
        self.lo = np.array([bd[k][0] for k in self.names], float)
        self.hi = np.array([bd[k][1] for k in self.names], float)
        self.n = len(self.names)

    def decode(self, u):
        v = self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * (self.hi - self.lo)
        return dict(zip(self.names, v))

    def build(self, u):
        ss, pd, _ = K.build(self.kind, self.plate, self.decode(u),
                            alpha40=self.alpha40, removal=self.removal,
                            eta_hat=self.eta_hat)
        return ss, pd

    def order(self, u):
        return self.build(u)[0][0].shape[0]


def fitness_of(design, **kw):
    def f(u):
        try:
            ss, pd = design.build(u)
        except Exception:
            return -1e4
        return evaluate(design.plate, ss, pd=pd, **kw)
    return f


def optimise(design, seeds=None, verbose=False, **kw):
    """Run the PSO from each seed and keep the best."""
    seeds = C.OPT['seeds'] if seeds is None else seeds
    fit = fitness_of(design, **kw)
    best = (None, -np.inf, None)
    for sd in seeds:
        x, J, info = pso(fit, design.n, seed=sd, verbose=verbose)
        if J > best[1]:
            best = (x, J, info)
        if verbose:
            print(f'   seed {sd}: J = {J:+.4f}', flush=True)
    x, J, info = best
    ss, pd = design.build(x)
    return dict(kind=design.kind, x=x, J=J, ss=ss, pd=pd,
                params=design.decode(x), order=ss[0].shape[0],
                n_params=design.n, history=info['history'])
