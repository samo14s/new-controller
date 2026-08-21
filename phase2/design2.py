"""design2.py — one optimiser, one objective, the Stage 3-4 controller set."""
import numpy as np

import config as C
import ctrl2 as K2
from eval2 import evaluate
from pso import pso


class Design2:
    def __init__(self, kind, plant, plate, ss_mu=None):
        self.kind = kind
        self.plant = plant
        self.plate = plate
        self.ss_mu = ss_mu
        bd = C.BOUNDS2[kind]
        self.names = list(bd.keys())
        self.lo = np.array([bd[k][0] for k in self.names], float)
        self.hi = np.array([bd[k][1] for k in self.names], float)
        self.n = len(self.names)

    def decode(self, u):
        v = self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * (self.hi - self.lo)
        return dict(zip(self.names, v))

    def build(self, u):
        return K2.build(self.kind, self.plant, self.decode(u), self.ss_mu)


def optimise(design, seeds=None, verbose=False, **kw):
    seeds = C.OPT['seeds'] if seeds is None else seeds

    def fit(u):
        try:
            c = design.build(u)
        except Exception:
            return -1e4
        return evaluate(design.plate, c, **kw)

    best = (None, -np.inf, None)
    for sd in seeds:
        x, J, info = pso(fit, design.n, seed=sd, verbose=verbose)
        if J > best[1]:
            best = (x, J, info)
        if verbose:
            print(f'   seed {sd}: J = {J:+.4f}', flush=True)
    x, J, info = best
    c = design.build(x)
    return dict(kind=design.kind, x=x, J=J, ctrl=c, params=design.decode(x),
                n_params=c.n_params, order=c.order, history=info['history'])
