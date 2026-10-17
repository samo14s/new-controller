"""Shared helpers for the SADE verification (imports src/ read-only)."""
import os
import sys
import json
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

RPMS = (1000, 1950, 2500)
G_LO, G_HI = 0.0, 3000.0
QSTEP = 0.01            # objective cache quantum, V s/m (see README)


def qkey(g):
    return round(float(g) / QSTEP)


class CachedFitness:
    """fitness(rpm, g, 'force') with g quantised to QSTEP and memoised.

    The cache is shared by all runs of one script and stored on disk so that the
    brute-force grid and the SADE runs reuse each other's evaluations.
    """

    def __init__(self, rpm, kind="force", path=None):
        from objective import fitness
        self._fit = fitness
        self.rpm, self.kind = rpm, kind
        self.path = path or os.path.join(HERE, f"cache_{kind}_{rpm}.pkl")
        self.cache = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                self.cache = pickle.load(fh)
        self.n_calls = 0
        self.n_new = 0

    def detail(self, g):
        k = qkey(g)
        self.n_calls += 1
        if k not in self.cache:
            f, al, fd = self._fit(self.rpm, k * QSTEP, self.kind, detail=True)
            self.cache[k] = (float(f), float(al), float(fd))
            self.n_new += 1
        return self.cache[k]

    def __call__(self, g):
        return self.detail(np.atleast_1d(g)[0])[0]

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self.cache, fh)
        os.replace(tmp, self.path)


def dump_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
