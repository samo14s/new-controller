"""Shared setup for the scripts: paths, the testbed, the identified model."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
RESULTS = os.path.join(ROOT, 'results')
FIGS = os.path.join(RESULTS, 'figures')
DIGIT = os.path.join(ROOT, 'data', 'digitized')
os.makedirs(FIGS, exist_ok=True)

import numpy as np  # noqa: E402

from testbed import Testbed, identification_input  # noqa: E402
from sddmdc import SdDMDc  # noqa: E402

ID_SEED = 1                # noise realisation of the identification record
ROM_FILE = os.path.join(RESULTS, 'rom.npz')


class Tee:
    """print() to the console and to a results/*.txt log at once."""

    def __init__(self, name):
        self.f = open(os.path.join(RESULTS, name), 'w')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s)
        self.f.write(s + '\n')
        self.f.flush()


def identify(T=None, seed=ID_SEED, **kw):
    """Run the Sec. 3.1 protocol on the testbed (default: calibrated zeta)
    and identify the sd-DMDc model with the project's default options
    (normalize='group')."""
    if T is None:
        import calib
        T = Testbed(zeta=calib.load()['zeta'])
    u = identification_input(T.fs)
    out = T.run(u, seed=seed)
    opts = dict(normalize='group')
    opts.update(kw)
    M = SdDMDc(out['w'], out['v'] / T.s_sens, u, T.fs, s_sens=T.s_sens, **opts)
    return T, M, u, out
