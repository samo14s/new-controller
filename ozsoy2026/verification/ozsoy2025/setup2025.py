"""Model set-up for the second data set: Ozsoy, Sims & Ozturk 2025 (MSSP 224, 111942).

The 2025 article used the same flexible workpiece, proof-mass actuator, tool and cutting conditions as
the 2026 case study of this repository: its Table 2 gives 129.3 Hz, 1.34 %, 1.34e7 N/m for the
preloaded structure, the actuator 3 s^2/(s^2 + 15.834 s + 2785.6) N/V, a 16 mm four-flute tool with a
45 degree helix, K_t = 660 MPa, K_r = 180 MPa, down milling at half immersion and 0.05 mm per tooth,
i.e. the values of src/params.py (which uses 1.34 % although the 2026 article's table states 1.45 %).
Its saturation experiments (its Figs. 10 and 12, 1700-2250 rpm, DVF control) were made after the
workpiece had been clamped again: 128.1 Hz, 1.48 % and 1.13e7 N/m, with a DVF gain 'optimised for this
modification' whose value is not stated.

apply_structure(fn, zeta, k) sets these three values in every src module, as
verification/paper/mc_common.apply_params does for the Monte Carlo samples (which keep k fixed):
params.FN, WN, K, M = K / WN^2, and C through params.set_damping; the cutting and actuator constants
are not touched.  Every module of src reads params.M, C and K at call time (tdsim passes them to the
C simulator at each call), so the switch is complete; check() verifies it on the simulator.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import params as P  # noqa: E402

NOMINAL = dict(fn=129.3, zeta=0.0134, k=1.34e7)       # Table 2 of the 2025 article = src/params.py
RECLAMPED = dict(fn=128.1, zeta=0.0148, k=1.13e7)     # Section 3.2 of the 2025 article (its Figs. 10, 12)
G_FIG6 = 253.0                                        # V s/m, Table 1 of the 2025 article (DVF)


def apply_structure(fn, zeta, k):
    P.FN = float(fn)
    P.WN = 2 * np.pi * P.FN
    P.K = float(k)
    P.M = P.K / P.WN ** 2
    P.set_damping(float(zeta))
    return dict(fn=P.FN, zeta=P.ZETA, k=P.K, m=P.M, c=P.C)


def structure():
    return dict(fn=P.FN, zeta=P.ZETA, k=P.K, m=P.M, c=P.C)


def check():
    """The switch reaches the closed form (zoa) and the simulator (tdsim)."""
    import zoa
    out = {}
    for name, s in (("nominal", NOMINAL), ("reclamped", RECLAMPED)):
        apply_structure(**s)
        out[name] = dict(structure(), a_lim0_1900_mm=float(zoa.a_lim(1900.0, 0.0) * 1e3))
    apply_structure(**NOMINAL)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=1))
