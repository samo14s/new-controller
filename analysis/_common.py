"""Shared setup for the analysis scripts: puts baseline/plant on sys.path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLANT = os.path.join(ROOT, 'baseline', 'plant')
if PLANT not in sys.path:
    sys.path.insert(0, PLANT)

RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

# Table 4 of Du et al. (2024)
F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]
ZETA = (0.0031, 0.0017, 0.0027, 0.0056, 0.0035)
# Section 5: natural frequencies after a series of milling experiments
F_AFTER_MILLING = [632.0, 1162.0]

# Reference operating point "S" of the paper (Section 4.2)
RPM_S, AP_S, AE_S, FZ_S = 4900, 0.3e-3, 0.1e-3, 0.02e-3


def build_plate(calibrate=True, patch=True, PX=14, PZ=14):
    from chebyshev_plate import ChebyshevPlate
    p = ChebyshevPlate(PX=PX, PZ=PZ)
    if patch:
        p.add_piezo_patch()
    if calibrate:
        p.calibrate_frequencies(F_MEASURED)
    return p


class Tee:
    """Write to stdout and to a log file at the same time."""

    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, *a):
        line = ' '.join(str(x) for x in a)
        print(line)
        self.f.write(line + '\n')
        self.f.flush()
