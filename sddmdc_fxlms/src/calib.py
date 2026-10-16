"""
calib.py — the two quantities the paper does not publish but its control
results depend on, fitted once (scripts/calibrate.py) and read everywhere.

  zeta      modal damping ratio of the truth plate (all modes);
  mu_scale  map from the paper's step size to this testbed's:
            mu_testbed = mu_scale * mu_paper.  Tonal FxLMS adapts at a rate
            proportional to mu * A_ref^2 * |Gc| * |Gs| (theory.py), and the
            paper does not report the reference amplitude, the internal signal
            scalings or whether the factor 2 of Eq. (20) is implemented, so its
            mu is known only up to this one scalar.

Without results/calibration.json the defaults (zeta = 0.006, mu_scale = 1)
are used and the runs are labelled UNCALIBRATED.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CAL_FILE = os.path.join(os.path.dirname(_HERE), 'results', 'calibration.json')
DEFAULT = dict(zeta=0.006, mu_scale=1.0, calibrated=False)


def load():
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE) as f:
            d = json.load(f)
        d['calibrated'] = True
        return d
    return dict(DEFAULT)


def save(d):
    with open(CAL_FILE, 'w') as f:
        json.dump(d, f, indent=2)
