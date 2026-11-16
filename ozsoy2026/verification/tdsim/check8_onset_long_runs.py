"""Time-domain onset with long runs, against ZOA (audit check of the D1 window).

Just above the linear onset the chatter saturates on a small limit cycle.  After
a step entry the run needs more than the 300 revolutions of src/tdboundary.py
to settle on it, so D1 with 300 revolutions calls such runs stable and puts the
boundary too high (check7: 0.4-2.2 % above ZOA).  Here the onset of the same 8
cases is found with 2000-revolution runs (same D1 test on the whole run), by
bisection to 1e-3 of the ZOA limit.  Result: the onset is 0.1-0.7 % above ZOA,
so the 300-revolution boundaries are 0.05-1.5 % high.
Writes check8_onset_long_runs.json (about 3 h on one core).
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from tdsim import simulate              # noqa: E402
from tdboundary import classify_run     # noqa: E402
from zoa import a_lim                   # noqa: E402

CASES = [(500, 0.0), (500, 1200.0), (1000, 630.0), (1300, 991.0), (1600, 0.0),
         (2500, 0.0), (2500, 535.0), (3000, 712.0)]
NREV, RTOL = 2000, 1e-3


def stable(rpm, a, g):
    r = simulate(rpm, a, g, store=True, nrev=NREV)
    return classify_run(r["x"], r["spr"], r["diverged"])["stable"]


c7 = {(b["rpm"], b["g"]): b for b in json.load(open(os.path.join(HERE, "check7_final_scheme.json")))["boundary"]}
out = []
for rpm, g in CASES:
    z = a_lim(rpm, g)
    lo, hi = 0.99, 1.05
    assert stable(rpm, lo * z, g) and not stable(rpm, hi * z, g)
    while hi - lo > RTOL:
        mid = 0.5 * (lo + hi)
        if stable(rpm, mid * z, g):
            lo = mid
        else:
            hi = mid
    row = dict(rpm=rpm, g=g, zoa_mm=z * 1e3, onset_over_zoa=0.5 * (lo + hi),
               td300_over_zoa=1 + c7[(rpm, g)]["rel"])
    out.append(row)
    print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "check8_onset_long_runs.json"), "w"), indent=1)
