"""Check 6: slices that sit exactly on the entry/exit angle (measure-zero depths).

tdsim.c decides "in cut" with phi >= phist && phi <= phiex on angles computed
in floating point.  When the helix lag of a slice is an exact multiple of the
angular step 2 pi/spr, that slice lands exactly on the entry angle pi/2 at some
steps; rounding then includes it for some teeth and not for others, so the
cutting force is no longer exactly tooth-periodic and a small once-per-
revolution forcing appears (like runout).  For layer j = 0 and k = 1 this
happens at a* = (2 pi/spr) * nlay * R / (0.5 tan(beta)) = 5.8905 mm.
The run at a* is compared with depths 1e-9 relative away.  Writes
check6_slice_coincidence.json.
"""
import json
import common
import numpy as np
import params as P
import tdsim
import detector as Dt

a_star = (2 * np.pi / P.STEPS_PER_REV) * P.N_LAYERS * (P.D_TOOL / 2) / (0.5 * np.tan(P.HELIX))
rows = []
for a in (a_star, a_star * (1 + 1e-9), a_star * (1 - 1e-9)):
    r = tdsim.simulate(2600, a, 1200.0, store=True, nrev=300)
    e = np.diff(r["x"][::P.STEPS_PER_REV // P.N_TEETH])
    c = Dt.classify(r["x"], P.STEPS_PER_REV)
    rows.append(dict(rpm=2600, g=1200.0, a_mm=a * 1e3,
                     rms_e_first10pct_m=float(np.sqrt(np.mean(e[:120] ** 2))),
                     rms_e_last10pct_m=float(np.sqrt(np.mean(e[-120:] ** 2))),
                     D1_stable=c["stable_decay"]))
    print(rows[-1])
with open(common.out("check6_slice_coincidence.json"), "w") as f:
    json.dump(dict(a_star_mm=a_star * 1e3, rows=rows), f, indent=1)
