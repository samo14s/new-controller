"""Long runs where the 300-revolution boundaries reach 1.03-1.05 of ZOA (audit check of the D1 window).

Near the forced resonance (about 2000 rpm, where the tooth-passing harmonic
sits on the mode and the feasible gain is small) the boundaries of the C1 map
without the voltage limit (step_free) reach 1.03-1.05 of the ZOA limit at
1970-2030 rpm, and one C4 face entry (the C1/C2 gain plus g2 = 2000, at
2000 rpm) stops at the top of the ladder, 1.05.  check8 found the D1 window
biased by 0.05-1.5 % at eight other cases; here the onset of these cases is
found as in check8: 2000-revolution runs (same D1 test, judged after the
entry), bisection to 1e-3 of the ZOA limit.
Writes check10_forced_resonance_long_runs.json (a few minutes on one core).
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P                                      # noqa: E402
from tdboundary import classify                         # noqa: E402
from zoa import a_lim                                   # noqa: E402
from c1_unsafe_zone import ENTRIES                      # noqa: E402
import c4_comb as c4                                    # noqa: E402

NREV, RTOL = 2000, 1e-3
RES = os.path.join(HERE, "..", "..", "results")


def onset(rpm, g, a_ref, name, lo, hi, g2=0.0):
    """Bisection of the long-run onset between lo and hi (fractions of a_ref)."""
    vmax, entry = ENTRIES[name]

    def stable(f):
        return classify(rpm, f * a_ref, g, vmax, NREV, g2=g2, **entry)["stable"]
    assert stable(lo) and not stable(hi), (rpm, name)
    while hi - lo > RTOL:
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


m = np.load(os.path.join(RES, "c1_map.npz"))
P.set_damping(float(m["zeta"]))
out = dict(c1_step_free=[], c4_face_entry=None)
for rpm in range(1970, 2031, 10):
    i = int(np.nonzero(m["rpm"] == rpm)[0][0])
    g = float(m["g"][i])
    z = a_lim(rpm, g)
    row = dict(rpm=rpm, g=round(g, 2), zoa_mm=z * 1e3, td300_over_zoa=float(m["a_step_free"][i] / m["a_lim"][i]),
               onset_over_zoa=onset(rpm, g, z, "step_free", 0.99, 1.06))
    out["c1_step_free"].append(row)
    print(row, flush=True)

# C4: the C1/C2 gain plus g2 = 2000 at 2000 rpm, face entry with the 9 V limit
E = np.load(os.path.join(RES, "c4_entry.npz"))
des = list(E["designs"])
j, i = des.index("paper+2000"), int(np.nonzero(E["rpm"] == 2000.0)[0][0])
g1, g2 = float(E["g1"][i, j]), float(E["g2"][j])
P.set_damping(float(E["zeta"]))
z = c4.a_lim_ctrl(2000.0, g1, g2)
out["c4_face_entry"] = dict(rpm=2000, g1=round(g1, 2), g2=g2, a_lim_mm=z * 1e3,
                            td300_over_a_lim=float(E["a_entry"][i, j, list(E["names"]).index("edge_clip")] / E["a_lim"][i, j]),
                            onset_over_a_lim=onset(2000.0, g1, z, "edge_clip", 0.95, 1.05, g2=g2))
print(out["c4_face_entry"], flush=True)
json.dump(out, open(os.path.join(HERE, "check10_forced_resonance_long_runs.json"), "w"), indent=1)
