"""Context for ambiguous text numbers: global minimum of the ZOA SLD for a few constant gains.

Not a reproduction result - only used to interpret 'worst case 1.2 -> 3.6 mm' and the
single-gain minima of Fig. 11.  Uses the shared model in ../../src (zeta = 1.34 %).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import zoa

rpm = np.arange(500, 3001, 5.0)
out = {}
for g in (0.0, 237.0, 257.0, 300.0):
    a = zoa.sld(rpm, g) * 1e3
    j = int(np.argmin(a))
    out[str(g)] = dict(min_mm=round(float(a[j]), 3), at_rpm=float(rpm[j]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_minima.json"), "w"), indent=1)
