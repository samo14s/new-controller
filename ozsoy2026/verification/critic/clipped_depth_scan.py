"""Clipped DVF (9 V command limit): is stability monotone in depth?

Decay detector D1 (verification/tdsim/detector.py), 2048 steps/rev, 300 rev.
Scans the depth from 0.5 mm upward and reports every stable/chatter switch.
Writes clipped_depth_scan.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tdsim"))
import params as P   # noqa: E402
import detector      # noqa: E402

VMAX = P.F_MAX / P.GA_GAIN
CASES = [(2500, 3000.0), (2500, 1500.0), (3000, 3000.0), (1300, 3000.0), (500, 3000.0), (2500, 543.0)]
depths = np.arange(0.5, 22.01, 0.5)
out = []
for n, g in CASES:
    st = []
    for a in depths:
        c = detector.run_and_classify(n, a * 1e-3, g, 300, spr=2048, vmax=VMAX)
        st.append(bool(c["stable"]))
    st = np.array(st)
    first_chatter = float(depths[np.argmax(~st)]) if (~st).any() else None
    switches = [float(depths[i + 1]) for i in range(len(st) - 1) if st[i] != st[i + 1]]
    out.append(dict(rpm=n, g=g, depths_mm=depths.tolist(), stable=st.tolist(),
                    first_chatter_mm=first_chatter, switch_depths_mm=switches))
    print(n, g, "first chatter", first_chatter, "switches at", switches, flush=True)
with open(os.path.join(HERE, "clipped_depth_scan.json"), "w") as fh:
    json.dump(out, fh, indent=1)
