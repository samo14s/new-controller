"""Check 3c: re-test every D1 boundary of check3 with the final detector.

check3_stability.json was produced before detector.classify got its last
test (max |x(t)-x(t-T)| over the last 20 % < 0.25 fz), which catches limit
cycles that are already saturated during the engagement transient.  For each
case and each D1 configuration the run is repeated at 0.99 and 1.01 times the
reported boundary; the final detector must call the first stable and the
second unstable.  Writes check3c_recheck.json.
"""
import json
import common
import numpy as np
import detector as Dt

d = json.load(open(common.out("check3_stability.json")))
CONF = [("a_td512_n300", 512, 300), ("a_td512_n1000", 512, 1000),
        ("a_td2048_n1000", 2048, 1000), ("a_td8192_n300", 8192, 300)]
out, bad = [], []
for r in d["cases"]:
    for key, spr, nrev in CONF:
        ab = r[key]
        if ab is None or (isinstance(ab, float) and np.isnan(ab)):
            continue
        lo = Dt.run_and_classify(r["rpm"], 0.99 * ab, r["g"], nrev, spr)
        hi = Dt.run_and_classify(r["rpm"], 1.01 * ab, r["g"], nrev, spr)
        ok = lo["stable_decay"] and not hi["stable_decay"]
        row = dict(rpm=r["rpm"], g=r["g"], config=key, a_mm=ab * 1e3, stable_at_0p99=lo["stable_decay"],
                   stable_at_1p01=hi["stable_decay"], consistent=ok,
                   dx_end_over_fz_1p01=hi["dx_end_over_fz"])
        out.append(row)
        if not ok:
            bad.append(row)
        print(row, flush=True)
res = dict(n_checked=len(out), n_inconsistent=len(bad), inconsistent=bad, rows=out)
print(json.dumps(dict(n_checked=len(out), n_inconsistent=len(bad)), indent=1))
with open(common.out("check3c_recheck.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
