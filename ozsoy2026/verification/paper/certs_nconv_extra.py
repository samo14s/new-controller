"""Extra node counts for the N-convergence of the C2 certified kick at a point where the 2 N solve failed.

At 2500 rpm and 0.98 a_lim (g = 1500 V s/m) the C2 certificate exists with N = 20 nodes (1.14 mm/s)
but no candidate of the N = 40 solve passed the independent check (certs_nconv_a.json).  This script
certifies the same point with intermediate node counts, to tell a trend from a numerical failure.
Same function and settings as certs_checks.py nconv (c2.certify_kick, defaults).

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/certs_nconv_extra.py [rpm frac N1,N2,...]
Writes verification/paper/certs_nconv_extra.json (after every solve; resumable).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")

import params as P  # noqa: E402

P.set_damping(float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"]))

import c2_certificate as c2  # noqa: E402
from zoa import a_lim  # noqa: E402

G = 1500.0
OUT = os.path.join(HERE, "certs_nconv_extra.json")


def main():
    rpm = float(sys.argv[1]) if len(sys.argv) > 1 else 2500.0
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.98
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [24, 30]
    out = json.load(open(OUT)) if os.path.exists(OUT) else dict(note=__doc__.split("\n\n")[0], points={})
    al = a_lim(rpm, G)
    key = f"{rpm:.0f}_{frac}"
    rec = out["points"].setdefault(key, dict(rpm=rpm, frac=frac, a_mm=frac * al * 1e3, runs={}))
    for N in Ns:
        if str(N) in rec["runs"]:
            continue
        t0 = time.time()
        beta, note = c2.certify_kick(rpm, frac * al, G, N=N)
        rec["runs"][str(N)] = dict(beta_mmps=beta * 1e3, note=note, seconds=round(time.time() - t0, 1))
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        print(f"{key} N {N}: {beta * 1e3:.4f} mm/s ({time.time() - t0:.0f} s) {note}", flush=True)


if __name__ == "__main__":
    main()
