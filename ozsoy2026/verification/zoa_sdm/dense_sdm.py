"""SDM stability boundary on a dense speed grid (zeta = 1.34 %), straight and
helical cutter, g = 0 (25-rpm step) and g = 1500 V s/m (50-rpm step).
Writes dense_sdm.csv; used by make_figures.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sdm
import zoa_lobes as L

Z = 0.0134


def job(args):
    n, g, mode = args
    r = sdm.boundary_richardson(n, g, Z, helix=(mode == "helix"))
    fz = L.envelope([n], g, Z)[1][0]
    mu = r["mu"]
    kind = "flip" if abs(mu.imag) < 1e-3 and mu.real < 0 else "Hopf"
    return dict(g=g, rpm=n, mode=mode, a_mm=r["a_R"] * 1e3, a_2m_mm=r["a_2m"] * 1e3, m=r["m"],
                type=kind, fc_Hz=sdm.multiplier_freq(mu, n, fz))


if __name__ == "__main__":
    jobs = [(float(n), 0.0, md) for n in np.arange(500, 3001, 25) for md in ("straight", "helix")]
    jobs += [(float(n), 1500.0, md) for n in np.arange(500, 3001, 50) for md in ("straight", "helix")]
    with Pool(2) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    with open(os.path.join(HERE, "dense_sdm.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("done", len(rows))
