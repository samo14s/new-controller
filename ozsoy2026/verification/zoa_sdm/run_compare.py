"""ZOA (src/zoa.py) vs independent lobe-sweep ZOA vs first-order SDM at 12
spindle speeds, 4 DVF gains, 2 damping ratios.  Writes compare_12speeds.csv."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import sys
import time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import model as M
import zoa_lobes as L
import sdm

RPMS = np.array([500, 700, 900, 1100, 1300, 1500, 1700, 1950, 2200, 2500, 2800, 3000.0])
GAINS = (0.0, 300.0, 800.0, 1500.0)
ZETAS = (0.0134, 0.0145)


def src_zoa(zeta, g, rpms):
    sys.path.insert(0, SRC)
    import params as P
    import zoa
    P.set_damping(zeta)
    gcl = zoa._gcl_grid(g)
    out = [zoa.a_lim(n, g, gcl=gcl, return_freq=True) for n in rpms]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def job(args):
    zeta, g = args
    t0 = time.time()
    a_src, f_src = src_zoa(zeta, g, RPMS)
    a_lob, f_lob, k_lob = L.envelope(RPMS, g, zeta)
    a_ide, f_ide, _ = L.envelope(RPMS, g, zeta, ideal=True)
    rows = []
    for i, n in enumerate(RPMS):
        st = sdm.boundary_richardson(n, g, zeta)
        hx = sdm.boundary_richardson(n, g, zeta, helix=True)
        idl = sdm.boundary_richardson(n, g, zeta, ideal=True)
        avg = sdm.boundary_richardson(n, g, zeta, helix="mean")
        mu = st["mu"]
        kind = "flip" if abs(mu.imag) < 1e-3 and mu.real < 0 else ("fold" if abs(mu.imag) < 1e-3 else "Hopf")
        rows.append(dict(
            zeta=zeta, g=g, rpm=n,
            a_zoa_src_mm=a_src[i] * 1e3, fc_zoa_src_Hz=f_src[i],
            a_zoa_lobes_mm=a_lob[i] * 1e3, fc_zoa_lobes_Hz=f_lob[i], lobe_k=int(k_lob[i]),
            a_sdm_mm=st["a_R"] * 1e3, a_sdm_m_mm=st["a_m"] * 1e3, a_sdm_2m_mm=st["a_2m"] * 1e3,
            sdm_m=st["m"], sdm_mu_abs=abs(mu), sdm_mu_arg_deg=np.degrees(np.angle(mu)),
            sdm_type=kind, fc_sdm_Hz=sdm.multiplier_freq(mu, n, f_lob[i]),
            a_sdm_helix_mm=hx["a_R"] * 1e3, a_sdm_meanH_mm=avg["a_R"] * 1e3,
            fc_sdm_helix_Hz=sdm.multiplier_freq(hx["mu"], n, f_lob[i]),
            a_zoa_ideal_act_mm=a_ide[i] * 1e3, a_sdm_ideal_act_mm=idl["a_R"] * 1e3,
            rel_src_vs_lobes=a_src[i] / a_lob[i] - 1,
            rel_src_vs_sdm=a_src[i] / st["a_R"] - 1,
            rel_src_vs_sdm_meanH=a_src[i] / avg["a_R"] - 1,
            rel_src_vs_sdm_helix=a_src[i] / hx["a_R"] - 1,
            rel_sdm_helix_vs_straight=hx["a_R"] / st["a_R"] - 1,
            sdm_richardson_corr=st["a_R"] / st["a_2m"] - 1,
            rel_zoa_full_vs_ideal_act=a_lob[i] / a_ide[i] - 1,
            rel_sdm_full_vs_ideal_act=st["a_R"] / idl["a_R"] - 1,
        ))
    print(f"zeta={zeta} g={g} done in {time.time()-t0:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    combos = [(z, g) for z in ZETAS for g in GAINS]
    with Pool(2) as pool:
        res = pool.map(job, combos)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, "compare_12speeds.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote compare_12speeds.csv")
