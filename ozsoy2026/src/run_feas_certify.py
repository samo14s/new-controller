"""Certified operating depth under the reformulated condition.

The steady (forced-response) condition fixes the gain; it cannot see the
entry transient.  With the real actuator, entry saturates the command, and
at large gains this tips cuts that the linear analysis calls stable into
growing chatter (e.g. 2500 rpm, g = 1500: linear limit 17.2 mm, clipped
time-domain limit 12.9 mm).  The depth an operator can rely on is therefore

    a_cert = min(kappa * a_lim(g*),  a_TD,clipped(g*))

where a_TD,clipped is the time-domain limit from rest with |V| <= F_max/K_a,
searched only up to kappa*a_lim (so a_td_clipped = a_cert here), with every
depth on an 8 % ladder up to it tested.
    python run_feas_certify.py --tag g1500
Writes ../results/feas_cert_<tag>.npz and .csv
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VMAX = P.F_MAX / P.GA_GAIN


def work(args):
    rpm, g, zeta, kappa = args
    P.set_damping(zeta)
    from tdboundary import boundary
    from zoa import a_lim
    al = a_lim(rpm, g)
    # every depth up to the operating depth must settle, scanned in 8 % steps
    td = boundary(rpm, g, vmax=VMAX if g > 0 else 0.0, a_lo=0.25 * kappa * al,
                  a_hi=kappa * al, growth=1.08)
    return rpm, g, al * 1e3, kappa * al * 1e3, td * 1e3, min(kappa * al, td) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1500")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = np.load(os.path.join(RES, f"feas_{a.tag}.npz"))
    zeta, kappa = float(d["zeta"]), float(d["kappa"])
    jobs = [(float(n), float(g), zeta, kappa) for n, g in zip(d["rpm"], d["g_best"])]
    with Pool(a.workers) as pool:
        rows = pool.map(work, jobs)
    arr = np.array(rows)
    keys = ["rpm", "g", "a_lim", "a_op_linear", "a_td_clipped", "a_cert"]
    data = {k: arr[:, i] for i, k in enumerate(keys)}
    data.update(zeta=zeta, kappa=kappa, gmax=float(d["gmax"]))
    np.savez_compressed(os.path.join(RES, f"feas_cert_{a.tag}.npz"), **data)
    np.savetxt(os.path.join(RES, f"feas_cert_{a.tag}.csv"), arr, delimiter=",",
               header=",".join(keys), comments="", fmt="%.4f")
    lim = data["a_td_clipped"] < data["a_op_linear"] - 1e-9
    print(f"{a.tag}: clipped limit binds at {lim.sum()}/{len(arr)} speeds; "
          f"a_cert {data['a_cert'].min():.2f}-{data['a_cert'].max():.2f} mm, "
          f"mean {data['a_cert'].mean():.2f}")


if __name__ == "__main__":
    main()
