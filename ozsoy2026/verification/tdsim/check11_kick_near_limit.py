"""Critical kick at 0.999 of the linear limit (audit check of README 11.5).

The C1 basin campaign stops at 0.98 a_lim.  README 11.5 states that the basin
tends to a finite size at the limit, not to zero; the review's runs at 0.999
behind that statement used the draft's 4-pass chip memory and one phase.  Here
the same quantity with the current C1 protocol (full surface memory, upward
kick search on DV_GRID, 9 V, the C1 gain of results/c1_basin.npz) at phases
0, 4, 8 and 12 of 16.
Writes check11_kick_near_limit.json (a few minutes on four cores).
"""
import json
import os
import sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import params as P                                      # noqa: E402
from zoa import a_lim                                   # noqa: E402
from c1_unsafe_zone import _critical_kick, NREG         # noqa: E402

FRAC, PHASES, SPEEDS = 0.999, (0, 4, 8, 12), (1090.0, 2500.0, 2870.0)


def job(args):
    return args[0], args[4], _critical_kick(args)


if __name__ == "__main__":
    import tdsim                                        # noqa: F401  build the simulator before the workers start
    B = np.load(os.path.join(HERE, "..", "..", "results", "c1_basin.npz"))
    zeta = float(B["zeta"])
    P.set_damping(zeta)
    jobs, info = [], {}
    for n in SPEEDS:
        i = int(np.nonzero((B["rpm"] == n) & (B["vmax"] > 0))[0][0])
        g = float(B["g"][i])
        a = FRAC * a_lim(n, g)
        info[n] = dict(g=g, a_mm=a * 1e3)
        jobs += [(n, a, g, 9.0, p, zeta, 0.0) for p in PHASES]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    out = dict(frac=FRAC, phases=list(PHASES), nreg=NREG, by_rpm={})
    for n in SPEEDS:
        k = [r[2] * 1e3 for r in res if r[0] == n]
        out["by_rpm"][f"{n:.0f}"] = dict(info[n], kick_mmps=k, min=min(k), max=max(k))
        print(f"{n:.0f} rpm, {FRAC} a_lim: critical kick {min(k):.2f}-{max(k):.2f} mm/s over phases {PHASES}",
              flush=True)
    json.dump(out, open(os.path.join(HERE, "check11_kick_near_limit.json"), "w"), indent=1)
