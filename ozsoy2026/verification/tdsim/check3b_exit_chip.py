"""Check 3b: the linear core of tdsim without the exit nonlinearity.

In down milling the static chip fz sin(phi) vanishes at the exit angle pi, so
the h <= 0 switch acts at any vibration amplitude there; this saturates chatter
just above the linear limit (check3).  Here the exit is moved to pi - 0.35 rad
(static exit chip 0.34 fz = 17 um), so vibrations smaller than that never
switch the force off and the simulated loop is exactly linear near the
boundary.  The tdsim C routine is called directly with the modified exit angle
and its stability boundary (criteria D1 and D2) is compared with:
  * the semi-discretisation of the continuous periodic model for the same
    immersion (sd.py, Richardson-extrapolated m = 120/240),
  * the ZOA for the same immersion (= nominal a_lim scaled by alpha ratio),
  * the ZOA with the dt/2 step-hold delay for the same immersion.
Writes check3b_exit_chip.csv / .json.
"""
import json
import common
import numpy as np
import params as P
import tdsim
import zoa
import structure
import detector as Dt
import sd

PHI_OUT = np.pi - 0.35
SPEEDS = [500, 1300, 1950, 2600]
GAINS = [0.0, 600.0, 1200.0]
RATIO = sd.H_mean_zoa(1e-3) / sd.H_mean_numeric(1e-3, PHI_OUT)   # a scales with 1/alpha


def run(rpm, a, g, nrev, spr):
    n = spr * nrev
    bufs = [np.zeros(n + 1) for _ in range(5)]
    summ = np.zeros(5)
    tdsim._lib.simulate(float(rpm), float(a), float(g), P.M, P.C, P.K, P.GA_GAIN,
                        P.GA_A1, P.GA_A0, P.K_TC, P.K_RC, P.FZ, P.D_TOOL, P.N_TEETH,
                        P.HELIX, tdsim.PHI_ST, PHI_OUT, spr, nrev, P.N_LAYERS, 2, 0.0,
                        1, *bufs, summ)
    return Dt.classify(bufs[0], spr, bool(summ[4]))


def boundary(rpm, g, nrev, spr, guess, crit, rtol=1e-3):
    cls = lambda a: run(rpm, a, g, nrev, spr)[crit]
    a = guess
    if cls(a):
        while cls(a * 1.03):          # step up while still stable
            a *= 1.03
            assert a < 0.2
        lo, hi = a, a * 1.03
    else:
        while not cls(a / 1.03):      # step down while still unstable
            a /= 1.03
            assert a > 1e-5
        lo, hi = a / 1.03, a
    while (hi - lo) / lo > rtol:
        mid = 0.5 * (lo + hi)
        if cls(mid):
            lo = mid
        else:
            hi = mid
    ab = 0.5 * (lo + hi)
    assert cls(0.5 * ab) and cls(0.75 * ab), "lower crossing below the bracket"
    return ab


def zoh(rpm, g, spr):
    tau = 0.5 * 60.0 / (rpm * spr)
    gcl = structure.Gcl(1j * zoa.W_GRID, g) * np.exp(-1j * zoa.W_GRID * tau)
    return float(zoa.a_lim(rpm, g, gcl=gcl)) * RATIO


if __name__ != "__main__":
    raise SystemExit("run as a script")

rows = []
for g in GAINS:
    for rpm in SPEEDS:
        row = dict(rpm=rpm, g=g, phi_out=PHI_OUT)
        row["a_zoa"] = float(zoa.a_lim(rpm, g)) * RATIO
        a1, _ = sd.boundary(rpm, g, row["a_zoa"], m=120, phi_out=PHI_OUT, check_below=False)
        a2, isl = sd.boundary(rpm, g, row["a_zoa"], m=240, phi_out=PHI_OUT)
        row["a_sd"] = a2 + (a2 - a1) / 3
        row["sd_island"] = bool(isl)
        for spr in (512, 2048):
            row[f"a_zoh{spr}"] = zoh(rpm, g, spr)
            row[f"a_tdD1_{spr}"] = boundary(rpm, g, 300, spr, row[f"a_zoh{spr}"], "stable_decay")
            row[f"a_tdD2_{spr}"] = boundary(rpm, g, 300, spr, row[f"a_zoh{spr}"], "stable_growth")
        rows.append(row)
        print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)

keys = list(rows[0].keys())
with open(common.out("check3b_exit_chip.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r[k]) for k in keys) + "\n")


def st(a, b):
    v = np.array([r[a] / r[b] - 1 for r in rows])
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))


summary = dict(alpha_ratio=RATIO,
               sd_vs_zoa=st("a_sd", "a_zoa"),
               D1_512_vs_zoh512=st("a_tdD1_512", "a_zoh512"),
               D1_2048_vs_zoh2048=st("a_tdD1_2048", "a_zoh2048"),
               D1_2048_vs_sd=st("a_tdD1_2048", "a_sd"),
               D2_512_vs_D1_512=st("a_tdD2_512", "a_tdD1_512"),
               D2_2048_vs_D1_2048=st("a_tdD2_2048", "a_tdD1_2048"),
               richardson_D1_vs_sd=dict(zip(
                   ["min", "max", "mean"],
                   (lambda v: [float(v.min()), float(v.max()), float(v.mean())])(
                       np.array([(r["a_tdD1_2048"] + (r["a_tdD1_2048"] - r["a_tdD1_512"]) / 3)
                                 / r["a_sd"] - 1 for r in rows])))))
print(json.dumps(summary, indent=1))
with open(common.out("check3b_exit_chip.json"), "w") as f:
    json.dump(dict(rows=rows, summary=summary), f, indent=1, default=float)
