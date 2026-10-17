"""Check 3: time-domain stability boundary of src/tdsim.c vs src/zoa.a_lim.

For each spindle speed and DVF gain the axial depth is bisected with the
once-per-tooth chatter detector (detector.py).  Reported per case:

  a_zoa        src/zoa.a_lim (zero-order approximation, the paper's Step 1)
  a_sd         linear stability of the continuous periodic model with helix
               (semi-discretisation, sd.py, m = 240 and Richardson-extrapolated)
  a_sd_mean    same, H(t) replaced by its mean -> must reproduce a_zoa
  a_sd_nohelix periodic model with helix angle 0 (effect of the helix)
  a_zoh{spr}   ZOA with the extra delay dt/2 of the step-held cutting force,
               Gcl(iw) -> Gcl(iw) exp(-i w dt/2), dt = 60/(rpm spr)
  a_td{spr}_n{nrev}     tdsim boundary, criterion D1 (deviation from the
               tooth-periodic state must die out; bisection, rtol 1e-3)
  a_tdD2_{spr}_n{nrev}  tdsim boundary, criterion D2 (no growth beyond the
               engagement transient), see detector.py
  fc_*         chatter frequencies (ZOA, and FFT of x(t)-x(t-T) in the TD run
               just above the TD boundary)

Writes check3_stability.csv / .json and check3_sld.png.
"""
import json
import time
import common
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import zoa
import structure
import detector as Dt
import sd

RPMS = [500, 650, 800, 1100, 1300, 1600, 1950, 2300, 2600, 3000]
GAINS = [0.0, 600.0, 1200.0]
FINE_SPR_RPMS = (500, 1300, 2600)        # spr = 8192 only here (cost)


def a_zoa_zoh(rpm, g, spr):
    tau = 0.5 * 60.0 / (rpm * spr)
    gcl = structure.Gcl(1j * zoa.W_GRID, g) * np.exp(-1j * zoa.W_GRID * tau)
    return float(zoa.a_lim(rpm, g, gcl=gcl))


def td_boundary(rpm, g, nrev, spr, guess, rtol=1e-3, criterion="stable_decay"):
    """Bracket around guess with 3 % steps, then bisect; check 0.5x, 0.75x stable."""
    cls = lambda a: Dt.run_and_classify(rpm, a, g, nrev, spr)[criterion]
    a = guess
    if cls(a):
        lo = a
        while True:
            a *= 1.03
            if not cls(a):
                hi = a
                break
    else:
        hi = a
        while True:
            a /= 1.03
            if cls(a):
                lo = a
                break
    while (hi - lo) / lo > rtol:
        mid = 0.5 * (lo + hi)
        if cls(mid):
            lo = mid
        else:
            hi = mid
    ab = 0.5 * (lo + hi)
    island = not (cls(0.5 * ab) and cls(0.75 * ab))
    return ab, island


def td_chatter_freq(rpm, g, a, nrev=300, spr=P.STEPS_PER_REV):
    r = Dt.run_and_classify(rpm, a, g, nrev, spr)["run"]
    nd = spr // P.N_TEETH
    x = r["x"]
    e = x[nd:] - x[:-nd]                        # removes the tooth-periodic part
    dt = 60.0 / (rpm * spr)
    f, S = Dt.chatter_frequency(e, dt, frac=0.5)
    ok = f > 2.0
    return float(f[ok][np.argmax(S[ok])])


def sd_freq(rpm, a, g, fc_hint, m=240):
    mu = sd.critical_multiplier(rpm, a, g, m=m)
    ft = P.N_TEETH * rpm / 60.0
    frac = np.angle(mu) / (2 * np.pi)
    cands = ft * (np.arange(-1, 20)[:, None] + np.array([frac, -frac])[None, :])
    cands = cands[cands > 0]
    return float(cands[np.argmin(np.abs(cands - fc_hint))]), complex(mu)


def one_case(args):
    rpm, g = args
    t0 = time.time()
    row = dict(rpm=rpm, g=g)
    az, fz_ = zoa.a_lim(rpm, g, return_freq=True)
    row["a_zoa"], row["fc_zoa"] = float(az), float(fz_)
    # semi-discretisation references
    a120, _ = sd.boundary(rpm, g, az, m=120, check_below=False)
    a240, isl = sd.boundary(rpm, g, az, m=240)
    row["a_sd_m120"], row["a_sd_m240"] = a120, a240
    row["a_sd"] = a240 + (a240 - a120) / 3.0              # O(1/m^2) extrapolation
    row["sd_island_below"] = bool(isl)
    m120, _ = sd.boundary(rpm, g, az, m=120, mode="mean", check_below=False)
    m240, _ = sd.boundary(rpm, g, az, m=240, mode="mean", check_below=False)
    row["a_sd_mean"] = m240 + (m240 - m120) / 3.0
    n120, _ = sd.boundary(rpm, g, az, m=120, helix=0.0, check_below=False)
    n240, _ = sd.boundary(rpm, g, az, m=240, helix=0.0, check_below=False)
    row["a_sd_nohelix"] = n240 + (n240 - n120) / 3.0
    row["fc_sd"], mu = sd_freq(rpm, a240, g, row["fc_zoa"])
    row["sd_crit_mult_angle_deg"] = float(np.degrees(np.angle(mu)))
    # ZOA with the step-hold delay
    for spr in (512, 2048, 8192):
        row[f"a_zoh{spr}"] = a_zoa_zoh(rpm, g, spr)
    # time domain, criterion D1 (main) and D2 (growth)
    for spr, nrev in ((512, 300), (512, 1000), (2048, 1000)):
        ab, isl = td_boundary(rpm, g, nrev, spr, row[f"a_zoh{spr}"])
        row[f"a_td{spr}_n{nrev}"] = ab
        row[f"td{spr}_n{nrev}_island"] = isl
    if rpm in FINE_SPR_RPMS:
        ab, isl = td_boundary(rpm, g, 300, 8192, row["a_zoh8192"])
        row["a_td8192_n300"] = ab
    else:
        row["a_td8192_n300"] = np.nan
    for spr, nrev in ((512, 300), (2048, 300)):
        ab, isl = td_boundary(rpm, g, nrev, spr, row[f"a_zoh{spr}"],
                              criterion="stable_growth")
        row[f"a_tdD2_{spr}_n{nrev}"] = ab
        row[f"tdD2_{spr}_n{nrev}_island"] = isl
    row["fc_td512"] = td_chatter_freq(rpm, g, 1.02 * row["a_td512_n1000"])
    row["seconds"] = time.time() - t0
    print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items()},
          flush=True)
    return row


if __name__ == "__main__":
    cases = [(r, g) for g in GAINS for r in RPMS]
    with Pool(2) as pool:
        rows = pool.map(one_case, cases, chunksize=1)

    keys = list(rows[0].keys())
    with open(common.out("check3_stability.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def stats(num, den):
        v = np.array([r[num] / r[den] - 1 for r in rows if np.isfinite(r[num])])
        return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))

    summary = {
        "sd_mean_vs_zoa": stats("a_sd_mean", "a_zoa"),
        "sd_periodic_vs_zoa": stats("a_sd", "a_zoa"),
        "sd_nohelix_vs_zoa": stats("a_sd_nohelix", "a_zoa"),
        "zoh512_vs_zoa": stats("a_zoh512", "a_zoa"),
        "D1_td512_n1000_vs_zoa": stats("a_td512_n1000", "a_zoa"),
        "D1_td512_n1000_vs_zoh512": stats("a_td512_n1000", "a_zoh512"),
        "D1_td2048_n1000_vs_zoa": stats("a_td2048_n1000", "a_zoa"),
        "D1_td2048_n1000_vs_zoh2048": stats("a_td2048_n1000", "a_zoh2048"),
        "D1_td8192_n300_vs_zoa": stats("a_td8192_n300", "a_zoa"),
        "D1_td8192_n300_vs_sd": stats("a_td8192_n300", "a_sd"),
        "D1_td512_n300_vs_n1000": stats("a_td512_n300", "a_td512_n1000"),
        "D2_td512_n300_vs_zoa": stats("a_tdD2_512_n300", "a_zoa"),
        "D2_td512_n300_vs_D1_n1000": stats("a_tdD2_512_n300", "a_td512_n1000"),
        "D2_td2048_n300_vs_zoa": stats("a_tdD2_2048_n300", "a_zoa"),
        "fc_td512_vs_fc_zoa": stats("fc_td512", "fc_zoa"),
        "any_td_island": any(r[k] for r in rows for k in r if k.endswith("_island")),
    }
    print(json.dumps(summary, indent=1))
    with open(common.out("check3_stability.json"), "w") as f:
        json.dump(dict(cases=rows, summary=summary), f, indent=1, default=float)

    # ---- figure -----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    grid = np.arange(500, 3001, 5)
    for ax, g in zip(axs, GAINS):
        ax.plot(grid, zoa.sld(grid, g) * 1e3, "k-", lw=1.2, label="ZOA (src/zoa.py)")
        tau_curve = [a_zoa_zoh(n, g, 512) * 1e3 for n in grid]
        ax.plot(grid, tau_curve, "C1--", lw=1, label="ZOA + dt/2 hold delay, spr 512")
        rr = [r for r in rows if r["g"] == g]
        ax.plot([r["rpm"] for r in rr], [r["a_sd"] * 1e3 for r in rr], "ks", mfc="none",
                ms=8, label="semi-discretisation (periodic, helix)")
        ax.plot([r["rpm"] for r in rr], [r["a_td512_n1000"] * 1e3 for r in rr], "C1o",
                label="tdsim spr 512 (D1, 1000 rev)")
        ax.plot([r["rpm"] for r in rr], [r["a_tdD2_512_n300"] * 1e3 for r in rr], "C3x",
                label="tdsim spr 512 (D2 growth, 300 rev)")
        ax.plot([r["rpm"] for r in rr], [r["a_td2048_n1000"] * 1e3 for r in rr], "C0^",
                label="tdsim spr 2048 (D1, 1000 rev)")
        ax.plot([r["rpm"] for r in rr], [r["a_td8192_n300"] * 1e3 for r in rr], "C2v",
                label="tdsim spr 8192 (D1, 300 rev)")
        top = max(r["a_zoa"] for r in rr) * 1e3
        ax.set_ylim(0, min(1.1 * top, 45))
        ax.set_title(f"g = {g:.0f} V s/m", fontsize=10)
        ax.set_xlabel("spindle speed (rpm)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("limiting axial depth (mm)")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(common.out("check3_sld.png"), dpi=130)
