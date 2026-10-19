"""C1: saturation-induced unsafe zones in actively controlled milling.

With the actuator command clipped (|V| <= F_max/K_a), a cut that the linear
analysis and the unclipped simulation call stable can still chatter.  Entry
excites the chatter mode; once the command saturates, the effective DVF gain
drops below the gain that the depth needs, and the mode grows.  Enter the same
cut softly and it settles onto the forced periodic orbit.  The cut is
bistable, and this script maps that zone and its basin of attraction.

  map        per speed, with the gain map of run_feasibility_map.py, the
             depth limit for five entries: step (clipped and unclipped),
             face entry, depth ramp and feed ramp (clipped).  Also stores
             a_lim(g) and a_lim(0).
  gainplane  the same limits as a function of the gain, at two speeds
  basin      critical velocity kick on the settled cut versus depth.  The
             cut is reached by a face entry and kicked at 16 phases per
             tooth period.  Clipped, with the unclipped cut as a control.
  df         describing-function estimate of the critical chatter
             amplitude at every speed, and the step-entry limit it predicts
             from the entry transient of the unclipped cut
    python c1_unsafe_zone.py map --workers 4
Writes ../results/c1_<command>.npz (and .csv for map)
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VMAX = P.F_MAX / P.GA_GAIN
ENTRIES = {
    "step_clip": (VMAX, {}),
    "step_free": (0.0, {}),
    "edge_clip": (VMAX, {"edge": True}),
    "ramp_clip": (VMAX, {"ramp_a": 10.0}),
    "feed_clip": (VMAX, {"ramp_f": 10.0, "f0": 0.0}),
}
# depth ladder of every boundary: 8 % steps from a quarter of a_lim, and
# a_hi = 1.05 a_lim so that the unclipped limit (0.4-2.2 % above a_lim) is seen
LADDER = dict(growth=1.08, lo=0.25, hi=1.05)
BASIN_RPM = (1090, 1420, 2500, 2870)
BASIN_FRAC = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
FREE_FRAC = (0.9, 0.98)
N_PHASE = 16
KICK_REV, WINDOW = 100, 300
DV_LO, DV_HI, N_BISECT = 1e-4, 1.0, 9


def gain_map(tag="g1500"):
    d = np.load(os.path.join(RES, f"feas_{tag}.npz"))
    return d["rpm"], d["g_best"], float(d["zeta"])


# ----------------------------------------------------------------- map ----
def _limit(args):
    rpm, g, name, zeta = args
    P.set_damping(zeta)
    from tdboundary import boundary
    from zoa import a_lim
    al = a_lim(rpm, g)
    if not np.isfinite(al):
        return np.nan
    vmax, entry = ENTRIES[name]
    if g == 0:
        vmax = 0.0
    return boundary(rpm, g, vmax=vmax, a_lo=LADDER["lo"] * al, a_hi=LADDER["hi"] * al,
                    growth=LADDER["growth"], **entry) * 1e3


def run_map(a):
    from zoa import a_lim
    rpm, g, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    names = list(ENTRIES)
    jobs = [(float(n), float(gg), k, zeta) for n, gg in zip(rpm, g) for k in names]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_limit, jobs, chunksize=4)).reshape(len(rpm), len(names))
    data = dict(rpm=rpm, g=g, zeta=zeta,
                a_lim=np.array([a_lim(n, gg) for n, gg in zip(rpm, g)]) * 1e3,
                a_lim0=np.array([a_lim(n, 0.0) for n in rpm]) * 1e3)
    data.update({f"a_{k}": lim[:, i] for i, k in enumerate(names)})
    np.savez_compressed(os.path.join(RES, "c1_map.npz"), **data)
    keys = ["rpm", "g", "a_lim", "a_lim0"] + [f"a_{k}" for k in names]
    np.savetxt(os.path.join(RES, "c1_map.csv"), np.column_stack([data[k] for k in keys]),
               delimiter=",", header=",".join(keys), comments="", fmt="%.4f")
    gap = data["a_step_free"] - data["a_step_clip"]
    print(f"step entry, clipped below unclipped by > 1 % at "
          f"{np.sum(gap > 0.01 * data['a_step_free'])}/{len(rpm)} speeds")
    for k in names:
        r = data[f"a_{k}"] / data["a_lim"]
        print(f"  {k:10s} a/a_lim: min {np.nanmin(r):.3f} median {np.nanmedian(r):.3f}")


# ----------------------------------------------------------- gainplane ----
def run_gainplane(a):
    from zoa import a_lim
    rpm_map, g_map, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    speeds, gains = (1090.0, 2500.0), np.arange(0.0, 3001.0, 150.0)
    names = ["step_clip", "step_free", "edge_clip"]
    jobs = [(n, gg, k, zeta) for n in speeds for gg in gains for k in names]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_limit, jobs, chunksize=2)).reshape(len(speeds), len(gains), len(names))
    data = dict(rpm=np.array(speeds), gains=gains, zeta=zeta,
                g_map=np.interp(speeds, rpm_map, g_map),
                a_lim=np.array([[a_lim(n, gg) for gg in gains] for n in speeds]) * 1e3)
    data.update({f"a_{k}": lim[:, :, i] for i, k in enumerate(names)})
    np.savez_compressed(os.path.join(RES, "c1_gainplane.npz"), **data)
    for i, n in enumerate(speeds):
        j = int(np.nanargmax(data["a_step_clip"][i]))
        print(f"{n:.0f} rpm: step-entry clipped limit peaks at g = {gains[j]:.0f} "
              f"({data['a_step_clip'][i, j]:.2f} mm); map gain {data['g_map'][i]:.0f}")


# --------------------------------------------------------------- basin ----
def _settled_stable(rpm, a, g, vmax):
    from tdsim import simulate
    from tdboundary import classify_run
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=KICK_REV + WINDOW, edge=True)
    return classify_run(r["x"][KICK_REV * P.STEPS_PER_REV:], P.STEPS_PER_REV, r["diverged"])["stable"]


def _kick_chatters(rpm, a, g, vmax, ks, dv):
    from tdsim import simulate
    from tdboundary import classify_run
    spr = P.STEPS_PER_REV
    nd = spr // P.N_TEETH
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=KICK_REV + WINDOW, edge=True,
                 kick_step=ks, kick_dv=dv)
    st = (ks // nd + 1) * nd
    return not classify_run(r["x"][st:], spr, r["diverged"])["stable"]


def _critical_kick(args):
    """Smallest kick (m/s) that turns the settled cut into chatter, by
    bisection in log(dv); inf if DV_HI does not, nan if the unkicked cut
    does not settle."""
    rpm, a, g, vmax, phase, zeta = args
    P.set_damping(zeta)
    nd = P.STEPS_PER_REV // P.N_TEETH
    ks = KICK_REV * P.STEPS_PER_REV + (phase * nd) // N_PHASE
    if _kick_chatters(rpm, a, g, vmax, ks, DV_LO):
        return np.nan
    if not _kick_chatters(rpm, a, g, vmax, ks, DV_HI):
        return np.inf
    lo, hi = np.log(DV_LO), np.log(DV_HI)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if _kick_chatters(rpm, a, g, vmax, ks, np.exp(mid)):
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def run_basin(a):
    from zoa import a_lim
    rpm_map, g_map, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    cases = []
    for n in BASIN_RPM:
        g = float(np.interp(n, rpm_map, g_map))
        al = a_lim(n, g)
        for f in BASIN_FRAC:
            cases.append((n, g, al, f, VMAX))
        for f in FREE_FRAC:
            cases.append((n, g, al, f, 0.0))
    jobs = [(float(n), f * al, g, v, p, zeta) for (n, g, al, f, v) in cases for p in range(N_PHASE)]
    with Pool(a.workers) as pool:
        dv = np.array(pool.map(_critical_kick, jobs, chunksize=1)).reshape(len(cases), N_PHASE)
    arr = np.array([(n, g, al * 1e3, f, v) for (n, g, al, f, v) in cases])
    np.savez_compressed(os.path.join(RES, "c1_basin.npz"), rpm=arr[:, 0], g=arr[:, 1],
                        a_lim=arr[:, 2], frac=arr[:, 3], vmax=arr[:, 4], dv=dv, zeta=zeta,
                        n_phase=N_PHASE, kick_rev=KICK_REV, window=WINDOW)
    for row, d in zip(arr, dv):
        lab = "clip" if row[4] > 0 else "free"
        fin = d[np.isfinite(d)]
        print(f"{row[0]:.0f} rpm {row[3]:.2f} a_lim {lab}: dv* min {np.nanmin(d)*1e3 if fin.size else np.nan:.2f} "
              f"median {np.nanmedian(d)*1e3:.2f} mm/s  (inf {np.isinf(d).sum()}, nan {np.isnan(d).sum()})")


# ------------------------------------------------------------------ df ----
DF_FRAC = np.linspace(0.05, 1.0, 39)
_TH = (np.arange(128) + 0.5) / 128 * 2 * np.pi


def df_gain(A, b, vmax=VMAX):
    """Describing-function gain on a chatter component of amplitude A (V) in
    the presence of the forced command b(t) (V), averaged over the forced
    period (dual-input DF for incommensurate signals)."""
    A = np.atleast_1d(A)
    s = np.sin(_TH)
    u = np.asarray(b)[:, None, None] + A[None, :, None] * s[None, None, :]
    return 2 * (np.clip(u, -vmax, vmax) * s).mean(axis=(0, 2)) / A


def critical_amplitude(Nstar, b, vmax=VMAX):
    """Smallest amplitude (V) at which df_gain falls to Nstar (0 or inf at the ends)."""
    if Nstar <= 0:
        return np.inf
    A = np.logspace(-3, 3, 1201) * vmax
    Nv = df_gain(A, b, vmax)
    i = np.where(Nv <= Nstar)[0]
    if not i.size:
        return np.inf
    if i[0] == 0:
        return 0.0
    k = i[0]
    return float(np.exp(np.interp(Nstar, [Nv[k], Nv[k - 1]], [np.log(A[k]), np.log(A[k - 1])])))


def effective_gain_needed(al_of_g, gains, a):
    """Largest g' <= g with a_lim(g') = a, scanning down from the top (0 if none)."""
    if a >= al_of_g[-1]:
        return gains[-1]
    for i in range(len(gains) - 1, 0, -1):
        if al_of_g[i - 1] < a <= al_of_g[i]:
            return float(np.interp(a, [al_of_g[i - 1], al_of_g[i]], [gains[i - 1], gains[i]]))
    return 0.0


def _df_speed(args):
    rpm, g, zeta = args
    P.set_damping(zeta)
    from zoa import a_lim
    from tdsim import simulate
    from feasibility import forced_velocity
    gains = np.linspace(0.0, g, 41)
    alg = np.array([a_lim(rpm, x) for x in gains])
    al = alg[-1]
    spr = P.STEPS_PER_REV
    nd = spr // P.N_TEETH
    out = np.full((len(DF_FRAC), 6), np.nan)
    for i, f in enumerate(DF_FRAC):
        a = f * al
        ge = effective_gain_needed(alg, gains, a)
        vf = forced_velocity(rpm, g, [a], n_t=nd)[0]           # one tooth period, at the steps
        b = -g * vf
        A1 = critical_amplitude(ge / g, np.zeros(1))
        A2 = critical_amplitude(ge / g, b)
        # entry transient of the unclipped step entry, forced orbit removed
        r = simulate(rpm, a, g, store=True, nrev=30)
        vtr = r["v"] - np.tile(vf, 30 * P.N_TEETH + 1)[: len(r["v"])]
        Atr = g * float(np.abs(vtr).max())
        out[i] = (a * 1e3, ge, A1, A2, Atr, g * np.abs(vf).max())
    return out


def run_df(a):
    rpm, g, zeta = gain_map(a.tag)
    with Pool(a.workers) as pool:
        res = np.array(pool.map(_df_speed, [(float(n), float(gg), zeta) for n, gg in zip(rpm, g)]))
    a_mm, ge, A1, A2, Atr, bpk = (res[:, :, k] for k in range(6))

    def first_cross(thr):
        out = np.full(len(rpm), np.nan)
        for j in range(len(rpm)):
            k = np.where(Atr[j] >= thr[j])[0]
            out[j] = a_mm[j, k[0]] if k.size else a_mm[j, -1]
        return out

    pred1, pred2 = first_cross(A1), first_cross(A2)
    np.savez_compressed(os.path.join(RES, "c1_df.npz"), rpm=rpm, g=g, zeta=zeta, frac=DF_FRAC,
                        a=a_mm, g_eff=ge, A_single=A1, A_dual=A2, A_entry=Atr, b_peak=bpk,
                        a_pred_single=pred1, a_pred_dual=pred2)
    m = os.path.join(RES, "c1_map.npz")
    if os.path.exists(m):
        t = np.load(m)["a_step_clip"]
        for lab, p in (("single", pred1), ("dual", pred2)):
            e = (p - t) / t
            print(f"DF {lab}-input predicted step-entry limit vs simulation: median {np.nanmedian(e)*100:+.1f} %, "
                  f"|err| < 10 % at {np.sum(np.abs(e) < 0.10)}/{len(t)} speeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["map", "gainplane", "basin", "df"])
    ap.add_argument("--tag", default="g1500")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dict(map=run_map, gainplane=run_gainplane, basin=run_basin, df=run_df)[a.command](a)


if __name__ == "__main__":
    main()
