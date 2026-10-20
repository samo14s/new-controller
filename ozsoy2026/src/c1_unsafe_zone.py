"""C1: saturation-induced bistability in actively controlled milling.

With the actuator command clipped (|V| <= F_max/K_a), a cut that the linear
analysis calls stable can still chatter.  A large enough disturbance makes the
command saturate, which lowers the effective DVF gain below the one the depth
needs, so the chatter mode grows.  Without the limit the same cut always
recovers.  The settled cut and chatter therefore coexist (bistability).  The
step entry is one such disturbance.  It reveals only the top of the bistable
band, where the entry transient exceeds the basin of the stable cut.

All runs remember the last NREG passes (multiple regeneration).  With a
single pass, a tooth that leaves the cut is treated as having removed
material, and clipped chatter runs away instead of settling on a bounded
cycle.

  map        per speed, with the gain map of run_feasibility_map.py, the
             depth limit for four entries: step (clipped and unclipped),
             face entry and feed ramp (clipped); plus a_lim(g) and a_lim(0).
             (The idealised depth ramp of tdsim is left out: a real ramp-in
             shifts the regeneration delay.)
  band       lower edge of the bistable band every 20 rpm: the smallest
             depth at which a kick of DV_HI on the settled cut (4 phases)
             ends in chatter
  basin      critical kick versus depth at four speeds, 16 phases per tooth
             period, clipped and (control) unclipped.  Also the kick-to-mode
             factor: the slow (chatter) mode amplitude that a unit kick puts
             into the linear cut.
  gainplane  the depth limits as a function of the gain, at two speeds
  df         dual-input describing-function threshold at the basin speeds
    python c1_unsafe_zone.py map --workers 4
Writes ../results/c1_<command>.npz (and .csv for map and band)
"""
import argparse
import hashlib
import os
from multiprocessing import Pool
import numpy as np
import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VMAX = P.F_MAX / P.GA_GAIN
SPR = P.STEPS_PER_REV
ND = SPR // P.N_TEETH
NREG = 4
ENTRIES = {
    "step_clip": (VMAX, {"nreg": NREG}),
    "step_free": (0.0, {"nreg": NREG}),
    "edge_clip": (VMAX, {"edge": True, "nreg": NREG}),
    "feed_clip": (VMAX, {"ramp_f": 10.0, "f0": 0.0, "nreg": NREG}),
}
# depth ladder of every boundary: 8 % steps from a quarter of a_lim up to
# 1.05 a_lim, so that a time-domain limit slightly above a_lim is seen
LADDER = dict(growth=1.08, lo=0.25, hi=1.05)
BASIN_RPM = (1090, 1420, 2500, 2870)
BASIN_FRAC = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
FREE_FRAC = (0.3, 0.6, 0.9, 0.98)
N_PHASE = 16
KICK_REV, WINDOW = 100, 300
DV_LO, DV_HI, N_BISECT = 1e-4, 3.0, 9       # m/s; 3 m/s is a 3.7 mm swing, below the 1 cm abort
BAND_STEP, BAND_PHASES, BAND_BISECT = 20, 4, 7
F_HP = 40.0                                  # Hz, high-pass for the kick-to-mode factor


def build_id():
    """Hash of the simulator source the loaded library was built from, stored
    with every result (taken when tdsim is first imported)."""
    from tdsim import SOURCE_SHA
    return SOURCE_SHA[:12]


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
    """--entries recomputes only those columns of an existing c1_map.npz."""
    from zoa import a_lim
    rpm, g, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    names = list(ENTRIES)
    todo = a.entries.split(",") if a.entries else names
    fn = os.path.join(RES, "c1_map.npz")
    jobs = [(float(n), float(gg), k, zeta) for n, gg in zip(rpm, g) for k in todo]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_limit, jobs, chunksize=4)).reshape(len(rpm), len(todo))
    data = dict(np.load(fn)) if a.entries and os.path.exists(fn) else dict(
        rpm=rpm, g=g, zeta=zeta,
        a_lim=np.array([a_lim(n, gg) for n, gg in zip(rpm, g)]) * 1e3,
        a_lim0=np.array([a_lim(n, 0.0) for n in rpm]) * 1e3)
    data.update({f"a_{k}": lim[:, i] for i, k in enumerate(todo)}, nreg=NREG, build=build_id())
    data = {k: v for k, v in data.items()                    # drop retired entries
            if not k.startswith("a_") or k in ("a_lim", "a_lim0") or k[2:] in ENTRIES}
    np.savez_compressed(fn, **data)
    keys = ["rpm", "g", "a_lim", "a_lim0"] + [f"a_{k}" for k in names]
    np.savetxt(os.path.join(RES, "c1_map.csv"), np.column_stack([data[k] for k in keys]),
               delimiter=",", header=",".join(keys), comments="", fmt="%.4f")
    for k in names:
        r = data[f"a_{k}"] / data["a_lim"]
        print(f"  {k:10s} a/a_lim: min {np.nanmin(r):.3f} median {np.nanmedian(r):.3f}, "
              f"< 0.97 at {np.sum(r < 0.97)}/{len(r)} speeds")


# ---------------------------------------------------------- kick tools ----
def _kick_chatters(rpm, a, g, vmax, ks, dv, g2=0.0):
    """Settle the cut with a face entry, kick x' by dv at step ks, and judge
    the 300 revolutions after the kick (g2: comb term of C4)."""
    from tdsim import simulate
    from tdboundary import classify_run
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=KICK_REV + WINDOW, edge=True,
                 kick_step=ks, kick_dv=dv, nreg=NREG, g2=g2)
    st = (ks // ND + 1) * ND
    return not classify_run(r["x"][st:], SPR, r["diverged"])["stable"]


def _critical_kick(args):
    """Smallest kick (m/s) that turns the settled cut into chatter, by
    bisection in log(dv); inf if DV_HI does not, nan if the unkicked cut
    does not settle.  args: (rpm, a, g, vmax, phase, zeta[, g2])."""
    rpm, a, g, vmax, phase, zeta = args[:6]
    g2 = args[6] if len(args) > 6 else 0.0
    P.set_damping(zeta)
    ks = KICK_REV * SPR + (phase * ND) // N_PHASE
    if _kick_chatters(rpm, a, g, vmax, ks, DV_LO, g2):
        return np.nan
    if not _kick_chatters(rpm, a, g, vmax, ks, DV_HI, g2):
        return np.inf
    lo, hi = np.log(DV_LO), np.log(DV_HI)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if _kick_chatters(rpm, a, g, vmax, ks, np.exp(mid), g2):
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def _mode_factor(args):
    """Chatter-mode velocity amplitude per unit kick in the linear cut.

    The unclipped settled cut is kicked by a small dv.  The velocity
    deviation from the unkicked run is high-passed above F_HP, which removes
    the actuator mode (about 8 Hz) and keeps the structural, regenerative
    modes.  Its per-tooth peak decays as the sum of those modes; after the
    fast ones have died out, the slowest is fitted and extrapolated back to
    the kick: p = A_mode(kick) / dv.  Returns (p, decay per tooth period, r2
    of the fit); p is nan when the fit is poor (r2 < 0.98).
    """
    rpm, a, g, zeta = args
    P.set_damping(zeta)
    from tdsim import simulate
    dv0, ks, nt = 1e-3, KICK_REV * SPR, 160
    r0 = simulate(rpm, a, g, store=True, nrev=KICK_REV + 40, edge=True, nreg=NREG)
    r1 = simulate(rpm, a, g, store=True, nrev=KICK_REV + 40, edge=True, nreg=NREG,
                  kick_step=ks, kick_dv=dv0)
    d = r1["v"][ks:ks + nt * ND] - r0["v"][ks:ks + nt * ND]
    fr = np.fft.rfftfreq(len(d), 60.0 / (rpm * SPR))
    D = np.fft.rfft(d)
    D[fr < F_HP] = 0.0
    env = np.abs(np.fft.irfft(D, len(d))).reshape(nt, ND).max(axis=1)
    k = np.arange(nt)
    k0 = 6                                   # fast modes gone; fit until the envelope fell 1e3
    below = np.where(env[k0:] < 1e-3 * env[k0])[0]
    k1 = min(k0 + (below[0] if below.size else nt), nt - 8)       # nt - 8: filter edge
    use = (k >= k0) & (k < k1)
    if use.sum() < 8:
        return np.nan, np.nan, np.nan
    slope, icpt = np.polyfit(k[use], np.log(env[use]), 1)
    res = np.log(env[use]) - (slope * k[use] + icpt)
    r2 = 1 - res.var() / np.log(env[use]).var()
    p = float(np.exp(icpt) / dv0) if r2 >= 0.98 else np.nan
    return p, float(np.exp(slope)), float(r2)


# --------------------------------------------------------------- basin ----
def run_basin(a):
    from zoa import a_lim
    rpm_map, g_map, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    cases = []
    for n in BASIN_RPM:
        g = float(np.interp(n, rpm_map, g_map))
        al = a_lim(n, g)
        cases += [(n, g, al, f, VMAX) for f in BASIN_FRAC] + [(n, g, al, f, 0.0) for f in FREE_FRAC]
    jobs = [(float(n), f * al, g, v, p, zeta) for (n, g, al, f, v) in cases for p in range(N_PHASE)]
    clipped = [(float(n), f * al, g, zeta) for (n, g, al, f, v) in cases if v > 0]
    with Pool(a.workers) as pool:
        dv = np.array(pool.map(_critical_kick, jobs, chunksize=1)).reshape(len(cases), N_PHASE)
        mf = np.array(pool.map(_mode_factor, clipped))
    arr = np.array([(n, g, al * 1e3, f, v) for (n, g, al, f, v) in cases])
    p_mode, decay, r2 = (np.full(len(cases), np.nan) for _ in range(3))
    clip = arr[:, 4] > 0
    p_mode[clip], decay[clip], r2[clip] = mf[:, 0], mf[:, 1], mf[:, 2]
    # mean cutting impulse of one tooth period, to put a kick in physical terms
    fbar = (P.N_TEETH * arr[:, 2] * 1e-3 * arr[:, 3] * P.FZ / (2 * np.pi)) * (P.K_TC * np.pi / 4 + P.K_RC / 2)
    tooth_impulse = fbar * 60.0 / (arr[:, 0] * P.N_TEETH)
    np.savez_compressed(os.path.join(RES, "c1_basin.npz"), rpm=arr[:, 0], g=arr[:, 1],
                        a_lim=arr[:, 2], frac=arr[:, 3], vmax=arr[:, 4], dv=dv, p_mode=p_mode,
                        decay=decay, fit_r2=r2, tooth_impulse=tooth_impulse, mass=P.M, zeta=zeta,
                        n_phase=N_PHASE, kick_rev=KICK_REV, window=WINDOW, dv_hi=DV_HI,
                        nreg=NREG, build=build_id())
    for row, d, p in zip(arr, dv, p_mode):
        lab = "clip" if row[4] > 0 else "free"
        fin = d[np.isfinite(d)]
        print(f"{row[0]:.0f} rpm {row[3]:.2f} a_lim {lab}: dv* min "
              f"{fin.min()*1e3 if fin.size else np.inf:.2f} median {np.median(d)*1e3:.2f} mm/s "
              f"(inf {np.isinf(d).sum()}, nan {np.isnan(d).sum()}), p_mode {p:.3f}")


# ---------------------------------------------------------------- band ----
def _band_low(args):
    """Smallest depth (mm) at which a DV_HI kick on the settled cut, at any of
    BAND_PHASES phases, ends in chatter (bisection, assuming the basin
    shrinks with depth); nan if not even 0.98 a_lim does."""
    rpm, g, zeta = args
    P.set_damping(zeta)
    from zoa import a_lim
    al = a_lim(rpm, g)

    def chatters(f):
        return any(_kick_chatters(rpm, f * al, g, VMAX, KICK_REV * SPR + p * ND // BAND_PHASES, DV_HI)
                   for p in range(BAND_PHASES))

    lo, hi = 0.05, 0.98
    if not chatters(hi):
        return np.nan
    if chatters(lo):
        return lo * al * 1e3
    for _ in range(BAND_BISECT):
        mid = 0.5 * (lo + hi)
        if chatters(mid):
            hi = mid
        else:
            lo = mid
    return hi * al * 1e3


def run_band(a):
    rpm_map, g_map, zeta = gain_map(a.tag)
    m = (rpm_map - rpm_map[0]) % BAND_STEP == 0
    rpm, g = rpm_map[m], g_map[m]
    with Pool(a.workers) as pool:
        low = np.array(pool.map(_band_low, [(float(n), float(gg), zeta) for n, gg in zip(rpm, g)], chunksize=1))
    np.savez_compressed(os.path.join(RES, "c1_band.npz"), rpm=rpm, g=g, a_band_low=low, zeta=zeta,
                        dv_hi=DV_HI, phases=BAND_PHASES, nreg=NREG, build=build_id())
    np.savetxt(os.path.join(RES, "c1_band.csv"), np.column_stack([rpm, g, low]), delimiter=",",
               header="rpm,g,a_band_low", comments="", fmt="%.4f")
    print(f"band lower edge found at {np.isfinite(low).sum()}/{len(low)} speeds, "
          f"{np.nanmin(low):.2f}-{np.nanmax(low):.2f} mm")


# ----------------------------------------------------------- gainplane ----
def run_gainplane(a):
    from zoa import a_lim
    from feasibility import demand_envelope
    rpm_map, g_map, zeta = gain_map(a.tag)
    P.set_damping(zeta)
    speeds, gains = (1090.0, 2500.0), np.arange(0.0, 3001.0, 150.0)
    names = ["step_clip", "step_free", "edge_clip"]
    jobs = [(n, gg, k, zeta) for n in speeds for gg in gains for k in names]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_limit, jobs, chunksize=2)).reshape(len(speeds), len(gains), len(names))
    data = dict(rpm=np.array(speeds), gains=gains, zeta=zeta, nreg=NREG, build=build_id(),
                g_map=np.interp(speeds, rpm_map, g_map),
                a_lim=np.array([[a_lim(n, gg) for gg in gains] for n in speeds]) * 1e3,
                # peak command of the settled forced orbit over all depths up to a_lim:
                # above 9 V the orbit itself saturates and a soft entry cannot avoid it
                orbit_peak_V=np.array([[demand_envelope(n, gg, kappa=1.0)[1] / P.GA_GAIN
                                        for gg in gains] for n in speeds]))
    data.update({f"a_{k}": lim[:, :, i] for i, k in enumerate(names)})
    np.savez_compressed(os.path.join(RES, "c1_gainplane.npz"), **data)
    for i, n in enumerate(speeds):
        sat = gains[data["orbit_peak_V"][i] > VMAX]
        j = int(np.nanargmax(data["a_step_clip"][i]))
        print(f"{n:.0f} rpm: forced orbit saturates below a_lim for g >= "
              f"{sat.min() if sat.size else np.nan:.0f}; step-entry clipped limit peaks at "
              f"g = {gains[j]:.0f} ({data['a_step_clip'][i, j]:.2f} mm); map gain {data['g_map'][i]:.0f}")


# ------------------------------------------------------------------ df ----
DF_FRAC = np.linspace(0.05, 1.0, 96)
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
    from feasibility import forced_velocity
    gains = np.linspace(0.0, g, 81)
    alg = np.array([a_lim(rpm, x) for x in gains])
    out = np.full((len(DF_FRAC), 4), np.nan)
    for i, f in enumerate(DF_FRAC):
        a = f * alg[-1]
        ge = effective_gain_needed(alg, gains, a)
        b = -g * forced_velocity(rpm, g, [a], n_t=ND)[0]
        out[i] = (a * 1e3, ge, critical_amplitude(ge / g, np.zeros(1)), critical_amplitude(ge / g, b))
    return out


def run_df(a):
    rpm_map, g_map, zeta = gain_map(a.tag)
    rpm = np.array(BASIN_RPM, float)
    g = np.interp(rpm, rpm_map, g_map)
    with Pool(a.workers) as pool:
        res = np.array(pool.map(_df_speed, [(n, gg, zeta) for n, gg in zip(rpm, g)]))
    np.savez_compressed(os.path.join(RES, "c1_df.npz"), rpm=rpm, g=g, zeta=zeta, frac=DF_FRAC,
                        a=res[:, :, 0], g_eff=res[:, :, 1], A_single=res[:, :, 2], A_dual=res[:, :, 3])
    print("df done:", ", ".join(f"{n:.0f} rpm" for n in rpm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["map", "band", "basin", "gainplane", "df"])
    ap.add_argument("--tag", default="g1500")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--entries", default="", help="map: comma-separated subset of ENTRIES")
    a = ap.parse_args()
    dict(map=run_map, band=run_band, basin=run_basin, gainplane=run_gainplane, df=run_df)[a.command](a)


if __name__ == "__main__":
    main()
