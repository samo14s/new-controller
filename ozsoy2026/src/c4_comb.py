"""C4: actuator authority and the certified basin with a delayed-difference term.

Problem (C1, C2).  Under DVF the forced vibration of the cut takes most of the
9 V: the orbit's command b(t) = -g x_p'(t) reaches 6-8.6 V at 2500 rpm, which
leaves a margin u0 = 9 - b_max of a few tenths of a volt to 3 V for the
regenerative deviation, and the certified basin shrinks with it (C2).

Controller.  u = -g1 x' - g2 (x' - x'(t - T)) - kd (x - x(t - T)).
This is DVF plus PD feedback on the displacement difference over one tooth
period.  The law is not new: PD on x(t) - x(t - T) with the same argument
(a lower actuator voltage, since the forced vibration is not fought) is Li et
al. 2022 (Int. J. Adv. Manuf. Technol., doi 10.1007/s00170-022-09128-w); its
non-invasiveness on the periodic orbit is that of Pyragas delayed feedback
(Phys. Lett. A 1992), applied to milling with a bound on the control action by
van de Wouw et al. 2015 (IFAC-PapersOnLine, doi 10.1016/j.ifacol.2015.09.400)
and van Dijk, van de Wouw & Nijmeijer 2015 (Int. J. Robust Nonlinear Control
25(17):3495-3514, doi 10.1002/rnc.3280: fixed-structure delayed output
feedback that vanishes on the periodic solution, with the actuator forces as
a design criterion); see also Mancisidor et al. 2019 (Mechatronics, doi
10.1016/j.mechatronics.2019.102276, delayed feedback against the regenerative
effect in turning), Du et al. 2022, 2024 (time-delay feedback) and Li et al.
2021 (comb filtering of the spindle harmonics, Mech. Syst. Signal Process.).
What C4 adds is narrower: the analysis under a hard 9 V clip, i.e. the
authority budget of the forced orbit, the certified basin with the clip (C2's
LMIs with the delayed terms in the sector argument), the gains that maximise
it, and the limits of the idealisation (c4_robust.py).

Properties.  On any tooth-periodic motion x(t) = x(t - T), so the delayed
terms vanish on the forced orbit (1 - exp(-sT) is zero at every tooth
harmonic): its command is -g1 x_p' whatever g2 and kd are.  For g1, g2 >= 0,
K(s) = g1 + g2 (1 - exp(-sT)) has Re K >= g1 in Re s >= 0 (positive real), so
with an ideal collocated actuator the velocity loop is passive like DVF.  Sign
of kd: the regenerative force is KAPPA a (x - x(t - T)) with KAPPA < 0, so
kd < 0 opposes it (each -1e5 V/m adds about 1 mm to a_lim) and kd > 0 adds to it.

Fair baseline.  The paper's gain bound g <= 1500 is an assumption of this
reproduction, not the actuator's limit.  Both families are therefore compared
under the same authority rule as feasibility.py (the settled forced demand
stays below F_max at every depth up to 0.9 a_lim of the design) with the same
bound g1 <= 3000; the DVF optimum is then feas_g3000's.

Model and certificate: those of C2 (c2_certificate.py) with the delayed terms
in the deadzone argument K x and in the linear part; the history velocity
x'(t - T) is the derivative of the Chebyshev interpolant of the history at
theta = -T.  The orbit, its command bound b_max and the sector condition are
unchanged (b depends on g1 only).  The interpolant cannot follow the jump of
x'(t - T) one tooth period after a velocity kick (an error of order g2 dv / 2
in the command during that period); the simulator validates the result.

    python c4_comb.py limits     -> ../results/c4_limits.npz   (linear a_lim per design, authority-feasible g1)
    python c4_comb.py gainmap    -> ../results/c4_gainmap.npz  (certified kick over the gain grid, speed x depth)
    python c4_comb.py entry      -> ../results/c4_entry.npz    (simulated step and face entries per design)
    python c4_comb.py validate   -> ../results/c4_validate.npz (best DVF vs best delayed design: certificate,
                                                                 averaged model and simulated critical kicks)
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P
import c2_certificate as c2
from zoa import a_lim

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VMAX = c2.VMAX
G_REF = 1500.0                                 # the paper's DVF gain (C1, C2)
G1_MAX = 3000.0                                # gain bound of both families (feas_g3000)
FEAS_KAPPA = 0.9                               # depth range of the authority rule (feasibility.KAPPA)


# --------------------------------------------------------------- model ----
def model(rpm, a, g1, g2=0.0, kd=0.0, N=None):
    """C2's collocated model with u = -g1 x' - g2 (x' - x'(t-T)) - kd (x - x(t-T))
    (linear part inside A, deadzone argument K x)."""
    N = N or c2.nodes_for(rpm)
    A, B, _ = c2.model(rpm, a, 0.0, N)         # cut, structure and actuator without feedback
    T = 60.0 / (rpm * P.N_TEETH)
    D, _ = c2.cheb(N)
    Dt = (2.0 / T) * D
    n = 4 + N
    K = np.zeros(n)
    K[1] = -(g1 + g2)
    K[0] = -kd
    K[n - 1] += kd                              # x(t - T) = phi_N
    K[0] += g2 * Dt[N, 0]                       # x'(t - T) = sum_k Dt[N, k] phi_k, phi_0 = x
    K[4:] += g2 * Dt[N, 1:]
    return A + np.outer(B, K), B, K


def lead(rpm, a, g1, g2=0.0, kd=0.0, N=None):
    return float(np.linalg.eigvals(model(rpm, a, g1, g2, kd, N)[0]).real.max())


def a_lim_ctrl(rpm, g1, g2=0.0, kd=0.0, a_max=0.2, rtol=1e-6):
    """First unstable depth (m) of the collocated model: upward scan by 3 %
    from 0.3 mm (stability need not be monotone in depth with the comb
    terms), then bisection.  inf if stable up to a_max."""
    if lead(rpm, 1e-6, g1, g2, kd) >= 0:
        return 0.0
    lo, a = 1e-6, 0.3e-3
    while lead(rpm, a, g1, g2, kd) < 0:
        lo, a = a, a * 1.03
        if a > a_max:
            return np.inf
    hi = a
    while hi - lo > rtol * hi:
        m = 0.5 * (lo + hi)
        if lead(rpm, m, g1, g2, kd) < 0:
            lo = m
        else:
            hi = m
    return lo


# --------------------------------------------------------- certificate ----
def certify_kick(rpm, a, g1, g2=0.0, settings=c2.TIGHT, full=False, delta=c2.DELTA):
    A, B, K = model(rpm, a, g1, g2)
    _, _, bmax = c2.orbit(rpm, g1, a)
    e = np.zeros(len(B))
    e[1] = 1.0
    return c2.certify(A, B, K, VMAX - bmax, e, rpm, settings, full=full, delta=delta)


def certify_step(rpm, a, g1, g2=0.0, settings=c2.TIGHT, delta=c2.DELTA):
    A, B, K = model(rpm, a, g1, g2)
    x0, _, bmax = c2.orbit(rpm, g1, a)
    return c2.certify(A, B, K, VMAX - bmax, -x0, rpm, settings, delta=delta)


def orbit_saturation_depth(rpm, g1, a_max=50e-3, da=0.25e-3):
    """Smallest depth (m) on a da grid at which the forced orbit's command
    bound b_max reaches the limit (the orbit itself saturates); inf if none
    up to a_max.  Independent of g2.  b_max is not monotone in depth (the
    helix cancels tooth harmonics at some depths), so the orbit may fit
    again deeper down; run_limits also stores b_max on a depth grid."""
    for a in np.arange(da, a_max + da / 2, da):
        if c2.orbit(rpm, g1, a)[2] >= VMAX:
            return float(a)
    return np.inf


def demand_envelope_ctrl(rpm, g1, g2=0.0, kd=0.0, kappa=FEAS_KAPPA, n_depth=80):
    """(a_lim of the design, worst settled forced demand (N) over all depths up
    to kappa a_lim): feasibility.demand_envelope with the delayed terms in
    a_lim (the forced demand depends on g1 only)."""
    from feasibility import forced_demand
    al = a_lim_ctrl(rpm, g1, g2, kd)
    if not np.isfinite(al) or al <= 0 or g1 == 0:
        return al, 0.0
    depths = np.linspace(kappa * al / n_depth, kappa * al, n_depth)
    return al, float(forced_demand(rpm, g1, depths).max())


def feasible(rpm, g1, g2=0.0, kd=0.0, guard=1.0):
    al, d = demand_envelope_ctrl(rpm, g1, g2, kd)
    return bool(np.isfinite(al) and al > 0 and d <= guard * P.F_MAX)


def feasible_g1(rpm, g2=0.0, kd=0.0, g_hi=G1_MAX, tol=0.25, guard=1.0):
    """Largest authority-feasible g1 <= g_hi (bisection, as
    feasibility.edge_gain); with g2 = kd = 0 it is the DVF optimum.  guard < 1
    keeps a band of the limit free (the design's forced demand then peaks at
    guard x 9 V instead of exactly 9 V)."""
    if feasible(rpm, g_hi, g2, kd, guard):
        return g_hi
    lo, hi = 0.0, g_hi
    if not feasible(rpm, lo, g2, kd, guard):
        # without DVF the loop itself can be unstable (P term): the feasible
        # gains then start above 0; take the largest feasible point of a
        # 100 V s/m grid and bisect above it
        grid = [g for g in np.arange(g_hi - 100.0, 0.0, -100.0) if feasible(rpm, g, g2, kd, guard)]
        if not grid:
            return np.nan
        lo, hi = grid[0], grid[0] + 100.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(rpm, mid, g2, kd, guard):
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------------------ campaigns ----
LIM_G2 = (0.0, 500.0, 1000.0, 2000.0, 4000.0)  # comb gains added to the paper's DVF gain
FEAS_DESIGNS = (("dvf", 0.0, 0.0), ("d1000", 1000.0, 0.0), ("d2000", 2000.0, 0.0), ("d4000", 4000.0, 0.0),
                ("p3e5", 0.0, -3e5), ("p1e6", 0.0, -1e6))   # (name, g2, kd), g1 = feasible_g1
LIM_DEPTH = np.arange(0.5e-3, 50.01e-3, 0.5e-3)  # depth grid of the orbit-command map (m)


def paper_gains():
    """The DVF gain of C1 and C2 at each speed (feasibility study with g <= 1500:
    1500, lower from 1530 to 2450 rpm where the forced vibration would saturate)."""
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    return d["rpm"], d["g_best"], float(d["zeta"])


G1_UNCAPPED = 20000.0                          # gain bound of the check without the 3000 cap


def _limits(args):
    rpm, g1, zeta = args
    P.set_damping(zeta)
    paper = [a_lim_ctrl(rpm, g1, g2) for g2 in LIM_G2]
    bmax = [c2.orbit(rpm, g1, a)[2] for a in LIM_DEPTH]
    gf = [feasible_g1(rpm, g2, kd) for _, g2, kd in FEAS_DESIGNS]
    af = [a_lim_ctrl(rpm, g, g2, kd, a_max=0.5) if np.isfinite(g) else np.nan
          for g, (_, g2, kd) in zip(gf, FEAS_DESIGNS)]
    # where DVF sits on the 3000 bound, the same rule without the bound for DVF and d2000
    unc = [np.nan] * 4
    if gf[0] >= G1_MAX:
        gu = [feasible_g1(rpm, g2, g_hi=G1_UNCAPPED) for g2 in (0.0, 2000.0)]
        unc = gu + [a_lim_ctrl(rpm, g, g2, a_max=0.5) for g, g2 in zip(gu, (0.0, 2000.0))]
    return paper, orbit_saturation_depth(rpm, g1), bmax, gf, af, unc


def run_limits(a):
    """Per speed: (1) the paper's DVF gain g1 with the comb gains LIM_G2 (linear
    a_lim), with the forced orbit's command bound on a depth grid; (2) the
    authority-feasible designs FEAS_DESIGNS (g1 <= 3000 under the rule of
    feasibility.py) and their linear a_lim."""
    rpm, g1, zeta = paper_gains()
    with Pool(a.workers) as pool:
        res = pool.map(_limits, [(float(n), float(g), zeta) for n, g in zip(rpm, g1)], chunksize=2)
    al = np.array([r[0] for r in res]) * 1e3
    sat = np.array([r[1] for r in res]) * 1e3
    bmax = np.array([r[2] for r in res])
    gf = np.array([r[3] for r in res])
    af = np.array([r[4] for r in res]) * 1e3
    unc = np.array([r[5] for r in res])
    np.savez_compressed(os.path.join(RES, "c4_limits.npz"), rpm=rpm, g1=g1, g2=np.array(LIM_G2), a_lim=al,
                        a_sat=sat, depth=LIM_DEPTH * 1e3, bmax=bmax,
                        a_lim_zoa=np.array([a_lim(n, g) for n, g in zip(rpm, g1)]) * 1e3,
                        feas_names=np.array([d[0] for d in FEAS_DESIGNS]), feas_g2=np.array([d[1] for d in FEAS_DESIGNS]),
                        feas_kd=np.array([d[2] for d in FEAS_DESIGNS]), feas_g1=gf, feas_a_lim=af, g1_max=G1_MAX,
                        uncapped_g1=unc[:, :2], uncapped_a_lim=unc[:, 2:] * 1e3, zeta=zeta)
    for j, g2 in enumerate(LIM_G2):
        r = al[:, j] / al[:, 0]
        print(f"paper g1 + g2 = {g2:5.0f}: a_lim {np.nanmin(al[:, j]):.2f}-{np.nanmax(al[:, j]):.2f} mm, "
              f"ratio to DVF {np.nanmin(r):.2f}-{np.nanmax(r):.2f} (median {np.nanmedian(r):.2f})")
    print(f"paper g1: orbit command >= 9 V on {np.mean(bmax >= VMAX) * 100:.1f} % of the speed x depth grid")
    m = np.isfinite(unc[:, 2])
    if m.any():
        r = unc[m, 3] / unc[m, 2]
        print(f"without the 3000 bound at the {m.sum()} speeds where DVF sits on it: DVF g1 "
              f"{np.nanmin(unc[m, 0]):.0f}-{np.nanmax(unc[m, 0]):.0f}, d2000 / DVF {r.min():.2f}-{r.max():.2f} "
              f"(median {np.median(r):.2f})")
    for j, (name, g2, kd) in enumerate(FEAS_DESIGNS):
        r = af[:, j] / af[:, 0]
        print(f"feasible {name:6s} (g2 {g2:5.0f}, kd {kd:8.0f}): g1 {np.nanmin(gf[:, j]):6.0f}-{np.nanmax(gf[:, j]):6.0f}, "
              f"a_lim {np.nanmin(af[:, j]):.2f}-{np.nanmax(af[:, j]):.2f} mm, ratio to feasible DVF "
              f"{np.nanmin(r):.2f}-{np.nanmax(r):.2f} (median {np.nanmedian(r):.2f})")


GM_RPM = (1090.0, 1420.0, 2500.0, 2870.0)        # the C1 basin speeds
GM_DEPTH = tuple(np.arange(2.0, 34.1, 2.0))       # mm
GM_G1 = (500.0, 1000.0, G_REF, 2000.0, G1_MAX)
GM_G2 = (0.0, 500.0, 1000.0, 2000.0, 4000.0)
GM_G1_DVF = tuple(np.arange(500.0, G1_MAX + 1.0, 100.0))   # DVF alone on a finer g1 grid (26 gains vs 20 delayed pairs)


def _gainmap(args):
    rpm, a, g1, g2, zeta = args
    P.set_damping(zeta)
    sigma = -lead(rpm, a, g1, g2)
    bmax = c2.orbit(rpm, g1, a)[2]
    if sigma <= 0:
        return 0.0, sigma, bmax, "linearly unstable"
    if bmax >= VMAX:
        return 0.0, sigma, bmax, "orbit saturates"
    beta, msg = certify_kick(rpm, a, g1, g2, settings=c2.FAST)
    return beta, sigma, bmax, msg


def run_gainmap(a):
    """Certified kick (FAST solver settings, checked) at the basin speeds on
    an absolute depth grid, for every (g1, g2) of the gain grid.  Points
    already in c4_gainmap.npz are reused (the certificate is deterministic).
    DVF alone is also mapped on the finer grid GM_G1_DVF, so that the DVF
    baseline is not handicapped by having fewer candidates than the delayed
    designs."""
    zeta = float(np.load(os.path.join(RES, "feas_g1500.npz"))["zeta"])
    fn = os.path.join(RES, "c4_gainmap.npz")
    old = {}
    if os.path.exists(fn):
        o = np.load(fn)
        for i, n in enumerate(o["rpm"]):
            for j, d in enumerate(o["depth"]):
                for p, g1 in enumerate(o["g1"]):
                    for q, g2 in enumerate(o["g2"]):
                        old[(float(n), float(d), float(g1), float(g2))] = (
                            o["kick"][i, j, p, q], o["sigma"][i, j, p, q], o["bmax"][i, j, p, q], str(o["msg"][i, j, p, q]))
    keys = [(n, float(d), g1, g2) for n in GM_RPM for d in GM_DEPTH for g1 in GM_G1 for g2 in GM_G2]
    dkeys = [(n, float(d), float(g1), 0.0) for n in GM_RPM for d in GM_DEPTH for g1 in GM_G1_DVF]
    if os.path.exists(fn) and "g1_dvf" in o.files:
        for i, n in enumerate(o["rpm"]):
            for j, d in enumerate(o["depth"]):
                for p, g1 in enumerate(o["g1_dvf"]):
                    old[(float(n), float(d), float(g1), 0.0)] = (
                        o["kick_dvf"][i, j, p], o["sigma_dvf"][i, j, p], o["bmax_dvf"][i, j, p], str(o["msg_dvf"][i, j, p]))
    todo = [k for k in dict.fromkeys(keys + dkeys) if k not in old]
    print(f"{len(keys)} points, {len(keys) - len(todo)} reused, {len(todo)} to compute", flush=True)
    with Pool(a.workers) as pool:
        new = pool.map(_gainmap, [(n, d * 1e-3, g1, g2, zeta) for (n, d, g1, g2) in todo], chunksize=1)
    old.update(dict(zip(todo, new)))
    res = [old[k] for k in keys]
    shape = (len(GM_RPM), len(GM_DEPTH), len(GM_G1), len(GM_G2))
    kick, sigma, bmax = (np.array([r[k] for r in res], float).reshape(shape) for k in range(3))
    msg = np.array([r[3] for r in res]).reshape(shape)
    dres = [old[k] for k in dkeys]
    dshape = (len(GM_RPM), len(GM_DEPTH), len(GM_G1_DVF))
    kd_, sd_, bd_ = (np.array([r[k] for r in dres], float).reshape(dshape) for k in range(3))
    np.savez_compressed(fn, rpm=np.array(GM_RPM), depth=np.array(GM_DEPTH),
                        g1=np.array(GM_G1), g2=np.array(GM_G2), kick=kick, sigma=sigma, bmax=bmax, msg=msg,
                        g1_dvf=np.array(GM_G1_DVF), kick_dvf=kd_, sigma_dvf=sd_, bmax_dvf=bd_,
                        msg_dvf=np.array([r[3] for r in dres]).reshape(dshape),
                        a_lim_ref=np.array([a_lim(n, G_REF) for n in GM_RPM]) * 1e3, zeta=zeta)
    for i, n in enumerate(GM_RPM):
        best = kick[i].reshape(len(GM_DEPTH), -1).max(axis=1)
        dvf = kd_[i].max(axis=1)
        print(f"{n:.0f} rpm: best certified kick (mm/s) " + " ".join(f"{d:.0f}:{b*1e3:.1f}" for d, b in zip(GM_DEPTH, best)))
        print(f"{'':9s}best DVF, fine g1 grid   " + " ".join(f"{d:.0f}:{b*1e3:.1f}" for d, b in zip(GM_DEPTH, dvf)))


ENT_RPM = tuple(np.arange(600.0, 3000.1, 100.0))
ENT_NAMES = ("step_clip", "edge_clip")


def entry_designs():
    """(label, g1 per ENT_RPM, g2): the paper's gain plus 1000 and 2000, and the
    authority-feasible DVF and d2000 designs of c4_limits.npz."""
    rpm_map, g_map, _ = paper_gains()
    L = np.load(os.path.join(RES, "c4_limits.npz"))
    names = list(L["feas_names"])
    gf = lambda name: np.interp(ENT_RPM, L["rpm"], L["feas_g1"][:, names.index(name)])
    gp = np.interp(ENT_RPM, rpm_map, g_map)
    return [("paper+1000", gp, 1000.0), ("paper+2000", gp, 2000.0), ("feas_dvf", gf("dvf"), 0.0),
            ("feas_d2000", gf("d2000"), 2000.0)]


def _entry(args):
    """Simulated entry limit (mm) of a design, with the C1 entries and depth
    ladder taken relative to the design's own linear limit."""
    from tdboundary import boundary
    from c1_unsafe_zone import ENTRIES, LADDER
    rpm, g1, g2, name, zeta = args
    P.set_damping(zeta)
    al = a_lim_ctrl(rpm, g1, g2)
    if not np.isfinite(al) or al <= 0:
        return np.nan
    vmax, entry = ENTRIES[name]
    return boundary(rpm, g1, vmax=vmax, a_lo=LADDER["lo"] * al, a_hi=LADDER["hi"] * al,
                    growth=LADDER["growth"], g2=g2, **entry) * 1e3


def run_entry(a):
    """Step and face entries with the 9 V limit in the simulator (C1
    protocol: nreg 4, D1 classifier) for the designs of entry_designs()."""
    import tdsim                                  # build and fix the simulator before the workers start
    zeta = paper_gains()[2]
    des = entry_designs()
    jobs = [(n, float(g1[i]), g2, k, zeta) for i, n in enumerate(ENT_RPM) for _, g1, g2 in des for k in ENT_NAMES]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_entry, jobs, chunksize=1)).reshape(len(ENT_RPM), len(des), len(ENT_NAMES))
    al = np.array([[a_lim_ctrl(n, float(g1[i]), g2) for _, g1, g2 in des] for i, n in enumerate(ENT_RPM)]) * 1e3
    np.savez_compressed(os.path.join(RES, "c4_entry.npz"), rpm=np.array(ENT_RPM),
                        designs=np.array([d[0] for d in des]), g1=np.array([d[1] for d in des]).T,
                        g2=np.array([d[2] for d in des]), names=np.array(ENT_NAMES), a_entry=lim, a_lim=al,
                        zeta=zeta, build=tdsim.SOURCE_SHA[:12])
    for j, (lab, _, _) in enumerate(des):
        for k, name in enumerate(ENT_NAMES):
            r = lim[:, j, k] / al[:, j]
            print(f"{lab:10s} {name:9s}: {np.nanmin(lim[:, j, k]):.2f}-{np.nanmax(lim[:, j, k]):.2f} mm, "
                  f"a/a_lim min {np.nanmin(r):.3f} median {np.nanmedian(r):.3f}")


VAL_DEPTH = (6.0, 10.0, 14.0, 18.0, 22.0, 26.0)  # mm, on the gain-map grid


def best_gains(gm, i, j, comb=None):
    """(g1, g2, kick) of the largest certified kick at gain-map point (i, j);
    comb = False: DVF only (g2 = 0); True: g2 > 0 only; None: any."""
    k = np.array(gm["kick"][i, j], float)
    g2 = np.asarray(gm["g2"])
    if comb is not None:
        k[:, (g2 > 0) != comb] = -1.0
    p, q = np.unravel_index(np.argmax(k), k.shape)
    return float(gm["g1"][p]), float(g2[q]), float(max(k[p, q], 0.0))


def _validate(args):
    rpm, a, g1, g2, zeta, offaxis = args
    P.set_damping(zeta)
    A, B, K = model(rpm, a, g1, g2)
    _, b, bmax = c2.orbit(rpm, g1, a)
    e = np.zeros(len(B))
    e[1] = 1.0
    size, note, cert = c2.certify(A, B, K, VMAX - bmax, e, rpm, full=True)
    Tp = 60.0 / (rpm * P.N_TEETH)
    vmax, vend = c2.proof_check(A, B, K, b, Tp, cert) if cert is not None else (np.nan, np.nan)
    off = c2.td_offaxis(rpm, a, g1, cert, g2=g2) if (cert is not None and offaxis) else (0, 0)
    return size, note, vmax, vend, c2.model_critical_kick(A, B, K, b, Tp), off[0], off[1]


def run_validate(a):
    """At the basin speeds and VAL_DEPTH, for the gain map's best DVF design
    (g2 = 0) and its best delayed design (g2 > 0): the simulated critical kick
    (C1 protocol, 16 phases; nan where the unkicked cut already chatters),
    the certificate (tight settings) with its proof check, the averaged
    model's critical kick, and for the delayed design the off-axis test (C2)."""
    import tdsim
    from c1_unsafe_zone import _critical_kick, N_PHASE
    gm = np.load(os.path.join(RES, "c4_gainmap.npz"))
    zeta = float(gm["zeta"])
    pts = []                                      # (rpm, depth mm, g1, g2, kind)
    for i, n in enumerate(gm["rpm"]):
        for d in VAL_DEPTH:
            j = int(np.argmin(np.abs(gm["depth"] - d)))
            for kind, comb in (("dvf", False), ("comb", True)):
                g1, g2, k = best_gains(gm, i, j, comb)
                if k > 0:
                    pts.append((float(n), d, g1, g2, kind))
    td_jobs = [(n, d * 1e-3, g1, VMAX, p, zeta, g2) for (n, d, g1, g2, _) in pts for p in range(N_PHASE)]
    cjobs = [(n, d * 1e-3, g1, g2, zeta, kind == "comb") for (n, d, g1, g2, kind) in pts]
    with Pool(a.workers) as pool:
        dv = np.array(pool.map(_critical_kick, td_jobs, chunksize=1)).reshape(len(pts), N_PHASE)
        res = pool.map(_validate, cjobs, chunksize=1)
    cert, vmax, vend, mk, obad, oruns = (np.array([r[k] for r in res], float) for k in (0, 2, 3, 4, 5, 6))
    kind = np.array([p[4] for p in pts])
    unsettled = np.isnan(dv).any(axis=1)          # the unkicked cut chatters at some phase
    td_min = np.where(unsettled, np.nan, np.nanmin(np.where(np.isnan(dv), np.inf, dv), axis=1))
    np.savez_compressed(os.path.join(RES, "c4_validate.npz"), rpm=np.array([p[0] for p in pts]),
                        depth=np.array([p[1] for p in pts]), g1=np.array([p[2] for p in pts]),
                        g2=np.array([p[3] for p in pts]), kind=kind, dv=dv, td_min=td_min, unsettled=unsettled,
                        cert=cert, model_kick=mk, proof_vmax=vmax, proof_vend=vend, offaxis_bad=obad,
                        offaxis_runs=oruns, msg=np.array([r[1] for r in res]), zeta=zeta,
                        build=tdsim.SOURCE_SHA[:12])
    for (n, d, g1, g2, kd_), c, m, t, v, ob, orr, u in zip(pts, cert, mk, td_min, vmax, obad, oruns, unsettled):
        ts = "chatters unkicked" if u else f"{t*1e3:8.2f}"
        print(f"{n:.0f} rpm {d:4.0f} mm {kd_:4s} g1 {g1:5.0f} g2 {g2:5.0f}: cert {c*1e3:7.3f}, model {m*1e3:8.2f}, "
              f"TD {ts} mm/s; max V {v:.4f}; off-axis {ob:.0f}/{orr:.0f}")
    ok = np.isfinite(td_min) & (cert > 0)
    fm = np.isfinite(mk) & (cert > 0)
    print(f"certificate below the simulated critical kick at {np.sum(cert[ok] <= td_min[ok])}/{ok.sum()} points; "
          f"below the model's at {np.sum(cert[fm] <= mk[fm])}/{fm.sum()}; unsettled {unsettled.sum()}; "
          f"off-axis chatter {obad.sum():.0f}/{oruns.sum():.0f}")
    for n in gm["rpm"]:
        for d in VAL_DEPTH:
            sel = [(k, t) for (nn, dd, _, _, k), t in zip(pts, td_min) if nn == n and dd == d]
            if len(sel) == 2:
                print(f"  {n:.0f} rpm {d:.0f} mm: simulated critical kick DVF {sel[0][1]*1e3:.2f}, "
                      f"delayed {sel[1][1]*1e3:.2f} mm/s")


VAL2_DEEP = (30.0, 34.0)                        # mm: the deepest certified depths, delayed designs only


def _same_g1(args):
    """Best g2 > 0 (by the certificate) at a given g1: the delayed design with
    the same orbit command, hence the same authority budget, as that DVF."""
    rpm, a, g1, zeta = args
    best = (0.0, np.nan)
    for g2 in GM_G2[1:]:
        k = _gainmap((rpm, a, g1, g2, zeta))[0]
        if k > best[0]:
            best = (k, g2)
    return best


def run_validate2(a):
    """Supplement to run_validate after the second review.
      dvf_fine  the best DVF of the fine g1 grid, where it differs from run_validate's
      same_g1   that DVF's g1 plus its best g2 > 0: same orbit command and authority
                budget, so a change of the simulated basin is due to the delayed term
      comb      the best delayed design at VAL2_DEEP (certificate with its checks,
                averaged model, off-axis test), where only delayed designs are certified
    Each with the simulated critical kick (C1 protocol, 16 phases)."""
    import tdsim
    from c1_unsafe_zone import _critical_kick, N_PHASE
    gm = np.load(os.path.join(RES, "c4_gainmap.npz"))
    v1 = np.load(os.path.join(RES, "c4_validate.npz"))
    zeta = float(gm["zeta"])
    pts, sjobs = [], []
    for i, n in enumerate(gm["rpm"]):
        for d in VAL_DEPTH + VAL2_DEEP:
            j = int(np.argmin(np.abs(gm["depth"] - d)))
            kd_ = gm["kick_dvf"][i, j]
            if kd_.max() > 0:
                g1 = float(gm["g1_dvf"][int(np.argmax(kd_))])
                old = (v1["rpm"] == n) & (v1["depth"] == d) & (v1["kind"] == "dvf")
                if not (old.any() and float(v1["g1"][old][0]) == g1):
                    pts.append((float(n), d, g1, 0.0, "dvf_fine"))
                sjobs.append((float(n), d, g1))
            if d in VAL2_DEEP:
                g1, g2, k = best_gains(gm, i, j, True)
                if k > 0:
                    pts.append((float(n), d, g1, g2, "comb"))
    with Pool(a.workers) as pool:
        sg = pool.map(_same_g1, [(n, d * 1e-3, g1, zeta) for n, d, g1 in sjobs], chunksize=1)
        pts += [(n, d, g1, g2, "same_g1") for (n, d, g1), (k, g2) in zip(sjobs, sg) if k > 0]
        td_jobs = [(n, d * 1e-3, g1, VMAX, p, zeta, g2) for (n, d, g1, g2, _) in pts for p in range(N_PHASE)]
        dv = np.array(pool.map(_critical_kick, td_jobs, chunksize=1)).reshape(len(pts), N_PHASE)
        res = pool.map(_validate, [(n, d * 1e-3, g1, g2, zeta, kind == "comb") for (n, d, g1, g2, kind) in pts],
                       chunksize=1)
    cert, vmax, vend, mk, obad, oruns = (np.array([r[k] for r in res], float) for k in (0, 2, 3, 4, 5, 6))
    unsettled = np.isnan(dv).any(axis=1)
    td_min = np.where(unsettled, np.nan, np.nanmin(np.where(np.isnan(dv), np.inf, dv), axis=1))
    np.savez_compressed(os.path.join(RES, "c4_validate2.npz"), rpm=np.array([p[0] for p in pts]),
                        depth=np.array([p[1] for p in pts]), g1=np.array([p[2] for p in pts]),
                        g2=np.array([p[3] for p in pts]), kind=np.array([p[4] for p in pts]), dv=dv, td_min=td_min,
                        unsettled=unsettled, cert=cert, model_kick=mk, proof_vmax=vmax, proof_vend=vend,
                        offaxis_bad=obad, offaxis_runs=oruns, msg=np.array([r[1] for r in res]), zeta=zeta,
                        build=tdsim.SOURCE_SHA[:12])
    for (n, d, g1, g2, kind), c, m, t, u, ob, orr in zip(pts, cert, mk, td_min, unsettled, obad, oruns):
        ts = "chatters unkicked" if u else f"{t*1e3:8.2f}"
        print(f"{n:.0f} rpm {d:4.0f} mm {kind:8s} g1 {g1:5.0f} g2 {g2:5.0f}: cert {c*1e3:7.3f}, model {m*1e3:8.2f}, "
              f"TD {ts} mm/s; off-axis {ob:.0f}/{orr:.0f}")
    ok = np.isfinite(td_min) & (cert > 0)
    print(f"certificate below the simulated critical kick at {np.sum(cert[ok] <= td_min[ok])}/{ok.sum()} points; "
          f"off-axis chatter {obad.sum():.0f}/{oruns.sum():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["limits", "gainmap", "entry", "validate", "validate2"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dict(limits=run_limits, gainmap=run_gainmap, entry=run_entry, validate=run_validate,
         validate2=run_validate2)[a.command](a)


if __name__ == "__main__":
    main()
