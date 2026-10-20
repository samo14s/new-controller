"""C4: authority allocation with an orbit-invisible comb term.

Problem (C1, C2).  Under DVF the forced vibration of the cut takes most of the
9 V: the orbit's command b(t) = -g x_p'(t) reaches 6-8.6 V at 2500 rpm, which
leaves a margin u0 = 9 - b_max of a few tenths of a volt to 3 V for the
regenerative deviation, and the certified basin shrinks with it (C2).

Controller.  u = -g1 x' - g2 (x' - x'(t - T)) - kd (x - x(t - T)).
On any tooth-periodic motion x(t) = x(t - T), so the comb terms vanish on the
forced orbit: its command is -g1 x_p' whatever g2 and kd are (the transfer
function 1 - exp(-sT) is zero at every tooth-passing harmonic).  They act only
on the deviation from the orbit, i.e. on chatter, without using authority on
the forced vibration.  The design trades g1 (damping, but authority used on the
orbit) against g2 (damping of the deviation, no authority used on the orbit).
kd (proportional chip feedback) was tried and dropped: it destabilises the cut
through the actuator mode (probe in the README).

Model and certificate: those of C2 (c2_certificate.py) with the comb terms in
the deadzone argument K x and in the linear part; the history velocity
x'(t - T) is the derivative of the Chebyshev interpolant of the history at
theta = -T.  The orbit, its command bound b_max and the sector condition are
unchanged (b depends on g1 only).

    python c4_comb.py limits     -> ../results/c4_limits.npz   (linear a_lim per comb gain and orbit saturation, vs speed)
    python c4_comb.py gainmap    -> ../results/c4_gainmap.npz  (best certified kick over the gain grid, speed x depth)
    python c4_comb.py validate   -> ../results/c4_validate.npz (certificate, model and simulated critical kicks at
                                                                 the C1 basin speeds for the chosen designs)
    python c4_comb.py entry      -> ../results/c4_entry.npz    (simulated step-entry limit per design)
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


# ------------------------------------------------------------ campaigns ----
LIM_G2 = (0.0, 500.0, 1000.0, 2000.0, 4000.0)  # comb gains of the limit map, added to the paper's DVF gain


def paper_gains():
    """The DVF gain of C1 and C2 at each speed (feasibility study: 1500, lower
    from 1530 to 2450 rpm where the forced vibration would saturate)."""
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    return d["rpm"], d["g_best"], float(d["zeta"])


LIM_DEPTH = np.arange(0.5e-3, 50.01e-3, 0.5e-3)  # depth grid of the orbit-command map (m)


def _limits(args):
    rpm, g1, zeta = args
    P.set_damping(zeta)
    bmax = [c2.orbit(rpm, g1, a)[2] for a in LIM_DEPTH]
    return [a_lim_ctrl(rpm, g1, g2) for g2 in LIM_G2], orbit_saturation_depth(rpm, g1), bmax


def run_limits(a):
    """Linear stability limit of the collocated model vs speed with the
    paper's DVF gain g1 and the comb gains LIM_G2, and the depth at which
    the forced orbit alone saturates (it depends on g1 only)."""
    rpm, g1, zeta = paper_gains()
    with Pool(a.workers) as pool:
        res = pool.map(_limits, [(float(n), float(g), zeta) for n, g in zip(rpm, g1)], chunksize=4)
    al = np.array([r[0] for r in res]) * 1e3
    sat = np.array([r[1] for r in res]) * 1e3
    bmax = np.array([r[2] for r in res])
    np.savez_compressed(os.path.join(RES, "c4_limits.npz"), rpm=rpm, g1=g1, g2=np.array(LIM_G2), a_lim=al,
                        a_sat=sat, depth=LIM_DEPTH * 1e3, bmax=bmax,
                        a_lim_zoa=np.array([a_lim(n, g) for n, g in zip(rpm, g1)]) * 1e3, zeta=zeta)
    for j, g2 in enumerate(LIM_G2):
        r = al[:, j] / al[:, 0]
        print(f"g2 = {g2:5.0f}: a_lim {np.nanmin(al[:, j]):.2f}-{np.nanmax(al[:, j]):.2f} mm, "
              f"ratio to DVF {np.nanmin(r):.2f}-{np.nanmax(r):.2f} (median {np.nanmedian(r):.2f})")
    fin = np.isfinite(sat)
    print(f"orbit saturates below 50 mm at {fin.sum()}/{len(rpm)} speeds"
          + (f", from {sat[fin].min():.2f} mm" if fin.any() else "")
          + f"; on {np.mean(bmax >= VMAX) * 100:.1f} % of the speed x depth grid")


GM_RPM = (1090.0, 1420.0, 2500.0, 2870.0)        # the C1 basin speeds
GM_DEPTH = tuple(np.arange(2.0, 34.1, 2.0))       # mm
GM_G1 = (500.0, 1000.0, G_REF)
GM_G2 = (0.0, 500.0, 1000.0, 2000.0, 4000.0)


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
    an absolute depth grid, for every (g1, g2) of the gain grid."""
    zeta = float(np.load(os.path.join(RES, "feas_g1500.npz"))["zeta"])
    jobs = [(n, d * 1e-3, g1, g2, zeta) for n in GM_RPM for d in GM_DEPTH for g1 in GM_G1 for g2 in GM_G2]
    with Pool(a.workers) as pool:
        res = pool.map(_gainmap, jobs, chunksize=1)
    shape = (len(GM_RPM), len(GM_DEPTH), len(GM_G1), len(GM_G2))
    kick, sigma, bmax = (np.array([r[k] for r in res]).reshape(shape) for k in range(3))
    msg = np.array([r[3] for r in res]).reshape(shape)
    np.savez_compressed(os.path.join(RES, "c4_gainmap.npz"), rpm=np.array(GM_RPM), depth=np.array(GM_DEPTH),
                        g1=np.array(GM_G1), g2=np.array(GM_G2), kick=kick, sigma=sigma, bmax=bmax, msg=msg,
                        a_lim_ref=np.array([a_lim(n, G_REF) for n in GM_RPM]) * 1e3, zeta=zeta)
    for i, n in enumerate(GM_RPM):
        best = kick[i].reshape(len(GM_DEPTH), -1).max(axis=1)
        ref = kick[i, :, GM_G1.index(G_REF), 0]
        print(f"{n:.0f} rpm: best certified kick (mm/s) " + " ".join(f"{d:.0f}:{b*1e3:.1f}" for d, b in zip(GM_DEPTH, best)))
        print(f"{'':9s}DVF 1500               " + " ".join(f"{d:.0f}:{b*1e3:.1f}" for d, b in zip(GM_DEPTH, ref)))


ENT_RPM = tuple(np.arange(600.0, 3000.1, 100.0))
ENT_G2 = (1000.0, 2000.0)                        # comb gains added to the paper's DVF gain
ENT_NAMES = ("step_clip", "edge_clip")


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
    protocol: nreg 4, D1 classifier) for the comb designs; DVF is in c1_map."""
    from c1_unsafe_zone import build_id
    rpm_map, g_map, zeta = paper_gains()
    g1 = np.interp(ENT_RPM, rpm_map, g_map)
    jobs = [(n, float(g), g2, k, zeta) for n, g in zip(ENT_RPM, g1) for g2 in ENT_G2 for k in ENT_NAMES]
    with Pool(a.workers) as pool:
        lim = np.array(pool.map(_entry, jobs, chunksize=1)).reshape(len(ENT_RPM), len(ENT_G2), len(ENT_NAMES))
    al = np.array([[a_lim_ctrl(n, float(g), g2) for g2 in ENT_G2] for n, g in zip(ENT_RPM, g1)]) * 1e3
    np.savez_compressed(os.path.join(RES, "c4_entry.npz"), rpm=np.array(ENT_RPM), g1=g1, g2=np.array(ENT_G2),
                        names=np.array(ENT_NAMES), a_entry=lim, a_lim=al, zeta=zeta, build=build_id())
    for j, g2 in enumerate(ENT_G2):
        for k, name in enumerate(ENT_NAMES):
            r = lim[:, j, k] / al[:, j]
            print(f"g2 {g2:.0f} {name:9s}: {np.nanmin(lim[:, j, k]):.2f}-{np.nanmax(lim[:, j, k]):.2f} mm, "
                  f"a/a_lim min {np.nanmin(r):.3f} median {np.nanmedian(r):.3f}")


VAL_DEPTH = (6.0, 10.0, 14.0, 18.0, 22.0, 26.0)  # mm, on the gain-map grid


def best_gains(gm, i, j):
    """(g1, g2, kick) of the largest certified kick at gain-map point (i, j)."""
    k = gm["kick"][i, j]
    p, q = np.unravel_index(np.argmax(k), k.shape)
    return float(gm["g1"][p]), float(gm["g2"][q]), float(k[p, q])


def _validate(args):
    rpm, a, g1, g2, zeta = args
    P.set_damping(zeta)
    A, B, K = model(rpm, a, g1, g2)
    _, b, bmax = c2.orbit(rpm, g1, a)
    e = np.zeros(len(B))
    e[1] = 1.0
    size, note, cert = c2.certify(A, B, K, VMAX - bmax, e, rpm, full=True)
    Tp = 60.0 / (rpm * P.N_TEETH)
    vmax, vend = c2.proof_check(A, B, K, b, Tp, cert) if cert is not None else (np.nan, np.nan)
    off = c2.td_offaxis(rpm, a, g1, cert, g2=g2) if cert is not None else (0, 0)
    return size, note, vmax, vend, c2.model_critical_kick(A, B, K, b, Tp), off[0], off[1]


def run_validate(a):
    """At the basin speeds and VAL_DEPTH: for the gain-map's best design and,
    where it is linearly stable, DVF 1500: the simulated critical kick (C1
    protocol, 16 phases).  For the best design also the certificate (tight
    settings), its proof check, the averaged model's critical kick and the
    off-axis test (C2)."""
    from c1_unsafe_zone import _critical_kick, N_PHASE, build_id
    gm = np.load(os.path.join(RES, "c4_gainmap.npz"))
    zeta = float(gm["zeta"])
    pts = []                                      # (rpm, depth mm, g1, g2, kind)
    for i, n in enumerate(gm["rpm"]):
        for d in VAL_DEPTH:
            j = int(np.argmin(np.abs(gm["depth"] - d)))
            g1, g2, k = best_gains(gm, i, j)
            if k > 0:
                pts.append((float(n), d, g1, g2, "best"))
                if d * 1e-3 < a_lim(n, G_REF) and (g1, g2) != (G_REF, 0.0):
                    pts.append((float(n), d, G_REF, 0.0, "dvf"))
    td_jobs = [(n, d * 1e-3, g1, VMAX, p, zeta, g2) for (n, d, g1, g2, _) in pts for p in range(N_PHASE)]
    best = [(n, d * 1e-3, g1, g2, zeta) for (n, d, g1, g2, kind) in pts if kind == "best"]
    with Pool(a.workers) as pool:
        dv = np.array(pool.map(_critical_kick, td_jobs, chunksize=1)).reshape(len(pts), N_PHASE)
        res = pool.map(_validate, best, chunksize=1)
    kind = np.array([p[4] for p in pts])
    nb = len(best)
    cert, vmax, vend, mk, obad, oruns = (np.array([r[k] for r in res]) for k in (0, 2, 3, 4, 5, 6))
    np.savez_compressed(os.path.join(RES, "c4_validate.npz"), rpm=np.array([p[0] for p in pts]),
                        depth=np.array([p[1] for p in pts]), g1=np.array([p[2] for p in pts]),
                        g2=np.array([p[3] for p in pts]), kind=kind, dv=dv, td_min=dv.min(axis=1),
                        cert=cert, model_kick=mk, proof_vmax=vmax, proof_vend=vend, offaxis_bad=obad,
                        offaxis_runs=oruns, msg=np.array([r[1] for r in res]), zeta=zeta, build=build_id())
    tdb = dv[kind == "best"].min(axis=1)
    for (n, d, g1, g2, _), c, m, t, v, ob, orr in zip([p for p in pts if p[4] == "best"], cert, mk, tdb, vmax,
                                                       obad, oruns):
        ref = [dv[i].min() for i, p in enumerate(pts) if p[4] == "dvf" and p[0] == n and p[1] == d]
        refs = f"{ref[0]*1e3:8.2f}" if ref else "     -  "
        print(f"{n:.0f} rpm {d:4.0f} mm  g1 {g1:5.0f} g2 {g2:5.0f}: cert {c*1e3:7.3f}, model {m*1e3:8.2f}, "
              f"TD {t*1e3:8.2f} mm/s (DVF 1500 TD {refs}); max V {v:.4f}; off-axis {ob}/{orr}")
    ok = np.isfinite(tdb) & (cert > 0)
    print(f"certificate below the simulated critical kick at {np.sum(cert[ok] <= tdb[ok])}/{ok.sum()} points; "
          f"below the model's at {np.sum(cert[np.isfinite(mk)] <= mk[np.isfinite(mk)])}/{np.isfinite(mk).sum()}; "
          f"off-axis chatter {obad.sum()}/{oruns.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["limits", "gainmap", "entry", "validate"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dict(limits=run_limits, gainmap=run_gainmap, entry=run_entry, validate=run_validate)[a.command](a)


if __name__ == "__main__":
    main()
