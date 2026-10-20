"""C4 robustness: what the idealised model of C4 leaves out (closed form, ZOA).

The C4 results (c4_comb.py) hold for one structural mode, an instantaneous
loop, a comb delay equal to the tooth period and identical teeth.  Each of
these is relaxed here, in the frequency domain of the averaged (ZOA) model.
The controller is

    U(s) = -exp(-s tau) [ s (g1 + g2 Ec(s)) + kd Ec(s) ] X(s),   Ec = 1 - exp(-s Tc),

and the regenerative force KAPPA a E(s) X(s), E = 1 - exp(-s T), so

    Delta(s) = 1 - G(s) [ KAPPA a E(s) - Ga(s) exp(-s tau) (s (g1 + g2 Ec) + kd Ec) ].

With tau = 0, Tc = T and one mode, Delta = 0 is the characteristic equation
of the collocated model of c4_comb (tests: same stability limits).

  stability     number of zeros of Delta in Re s > 0 by the argument
                principle along s = i w (Delta(0) = Delta(i inf) = 1, G and
                Ga stable): Z = -(change of arg Delta over w in [0, inf)) / pi
  loop delay    tau > 0 (sampling, hold, filters, amplifier)
  second mode   G = G1 + G2 at the same point, mode 2 at f2 with stiffness
                kratio K and damping zeta2 (not identified for this set-up)
  comb delay    Tc = rho T: a speed error or a delay rounded to samples; the
                comb then no longer vanishes on the forced orbit
  runout        eccentric radial runout r_i = rho_r cos(2 pi i / N + psi)
                makes the forced orbit spindle-periodic; the comb, zero only
                at tooth harmonics, then acts on it (spindle harmonics k not a
                multiple of N), and so does the regeneration
  spindle comb  Tc = N T: zero at every spindle harmonic, runout included

Forced orbit with runout: one axial slice gives the static force over a
revolution (the same chip as tdsim.c: h = fz sin(phi) + r_i - r_(i+1), cut only
where h > 0 inside the immersion); its spindle harmonics W_k, integrated over
the depth with the helix lag, give F_k(a), and X_k = G F_k / Delta(i k Omega).
The command's peak over the revolution is compared with the 9 V limit.

    python c4_robust.py        -> ../results/c4_robust.npz
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import params as P
from structure import Ga
import c2_certificate as c2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
KAPPA = c2.KAPPA
VMAX = c2.VMAX
W_MAX = 2 * np.pi * 5000.0                 # rad/s, end of the argument-principle sweep
DW = 0.25                                  # rad/s, its step (resolves 2 zeta w_n = 22 rad/s)


def G_struct(s, mode2=None):
    """Receptance at the cutting point: the identified mode, plus an optional
    second mode (f2 Hz, stiffness ratio to K, damping ratio)."""
    g = 1.0 / (P.M * s ** 2 + P.C * s + P.K)
    if mode2 is not None:
        f2, kr, z2 = mode2
        K2 = kr * P.K
        M2 = K2 / (2 * np.pi * f2) ** 2
        g = g + 1.0 / (M2 * s ** 2 + 2 * z2 * np.sqrt(K2 * M2) * s + K2)
    return g


def controller(s, rpm, g1, g2=0.0, kd=0.0, tau=0.0, rho=1.0):
    """C(s) with U = -C X (V per m of displacement)."""
    Tc = rho * 60.0 / (rpm * P.N_TEETH)
    Ec = 1.0 - np.exp(-s * Tc)
    return np.exp(-s * tau) * (s * (g1 + g2 * Ec) + kd * Ec)


def delta(s, rpm, a, g1, g2=0.0, kd=0.0, tau=0.0, rho=1.0, mode2=None):
    T = 60.0 / (rpm * P.N_TEETH)
    E = 1.0 - np.exp(-s * T)
    return 1.0 - G_struct(s, mode2) * (KAPPA * a * E - Ga(s) * controller(s, rpm, g1, g2, kd, tau, rho))


_W = np.arange(0.0, W_MAX + DW, DW)


def unstable_zeros(rpm, a, g1, g2=0.0, kd=0.0, tau=0.0, rho=1.0, mode2=None):
    """Zeros of Delta in Re s > 0 (argument principle on the imaginary axis)."""
    d = delta(1j * _W, rpm, a, g1, g2, kd, tau, rho, mode2)
    ph = np.unwrap(np.angle(d))
    return int(round(-(ph[-1] - ph[0]) / np.pi))


def a_lim_cf(rpm, g1, g2=0.0, kd=0.0, tau=0.0, rho=1.0, mode2=None, a_max=0.2, rtol=1e-4):
    """First depth (m) with a zero in Re s > 0: scan by 3 % from 0.3 mm, then
    bisection.  0 if the loop is unstable without cutting, inf if stable up
    to a_max."""
    z = lambda a: unstable_zeros(rpm, a, g1, g2, kd, tau, rho, mode2)
    if z(1e-7) != 0:
        return 0.0
    lo, a = 1e-7, 0.3e-3
    while z(a) == 0:
        lo, a = a, a * 1.03
        if a > a_max:
            return np.inf
    hi = a
    while hi - lo > rtol * hi:
        m = 0.5 * (lo + hi)
        if z(m) == 0:
            lo = m
        else:
            hi = m
    return lo


# ------------------------------------------------------ forced orbit ----
N_THETA = 16384                            # samples of one revolution
K_HARM = P.N_TEETH * 40                    # spindle harmonics kept (40 tooth harmonics)
_PHI_ST, _PHI_EX = P.immersion_angles()


def slice_force_revolution(runout=0.0, psi=0.0, n_theta=N_THETA, k_harm=K_HARM):
    """W_k: spindle harmonics of the static force per unit depth (N/m) of one
    axial slice over a revolution, k = 0..k_harm, with eccentric runout."""
    N = P.N_TEETH
    th = np.arange(n_theta) / n_theta * 2 * np.pi
    r = runout * np.cos(2 * np.pi * np.arange(N) / N + psi)
    w = np.zeros(n_theta)
    for i in range(N):
        phi = np.mod(th + 2 * np.pi * i / N, 2 * np.pi)
        cut = (phi >= _PHI_ST) & (phi <= _PHI_EX)
        h = P.FZ * np.sin(phi) + r[i] - r[(i + 1) % N]
        w += np.where(cut & (h > 0), -(P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * h, 0.0)
    return (np.fft.rfft(w) / n_theta)[: k_harm + 1]


def orbit_command(rpm, a, g1, g2=0.0, kd=0.0, tau=0.0, rho=1.0, runout=0.0, psi=0.0, mode2=None,
                  n_t=N_THETA // 4, W=None):
    """Peak |command| (V) of the settled forced motion over one revolution,
    and the harmonics of the command (complex, k = 0..K_HARM)."""
    W = slice_force_revolution(runout, psi) if W is None else W
    k = np.arange(len(W))
    q = k * np.tan(P.HELIX) / (0.5 * P.D_TOOL)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = W * np.where(k == 0, a, (1 - np.exp(-1j * q * a)) / (1j * q))
    Om = 2 * np.pi * rpm / 60.0
    s = 1j * k * Om
    X = np.zeros_like(F)
    X[0] = F[0] / P.K if mode2 is None else F[0] * G_struct(0.0, mode2)
    X[1:] = G_struct(s[1:], mode2) * F[1:] / delta(s[1:], rpm, a, g1, g2, kd, tau, rho, mode2)
    U = np.zeros_like(F)
    U[1:] = -controller(s[1:], rpm, g1, g2, kd, tau, rho) * X[1:]
    t = np.arange(n_t) / n_t * 2 * np.pi
    u = 2 * np.real(np.exp(1j * np.outer(t, k[1:])) @ U[1:])
    return float(np.abs(u).max()), U


def worst_orbit_command(rpm, depths, g1, g2=0.0, kd=0.0, rho=1.0, runout=0.0, n_psi=8, tau=0.0, mode2=None):
    """Largest peak command over depths and n_psi runout phases."""
    best = 0.0
    for psi in (np.arange(n_psi) * 2 * np.pi / n_psi if runout else [0.0]):
        W = slice_force_revolution(runout, psi)
        for a in depths:
            best = max(best, orbit_command(rpm, a, g1, g2, kd, tau, rho, W=W)[0])
    return best


# ------------------------------------------------------------ campaign ----
ROB_RPM = (1090.0, 1420.0, 2500.0, 2870.0)      # the C1 basin speeds
EPS = (-0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03)    # comb-delay error Tc / T - 1
RUNOUT = (0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6)  # eccentric radial runout (m)
TAU = (0.0, 50e-6, 100e-6, 200e-6, 300e-6, 400e-6)                   # loop delay (s)
MODE2 = (None, (400.0, 6.0, 0.01), (800.0, 40.0, 0.01))              # second mode: f2 (Hz), K2 / K, zeta2
N_DEPTH = 40                                    # depths in (0, 0.9 a_lim] for the worst orbit command


def designs(rpm):
    """(label, g1, g2, rho) at a speed: the authority-feasible DVF and d2000
    designs of c4_limits.npz, and d2000 with a spindle-period comb."""
    L = np.load(os.path.join(RES, "c4_limits.npz"))
    names = list(L["feas_names"])
    g = lambda name: float(np.interp(rpm, L["rpm"], L["feas_g1"][:, names.index(name)]))
    return [("DVF", g("dvf"), 0.0, 1.0), ("tooth comb", g("d2000"), 2000.0, 1.0),
            ("spindle comb", g("d2000"), 2000.0, float(P.N_TEETH))]


def _robust(args):
    rpm, lab, g1, g2, rho, zeta = args
    P.set_damping(zeta)
    a0 = a_lim_cf(rpm, g1, g2, rho=rho)
    depths = np.linspace(0.9 * a0 / N_DEPTH, 0.9 * a0, N_DEPTH)
    out = dict(a_lim=a0)
    if g2 > 0:
        out["eps_a_lim"] = [a_lim_cf(rpm, g1, g2, rho=rho * (1 + e)) for e in EPS]
        out["eps_cmd"] = [worst_orbit_command(rpm, depths, g1, g2, rho=rho * (1 + e)) for e in EPS]
    out["runout_cmd"] = [worst_orbit_command(rpm, depths, g1, g2, rho=rho, runout=r) for r in RUNOUT]
    out["tau_a_lim"] = [[a_lim_cf(rpm, g1, g2, tau=t, rho=rho, mode2=m) for t in TAU] for m in MODE2]
    return out


TD_ORBIT = [(2500.0, 20e-3, 1500.0, 2000.0, 1.0, 2e-6, 0.0), (2500.0, 20e-3, 1500.0, 2000.0, 1.0, 5e-6, 1.0),
            (2500.0, 10e-3, 1500.0, 0.0, 1.0, 5e-6, 0.5), (2500.0, 20e-3, 1500.0, 2000.0, 0.99, 0.0, 0.0),
            (2500.0, 20e-3, 1500.0, 2000.0, 1.02, 0.0, 0.0), (1420.0, 15e-3, 1500.0, 2000.0, 0.98, 2e-6, 2.0),
            (2500.0, 20e-3, 1500.0, 2000.0, 4.0, 2e-6, 0.0), (1090.0, 25e-3, 1500.0, 2000.0, 4.0, 3e-6, 0.7)]


def _td_orbit(c):
    """Settled (unclipped, face entry, nreg 4) peak command in the simulator
    against the closed form."""
    from tdsim import simulate
    rpm, a, g1, g2, rho, ro, psi = c
    spr = P.STEPS_PER_REV
    r = simulate(rpm, a, g1, store=True, nrev=300, edge=True, nreg=4, g2=g2, comb_delay=rho, runout=ro,
                 runout_phase=psi)
    drift = float(np.abs(r["x"][-spr:] - r["x"][-2 * spr:-spr]).max() / np.abs(r["x"][-spr:]).max())
    return float(np.abs(r["vcmd"][-spr:]).max()), orbit_command(rpm, a, g1, g2, rho=rho, runout=ro, psi=psi)[0], drift


TD_LIMIT = [(2500.0, 1500.0, 2000.0, 4.0), (1090.0, 1500.0, 2000.0, 4.0), (2500.0, 1500.0, 2000.0, 0.98),
            (2500.0, 1500.0, 2000.0, 1.02)]


def _td_limit(c):
    """Unclipped step-entry stability limit in the simulator against a_lim_cf."""
    from tdboundary import boundary
    rpm, g1, g2, rho = c
    am = a_lim_cf(rpm, g1, g2, rho=rho)
    return boundary(rpm, g1, a_lo=0.8 * am, a_hi=1.3 * am, growth=1.04, g2=g2, comb_delay=rho), am


TD_FACE_EPS = (-0.02, -0.01, 0.0, 0.01, 0.02)


def _td_face(e):
    """Face-entry limit with the 9 V clip (C1 protocol) of the feasible d2000
    design at 2500 rpm with a comb-delay error e."""
    from tdboundary import boundary
    from c1_unsafe_zone import ENTRIES, LADDER
    rpm = 2500.0
    _, g1, g2, _ = designs(rpm)[1]
    am = a_lim_cf(rpm, g1, g2, rho=1 + e)
    vmax, entry = ENTRIES["edge_clip"]
    return boundary(rpm, g1, vmax=vmax, a_lo=LADDER["lo"] * am, a_hi=LADDER["hi"] * am, growth=LADDER["growth"],
                    g2=g2, comb_delay=1 + e, **entry), am


def run(a):
    import tdsim
    zeta = float(np.load(os.path.join(RES, "feas_g1500.npz"))["zeta"])
    jobs = [(n, lab, g1, g2, rho, zeta) for n in ROB_RPM for (lab, g1, g2, rho) in designs(n)]
    with Pool(a.workers) as pool:
        res = pool.map(_robust, jobs, chunksize=1)
        tdo = pool.map(_td_orbit, TD_ORBIT, chunksize=1)
        tdl = pool.map(_td_limit, TD_LIMIT, chunksize=1)
        tdf = pool.map(_td_face, TD_FACE_EPS, chunksize=1)
    nd = len(designs(ROB_RPM[0]))
    shp = (len(ROB_RPM), nd)
    nan_eps = [np.nan] * len(EPS)
    out = dict(rpm=np.array(ROB_RPM), labels=np.array([d[0] for d in designs(ROB_RPM[0])]),
               g1=np.array([j[2] for j in jobs]).reshape(shp), g2=np.array([j[3] for j in jobs]).reshape(shp),
               rho=np.array([j[4] for j in jobs]).reshape(shp), eps=np.array(EPS), runout=np.array(RUNOUT),
               tau=np.array(TAU), mode2=np.array([m if m else (np.nan,) * 3 for m in MODE2]),
               a_lim=np.array([r["a_lim"] for r in res]).reshape(shp) * 1e3,
               eps_a_lim=np.array([r.get("eps_a_lim", nan_eps) for r in res]).reshape(shp + (len(EPS),)) * 1e3,
               eps_cmd=np.array([r.get("eps_cmd", nan_eps) for r in res]).reshape(shp + (len(EPS),)),
               runout_cmd=np.array([r["runout_cmd"] for r in res]).reshape(shp + (len(RUNOUT),)),
               tau_a_lim=np.array([r["tau_a_lim"] for r in res]).reshape(shp + (len(MODE2), len(TAU))) * 1e3,
               td_orbit_cases=np.array(TD_ORBIT), td_orbit=np.array(tdo),
               td_limit_cases=np.array(TD_LIMIT), td_limit=np.array(tdl) * 1e3,
               td_face_eps=np.array(TD_FACE_EPS), td_face=np.array(tdf) * 1e3,
               zeta=zeta, build=tdsim.SOURCE_SHA[:12])
    np.savez_compressed(os.path.join(RES, "c4_robust.npz"), **out)
    for i, n in enumerate(ROB_RPM):
        for j, lab in enumerate(out["labels"]):
            rc = out["runout_cmd"][i, j]
            tol = np.interp(VMAX, rc, RUNOUT) * 1e6 if rc[-1] >= VMAX and rc[0] < VMAX else (
                np.inf if rc[-1] < VMAX else 0.0)
            line = (f"{n:.0f} rpm {lab:12s} g1 {out['g1'][i, j]:6.0f} g2 {out['g2'][i, j]:5.0f}: a_lim {out['a_lim'][i, j]:6.2f} mm; "
                    f"runout tolerance {tol:.2f} um; a_lim at tau = 400 us: {out['tau_a_lim'][i, j, 0, -1]:.2f} "
                    f"(mode 400 Hz: {out['tau_a_lim'][i, j, 1, -1]:.2f}, 800 Hz: {out['tau_a_lim'][i, j, 2, -1]:.2f})")
            if out["g2"][i, j] > 0:
                line += (f"; eps -2/+2 %: a_lim {out['eps_a_lim'][i, j, 1]:.2f}/{out['eps_a_lim'][i, j, -2]:.2f}, "
                         f"worst orbit {out['eps_cmd'][i, j, 1]:.2f}/{out['eps_cmd'][i, j, -2]:.2f} V")
            print(line)
    for c, (td, cf, dr) in zip(TD_ORBIT, tdo):
        print(f"TD orbit {c}: simulator {td:.3f} V, closed form {cf:.3f} V (ratio {td / cf:.3f}, drift {dr:.0e})")
    for c, (td, cf) in zip(TD_LIMIT, out["td_limit"]):
        print(f"TD limit {c}: simulator {td:.2f} mm, closed form {cf:.2f} mm (ratio {td / cf:.4f})")
    for e, (td, cf) in zip(TD_FACE_EPS, out["td_face"]):
        print(f"face entry, 9 V, 2500 rpm, comb-delay error {e * 100:+.0f} %: {td:.2f} mm (linear limit {cf:.2f} mm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    run(ap.parse_args())
