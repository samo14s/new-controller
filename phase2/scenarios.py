"""
scenarios.py — PHASES 12, 13, 14: the comparison protocol.
==========================================================
Nothing changes between controllers except the controller: same plate, same
patch, same sensor, same sign, same cutting parameters, same disturbance, same
initial conditions, same integrator, same Floquet resolution.

SCENARIOS
  S1  nominal                 Delta = 0, tau = tau_0
  S2  material removal        eta = 0 .. eta_max (physics) and the paper's
                              symmetric +/-10 % box, for comparison
  S3  stiffness and damping   zeta x {0.8, 1.0, 1.2}, stiffness +/-10 %
  S4  delay                   tau / tau_0 = 1, 1.25, 1.5, 1.75, 2 (spindle speed)

METRICS
  a_p,lim   axial stability limit by Floquet bisection on the 5-mode model
  A_max     peak displacement at the sensor, um
  A_rms     root mean square displacement, um
  T_s       settling time to the steady envelope, s
  E_u       control effort  int u^2 dt, V^2 s
  u_max     peak voltage, V
  delta_max certified uncertainty margin  (certify.py)
  tau_max   delay margin, from the Floquet sweep over spindle speed
"""
import numpy as np

import config as C
from objective import limits
from simulate import MillingSimulation
from sim_ctrl import Combined
from closed_loop import limit as floquet_limit


# ---------------------------------------------------------------------------
def mode_scale_from_removal(removal, eta, n):
    """omega(eta) / omega(0) per mode, from the physics-based family."""
    f0 = removal.frequencies(0.0)
    fe = removal.frequencies(eta)
    r = np.asarray(fe[:n], float) / np.asarray(f0[:n], float)
    if r.size < n:
        r = np.concatenate([r, np.ones(n - r.size)])
    return r


def paper_box_scale(dm, dk, n):
    """omega ratio for the paper's symmetric box: omega^2 = (1+dk)/(1+dm)."""
    return np.full(n, np.sqrt((1.0 + dk) / (1.0 + dm)))


# ---------------------------------------------------------------------------
def time_metrics(res, dt=None):
    y = np.asarray(res['y_obs'], float)
    u = np.asarray(res['u'], float)
    t = np.asarray(res['t'], float)
    dt = res['dt'] if dt is None else dt
    out = dict(diverged=bool(res['diverged']),
               t_div=res['t_div'],
               A_max=float(np.max(np.abs(y))) * 1e6,
               A_rms=float(np.sqrt(np.mean(y ** 2))) * 1e6,
               E_u=float(np.sum(u ** 2) * dt),
               u_max=float(np.max(np.abs(u))),
               u_mean=float(np.mean(np.abs(u[int(0.05 * u.size):]))))
    # settling time: last instant at which |y| leaves a band of 1.5 times the
    # RMS of the final quarter of the record
    n = y.size
    env = 1.5 * float(np.sqrt(np.mean(y[int(0.75 * n):] ** 2)))
    idx = np.where(np.abs(y) > env)[0]
    out['T_s'] = float(t[idx[-1]]) if idx.size else 0.0
    return out


def settling_free(plate, ss, pd, y0_um=10.0, T=0.5, n_sub=None, band=0.05):
    """Settling time of the FREE response (no cutting), from an initial modal
    displacement.  A milling pass is forced continuously, so the settling time
    of the cut is not a controller property; the free decay is."""
    n_sub = C.N_SUB_TIME if n_sub is None else n_sub
    sim = MillingSimulation(plate, C.RPM_S, 1e-9, ae=C.AE, fz=0.0,
                            sign=C.SIGN, n_modes=C.N_MODES, n_sub=n_sub)
    dt, tau = sim.dt, sim.tau
    ctrl = None if ss is None else Combined(ss, pd, dt, tau)
    n = sim.n
    q0 = np.zeros(n)
    D = sim.D_obs
    q0 += D / float(D @ D) * (y0_um * 1e-6)
    res = _free_run(sim, ctrl, q0, T)
    y = np.abs(res['y'])
    thr = band * float(np.max(y))
    idx = np.where(y > thr)[0]
    return dict(T_s=float(res['t'][idx[-1]]) if idx.size else 0.0,
                E_u=float(np.sum(res['u'] ** 2) * dt),
                u_max=float(np.max(np.abs(res['u']))),
                A_max=float(np.max(y)) * 1e6)


def _free_run(sim, ctrl, q0, T):
    n, dt = sim.n, sim.dt
    nstep = int(round(T / dt)) + 1
    q = q0.copy()
    qd = np.zeros(n)
    qdd = np.zeros(n)
    g, b = sim.g, sim.b
    S0inv = sim.S0inv
    t = np.arange(nstep) * dt
    y = np.zeros(nstep)
    u_s = np.zeros(nstep)
    for k in range(1, nstep):
        yk = float(sim.D_obs @ q)
        ydk = float(sim.D_obs @ qd)
        y[k - 1] = yk
        u = 0.0 if ctrl is None else float(ctrl(y=yk, yd=ydk, t=t[k], k=k))
        u_s[k] = u
        qdp = qd + (1 - g) * dt * qdd
        qp = q + dt * qd + (0.5 - b) * dt ** 2 * qdd
        rhs = sim.H * u - sim.C @ qdp - sim.K @ qp
        qdd = S0inv @ rhs
        qd = qdp + g * dt * qdd
        q = qp + b * dt ** 2 * qdd
    y[-1] = float(sim.D_obs @ q)
    return dict(t=t, y=y, u=u_s)


def run_time(plate, ss, pd, rpm, ap, mode_scale=None, zeta_scale=1.0,
             T=None, moving=True, n_sub=None, ctrl_obj=None):
    n_sub = C.N_SUB_TIME if n_sub is None else n_sub
    sim = MillingSimulation(plate, rpm, ap, ae=C.AE, fz=C.FZ, sign=C.SIGN,
                            n_modes=C.N_MODES, n_sub=n_sub,
                            mode_scale=mode_scale, zeta_scale=zeta_scale)
    if ss is None and ctrl_obj is None:
        ctrl = None
    elif ctrl_obj is not None:
        ctrl = ctrl_obj
        if hasattr(ctrl, 'reset'):
            ctrl.reset()
    else:
        ctrl = Combined(ss, pd, sim.dt, sim.tau)
    return time_metrics(sim.run(controller=ctrl, T=T, moving=moving))


# ---------------------------------------------------------------------------
def floquet_limits(plate, ss, pd, rpm, positions=None, m=None, mode_scale=None,
                   zeta_scale=1.0, lo=0.005e-3, hi=3.0e-3, tol=5e-6):
    """a_p,lim at each position, on a plate optionally perturbed."""
    pos = C.POSITIONS if positions is None else positions
    m = C.M_FLOQUET if m is None else m
    pl = _perturbed(plate, mode_scale, zeta_scale)
    return np.array([floquet_limit(pl, rpm, fr * pl.lp, ctrl=ss, pd=pd,
                                   lo=lo, hi=hi, tol=tol, n_modes=C.N_MODES,
                                   m=m, n_period=C.N_PERIOD,
                                   coeff_mode='time', coeff_scale=C.SIGN,
                                   ae=C.AE)
                     for fr in pos])


class _Shim:
    """A plate view with perturbed modal parameters, same mode shapes."""

    def __init__(self, plate, omega, zeta):
        self._p = plate
        self.omega_n = omega
        self.zeta_modes = zeta

    def __getattr__(self, k):
        return getattr(self._p, k)


def _perturbed(plate, mode_scale=None, zeta_scale=1.0):
    if mode_scale is None and zeta_scale == 1.0:
        return plate
    w = np.asarray(plate.omega_n, float).copy()
    z = np.asarray(plate.zeta_modes, float).copy() * zeta_scale
    if mode_scale is not None:
        ms = np.asarray(mode_scale, float)
        w[:ms.size] = w[:ms.size] * ms
    return _Shim(plate, w, z)


# ---------------------------------------------------------------------------
def delay_margin(plate, ss, pd, ap, rpm_lo=1500, rpm_hi=9000, n=31, m=None):
    """Spindle-speed sweep at fixed depth: which delays keep the loop stable.

    tau = 60 / (N_T rpm), so a low spindle speed is a long delay.  Reported as
    the widest interval of tau / tau_0 containing the design point over which
    every speed tested is stable.
    """
    from closed_loop import is_stable
    m = C.M_FLOQUET if m is None else m
    rpms = np.unique(np.concatenate([np.linspace(rpm_lo, rpm_hi, n),
                                     [C.RPM_S]]))
    tau0 = 60.0 / (3 * C.RPM_S)
    ok = []
    for r in rpms:
        st = all(is_stable(plate, float(r), ap, fr * plate.lp, ctrl=ss, pd=pd,
                           n_modes=C.N_MODES, m=m, n_period=C.N_PERIOD,
                           coeff_mode='time', coeff_scale=C.SIGN,
                           ae=C.AE)[0]
                 for fr in (0.0, 0.5, 1.0))
        ok.append(st)
    ok = np.array(ok)
    i0 = int(np.argmin(np.abs(rpms - C.RPM_S)))
    if not ok[i0]:
        return dict(rpms=rpms, ok=ok, tau_lo=np.nan, tau_hi=np.nan,
                    ratio_lo=np.nan, ratio_hi=np.nan)
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    tau_hi = 60.0 / (3 * rpms[lo])          # low speed  = long delay
    tau_lo = 60.0 / (3 * rpms[hi])
    return dict(rpms=rpms, ok=ok, tau_lo=tau_lo, tau_hi=tau_hi,
                ratio_lo=tau_lo / tau0, ratio_hi=tau_hi / tau0)
