"""
eval2.py — the one evaluation path, shared by all four controllers.
====================================================================
Same plant, same actuator, same sensor, same conditions, same numerical grid,
same objective, same constraints.  A controller that schedules on the milling
position is asked for its gain AT that position; a fixed one returns the same
gain whatever it is asked.  Nothing else differs.

Objective (to maximise), the same for every structure:

    J = - mean_{a_p in probes} max_{x_P in positions} log rho( a_p, x_P )

with rho the Floquet spectral radius of the delayed periodic closed loop on the
FIVE-mode evaluation model.  J > 0 means stable at every probe depth and every
position, and J grows with the margin.

Constraints, screened before any Floquet work:
  1. nominal closed loop (no cutting) has all poles <= -1 s^-1, at every
     scheduling point;
  2. modulus margin max_w |S| <= MS_MAX;
  3. actuator effort max_w |K S P_f| <= V_PER_N volts per newton.
"""
import numpy as np

import config as C
from closed_loop import period_maps, spectral_radius, limit as _floq_limit
from fopid import ss_frf
from plate_model import plant_frf, plant_vectors

_F_CON = np.logspace(0.5, 4.1, 140)


# ---------------------------------------------------------------------------
def _ss_arrays(ss):
    if ss is None:
        return (np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)),
                np.zeros((1, 1)))
    A, B, Cc, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    B = B.reshape(-1, 1)
    Cc = Cc.reshape(1, -1)
    D = D.reshape(1, 1)
    return A.reshape(B.shape[0], B.shape[0]), B, Cc, D


def nominal_poles(plate, ss, pd=None, n_modes=None):
    n = C.N_MODES if n_modes is None else n_modes
    w, z, H, D_obs, _ = plant_vectors(plate, n)
    Ac, Bc, Cc, Dc = _ss_arrays(ss)
    nc = Ac.shape[0]
    A = np.zeros((2 * n + nc, 2 * n + nc))
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -np.diag(w ** 2) + float(Dc[0, 0]) * np.outer(H, D_obs)
    A[n:2 * n, n:2 * n] = -np.diag(2 * z * w)
    if nc:
        A[n:2 * n, 2 * n:] = np.outer(H, Cc.ravel())
        A[2 * n:, :n] = np.outer(Bc.ravel(), D_obs)
        A[2 * n:, 2 * n:] = Ac
    if pd is not None:
        A[n:2 * n, :n] += float(pd[0]) * np.outer(H, D_obs)
        A[n:2 * n, n:2 * n] += float(pd[1]) * np.outer(H, D_obs)
    return np.linalg.eigvals(A)


def frequency_metrics(plate, ss, pd=None, positions=None, f=None, tau=None):
    f = _F_CON if f is None else np.asarray(f, float)
    pos = C.POSITIONS_DESIGN if positions is None else positions
    om = 2 * np.pi * f
    K = ss_frf(ss, om) if ss is not None else np.zeros_like(om, complex)
    if pd is not None:
        tau = 60.0 / (3 * C.RPM_DESIGN) if tau is None else tau
        K = K + (float(pd[0]) + 1j * om * float(pd[1])) * np.exp(-1j * om * tau)
    Pu, _ = plant_frf(plate, f, C.N_MODES)
    S = 1.0 / (1.0 - Pu * K)
    Ms = float(np.max(np.abs(S)))
    v = 0.0
    for fr in pos:
        _, Pf = plant_frf(plate, f, C.N_MODES, x_force=fr * plate.lp)
        v = max(v, float(np.max(np.abs(K * S * Pf))))
    return Ms, v


# ---------------------------------------------------------------------------
def floquet_log_rho(plate, ctrl, rpm, ap, x_pos, eta=0.0, m=None,
                    n_period=None, sign=None):
    ss, pd = ctrl.at(x_pos, eta)
    m = C.M_FLOQUET_PSO if m is None else m
    npd = C.N_PERIOD if n_period is None else n_period
    maps, _ = period_maps(plate, rpm, ap, x_pos, ctrl=ss, pd=pd,
                          n_modes=C.N_MODES, m=m, coeff_mode='time',
                          coeff_scale=C.SIGN if sign is None else sign,
                          ae=C.AE)
    rho = spectral_radius(maps, m, maps[0][0].shape[0], npd)
    if not np.isfinite(rho):
        return 50.0
    return float(np.clip(np.log(max(rho, 1e-300)), -50.0, 50.0))


def evaluate(plate, ctrl, rpm=None, probes=None, positions=None, m=None,
             detail=False):
    try:
        return _evaluate(plate, ctrl, rpm, probes, positions, m, detail)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info = dict(feasible=False, reason='numerical failure', Ms=np.nan,
                    V=np.nan, J=-1e4, max_re=np.nan)
        return (info['J'], info) if detail else info['J']


def _evaluate(plate, ctrl, rpm, probes, positions, m, detail):
    rpm = C.RPM_DESIGN if rpm is None else rpm
    probes = C.AP_PROBE if probes is None else probes
    pos = C.POSITIONS_DESIGN if positions is None else positions
    info = dict(feasible=False, reason='', Ms=np.nan, V=np.nan, J=-np.inf)

    mre = -np.inf
    for fr in pos:
        ss, pd = ctrl.at(fr * plate.lp)
        mre = max(mre, float(np.max(nominal_poles(plate, ss, pd).real)))
    info['max_re'] = mre
    if not np.isfinite(mre) or mre > -1.0:
        info['reason'] = 'nominal loop unstable'
        info['J'] = -1e3 - max(mre, 0.0)
        return (info['J'], info) if detail else info['J']

    Ms, V = -np.inf, -np.inf
    for fr in pos:
        ss, pd = ctrl.at(fr * plate.lp)
        a, b = frequency_metrics(plate, ss, pd, pos)
        Ms, V = max(Ms, a), max(V, b)
    info['Ms'], info['V'] = Ms, V
    pen = 0.0
    if Ms > C.MS_MAX:
        pen += 10.0 * (Ms / C.MS_MAX - 1.0)
    if V > C.V_PER_N:
        pen += 10.0 * (V / C.V_PER_N - 1.0)
    if pen > 0.0:
        info['reason'] = f'constraint violated (Ms={Ms:.2f}, V={V:.0f} V/N)'
        info['J'] = -100.0 - pen
        return (info['J'], info) if detail else info['J']

    margins = []
    for ap in probes:
        margins.append(max(floquet_log_rho(plate, ctrl, rpm, ap, fr * plate.lp,
                                           m=m) for fr in pos))
    J = -float(np.mean(margins))
    info.update(feasible=True, J=J, margins=margins)
    return (J, info) if detail else J


# ---------------------------------------------------------------------------
def limits(plate, ctrl, rpm=None, positions=None, m=None, eta=0.0,
           lo=0.005e-3, hi=3.0e-3, tol=5e-6, sign=None):
    """a_p,lim at each position, full-resolution Floquet bisection."""
    rpm = C.RPM_S if rpm is None else rpm
    pos = C.POSITIONS if positions is None else positions
    m = C.M_FLOQUET if m is None else m
    out = []
    for fr in pos:
        x = fr * plate.lp
        ss, pd = ctrl.at(x, eta)
        out.append(_floq_limit(plate, rpm, x, ctrl=ss, pd=pd, lo=lo, hi=hi,
                               tol=tol, n_modes=C.N_MODES, m=m,
                               n_period=C.N_PERIOD, coeff_mode='time',
                               coeff_scale=C.SIGN if sign is None else sign,
                               ae=C.AE))
    return np.array(out)
