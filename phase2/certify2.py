"""
certify2.py — STAGES 5 and 6: stability over the position domain, and margins.
==============================================================================
Stage 5 asks for a condition valid over the admissible milling-position domain,
for a closed loop that keeps the regenerative delay:

    xdot(t) = [A(x_P, t) - B K(x_P)] x(t) + A_d(x_P, t) x(t - tau)

Two instruments, and the difference between them is reported rather than hidden.

(1) EXACT crossing analysis.  For the frozen-coefficient loop, a characteristic
    root reaches the imaginary axis at frequency omega only if some eigenvalue
    of  G(omega) = (j omega I - A_cl)^-1 A_d,cl  has unit modulus, and then

        e^{-j omega tau} = 1 / lambda_k(omega)   =>   tau = (arg lambda_k + 2 pi m) / omega .

    The smallest positive such tau over all omega and all k is tau_max: the loop
    is stable for every delay in [0, tau_max).  If no eigenvalue ever reaches
    unit modulus the loop is stable for EVERY delay and tau_max = infinity.
    This answers Stage 6 exactly, with no Lyapunov conservatism.

(2) COMMON-P Lyapunov-Krasovskii certificate over the whole vertex family.  It
    is more conservative, and it buys the one thing (1) cannot give: validity
    when the position, the removal state and the milling-force coefficient move
    ARBITRARILY FAST inside their domains -- which is what actually happens,
    since alpha_4(t) has a peak-to-mean ratio of 12.4 within one tooth period.

The vertex family is where the proposed controller earns its margin.  For a
fixed controller the position is an UNCERTAINTY axis and every position must be
covered by one gain.  For the scheduled controller the position is a SCHEDULING
axis: each vertex carries the gain designed for it, and the set the certificate
must span is correspondingly smaller.
"""
import numpy as np
from scipy.linalg import matrix_balance

import config as C
import lk_lmi as L


# ---------------------------------------------------------------------------
def augment(A, Ad, B, Cy, ss, pd, n):
    """Closed loop in physical coordinates."""
    if ss is None:
        Ac = np.zeros((0, 0))
        Bc = np.zeros((0, 1))
        Cc = np.zeros((1, 0))
        Dc = np.zeros((1, 1))
    else:
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in ss]
        Bc = Bc.reshape(-1, 1)
        Cc = Cc.reshape(1, -1)
        Dc = Dc.reshape(1, 1)
        Ac = Ac.reshape(Bc.shape[0], Bc.shape[0])
    nc = Ac.shape[0]
    npl = A.shape[0]
    ncl = npl + nc
    Acl = np.zeros((ncl, ncl))
    Adcl = np.zeros((ncl, ncl))
    Acl[:npl, :npl] = A + float(Dc[0, 0]) * (B @ Cy)
    Adcl[:npl, :npl] = Ad
    if nc:
        Acl[:npl, npl:] = B @ Cc
        Acl[npl:, :npl] = Bc @ Cy
        Acl[npl:, npl:] = Ac
    if pd is not None:
        Cyq = np.zeros((1, npl))
        Cyv = np.zeros((1, npl))
        Cyq[0, :n] = Cy[0, :n]
        Cyv[0, n:] = Cy[0, :n]
        Adcl[:npl, :npl] += B @ (float(pd[0]) * Cyq + float(pd[1]) * Cyv)
    return Acl, Adcl, npl


def _scale(Acl, Adcl, n, npl, ws):
    d = np.concatenate([ws * np.ones(n), np.ones(n),
                        np.ones(Acl.shape[0] - npl)])
    S = np.diag(d)
    Si = np.diag(1.0 / d)
    return S @ Acl @ Si / ws, S @ Adcl @ Si / ws


def vertices(plant, ctrl, positions=None, etas=(0.0,), a_scale=1.0,
             zetas=(1.0,), xis=(1.0,), n_pos=9, ws=None, balance=True):
    """[(A_cl, A_d,cl)] scaled, plus n_plant.

    `positions` is the SCHEDULING domain, and each vertex uses the controller
    the schedule assigns to it.  A fixed controller returns the same gain
    everywhere, which is exactly why it has to cover the whole domain with one
    design.
    """
    ws = 2 * np.pi * 800.0 if ws is None else ws
    xs = (np.linspace(0.0, plant.plate.lp, n_pos) if positions is None
          else np.asarray(positions, float))
    a_nom = plant.a40
    a_lo = a_nom + (C.ALPHA_LO * plant.abar4 - a_nom) * a_scale
    a_hi = a_nom + (C.ALPHA_HI * plant.abar4 - a_nom) * a_scale
    out = []
    npl_s = None
    Tc = None
    for x in xs:
        for eta in etas:
            for xi in xis:
                for z in zetas:
                    ss, pd = ctrl.at(float(x), float(eta))
                    for a in (a_lo, a_hi):
                        A, Ad, B, _, Cy = plant.matrices(
                            x_pos=float(x), a4=a, eta=float(eta), xi=float(xi),
                            zeta_scale=float(z))
                        Acl, Adcl, npl = augment(A, Ad, B, Cy, ss, pd, plant.n)
                        Acl, Adcl = _scale(Acl, Adcl, plant.n, npl, ws)
                        if balance and Acl.shape[0] > npl:
                            if Tc is None:
                                try:
                                    _, T = matrix_balance(Acl[npl:, npl:])
                                    dd = np.concatenate(
                                        [np.ones(npl), np.abs(np.diag(T))])
                                except Exception:
                                    dd = np.ones(Acl.shape[0])
                                if (not np.all(np.isfinite(dd))
                                        or np.min(dd) <= 0):
                                    dd = np.ones(Acl.shape[0])
                                Tc = np.diag(dd)
                                Tci = np.diag(1.0 / dd)
                            Ab, Adb = Tci @ Acl @ Tc, Tci @ Adcl @ Tc
                            if np.all(np.isfinite(Ab)) and np.all(np.isfinite(Adb)):
                                Acl, Adcl = Ab, Adb
                        out.append((Acl, Adcl))
                        npl_s = npl
    return out, npl_s, ws


# ---------------------------------------------------------------------------
def _wgrid(A, n=250, n_local=41, span=8.0):
    """Frequency grid that RESOLVES the resonances.

    The plate has zeta ~ 0.003, so the half-power width of a mode is omega/2Q ~
    10 rad/s while a 400-point log grid up to 3e4 rad/s has a spacing of ~100
    rad/s near the modes.  A plain log grid therefore steps straight over every
    peak and reports a spectral radius that is far too small.  Each closed-loop
    eigenvalue gets its own local grid, several damping widths wide.
    """
    ev = np.linalg.eigvals(A)
    r = np.abs(ev)
    hi = 5.0 * max(float(r.max()), 1.0)
    g = [np.array([0.0]), np.logspace(-3, np.log10(hi), n)]
    for lam in ev:
        wk = abs(float(np.imag(lam)))
        if wk <= 0.0:
            continue
        half = max(abs(float(np.real(lam))), 1e-6 * wk)
        g.append(np.linspace(max(wk - span * half, 0.0), wk + span * half,
                             n_local))
    return np.unique(np.concatenate(g))


def crossing_delays(Acl, Adcl, w=None):
    """(tau_max, peak) for one frozen-coefficient closed loop.

    tau_max = inf means stable for every delay.  peak is
    sup_omega rho((j omega I - A)^-1 A_d): below 1 no crossing exists at all.
    """
    if not (np.all(np.isfinite(Acl)) and np.all(np.isfinite(Adcl))):
        return 0.0, np.inf
    try:
        if float(np.max(np.linalg.eigvals(Acl).real)) >= 0.0:
            return 0.0, np.inf
    except np.linalg.LinAlgError:
        return 0.0, np.inf
    w = _wgrid(Acl) if w is None else w
    I = np.eye(Acl.shape[0])
    peak = 0.0
    tau = np.inf
    for wk in w:
        try:
            lam = np.linalg.eigvals(np.linalg.solve(1j * wk * I - Acl, Adcl))
        except np.linalg.LinAlgError:
            return 0.0, np.inf
        mag = np.abs(lam)
        peak = max(peak, float(mag.max()))
        if wk <= 0.0:
            continue
        for lk, mk in zip(lam, mag):
            if mk >= 1.0:
                th = float(np.angle(lk)) % (2 * np.pi)
                tau = min(tau, th / wk)
    return tau, peak


def analyse(plant, ctrl, **kw):
    """(tau_max, peak, all_delay_stable) over the whole vertex family.

    tau_max comes back in SECONDS.  The vertices live in scaled time (t~ = ws t)
    for conditioning, so the crossing delays are produced in scaled units and
    are divided by ws on the way out.
    """
    V, npl, ws = vertices(plant, ctrl, **kw)
    tau = np.inf
    peak = 0.0
    for (A, Ad) in V:
        t, p = crossing_delays(A, Ad)
        tau = min(tau, t)
        peak = max(peak, p)
    return (tau if np.isinf(tau) else tau / ws), peak, bool(peak < 1.0)


def di_stable(plant, ctrl, **kw):
    return analyse(plant, ctrl, **kw)[2]


# ---------------------------------------------------------------------------
def depth_bisect(plant_of, ctrl_of, lo=2e-6, hi=4e-3, tol=2e-6, n_iter=24,
                 **kw):
    """Largest a_p at which the whole family is stable for EVERY delay."""
    def ok(ap):
        p = plant_of(ap)
        return di_stable(p, ctrl_of(p), **kw)

    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def margin_bisect(plant, ctrl, base_kw=None, lo=0.0, hi=8.0, tol=1e-2,
                  n_iter=20, eta_max=None, zeta_span=None):
    """Largest inflation of the uncertainty set still stable for every delay.

    The inflation scales the three genuinely uncertain quantities together:
    alpha_4 about its nominal, the removal range, and the damping multiplier.
    The milling position is NOT inflated: it is a scheduling signal for the
    proposed controller and a fixed domain for the others.
    """
    base_kw = dict(base_kw or {})
    eta_max = C.ETA_MAX if eta_max is None else eta_max
    zeta_span = (C.ZETA_LO, C.ZETA_HI) if zeta_span is None else zeta_span

    def ok(sc):
        kw = dict(base_kw)
        kw['a_scale'] = sc
        kw['etas'] = tuple(np.linspace(0.0, eta_max * sc, 3))
        kw['zetas'] = (1.0 + (zeta_span[0] - 1.0) * sc,
                       1.0 + (zeta_span[1] - 1.0) * sc)
        return di_stable(plant, ctrl, **kw)

    if not ok(max(lo, 1e-3)):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
def lk_common_P(plant, ctrl, **kw):
    """Common-P Lyapunov-Krasovskii certificate over the vertex family."""
    V, npl, _ = vertices(plant, ctrl, **kw)
    return L.certificate_di(V, eps=0.0, n_plant=npl)
