"""
certify.py — assembling the closed loop for the Lyapunov-Krasovskii certificate.
================================================================================
Physical closed loop, with the controller and the delayed PD of Eq. (30):

    xdot_p = A x_p + A_d x_p(t-tau) + B u
    xdot_c = A_c x_c + B_c y ,           y = C_y x_p
    u      = C_c x_c + D_c y + K_Pp y(t-tau) + K_Pd ydot(t-tau)

    =>  A_cl   = [ A + B D_c C_y     B C_c ]
                 [ B_c C_y           A_c   ]
        A_d,cl = [ A_d + B (K_Pp C_yq + K_Pd C_yv)   0 ]
                 [ 0                                 0 ]

Both are then carried into the scaled coordinates of `uncertainty.py`, and the
controller block alone is diagonally balanced.  That last step is safe for the
certificate because the norm-bounded residual acts only on the plant block, and
a transformation that is the identity there leaves both E and F untouched.
"""
import numpy as np
from scipy.linalg import matrix_balance

import lk_lmi as L


def _augment(A, Ad, B, Cy, ss, pd, n):
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


def closed_loop_vertices(uset, ss=None, pd=None, scale=1.0, n_eta=3, n_x=9,
                         balance=True, eta_range=None):
    """[(A_cl, A_d,cl)] in scaled coordinates, plus n_plant."""
    n = uset.n
    ws = uset.ws
    out = []
    npl_s = None
    Tc = None
    for (A, Ad, B, Cy) in uset.vertices(scale, n_eta, n_x, eta_range):
        Acl, Adcl, npl = _augment(A, Ad, B, Cy, ss, pd, n)
        S = np.diag(np.concatenate([ws * np.ones(n), np.ones(n),
                                    np.ones(Acl.shape[0] - npl)]))
        Si = np.linalg.inv(S)
        Acl = S @ Acl @ Si / ws
        Adcl = S @ Adcl @ Si / ws
        if balance and Acl.shape[0] > npl:
            if Tc is None:
                try:
                    _, T = matrix_balance(Acl[npl:, npl:])
                    d = np.concatenate([np.ones(npl), np.abs(np.diag(T))])
                except Exception:
                    d = np.ones(Acl.shape[0])
                if not np.all(np.isfinite(d)) or np.min(d) <= 0:
                    d = np.ones(Acl.shape[0])
                Tc = np.diag(d)
                Tci = np.diag(1.0 / d)
            Ab = Tci @ Acl @ Tc
            Adb = Tci @ Adcl @ Tc
            if np.all(np.isfinite(Ab)) and np.all(np.isfinite(Adb)):
                Acl, Adcl = Ab, Adb
        out.append((Acl, Adcl))
        npl_s = npl
    return out, npl_s


def certify(uset, ss=None, pd=None, scale=1.0, eps=None, n_eta=3, n_x=9,
            delay_independent=True, tau=None, **kw):
    V, npl = closed_loop_vertices(uset, ss, pd, scale, n_eta, n_x)
    if eps is None:
        eps = uset.gap_bound(uset.vertices(scale, n_eta, n_x), scale=scale,
                             scaled=True, n_eta=5, n_xi=3, n_x=17, n_a=3,
                             n_z=2)
    if delay_independent:
        return L.certificate_di(V, eps=eps, n_plant=npl, **kw), eps
    return L.certificate(V, tau, eps=eps, n_plant=npl, **kw), eps


def certified_depth(uset_of, ss=None, pd=None, lo=1e-6, hi=3e-3, tol=2e-6,
                    eps=None, n_iter=22, **kw):
    """Largest axial depth for which the delay-independent certificate holds.

    uset_of(ap) -> UncertaintySet at that depth.  The answer is a depth below
    which NO spindle speed and no parameter value in the set can destabilise the
    loop -- the stability-lobe minimum, turned into a theorem.
    """
    def ok(ap):
        u = uset_of(ap)
        r, _ = certify(u, ss, pd, eps=eps, **kw)
        return r['feasible']

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


def uncertainty_margin(uset, ss=None, pd=None, lo=0.0, hi=4.0, tol=2e-2,
                       n_iter=16, **kw):
    """Largest inflation of the physics set still certified (Phase 10)."""
    def ok(sc):
        r, _ = certify(uset, ss, pd, scale=sc, **kw)
        return r['feasible']

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
# Exact delay-independent test (frequency domain)
# ---------------------------------------------------------------------------
def _wgrid(A, n=400):
    r = np.abs(np.linalg.eigvals(A))
    hi = 5.0 * max(float(r.max()), 1.0)
    return np.concatenate([[0.0], np.logspace(-3, np.log10(hi), n)])


def di_stable_exact(Acl, Adcl, w=None, tol=1.0):
    """Exact delay-independent stability of  xdot = A x + A_d x(t-tau).

    The characteristic function det(sI - A - A_d e^{-s tau}) has no root in the
    closed right half plane for EVERY tau >= 0 if and only if A is Hurwitz and

        sup_omega  rho( (j omega I - A)^{-1} A_d )  <  1,

    because e^{-j omega tau} sweeps the whole unit circle as tau varies.  This
    is exact -- no Lyapunov conservatism -- and costs one eigenvalue problem per
    frequency, so it is what the depth and margin bisections use.  The quadratic
    LK certificate is kept alongside because it buys something this test does
    not: it also covers ARBITRARILY FAST variation of the parameters.
    """
    if not (np.all(np.isfinite(Acl)) and np.all(np.isfinite(Adcl))):
        return False, np.inf
    try:
        ev = np.linalg.eigvals(Acl)
    except np.linalg.LinAlgError:
        return False, np.inf
    if float(np.max(ev.real)) >= 0.0:
        return False, np.inf
    w = _wgrid(Acl) if w is None else w
    n = Acl.shape[0]
    I = np.eye(n)
    peak = 0.0
    for wk in w:
        try:
            R = np.linalg.solve(1j * wk * I - Acl, Adcl)
        except np.linalg.LinAlgError:
            return False, np.inf
        peak = max(peak, float(np.max(np.abs(np.linalg.eigvals(R)))))
        if peak >= tol:
            return False, peak
    return peak < tol, peak


def di_stable_set(uset, ss=None, pd=None, scale=1.0, n_eta=3, n_x=9,
                  eta_range=None):
    """Exact delay-independent stability at every vertex of the set."""
    V, npl = closed_loop_vertices(uset, ss, pd, scale, n_eta, n_x,
                                  eta_range=eta_range)
    peak = 0.0
    for (A, Ad) in V:
        ok, p = di_stable_exact(A, Ad)
        peak = max(peak, p)
        if not ok:
            return False, peak
    return True, peak


def di_depth(uset_of, ss=None, pd=None, lo=2e-6, hi=4e-3, tol=2e-6,
             n_iter=24, **kw):
    """Largest depth for which every vertex is delay-independently stable."""
    def ok(ap):
        return di_stable_set(uset_of(ap), ss, pd, **kw)[0]

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


def di_margin(uset, ss=None, pd=None, lo=0.0, hi=8.0, tol=1e-2, n_iter=20,
              **kw):
    """Largest inflation of the physics set that stays delay-independently
    stable at the nominal depth (Phase 10)."""
    def ok(sc):
        return di_stable_set(uset, ss, pd, scale=sc, **kw)[0]

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
