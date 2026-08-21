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
                         balance=True):
    """[(A_cl, A_d,cl)] in scaled coordinates, plus n_plant."""
    n = uset.n
    ws = uset.ws
    out = []
    npl_s = None
    Tc = None
    for (A, Ad, B, Cy) in uset.vertices(scale, n_eta, n_x):
        Acl, Adcl, npl = _augment(A, Ad, B, Cy, ss, pd, n)
        S = np.diag(np.concatenate([ws * np.ones(n), np.ones(n),
                                    np.ones(Acl.shape[0] - npl)]))
        Si = np.linalg.inv(S)
        Acl = S @ Acl @ Si / ws
        Adcl = S @ Adcl @ Si / ws
        if balance and Acl.shape[0] > npl:
            if Tc is None:
                _, T = matrix_balance(Acl[npl:, npl:])
                d = np.concatenate([np.ones(npl), np.diag(T)])
                Tc = np.diag(d)
                Tci = np.diag(1.0 / d)
            Acl = Tci @ Acl @ Tc
            Adcl = Tci @ Adcl @ Tc
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
