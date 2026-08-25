"""
lk_lmi.py — PHASES 9, 10, 11: the delay-explicit stability certificate.
=======================================================================
Du et al. (2024) never prove closed-loop stability WITH the delay.  Eq. (16) puts
the delayed state inside the input F_PI, so the mu condition of Eq. (28) is a
condition on a delay-free system: the paper contains no tau_max and no inequality
of the form Omega(P, Q, K, tau, Delta) < 0.  This module supplies one.

Closed loop, after eliminating the controller:

    xdot_cl(t) = A_cl(theta) x_cl(t) + A_d,cl(theta) x_cl(t - tau)

with theta in the physics-based set of `uncertainty.py`, over-approximated by
conv{(A_k, A_dk)} plus a norm-bounded residual of radius eps.

Lyapunov-Krasovskii functional

    V = x^T P x + int_{t-tau}^{t} x^T Q x ds
              + tau int_{-tau}^{0} int_{t+s}^{t} xdot^T R xdot dr ds

Differentiating and bounding the double integral with Jensen's inequality gives,
for every vertex k, the condition

    Psi_k = [ A^T P + P A + Q - R    P A_d + R      tau A^T R  ]
            [ A_d^T P + R            -Q - R         tau A_d^T R ]  < 0
            [ tau R A                tau R A_d      -R          ]

which, with the norm-bounded residual handled by the S-procedure and one Schur
complement, becomes the LMI actually solved here:

    [ Psi_k + lam F^T F     E ]
    [ E^T                -lam I ]  < 0 ,   E = [P E0 ; 0 ; tau R E0].

Feasibility certifies asymptotic stability for EVERY delay in [0, tau], every
parameter value in the set, and arbitrarily fast variation of that value -- the
last point being what a common P buys and what a frozen mu analysis does not give.
"""
import numpy as np

try:
    import cvxpy as cp
except Exception:                                            # pragma: no cover
    cp = None


def _psi(A, Ad, P, Q, R, tau):
    return cp.bmat([
        [A.T @ P + P @ A + Q - R, P @ Ad + R,       tau * A.T @ R],
        [Ad.T @ P + R,            -Q - R,           tau * Ad.T @ R],
        [tau * R @ A,             tau * R @ Ad,     -R],
    ])


def certificate(vertices, tau, eps=0.0, n_plant=None, solver=None,
                verbose=False, kappa=1e7):
    """Solve the LK LMI.  Returns dict(feasible, slack, status, P, Q, R, lam).

    vertices : list of (A_cl, A_d_cl) pairs, in scaled coordinates
    tau      : delay bound, same time units as the vertices
    eps      : radius of the norm-bounded residual acting on the plant states
    n_plant  : how many leading states the residual acts on (default: all)
    """
    if cp is None:
        raise RuntimeError('cvxpy is required for the certificate')
    n = vertices[0][0].shape[0]
    npl = n if n_plant is None else int(n_plant)
    P = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n), symmetric=True)
    R = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(nonneg=True)

    Sp = np.zeros((npl, n))
    Sp[:, :npl] = np.eye(npl)
    E0 = eps * np.vstack([np.eye(npl), np.zeros((n - npl, npl))])
    SS = Sp.T @ Sp
    Zn = np.zeros((n, n))
    FtF = np.block([[SS, Zn, Zn], [Zn, SS, Zn], [Zn, Zn, Zn]])

    # Normalisation.  The condition is homogeneous in (P, Q, R, lam).  P is
    # pinned from below and the others are left free up to a large trace budget:
    # the Jensen term needs R >> P to recover the tau -> 0 limit, so R must not be
    # squeezed by the normalisation.  Feasibility is then s > 0.
    cons = [P >> np.eye(n), Q >> 1e-6 * np.eye(n), R >> 1e-6 * np.eye(n),
            cp.trace(P) + cp.trace(Q) + cp.trace(R) + lam <= kappa]
    for (A, Ad) in vertices:
        Psi = _psi(A, Ad, P, Q, R, tau)
        if eps > 0.0:
            Psi = Psi + lam * FtF
            Ecal = cp.vstack([P @ E0, np.zeros((n, npl)), tau * R @ E0])
            Th = cp.bmat([[Psi, Ecal], [Ecal.T, -lam * np.eye(npl)]])
        else:
            Th = Psi
        cons.append(Th << -s * np.eye(Th.shape[0]))

    prob = cp.Problem(cp.Maximize(s), cons)
    for sv in ([solver] if solver else ['CLARABEL', 'SCS']):
        try:
            prob.solve(solver=sv, verbose=verbose)
        except Exception:
            continue
        if prob.status in ('optimal', 'optimal_inaccurate'):
            break
    ok = (prob.status in ('optimal', 'optimal_inaccurate')
          and s.value is not None and float(s.value) > 1e-10)
    return dict(feasible=bool(ok),
                slack=None if s.value is None else float(s.value),
                status=prob.status,
                P=None if P.value is None else np.array(P.value),
                Q=None if Q.value is None else np.array(Q.value),
                R=None if R.value is None else np.array(R.value),
                lam=None if lam.value is None else float(lam.value))


def tau_max(vertices, eps=0.0, n_plant=None, lo=0.0, hi=100.0, tol=1e-2,
            n_iter=20, **kw):
    """Largest delay bound for which the certificate holds (bisection)."""
    if not certificate(vertices, max(lo, 1e-6), eps, n_plant, **kw)['feasible']:
        return 0.0
    if certificate(vertices, hi, eps, n_plant, **kw)['feasible']:
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if certificate(vertices, mid, eps, n_plant, **kw)['feasible']:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def margin(build_vertices, tau, eps_of=None, n_plant=None, lo=0.0, hi=3.0,
           tol=2e-2, n_iter=16, **kw):
    """Largest inflation `scale` of the uncertainty set still certified."""
    def ok(sc):
        V = build_vertices(sc)
        e = 0.0 if eps_of is None else eps_of(sc)
        return certificate(V, tau, e, n_plant, **kw)['feasible']

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


def certificate_di(vertices, eps=0.0, n_plant=None, solver=None,
                   verbose=False, kappa=1.0, tol=1e-9):
    """Delay-INDEPENDENT certificate: stability for EVERY tau >= 0.

        V = x^T P x + int_{t-tau}^t x^T Q x ds
        [ A^T P + P A + Q   P A_d ]
        [ A_d^T P           -Q    ] < 0

    For milling this is the natural statement: it certifies a cutting depth at
    which no spindle speed whatsoever -- no lobe -- can destabilise the loop,
    over the whole uncertainty set.  It is what the stability-lobe minimum is,
    made into a theorem.

    A SOLVER STATUS IS NOT A CERTIFICATE.  cvxpy returns `optimal_inaccurate`
    for points that violate the constraints by many orders of magnitude on this
    problem -- P with eigenvalues at -3e5 where P >> 1e-3 I was asked, a trace
    of 1.4e5 where the normalisation fixes it at 1, and Psi with a +1.9e7
    eigenvalue where Psi << -s I was asked.  Reporting that as feasible is a
    false positive, so the returned (P, Q) is VERIFIED here against the three
    inequalities it is supposed to satisfy, and `feasible` means the verified
    matrices really are a Lyapunov-Krasovskii pair.  `certified` carries that
    verdict, `status` keeps the solver's own word, and the residuals are
    returned so a marginal case can be judged rather than trusted.
    """
    if cp is None:
        raise RuntimeError('cvxpy is required for the certificate')
    n = vertices[0][0].shape[0]
    npl = n if n_plant is None else int(n_plant)
    P = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(nonneg=True)
    Sp = np.zeros((npl, n))
    Sp[:, :npl] = np.eye(npl)
    E0 = eps * np.vstack([np.eye(npl), np.zeros((n - npl, npl))])
    SS = Sp.T @ Sp
    Zn = np.zeros((n, n))
    FtF = np.block([[SS, Zn], [Zn, SS]])
    # Homogeneous in (P, Q, lam): fix the trace to 1 so every variable stays
    # O(1) and s is a relative margin.  Feasibility is s > 0.
    cons = [P >> 1e-3 * np.eye(n), Q >> 1e-8 * np.eye(n),
            cp.trace(P) + cp.trace(Q) + lam == 1.0]
    for (A, Ad) in vertices:
        Psi = cp.bmat([[A.T @ P + P @ A + Q, P @ Ad],
                       [Ad.T @ P, -Q]])
        if eps > 0.0:
            Psi = Psi + lam * FtF
            Ecal = cp.vstack([P @ E0, np.zeros((n, npl))])
            Psi = cp.bmat([[Psi, Ecal], [Ecal.T, -lam * np.eye(npl)]])
        cons.append(Psi << -s * np.eye(Psi.shape[0]))
    prob = cp.Problem(cp.Maximize(s), cons)
    for sv in ([solver] if solver else ['CLARABEL', 'SCS']):
        try:
            prob.solve(solver=sv, verbose=verbose)
        except Exception:
            continue
        if prob.status in ('optimal', 'optimal_inaccurate'):
            break
    solved = (prob.status in ('optimal', 'optimal_inaccurate')
              and s.value is not None and float(s.value) > tol)
    Pv = None if P.value is None else np.array(P.value)
    Qv = None if Q.value is None else np.array(Q.value)
    res = verify_di(vertices, Pv, Qv)
    return dict(feasible=bool(solved and res['certified']),
                certified=res['certified'],
                solver_feasible=bool(solved),
                slack=None if s.value is None else float(s.value),
                status=prob.status, residual=res,
                P=Pv, Q=Qv)


def verify_di(vertices, P, Q, rtol=1e-8):
    """Do these (P, Q) really satisfy the delay-independent LK inequalities?

    Returns the three residuals, each normalised by the size of the matrix it
    came from so the verdict does not depend on how the solver scaled its
    answer: P and Q must be positive definite, and every vertex Psi must be
    negative definite.  A solution that fails any of them is not a certificate,
    whatever the solver called it.
    """
    if P is None or Q is None:
        return dict(certified=False, reason='solver returned no matrices',
                    min_eig_P=np.nan, min_eig_Q=np.nan, max_eig_Psi=np.nan)
    Ps = 0.5 * (P + P.T)
    Qs = 0.5 * (Q + Q.T)
    mp = float(np.linalg.eigvalsh(Ps).min())
    mq = float(np.linalg.eigvalsh(Qs).min())
    worst = -np.inf
    for (A, Ad) in vertices:
        Psi = np.block([[A.T @ Ps + Ps @ A + Qs, Ps @ Ad],
                        [Ad.T @ Ps, -Qs]])
        Psi = 0.5 * (Psi + Psi.T)
        sc = max(float(np.abs(Psi).max()), 1e-300)
        worst = max(worst, float(np.linalg.eigvalsh(Psi).max()) / sc)
    nP = max(float(np.abs(Ps).max()), 1e-300)
    nQ = max(float(np.abs(Qs).max()), 1e-300)
    ok = (mp / nP > rtol) and (mq / nQ > rtol) and (worst < -rtol)
    return dict(certified=bool(ok), min_eig_P=mp, min_eig_Q=mq,
                rel_min_eig_P=mp / nP, rel_min_eig_Q=mq / nQ,
                max_eig_Psi=worst,
                reason='' if ok else 'the returned matrices violate the LMIs')
