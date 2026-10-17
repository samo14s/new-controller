"""First-order semi-discretization (Insperger & Stepan; Insperger, Stepan &
Turi 2008) for the closed-loop milling DDE

    ydot(t) = [A0 + (a H(t)/m) e2 e1^T] y(t) - (a H(t)/m) e2 w(t - T)

y = [w, wdot, q, qdot] (full actuator) or [w, wdot] (ideal actuator),
H(t) T-periodic (tooth period), delay T.  On [t_i, t_i + dt], dt = T/m, the
coefficients are frozen at their interval means and the delayed position is
interpolated linearly between w_{i-m} and w_{i-m+1}:

    y_{i+1} = P_i y_i + (Phi0_i - Phi1_i) b_i w_{i-m} + Phi1_i b_i w_{i-m+1}
    P_i = e^{A_i dt},  Phi0_i = int_0^dt e^{A_i(dt-s)} ds,
    Phi1_i = (1/dt) int_0^dt e^{A_i(dt-s)} s ds,  b_i = -(a H_i / m) e2.

Phi0, Phi1 come exactly from one block-triangular matrix exponential.
Stability: spectral radius of the monodromy matrix over one period < 1.
"""
import numpy as np
from scipy.linalg import expm
import model as M

GL_X, GL_W = np.polynomial.legendre.leggauss(8)


def interval_aH(rpm, a, m, helix=False):
    """Interval means of a*H(t) over the m sub-intervals of one tooth period.

    helix=False : straight flute (exact interval means)
    helix=True  : 45 deg helical cutter (exact in z, 8-point Gauss in t)
    helix='mean': H(t) replaced by its time average (N Kt / 4 pi) alpha_yy;
                  the SDM then solves exactly the averaged (ZOA) model, which
                  isolates SDM discretisation error from parametric effects.
    """
    T = 60.0 / (M.NT * rpm)
    Om = 2 * np.pi * rpm / 60.0
    edges = np.linspace(0.0, T, m + 1)
    if isinstance(helix, str) and helix == "mean":
        return np.full(m, a * M.NT * M.KT * M.ALPHA_YY / (4 * np.pi))
    if not helix:
        # straight flute: tooth 0 enters (phi = pi/2) at t = 0 and exits at T;
        # exactly one tooth engaged, H continuous inside the period
        ph = M.PHI_ST + Om * edges
        return a * M.H_avg_straight(ph[:-1], ph[1:])
    # helical: Gauss-Legendre in time inside each interval, exact in z
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1] - edges[0])
    t = mid[:, None] + half * GL_X[None, :]
    vals = M.aH_helical((M.PHI_ST + Om * t).ravel(), a).reshape(t.shape)
    return 0.5 * (vals * GL_W[None, :]).sum(axis=1)


def default_m(rpm, per_period=40):
    """Intervals per tooth period: >= 40 and >= per_period steps per natural
    period of the structural mode (the SDM error is O(1/m^2), see
    sdm_convergence.csv)."""
    T = 60.0 / (M.NT * rpm)
    return int(max(40, np.ceil(per_period * T * M.FN)))


def monodromy(rpm, a, g, zeta, m=None, helix=False, ideal=False):
    m = default_m(rpm) if m is None else m
    T = 60.0 / (M.NT * rpm)
    dt = T / m
    A0 = M.A_closed(g, zeta, ideal)
    d = A0.shape[0]
    aH = interval_aH(rpm, a, m, helix)
    Ai = np.repeat(A0[None], m, axis=0)
    Ai[:, 1, 0] += aH / M.MS
    big = np.zeros((m, 3 * d, 3 * d))
    big[:, :d, :d] = Ai * dt
    big[:, :d, d:2 * d] = np.eye(d) * dt
    big[:, d:2 * d, 2 * d:] = np.eye(d) * dt
    E = expm(big)
    P = E[:, :d, :d]
    Phi0 = E[:, :d, d:2 * d]
    Phi1 = E[:, :d, 2 * d:] / dt
    b = np.zeros((m, d))
    b[:, 1] = -aH / M.MS
    w0 = np.einsum("ijk,ik->ij", Phi0 - Phi1, b)   # multiplies w_{i-m}
    w1 = np.einsum("ijk,ik->ij", Phi1, b)          # multiplies w_{i-m+1}
    # state z_i = [y_i, w_{i-1}, ..., w_{i-m}]
    nz = d + m
    Phi = np.eye(nz)
    for i in range(m):
        new = np.empty_like(Phi)
        new[:d] = P[i] @ Phi[:d] + np.outer(w0[i], Phi[nz - 1]) + np.outer(w1[i], Phi[nz - 2])
        new[d] = Phi[0]
        new[d + 1:] = Phi[d:nz - 1]
        Phi = new
    return Phi


def mu_max(rpm, a, g, zeta, m=None, helix=False, ideal=False, return_mu=False):
    ev = np.linalg.eigvals(monodromy(rpm, a, g, zeta, m, helix, ideal))
    j = np.argmax(np.abs(ev))
    return (np.abs(ev[j]), ev[j]) if return_mu else np.abs(ev[j])


def boundary(rpm, g, zeta, m=None, helix=False, ideal=False,
             a0=0.05e-3, amax=0.4, ratio=1.06, rtol=2e-5):
    """Smallest a (m) with spectral radius 1: geometric scan then bisection.

    Independent of the ZOA (no ZOA value is used as a start).  Returns
    (a_lim, critical multiplier at a_lim*(1+rtol), number of evaluations).
    """
    nev = 0
    if mu_max(rpm, 0.0, g, zeta, m, helix, ideal) >= 1:
        return 0.0, np.nan, 1
    lo, hi = 0.0, None
    a = a0
    while a <= amax:
        nev += 1
        if mu_max(rpm, a, g, zeta, m, helix, ideal) >= 1:
            hi = a
            break
        lo = a
        a *= ratio
    if hi is None:
        return np.inf, np.nan, nev
    while (hi - lo) > rtol * hi:
        mid = 0.5 * (lo + hi)
        nev += 1
        if mu_max(rpm, mid, g, zeta, m, helix, ideal) >= 1:
            hi = mid
        else:
            lo = mid
    _, mu = mu_max(rpm, hi, g, zeta, m, helix, ideal, return_mu=True)
    return 0.5 * (lo + hi), mu, nev + 1


def multiplier_freq(mu, rpm, fref):
    """Chatter frequency implied by a critical multiplier mu = e^{i w T}:
    w = (+-arg mu + 2 pi k)/T (mu and its conjugate are both multipliers),
    choosing the sign and k that land nearest fref (Hz)."""
    T = 60.0 / (M.NT * rpm)
    best = None
    for th in (np.angle(mu), -np.angle(mu)):
        k = np.round((2 * np.pi * fref * T - th) / (2 * np.pi))
        f = (th + 2 * np.pi * k) / (2 * np.pi * T)
        if best is None or abs(f - fref) < abs(best - fref):
            best = f
    return best


def boundary_richardson(rpm, g, zeta, helix=False, ideal=False, m=None, rtol=2e-6):
    """Boundary at m and 2m intervals plus Richardson extrapolation
    a_R = a_2m + (a_2m - a_m)/3 (second-order convergence).

    The 2m search is bracketed around a_m (bracket verified; full scan if the
    bracket fails).  Returns dict with a_m, a_2m, a_R, critical multiplier.
    """
    m = default_m(rpm) if m is None else m
    a1, mu1, n1 = boundary(rpm, g, zeta, m, helix, ideal, rtol=rtol)
    lo, hi = 0.95 * a1, 1.05 * a1
    if (mu_max(rpm, lo, g, zeta, 2 * m, helix, ideal) < 1
            and mu_max(rpm, hi, g, zeta, 2 * m, helix, ideal) >= 1):
        while (hi - lo) > rtol * hi:
            mid = 0.5 * (lo + hi)
            if mu_max(rpm, mid, g, zeta, 2 * m, helix, ideal) >= 1:
                hi = mid
            else:
                lo = mid
        a2 = 0.5 * (lo + hi)
        _, mu2 = mu_max(rpm, hi, g, zeta, 2 * m, helix, ideal, return_mu=True)
        bracket = True
    else:
        a2, mu2, _ = boundary(rpm, g, zeta, 2 * m, helix, ideal, rtol=rtol)
        bracket = False
    return dict(m=m, a_m=a1, a_2m=a2, a_R=a2 + (a2 - a1) / 3.0, mu=mu2,
                bracketed=bracket)
