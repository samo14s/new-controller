"""Linear stability of the continuous-time periodic model by semi-discretisation.

Independent reference for the time-domain stability boundary (no time-stepping
of forces, no ZOH).  Linearising the chip-thickness model about the steady,
tooth-periodic cut (static chip fz sin(phi) > 0 everywhere inside [pi/2, pi],
so the h <= 0 switch is inactive) gives the periodic delay equation

    z' = A_cl z + b H(t) (x(t) - x(t - T)),   T = 60/(N rpm),   x = c^T z,

with z = [x, xdot, q, qdot] (structure + actuator + DVF), b = [0, 1/m, 0, 0] and
the specific directional factor of the y direction, integrated over the helix:

    H(t) = sum_j int_0^a g(phi_j(z,t)) (Kt sin phi cos phi - Kr cos^2 phi) dz.

Its tooth-period average is N a Kt alpha_yy / (4 pi), so replacing H by its mean
gives exactly the zero-order (ZOA) characteristic equation -- this is used to
validate both this code and src/zoa.py.

Method: first-order semi-discretisation (Insperger & Stepan 2004), m intervals
per period, H averaged over each interval, delayed term linearly interpolated,
current term integrated exactly; monodromy eigenvalues decide stability.
"""
import numpy as np
from scipy.linalg import expm
import ref_sim as R


def closed_loop_A(g):
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 0] = -R.K / R.M
    A[1, 1] = -(R.C + R.KA * g) / R.M
    A[1, 2] = -R.KA * R.A0 / R.M
    A[1, 3] = -R.KA * R.A1 / R.M
    A[2, 3] = 1.0
    A[3, 1] = -g
    A[3, 2] = -R.A0
    A[3, 3] = -R.A1
    return A


def _F(phi):
    """Antiderivative of Kt sin cos - Kr cos^2."""
    return R.KT * np.sin(phi) ** 2 / 2 - R.KR * (phi / 2 + np.sin(2 * phi) / 4)


def H_of_theta(theta, a, helix=R.BETA, phi_out=R.PHI_OUT):
    """Specific force factor (N/m) at spindle angle theta (array), exact in z."""
    theta = np.atleast_1d(theta)
    out = np.zeros_like(theta, dtype=float)
    kb = 2 * np.tan(helix) / R.D
    for j in range(R.NT):
        top = theta + 2 * np.pi * j / R.NT                 # slice z = 0
        if kb == 0.0:
            ph = np.mod(top, 2 * np.pi)
            ins = (ph >= R.PHI_IN) & (ph <= phi_out)
            out += np.where(ins, a * (R.KT * np.sin(ph) * np.cos(ph)
                                      - R.KR * np.cos(ph) ** 2), 0.0)
            continue
        bot = top - kb * a                                 # slice z = a
        # shift so that top lies in [0, 2 pi)
        sh = np.floor(top / (2 * np.pi)) * 2 * np.pi
        t0, b0 = top - sh, bot - sh
        for n in range(-4, 2):
            lo = np.maximum(b0, R.PHI_IN + 2 * np.pi * n)
            hi = np.minimum(t0, phi_out + 2 * np.pi * n)
            ok = hi > lo
            out += np.where(ok, (_F(hi - 2 * np.pi * n) - _F(lo - 2 * np.pi * n)) / kb, 0.0)
    return out


def H_mean_zoa(a):
    import zoa
    return R.NT * a * R.KT * zoa.ALPHA / (4 * np.pi)


def H_mean_numeric(a, phi_out=R.PHI_OUT, n=20000):
    th = 2 * np.pi * (np.arange(n) + 0.5) / n
    return float(np.mean(H_of_theta(th, a, phi_out=phi_out)))


def monodromy_radius(rpm, a, g, m=120, helix=R.BETA, mode="periodic", nsub=16,
                     return_mults=False, phi_out=R.PHI_OUT):
    T = 60.0 / (R.NT * rpm)
    dt = T / m
    Acl = closed_loop_A(g)
    b = np.array([0.0, 1.0 / R.M, 0.0, 0.0])
    # interval-averaged H
    if mode == "mean":
        Hi = np.full(m, H_mean_zoa(a) if phi_out == R.PHI_OUT
                     else H_mean_numeric(a, phi_out))
    else:
        Omega = 2 * np.pi * rpm / 60.0
        tt = (np.arange(m)[:, None] + (np.arange(nsub)[None, :] + 0.5) / nsub) * dt
        Hi = H_of_theta((Omega * tt).ravel(), a, helix, phi_out).reshape(m, nsub).mean(axis=1)
    n = 4 + m
    M = np.eye(n)
    I4 = np.eye(4)
    Z4 = np.zeros((4, 4))
    for i in range(m):
        Ai = Acl.copy()
        Ai[1, 0] += Hi[i] / R.M
        big = np.block([[Ai, I4, Z4], [Z4, Z4, I4], [Z4, Z4, Z4]]) * dt
        E = expm(big)
        Phi = E[:4, :4]
        G0 = E[:4, 4:8]                 # int_0^dt e^{A u} du
        G1 = E[:4, 8:12] / dt           # (1/dt) int_0^dt e^{A u} (dt - u) du
        Bx = -b * Hi[i]                  # delayed term coefficient on x(t - T)
        w0 = (G0 - G1) @ Bx             # multiplies x_{i-m}
        w1 = G1 @ Bx                    # multiplies x_{i-m+1}
        top = Phi @ M[:4] + np.outer(w0, M[n - 1]) + np.outer(w1, M[n - 2])
        newM = np.empty_like(M)
        newM[:4] = top
        newM[4] = M[0]
        newM[5:] = M[4:n - 1]
        M = newM
    mu = np.linalg.eigvals(M)
    r = np.max(np.abs(mu))
    if return_mults:
        return r, mu
    return r


def critical_multiplier(rpm, a, g, **kw):
    r, mu = monodromy_radius(rpm, a, g, return_mults=True, **kw)
    k = np.argmax(np.abs(mu))
    return mu[k]


def boundary(rpm, g, a_guess, rtol=1e-5, check_below=True, **kw):
    """Depth where the spectral radius crosses 1, bracketed near a_guess.

    With check_below, the radius is also evaluated at 0.25/0.5/0.75 of the
    result; any instability there (a lower crossing) is reported via 'island'.
    """
    from scipy.optimize import brentq
    f = lambda a: monodromy_radius(rpm, a, g, **kw) - 1.0
    a = 0.85 * a_guess
    fa = f(a)
    step = 1.04
    if fa < 0:
        while True:
            a2 = a * step
            f2 = f(a2)
            if f2 > 0:
                lo, hi = a, a2
                break
            a = a2
            if a > 0.2:
                return np.inf, False
    else:
        while True:
            a2 = a / step
            f2 = f(a2)
            if f2 < 0:
                lo, hi = a2, a
                break
            a = a2
            if a < 1e-6:
                return 0.0, False
    ab = brentq(f, lo, hi, xtol=1e-12, rtol=rtol)
    island = False
    if check_below:
        island = any(f(c * ab) > 0 for c in (0.25, 0.5, 0.75))
    return ab, island
