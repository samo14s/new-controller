"""C3: certified basin on the time-periodic model (periodic saturated-LMI certificate).

What C2 left open.  C2's theorem is about the averaged (ZOA) model: the
directional coefficient of the regenerative force is replaced by its mean
over the tooth period.  The real cut has a time-periodic coefficient
a H(t), and the forced command b(t) varies over the period while C2 used its
peak.  C3 proves the basin on the time-periodic collocated model, with a
periodic Lyapunov function and a time-resolved saturation margin.

Model.  The deviation from the forced orbit obeys
    x' = A(t) x - B psi(K x, t),   A(t) = A0 + a H(t) E,
A0 the collocated loop of C2 without cutting, E = (e_2 e_1' - e_2 e_N') / m
(regeneration x - x(t - T) on the velocity row), a H(t) the exact helical
integral over the depth of Kt sin(phi) cos(phi) - Kr cos(phi)^2 on the engaged
teeth.  Its mean is C2's KAPPA a.  The forced orbit is the same as in C2: the
regenerative term vanishes on any tooth-periodic motion.  The collocation of
the delay is the same as C2's (the history moves with the clock whatever A(t)).
Its Floquet multipliers match the semi-discretisation of verification/zoa_sdm
(floquet_check).

Certificate.  P(t) = W(t)^-1 with W continuous, periodic and affine on each
of M_C subintervals of the tooth period (node values W_k, W_M_C = W_0);
Y(t) = G(t) W(t) affine between node values Y_k; S_k > 0 constant on each.
V = x' P(t) x satisfies V' + 2 lam V < 0 with the generalised sector condition
at level u0(t) (pointwise in t) when, with z = P x,
    M1(t) = [[W' - A(t) W - W A(t)' - 2 lam W, B S - Y'], [S B' - Y, 2 S]] > 0
(x' P' x = -z' W' z gives the sign of W').  On subinterval k, W' is constant;
W(t), Y(t) are affine in the local time s in [0, 1], and A(t) = A0 + h(t) E
with h(t) in a proven enclosure [h_k-, h_k+].  M1 is affine in s for fixed h
and affine in h for fixed s, so on the box it is the bilinear interpolation of
its four corner values: a convex combination.  M1 > 0 at the two endpoints
times the two enclosure vertices therefore gives M1 > 0 on the whole
subinterval.  The sector condition: the deadzone of sat(b(t) + w) - b(t)
satisfies psi (psi - v) <= 0 whenever |w - v| <= V - |b(t)|; on subinterval k
the level u0_k = V - max |b| over it is used, and the inclusion of E(P(t))
in the sector region,
    M2 = [[W, W K' - Y'], [K W - Y, u0_k^2]] >= 0,
is affine in s, so it holds on the subinterval when it holds at both nodes.
The validity bound of C2, |x - x(t - T)| <= DELTA fz, and the size of
E(P(t)) along e, [[1, beta e'], [beta e, W]] >= 0, are affine too: nodes
suffice.  Then E(P(t0)) is a certified basin for a deviation applied at any
phase t0: the trajectory stays in E(P(t)) and V decays as e^(-2 lam t).
The certified kick is the smallest size over the phases.

Enclosure.  h(t) is sampled densely on each subinterval and widened by
L dt / 2, L a bound on |dh/dt| (the window of each tooth enters through the
two ends of the z-integral only).  b(t) is bounded as in C2: samples plus
max|b''| dt^2 / 8.

Numerics.  As C2: re-centred solves (constant coordinate change by the
Cholesky factor of the mean W), and every candidate is checked independently
of the solver at all 4 M_C corners (eigenvalues), the inclusion and validity
at every node (Cholesky; exceeded bounds repaired by scaling (W, Y, S)), and
the size at every node.

    python c3_periodic.py floquet     -> ../results/c3_floquet.npz
    python c3_periodic.py basin       -> ../results/c3_basin.npz
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
from scipy.linalg import expm, cholesky, solve_triangular
import params as P
import c2_certificate as c2
from c2_certificate import LMIBlock, solve_sdp, VMAX, TIGHT, DELTA
from zoa import a_lim, alpha_matrix, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
KAPZ = np.tan(P.HELIX) / (0.5 * P.D_TOOL)          # helix lag, rad per metre of depth
M_C = 16                                           # subintervals of the tooth period
N_ENC = 2048                                       # samples per subinterval for the enclosures (the pad
#                                                  L dt / 2 dominates the residual below about 1000)
N_PHASE = 16                                       # C1's kick phases per tooth period


# -------------------------------------------------------------- the cut ----
class Cut:
    """Radial immersion: entry/exit angles, mean directional coefficient
    (C2's KAPPA at this immersion) and the static slice-force harmonics."""

    def __init__(self, ae=P.RADIAL_IMMERSION):
        self.ae = ae
        self.st, self.ex = P.immersion_angles(P.MILLING, ae)
        self.alpha = alpha_matrix(P.MILLING, ae)[1, 1]
        self.kappa = P.N_TEETH * P.K_TC * self.alpha / (4 * np.pi)
        self.W = self._slice_force_harmonics()

    def _slice_force_harmonics(self, n_theta=4096, n_harm=40):
        """feasibility.slice_force_harmonics at this immersion."""
        pitch = 2 * np.pi / P.N_TEETH
        th = np.arange(n_theta) / n_theta * pitch
        w = np.zeros(n_theta)
        for i in range(P.N_TEETH):
            phi = np.mod(th + i * pitch, 2 * np.pi)
            cut = (phi >= self.st) & (phi <= self.ex)
            h = P.FZ * np.sin(phi)
            w += np.where(cut & (h > 0), -(P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * h, 0.0)
        return (np.fft.rfft(w) / n_theta)[: n_harm + 1]

    def a_lim(self, rpm, g):
        """ZOA limit: the characteristic equation scales with alpha only."""
        return a_lim(rpm, g) * ALPHA / self.alpha

    def force_harmonics(self, a):
        n = len(self.W)
        m = np.arange(n)
        q = m * P.N_TEETH * KAPZ
        with np.errstate(invalid="ignore", divide="ignore"):
            integ = np.where(m == 0, a, (1 - np.exp(-1j * q * a)) / (1j * q))
        return self.W * integ


HALF = Cut()


def _prim(p):
    """Antiderivative of Kt sin(p) cos(p) - Kr cos(p)^2."""
    return P.K_TC * np.sin(p) ** 2 / 2 - P.K_RC * (p / 2 + np.sin(2 * p) / 4)


def aH(t, rpm, a, cut=HALF):
    """a H(t) (N/m): dynamic force per unit x - x(t - T), exact in z.
    Tooth i of the slice at height z sits at Omega t + i 2 pi / N - KAPZ z
    (the simulator's convention)."""
    th = 2 * np.pi * rpm / 60.0 * np.atleast_1d(np.asarray(t, float))
    out = np.zeros_like(th)
    for j in range(P.N_TEETH):
        hi = th + j * 2 * np.pi / P.N_TEETH
        lo = hi - KAPZ * a
        for p in np.arange(np.floor((lo.min() - cut.ex) / (2 * np.pi)) - 1,
                           np.ceil((hi.max() - cut.st) / (2 * np.pi)) + 2):
            l = np.maximum(lo, cut.st + 2 * np.pi * p)
            u = np.minimum(hi, cut.ex + 2 * np.pi * p)
            m = u > l
            out[m] += (_prim(u[m] - 2 * np.pi * p) - _prim(l[m] - 2 * np.pi * p)) / KAPZ
    return out


def aH_rate_bound(rpm, cut=HALF):
    """L >= |d(a H)/dt|: each tooth contributes (f(phi_top) - f(phi_bottom)) / KAPZ
    times the spindle rate, f bounded on the window."""
    ph = np.linspace(cut.st, cut.ex, 4001)
    f = np.abs(P.K_TC * np.sin(ph) * np.cos(ph) - P.K_RC * np.cos(ph) ** 2).max() * 1.001
    return 2 * np.pi * rpm / 60.0 * P.N_TEETH * 2 * f / KAPZ


def enclosure(rpm, a, cut=HALF, mc=M_C, n_enc=N_ENC, order=2):
    """Per subinterval k (local time s in [0, 1]): a H = alpha_k + beta_k s + r
    with |r| <= eps_k.  order 1: a box (beta = 0); order 2: least-squares
    line plus the largest residual.  Samples at the midpoints of n_enc cells,
    widened by the rate bound of the residual times half a cell."""
    Tp = 60.0 / (rpm * P.N_TEETH)
    tau = Tp / mc
    sj = (np.arange(n_enc) + 0.5) / n_enc
    h = aH(((np.arange(mc)[:, None] + sj[None, :]) * tau).ravel(), rpm, a, cut).reshape(mc, n_enc)
    L = aH_rate_bound(rpm, cut)
    if order == 1:
        lo, hi = h.min(axis=1), h.max(axis=1)
        alpha, beta = 0.5 * (lo + hi), np.zeros(mc)
        eps = 0.5 * (hi - lo) + L * tau / n_enc / 2
    else:
        V = np.column_stack([np.ones(n_enc), sj])
        coef = np.linalg.lstsq(V, h.T, rcond=None)[0]
        alpha, beta = coef[0], coef[1]
        eps = np.abs(h - alpha[:, None] - beta[:, None] * sj[None, :]).max(axis=1) \
            + (L + np.abs(beta) / tau) * tau / n_enc / 2
    return alpha, beta, eps


def corners(alpha, beta, eps):
    """Corners of the decrease LMI on each subinterval: tuples (s, c_k, c_k1)
    with A W(s) replaced by (1 - s) A0 W_k + s A0 W_k1 + c_k E W_k + c_k1 E W_k1.
    With h = alpha + beta s + r and W(s) = (1 - s) W_k + s W_k1, h W(s) is
    multi-affine in ((s, sigma = s^2), r), (s, s^2) lies in the triangle
    (0, 0), (1/2, 0), (1, 1) and r in [-eps, eps]: the corners are the
    3 x 2 products (2 x 2 when beta = 0, where sigma does not enter)."""
    out = []
    for al, be, ep in zip(alpha, beta, eps):
        pts = ((0.0, 0.0), (1.0, 1.0)) if be == 0 else ((0.0, 0.0), (0.5, 0.0), (1.0, 1.0))
        out.append([(sv, (al + r) * (1 - sv) + be * (sv - sg), (al + r) * sv + be * sg)
                    for sv, sg in pts for r in (-ep, ep)])
    return out


def h_range(alpha, beta, eps):
    """Bounds of a H on each subinterval implied by the enclosure."""
    return np.minimum(alpha, alpha + beta) - eps, np.maximum(alpha, alpha + beta) + eps


# --------------------------------------------------------------- model ----
def model(rpm, g, N=None):
    """A0 (no cutting), E, B, K with A(t) = A0 + a H(t) E."""
    A0, B, K = c2.model(rpm, 0.0, g, N)
    n = len(B)
    E = np.zeros((n, n))
    E[1, 0] = 1.0 / P.M
    E[1, n - 1] = -1.0 / P.M
    return A0, E, B, K


GL2 = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])


def propagators(A0, E, h_of_t, Tp, L):
    """Step propagators over L equal steps of one period, 4th-order Magnus
    (two Gauss points); h_of_t gives a H at given times."""
    dt = Tp / L
    hq = h_of_t(((np.arange(L)[:, None] + GL2[None, :]) * dt).ravel()).reshape(L, 2)
    Ph = np.empty((L,) + A0.shape)
    C = E @ A0 - A0 @ E                                            # [A2, A1] = (h2 - h1) [E, A0]
    for i in range(L):
        h1, h2 = hq[i]
        Om = dt / 2 * (2 * A0 + (h1 + h2) * E) + np.sqrt(3) / 12 * dt * dt * (h2 - h1) * C
        Ph[i] = expm(Om)
    return Ph


def monodromy(rpm, a, g, cut=HALF, N=None, L=200):
    A0, E, B, K = model(rpm, g, N)
    Tp = 60.0 / (rpm * P.N_TEETH)
    Phi = np.eye(len(B))
    for Pi in propagators(A0, E, lambda t: aH(t, rpm, a, cut), Tp, L):
        Phi = Pi @ Phi
    return Phi


def spectral_radius(rpm, a, g, cut=HALF, N=None, L=200):
    return float(np.abs(np.linalg.eigvals(monodromy(rpm, a, g, cut, N, L))).max())


def floquet_a_lim(rpm, g, cut=HALF, N=None, L=200, lo_frac=0.1, hi_frac=1.5, ratio=1.04, rtol=1e-5):
    """Smallest depth with spectral radius 1 (geometric scan from lo_frac of
    the ZOA limit, then bisection)."""
    az = cut.a_lim(rpm, g)
    lo, a = 0.0, lo_frac * az
    while spectral_radius(rpm, a, g, cut, N, L) < 1:
        lo, a = a, a * ratio
        if a > hi_frac * az:
            return np.inf
    hi = a
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if spectral_radius(rpm, mid, g, cut, N, L) < 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------ forced orbit ----
N_DENSE = 4096


def orbit(rpm, g, a, cut=HALF, N=None, mc=M_C, n_t=256):
    """C2's forced orbit at this immersion; also u0_k = V - max |b| on each
    of the mc subintervals (samples plus max|b''| dt^2 / 8)."""
    from structure import Gcl
    N = N or c2.nodes_for(rpm)
    wT = 2 * np.pi * P.N_TEETH * rpm / 60.0
    F = cut.force_harmonics(a)
    m = np.arange(len(F))
    w = m * wT
    X = np.empty(len(F), complex)
    X[0] = F[0] / P.K
    X[1:] = Gcl(1j * w[1:], g) * F[1:]
    Q = np.zeros_like(X)
    Q[1:] = -g * 1j * w[1:] * X[1:] / ((1j * w[1:]) ** 2 + P.GA_A1 * 1j * w[1:] + P.GA_A0)

    def at(c, t, deriv=0):
        t = np.atleast_1d(t)
        z = (1j * w) ** deriv * c
        return z[0].real * (deriv == 0) + 2 * np.real(np.exp(1j * np.outer(t, w[1:])) @ z[1:])

    th = c2.history_nodes(rpm, N)
    state = np.concatenate([at(X, 0.0), at(X, 0.0, 1), at(Q, 0.0), at(Q, 0.0, 1), at(X, th)])
    Tp = 2 * np.pi / wT
    b = -g * at(X, np.arange(n_t) / n_t * Tp, 1)
    q = -(-N_DENSE // mc)                                          # samples per subinterval
    s = (np.arange(mc)[:, None] + np.linspace(0.0, 1.0, q + 1)[None, :]) * (Tp / mc)
    bd = np.abs(-g * at(X, s.ravel(), 1)).reshape(s.shape)
    pad = 2 * g * np.sum(w[1:] ** 3 * np.abs(X[1:])) * (Tp / (mc * q)) ** 2 / 8
    return state, b, bd.max(axis=1) + pad


# ----------------------------------------------------------------- SDP ----
def _add_sym(blk, Mz, vw, I, J, coef, r0=0, c0=0):
    """coef (Mz W + W Mz') into blk, W symmetric with upper-triangle variables vw
    at (I, J); block offset (r0, c0) for both rows and columns."""
    n = Mz.shape[0]
    off = I != J
    Io, vo = I[off], vw[off]
    C = np.arange(n)
    rr = np.concatenate([np.tile(C, len(I)), np.tile(C, len(Io))])
    cc = np.concatenate([np.repeat(J, n), np.repeat(Io, n)])
    vv = np.concatenate([np.repeat(vw, n), np.repeat(vo, n)])
    uu = np.concatenate([Mz[:, I].T.ravel(), Mz[:, J[off]].T.ravel()])
    blk.add(rr + r0, cc + c0, vv, coef * uu)
    blk.add(cc + r0, rr + c0, vv, coef * uu)


def _sym(x, I, J, n):
    W = np.zeros((n, n))
    W[I, J] = x
    W[J, I] = x
    return W


def lmi(A0, E, B, K, enc, u0n, e, Tm, Tp, eps=1e-9, lam=0.0, settings=None, valid=None, merged=True):
    """Largest beta with beta e in E(P(t)) for all t, P periodic piecewise
    affine as in the module docstring.  enc: (alpha, beta, eps) of a H on
    each subinterval (enclosure); u0n: sector level at each node (the smaller
    of the two subintervals meeting there).  Returns beta and (W_k, Y_k, S_k)
    in the coordinates x = Tm z.

    merged: the two corners r = -eps, +eps of each (s, sigma) are one LMI.
    E has rank one, E = p q' in these coordinates, so the r-term of the
    decrease matrix is -r (p v' + v p') with v = W(s) q, and Petersen's lemma
    (exact for a scalar |r| <= eps) gives
        M0 - r (p v' + v p') > 0 for all |r| <= eps
        <=>  exists mu > 0:  [[M0 - mu p p', eps v], [eps v', mu]] > 0.
    It halves the decrease blocks; check() still tests all corners."""
    mc, n = len(enc[0]), len(B)
    Ti = np.linalg.inv(Tm)
    A0s, Es, Bs, Ks, es = Ti @ A0 @ Tm, Ti @ E @ Tm, Ti @ B, K @ Tm, Ti @ e
    I, J = np.triu_indices(n)
    nw = len(I)
    per = nw + n + 1
    vW = [k * per + np.arange(nw) for k in range(mc)]
    vY = [k * per + nw + np.arange(n) for k in range(mc)]
    vS = [k * per + nw + n for k in range(mc)]
    vb = mc * per
    tau = Tp / mc
    C = np.arange(n)
    off = I != J
    blocks = []
    if merged:
        p = Ti[:, 1]
        q = Tm.T @ E[1]                                              # E = e_2 E[1], so Es = p q'
        rho = np.sqrt(np.linalg.norm(p) / (np.linalg.norm(q) * max(np.max(enc[2]), 1e-300)))
        p, q = p / rho, q * rho                                      # balance mu (about 1) against the rest
        assert np.allclose(np.outer(p, q), Es, atol=1e-12 * np.abs(Es).max())
        mus = []
        alpha, beta, rad = enc
        cks_all = [[(sv, al * (1 - sv) + be * (sv - sg), al * sv + be * sg, ep) for sv, sg in
                    (((0.0, 0.0), (1.0, 1.0)) if be == 0 else ((0.0, 0.0), (0.5, 0.0), (1.0, 1.0)))]
                   for al, be, ep in zip(alpha, beta, rad)]
    else:
        cks_all = [[c + (0.0,) for c in cc] for cc in corners(*enc)]
    nvar = vb + 1
    for k, cks in enumerate(cks_all):
        k1 = (k + 1) % mc
        for sv, ck, ck1, ep in cks:
            big = merged and ep > 0
            M1 = LMIBlock(n + 2 if big else n + 1)
            if big:                                                   # [[M0 - mu p p', ep v], [ep v', mu]]
                vmu = nvar
                nvar += 1
                mus.append(vmu)
                Ip, Jp = np.triu_indices(n)
                M1.add(Ip, Jp, vmu, -p[Ip] * p[Jp])
                M1.add(n + 1, n + 1, vmu, 1.0)
                for v, wgt in ((vW[k], 1 - sv), (vW[k1], sv)):
                    if wgt:
                        M1.add(I, n + 1, v, ep * wgt * q[J])
                        M1.add(J[off], n + 1, v[off], ep * wgt * q[I[off]])
                M1.add(n + 1, n + 1, -1, -eps)
            M1.add(I, J, vW[k1], 1.0 / tau)                         # W' = (W_k1 - W_k) / tau
            M1.add(I, J, vW[k], -1.0 / tau)
            for v, wgt, c in ((vW[k], 1 - sv, ck), (vW[k1], sv, ck1)):
                if wgt or c:
                    _add_sym(M1, wgt * A0s + c * Es, v, I, J, -1.0)
                if lam > 0 and wgt:
                    M1.add(I, J, v, -2 * lam * wgt)
            M1.add(C, n, vS[k], Bs)
            for v, wgt in ((vY[k], 1 - sv), (vY[k1], sv)):
                if wgt:
                    M1.add(C, n, v, -wgt)
            M1.add(n, n, vS[k], 2.0)
            M1.add(np.arange(n + 1), np.arange(n + 1), -1, -eps)
            blocks.append(M1)
    cs = valid[0] @ Tm if valid is not None else None
    for k in range(mc):
        M2 = LMIBlock(n + 1)                                         # [[W, W K' - Y'], [K W - Y, u0^2]]
        M2.add(I, J, vW[k], 1.0)
        M2.add(I, n, vW[k], Ks[J])
        M2.add(J[off], n, vW[k][off], Ks[I[off]])
        M2.add(C, n, vY[k], -1.0)
        M2.add(n, n, -1, u0n[k] ** 2)
        M3 = LMIBlock(n + 1)                                         # [[1, beta e'], [beta e, W]]
        M3.add(0, 0, -1, 1.0)
        M3.add(0, 1 + C, vb, es)
        M3.add(1 + I, 1 + J, vW[k], 1.0)
        blocks += [M2, M3]
        if cs is not None:                                           # c W c' <= h^2 (scalar, W > 0)
            M4 = LMIBlock(1)
            M4.add(0, 0, vW[k], -cs[I] * cs[J] * np.where(off, 2.0, 1.0))
            M4.add(0, 0, -1, valid[1] ** 2)
            blocks.append(M4)
    c = np.zeros(nvar)
    c[vb] = 1.0
    nonneg = [(v, eps) for v in vS] + [(vb, 0.0)] + ([(v, eps) for v in mus] if merged else [])
    x, status = solve_sdp(len(c), c, blocks, nonneg=nonneg, settings=SOLVER if settings is None else settings)
    W = np.array([_sym(x[v], I, J, n) for v in vW])
    Y = np.array([x[v] for v in vY])
    S = np.array([x[v] for v in vS])
    return x[vb], (W, Y, S), status


def check(A0, E, B, K, enc, u0n, e, Tm, Tp, W, Y, S, valid=None):
    """Check a candidate independently of the solver (as c2.check, at every
    corner of every subinterval and at every node).  Returns (size, scale c, message)."""
    mc, n = len(enc[0]), len(B)
    Ti = np.linalg.inv(Tm)
    A0s, Es, Bs, Ks, es = Ti @ A0 @ Tm, Ti @ E @ Tm, Ti @ B, K @ Tm, Ti @ e
    W = (W + W.transpose(0, 2, 1)) / 2
    tau = Tp / mc
    for k in range(mc):
        lw = np.linalg.eigvalsh(W[k])
        if lw.min() <= 0 or lw.max() / lw.min() > 1e10:
            return 0.0, 0.0, f"W_{k} not positive definite or too ill-conditioned"
    worst = np.inf
    nA, nE = np.linalg.norm(A0s, 2), np.linalg.norm(Es, 2)
    for k, cks in enumerate(corners(*enc)):
        k1 = (k + 1) % mc
        Wd = (W[k1] - W[k]) / tau
        for sv, ck, ck1 in cks:
            AW = ((1 - sv) * A0s + ck * Es) @ W[k] + (sv * A0s + ck1 * Es) @ W[k1]
            Ys = (1 - sv) * Y[k] + sv * Y[k1]
            off = (Bs * S[k] - Ys)[:, None]
            M1 = np.block([[Wd - AW - AW.T, off], [off.T, np.array([[2.0 * S[k]]])]])
            ev = np.linalg.eigvalsh((M1 + M1.T) / 2)
            terms = max(np.abs(ev).max(), (nA + (abs(ck) + abs(ck1)) * nE) * max(np.linalg.norm(W[k], 2),
                        np.linalg.norm(W[k1], 2)), (np.linalg.norm(W[k], 2) + np.linalg.norm(W[k1], 2)) / tau)
            tol = 1e3 * (n + 1) * np.finfo(float).eps * terms      # covers rounding in W' and A W
            if ev.min() <= tol:
                return 0.0, 0.0, f"decrease not strict on subinterval {k} ({ev.min() / np.abs(ev).max():+.1e})"
            worst = min(worst, ev.min() / np.abs(ev).max())
    c = 1.0
    sizes = []
    cv = valid[0] @ Tm if valid is not None else None
    for k in range(mc):
        L = cholesky(W[k], lower=True)
        z = solve_triangular(L, W[k] @ Ks - Y[k], lower=True)
        q = z @ z
        if q > 0:
            c = min(c, (1 - 1e-9) * u0n[k] ** 2 / q)
        if cv is not None:
            qv = cv @ W[k] @ cv
            if qv > 0:
                c = min(c, (1 - 1e-9) * valid[1] ** 2 / qv)
        y = solve_triangular(L, es, lower=True)
        sizes.append(1.0 / (y @ y))
    size = float(np.sqrt(c * min(sizes)) * (1 - 1e-9))
    return size, c, ("ok" if c == 1.0 else f"ok, shrunk by {c:.12g}") + f"; min eig ratio {worst:.1e}"


LAM_FRAC = (0.05, 0.02, 0.01)                   # decay margins, fractions of the Floquet rate; an
#                                                  infeasible problem is retried with a smaller one
SOLVER = dict(direct_solve_method="faer", max_threads=1, max_iter=120)   # defaults otherwise; the check decides
REG_RETRY = 1e-7                                # static regularisation of the retry (default 1e-8)
N_PASS = 1                                      # solves per margin: the warm start (a re-centred second
#                                                 pass added 0.4 % at 2500 rpm, 0.9 a_lim)


def floquet_rate(A0, E, h_of_t, Tp, L=200):
    Phi = np.eye(A0.shape[0])
    for Pi in propagators(A0, E, h_of_t, Tp, L):
        Phi = Pi @ Phi
    return -np.log(np.abs(np.linalg.eigvals(Phi)).max()) / Tp


def certify(rpm, a, g, cut=HALF, e=None, N=None, mc=M_C, averaged=False, resolved=True, order=2,
            settings=None, delta=DELTA, full=False, n_pass=N_PASS):
    """Certified size along e (default: unit velocity, the kick of C1) on the
    periodic model.  averaged=True replaces a H(t) by its mean (C2's model,
    with the periodic W and the time-resolved margin); resolved=False uses
    C2's single level V - max|b| at every phase.

    The first solve starts in the coordinates of C2's certificate on the
    averaged model (it is feasible for the periodic LMIs of the averaged
    model with the single level, so it is a well-conditioned start), then
    re-centres n_pass - 1 times.  With full=True returns a dict with the
    size, C2's size from the warm start, the size at each node (a kick at
    phase t_k), the margins and the certificate (Tm, W_k, Y_k, S_k)."""
    A0, E, B, K = model(rpm, g, N)
    n = len(B)
    Tp = 60.0 / (rpm * P.N_TEETH)
    if e is None:
        e = np.zeros(n)
        e[1] = 1.0
    Abar = A0 + cut.kappa * a * E
    if averaged:
        enc = (np.full(mc, cut.kappa * a), np.zeros(mc), np.zeros(mc))
        sigma = -np.linalg.eigvals(Abar).real.max()
    else:
        enc = enclosure(rpm, a, cut, mc, order=order)
        sigma = floquet_rate(A0, E, lambda t: aH(t, rpm, a, cut), Tp)
    _, _, bk = orbit(rpm, g, a, cut, N, mc)
    u0k = VMAX - (bk if resolved else np.full(mc, bk.max()))
    u0n = np.minimum(u0k, np.roll(u0k, 1))                       # node k closes subinterval k-1
    out = dict(sigma=sigma, u0_min=float(u0k.min()), u0_max=float(u0k.max()), c2_size=np.nan,
               phase_size=np.zeros(N_PHASE), size=0.0, cert=None)
    if sigma <= 0 or u0k.min() <= 0:
        out.update(note="linearly unstable" if sigma <= 0 else "orbit saturates")
        return out if full else (0.0, out["note"])
    valid = c2.validity(n, delta) if delta is not None else None
    s2, _, cert2 = c2.certify(Abar, B, K, u0k.min(), e, rpm, TIGHT, full=True, delta=delta)
    out["c2_size"] = s2
    start = cert2[0] @ cholesky(np.linalg.inv(cert2[1]), lower=True) if cert2 is not None else None
    best, note, cert = 0.0, "no candidate passed the check", None
    for f in LAM_FRAC:
        Tm = start if start is not None else np.diag(c2.scaling(Abar, rpm))
        for p in range(n_pass):
            for reg in (None, REG_RETRY):         # the first KKT solve can fail; retry regularised
                st = dict(SOLVER if settings is None else settings)
                if reg is not None:
                    st["static_regularization_constant"] = reg
                beta, (W, Y, S), status = lmi(A0, E, B, K, enc, u0n, e, Tm, Tp, lam=f * sigma,
                                              settings=st, valid=valid)
                size, c, msg = check(A0, E, B, K, enc, u0n, e, Tm, Tp, W, Y, S, valid=valid)
                if size > 0:
                    break
            if size > best:
                best = min(size, beta) if beta > 0 else size
                note = f"{status}; pass {p}, lam {f:g} sigma{', regularised' if reg else ''}; {msg}"
                cert = (Tm.copy(), c * W, c * Y, c * S)
            try:
                Tm = Tm @ cholesky(np.mean((W + W.transpose(0, 2, 1)) / 2, axis=0), lower=True)
            except np.linalg.LinAlgError:
                break
        if best > 0:
            break
    if cert is not None:                       # size at the N_PHASE kick phases of C1 (W(t) is affine between nodes)
        es = np.linalg.solve(cert[0], e)
        Wt = W_at(cert, np.arange(N_PHASE), N_PHASE)
        out["phase_size"] = np.array([(1 - 1e-9) / np.sqrt(es @ np.linalg.solve(Wk, es)) for Wk in Wt])
    out.update(size=best, note=note, cert=cert)
    return out if full else (best, note)


# ------------------------------------------- simulating the periodic model ----
def step_maps(A0, E, B, h_of_t, Tp, L):
    """Exact-in-the-input step maps over L steps of the period: x_{i+1} =
    Ad_i x_i - Bd_i psi_i (psi held over a step; A(t) frozen at its Gauss
    points through the 4th-order Magnus propagator, the input through the
    mid-step matrix)."""
    n = len(B)
    dt = Tp / L
    Ad = propagators(A0, E, h_of_t, Tp, L)
    hm = h_of_t((np.arange(L) + 0.5) * dt)
    Bd = np.empty((L, n))
    for i in range(L):
        Mx = np.zeros((n + 1, n + 1))
        Mx[:n, :n] = A0 + hm[i] * E
        Mx[:n, n] = B
        Bd[i] = expm(Mx * dt)[:n, n]
    return Ad, Bd


def run_model(Ad, Bd, K, b, X0, k0, nper, record=False):
    """c2.run_model with the periodic step maps (b on the same L steps)."""
    L = len(b)
    X = np.array(X0, float)
    idx = np.array(k0, int) % L
    hist = [X.copy()] if record else None
    for _ in range(nper * L):
        w = K @ X
        bb = b[idx]
        psi = w - (np.clip(bb + w, -VMAX, VMAX) - bb)
        X = np.einsum("kij,jk->ik", Ad[idx], X) - Bd[idx].T * psi
        idx = (idx + 1) % L
        if record:
            hist.append((X.copy(), idx.copy()))
    return (X, hist) if record else X


def W_at(cert, idx, L):
    """W(t) at step indices idx of a period of L steps (nodes every L / mc)."""
    _, W, _, _ = cert
    mc = len(W)
    s = (np.asarray(idx) % L) * mc / L
    k = np.floor(s).astype(int) % mc
    f = s - np.floor(s)
    return (1 - f)[:, None, None] * W[k] + f[:, None, None] * W[(k + 1) % mc]


def proof_check(Ad, Bd, K, b, cert, nruns=48, nper=40, seed=0):
    """Start at 0.999 of the boundary of E(P(t0)) in random directions and
    phases t0; return the largest V(t, x(t)) reached and the largest final V."""
    Tm = cert[0]
    L = len(b)
    rng = np.random.default_rng(seed)
    n = Tm.shape[0]
    k0 = rng.integers(0, L, nruns)
    U = rng.standard_normal((n, nruns))
    U /= np.linalg.norm(U, axis=0)
    W0 = W_at(cert, k0, L)
    Z0 = np.einsum("kij,jk->ik", np.linalg.cholesky(W0), U)
    _, H = run_model(Ad, Bd, K, b, 0.999 * Tm @ Z0, k0, nper, record=True)
    Ti = np.linalg.inv(Tm)
    vmax = vend = 0.0
    for X, idx in H[1:]:
        Z = Ti @ X
        Wt = W_at(cert, idx, L)
        V = np.einsum("ik,ki->k", Z, np.linalg.solve(Wt, Z.T[:, :, None])[:, :, 0])
        vmax, vend = max(vmax, V.max()), V.max()
    return float(vmax), float(vend)


# ------------------------------------------ off-axis test in the simulator ----
def td_offaxis(rpm, a, g, cert, nodes=None, kick_rev=100, window=300):
    """c2.td_offaxis with the periodic certificate: the boundary points of
    E(P(t_k)) (c2.boundary_directions) applied as full-state deviations at
    the phase t_k of node k (C1's kick protocol), at 4 nodes a quarter
    period apart by default.  Returns (chattering, runs)."""
    from tdsim import simulate
    from tdboundary import classify_run
    Tm, W = cert[0], cert[1]
    mc, n = len(W), Tm.shape[0]
    spr = P.STEPS_PER_REV
    nd = spr // P.N_TEETH
    assert nd % mc == 0, "nodes must fall on simulator steps"
    nodes = range(0, mc, max(1, mc // 4)) if nodes is None else nodes
    bad = runs = 0
    for k in nodes:
        ks = kick_rev * spr + k * nd // mc
        for X in c2.boundary_directions((Tm, np.linalg.inv(W[k])), n):
            r = simulate(rpm, a, g, vmax=VMAX, store=True, nrev=kick_rev + window, edge=True, nreg=4,
                         kick_step=ks, pert=c2.deviation_to_pert(rpm, X))
            st = (ks // nd + 1) * nd
            bad += not classify_run(r["x"][st:], spr, r["diverged"])["stable"]
            runs += 1
    return bad, runs


# ------------------------------------------------------------ campaigns ----
def gain_map():
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    return d["rpm"], d["g_best"], float(d["zeta"])


def critical_multiplier(rpm, a, g, cut=HALF, N=None):
    ev = np.linalg.eigvals(monodromy(rpm, a, g, cut, N))
    return ev[np.argmax(np.abs(ev))]


FLOQ_AE = (0.5, 0.1)


def _floquet(args):
    rpm, g, ae, zeta = args
    P.set_damping(zeta)
    cut = Cut(ae)
    az = cut.a_lim(rpm, g)
    af = floquet_a_lim(rpm, g, cut)
    if not np.isfinite(af):
        return az, af, np.nan, np.nan, np.nan
    mu = critical_multiplier(rpm, af * (1 + 1e-4), g, cut)
    N = c2.nodes_for(rpm)
    rho8 = spectral_radius(rpm, af, g, cut, N=N + 8)               # node convergence at the limit
    return az, af, mu.real, mu.imag, rho8


def run_floquet(a):
    """Stability limit of the periodic collocated model (Floquet) against the
    ZOA on the gain map of C2, at half and low radial immersion."""
    rpm_map, g_map, zeta = gain_map()
    rpm = np.arange(rpm_map[0], rpm_map[-1] + 1, a.step)
    g = np.interp(rpm, rpm_map, g_map)
    jobs = [(float(n), float(gg), ae, zeta) for ae in FLOQ_AE for n, gg in zip(rpm, g)]
    with Pool(a.workers) as pool:
        res = np.array(pool.map(_floquet, jobs, chunksize=1)).reshape(len(FLOQ_AE), len(rpm), 5)
    np.savez_compressed(os.path.join(RES, "c3_floquet.npz"), rpm=rpm, g=g, ae=np.array(FLOQ_AE),
                        a_zoa=res[..., 0], a_floquet=res[..., 1], mu=res[..., 2] + 1j * res[..., 3],
                        rho_n8=res[..., 4], zeta=zeta)
    for i, ae in enumerate(FLOQ_AE):
        r = res[i, :, 1] / res[i, :, 0]
        flip = (res[i, :, 2] < 0) & (np.abs(res[i, :, 3]) < 1e-6 * np.abs(res[i, :, 2]))
        print(f"ae/D {ae}: Floquet / ZOA {np.nanmin(r):.4f}-{np.nanmax(r):.4f} (median {np.nanmedian(r):.4f}); "
              f"flip at {np.sum(flip)} speeds; |rho(N+8) - 1| <= {np.nanmax(np.abs(res[i, :, 4] - 1)):.1e}")


def model_kick_by_phase(Ad, Bd, K, b, phases, nper=1200, lo=1e-4, hi=3.0, nbis=9):
    """c2.model_critical_kick on the given step maps, at each phase index
    separately (log bisection per phase)."""
    n, L = Bd.shape[1], len(b)
    e = np.zeros(n)
    e[1] = 1.0
    k0 = np.asarray(phases, int) % L

    def diverges(dv):
        with np.errstate(over="ignore", invalid="ignore"):
            X = run_model(Ad, Bd, K, b, e[:, None] * dv[None, :], k0, nper)
            nrm = np.linalg.norm(X, axis=0)
        return ~np.isfinite(nrm) | (nrm > 10 * dv)

    m = len(k0)
    out = np.full(m, np.nan)
    at_lo = diverges(np.full(m, lo))
    at_hi = diverges(np.full(m, hi))
    out[at_lo] = lo
    out[~at_hi] = np.inf
    todo = ~at_lo & at_hi
    a_, c_ = np.full(m, np.log(lo)), np.full(m, np.log(hi))
    for _ in range(nbis):
        mid = 0.5 * (a_ + c_)
        d = diverges(np.exp(mid))
        c_ = np.where(d, mid, c_)
        a_ = np.where(d, a_, mid)
    out[todo] = np.exp(c_[todo])
    return out


L_SIM = 256                                       # steps per tooth period of the model runs


def set_immersion(cut):
    """The simulator's engagement window (tdsim reads it at each call)."""
    import tdsim
    tdsim.PHI_ST, tdsim.PHI_EX = cut.st, cut.ex


def _point(rpm, g, a, cut, offaxis=True):
    """Certificate and model checks at one cut: the periodic certificate,
    C2's (warm start), the proof on the periodic model, the critical kick of
    the periodic and of the averaged model at C1's N_PHASE kick phases, and the
    off-axis simulator test of E(P(t_k)) at 4 phases."""
    r = certify(rpm, a, g, cut, full=True)
    A0, E, B, K = model(rpm, g)
    Tp = 60.0 / (rpm * P.N_TEETH)
    _, b, _ = orbit(rpm, g, a, cut, n_t=L_SIM)
    ph = np.arange(N_PHASE) * L_SIM // N_PHASE
    Ad, Bd = step_maps(A0, E, B, lambda t: aH(t, rpm, a, cut), Tp, L_SIM)
    Aa, Ba = step_maps(A0, E, B, lambda t: np.full(np.size(t), cut.kappa * a), Tp, L_SIM)
    cert = r["cert"]
    proof = proof_check(Ad, Bd, K, b, cert) if cert is not None else (np.nan, np.nan)
    set_immersion(cut)
    off = td_offaxis(rpm, a, g, cert) if (cert is not None and offaxis) else (0, 0)
    return dict(cert=r["size"], c2_cert=r["c2_size"], phase_cert=r["phase_size"], u0_min=r["u0_min"],
                u0_max=r["u0_max"], sigma=r["sigma"], proof_vmax=proof[0], proof_vend=proof[1],
                model_kick=model_kick_by_phase(Ad, Bd, K, b, ph), avg_kick=model_kick_by_phase(Aa, Ba, K, b, ph),
                offaxis_bad=off[0], offaxis_runs=off[1], msg=r["note"])


def _stack(res, keys):
    return {k: np.array([r[k] for r in res]) for k in keys}


KEYS = ("cert", "c2_cert", "phase_cert", "u0_min", "u0_max", "sigma", "proof_vmax", "proof_vend", "model_kick",
        "avg_kick", "offaxis_bad", "offaxis_runs", "msg")


def _report(out, td):
    for i in range(len(out["cert"])):
        print(f"{out['rpm'][i]:.0f} rpm {out['frac'][i]:.2f} a_lim: C3 {out['cert'][i]*1e3:7.3f} (C2 {out['c2_cert'][i]*1e3:7.3f}) "
              f"mm/s; per phase {out['phase_cert'][i].min()*1e3:.3f}-{out['phase_cert'][i].max()*1e3:.3f}; model periodic "
              f"{np.nanmin(out['model_kick'][i])*1e3:8.2f}, averaged {np.nanmin(out['avg_kick'][i])*1e3:8.2f}, "
              f"TD {np.nanmin(td[i])*1e3:8.2f}; max V {out['proof_vmax'][i]:.5f}; off-axis "
              f"{out['offaxis_bad'][i]}/{out['offaxis_runs'][i]}  [{out['msg'][i]}]", flush=True)
    c, c2s, ph = out["cert"], out["c2_cert"], out["phase_cert"]
    ok = c > 0
    q = c[ok] / c2s[ok]
    print(f"C3 / C2: {q.min():.3f}-{q.max():.3f} (median {np.median(q):.3f})" if ok.any() else "C3 / C2: -",
          f"at {ok.sum()}/{len(c)} points")
    mk, ak = out["model_kick"], out["avg_kick"]
    both = np.isfinite(mk) & np.isfinite(ak)
    print(f"periodic / averaged model critical kick, phase by phase: {np.min(mk[both] / ak[both]):.3f}-"
          f"{np.max(mk[both] / ak[both]):.3f} (median {np.median(mk[both] / ak[both]):.3f})")
    print(f"per-phase certificate below the periodic model's kick at the same phase: {np.sum((ph <= mk)[ok])}/{ok.sum() * N_PHASE}")
    print(f"C2 certificate below the periodic model's smallest kick: {np.sum(c2s <= np.nanmin(mk, axis=1))}/{len(c2s)}")
    fin = np.isfinite(td) & ok[:, None]
    tdp = td if td.shape[1] == N_PHASE else None
    if tdp is not None:
        print(f"per-phase certificate below the simulated kick at the same phase: {np.sum((ph <= td)[fin])}/{fin.sum()}")
    tdmin = np.array([np.min(r[np.isfinite(r)]) if np.isfinite(r).any() else np.inf for r in td])
    print(f"certificate below the simulated smallest kick: {np.sum(c[ok] <= tdmin[ok])}/{ok.sum()}")
    print(f"proof check max V {np.nanmax(out['proof_vmax']):.5f}; off-axis chatter {out['offaxis_bad'].sum()}/"
          f"{out['offaxis_runs'].sum()}")


def _basin(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    return _point(rpm, g, frac * a_lim(rpm, g), HALF)


def run_basin(a):
    """At the C1 basin points (half immersion): the periodic certificate (any
    phase and per phase), C2's from the warm start, the proof checked on the
    periodic model, the periodic and averaged models' critical kicks at the
    16 kick phases of C1, the simulated critical kick at those phases (C1),
    and boundary points of E(P(t_k)) at 4 phases in the simulator."""
    b = np.load(os.path.join(RES, "c1_basin.npz"))
    assert int(b["n_phase"]) == N_PHASE, "C1's kick phases"
    m = b["vmax"] > 0
    jobs = [(float(n), float(g), float(f), float(b["zeta"])) for n, g, f in zip(b["rpm"][m], b["g"][m], b["frac"][m])]
    if a.limit:
        jobs = jobs[:a.limit]
    with Pool(a.workers) as pool:
        res = pool.map(_basin, jobs, chunksize=1)
    out = _stack(res, KEYS)
    out.update(rpm=np.array([j[0] for j in jobs]), g=np.array([j[1] for j in jobs]),
               frac=np.array([j[2] for j in jobs]), td_kick=b["dv"][m][:len(jobs)], m_c=M_C, delta=DELTA)
    np.savez_compressed(os.path.join(RES, "c3_basin.npz"), **out)
    _report(out, out["td_kick"])


LOW_AE = 0.1
LOW_FRAC = (0.5, 0.8, 0.9, 0.98)
LOW_PHASES = (0, 4, 8, 12)                        # of C1's 16 kick phases


def _low_td(args):
    """C1's simulated critical kick at low immersion."""
    from c1_unsafe_zone import _critical_kick
    set_immersion(Cut(LOW_AE))
    return _critical_kick(args)


def _low(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    cut = Cut(LOW_AE)
    return _point(rpm, g, frac * cut.a_lim(rpm, g), cut)


def run_lowimm(a):
    """Low radial immersion (ae/D = LOW_AE), where a H(t) swings much more
    (0.5-1.3 of its mean at 10 mm): the same checks as run_basin at the C1
    speeds, with the simulated critical kick at 4 of C1's 16 phases."""
    from c1_unsafe_zone import BASIN_RPM
    rpm_map, g_map, zeta = gain_map()
    cut = Cut(LOW_AE)
    pts = [(float(n), float(np.interp(n, rpm_map, g_map)), f, zeta) for n in BASIN_RPM for f in LOW_FRAC]
    tdj = [(n, f * cut.a_lim(n, g), g, VMAX, p, z) for (n, g, f, z) in pts for p in LOW_PHASES]
    with Pool(a.workers) as pool:
        td = np.array(pool.map(_low_td, tdj, chunksize=1)).reshape(len(pts), len(LOW_PHASES))
        res = pool.map(_low, pts, chunksize=1)
    out = _stack(res, KEYS)
    out.update(rpm=np.array([p[0] for p in pts]), g=np.array([p[1] for p in pts]), frac=np.array([p[2] for p in pts]),
               a=np.array([p[2] * cut.a_lim(p[0], p[1]) for p in pts]), td_kick=td, td_phases=np.array(LOW_PHASES),
               ae=LOW_AE, m_c=M_C, delta=DELTA)
    np.savez_compressed(os.path.join(RES, "c3_lowimm.npz"), **out)
    _report(out, td)
    ph = out["phase_cert"][:, list(LOW_PHASES)]
    fin = np.isfinite(td) & (out["cert"] > 0)[:, None]
    print(f"per-phase certificate below the simulated kick at the same phase: {np.sum((ph <= td)[fin])}/{fin.sum()}")


DEC_RPM = (1090, 1420, 2500, 2870)
DEC_FRAC = (0.5, 0.9, 0.98)
DEC_VARIANTS = (("averaged, resolved", dict(averaged=True)),
                ("periodic, single level", dict(resolved=False)),
                ("periodic, resolved, m_c 8", dict(mc=8)),
                ("periodic, resolved, m_c 32", dict(mc=32)))


def _decomp(args):
    rpm, g, frac, zeta, kw = args
    P.set_damping(zeta)
    r = certify(rpm, frac * a_lim(rpm, g), g, full=True, **kw)
    return r["size"], r["c2_size"]


def run_decomp(a):
    """What each ingredient adds: the time-resolved margin on the averaged
    model, the periodic coefficient with C2's single level, and the number
    of subintervals (8, 16 from run_basin, 32)."""
    b = np.load(os.path.join(RES, "c1_basin.npz"))
    pts = [(float(n), float(g), float(f), float(b["zeta"])) for n, g, f, v in zip(b["rpm"], b["g"], b["frac"], b["vmax"])
           if v > 0 and int(n) in DEC_RPM and any(abs(f - x) < 1e-9 for x in DEC_FRAC)]
    jobs = [p + (kw,) for p in pts for _, kw in DEC_VARIANTS]
    with Pool(a.workers) as pool:
        res = np.array(pool.map(_decomp, jobs, chunksize=1)).reshape(len(pts), len(DEC_VARIANTS), 2)
    np.savez_compressed(os.path.join(RES, "c3_decomp.npz"), rpm=np.array([p[0] for p in pts]),
                        frac=np.array([p[2] for p in pts]), variants=np.array([v for v, _ in DEC_VARIANTS]),
                        size=res[..., 0], c2_size=res[:, 0, 1])
    for p, row in zip(pts, res):
        print(f"{p[0]:.0f} rpm {p[2]:.2f}: C2 {row[0, 1]*1e3:.3f}; " +
              "; ".join(f"{v} {s*1e3:.3f}" for (v, _), s in zip(DEC_VARIANTS, row[:, 0])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["floquet", "basin", "decomp", "lowimm"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=int, default=20, help="floquet: rpm spacing")
    ap.add_argument("--limit", type=int, default=0, help="basin: first points only (dry run)")
    a = ap.parse_args()
    dict(floquet=run_floquet, basin=run_basin, decomp=run_decomp, lowimm=run_lowimm)[a.command](a)


if __name__ == "__main__":
    main()
