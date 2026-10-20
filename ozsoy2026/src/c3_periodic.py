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
N_ENC = 128                                        # samples per subinterval for the enclosures


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


def enclosure(rpm, a, cut=HALF, mc=M_C, n_enc=N_ENC):
    """[h_k-, h_k+] containing a H(t) on each of the mc subintervals."""
    Tp = 60.0 / (rpm * P.N_TEETH)
    s = np.arange(mc * n_enc + 1) / (mc * n_enc) * Tp
    h = aH(s, rpm, a, cut)
    pad = aH_rate_bound(rpm, cut) * (Tp / (mc * n_enc)) / 2
    lo = np.array([h[k * n_enc:(k + 1) * n_enc + 1].min() for k in range(mc)]) - pad
    hi = np.array([h[k * n_enc:(k + 1) * n_enc + 1].max() for k in range(mc)]) + pad
    return lo, hi


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


def lmi(A0, E, B, K, hlo, hhi, u0n, e, Tm, Tp, eps=1e-9, lam=0.0, settings=TIGHT, valid=None):
    """Largest beta with beta e in E(P(t)) for all t, P periodic piecewise
    affine as in the module docstring.  hlo, hhi: enclosure of a H on each
    subinterval; u0n: sector level at each node (the smaller of the two
    subintervals meeting there).  Returns beta and (W_k, Y_k, S_k) in the
    coordinates x = Tm z."""
    mc, n = len(hlo), len(B)
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
    blocks = []
    for k in range(mc):
        k1 = (k + 1) % mc
        for node in (k, k1):
            for h in (hlo[k], hhi[k]):
                M1 = LMIBlock(n + 1)
                M1.add(I, J, vW[k1], 1.0 / tau)                     # W' = (W_k1 - W_k) / tau
                M1.add(I, J, vW[k], -1.0 / tau)
                _add_sym(M1, A0s + h * Es, vW[node], I, J, -1.0)
                if lam > 0:
                    M1.add(I, J, vW[node], -2 * lam)
                M1.add(C, n, vS[k], Bs)
                M1.add(C, n, vY[node], -1.0)
                M1.add(n, n, vS[k], 2.0)
                M1.add(np.arange(n + 1), np.arange(n + 1), -1, -eps)
                blocks.append(M1)
    off = I != J
    for k in range(mc):
        M2 = LMIBlock(n + 1)
        M2.add(I, J, vW[k], 1.0)
        M2.add(I, n, vW[k], Ks[J])
        M2.add(J[off], n, vW[k][off], Ks[I[off]])
        M2.add(C, n, vY[k], -1.0)
        M2.add(n, n, -1, u0n[k] ** 2)
        M3 = LMIBlock(n + 1)
        M3.add(0, 0, -1, 1.0)
        M3.add(0, 1 + C, vb, es)
        M3.add(1 + I, 1 + J, vW[k], 1.0)
        blocks += [M2, M3]
        if valid is not None:
            cs = valid[0] @ Tm
            M4 = LMIBlock(n + 1)
            M4.add(I, J, vW[k], 1.0)
            M4.add(I, n, vW[k], cs[J])
            M4.add(J[off], n, vW[k][off], cs[I[off]])
            M4.add(n, n, -1, valid[1] ** 2)
            blocks.append(M4)
    c = np.zeros(vb + 1)
    c[vb] = 1.0
    x, status = solve_sdp(len(c), c, blocks, nonneg=[(v, eps) for v in vS] + [(vb, 0.0)], settings=settings)
    W = np.array([_sym(x[v], I, J, n) for v in vW])
    Y = np.array([x[v] for v in vY])
    S = np.array([x[v] for v in vS])
    return x[vb], (W, Y, S), status


def check(A0, E, B, K, hlo, hhi, u0n, e, Tm, Tp, W, Y, S, valid=None):
    """Check a candidate independently of the solver (as c2.check, at every
    corner of every subinterval and at every node).  Returns (size, scale c, message)."""
    mc, n = len(hlo), len(B)
    Ti = np.linalg.inv(Tm)
    A0s, Es, Bs, Ks, es = Ti @ A0 @ Tm, Ti @ E @ Tm, Ti @ B, K @ Tm, Ti @ e
    W = (W + W.transpose(0, 2, 1)) / 2
    tau = Tp / mc
    for k in range(mc):
        lw = np.linalg.eigvalsh(W[k])
        if lw.min() <= 0 or lw.max() / lw.min() > 1e10:
            return 0.0, 0.0, f"W_{k} not positive definite or too ill-conditioned"
    worst = np.inf
    for k in range(mc):
        k1 = (k + 1) % mc
        Wd = (W[k1] - W[k]) / tau
        for node in (k, k1):
            for h in (hlo[k], hhi[k]):
                As = A0s + h * Es
                off = (Bs * S[k] - Y[node])[:, None]
                M1 = np.block([[Wd - As @ W[node] - W[node] @ As.T, off], [off.T, np.array([[2.0 * S[k]]])]])
                ev = np.linalg.eigvalsh((M1 + M1.T) / 2)
                scale = max(np.abs(ev).max(), np.linalg.norm(As, 2) * np.linalg.norm(W[node], 2),
                            (np.linalg.norm(W[k], 2) + np.linalg.norm(W[k1], 2)) / tau)
                tol = 1e3 * (n + 1) * np.finfo(float).eps * scale    # also covers rounding in W' and A W
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


LAM_FRAC = c2.LAM_FRAC
N_PASS = c2.N_PASS


def floquet_rate(A0, E, h_of_t, Tp, L=200):
    Phi = np.eye(A0.shape[0])
    for Pi in propagators(A0, E, h_of_t, Tp, L):
        Phi = Pi @ Phi
    return -np.log(np.abs(np.linalg.eigvals(Phi)).max()) / Tp


def certify(rpm, a, g, cut=HALF, e=None, N=None, mc=M_C, averaged=False, resolved=True,
            settings=TIGHT, delta=DELTA, full=False):
    """Certified size along e (default: unit velocity, the kick of C1) on the
    periodic model.  averaged=True replaces a H(t) by its mean (C2's model,
    with the periodic W and the time-resolved margin); resolved=False uses
    C2's single level V - max|b| at every phase."""
    A0, E, B, K = model(rpm, g, N)
    n = len(B)
    Tp = 60.0 / (rpm * P.N_TEETH)
    if e is None:
        e = np.zeros(n)
        e[1] = 1.0
    if averaged:
        hlo = hhi = np.full(mc, cut.kappa * a)
        sigma = -np.linalg.eigvals(A0 + cut.kappa * a * E).real.max()
    else:
        hlo, hhi = enclosure(rpm, a, cut, mc)
        sigma = floquet_rate(A0, E, lambda t: aH(t, rpm, a, cut), Tp)
    _, _, bk = orbit(rpm, g, a, cut, N, mc)
    u0k = VMAX - (bk if resolved else np.full(mc, bk.max()))
    u0n = np.minimum(u0k, np.roll(u0k, 1))                       # node k closes subinterval k-1
    out = dict(sigma=sigma, u0_min=float(u0k.min()), u0_max=float(u0k.max()))
    if sigma <= 0 or u0k.min() <= 0:
        out.update(size=0.0, note="linearly unstable" if sigma <= 0 else "orbit saturates", cert=None)
        return out if full else (0.0, out["note"])
    valid = c2.validity(n, delta) if delta is not None else None
    best, note, cert = 0.0, "no candidate passed the check", None
    for f in LAM_FRAC:
        Tm = np.diag(c2.scaling(A0 + cut.kappa * a * E, rpm))
        for p in range(N_PASS):
            beta, (W, Y, S), status = lmi(A0, E, B, K, hlo, hhi, u0n, e, Tm, Tp, lam=f * sigma,
                                          settings=settings, valid=valid)
            size, c, msg = check(A0, E, B, K, hlo, hhi, u0n, e, Tm, Tp, W, Y, S, valid=valid)
            if size > best:
                best = min(size, beta) if beta > 0 else size
                note = f"{status}; pass {p}, lam {f:g} sigma; {msg}"
                cert = (Tm.copy(), c * W, c * Y, c * S)
            try:
                Tm = Tm @ cholesky(np.mean((W + W.transpose(0, 2, 1)) / 2, axis=0), lower=True)
            except np.linalg.LinAlgError:
                break
        if best > 0:
            break
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


def model_critical_kick(Ad, Bd, K, b, nphase=4, nper=1200, lo=1e-4, hi=3.0, nbis=9):
    """c2.model_critical_kick on the periodic model."""
    n, L = Bd.shape[1], len(b)
    e = np.zeros(n)
    e[1] = 1.0
    k0 = (np.arange(nphase) * L) // nphase

    def diverges(dv):
        with np.errstate(over="ignore", invalid="ignore"):
            X = run_model(Ad, Bd, K, b, np.outer(e, np.full(nphase, dv)), k0, nper)
            nrm = np.linalg.norm(X, axis=0)
        return ~np.isfinite(nrm) | (nrm > 10 * dv)

    if np.all(diverges(lo)):
        return lo
    if not np.any(diverges(hi)):
        return np.inf
    a, c = np.log(lo), np.log(hi)
    for _ in range(nbis):
        m = 0.5 * (a + c)
        if np.any(diverges(np.exp(m))):
            c = m
        else:
            a = m
    return float(np.exp(c))
