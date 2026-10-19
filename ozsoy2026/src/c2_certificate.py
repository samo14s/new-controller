"""C2: certified basin of the clipped DVF cut (saturated-LMI certificate).

Model.  The deviation from the forced periodic orbit obeys the averaged
(ZOA) milling DDE with the DVF loop (verification/zoa_sdm: ZOA and SDM agree to
2e-5 here).  The delayed displacement is carried by its values at N Chebyshev
nodes on [-T, 0] (pseudospectral collocation, Breda-Maset-Vermiglio), which
gives an ODE of order 4 + N with spectral accuracy; N is chosen per speed.
The clipped command is u = K x - psi(K x), where psi is the deadzone of the
command around the forced orbit's command b(t).  The linear part K x stays
inside the dynamics:  x' = A x - B psi(K x).

Sector.  For |b(t)| <= b_max < V, the deadzone of sat(b + w) - b satisfies the
generalised sector condition with level u0 = V - b_max, at every t (see
docs in the report, section 12): if |w - v| <= u0 then psi (psi - v) <= 0.
With v = G x, a quadratic V(x) = x' P x and T > 0 this gives the LMIs
(W = P^-1, Y = G W, S = 1/T; Gomes da Silva & Tarbouriech 2005):
    [[-(A W + W A'), B S - Y'], [S B' - Y, 2 S]] > 0
    [[W, W K' - Y'], [K W - Y, u0^2]] >= 0
so E(P) = {x' P x <= 1} is contractive and invariant, and every deviation
inside it returns to the orbit.  The size of E(P) along a direction e is
maximised through [[1, beta e'], [beta e, W]] >= 0.

Directions certified here:
  kick   e = unit velocity (the kick of C1): beta is a certified kick (m/s)
  step   e = minus the orbit's state at t = 0, history included (the cut
         starting at rest on a fresh surface): beta >= 1 certifies the entry

Every solution is re-checked in the original P-form with exact eigenvalues.
If the check fails, the problem is solved again with a larger margin.
    python c2_certificate.py validate      -> ../results/c2_validate.npz
    python c2_certificate.py stepmap       -> ../results/c2_stepmap.npz/.csv
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
import clarabel
import params as P
from zoa import a_lim, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VMAX = P.F_MAX / P.GA_GAIN
KAPPA = P.N_TEETH * P.K_TC * ALPHA / (4 * np.pi)      # dynamic force = KAPPA a (x - x(t - T))
R2 = np.sqrt(2.0)
TIGHT = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=400)


# --------------------------------------------------------------- model ----
def cheb(N):
    """Chebyshev differentiation matrix on x_j = cos(pi j / N) (Trefethen)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(N + 1))
    return D - np.diag(D.sum(axis=1)), x


def nodes_for(rpm):
    """Collocation nodes: enough for the delays per structural period."""
    T = 60.0 / (rpm * P.N_TEETH)
    return int(16 + 4 * np.ceil(T * P.FN))


def model(rpm, a, g, N=None):
    """x' = A x + B (-psi), argument of the deadzone K x;
    x = [x, xdot, q, qdot, phi_1 .. phi_N], phi_j = x(t + theta_j)."""
    N = N or nodes_for(rpm)
    T = 60.0 / (rpm * P.N_TEETH)
    D, _ = cheb(N)
    Dt = (2.0 / T) * D                       # theta = T (xc - 1) / 2: node 0 at 0, node N at -T
    n = 4 + N
    A = np.zeros((n, n))
    A[0, 1] = 1.0
    A[1, 0] = (-P.K + KAPPA * a) / P.M
    A[1, 1] = -P.C / P.M
    A[1, 2] = -P.GA_GAIN * P.GA_A0 / P.M
    A[1, 3] = -P.GA_GAIN * P.GA_A1 / P.M
    A[2, 3] = 1.0
    A[3, 2] = -P.GA_A0
    A[3, 3] = -P.GA_A1
    B = np.zeros(n)
    B[1] = P.GA_GAIN / P.M
    B[3] = 1.0
    K = np.zeros(n)
    K[1] = -g
    A += np.outer(B, K)                      # linear DVF inside the dynamics
    A[1, n - 1] += -KAPPA * a / P.M          # x(t - T) = phi_N
    A[4:, 0] = Dt[1:, 0]                     # phi_0 = x
    A[4:, 4:] = Dt[1:, 1:]
    return A, B, K


def history_nodes(rpm, N=None):
    N = N or nodes_for(rpm)
    T = 60.0 / (rpm * P.N_TEETH)
    return T * (cheb(N)[1][1:] - 1) / 2


# ------------------------------------------------------------ forced orbit ----
def orbit(rpm, g, a, N=None, n_t=256):
    """Forced periodic orbit of the cut (regeneration vanishes on it).

    Returns the model state at t = 0 (history at the collocation nodes) and
    the command b(t) = -g v(t) over one tooth period.
    """
    from feasibility import force_harmonics, N_HARM
    from structure import Gcl
    N = N or nodes_for(rpm)
    wT = 2 * np.pi * P.N_TEETH * rpm / 60.0
    m = np.arange(N_HARM + 1)
    F = force_harmonics([a])[0]                                    # F_m, m = 0..N_HARM
    w = m * wT
    X = np.empty(N_HARM + 1, complex)
    X[0] = F[0] / P.K                                              # static: Ga(0) = 0
    X[1:] = Gcl(1j * w[1:], g) * F[1:]
    Q = np.zeros_like(X)
    Q[1:] = -g * 1j * w[1:] * X[1:] / ((1j * w[1:]) ** 2 + P.GA_A1 * 1j * w[1:] + P.GA_A0)

    def at(c, t, deriv=0):
        t = np.atleast_1d(t)
        z = (1j * w) ** deriv * c
        return z[0].real * (deriv == 0) + 2 * np.real(np.exp(1j * np.outer(t, w[1:])) @ z[1:])

    th = history_nodes(rpm, N)
    state = np.concatenate([at(X, 0.0), at(X, 0.0, 1), at(Q, 0.0), at(Q, 0.0, 1), at(X, th)])
    tt = np.arange(n_t) / n_t * 2 * np.pi / wT
    b = -g * at(X, tt, 1)
    return state, b


# ----------------------------------------------------------------- SDP ----
class LMIBlock:
    """Sparse symmetric LMI  F0 + sum_i x_i F_i >= 0  in Clarabel's PSD
    triangle format (upper triangle, column-major, off-diagonals x sqrt 2).
    Entries are added once for each off-diagonal pair (row <= col kept)."""

    def __init__(self, size):
        self.N = size
        self.parts = []

    def add(self, r, c, var, val):
        r, c, var, val = np.broadcast_arrays(*(np.atleast_1d(np.asarray(z)) for z in (r, c, var, val)))
        up = r <= c
        self.parts.append((r[up], c[up], var[up], val[up].astype(float)))

    def assemble(self, nvar):
        r, c, v, x = (np.concatenate(z) for z in zip(*self.parts))
        lo, hi = np.minimum(r, c), np.maximum(r, c)
        idx = hi * (hi + 1) // 2 + lo
        x = np.where(r == c, x, x * R2)
        L = self.N * (self.N + 1) // 2
        const = v < 0
        b = np.zeros(L)
        np.add.at(b, idx[const], x[const])
        A = sp.coo_matrix((-x[~const], (idx[~const], v[~const])), shape=(L, nvar)).tocsc()
        return A, b


def solve_sdp(nvar, c_obj, blocks, nonneg=(), settings=None):
    """maximise c_obj' x subject to the LMI blocks and x_v >= lower."""
    As, bs, cones = [], [], []
    if nonneg:
        vs = np.array([v for v, _ in nonneg])
        As.append(sp.coo_matrix((-np.ones(len(vs)), (np.arange(len(vs)), vs)), shape=(len(vs), nvar)).tocsc())
        bs.append(-np.array([lb for _, lb in nonneg]))
        cones.append(clarabel.NonnegativeConeT(len(vs)))
    for blk in blocks:
        A, b = blk.assemble(nvar)
        As.append(A)
        bs.append(b)
        cones.append(clarabel.PSDTriangleConeT(blk.N))
    s = clarabel.DefaultSettings()
    s.verbose = False
    for k, v in (settings or {}).items():
        setattr(s, k, v)
    sol = clarabel.DefaultSolver(sp.csc_matrix((nvar, nvar)), -np.asarray(c_obj, float),
                                 sp.vstack(As).tocsc(), np.concatenate(bs), cones, s).solve()
    return np.array(sol.x), str(sol.status)


# --------------------------------------------------------- certificate ----
def scaling(A, rpm, steps=4000, revs=50):
    """State scales: peak of each state in the linear response to a 1 mm/s kick."""
    n = A.shape[0]
    E = expm(A * revs * 60.0 / rpm / steps)
    x = np.zeros(n)
    x[1] = 1e-3
    mx = np.abs(x).copy()
    for _ in range(steps):
        x = E @ x
        mx = np.maximum(mx, np.abs(x))
    mx[mx == 0] = 1.0
    return mx


def lmi(A, B, K, u0, e, d, eps=1e-9, lam=0.0, settings=TIGHT):
    """Largest beta with beta e in a certified ellipsoid.  Returns beta and
    (W, Y, S) in the scaled coordinates x = diag(d) xs.  lam > 0 asks for the
    decay rate V' <= -2 lam V inside the ellipsoid (a margin that scales with
    W, so the solution stays strictly feasible after round-off)."""
    n = len(B)
    Di = 1 / d
    As, Bs, Ks, es = Di[:, None] * A * d[None, :], Di * B, K * d, Di * e
    I, J = np.triu_indices(n)
    nw = len(I)
    vW, vY, vS, vb = np.arange(nw), nw + np.arange(n), nw + n, nw + n + 1
    off = I != J
    Io, vo = I[off], vW[off]
    C = np.arange(n)
    M1 = LMIBlock(n + 1)                       # [[-(A W + W A') - 2 lam W, B S - Y'], [S B' - Y, 2 S]] - eps I
    rr = np.concatenate([np.tile(C, nw), np.tile(C, len(Io))])
    cc = np.concatenate([np.repeat(J, n), np.repeat(Io, n)])
    vv = np.concatenate([np.repeat(vW, n), np.repeat(vo, n)])
    uu = np.concatenate([As[:, I].T.ravel(), As[:, J[off]].T.ravel()])    # U = A E
    M1.add(rr, cc, vv, -uu)
    M1.add(cc, rr, vv, -uu)                    # + U'
    if lam > 0:
        M1.add(I, J, vW, -2 * lam)
    M1.add(C, n, vS, Bs)
    M1.add(C, n, vY, -1.0)
    M1.add(n, n, vS, 2.0)
    M1.add(np.arange(n + 1), np.arange(n + 1), -1, -eps)
    M2 = LMIBlock(n + 1)                       # [[W, W K' - Y'], [K W - Y, u0^2]]
    M2.add(I, J, vW, 1.0)
    M2.add(I, n, vW, Ks[J])
    M2.add(J[off], n, vo, Ks[Io])
    M2.add(C, n, vY, -1.0)
    M2.add(n, n, -1, u0 ** 2)
    M3 = LMIBlock(n + 1)                       # [[1, beta e'], [beta e, W]]
    M3.add(0, 0, -1, 1.0)
    M3.add(0, 1 + C, vb, es)
    M3.add(1 + I, 1 + J, vW, 1.0)
    c = np.zeros(nw + n + 2)
    c[vb] = 1.0
    x, status = solve_sdp(len(c), c, [M1, M2, M3], nonneg=[(vS, eps), (vb, 0.0)], settings=settings)
    W = np.zeros((n, n))
    W[I, J] = x[:nw]
    W[J, I] = x[:nw]
    return x[vb], (W, x[nw:nw + n], x[nw + n]), status


def check(A, B, K, u0, e, d, W, Y, S, rel=1e-9):
    """Re-check the certificate from (W, Y, S) in the P-form, exactly.

    Returns the certified size along e (0 if a condition fails):
    decrease  [[A'P + P A, -P B + G' T], [-B' P + T G, -2 T]] < 0 by a
              relative margin rel (the decay-rate margin of lmi() supplies it),
    inclusion (K - G) P^-1 (K - G)' <= u0^2, with P = W^-1, G = Y P, T = 1/S.
    """
    Di = 1 / d
    As, Bs, Ks, es = Di[:, None] * A * d[None, :], Di * B, K * d, Di * e
    lw = np.linalg.eigvalsh(W)
    if lw.min() <= 0:
        return 0.0, "W not positive"
    Pm = np.linalg.inv(W)
    Pm = (Pm + Pm.T) / 2
    G = Y @ Pm
    T = 1.0 / S
    M = np.block([[As.T @ Pm + Pm @ As, (-Pm @ Bs + G * T)[:, None]],
                  [(-Bs @ Pm + T * G)[None, :], np.array([[-2 * T]])]])
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    if ev.max() >= -rel * np.abs(ev).max():
        return 0.0, f"decrease fails ({ev.max() / np.abs(ev).max():+.1e})"
    incl = (Ks - G) @ W @ (Ks - G) / u0 ** 2
    shrink = 1.0 / np.sqrt(incl) if incl > 1 else 1.0          # shrink E(P) if it pokes out
    return shrink / np.sqrt(es @ Pm @ es), "ok"


LAM_FRAC = (0.05, 0.2, 0.5)                    # decay-rate margins tried, as fractions of the linear rate


def certify(A, B, K, u0, e, rpm):
    """Certified size along e, checked in the P-form; the decay-rate margin
    is raised until the check passes."""
    sigma = -np.linalg.eigvals(A).real.max()
    if sigma <= 0:
        return 0.0, "linearly unstable"
    if u0 <= 0:
        return 0.0, "orbit saturates"
    d = scaling(A, rpm)
    for f in LAM_FRAC:
        beta, (W, Y, S), status = lmi(A, B, K, u0, e, d, lam=f * sigma)
        size, msg = check(A, B, K, u0, e, d, W, Y, S)
        if size > 0:
            return min(size, beta), f"{status}; checked, lam = {f:g} sigma"
    return 0.0, f"{status}; {msg}"


def lmi_periodic(A, B, K, u0s, e, d, Tp, lam=0.0, kick_all_phases=True, settings=TIGHT):
    """Periodic certificate: W(t), Y(t), S(t) piecewise linear over the tooth
    period on len(u0s) intervals, W(Tp) = W(0).  Interval i has the margin
    u0s[i] = V - max |b(t)| over that interval.  Decrease on interval i:
        [[Wdot - (A W + W A') - 2 lam W, B S - Y'], [S B' - Y, 2 S]] >= 0
    with Wdot = (W_{i+1} - W_i) / dt; affine in t, so it is imposed at both
    ends.  The size along e is imposed at every knot (a kick at any phase) or
    only at t = 0 (the entry).  Returns beta and the knot values (scaled)."""
    n, Kp = len(B), len(u0s)
    dt = Tp / Kp
    Di = 1 / d
    As, Bs, Ks, es = Di[:, None] * A * d[None, :], Di * B, K * d, Di * e
    I, J = np.triu_indices(n)
    nw = len(I)
    per = nw + n + 1                                   # W, Y, S per knot
    vb = Kp * per
    off = I != J
    Io = I[off]
    C = np.arange(n)
    rr = np.concatenate([np.tile(C, nw), np.tile(C, len(Io))])
    cc = np.concatenate([np.repeat(J, n), np.repeat(Io, n)])
    idx = np.concatenate([np.repeat(np.arange(nw), n), np.repeat(np.arange(nw)[off], n)])
    uu = np.concatenate([As[:, I].T.ravel(), As[:, J[off]].T.ravel()])

    def vW(k): return (k % Kp) * per + np.arange(nw)
    def vY(k): return (k % Kp) * per + nw + np.arange(n)
    def vS(k): return (k % Kp) * per + nw + n

    blocks = []
    for i in range(Kp):
        for k in (i, i + 1):                           # both ends of interval i
            M1 = LMIBlock(n + 1)
            M1.add(rr, cc, vW(k)[idx], -uu)
            M1.add(cc, rr, vW(k)[idx], -uu)
            M1.add(I, J, vW(i + 1), 1.0 / dt)           # Wdot
            M1.add(I, J, vW(i), -1.0 / dt)
            if lam > 0:
                M1.add(I, J, vW(k), -2 * lam)
            M1.add(C, n, vS(k), Bs)
            M1.add(C, n, vY(k), -1.0)
            M1.add(n, n, vS(k), 2.0)
            blocks.append(M1)
            M2 = LMIBlock(n + 1)
            M2.add(I, J, vW(k), 1.0)
            M2.add(I, n, vW(k), Ks[J])
            M2.add(J[off], n, vW(k)[off], Ks[Io])
            M2.add(C, n, vY(k), -1.0)
            M2.add(n, n, -1, u0s[i] ** 2)
            blocks.append(M2)
    for k in (range(Kp) if kick_all_phases else (0,)):
        M3 = LMIBlock(n + 1)
        M3.add(0, 0, -1, 1.0)
        M3.add(0, 1 + C, vb, es)
        M3.add(1 + I, 1 + J, vW(k), 1.0)
        blocks.append(M3)
    nvar = vb + 1
    c = np.zeros(nvar)
    c[vb] = 1.0
    x, status = solve_sdp(nvar, c, blocks, nonneg=[(vS(k), 1e-12) for k in range(Kp)] + [(vb, 0.0)],
                          settings=settings)
    knots = []
    for k in range(Kp):
        W = np.zeros((n, n))
        W[I, J] = x[vW(k)]
        W[J, I] = x[vW(k)]
        knots.append((W, x[vY(k)], x[vS(k)]))
    return x[vb], knots, status


def check_periodic(A, B, K, u0s, e, d, Tp, knots, lam, kick_all_phases=True, rel=1e-9, nsub=8):
    """Exact re-check of the periodic certificate.  On each interval the
    decrease matrix is affine in t; it is checked at the ends and at nsub
    interior points in the P-form P(t) = W(t)^-1 with Pdot = -P Wdot P.
    Returns the certified size along e (0 if a check fails)."""
    Kp = len(u0s)
    dt = Tp / Kp
    Di = 1 / d
    As, Bs, Ks, es = Di[:, None] * A * d[None, :], Di * B, K * d, Di * e
    size = np.inf
    for i in range(Kp):
        W0, Y0, S0 = knots[i]
        W1, Y1, S1 = knots[(i + 1) % Kp]
        Wd = (W1 - W0) / dt
        for s in np.linspace(0, 1, nsub + 2):
            W = (1 - s) * W0 + s * W1
            Y = (1 - s) * Y0 + s * Y1
            S = (1 - s) * S0 + s * S1
            if np.linalg.eigvalsh(W).min() <= 0 or S <= 0:
                return 0.0, "W or S not positive"
            Pm = np.linalg.inv(W)
            Pm = (Pm + Pm.T) / 2
            G, T = Y @ Pm, 1.0 / S
            Pd = -Pm @ Wd @ Pm
            M = np.block([[Pd + As.T @ Pm + Pm @ As, (-Pm @ Bs + G * T)[:, None]],
                          [(-Bs @ Pm + T * G)[None, :], np.array([[-2 * T]])]])
            ev = np.linalg.eigvalsh((M + M.T) / 2)
            if ev.max() >= -rel * np.abs(ev).max():
                return 0.0, f"decrease fails on interval {i} ({ev.max() / np.abs(ev).max():+.1e})"
            incl = (Ks - G) @ W @ (Ks - G) / u0s[i] ** 2
            if incl > 1 + 1e-9:
                return 0.0, f"inclusion fails on interval {i} ({incl:.6f})"
            if kick_all_phases or (i == 0 and s == 0):
                size = min(size, 1.0 / np.sqrt(es @ Pm @ es))
    return size, "ok"


def margins(b, Kp):
    """Margin per interval: V - max |b| over each of Kp equal parts of the period."""
    parts = np.array_split(np.abs(b), Kp)
    return np.array([VMAX - p.max() for p in parts])


def certify_periodic(A, B, K, b, e, rpm, Kp=16, kick_all_phases=True):
    sigma = -np.linalg.eigvals(A).real.max()
    if sigma <= 0:
        return 0.0, "linearly unstable"
    u0s = margins(b, Kp)
    if u0s.min() <= 0:
        return 0.0, "orbit saturates"
    d = scaling(A, rpm)
    Tp = 60.0 / (rpm * P.N_TEETH)
    for f in LAM_FRAC:
        beta, knots, status = lmi_periodic(A, B, K, u0s, e, d, Tp, lam=f * sigma,
                                           kick_all_phases=kick_all_phases)
        size, msg = check_periodic(A, B, K, u0s, e, d, Tp, knots, f * sigma, kick_all_phases)
        if size > 0:
            return min(size, beta), f"{status}; checked, lam = {f:g} sigma"
    return 0.0, f"{status}; {msg}"


def certify_kick(rpm, a, g, N=None):
    A, B, K = model(rpm, a, g, N)
    _, b = orbit(rpm, g, a, N)
    e = np.zeros(len(B))
    e[1] = 1.0
    return certify(A, B, K, VMAX - np.abs(b).max(), e, rpm)


def certify_step(rpm, a, g, N=None):
    """beta >= 1: the cut entered at rest on a fresh surface is certified."""
    A, B, K = model(rpm, a, g, N)
    x0, b = orbit(rpm, g, a, N)
    return certify(A, B, K, VMAX - np.abs(b).max(), -x0, rpm)


def certify_kick_periodic(rpm, a, g, N=None, Kp=16):
    A, B, K = model(rpm, a, g, N)
    _, b = orbit(rpm, g, a, N)
    e = np.zeros(len(B))
    e[1] = 1.0
    return certify_periodic(A, B, K, b, e, rpm, Kp)


def certify_step_periodic(rpm, a, g, N=None, Kp=16):
    A, B, K = model(rpm, a, g, N)
    x0, b = orbit(rpm, g, a, N)
    return certify_periodic(A, B, K, b, -x0, rpm, Kp, kick_all_phases=False)


# ------------------------------------------------------------ campaigns ----
def _validate(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    a = frac * a_lim(rpm, g)
    beta, msg = certify_kick(rpm, a, g)
    return beta, msg


def run_validate(a):
    """Certified kick at the C1 basin points, against the simulated one."""
    b = np.load(os.path.join(RES, "c1_basin.npz"))
    m = b["vmax"] > 0
    jobs = [(float(n), float(g), float(f), float(b["zeta"])) for n, g, f in zip(b["rpm"][m], b["g"][m], b["frac"][m])]
    with Pool(a.workers) as pool:
        res = pool.map(_validate, jobs, chunksize=1)
    cert = np.array([r[0] for r in res])
    td_min = b["dv"][m].min(axis=1)
    np.savez_compressed(os.path.join(RES, "c2_validate.npz"), rpm=b["rpm"][m], g=b["g"][m], frac=b["frac"][m],
                        cert=cert, td_min=td_min, td_median=np.median(b["dv"][m], axis=1),
                        msg=np.array([r[1] for r in res]))
    for (n, g, f, _), c, t, (_, msg) in zip(jobs, cert, td_min, res):
        print(f"{n:.0f} rpm {f:.2f} a_lim: certified {c*1e3:7.3f} mm/s, TD min {t*1e3:8.2f} mm/s, "
              f"ratio {t / c if c > 0 else np.inf:6.2f}  [{msg}]")
    ok = np.isfinite(td_min) & (cert > 0)
    print(f"sound (certified <= TD min) at {np.sum(cert[ok] <= td_min[ok])}/{ok.sum()} points")


def _step_depth(args):
    """Largest depth with a certified step entry (bisection on depth)."""
    rpm, g, zeta = args
    P.set_damping(zeta)
    al = a_lim(rpm, g)
    lo, hi = 0.0, 0.999
    if certify_step(rpm, hi * al, g)[0] >= 1:
        return hi * al * 1e3
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        if certify_step(rpm, mid * al, g)[0] >= 1:
            lo = mid
        else:
            hi = mid
    return lo * al * 1e3


def run_stepmap(a):
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    rpm, g = d["rpm"], d["g_best"]
    sel = (rpm - rpm[0]) % a.step == 0
    rpm, g = rpm[sel], g[sel]
    with Pool(a.workers) as pool:
        cert = np.array(pool.map(_step_depth, [(float(n), float(gg), float(d["zeta"])) for n, gg in zip(rpm, g)],
                                 chunksize=1))
    np.savez_compressed(os.path.join(RES, "c2_stepmap.npz"), rpm=rpm, g=g, a_cert_step=cert)
    np.savetxt(os.path.join(RES, "c2_stepmap.csv"), np.column_stack([rpm, g, cert]), delimiter=",",
               header="rpm,g,a_cert_step", comments="", fmt="%.4f")
    print(f"certified step-entry depth at {len(rpm)} speeds: {cert.min():.2f}-{cert.max():.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "stepmap"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=int, default=20, help="stepmap: rpm spacing")
    a = ap.parse_args()
    dict(validate=run_validate, stepmap=run_stepmap)[a.command](a)


if __name__ == "__main__":
    main()
