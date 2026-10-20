"""C2: certified basin of the clipped DVF cut (saturated-LMI certificate).

What is proved, and for which model.  The certificate is a theorem about
the averaged model below, not about the time-domain simulator.  The averaged
model is validated against the simulator separately (run_validate: its
critical kick against the simulated one).

Model.  The deviation from the forced periodic orbit obeys the averaged
(ZOA) milling DDE with the DVF loop (verification/zoa_sdm: ZOA and SDM agree
to 2e-5 on the stability limit here).  The delayed displacement is carried
by its values at N Chebyshev nodes on [-T, 0] (pseudospectral collocation,
Breda-Maset-Vermiglio).  This gives an ODE of order 4 + N whose rightmost
eigenvalues match the DDE's to 3e-8; N is chosen per speed.  The clipped
command is u = K x - psi(K x), where psi is the deadzone of the command
around the forced orbit's command b(t).  The linear part K x stays inside the
dynamics:  x' = A x - B psi(K x).  Left out (the simulator has them): the
time-periodic cutting coefficient (+-12 % around its average) and the chip
going to zero near the exit angle.

Sector.  Let b_max bound |b(t)| over the whole period (dense samples plus an
inter-sample bound from the orbit's harmonics).  For |b(t)| <= b_max < V,
the deadzone of sat(b + w) - b satisfies the generalised sector condition
with level u0 = V - b_max at every t: if |w - v| <= u0 then psi (psi - v) <= 0.
With v = G x, a quadratic V(x) = x' P x and T > 0 this gives the LMIs
(W = P^-1, Y = G W, S = 1/T; Gomes da Silva & Tarbouriech 2005):
    M1 = [[-(A W + W A') - 2 lam W, B S - Y'], [S B' - Y, 2 S]] > 0
    M2 = [[W, W K' - Y'], [K W - Y, u0^2]] >= 0
so E(P) = {x' P x <= 1} is invariant and every deviation inside it returns
to the orbit (V' <= -2 lam V).  The size of E(P) along a direction e is
maximised through [[1, beta e'], [beta e, W]] >= 0.

Validity.  The averaged model cuts linearly; the real chip clips at zero.  A
large deviation lifts the tool out of the cut, and then the model no longer
describes the cut.  The review found such states inside an unconstrained
E(P) (history deviations of 5 fz) that chatter in the simulator.  So E(P) is
also kept inside |x - x(t - T)| <= DELTA fz:
    M4 = [[W, W c'], [c W, (DELTA fz)^2]] >= 0,   c = e_x - e_phi_N.
E(P) is invariant, so the bound holds along the whole trajectory.

Directions certified here:
  kick   e = unit velocity (the kick of C1): beta is a certified kick (m/s)
  step   e = minus the orbit's state at t = 0, history included (the cut
         starting at rest on a fresh surface): beta >= 1 certifies the entry

Numerics.  Clarabel stops short of the optimum on these problems, so each
problem is solved again in the coordinates whitened by the previous W
(re-centring).  Every candidate is then checked independently of the solver:
  * M1 at lam = 0 must have its smallest eigenvalue above a round-off bound;
  * the inclusion and the size are computed through a Cholesky factor of W;
  * an inclusion that is exceeded is repaired by shrinking (W, Y).
    python c2_certificate.py validate      -> ../results/c2_validate.npz
    python c2_certificate.py surface       -> ../results/c2_surface.npz (kick, 40 rpm x 7 depths)
    python c2_certificate.py step          -> ../results/c2_step.npz (step entry, C1 basin speeds)
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm, cholesky, solve_triangular
import clarabel
import params as P
from zoa import a_lim, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VMAX = P.F_MAX / P.GA_GAIN
KAPPA = P.N_TEETH * P.K_TC * ALPHA / (4 * np.pi)      # dynamic force = KAPPA a (x - x(t - T))
R2 = np.sqrt(2.0)
TIGHT = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=400)
DELTA = 0.5          # validity of the linear cutting model: |x - x(t - T)| <= DELTA fz inside E(P)
FAST = {}                                   # solver defaults (faster; re-centring recovers most of the gap)


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
N_DENSE = 4096


def orbit(rpm, g, a, N=None, n_t=256):
    """Forced periodic orbit of the cut (regeneration vanishes on it).

    Returns the model state at t = 0 (history at the collocation nodes), the
    command b(t) = -g v(t) on n_t samples of one tooth period, and b_max, a
    bound on |b(t)| over the whole period: the largest of N_DENSE samples plus
    max|b''| dt^2 / 8, with |b''| <= 2 g sum_m w_m^3 |X_m|.  The orbit is
    truncated to the N_HARM tooth-passing harmonics of feasibility.py.
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
    Tp = 2 * np.pi / wT
    b = -g * at(X, np.arange(n_t) / n_t * Tp, 1)
    bd = -g * at(X, np.arange(N_DENSE) / N_DENSE * Tp, 1)
    b2 = 2 * g * np.sum(w[1:] ** 3 * np.abs(X[1:]))
    bmax = np.abs(bd).max() + b2 * (Tp / N_DENSE) ** 2 / 8
    return state, b, bmax


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


def transform(A, B, K, e, Tm):
    """Coordinates x = Tm z (Tm a vector means diag(Tm))."""
    Tm = np.diag(Tm) if np.ndim(Tm) == 1 else Tm
    Ti = np.linalg.inv(Tm)
    return Ti @ A @ Tm, Ti @ B, K @ Tm, Ti @ e


def lmi(A, B, K, u0, e, Tm, eps=1e-9, lam=0.0, settings=TIGHT, valid=None):
    """Largest beta with beta e in a certified ellipsoid.  Returns beta and
    (W, Y, S) in the coordinates x = Tm z.  lam > 0 asks for the decay rate
    V' <= -2 lam V inside the ellipsoid; it is a margin that scales with W,
    so the solution stays strictly feasible after round-off.  valid = (c, h)
    adds c x <= h on E(P) (the validity bound)."""
    n = len(B)
    As, Bs, Ks, es = transform(A, B, K, e, Tm)
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
    blocks = [M1, M2, M3]
    if valid is not None:
        cs = valid[0] @ (np.diag(Tm) if np.ndim(Tm) == 1 else Tm)
        M4 = LMIBlock(n + 1)                   # [[W, W c'], [c W, h^2]]
        M4.add(I, J, vW, 1.0)
        M4.add(I, n, vW, cs[J])
        M4.add(J[off], n, vo, cs[Io])
        M4.add(n, n, -1, valid[1] ** 2)
        blocks.append(M4)
    c = np.zeros(nw + n + 2)
    c[vb] = 1.0
    x, status = solve_sdp(len(c), c, blocks, nonneg=[(vS, eps), (vb, 0.0)], settings=settings)
    W = np.zeros((n, n))
    W[I, J] = x[:nw]
    W[J, I] = x[:nw]
    return x[vb], (W, x[nw:nw + n], x[nw + n]), status


def check(A, B, K, u0, e, Tm, W, Y, S, valid=None):
    """Check a candidate (W, Y, S) independently of the solver.

    Decrease: M1 at lam = 0 must be positive definite.  The test is its
    smallest computed eigenvalue against 1e3 n eps ||M1|| (eigvalsh is
    backward stable, so this is beyond round-off).  W must be positive
    definite, with condition number below 1e10.
    Inclusion: q = (K W - Y) W^-1 (K W - Y)' <= u0^2; if not, (W, Y) is
    scaled by c = u0^2 / q, which keeps M1 > 0.  The validity bound
    c_v W c_v' <= h^2 is enforced the same way.
    Size: beta = sqrt(c / e' W^-1 e).  Both quadratic forms come from a
    Cholesky factor of W and lose 1e-9 for safety.
    Returns (size, message); size 0 means the candidate is rejected.
    """
    As, Bs, Ks, es = transform(A, B, K, e, Tm)
    n = len(Bs)
    W = (W + W.T) / 2
    lw = np.linalg.eigvalsh(W)
    if lw.min() <= 0 or lw.max() / lw.min() > 1e10:
        return 0.0, "W not positive definite or too ill-conditioned"
    M1 = np.block([[-(As @ W + W @ As.T), (Bs * S - Y)[:, None]],
                   [(Bs * S - Y)[None, :], np.array([[2.0 * S]])]])
    M1 = (M1 + M1.T) / 2
    ev = np.linalg.eigvalsh(M1)
    tol = 1e3 * (n + 1) * np.finfo(float).eps * np.abs(ev).max()
    if ev.min() <= tol:
        return 0.0, f"decrease not strict ({ev.min() / np.abs(ev).max():+.1e})"
    L = cholesky(W, lower=True)
    z = solve_triangular(L, W @ Ks - Y, lower=True)
    q = z @ z
    c = min(1.0, (1 - 1e-9) * u0 ** 2 / q) if q > 0 else 1.0
    if valid is not None:
        cv = valid[0] @ (np.diag(Tm) if np.ndim(Tm) == 1 else Tm)
        qv = cv @ W @ cv
        if qv > 0:
            c = min(c, (1 - 1e-9) * valid[1] ** 2 / qv)
    y = solve_triangular(L, es, lower=True)
    return float(np.sqrt(c / (y @ y)) * (1 - 1e-9)), ("ok" if c == 1.0 else f"ok, shrunk by {c:.12g}")


LAM_FRAC = (0.05, 0.2)                         # decay-rate margins tried, as fractions of the linear rate
N_PASS = 3                                      # solves per margin: the first, then re-centred ones


def validity(n, delta=DELTA):
    """(c, h): the regenerative chip deviation x - x(t - T) bounded by delta fz."""
    c = np.zeros(n)
    c[0], c[n - 1] = 1.0, -1.0
    return c, delta * P.FZ


def certify(A, B, K, u0, e, rpm, settings=TIGHT, full=False, delta=DELTA):
    """Largest checked size along e, over re-centred solves, with E(P) kept
    inside the validity bound (delta = None drops it).  With full=True also
    returns the certificate (Tm, P) with V(x) = z' P z, z = Tm^-1 x."""
    valid = validity(len(B), delta) if delta is not None else None
    sigma = -np.linalg.eigvals(A).real.max()
    if sigma <= 0:
        return (0.0, "linearly unstable", None) if full else (0.0, "linearly unstable")
    if u0 <= 0:
        return (0.0, "orbit saturates", None) if full else (0.0, "orbit saturates")
    best, note, cert = 0.0, "no candidate passed the check", None
    for f in LAM_FRAC:
        Tm = np.diag(scaling(A, rpm))
        for p in range(N_PASS):
            beta, (W, Y, S), status = lmi(A, B, K, u0, e, Tm, lam=f * sigma, settings=settings, valid=valid)
            size, msg = check(A, B, K, u0, e, Tm, W, Y, S, valid=valid)
            if size > best:
                best = min(size, beta) if beta > 0 else size
                note = f"{status}; pass {p}, lam {f:g} sigma; {msg}"
                c = float(msg.split("by ")[1].rstrip(")")) if "shrunk" in msg else 1.0
                cert = (Tm.copy(), np.linalg.inv(c * (W + W.T) / 2))
            try:
                Tm = Tm @ cholesky((W + W.T) / 2, lower=True)
            except np.linalg.LinAlgError:
                break
        if best > 0:
            break
    return (best, note, cert) if full else (best, note)


def certify_kick(rpm, a, g, N=None, settings=TIGHT, delta=DELTA):
    A, B, K = model(rpm, a, g, N)
    _, _, bmax = orbit(rpm, g, a, N)
    e = np.zeros(len(B))
    e[1] = 1.0
    return certify(A, B, K, VMAX - bmax, e, rpm, settings, delta=delta)


def certify_step(rpm, a, g, N=None, settings=TIGHT, delta=DELTA):
    """beta >= 1: the cut entered at rest on a fresh surface is certified."""
    A, B, K = model(rpm, a, g, N)
    x0, _, bmax = orbit(rpm, g, a, N)
    return certify(A, B, K, VMAX - bmax, -x0, rpm, settings, delta=delta)


# ------------------------------------------------- simulating the model ----
def _zoh(A, B, h):
    n = len(B)
    Mx = np.zeros((n + 1, n + 1))
    Mx[:n, :n] = A
    Mx[:n, n] = B
    E = expm(Mx * h)
    return E[:n, :n], E[:n, n]


def run_model(A, B, K, b, Tp, X0, k0, nper, record=False):
    """Simulate the averaged model with the true deadzone around b(t).

    b holds len(b) samples of one tooth period Tp; the step is Tp/len(b),
    the linear part is integrated exactly and the deadzone is held over a
    step.  X0: n x m start deviations, k0: phase index of each.  Runs nper
    periods; returns the final states, and every state if record.
    """
    L = len(b)
    Ad, Bd = _zoh(A, B, Tp / L)
    X = np.array(X0, float)
    idx = np.array(k0, int) % L
    hist = [X.copy()] if record else None
    for _ in range(nper * L):
        w = K @ X
        bb = b[idx]
        psi = w - (np.clip(bb + w, -VMAX, VMAX) - bb)
        X = Ad @ X - np.outer(Bd, psi)
        idx = (idx + 1) % L
        if record:
            hist.append(X.copy())
    return (X, np.array(hist)) if record else X


def proof_check(A, B, K, b, Tp, cert, nruns=48, nper=40, seed=0):
    """Start on the boundary of the certified ellipsoid (0.999 of it) in
    random directions and phases; return the largest V reached and the
    largest final V.  The theorem says max V <= 1 (up to the step) and
    V -> 0."""
    Tm, Pz = cert
    rng = np.random.default_rng(seed)
    n = len(B)
    Lp = cholesky(np.linalg.inv(Pz), lower=True)                    # Wz = Lp Lp'
    U = rng.standard_normal((n, nruns))
    U /= np.linalg.norm(U, axis=0)
    X0 = 0.999 * Tm @ Lp @ U
    k0 = rng.integers(0, len(b), nruns)
    _, H = run_model(A, B, K, b, Tp, X0, k0, nper, record=True)
    Ti = np.linalg.inv(Tm)
    Z = np.einsum("ij,tjk->tik", Ti, H)
    V = np.einsum("tik,ij,tjk->tk", Z, Pz, Z)
    return float(V.max()), float(V[-1].max())


def model_critical_kick(A, B, K, b, Tp, nphase=4, nper=1200, lo=1e-4, hi=3.0, nbis=9):
    """Smallest velocity kick on the orbit that makes the averaged model
    diverge, minimum over nphase phases (log bisection, all phases at once).
    Diverged: the state norm (velocity-dominated, so comparable with the
    kick) is not finite or above 10 dv after nper periods; a decaying run is
    far below dv by then (the linear decay per tooth is <= 0.994 up to
    0.98 a_lim, 0.994^1200 < 1e-3)."""
    n, L = len(B), len(b)
    e = np.zeros(n)
    e[1] = 1.0
    k0 = (np.arange(nphase) * L) // nphase

    def diverges(dv):
        with np.errstate(over="ignore", invalid="ignore"):
            X = run_model(A, B, K, b, Tp, np.outer(e, np.full(nphase, dv)), k0, nper)
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


# ------------------------------------------ off-axis test in the simulator ----
def deviation_to_pert(rpm, X, N=None, spr=P.STEPS_PER_REV):
    """A model deviation (x, x', q, q', phi_1..phi_N) as a tdsim full-state
    perturbation: the history is the Chebyshev interpolant of the nodes,
    sampled at the simulator's steps over the last tooth period, with its
    derivative as the past velocity."""
    from numpy.polynomial import chebyshev as ch
    N = N or nodes_for(rpm)
    T = 60.0 / (rpm * P.N_TEETH)
    nd = spr // P.N_TEETH
    _, xc = cheb(N)
    coef = ch.chebfit(xc, np.concatenate([[X[0]], X[4:4 + N]]), N)
    u = 1 - 2 * np.arange(nd + 1) / nd                  # theta = -k T / nd  ->  xc = 1 + 2 theta / T
    return dict(x=X[0], v=X[1], q=X[2], qd=X[3], hx=ch.chebval(u, coef),
                hv=ch.chebval(u, ch.chebder(coef)) * 2 / T)


def boundary_directions(cert, n, nrand=6, seed=0):
    """Boundary points of E(P): those maximising x, x', q, q', x(t-T) and
    x - x(t-T) (both signs), and nrand random ones."""
    Tm, Pz = cert
    Wz = np.linalg.inv(Pz)
    pts = []
    for idx in ([0], [1], [2], [3], [n - 1], [0, n - 1]):
        c = np.zeros(n)
        c[idx[0]] = 1.0
        if len(idx) == 2:
            c[idx[1]] = -1.0
        cz = c @ Tm
        z = Wz @ cz / np.sqrt(cz @ Wz @ cz)
        pts += [Tm @ z, -(Tm @ z)]
    rng = np.random.default_rng(seed)
    L = cholesky(Wz, lower=True)
    for _ in range(nrand):
        u = rng.standard_normal(n)
        pts.append(Tm @ L @ (u / np.linalg.norm(u)))
    return pts


def td_offaxis(rpm, a, g, cert, scale=1.0, nphase=4, kick_rev=100, window=300):
    """Apply scale x each boundary point at nphase phases with the C1 protocol
    (face entry, nreg 4, |V| <= 9 V, 300-rev window); return (chattering, runs)."""
    from tdsim import simulate
    from tdboundary import classify_run
    spr = P.STEPS_PER_REV
    nd = spr // P.N_TEETH
    n = cert[0].shape[0]
    bad = runs = 0
    for X in boundary_directions(cert, n):
        pert = deviation_to_pert(rpm, scale * X)
        for p in range(nphase):
            ks = kick_rev * spr + (p * nd) // nphase
            r = simulate(rpm, a, g, vmax=VMAX, store=True, nrev=kick_rev + window, edge=True, nreg=4,
                         kick_step=ks, pert=pert)
            st = (ks // nd + 1) * nd
            bad += not classify_run(r["x"][st:], spr, r["diverged"])["stable"]
            runs += 1
    return bad, runs


# ------------------------------------------------------------ campaigns ----
def _validate(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    a = frac * a_lim(rpm, g)
    A, B, K = model(rpm, a, g)
    _, b, bmax = orbit(rpm, g, a)
    e = np.zeros(len(B))
    e[1] = 1.0
    size, note, cert = certify(A, B, K, VMAX - bmax, e, rpm, full=True)
    Tp = 60.0 / (rpm * P.N_TEETH)
    vmax, vend = proof_check(A, B, K, b, Tp, cert) if cert is not None else (np.nan, np.nan)
    off = td_offaxis(rpm, a, g, cert) if cert is not None else (0, 0)
    return size, note, vmax, vend, model_critical_kick(A, B, K, b, Tp), off[0], off[1]


def run_validate(a):
    """At the C1 basin points: the certified kick, the proof checked on its
    own model (boundary starts), the averaged model's critical kick, the
    simulated (time-domain) critical kick, and boundary points of E(P) in
    18 directions x 4 phases applied as full-state deviations in the
    simulator (off-axis test)."""
    b = np.load(os.path.join(RES, "c1_basin.npz"))
    m = b["vmax"] > 0
    jobs = [(float(n), float(g), float(f), float(b["zeta"])) for n, g, f in zip(b["rpm"][m], b["g"][m], b["frac"][m])]
    with Pool(a.workers) as pool:
        res = pool.map(_validate, jobs, chunksize=1)
    cert, vmax, vend, mk, obad, oruns = (np.array([r[k] for r in res]) for k in (0, 2, 3, 4, 5, 6))
    td_min = b["dv"][m].min(axis=1)
    np.savez_compressed(os.path.join(RES, "c2_validate.npz"), rpm=b["rpm"][m], g=b["g"][m], frac=b["frac"][m],
                        cert=cert, model_kick=mk, td_min=td_min, td_median=np.median(b["dv"][m], axis=1),
                        proof_vmax=vmax, proof_vend=vend, offaxis_bad=obad, offaxis_runs=oruns,
                        delta=DELTA, msg=np.array([r[1] for r in res]))
    for (n, g, f, _), c, k, t, v, ob, orr, (_, msg, *_) in zip(jobs, cert, mk, td_min, vmax, obad, oruns, res):
        print(f"{n:.0f} rpm {f:.2f} a_lim: certified {c*1e3:7.3f}, model {k*1e3:8.2f}, TD min {t*1e3:8.2f} mm/s; "
              f"max V {v:.5f}; off-axis chatter {ob}/{orr}  [{msg}]")
    fin = np.isfinite(mk) & (cert > 0)
    print(f"certificate below the model's critical kick at {np.sum(cert[fin] <= mk[fin])}/{fin.sum()} points; "
          f"proof check max V {np.nanmax(vmax):.5f}")
    both = np.isfinite(mk) & np.isfinite(td_min)
    r = mk[both] / td_min[both]
    print(f"averaged model vs simulation (critical kick): {r.min():.2f}-{r.max():.2f}, median {np.median(r):.2f}")
    ok = np.isfinite(td_min) & (cert > 0)
    print(f"certificate below the simulated critical kick at {np.sum(cert[ok] <= td_min[ok])}/{ok.sum()} points; "
          f"TD / certificate {np.min(td_min[ok] / cert[ok]):.2f}-{np.max(td_min[ok] / cert[ok]):.2f}")
    print(f"off-axis boundary points chattering in the simulator: {obad.sum()}/{oruns.sum()} runs "
          f"(at {np.sum(obad > 0)} of {len(obad)} points)")


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


SURF_FRAC = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def _surface(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    a = frac * a_lim(rpm, g)
    beta, msg = certify_kick(rpm, a, g, settings=FAST)
    return beta, VMAX - orbit(rpm, g, a)[2], msg


def run_surface(a):
    """Certified kick on a grid of speeds and depths (solver defaults)."""
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    rpm, g = d["rpm"], d["g_best"]
    sel = (rpm - rpm[0]) % a.step == 0
    rpm, g = rpm[sel], g[sel]
    jobs = [(float(n), float(gg), f, float(d["zeta"])) for n, gg in zip(rpm, g) for f in SURF_FRAC]
    with Pool(a.workers) as pool:
        res = pool.map(_surface, jobs, chunksize=1)
    shape = (len(rpm), len(SURF_FRAC))
    beta = np.array([r[0] for r in res]).reshape(shape)
    u0 = np.array([r[1] for r in res]).reshape(shape)
    al = np.array([a_lim(n, gg) for n, gg in zip(rpm, g)]) * 1e3
    np.savez_compressed(os.path.join(RES, "c2_surface.npz"), rpm=rpm, g=g, frac=np.array(SURF_FRAC), a_lim=al,
                        cert=beta, u0=u0, msg=np.array([r[2] for r in res]).reshape(shape))
    print(f"surface: {len(rpm)} speeds x {len(SURF_FRAC)} depths, checked {np.sum(beta > 0)}/{beta.size}; "
          f"certified kick at 0.9 a_lim {np.nanmin(beta[:, 4])*1e3:.2f}-{np.nanmax(beta[:, 4])*1e3:.2f} mm/s")


STEP_FRAC = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)


def _step(args):
    rpm, g, frac, zeta = args
    P.set_damping(zeta)
    return certify_step(rpm, frac * a_lim(rpm, g), g)


def run_step(a):
    """Step-entry certificate at the C1 basin speeds (beta >= 1 certifies)."""
    from c1_unsafe_zone import BASIN_RPM
    d = np.load(os.path.join(RES, "feas_g1500.npz"))
    jobs = [(float(n), float(np.interp(n, d["rpm"], d["g_best"])), f, float(d["zeta"]))
            for n in BASIN_RPM for f in STEP_FRAC]
    with Pool(a.workers) as pool:
        res = pool.map(_step, jobs, chunksize=1)
    beta = np.array([r[0] for r in res]).reshape(len(BASIN_RPM), len(STEP_FRAC))
    np.savez_compressed(os.path.join(RES, "c2_step.npz"), rpm=np.array(BASIN_RPM, float),
                        frac=np.array(STEP_FRAC), beta=beta)
    for n, row in zip(BASIN_RPM, beta):
        print(f"{n} rpm: step-entry beta " + " ".join(f"{f:.1f}:{b:.3f}" for f, b in zip(STEP_FRAC, row)))
    print(f"largest step-entry beta {beta.max():.3f} (>= 1 would certify the entry)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "stepmap", "surface", "step"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=int, default=40, help="stepmap/surface: rpm spacing")
    a = ap.parse_args()
    dict(validate=run_validate, stepmap=run_stepmap, surface=run_surface, step=run_step)[a.command](a)


if __name__ == "__main__":
    main()
