"""Loss of contact near the tool exit as a second sector nonlinearity in the C2 certificate.

PURPOSE (referee request).  C2 (src/c2_certificate.py, paper Section 5) is a theorem about the
N-node collocation of the averaged (ZOA) milling model, whose regenerative force is linear,
KAPPA a (x - x(t - T)) (KAPPA < 0; c2.model puts it in A[1, 0] and A[1, n-1]).  The validity bound
M4, |x - x(t - T)| <= delta fz on the certified ellipsoid (delta = 0.5), excludes gross loss of
contact, but inside it the chip still vanishes near the exit: on the engaged arc phi in [pi/2, pi]
(half-immersion down milling) h = fz sin(phi) + r |cos(phi)|, r = x - x(t - T), which for r < 0 is
negative where tan(pi - phi) < |r| / fz, i.e. over the last arctan(delta) = 26.6 deg of the arc at
|r| = delta fz.  Here the clipped chip h+ = max(h, 0) enters the certified model as a second
sector-bounded nonlinearity and the certified kick is recomputed (beta2, against beta1 without it).

DERIVATION (checked; paper Eqs. chip, fcut, aHt, aHmean, coll and Proposition 1).
  Force.  f_c = -sum_j int_0^a gamma_j c(phi_j) h_j+ dz with c(phi) = Ktc sin(phi) + Krc |cos(phi)| >= 0
  on the arc.  Writing h+ = h + max(-h, 0),
      f_c = f_s(t) + aH(t) r - Lambda(t, r),
      Lambda(t, r) = sum_j int_0^a gamma_j c(phi_j) max(|cos phi_j| |r| - fz sin phi_j, 0) dz  (r < 0),
      Lambda(t, r) = 0  (r >= 0: h >= fz sin(phi) >= 0 on the arc).
  On the forced orbit r = 0, so the orbit and b(t) are unchanged.  C2 replaces aH(t) by its mean
  KAPPA a; the force deviation becomes KAPPA a r - psi2 with psi2 = Lambda(t, r) kept exact and
  time-periodic, and the collocated model
      xi' = A xi - B psi1(K xi, t) - (e2 / m) psi2(t, chi xi),    chi = c2.validity(n)[0], chi xi = r,
  e2 the velocity row, m = P.M (the lost force acts on the workpiece mass).
  Sector.  psi2 >= 0 and KAPPA a r > 0 for r < 0, psi2 = 0 for r >= 0: psi2 has the sign of KAPPA a r.
  Each slice term is convex in |r| and zero at r = 0, so Lambda(t, r) / |r| is non-decreasing in |r|,
  and on |r| <= delta fz
      0 <= psi2 / (KAPPA a r) <= Lambda(t, delta fz) / (|KAPPA| a delta fz) <= theta(a, delta),
      theta(a, delta) = max_t Lambda(t, delta fz) / (|KAPPA| a delta fz),
  i.e. psi2 (psi2 - theta KAPPA a chi xi) <= 0 wherever |chi xi| <= delta fz, which M4 keeps on E(P)
  (so M4 must stay: theta holds only there).  The engagement comes from the helix,
  phi_j(t, z) = Omega t + 2 pi j / Nt - z tan(eta) / R.  theta is computed exactly in z (closed-form
  slice integrals; a midpoint quadrature of the paper's equations checks them) and as an upper bound
  over the tooth period (2^18 samples plus a Lipschitz pad, |dLambda / d(Omega t)| <= fz delta
  |(Ktc, Krc)| / kz).  It depends on a and delta, not on the speed.  theta_bar(delta) is the same
  share for the lost force averaged over the tooth period (independent of a): the sector of the
  averaged nonlinearity, as the ZOA averages the linear part; the instantaneous lost force exceeds
  it, so beta2 with theta_bar is a reference, not a theorem for the model above.
  LMI.  V = xi' P xi, W = P^-1, Y = Gamma W, S1 = 1/T1, S2 = 1/T2.  On E(P) both sectors hold, so
      V' <= V' - 2 T1 psi1 (psi1 - Gamma xi) - 2 T2 psi2 (psi2 - theta KAPPA a chi xi),
  a quadratic form in (xi, psi1, psi2) with blocks [P A + A' P, -P B + T1 Gamma', -P E + T2 theta
  KAPPA a chi'; ., -2 T1, 0; ., 0, -2 T2], E = e2 / m.  The congruence with diag(W, S1, S2) and a
  change of sign give
      M1' = [[-(A W + W A') - 2 lam W, B S1 - Y', E S2 - theta KAPPA a W chi'],
             [S1 B' - Y,              2 S1,     0                         ],
             [S2 E' - theta KAPPA a chi W, 0,   2 S2                      ]] > 0,
  linear in (W, Y, S1, S2), as stated in the request.  M2, M3, M4 are c2's.  In the scaled
  coordinates x = Tm z, E transforms like B (Tm^-1 E) and chi like the validity row (chi Tm).
  M1 is the leading principal block of M1', so every certificate here is also one of C2:
  beta2 <= beta1.  Dropping the psi1 row and column leaves the circle-criterion LMI of the sector
  [0, theta]; by the KYP lemma it has a solution only if Re[1 + theta G(jw)] > 0 for all w, with
  G(s) = KAPPA a chi (s I - A)^-1 E = L / (1 - L), L(s) = KAPPA a (1 - e^{-sT}) Gcl(s) (transfer from
  -psi2 to KAPPA a r).  theta_circle = 1 / max_w(-Re G) is an a-priori bound: for theta above it no
  certificate exists, whatever the clip (computed on the delay equation and on the collocation).

PROTOCOL.  Damping set first from results/gainmap_main.npz (zeta = 0.0134).  certify2 = c2.certify
with lmi2 / check2: the same scaling, LAM_FRAC margins (0.05, 0.2) sigma, N_PASS = 3 re-centred
solves, TIGHT Clarabel settings, eps = 1e-9 (also S2 >= eps), u0 = VMAX - b_max from c2.orbit, kick
direction e = e2, and the same check: W > 0 (condition < 1e10), S1, S2 > 0, smallest eigenvalue of
M1' at lam = 0 above 1e3 (n + 2) eps_mach max|eig| (the order of M1' replaces c2's n + 1), inclusion
and validity through a Cholesky factor of W with the homogeneous rescaling, and the smaller of the
checked size and the solver's beta.  Tests: theta = 0 reproduces c2.certify_kick (2500 rpm,
0.9 a_lim; 1090 rpm, 0.5 a_lim); the assembled blocks equal c2.lmi's; perturbed candidates are
rejected; a sampled run of the collocated model with the true dead zone and the true lost force
psi2(t, r) from boundary points of a certified ellipsoid.  Campaign: the 48 C2 points of
results/c2_validate.npz (g = 1500), beta2 with theta(a, 0.5) and beta1 = the theta = 0 path (like for
like); at the four speeds at 0.9 a_lim also beta2 with theta_bar(0.5), and beta2 with delta = 1
(theta(a, 1)) against beta1 with delta = 1.  Points at 0.5, 0.9 and 0.98 a_lim and the delta
comparison first; the JSON is rewritten after every point.  Both betas use c2.orbit's b_max; the
paper's rescaling to the rigorous bound (factor (u0 - d) / u0, Appendix B) scales both alike and
leaves beta2 / beta1 unchanged.

RUN (from the repository root, one process):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/contact_sector.py theta
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/contact_sector.py time
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/contact_sector.py test
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/contact_sector.py campaign
Each command updates its own section of verification/paper/contact_sector.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "contact_sector.json")

import params as P                                                     # noqa: E402

ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
P.set_damping(ZETA)                                                    # before any model is built

import c2_certificate as c2                                            # noqa: E402
from c2_certificate import (model, orbit, nodes_for, LMIBlock, solve_sdp, scaling, transform,  # noqa: E402
                            validity, VMAX, KAPPA, TIGHT, LAM_FRAC, N_PASS)
from zoa import a_lim                                                  # noqa: E402

G_DVF = 1500.0
DELTA = 0.5
EPS = 1e-9                                       # c2.lmi's strictness margin (M1 - eps I >= 0, S >= eps)

# ---------------------------------------------------------- cutter geometry ----
R_TOOL = 0.5 * P.D_TOOL
KZ = np.tan(P.HELIX) / R_TOOL                    # helix lag, rad per metre of depth (125 rad/m)
PITCH = 2 * np.pi / P.N_TEETH                    # tooth pitch angle
PHI_ST, PHI_EX = P.immersion_angles()            # engaged arc of half-immersion down milling
assert np.isclose(PHI_ST, np.pi / 2) and np.isclose(PHI_EX, np.pi), "derivation assumes the arc [pi/2, pi]"
N_TGRID = 2 ** 18                                # samples of the tooth period for max_t
N_ZQUAD = 20000                                  # midpoint slices of the quadrature check


def _H(psi, d):
    """int_0^psi (Ktc sin u + Krc cos u)(d cos u - sin u) du, u = pi - phi measured back from the exit:
    lost force of one slice per unit depth and per fz at |r| = d fz (valid for psi <= arctan d)."""
    Kt, Kr = P.K_TC, P.K_RC
    s2 = np.sin(2 * psi) / 4
    return (Kt * d - Kr) * np.sin(psi) ** 2 / 2 - Kt * (psi / 2 - s2) + Kr * d * (psi / 2 + s2)


def _J(psi):
    """int_0^psi (Ktc sin u + Krc cos u) cos u du: |aH| of one slice per unit depth, back from the exit."""
    return P.K_TC * np.sin(psi) ** 2 / 2 + P.K_RC * (psi / 2 + np.sin(2 * psi) / 4)


def _windows(theta_s, a, add):
    """Call add(lo, hi, p) for the slice windows [Omega t + 2 pi j / Nt - kz a, Omega t + 2 pi j / Nt]
    of every tooth and every 2 pi copy p that can meet the arc [pi/2, pi]."""
    for j in range(P.N_TEETH):
        hi = theta_s + j * PITCH
        lo = hi - KZ * a
        for p in range(int(np.floor((np.min(lo) - PHI_EX) / (2 * np.pi))) - 1,
                       int(np.ceil((np.max(hi) - PHI_ST) / (2 * np.pi))) + 2):
            add(lo, hi, p)


def lost_force(theta_s, r, a):
    """psi2 = Lambda(t, r) in N: force lost to contact loss at spindle angle theta_s = Omega t and
    chip deviation r = x - x(t - T) (m; arrays broadcast), all teeth, exact in z."""
    theta_s, r = np.broadcast_arrays(np.asarray(theta_s, float), np.asarray(r, float))
    d = np.maximum(-r, 0.0) / P.FZ
    psi0 = np.arctan(d)
    out = np.zeros(theta_s.shape)

    def add(lo, hi, p):
        l = np.maximum(lo, np.pi - psi0 + 2 * np.pi * p)
        u = np.minimum(hi, np.pi + 2 * np.pi * p)
        m = u > l
        out[m] += _H(np.pi - (l[m] - 2 * np.pi * p), d[m]) - _H(np.pi - (u[m] - 2 * np.pi * p), d[m])

    _windows(theta_s, a, add)
    return P.FZ / KZ * out


def linear_coefficient(theta_s, a):
    """|aH(t)| in N/m (paper Eq. aHt), exact in z."""
    theta_s = np.asarray(theta_s, float)
    out = np.zeros(theta_s.shape)

    def add(lo, hi, p):
        l = np.maximum(lo, PHI_ST + 2 * np.pi * p)
        u = np.minimum(hi, PHI_EX + 2 * np.pi * p)
        m = u > l
        out[m] += _J(np.pi - (l[m] - 2 * np.pi * p)) - _J(np.pi - (u[m] - 2 * np.pi * p))

    _windows(theta_s, a, add)
    return out / KZ


def forces_quadrature(theta_s, r, a, nz=N_ZQUAD):
    """Independent check from the paper's Eqs. (phi), (chip), (fcut): midpoint quadrature in z of the
    linear force -sum int gamma c h dz and the clipped one -sum int gamma c h+ dz.  Their difference
    is Lambda; the linear one at r and at 0 gives aH(t)."""
    theta_s = np.atleast_1d(np.asarray(theta_s, float))
    z = (np.arange(nz) + 0.5) / nz * a
    lin = np.zeros(len(theta_s))
    clip = np.zeros(len(theta_s))
    for j in range(P.N_TEETH):
        ph = np.mod(theta_s[:, None] + 2 * np.pi * j / P.N_TEETH - KZ * z[None, :], 2 * np.pi)
        on = (ph >= PHI_ST) & (ph <= PHI_EX)
        h = P.FZ * np.sin(ph) - r * np.cos(ph)
        c = P.K_TC * np.sin(ph) - P.K_RC * np.cos(ph)
        lin -= np.sum(np.where(on, c * h, 0.0), axis=1) * (a / nz)
        clip -= np.sum(np.where(on, c * np.maximum(h, 0.0), 0.0), axis=1) * (a / nz)
    return lin, clip


def theta(a, delta=DELTA, n_t=N_TGRID, detail=False):
    """Sector width theta(a, delta): an upper bound on max_t Lambda(t, delta fz) / (|KAPPA| a delta fz).
    Lambda is Lipschitz in Omega t with constant fz delta |(Ktc, Krc)| / kz (at most one tooth top and
    one bottom lie in the loss region, which is narrower than the pitch), so the grid maximum plus
    half a grid step times that constant bounds the maximum over the period."""
    th = np.arange(n_t) / n_t * PITCH
    lam = lost_force(th, -delta * P.FZ, a)
    den = abs(KAPPA) * a * delta * P.FZ
    lip = P.FZ * delta * np.hypot(P.K_TC, P.K_RC) / KZ
    up = (lam.max() + lip * (PITCH / n_t) / 2) / den
    if not detail:
        return float(up)
    k = int(np.argmax(lam))
    h = PITCH / n_t
    res = minimize_scalar(lambda t: -lost_force(np.array([t]), -delta * P.FZ, a)[0],
                          bounds=(th[k] - h, th[k] + h), method="bounded", options=dict(xatol=1e-13))
    lin = linear_coefficient(th, a)
    ok = lin > 1e-9 * lin.max()
    return dict(theta=float(up), theta_grid=float(lam.max() / den), theta_refined=float(max(-res.fun, lam.max()) / den),
                lipschitz_pad=float(lip * h / 2 / den), theta_mean=float(lam.mean() / den),
                spindle_angle_of_max_deg=float(np.degrees(th[k])),
                theta_instantaneous=float(np.max(lam[ok] / (lin[ok] * delta * P.FZ))),
                mean_aH_over_KAPPA_a=float(lin.mean() / (abs(KAPPA) * a)),
                aH_range_over_KAPPA_a=[float(lin.min() / (abs(KAPPA) * a)), float(lin.max() / (abs(KAPPA) * a))],
                share_of_period_with_loss=float(np.mean(lam > 0)))


def theta_bar(delta=DELTA):
    """Lost force averaged over the tooth period over |KAPPA| a delta fz (closed form, any a): the
    teeth tile the angle, so the mean is a Nt / (2 pi) fz H(arctan delta) and |KAPPA| a is
    a Nt / (2 pi) J(pi / 2)."""
    return float(_H(np.arctan(delta), delta) / (delta * _J(np.pi / 2)))


# ------------------------------------------- a-priori bound (KYP / circle) ----
def e2m(n):
    """E = e2 / m: the lost force enters the velocity row."""
    E = np.zeros(n)
    E[1] = 1.0 / P.M
    return E


def circle_theta(rpm, a, g, A=None, lam=0.0, fmax=1500.0, df=0.002):
    """theta_circle = 1 / max_w(-Re G(jw - lam)), G = L / (1 - L) (delay equation) and, with A,
    G = KAPPA a chi (s I - A)^-1 e2/m (collocation).  Necessary for M1' (lam the decay margin):
    beyond it the circle-criterion block of M1' has no solution.  Returns (dde, colloc, f_at_max)."""
    from structure import Gcl
    T = 60.0 / (rpm * P.N_TEETH)
    f = np.arange(1, int(round(fmax / df)) + 1) * df
    s = 2j * np.pi * f - lam

    def G_dde(s):
        L = KAPPA * a * (1 - np.exp(-s * T)) * Gcl(s, g)
        return L / (1 - L)

    def refine(G, k):
        lo, hi = max(f[k] - df, df / 10), f[k] + df
        r = minimize_scalar(lambda x: G(np.array([2j * np.pi * x - lam])).real[0], bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-9))
        return min(r.fun, G(s[k:k + 1]).real[0]), (r.x if r.fun < G(s[k:k + 1]).real[0] else f[k])

    gd = G_dde(s).real
    md, fd = refine(G_dde, int(np.argmin(gd)))
    out = [np.inf if md >= 0 else -1.0 / md, None, float(fd)]
    if A is not None:
        n = A.shape[0]
        ev, V = np.linalg.eig(A)
        u = validity(n)[0] @ V
        v = np.linalg.solve(V, e2m(n))

        def G_col(s):
            return np.concatenate([KAPPA * a * ((u * v)[None, :] / (sc[:, None] - ev[None, :])).sum(axis=1)
                                   for sc in np.array_split(s, max(1, len(s) // 20000))])

        k = np.argsort(gd)[:1]                                      # spot check of the eigen-expansion
        direct = KAPPA * a * validity(n)[0] @ np.linalg.solve(s[k][0] * np.eye(n) - A, e2m(n))
        assert abs(G_col(s[k])[0] - direct) <= 1e-6 * abs(direct) + 1e-12
        mc, _ = refine(G_col, int(np.argmin(G_col(s).real)))
        out[1] = np.inf if mc >= 0 else -1.0 / mc
    return tuple(out)


# ------------------------------------------------------------------ LMIs ----
def lmi2(A, B, K, u0, e, Tm, theta, kappa_a, eps=EPS, lam=0.0, settings=TIGHT, valid=None, border=None):
    """c2.lmi with the contact-loss channel: M1 is bordered by the column E S2 - theta kappa_a W chi'
    and the entry 2 S2 (S2 >= eps, M1' - eps I >= 0); M2, M3, M4 are c2's.  kappa_a = KAPPA a.
    Variables in c2's order (W, Y, S, beta), then S2.  Returns beta and (W, Y, S, S2) in the
    coordinates x = Tm z.
    border (default: theta != 0).  At theta = 0 the sector is {psi2 = 0} and the sup over S2 > 0 is
    C2's (S2 -> 0+), so the border is left out and the problem is c2.lmi's; border=True keeps it (a
    degenerate problem: the optimal S2 sits at eps, test 'theta0_bordered')."""
    if valid is None:
        raise ValueError("the contact sector holds only on |x - x(t - T)| <= delta fz: pass valid")
    border = (theta != 0) if border is None else border
    if not border and theta != 0:
        raise ValueError("theta != 0 needs the border")
    n = len(B)
    Tmm = np.diag(Tm) if np.ndim(Tm) == 1 else Tm
    As, Bs, Ks, es = transform(A, B, K, e, Tmm)
    Es = transform(A, e2m(n), K, e, Tmm)[1]                            # E transforms like B
    cs = valid[0] @ Tmm                                                 # chi like the validity row
    I, J = np.triu_indices(n)
    nw = len(I)
    vW, vY, vS, vb, vS2 = np.arange(nw), nw + np.arange(n), nw + n, nw + n + 1, nw + n + 2
    off = I != J
    Io, vo = I[off], vW[off]
    C = np.arange(n)
    nb = n + 2 if border else n + 1
    M1 = LMIBlock(nb)          # [[-(A W + W A') - 2 lam W, B S - Y', E S2 - th k W chi'], [., 2 S, 0], [., 0, 2 S2]] - eps I
    rr = np.concatenate([np.tile(C, nw), np.tile(C, len(Io))])
    cc = np.concatenate([np.repeat(J, n), np.repeat(Io, n)])
    vv = np.concatenate([np.repeat(vW, n), np.repeat(vo, n)])
    uu = np.concatenate([As[:, I].T.ravel(), As[:, J[off]].T.ravel()])    # U = A W
    M1.add(rr, cc, vv, -uu)
    M1.add(cc, rr, vv, -uu)                                             # + U'
    if lam > 0:
        M1.add(I, J, vW, -2 * lam)
    M1.add(C, n, vS, Bs)
    M1.add(C, n, vY, -1.0)
    M1.add(n, n, vS, 2.0)
    if border:
        M1.add(C, n + 1, vS2, Es)                                       # E S2
        if theta != 0:
            tk = theta * kappa_a                                        # - theta kappa_a (W chi')_i
            M1.add(I, n + 1, vW, -tk * cs[J])
            M1.add(J[off], n + 1, vo, -tk * cs[Io])
        M1.add(n + 1, n + 1, vS2, 2.0)
    M1.add(np.arange(nb), np.arange(nb), -1, -eps)
    M2 = LMIBlock(n + 1)                                                # [[W, W K' - Y'], [K W - Y, u0^2]]
    M2.add(I, J, vW, 1.0)
    M2.add(I, n, vW, Ks[J])
    M2.add(J[off], n, vo, Ks[Io])
    M2.add(C, n, vY, -1.0)
    M2.add(n, n, -1, u0 ** 2)
    M3 = LMIBlock(n + 1)                                                # [[1, beta e'], [beta e, W]]
    M3.add(0, 0, -1, 1.0)
    M3.add(0, 1 + C, vb, es)
    M3.add(1 + I, 1 + J, vW, 1.0)
    M4 = LMIBlock(n + 1)                                                # [[W, W chi'], [chi W, h^2]]
    M4.add(I, J, vW, 1.0)
    M4.add(I, n, vW, cs[J])
    M4.add(J[off], n, vo, cs[Io])
    M4.add(n, n, -1, valid[1] ** 2)
    nvar = nw + n + 3 if border else nw + n + 2
    c = np.zeros(nvar)
    c[vb] = 1.0
    nonneg = [(vS, eps), (vb, 0.0)] + ([(vS2, eps)] if border else [])
    x, status = solve_sdp(len(c), c, [M1, M2, M3, M4], nonneg=nonneg, settings=settings)
    W = np.zeros((n, n))
    W[I, J] = x[:nw]
    W[J, I] = x[:nw]
    return x[vb], (W, x[nw:nw + n], x[nw + n], x[vS2] if border else 0.0), status


def decrease_matrix(As, Bs, Es, cs, W, Y, S, S2, theta, kappa_a, border=True):
    """M1' at lam = 0 in the scaled coordinates (c2's M1 without the border)."""
    bs = Bs * S - Y
    if not border:
        M = np.block([[-(As @ W + W @ As.T), bs[:, None]], [bs[None, :], np.array([[2.0 * S]])]])
        return (M + M.T) / 2
    col = Es * S2 - theta * kappa_a * (W @ cs)
    M = np.block([[-(As @ W + W @ As.T), bs[:, None], col[:, None]],
                  [bs[None, :], np.array([[2.0 * S, 0.0]])],
                  [col[None, :], np.array([[0.0, 2.0 * S2]])]])
    return (M + M.T) / 2


def check2(A, B, K, u0, e, Tm, W, Y, S, S2, theta, kappa_a, valid, border=None):
    """c2.check with the extended decrease matrix M1' and S1, S2 > 0 required.  Returns (size,
    message, c): size 0 rejects the candidate; c <= 1 is the homogeneous rescaling of (W, Y, S, S2)
    that repairs an exceeded inclusion or validity bound (M1' is homogeneous in them).  Without the
    border (theta = 0) it is c2.check (tolerance 1e3 (n + 1) eps_mach)."""
    border = (theta != 0) if border is None else border
    if not border and theta != 0:
        raise ValueError("theta != 0 needs the border")
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(Y)) and np.isfinite(S) and np.isfinite(S2)):
        return 0.0, "non-finite candidate", 0.0
    if valid is None:
        raise ValueError("the contact sector needs the validity bound")
    Tmm = np.diag(Tm) if np.ndim(Tm) == 1 else Tm
    As, Bs, Ks, es = transform(A, B, K, e, Tmm)
    n = len(Bs)
    Es = transform(A, e2m(n), K, e, Tmm)[1]
    cs = valid[0] @ Tmm
    W = (W + W.T) / 2
    lw = np.linalg.eigvalsh(W)
    if lw.min() <= 0 or lw.max() / lw.min() > 1e10:
        return 0.0, "W not positive definite or too ill-conditioned", 0.0
    if not (S > 0 and (S2 > 0 or not border)):
        return 0.0, "multiplier not positive", 0.0
    ev = np.linalg.eigvalsh(decrease_matrix(As, Bs, Es, cs, W, Y, S, S2, theta, kappa_a, border))
    tol = 1e3 * len(ev) * np.finfo(float).eps * np.abs(ev).max()
    if ev.min() <= tol:
        return 0.0, f"decrease not strict ({ev.min() / np.abs(ev).max():+.1e})", 0.0
    L = cholesky(W, lower=True)
    z = solve_triangular(L, W @ Ks - Y, lower=True)
    q = z @ z
    c = min(1.0, (1 - 1e-9) * u0 ** 2 / q) if q > 0 else 1.0
    qv = cs @ W @ cs
    if qv > 0:
        c = min(c, (1 - 1e-9) * valid[1] ** 2 / qv)
    y = solve_triangular(L, es, lower=True)
    return (float(np.sqrt(c / (y @ y)) * (1 - 1e-9)), ("ok" if c == 1.0 else f"ok, shrunk by {c:.12g}"), c)


def certify2(A, B, K, u0, e, rpm, theta, kappa_a, settings=TIGHT, delta=DELTA, full=False, border=None):
    """c2.certify with lmi2 / check2: largest checked size along e over the same margins and
    re-centred passes, E(P) inside the validity bound (required).  With full=True also returns the
    certificate dict(Tm, P, S, S2) (V = z' P z, z = Tm^-1 x) and the per-pass record."""
    valid = validity(len(B), delta)
    sigma = -np.linalg.eigvals(A).real.max()
    if sigma <= 0:
        return (0.0, "linearly unstable", None, []) if full else (0.0, "linearly unstable")
    if u0 <= 0:
        return (0.0, "orbit saturates", None, []) if full else (0.0, "orbit saturates")
    best, note, cert, passes = 0.0, "no candidate passed the check", None, []
    for f in LAM_FRAC:
        Tm = np.diag(scaling(A, rpm))
        for p in range(N_PASS):
            t0 = time.time()
            beta, (W, Y, S, S2), status = lmi2(A, B, K, u0, e, Tm, theta, kappa_a, lam=f * sigma,
                                               settings=settings, valid=valid, border=border)
            size, msg, c = check2(A, B, K, u0, e, Tm, W, Y, S, S2, theta, kappa_a, valid, border=border)
            passes.append(dict(lam_frac=f, p=p, status=status, beta_solver=float(beta), size_checked=size,
                               msg=msg, S2=float(S2), seconds=round(time.time() - t0, 2)))
            if size > best:
                best = min(size, beta) if beta > 0 else size
                note = f"{status}; pass {p}, lam {f:g} sigma; {msg}"
                Ws = c * (W + W.T) / 2
                cert = dict(Tm=Tm.copy(), P=np.linalg.inv(Ws), W=Ws, Y=c * Y, S=c * S, S2=c * S2)
            try:
                Tm = Tm @ cholesky((W + W.T) / 2, lower=True)
            except (np.linalg.LinAlgError, ValueError):
                break
        if best > 0:
            break
    return (best, note, cert, passes) if full else (best, note)


def setup(rpm, a, g=G_DVF, N=None):
    A, B, K = model(rpm, a, g, N)
    x0, b, bmax = orbit(rpm, g, a, N)
    e = np.zeros(len(B))
    e[1] = 1.0
    return A, B, K, e, b, bmax


def certify2_kick(rpm, a, theta, g=G_DVF, N=None, settings=TIGHT, delta=DELTA, full=False, border=None):
    A, B, K, e, _, bmax = setup(rpm, a, g, N)
    return certify2(A, B, K, VMAX - bmax, e, rpm, theta, KAPPA * a, settings, delta=delta, full=full, border=border)


# ------------------------------------------------ sampled run of the model ----
def run_model2(A, B, K, b, Tp, a, X0, k0, nper, chi, Q):
    """c2.run_model plus the true lost force psi2(t, chi x) through E (zero-order hold over each of
    the len(b) steps of the tooth period; step k at spindle angle k PITCH / len(b), as the orbit).
    Tracks V = x' Q x and |chi x| on the fly; returns their maxima, the final V and the
    contact-loss statistics."""
    L = len(b)
    Ad, Bd = c2._zoh(A, B, Tp / L)
    _, Ed = c2._zoh(A, e2m(len(B)), Tp / L)
    X = np.array(X0, float)
    idx = np.array(k0, int) % L
    vmax = vmax1 = float(np.max(np.einsum("ik,ij,jk->k", X, Q, X)))
    rmax, ratio, active = float(np.max(np.abs(chi @ X))), 0.0, 0
    for step in range(nper * L):
        w = K @ X
        bb = b[idx]
        psi1 = w - (np.clip(bb + w, -VMAX, VMAX) - bb)
        r = chi @ X
        psi2 = lost_force(idx * PITCH / L, r, a)
        on = psi2 > 0
        active += int(on.sum())
        if np.any(on):
            ratio = max(ratio, float(np.max(psi2[on] / (KAPPA * a * r[on]))))
        X = Ad @ X - np.outer(Bd, psi1) - np.outer(Ed, psi2)
        idx = (idx + 1) % L
        V = np.einsum("ik,ij,jk->k", X, Q, X)
        vmax = max(vmax, float(V.max()))
        if step >= L:
            vmax1 = max(vmax1, float(V.max()))
        rmax = max(rmax, float(np.max(np.abs(chi @ X))))
    return vmax, vmax1, float(V.max()), rmax, ratio, active


def proof_check2(rpm, a, cert, nphase=8, nper=40, delta=DELTA, g=G_DVF):
    """Start on 0.999 of the boundary of the certified ellipsoid (c2.boundary_directions: 12
    axis-maximising points incl. +-(x - x(t-T)), 6 random) at nphase phases and run the collocated
    model with the true dead zone and the true lost force.  The theorem says V <= 1 and
    |x - x(t - T)| <= delta fz along every run (up to the step); a sampled consistency check only."""
    A, B, K, e, b, bmax = setup(rpm, a, g)
    n = len(B)
    Tp = 60.0 / (rpm * P.N_TEETH)
    Tm, Pz = cert["Tm"], cert["P"]
    pts = c2.boundary_directions((Tm, Pz), n)
    X0 = 0.999 * np.array([p for p in pts for _ in range(nphase)]).T
    k0 = np.tile((np.arange(nphase) * len(b)) // nphase, len(pts))
    chi, h = validity(n, delta)
    Ti = np.linalg.inv(Tm)
    vmax, vmax1, vend, rmax, ratio, active = run_model2(A, B, K, b, Tp, a, X0, k0, nper, chi, Ti.T @ Pz @ Ti)
    return dict(runs=int(X0.shape[1]), periods=nper, steps_per_period=len(b), max_V=vmax, max_V_after_first_period=vmax1,
                final_V_max=vend, max_r_over_delta_fz=rmax / h, max_psi2_over_KAPPA_a_r=ratio,
                steps_with_contact_loss=active, steps=int(nper * len(b) * X0.shape[1]))


# ------------------------------------------------------------ bookkeeping ----
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return str(float(o))
    return o


def load():
    try:
        with open(OUT) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(**sections):
    out = load()
    out["description"] = __doc__.split("\n\n")[0]
    out.update(_clean(sections))
    out["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    os.replace(tmp, OUT)


def points():
    d = np.load(os.path.join(RES, "c2_validate.npz"))
    out = []
    for i, (n, f, g, cst) in enumerate(zip(d["rpm"], d["frac"], d["g"], d["cert"])):
        assert float(g) == G_DVF
        al = a_lim(float(n), float(g))
        out.append(dict(i=i, rpm=float(n), frac=float(f), g=float(g), a_lim_mm=al * 1e3, a=float(f) * al,
                        a_mm=float(f) * al * 1e3, beta_c2_stored_mmps=float(cst) * 1e3, N=nodes_for(float(n))))
    return out


# --------------------------------------------------------------- commands ----
def cmd_theta(_):
    t0 = time.time()
    pts = points()
    tab = []
    for a_mm in (1, 2, 2.3, 3, 4, 5, 6, 8, 10, 12, 12.566, 14, 15, 16, 18, 20, 21, 25):
        a = a_mm * 1e-3
        tab.append(dict(a_mm=a_mm, theta_05=theta(a, 0.5), theta_1=theta(a, 1.0)))
    per = []
    for p in pts:
        d05, d1 = theta(p["a"], 0.5, detail=True), theta(p["a"], 1.0, detail=True)
        A, *_ = setup(p["rpm"], p["a"])
        cd, cc, fc = circle_theta(p["rpm"], p["a"], G_DVF, A=A)
        per.append(dict(rpm=p["rpm"], frac=p["frac"], a_mm=p["a_mm"], theta_05=d05["theta"], theta_1=d1["theta"],
                        theta_05_detail=d05, theta_1_detail=d1, theta_circle_dde=cd, theta_circle_colloc=cc,
                        f_circle_hz=fc))
        print(f"{p['rpm']:.0f} rpm {p['frac']:.2f} a_lim ({p['a_mm']:6.2f} mm): theta(0.5) {d05['theta']:.4f} "
              f"(inst {d05['theta_instantaneous']:.4f}), theta(1) {d1['theta']:.4f}; circle bound {cd:.4f} (DDE) "
              f"{cc:.4f} (colloc) at {fc:.1f} Hz")
    # quadrature check of the closed form (Lambda and aH) at a few depths and both deltas
    qc = []
    ths = np.linspace(0, PITCH, 97)[:-1]
    for a_mm in (2.3, 5.0, 10.0, 15.0, 21.0):
        a = a_mm * 1e-3
        for dl in (0.5, 1.0):
            lin, clip = forces_quadrature(ths, -dl * P.FZ, a)
            lin0, _ = forces_quadrature(ths, 0.0, a)
            lq = lin - clip
            lc = lost_force(ths, -dl * P.FZ, a)
            aHq = (lin - lin0) / (-dl * P.FZ)
            aHc = -linear_coefficient(ths, a)
            qc.append(dict(a_mm=a_mm, delta=dl, max_abs_err_lost_rel=float(np.abs(lq - lc).max() / np.abs(lc).max()),
                           max_abs_err_aH_rel=float(np.abs(aHq - aHc).max() / np.abs(aHc).max())))
    import c3_periodic as c3
    th_t = ths / (2 * np.pi * 2500.0 / 60.0)
    c3err = max(float(np.abs(c3.aH(th_t, 2500.0, a_mm * 1e-3) + linear_coefficient(ths, a_mm * 1e-3)).max()
                      / np.abs(linear_coefficient(ths, a_mm * 1e-3)).max()) for a_mm in (2.3, 5.0, 10.0, 15.0, 21.0))
    th05 = [q["theta_05"] for q in per]
    out = dict(theta_bar_05=theta_bar(0.5), theta_bar_1=theta_bar(1.0),
               exit_arc_deg_05=float(np.degrees(np.arctan(0.5))), exit_arc_deg_1=45.0,
               share_of_alpha_yy_in_exit_arc_05=float(_J(np.arctan(0.5)) / _J(np.pi / 2)),
               theta_05_range_48=[min(th05), max(th05)], theta_vs_depth=tab, per_point=per,
               quadrature_check=dict(nz=N_ZQUAD, n_times=len(ths), cases=qc,
                                     closed_form_aH_vs_c3_periodic_aH_rel=c3err),
               n_tgrid=N_TGRID, runtime_s=round(time.time() - t0, 1))
    save(theta=out)
    print(f"theta_bar(0.5) = {out['theta_bar_05']:.4f}, theta_bar(1) = {out['theta_bar_1']:.4f}; theta(a, 0.5) over the "
          f"48 points {min(th05):.4f}-{max(th05):.4f}; quadrature check max rel err "
          f"{max(q['max_abs_err_lost_rel'] for q in qc):.1e} (Lambda), {max(q['max_abs_err_aH_rel'] for q in qc):.1e} "
          f"(aH), aH vs c3_periodic {c3err:.1e}")


def cmd_time(_):
    rpm = 2500.0
    a = 0.9 * a_lim(rpm, G_DVF)
    A, B, K, e, _, bmax = setup(rpm, a)
    n = len(B)
    Tm = np.diag(scaling(A, rpm))
    sigma = -np.linalg.eigvals(A).real.max()
    valid = validity(n)
    t0 = time.time()
    beta1, _, st1 = c2.lmi(A, B, K, VMAX - bmax, e, Tm, lam=0.05 * sigma, valid=valid)
    t1 = time.time()
    th = theta(a)
    beta2, (W, Y, S, S2), st2 = lmi2(A, B, K, VMAX - bmax, e, Tm, th, KAPPA * a, lam=0.05 * sigma, valid=valid)
    t2 = time.time()
    nw = n * (n + 1) // 2
    out = dict(rpm=rpm, frac=0.9, n=n, unknowns_c2=nw + n + 2, unknowns_contact=nw + n + 3,
               c2_lmi_s=round(t1 - t0, 2), lmi2_s=round(t2 - t1, 2), beta_c2_pass0=float(beta1), status_c2=st1,
               beta_contact_pass0=float(beta2), status_contact=st2, theta=th, note="first pass only, lam 0.05 sigma")
    save(timing=out)
    print(json.dumps(out, indent=1))


def _block_structure_test(rpm, frac):
    """The blocks assembled by lmi2 contain c2.lmi's exactly (all captured through a stub solver): at
    theta = 0 without the border they are c2's; with the border (forced) M1 is the leading block of
    M1', M2-M4 are identical and S2 only borders M1; theta > 0 changes only the W chi' entries of the
    border column."""
    a = frac * a_lim(rpm, G_DVF)
    A, B, K, e, _, bmax = setup(rpm, a)
    n = len(B)
    Tm = np.diag(scaling(A, rpm))
    sigma = -np.linalg.eigvals(A).real.max()
    cap = {}
    nvar = n * (n + 1) // 2 + n + 2

    def stub(key):
        def f(nv, c_obj, blocks, nonneg=(), settings=None):
            cap[key] = (nv, np.asarray(c_obj), [blk.assemble(nv) for blk in blocks], [blk.N for blk in blocks], list(nonneg))
            return np.ones(nv), "stub"
        return f

    orig2, orig = c2.solve_sdp, globals()["solve_sdp"]
    try:
        c2.solve_sdp = stub("c2")
        c2.lmi(A, B, K, VMAX - bmax, e, Tm, lam=0.05 * sigma, valid=validity(n))
        globals()["solve_sdp"] = stub("cs0")
        lmi2(A, B, K, VMAX - bmax, e, Tm, 0.0, KAPPA * a, lam=0.05 * sigma, valid=validity(n))
        globals()["solve_sdp"] = stub("cs")
        lmi2(A, B, K, VMAX - bmax, e, Tm, 0.0, KAPPA * a, lam=0.05 * sigma, valid=validity(n), border=True)
        globals()["solve_sdp"] = stub("cs_theta")
        lmi2(A, B, K, VMAX - bmax, e, Tm, theta(a), KAPPA * a, lam=0.05 * sigma, valid=validity(n))
    finally:
        c2.solve_sdp, globals()["solve_sdp"] = orig2, orig
    (nv1, c1, bl1, N1, nn1), (nv2, c2o, bl2, N2, nn2) = cap["c2"], cap["cs"]
    nv0, c0, bl0, N0, nn0 = cap["cs0"]
    same0 = nv0 == nv1 and np.array_equal(c0, c1) and N0 == N1 and nn0 == nn1 and all(
        abs(Ak - Al).max() == 0 and np.array_equal(bk, bl) for (Ak, bk), (Al, bl) in zip(bl1, bl0))
    ok = nv2 == nv1 + 1 and np.array_equal(c2o[:nv1], c1) and c2o[nv1] == 0 and nn2[:2] == nn1
    ok &= N2[0] == N1[0] + 1 and N2[1:] == N1[1:]
    L1 = N1[0] * (N1[0] + 1) // 2
    Aa, ba = bl1[0]
    Ab, bb = bl2[0]
    ok &= (abs(Ab[:L1, :nv1] - Aa).max() == 0) and (abs(Ab[:L1, nv1:]).max() == 0) and np.array_equal(bb[:L1], ba)
    for (Ak, bk), (Al, bl) in zip(bl1[1:], bl2[1:]):
        ok &= (abs(Al[:, :nv1] - Ak).max() == 0) and (abs(Al[:, nv1:]).max() == 0) and np.array_equal(bk, bl)
    # with theta > 0 the only change is the W chi' part of the new column of M1' (the leading block
    # differs only by the order in which the sparse assembly sums duplicate entries: round-off)
    Ac, bc = cap["cs_theta"][2][0]
    diff = (Ac - Ab).tocsr()
    lead = float(abs(diff[:L1]).max() / abs(Ab[:L1]).max())
    tail = diff[L1:].tocoo()
    newcol_ok = bool(np.array_equal(bc, bb) and np.all(tail.col[tail.data != 0] < nvar - 2 - n) and tail.nnz > 0)
    return dict(rpm=rpm, frac=frac, n=n, theta0_problem_bitwise_identical_to_c2=bool(same0),
                theta0_bordered_blocks_contain_c2_bitwise=bool(ok), unknowns=[int(nv1), int(nv2)],
                theta_changes_only_W_entries_of_new_column=newcol_ok, theta_leading_block_rel_roundoff=lead,
                passed=bool(same0 and ok and newcol_ok and lead < 1e-14))


def _perturbation_tests(rpm, frac, cert, th):
    """The extended check on the certified candidate and on deliberately perturbed ones."""
    a = frac * a_lim(rpm, G_DVF)
    A, B, K, e, _, bmax = setup(rpm, a)
    n = len(B)
    u0, valid = VMAX - bmax, validity(n)
    Tm, W, Y, S, S2 = cert["Tm"], cert["W"], cert["Y"], cert["S"], cert["S2"]
    ka = KAPPA * a
    res = {}

    def chk(name, *args, **kw):
        z = dict(A=A, W=W, Y=Y, S=S, S2=S2, theta=th)
        z.update(kw)
        size, msg, _ = check2(z["A"], B, K, u0, e, Tm, z["W"], z["Y"], z["S"], z["S2"], z["theta"], ka, valid)
        res[name] = dict(size_mmps=size * 1e3, msg=msg)
    chk("certified candidate")
    chk("S2 x 1e-3", S2=S2 * 1e-3)
    chk("S2 x 1e3", S2=S2 * 1e3)
    chk("S2 negative", S2=-S2)
    chk("theta x 3", theta=3 * th)
    chk("sign of the contact column flipped (theta -> -theta)", theta=-th)
    A2, _, _ = model(rpm, 1.05 * a_lim(rpm, G_DVF), G_DVF)
    chk("model at 1.05 a_lim", A=A2)
    # the leading block is a C2 certificate: c2.check accepts (W, Y, S)
    size_c2, msg_c2 = c2.check(A, B, K, u0, e, Tm, W, Y, S, valid=valid)
    res["c2.check of (W, Y, S1)"] = dict(size_mmps=size_c2 * 1e3, msg=msg_c2)
    # does the C2 (theta = 0) certificate also certify the contact-loss model for some S2?  Scan S2
    # over 12 decades around the scale that balances the two parts of the new column.
    b1, note1, cert1, _ = certify2(A, B, K, u0, e, rpm, 0.0, ka, full=True)
    T1 = cert1["Tm"]
    Es1 = transform(A, e2m(n), K, e, T1)[1]
    s2c = np.linalg.norm(th * ka * (cert1["W"] @ (valid[0] @ T1))) / np.linalg.norm(Es1)
    found = None
    for s2 in s2c * np.logspace(-6, 6, 241):
        size, msg, _ = check2(A, B, K, u0, e, T1, cert1["W"], cert1["Y"], cert1["S"], s2, th, ka, valid)
        if size > 0:
            found = (s2, size)
            break
    res["theta = 0 certificate checked at theta (S2 scanned over 12 decades)"] = dict(
        passes=found is not None, size_mmps=None if found is None else found[1] * 1e3, beta_theta0_mmps=b1 * 1e3)
    controls = ("certified candidate", "S2 x 1e3", "c2.check of (W, Y, S1)",
                "theta = 0 certificate checked at theta (S2 scanned over 12 decades)")
    rejected = [k for k, v in res.items() if k not in controls and v["size_mmps"] == 0.0]
    return res, rejected


def cmd_test(_):
    t0 = time.time()
    out = dict(blocks=[_block_structure_test(2500.0, 0.9), _block_structure_test(1090.0, 0.5)])
    print("block structure:", out["blocks"])
    rep = []
    for rpm, frac in ((2500.0, 0.9), (1090.0, 0.5)):
        a = frac * a_lim(rpm, G_DVF)
        t1 = time.time()
        b_c2, n_c2 = c2.certify_kick(rpm, a, G_DVF)
        t2 = time.time()
        b_0, n_0, cert0, pas0 = certify2_kick(rpm, a, 0.0, full=True)
        t3 = time.time()
        b_b, n_b, certb, pasb = certify2_kick(rpm, a, 0.0, full=True, border=True)
        t4 = time.time()
        rep.append(dict(rpm=rpm, frac=frac, a_mm=a * 1e3, beta_c2_certify_kick_mmps=b_c2 * 1e3, note_c2=n_c2,
                        beta_theta0_mmps=b_0 * 1e3, note_theta0=n_0, rel_diff=(b_0 - b_c2) / b_c2,
                        passes_theta0=pas0, seconds_c2=round(t2 - t1, 1), seconds_theta0=round(t3 - t2, 1),
                        theta0_bordered=dict(
                            beta_mmps=b_b * 1e3, note=n_b, rel_diff=(b_b - b_c2) / b_c2, passes=pasb,
                            S2_of_kept_candidate=None if certb is None else certb["S2"], seconds=round(t4 - t3, 1),
                            explanation="S2 border kept at theta = 0: the optimal S2 goes to its eps-level bound, so "
                                        "M1' has an eigenvalue of order S2 at the round-off tolerance of the check "
                                        "(passes rejected as 'decrease not strict' with ratios ~1e-14-1e-12) and the "
                                        "solver stops at a different point; the sup over S2 > 0 equals C2's")))
        print(f"{rpm:.0f} rpm {frac} a_lim: c2.certify_kick {b_c2*1e3:.7f} mm/s [{n_c2}], theta = 0 path "
              f"{b_0*1e3:.7f} mm/s [{n_0}], rel diff {(b_0 - b_c2) / b_c2:+.2e}; with the S2 border forced "
              f"{b_b*1e3:.7f} mm/s [{n_b}], rel diff {(b_b - b_c2) / b_c2:+.2e} ({t2-t1:.0f}, {t3-t2:.0f}, {t4-t3:.0f} s)",
              flush=True)
        save(tests=dict(out, theta0_reproduction=rep, status="running"))
    out["theta0_reproduction"] = rep
    # a certificate of the contact-loss model, its perturbations and a sampled run
    pert = []
    for rpm, frac in ((1090.0, 0.5), (2500.0, 0.9)):
        a = frac * a_lim(rpm, G_DVF)
        th = theta(a)
        b2, n2, cert2, pas2 = certify2_kick(rpm, a, th, full=True)
        item = dict(rpm=rpm, frac=frac, theta=th, beta2_mmps=b2 * 1e3, note=n2, passes=pas2)
        if cert2 is not None:
            res, rejected = _perturbation_tests(rpm, frac, cert2, th)
            item.update(perturbations=res, rejected=rejected, sampled_run=proof_check2(rpm, a, cert2))
            print(f"{rpm:.0f} rpm {frac} a_lim theta {th:.4f}: beta2 {b2*1e3:.4f} mm/s; " + "; ".join(
                f"{k}: {v.get('size_mmps')} {v.get('msg', '')}" for k, v in res.items()))
            print("sampled run:", item["sampled_run"])
        else:
            print(f"{rpm:.0f} rpm {frac} a_lim theta {th:.4f}: no certificate [{n2}]")
        pert.append(item)
        save(tests=dict(out, perturbation=pert, status="running"))
    out["perturbation"] = pert
    out["runtime_s"] = round(time.time() - t0, 1)
    out["status"] = "done"
    save(tests=out)


def _one(p, th, delta, label):
    t0 = time.time()
    try:
        beta, note, cert, passes = certify2_kick(p["rpm"], p["a"], th, delta=delta, full=True)
    except Exception as exc:                                            # recorded, not fatal
        beta, note, cert, passes = 0.0, f"error: {exc!r}", None, []
    return dict(label=label, theta=th, delta=delta, beta_mmps=beta * 1e3, note=note, n_solves=len(passes),
                passes=[dict(lam_frac=q["lam_frac"], p=q["p"], status=q["status"], beta_solver_mmps=q["beta_solver"] * 1e3,
                             size_checked_mmps=q["size_checked"] * 1e3, msg=q["msg"], seconds=q["seconds"]) for q in passes],
                S2=None if cert is None else cert["S2"], seconds=round(time.time() - t0, 1))


def summarize(rows):
    done = [r for r in rows if "beta2_mmps" in r and r["beta1_mmps"] > 0]
    if not done:
        return {}
    rat = np.array([r["beta2_mmps"] / r["beta1_mmps"] for r in done])
    k = int(np.argmin(rat))
    cert2 = [r for r in done if r["beta2_mmps"] > 0]
    s = dict(points_done=len(done), certified_with_contact=len(cert2),
             not_certified=[dict(rpm=r["rpm"], frac=r["frac"], theta=r["theta_05"], circle=r["theta_circle_dde"],
                                 note=r["note2"]) for r in done if r["beta2_mmps"] == 0],
             ratio_min=float(rat.min()), ratio_max=float(rat.max()), ratio_median=float(np.median(rat)),
             worst_point=dict(rpm=done[k]["rpm"], frac=done[k]["frac"], a_mm=done[k]["a_mm"], theta=done[k]["theta_05"],
                              beta1_mmps=done[k]["beta1_mmps"], beta2_mmps=done[k]["beta2_mmps"], ratio=float(rat[k])),
             beta2_le_beta1_all=bool(np.all(rat <= 1 + 1e-6)),
             beta1_vs_stored_rel_max=float(max(abs(r["beta1_mmps"] / r["beta_c2_stored_mmps"] - 1) for r in done)))
    if cert2:
        rc = np.array([r["beta2_mmps"] / r["beta1_mmps"] for r in cert2])
        s.update(ratio_certified_min=float(rc.min()), ratio_certified_max=float(rc.max()),
                 ratio_certified_median=float(np.median(rc)),
                 beta2_range_mmps=[float(min(r["beta2_mmps"] for r in cert2)), float(max(r["beta2_mmps"] for r in cert2))])
    by = {}
    for r in done:
        by.setdefault(f"{r['frac']:.2f}", []).append(round(r["beta2_mmps"] / r["beta1_mmps"], 4))
    s["ratio_by_depth"] = by
    return s


def cmd_campaign(a):
    t0 = time.time()
    prev = load().get("campaign", {})
    pts = points()
    first = [p for p in pts if any(np.isclose(p["frac"], f) for f in (0.5, 0.9, 0.98))]
    rest = [p for p in pts if p not in first]
    rows = {(r["rpm"], r["frac"]): r for r in prev.get("points", []) if "beta2_mmps" in r} if a.resume else {}
    trade = {r["rpm"]: r for r in prev.get("tradeoff_090", []) if "beta1_delta1_mmps" in r} if a.resume else {}
    tb = theta_bar(0.5)

    def write(status):
        lst = sorted(rows.values(), key=lambda r: r["i"])
        save(campaign=dict(points=lst, tradeoff_090=[trade[k] for k in sorted(trade)], summary=summarize(lst),
                           theta_bar_05=tb, delta=DELTA, g=G_DVF, zeta=ZETA, status=status,
                           runtime_s=round(time.time() - t0 + prev.get("runtime_s", 0.0) * bool(a.resume), 1),
                           order="0.5/0.9/0.98 a_lim first, then the delta trade-off at 0.9 a_lim, then the other 36"))

    def do_point(p):
        if (p["rpm"], p["frac"]) in rows:
            return
        th = theta(p["a"], DELTA)
        A, *_ = setup(p["rpm"], p["a"])
        cd, cc, fc = circle_theta(p["rpm"], p["a"], G_DVF, A=A)
        sigma = -np.linalg.eigvals(A).real.max()
        cl = circle_theta(p["rpm"], p["a"], G_DVF, lam=LAM_FRAC[0] * sigma)[0]
        r1 = _one(p, 0.0, DELTA, "beta1 (theta = 0)")
        r2 = _one(p, th, DELTA, "beta2 (theta(a, 0.5))")
        row = dict(p, theta_05=th, theta_circle_dde=cd, theta_circle_colloc=cc, theta_circle_dde_lam005=cl,
                   f_circle_hz=fc, u0_V=float(VMAX - orbit(p["rpm"], G_DVF, p["a"])[2]),
                   beta1_mmps=r1["beta_mmps"], note1=r1["note"], beta2_mmps=r2["beta_mmps"], note2=r2["note"],
                   ratio=(r2["beta_mmps"] / r1["beta_mmps"] if r1["beta_mmps"] > 0 else None),
                   beta1_vs_stored_rel=(r1["beta_mmps"] / p["beta_c2_stored_mmps"] - 1), runs=[r1, r2])
        rows[(p["rpm"], p["frac"])] = row
        print(f"[{time.time()-t0:7.0f} s] {p['rpm']:.0f} rpm {p['frac']:.2f} a_lim ({p['a_mm']:.2f} mm): theta {th:.4f} "
              f"(circle {cd:.3f}); beta1 {r1['beta_mmps']:.4f} (stored {p['beta_c2_stored_mmps']:.4f}), beta2 "
              f"{r2['beta_mmps']:.4f} mm/s, ratio {row['ratio']}  [{r2['note']}] ({r1['seconds']:.0f} + {r2['seconds']:.0f} s)",
              flush=True)
        write("running")

    def do_trade(n):
        if n in trade:
            return
        p = next(q for q in pts if q["rpm"] == n and np.isclose(q["frac"], 0.9))
        th1 = theta(p["a"], 1.0)
        A, *_ = setup(p["rpm"], p["a"])
        rb = _one(p, tb, DELTA, "beta2 (theta_bar(0.5), reference)")
        r21 = _one(p, th1, 1.0, "beta2 (delta = 1, theta(a, 1))")
        r11 = _one(p, 0.0, 1.0, "beta1 (delta = 1, theta = 0)")
        base = rows.get((n, 0.9), {})
        sigma = -np.linalg.eigvals(A).real.max()
        trade[n] = dict(rpm=n, frac=0.9, a_mm=p["a_mm"], theta_bar_05=tb, theta_1=th1,
                        theta_circle_dde=circle_theta(p["rpm"], p["a"], G_DVF, A=A)[0],
                        theta_circle_dde_lam005=circle_theta(p["rpm"], p["a"], G_DVF, lam=LAM_FRAC[0] * sigma)[0],
                        beta2_theta_bar_mmps=rb["beta_mmps"], note_theta_bar=rb["note"],
                        beta2_delta1_mmps=r21["beta_mmps"], note2_delta1=r21["note"],
                        beta1_delta1_mmps=r11["beta_mmps"], note1_delta1=r11["note"],
                        beta1_delta05_mmps=base.get("beta1_mmps"), beta2_delta05_mmps=base.get("beta2_mmps"),
                        theta_05=base.get("theta_05"), runs=[rb, r21, r11])
        print(f"[{time.time()-t0:7.0f} s] trade-off {n:.0f} rpm 0.9 a_lim: theta_bar {tb:.4f} -> beta2 {rb['beta_mmps']:.4f}; "
              f"delta 1: theta {th1:.4f} beta2 {r21['beta_mmps']:.4f} vs beta1 {r11['beta_mmps']:.4f} mm/s; delta 0.5: "
              f"beta1 {base.get('beta1_mmps')} beta2 {base.get('beta2_mmps')}", flush=True)
        write("running")

    for p in first:
        do_point(p)
    for n in (1090.0, 1420.0, 2500.0, 2870.0):
        do_trade(n)
    if not a.first_only:
        for p in rest:
            do_point(p)
    write("done" if (len(rows) == len(pts) or a.first_only) else "partial")
    print(json.dumps(load()["campaign"]["summary"], indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["theta", "time", "test", "campaign"])
    ap.add_argument("--resume", action="store_true", help="campaign: keep the points already in the JSON")
    ap.add_argument("--first-only", action="store_true", help="campaign: only 0.5/0.9/0.98 a_lim and the trade-off")
    a = ap.parse_args()
    dict(theta=cmd_theta, time=cmd_time, test=cmd_test, campaign=cmd_campaign)[a.command](a)


if __name__ == "__main__":
    main()
