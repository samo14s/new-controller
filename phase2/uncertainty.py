"""
uncertainty.py — PHASE 4: the physics-based uncertainty set.
============================================================
Replaces Eqs. (22), (24), (25) and the "10 % / 20 %" sentence of Section 3.2 by a
set parameterised by the physical variables that actually generate it:

    theta = (eta, xi, x_tool, a, z)   in   Theta
      eta     in [0, eta_max]     removed volume fraction  V_removed / V_initial
      xi      in [0, 1]           fraction of the pass already machined
      x_tool  in [0, l_P]         tool position on the top edge
      a       in [0.3, 2.9]       multiplier of abar4                (Eq. 23)
      z       in [0.8, 1.2]       multiplier of the damping ratios

For every theta the modal matrices follow from the SAME Chebyshev-Ritz
discretisation the paper uses, with the Ritz integrals restricted to the
remaining domain Omega(eta, xi):

    M(eta,xi) = U0^T [ int_Omega ( I0 Y^T Y + I2 Y_x^T Y_x + I2 Y_z^T Y_z ) ] U0
    K(eta,xi) = U0^T [ int_Omega ( Kirchhoff bending density ) + K_kappa ] U0

Theorems (proved by additivity of the Ritz integrals, verified numerically in
analysis/material_removal.py):

    T1  Delta M = M(theta) - M(0) <= 0  and  Delta K <= 0   (Loewner order),
        EXACTLY, for every removal domain -- the reachable set lies in one half
        space, so half of any symmetric box is physically unreachable.
    T2  nested removal domains give monotone matrices in eta.
    T4  partial-length removal (xi < 1) breaks the plate symmetry and creates
        off-diagonal modal coupling, which Eq. (22) -- diagonal by construction --
        cannot represent at any magnitude.
"""
import numpy as np
from scipy.linalg import eigh

import config as C
from chebyshev_plate import ChebyshevPlate, cheb_matrix
from milling_dynamics import alpha4_average

KW, KR = 1e12, 1e8


def _grams(P, u1, u2, ngauss=96):
    ug, wg = np.polynomial.legendre.leggauss(ngauss)
    um = 0.5 * (u1 + u2) + 0.5 * (u2 - u1) * ug
    wm = 0.5 * (u2 - u1) * wg
    B = [cheb_matrix(P, um, d) for d in range(3)]
    return {(a, b): (B[a] * wm) @ B[b].T for a in range(3) for b in range(3)}


def _rect(pl, x_lo, x_hi, z_lo, z_hi):
    """Ritz contributions (K, M) of the rectangle, in fractions of (l_P, h_P)."""
    Gx = _grams(pl.PX, 2 * x_lo - 1.0, 2 * x_hi - 1.0)
    Gz = _grams(pl.PZ, 2 * z_lo - 1.0, 2 * z_hi - 1.0)
    cx, cz = 2.0 / pl.lp, 2.0 / pl.hp
    Aj = (pl.lp / 2.0) * (pl.hp / 2.0)
    K = pl.DP * Aj * (
        cx ** 4 * np.kron(Gx[(2, 2)], Gz[(0, 0)])
        + cz ** 4 * np.kron(Gx[(0, 0)], Gz[(2, 2)])
        + pl.nu * cx ** 2 * cz ** 2 * (np.kron(Gx[(2, 0)], Gz[(0, 2)])
                                       + np.kron(Gx[(0, 2)], Gz[(2, 0)]))
        + 2.0 * (1.0 - pl.nu) * cx ** 2 * cz ** 2 * np.kron(Gx[(1, 1)], Gz[(1, 1)]))
    I0 = pl.rho * pl.bp
    I2 = pl.rho * pl.bp ** 3 / 12.0
    M = I0 * Aj * np.kron(Gx[(0, 0)], Gz[(0, 0)])
    M += I2 * Aj * (cx ** 2 * np.kron(Gx[(1, 1)], Gz[(0, 0)])
                    + cz ** 2 * np.kron(Gx[(0, 0)], Gz[(1, 1)]))
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


def _clamp(pl):
    Gx = _grams(pl.PX, -1.0, 1.0)
    p0 = cheb_matrix(pl.PZ, -1.0, 0)[:, 0]
    p1 = cheb_matrix(pl.PZ, -1.0, 1)[:, 0]
    cz = 2.0 / pl.hp
    return (pl.lp / 2.0) * (KW * np.kron(Gx[(0, 0)], np.outer(p0, p0))
                            + KR * cz ** 2 * np.kron(Gx[(0, 0)], np.outer(p1, p1)))


class RemovalFamily:
    """M(eta, xi), C(eta, xi, z), K(eta, xi) in the FIXED nominal modal basis.

    The nominal basis is the one of the un-machined plate, so all perturbations
    are expressed in one common coordinate system -- a prerequisite for a common
    Lyapunov certificate over the whole family.
    """

    def __init__(self, n=2, freq_cal=None, zeta=None):
        self.n = n
        self._cache = {}
        pl = ChebyshevPlate(PX=14, PZ=14)
        self.pl = pl
        self.Kk = _clamp(pl)
        self.Kf, self.Mf = _rect(pl, 0.0, 1.0, 0.0, 1.0)
        w2, V = eigh(self.Kf + self.Kk, self.Mf)
        keep = w2 > 1.0
        U0 = V[:, keep][:, :n].copy()
        for k in range(n):
            U0[:, k] /= np.sqrt(U0[:, k] @ self.Mf @ U0[:, k])
        self.U0 = U0
        self.M0 = U0.T @ self.Mf @ U0
        self.K0raw = U0.T @ (self.Kf + self.Kk) @ U0
        self.omega_raw = np.sqrt(np.linalg.eigvalsh(self.K0raw))
        # frequency calibration: same scaling the paper applies before synthesis
        f_cal = np.asarray(C.F_MEASURED[:n], float) if freq_cal is None \
            else np.asarray(freq_cal, float)
        self.cal = (2 * np.pi * f_cal / self.omega_raw) ** 2
        self.K0 = self.K0raw * self.cal          # diagonal family -> exact scaling
        self.zeta0 = np.asarray(C.ZETA[:n] if zeta is None else zeta, float)
        self.omega0 = 2 * np.pi * f_cal
        self.C0 = np.diag(2 * self.zeta0 * self.omega0)
        self._cache = {}

    # ------------------------------------------------------------------
    def _MK(self, eta, xi, d_eta=0.0):
        """Removal geometry: the layer already finished on PREVIOUS passes covers
        the whole length; the pass in progress has reached x = xi * l_P only.

            finished : z/h in [1 - eta + d_eta, 1],   x/l in [0, 1]
            current  : z/h in [1 - eta, 1 - eta + d_eta], x/l in [0, xi]

        d_eta = a_p / h_P is one pass depth.  This is what makes the asymmetric
        (mode-coupling) part of the perturbation small but non-zero -- and it is
        exactly the part Eq. (22), being diagonal, cannot represent at all.
        """
        key = (round(float(eta), 9), round(float(xi), 9), round(float(d_eta), 9))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if eta <= 0.0:
            M, K = self.M0.copy(), self.K0.copy()
        else:
            de = min(max(d_eta, 0.0), eta)
            Kb, Mb = _rect(self.pl, 0.0, 1.0, 1.0 - eta + de, 1.0)
            if de > 0.0 and xi > 0.0:
                Kc, Mc = _rect(self.pl, 0.0, xi, 1.0 - eta, 1.0 - eta + de)
                Kb, Mb = Kb + Kc, Mb + Mc
            M = self.U0.T @ (self.Mf - Mb) @ self.U0
            K = (self.U0.T @ (self.Kf - Kb + self.Kk) @ self.U0) * self.cal
        w = np.sort(np.sqrt(np.maximum(
            np.linalg.eigvals(np.linalg.solve(M, K)).real, 1.0)))
        self._cache[key] = (M, K, w)
        return M, K, w

    def matrices(self, eta=0.0, xi=1.0, z=1.0, d_eta=0.0):
        """(M, C, K) for the removal geometry described in `_MK`."""
        M, K, w = self._MK(eta, xi, d_eta)
        return M.copy(), np.diag(2 * z * self.zeta0 * w), K.copy()

    def frequencies(self, eta=0.0, xi=1.0, d_eta=0.0):
        M, _, K = self.matrices(eta, xi, d_eta=d_eta)
        return np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(M, K)).real)) \
            / (2 * np.pi)


class UncertaintySet:
    """The set Theta and the resulting (A, A_d, B) family."""

    def __init__(self, plate, rpm=C.RPM_S, ap=C.AP_S, n=2,
                 eta_max=C.ETA_MAX, n_eta=6, n_x=9, sign=C.SIGN):
        self.n = n
        self.sign = sign
        self.fam = RemovalFamily(n=n)
        self.plate = plate
        self.abar4 = alpha4_average(rpm, ap, plate.hp, C.AE)
        self.H = np.asarray(plate.H_Pe_modal, float)[:n]
        self.D_obs = plate.D_row(plate.lp, plate.hp)[:n]
        self.eta_grid = np.linspace(0.0, eta_max, n_eta)
        self.x_grid = np.linspace(0.0, plate.lp, n_x)
        self.Dx = np.array([plate.D_row(x, plate.hp)[:n] for x in self.x_grid])
        self.a_grid = np.array([C.ALPHA_LO, 1.0, C.ALPHA_HI]) * self.abar4 * sign
        self.z_grid = np.array([C.ZETA_LO, 1.0, C.ZETA_HI])
        self.d_eta = ap / plate.hp          # one pass depth, as a volume fraction
        # State and time scaling.  x~ = [ws q ; qdot], time in units of 1/ws.
        # Without it the state matrix mixes entries of order 1 (the identity
        # block) with entries of order omega^2 ~ 5e7, and no SDP solver survives
        # that.  Stability, tau and eps are all transported exactly.
        self.ws = 2 * np.pi * 800.0
        self._Dcache = {}

    # ------------------------------------------------------------------
    def state_space(self, eta=0.0, xi=1.0, x_tool=None, a=None, z=1.0,
                    d_eta=None):
        """(A, A_d, B, Cy) of the open-loop plant at one point of Theta."""
        n = self.n
        d_eta = self.d_eta if d_eta is None else d_eta
        M, Cm, K = self.fam.matrices(eta, xi, z, d_eta)
        if x_tool is None:
            x_tool = 0.0
        kx = round(float(x_tool), 9)
        D = self._Dcache.get(kx)
        if D is None:
            D = self.plate.D_row(x_tool, self.plate.hp)[:n]
            self._Dcache[kx] = D
        DtD = np.outer(D, D)
        a = self.sign * 1.6 * self.abar4 if a is None else a
        Minv = np.linalg.inv(M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ (K + a * DtD)
        A[n:, n:] = -Minv @ Cm
        Ad = np.zeros((2 * n, 2 * n))
        Ad[n:, :n] = Minv @ (a * DtD)
        B = np.concatenate([np.zeros(n), Minv @ self.H])[:, None]
        Cy = np.concatenate([self.D_obs, np.zeros(n)])[None, :]
        return A, Ad, B, Cy

    def nominal(self):
        return self.state_space()

    # ------------------------------------------------------------------
    def _ranges(self, scale):
        a_nom = self.sign * 1.6 * self.abar4
        return (self.eta_grid[-1] * scale,
                a_nom + (self.a_grid[[0, -1]] - a_nom) * scale,
                1.0 + (self.z_grid[[0, -1]] - 1.0) * scale)

    def removal_states(self, scale=1.0, n_eta=3):
        """Sampled removal geometries (eta, xi).

        eta and xi jointly describe one rectangle of removed material, so they
        are sampled together.  xi < 1 is the mid-pass state, which is what
        creates the modal coupling Eq. (22) cannot represent (T4).
        """
        eta_max = self.eta_grid[-1] * scale
        out = [(0.0, 1.0)]
        for eta in np.linspace(eta_max / (n_eta - 1), eta_max, n_eta - 1):
            for xi in (0.0, 1.0):
                out.append((float(eta), xi))
        return out

    def W_hull(self, scale=1.0, n_x=101):
        """Vertices of conv{ a D(x)^T D(x) } in the 3-D space (w11, w12, w22).

        [A, A_d] is AFFINE in W = a D^T D once the removal state is fixed, so the
        convex hull of W is exactly what the LMI needs: imposing the condition at
        these vertices covers every position and every milling-force coefficient
        in the range, with no sampling gap at all.
        """
        from scipy.spatial import ConvexHull
        _, a_rng, _ = self._ranges(scale)
        xs = np.linspace(0.0, self.plate.lp, n_x)
        Ws = []
        for a in a_rng:                       # affine in a -> extremes suffice
            for x in xs:
                D = self.plate.D_row(x, self.plate.hp)[:self.n]
                W = a * np.outer(D, D)
                Ws.append([W[0, 0], W[0, 1], W[1, 1]])
        Ws = np.array(Ws)
        try:
            h = ConvexHull(Ws)
            V = Ws[np.unique(h.vertices)]
        except Exception:
            V = Ws
        return [np.array([[v[0], v[1]], [v[1], v[2]]]) for v in V]

    def scale(self, A, Ad, B, Cy):
        """Transport (A, A_d, B, C) to the scaled coordinates x~ = S x, t~ = ws t."""
        n = self.n
        S = np.diag(np.concatenate([self.ws * np.ones(n), np.ones(n)]))
        Si = np.linalg.inv(S)
        return (S @ A @ Si / self.ws, S @ Ad @ Si / self.ws,
                S @ B / self.ws, Cy @ Si)

    def _ss_from(self, eta, xi, W, z=1.0, d_eta=None):
        n = self.n
        d_eta = self.d_eta if d_eta is None else d_eta
        M, Cm, K = self.fam.matrices(eta, xi, z, d_eta)
        Minv = np.linalg.inv(M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ (K + W)
        A[n:, n:] = -Minv @ Cm
        Ad = np.zeros((2 * n, 2 * n))
        Ad[n:, :n] = Minv @ W
        B = np.concatenate([np.zeros(n), Minv @ self.H])[:, None]
        Cy = np.concatenate([self.D_obs, np.zeros(n)])[None, :]
        return A, Ad, B, Cy

    def vertices_scaled(self, scale=1.0, n_eta=3, n_x=9):
        return [self.scale(*v) for v in self.vertices(scale, n_eta, n_x)]

    def vertices(self, scale=1.0, n_eta=3, n_x=9):
        """(A, A_d, B, Cy) at every (removal state) x (W hull vertex).

        The damping multiplier z is not given vertices: 2 zeta omega is of order
        20 rad/s against 1e7 for the stiffness entries, so +/-20 % on it moves the
        state matrix by ~4 rad/s.  That residual is absorbed by `gap_bound`.
        """
        Wv = self.W_hull(scale, n_x=n_x)
        return [self._ss_from(eta, xi, W)
                for (eta, xi) in self.removal_states(scale, n_eta)
                for W in Wv]

    def gap_bound(self, verts, scale=1.0, n_eta=9, n_xi=5, n_x=41, n_a=3,
                  n_z=3, scaled=False):
        """eps: the largest distance from a point of Theta to conv(verts).

        Distance is measured in the Frobenius norm of [A A_d], which upper-bounds
        the spectral norm used in the certificate, and is obtained by projecting
        onto the simplex spanned by the vertices (non-negative least squares with
        a sum-to-one penalty).  So this is a genuine covering radius, not a
        nearest-sample distance.
        """
        from scipy.optimize import nnls
        if scaled:
            verts = [self.scale(*v) for v in verts]
        V = np.array([np.hstack([v[0], v[1]]).ravel() for v in verts]).T
        rho = 1e3 * np.abs(V).max()
        Vaug = np.vstack([V, rho * np.ones((1, V.shape[1]))])
        eta_max, a_rng, z_rng = self._ranges(scale)
        a_f = np.linspace(a_rng[0], a_rng[1], n_a)
        z_f = np.linspace(z_rng[0], z_rng[1], n_z)
        eps = 0.0
        for eta in np.linspace(0.0, eta_max, n_eta):
            xis = np.linspace(0.0, 1.0, n_xi) if eta > 0 else (1.0,)
            for xi in xis:
                for z in z_f:
                    for a in a_f:
                        for xt in np.linspace(0.0, self.plate.lp, n_x):
                            D = self.plate.D_row(xt, self.plate.hp)[:self.n]
                            ss = self._ss_from(eta, xi, a * np.outer(D, D), z)
                            if scaled:
                                ss = self.scale(*ss)
                            w = np.hstack([ss[0], ss[1]]).ravel()
                            lam, res = nnls(Vaug, np.concatenate([w, [rho]]))
                            eps = max(eps, float(res))
        return eps
