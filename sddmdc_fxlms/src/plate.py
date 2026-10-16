"""
plate.py — the TRUTH structure of the virtual testbed.
=======================================================
Cantilever orthotropic Kirchhoff plate with surface-bonded piezoelectric patches,
discretised by Rayleigh-Ritz.  It stands in for the physical specimen of

    Q. Ouyang, L. Dong, J. Liu, J. Zhou, "From data-driven DMDc modeling to
    control parameter optimization: A framework for adaptive vibration control
    of thin-walled composites", Thin-Walled Struct. 228 (2026) 115090,

Section 3.1, Table 2 and Fig. 4(a).  The paper never writes a structural model
(it is data-driven by design); this file is therefore NOT a reproduction of an
equation of the paper but the "physical plate" that the sd-DMDc of Section 2.2
will be asked to identify.  Fidelity is deliberately HIGHER than what the
identified model can represent (many modes, patch mass/stiffness, finite patch
areas), so that identification error exists and is measurable -- the
"inverse crime" of generating data with the identification model is avoided.

Geometry (Fig. 4a): x along the 480 mm side, y along the 240 mm side, clamped
edge x = 0, origin at the bottom-left corner of the plate.  All transducer
coordinates are the LOWER-LEFT corner of the patch (red dots of Fig. 4a), long
side of every MFC along x (measured on Fig. 4a: 38 mm <-> 78 px at 2.05 px/mm).

Ritz basis:  w(x, y, t) = sum_mn q_mn(t) X_m(xi) Y_n(eta),
    xi = 2x/a - 1, eta = 2y/b - 1,
    X_m = (1 + xi)^2 P_m(xi)  (clamped at xi = -1 exactly: w = w_x = 0),
    Y_n = P_n(eta)            (free-free),
with Legendre polynomials P.  Plate and patch properties are piecewise constant
on rectangles, so every matrix is a sum of Kronecker products of 1-D Gram
matrices -- assembly is exact up to Gauss quadrature and fast.

Strain energy  U = 1/2 int (D11 w_xx^2 + 2 D12 w_xx w_yy + D22 w_yy^2
                            + 4 D66 w_xy^2) dA
Patches add (i) mass rho_p t_p, (ii) bending stiffness of the composite
section about its shifted neutral axis, (iii) piezo coupling:
  * actuator: induced moment per unit length M_x = Q_p Lx t_p z_p,
    M_y = Q_p Ly t_p z_p (z_p = patch mid-plane height above the neutral axis),
    generalised force f_i = -int_patch (M_x phi_i,xx + M_y phi_i,yy) dA;
  * strain sensor: mean longitudinal surface strain over the active area,
    eps_x = -(z_s / A_s) int_s w_xx dA  (the paper's 1-D epsilon_xx, Sec. 2.1).
"""
import numpy as np
from numpy.polynomial import legendre as _L
from scipy.linalg import eigh

# --------------------------------------------------------------------------
# Table 2 of the paper (SI units).  Values flagged UNSTATED are not in the
# paper and are documented assumptions of the virtual testbed.
# --------------------------------------------------------------------------
PLATE = dict(a=0.480, b=0.240, h=0.003, rho=1560.0,
             E1=60.35e9, E2=60.35e9, G12=7.17e9, nu12=0.28)

# PZT-5H exciter.  Table 2 lists G12 = 63 GPa (= E): physically impossible for
# an isotropic ceramic (E/(2(1+nu)) = 24.0 GPa); we use the isotropic value.
PZT = dict(lx=0.040, ly=0.040, t=0.001, rho=7500.0, E=63e9, nu=0.31,
           d33=6.5e-10, d31=-3.5e-10)

# MFC M-2814-P1 actuator (d33 mode).  Overall 38 x 20 x 0.18 mm (Table 2).
# Active area 28 x 14 mm and IDE pitch 0.5 mm are datasheet-type values
# (UNSTATED in the paper).
MFC_P1 = dict(lx=0.038, ly=0.020, t=0.18e-3, rho=5440.0, E1=30.34e9,
              E2=30.34e9, G12=5.51e9, nu=0.303, d33=3.7e-10, d31=-1.7e-10,
              active_lx=0.028, active_ly=0.014, ide_pitch=0.5e-3)

# MFC M-2814-P2 sensor (d31 mode).  Overall 37 x 18 x 0.18 mm.
MFC_P2 = dict(lx=0.037, ly=0.018, t=0.18e-3, rho=5440.0, E1=30.34e9,
              E2=30.34e9, G12=5.51e9, nu=0.303, d33=3.7e-10, d31=-1.7e-10,
              active_lx=0.028, active_ly=0.014)

# Section 3.1: lower-left corners [m].  The text puts the PZT at (20, 140) mm
# while Fig. 4(a) draws it at (21.5, 70.7) mm and the photo of Fig. 3(a) also
# shows it in the lower half (y < 120 mm); all other patches match the text to
# 3 mm.  Two sources against one: the default is y = 70 mm, the text value is
# kept as a variant.  E only enters the primary path G_p.
POS = dict(E=(0.020, 0.070), A1=(0.010, 0.020), A2=(0.010, 0.200),
           S1=(0.130, 0.010), S2=(0.220, 0.210))
POS_E_TEXT = (0.020, 0.140)
POS_E_FIGURE = (0.020, 0.070)

# Fig. 4(a): 7 x 5 laser grid, x = 120..480 mm step 60, y = 0..240 step 60.
# Point 1 = (120, 0), point 7 = (480, 0), point 29 = (120, 240), 35 = (480, 240).
GRID_X = np.arange(0.120, 0.4801, 0.060)
GRID_Y = np.arange(0.000, 0.2401, 0.060)
GRID = np.array([(x, y) for y in GRID_Y for x in GRID_X])   # 35 x 2, row-major

# Table 4: experimental and FEM natural frequencies [Hz]
F_EXP = np.array([14.02, 30.49, 86.96, 117.07, 242.61, 276.07])
F_FEM = np.array([14.52, 31.63, 90.81, 119.16, 235.25, 285.97])
F_SDDMDC = np.array([14.05, 30.61, 87.25, 117.95, 243.85, 277.45])


# --------------------------------------------------------------------------
# 1-D bases
# --------------------------------------------------------------------------
def _leg_coeffs(m):
    c = np.zeros(m + 1)
    c[m] = 1.0
    return c


def basis_x(M, xi, deriv=0):
    """(M, len(xi)): d^k/dxi^k [ (1 + xi)^2 P_m(xi) ], m = 0..M-1."""
    xi = np.atleast_1d(np.asarray(xi, float))
    out = np.zeros((M, xi.size))
    clamp = np.array([1.0, 2.0, 1.0])               # (1 + xi)^2, power basis
    for m in range(M):
        pc = _L.leg2poly(_leg_coeffs(m))
        poly = np.polynomial.polynomial.polymul(clamp, pc)
        if deriv:
            poly = np.polynomial.polynomial.polyder(poly, deriv)
        out[m] = np.polynomial.polynomial.polyval(xi, poly)
    return out


def basis_y(N, eta, deriv=0):
    """(N, len(eta)): d^k/deta^k P_n(eta), n = 0..N-1."""
    eta = np.atleast_1d(np.asarray(eta, float))
    out = np.zeros((N, eta.size))
    for n in range(N):
        c = _leg_coeffs(n)
        if deriv:
            c = _L.legder(c, deriv)
        out[n] = _L.legval(eta, c)
    return out


def _grams(fun, P, u1=-1.0, u2=1.0, ng=48):
    """Gram matrices G[(i, j)] = int_{u1}^{u2} B^(i) B^(j) du and moment
    vectors v[i] = int_{u1}^{u2} B^(i) du on a sub-interval of [-1, 1]."""
    ug, wg = _L.leggauss(ng)
    um = 0.5 * (u1 + u2) + 0.5 * (u2 - u1) * ug
    wm = 0.5 * (u2 - u1) * wg
    B = [fun(P, um, d) for d in range(3)]
    G = {(i, j): (B[i] * wm) @ B[j].T for i in range(3) for j in range(3)}
    v = [B[i] @ wm for i in range(3)]
    return G, v


# --------------------------------------------------------------------------
# Section stiffness helpers
# --------------------------------------------------------------------------
def _Q_ortho(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1
    den = 1.0 - nu12 * nu21
    return dict(Q11=E1 / den, Q22=E2 / den, Q12=nu12 * E2 / den, Q66=G12)


def _composite_section(h, Qb, t, Qp):
    """Neutral axis z_n (from plate mid-plane) and bending stiffnesses of the
    plate + one-sided patch, using the x-direction moduli for z_n."""
    zb, zp = 0.0, h / 2 + t / 2
    zn = (Qb['Q11'] * h * zb + Qp['Q11'] * t * zp) / (Qb['Q11'] * h + Qp['Q11'] * t)
    Ib = h**3 / 12 + h * (zb - zn)**2
    Ip = t**3 / 12 + t * (zp - zn)**2
    D = {k: Qb[q] * Ib + Qp[q] * Ip
         for k, q in (('D11', 'Q11'), ('D22', 'Q22'), ('D12', 'Q12'), ('D66', 'Q66'))}
    return zn, D


class OrthoPlate:
    """Cantilever orthotropic plate + patches.

    Attributes after build: K, M (Ritz), omega [rad/s], freq [Hz], Phi
    (mass-normalised Ritz eigenvectors, n_dof x n_modes), and for each patch
    name the coupling row/column registered with add_actuator / add_sensor.
    """

    def __init__(self, M=14, N=10, plate=None, ng=56):
        self.p = dict(PLATE if plate is None else plate)
        self.Mx, self.Ny, self.ng = M, N, ng
        p = self.p
        self.Qb = _Q_ortho(p['E1'], p['E2'], p['G12'], p['nu12'])
        h = p['h']
        self.Db = {k: self.Qb[q] * h**3 / 12 for k, q in
                   (('D11', 'Q11'), ('D22', 'Q22'), ('D12', 'Q12'), ('D66', 'Q66'))}
        self.cx, self.cy = 2.0 / p['a'], 2.0 / p['b']
        self.Aj = p['a'] * p['b'] / 4.0
        Gx, _ = _grams(basis_x, M, ng=ng)
        Gy, _ = _grams(basis_y, N, ng=ng)
        self.K = self._stiff(self.Db, Gx, Gy)
        self.M = p['rho'] * h * self.Aj * np.kron(Gx[(0, 0)], Gy[(0, 0)])
        self.act, self.sen = {}, {}
        self.patches = []

    # -- assembly ----------------------------------------------------------
    def _stiff(self, D, Gx, Gy):
        cx, cy, Aj = self.cx, self.cy, self.Aj
        return Aj * (D['D11'] * cx**4 * np.kron(Gx[(2, 2)], Gy[(0, 0)])
                     + D['D22'] * cy**4 * np.kron(Gx[(0, 0)], Gy[(2, 2)])
                     + D['D12'] * cx**2 * cy**2 * (np.kron(Gx[(2, 0)], Gy[(0, 2)])
                                                   + np.kron(Gx[(0, 2)], Gy[(2, 0)]))
                     + 4 * D['D66'] * cx**2 * cy**2 * np.kron(Gx[(1, 1)], Gy[(1, 1)]))

    def _sub(self, x0, y0, lx, ly):
        a, b = self.p['a'], self.p['b']
        xi1, xi2 = 2 * x0 / a - 1, 2 * (x0 + lx) / a - 1
        et1, et2 = 2 * y0 / b - 1, 2 * (y0 + ly) / b - 1
        if not (-1 - 1e-12 <= xi1 < xi2 <= 1 + 1e-12 and -1 - 1e-12 <= et1 < et2 <= 1 + 1e-12):
            raise ValueError("patch outside the plate")
        Gx, vx = _grams(basis_x, self.Mx, xi1, xi2, ng=24)
        Gy, vy = _grams(basis_y, self.Ny, et1, et2, ng=24)
        return Gx, vx, Gy, vy

    def add_patch(self, name, x0, y0, prop, structural=True, iso=False):
        """Mass + stiffness of a bonded patch on the top face (z > 0)."""
        lx, ly, t = prop['lx'], prop['ly'], prop['t']
        if iso:
            G = prop['E'] / (2 * (1 + prop['nu']))
            Qp = _Q_ortho(prop['E'], prop['E'], G, prop['nu'])
        else:
            Qp = _Q_ortho(prop['E1'], prop['E2'], prop['G12'], prop['nu'])
        zn, Dc = _composite_section(self.p['h'], self.Qb, t, Qp)
        Gx, vx, Gy, vy = self._sub(x0, y0, lx, ly)
        if structural:
            dD = {k: Dc[k] - self.Db[k] for k in Dc}
            self.K = self.K + self._stiff(dD, Gx, Gy)
            self.M = self.M + prop['rho'] * t * self.Aj * np.kron(Gx[(0, 0)], Gy[(0, 0)])
        self.patches.append(dict(name=name, x0=x0, y0=y0, prop=prop, Qp=Qp,
                                 zn=zn, iso=iso))
        return Qp, zn

    def add_actuator(self, name, x0, y0, prop, kind, structural=True):
        """kind = 'pzt' (d31, isotropic in-plane, through-thickness field) or
        'mfc_p1' (d33 interdigitated, induced strain along x = fibre)."""
        iso = kind == 'pzt'
        Qp, zn = self.add_patch(name, x0, y0, prop, structural, iso=iso)
        t = prop['t']
        zp = self.p['h'] / 2 + t / 2 - zn
        if kind == 'pzt':
            Lx = Ly = prop['d31'] / t                        # strain per volt
            ax0, ay0, alx, aly = x0, y0, prop['lx'], prop['ly']
        elif kind == 'mfc_p1':
            Lx = prop['d33'] / prop['ide_pitch']
            Ly = prop['d31'] / prop['ide_pitch']
            # active area centred in the package
            ax0 = x0 + 0.5 * (prop['lx'] - prop['active_lx'])
            ay0 = y0 + 0.5 * (prop['ly'] - prop['active_ly'])
            alx, aly = prop['active_lx'], prop['active_ly']
        else:
            raise ValueError(kind)
        # induced stress resultants of the free strain, moment arm zp
        sx = Qp['Q11'] * Lx + Qp['Q12'] * Ly
        sy = Qp['Q12'] * Lx + Qp['Q22'] * Ly
        Mx, My = sx * t * zp, sy * t * zp                    # N m / m per volt
        Gx, vx, Gy, vy = self._sub(ax0, ay0, alx, aly)
        cx, cy, Aj = self.cx, self.cy, self.Aj
        f = -Aj * (Mx * cx**2 * np.kron(vx[2], vy[0]) + My * cy**2 * np.kron(vx[0], vy[2]))
        self.act[name] = dict(f=f, Mx=Mx, My=My, kind=kind)
        return f

    def add_sensor(self, name, x0, y0, prop, structural=True):
        """MFC strain sensor: row giving the mean eps_x over the active area at
        the sensor mid-plane."""
        Qp, zn = self.add_patch(name, x0, y0, prop, structural, iso=False)
        t = prop['t']
        zs = self.p['h'] / 2 + t / 2 - zn
        ax0 = x0 + 0.5 * (prop['lx'] - prop['active_lx'])
        ay0 = y0 + 0.5 * (prop['ly'] - prop['active_ly'])
        alx, aly = prop['active_lx'], prop['active_ly']
        Gx, vx, Gy, vy = self._sub(ax0, ay0, alx, aly)
        row = -zs * self.cx**2 * self.Aj * np.kron(vx[2], vy[0]) / (alx * aly)
        self.sen[name] = dict(row=row, zs=zs)
        return row

    # -- outputs -------------------------------------------------------------
    def w_rows(self, pts):
        """(n_pts, n_dof): transverse displacement w at points (x, y)."""
        pts = np.atleast_2d(pts)
        xi = 2 * pts[:, 0] / self.p['a'] - 1
        et = 2 * pts[:, 1] / self.p['b'] - 1
        X = basis_x(self.Mx, xi)
        Y = basis_y(self.Ny, et)
        return np.einsum('mp,np->pmn', X, Y).reshape(len(pts), -1)

    # -- modal analysis ----------------------------------------------------------
    def modes(self, n_modes=40):
        K = 0.5 * (self.K + self.K.T)
        M = 0.5 * (self.M + self.M.T)
        w2, V = eigh(K, M)
        w2, V = w2[:n_modes], V[:, :n_modes]
        V = V / np.sqrt(np.einsum('ij,ij->j', V, M @ V))
        # deterministic sign: positive displacement at the free corner (a, b/2)
        tip = self.w_rows([(self.p['a'], 0.37 * self.p['b'])])[0]
        V = V * np.where(tip @ V < 0, -1.0, 1.0)
        self.omega = np.sqrt(np.maximum(w2, 0.0))
        self.freq = self.omega / (2 * np.pi)
        self.Phi = V
        return self.freq


def build_paper_plate(M=14, N=10, n_modes=40, plate=None, pos=None,
                      structural=True):
    """The plate of Table 2 with the five transducers of Section 3.1."""
    pos = dict(POS if pos is None else pos)
    P = OrthoPlate(M, N, plate=plate)
    P.add_actuator('E', *pos['E'], PZT, 'pzt', structural)
    P.add_actuator('A1', *pos['A1'], MFC_P1, 'mfc_p1', structural)
    P.add_actuator('A2', *pos['A2'], MFC_P1, 'mfc_p1', structural)
    P.add_sensor('S1', *pos['S1'], MFC_P2, structural)
    P.add_sensor('S2', *pos['S2'], MFC_P2, structural)
    P.modes(n_modes)
    return P


def mode_type(P, k, npts=21):
    """'B' (bending) or 'T' (torsion) from the symmetry of mode k about y = b/2."""
    xs = np.full(npts, P.p['a'])
    ys = np.linspace(0, P.p['b'], npts)
    w = P.w_rows(np.c_[xs, ys]) @ P.Phi[:, k]
    sym = np.linalg.norm(w + w[::-1])
    asym = np.linalg.norm(w - w[::-1])
    return 'B' if sym >= asym else 'T'
