"""Independent re-statement of the 1-DOF closed-loop model (no import of src/).

Numbers are re-typed from the paper (Table 1, Table 2, Eq. 3).  Damping is a
function argument everywhere so both 1.34 % and 1.45 % can be run side by side.

Sign convention (derived in README.md, Sec. 1):
  w      workpiece displacement normal to the feed (y), m
  f_w    force acting on the workpiece, N
  phi    Altintas immersion angle, clockwise from +y; down milling, half
         immersion -> tooth engaged for phi in [pi/2, pi]
  h      = fz sin(phi) - (w(t) - w(t-T)) cos(phi)
  f_w    = -sum_j g_j (Kt sin(phi) - Kr cos(phi)) a h
  dynamic part:  f_w,dyn = a H(t) (w(t) - w(t-T)),
                 H(t) = sum_j g_j [Kt sin(phi_j) cos(phi_j) - Kr cos^2(phi_j)]
  time average:  <H> = (N Kt / 4 pi) alpha_yy,  alpha_yy = -1 - (Kr/Kt) pi/2
"""
import numpy as np

# structure (Table 1)
FN = 129.3
WN = 2 * np.pi * FN
KS = 1.34e7
MS = KS / WN**2
ZETAS = {"z134": 0.0134, "z145": 0.0145}

# actuator Eq. (3):  Ga = KA s^2 / (s^2 + A1 s + A0)  [N/V]
KA, A1, A0 = 3.0, 15.834, 2785.6

# cutting (Table 2)
NT = 4
DT = 16e-3
HELIX = np.deg2rad(45.0)
KT = 660e6
KR = 180e6
PHI_ST, PHI_EX = np.pi / 2, np.pi          # down milling, half immersion
ALPHA_YY = -1.0 - (KR / KT) * np.pi / 2    # derived independently (README Sec. 1)


def cs(zeta):
    return 2 * zeta * np.sqrt(KS * MS)


def G(s, zeta):
    return 1.0 / (MS * s**2 + cs(zeta) * s + KS)


def Ga(s, ideal=False):
    if ideal:                       # actuator mode removed: flat 3 N/V
        return KA + 0 * s
    return KA * s**2 / (s**2 + A1 * s + A0)


def Gcl(s, g, zeta, ideal=False):
    """Closed-loop receptance with V = -g * wdot, f_act = Ga V."""
    return 1.0 / (1.0 / G(s, zeta) + g * s * Ga(s, ideal))


def A_closed(g, zeta, ideal=False):
    """State matrix of the cutting-free closed loop.

    full actuator: states [w, wdot, q, qdot], qddot + A1 qdot + A0 q = V,
                   f_act = KA * qddot = KA (V - A1 qdot - A0 q),  V = -g wdot
    ideal actuator: states [w, wdot], f_act = -KA g wdot
    """
    c = cs(zeta)
    if ideal:
        return np.array([[0.0, 1.0], [-KS / MS, -(c + KA * g) / MS]])
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-KS / MS, -(c + KA * g) / MS, -KA * A0 / MS, -KA * A1 / MS],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -g, -A0, -A1],
    ])


def H_avg_straight(phi_a, phi_b):
    """Mean over [phi_a, phi_b] (inside one engagement window) of
    Kt sin cos - Kr cos^2 (straight flute, one tooth)."""
    F = lambda p: -KT * np.cos(2 * p) / 4 - KR * (p / 2 + np.sin(2 * p) / 4)
    return (F(phi_b) - F(phi_a)) / (phi_b - phi_a)


def _prim(p):
    """Antiderivative of Kt sin(p)cos(p) - Kr cos^2(p) in p."""
    return KT * np.sin(p) ** 2 / 2 - KR * (p / 2 + np.sin(2 * p) / 4)


def aH_helical(phi0, a, helix=HELIX):
    """a*H for a helical cutter: sum over teeth of the integral over z in [0,a]
    of window(phi) (Kt sin cos - Kr cos^2), phi(z) = phi0_j - kappa z.

    phi0 : array of tooth-0 bottom angles (rad).  Returns a*H (N/m^2 * m).
    Exact in z (antiderivative), so no axial discretisation error.
    """
    kap = 2 * np.tan(helix) / DT
    phi0 = np.atleast_1d(phi0)
    out = np.zeros_like(phi0, dtype=float)
    for j in range(NT):
        hi = phi0 + j * 2 * np.pi / NT
        lo = hi - kap * a
        # bring the window copies that can overlap [lo, hi]
        pmin = np.floor((lo - PHI_EX) / (2 * np.pi)).min() - 1
        pmax = np.ceil((hi - PHI_ST) / (2 * np.pi)).max() + 1
        for p in np.arange(pmin, pmax + 1):
            ws, we = PHI_ST + 2 * np.pi * p, PHI_EX + 2 * np.pi * p
            l = np.maximum(lo, ws)
            u = np.minimum(hi, we)
            m = u > l
            out[m] += (_prim(u[m] - 2 * np.pi * p) - _prim(l[m] - 2 * np.pi * p)) / kap
    return out
