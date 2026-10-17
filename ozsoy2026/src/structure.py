"""Plant, actuator and DVF loop of Sec. 2.2-2.3 (Eqs. 1, 3, 4).

Convention: x is the workpiece displacement at the cutting point along the
flexible direction, f the force on the workpiece.  The controller commands a
voltage V = -g * xdot; the actuator turns it into force through Ga(s) (N/V).
"""
import numpy as np
import params as P
from params import GA_GAIN, GA_A1, GA_A0


def G(s):
    """Receptance of the preloaded structure, m/N."""
    return 1.0 / (P.M * s**2 + P.C * s + P.K)


def Ga(s):
    """Actuator force per volt, Eq. (3)."""
    return GA_GAIN * s**2 / (s**2 + GA_A1 * s + GA_A0)


def Gcl(s, g):
    """Closed-loop receptance under DVF: G / (1 + g s Ga G), Eq. (4)."""
    Gs = G(s)
    return Gs / (1.0 + g * s * Ga(s) * Gs)


def closed_loop_poles(g):
    """Roots of the characteristic polynomial of Eq. (4).

    (M s^2 + C s + K)(s^2 + a1 s + a0) + g * GA_GAIN * s^3 = 0
    """
    p_struct = np.array([P.M, P.C, P.K])
    p_act = np.array([1.0, GA_A1, GA_A0])
    poly = np.polymul(p_struct, p_act)
    poly = poly + np.array([0.0, g * GA_GAIN, 0.0, 0.0, 0.0])
    return np.roots(poly)


def state_space(g):
    """Continuous closed-loop model, states z = [x, xdot, q, qdot].

    The actuator is realised as Ga = GA_GAIN * (1 - (a1 s + a0)/(s^2 + a1 s + a0)),
    so with qddot + a1 qdot + a0 q = V the actuator force is
    f_act = GA_GAIN * (V - a1 qdot - a0 q) and V = -g xdot.
    Input: cutting force on the workpiece.  Returns (A, B).
    """
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    # M xddot = -K x - C xdot + f_cut + GA_GAIN*(-g xdot - a1 qdot - a0 q)
    A[1, 0] = -P.K / P.M
    A[1, 1] = -(P.C + GA_GAIN * g) / P.M
    A[1, 2] = -GA_GAIN * GA_A0 / P.M
    A[1, 3] = -GA_GAIN * GA_A1 / P.M
    A[2, 3] = 1.0
    # qddot = V - a1 qdot - a0 q, V = -g xdot
    A[3, 1] = -g
    A[3, 2] = -GA_A0
    A[3, 3] = -GA_A1
    B = np.array([0.0, 1.0 / P.M, 0.0, 0.0])
    return A, B
