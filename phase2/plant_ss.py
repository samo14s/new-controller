"""
plant_ss.py — STAGE 2: the one common controlled-plant state space.
====================================================================
Every controller in this study reads the plant from here and nowhere else.  The
form is the one the research plan fixes,

    xdot(t) = A(x_P, t) x(t) + A_d(x_P, t) x(t - tau) + B u(t) + E F(t)
    y(t)    = C_y x(t)

obtained from Eq. (13) of Du, Liu, Dai & Long, IJMS 274 (2024) 109257,

    M q.. + C q. + (K + a4(t) D(x_P)^T D(x_P)) q(t)
          - a4(t) D(x_P)^T D(x_P) q(t - tau) = f_tau a3(t) D(x_P)^T + H_Pe u(t)

with x = [q ; qdot] and

    A   = [ 0                          I        ]     A_d = [ 0            0 ]
          [ -M^-1 (K + a4 D^T D)   -M^-1 C      ]           [ M^-1 a4 D^T D 0 ]

    B   = [ 0 ; M^-1 H_Pe ]      E = [ 0 ; M^-1 D(x_P)^T ]      C_y = [ D_obs 0 ]

Three things are kept EXPLICIT rather than absorbed into an uncertainty block,
because the proposed controller uses them:

  x_P    the milling position on the top edge.  It enters only through D(x_P),
         and it is KNOWN -- the machine commands the feed, so the tool position
         is a measured signal, not an unknown parameter.
  eta    the removed volume fraction, which sets M(eta), C(eta), K(eta).  Also
         known: it follows from the pass schedule.
  a4(t)  the milling-force coefficient.  This one is genuinely uncertain, and it
         is what stays in the uncertainty set.

Nothing here is controller-specific.  A controller that wants a reduced or
frozen version of this plant must derive it from these matrices.
"""
import numpy as np

import config as C
from milling_dynamics import alpha4_average, alpha4_series, alpha34, N_TEETH
from plate_model import plant_vectors
from uncertainty import RemovalFamily


class ControlledPlant:
    """The controlled milling plant, Stage 1 + Stage 2 of the plan."""

    def __init__(self, plate, rpm=None, ap=None, ae=None, fz=None,
                 n=None, sign=None, removal=None):
        self.plate = plate
        self.rpm = C.RPM_S if rpm is None else rpm
        self.ap = C.AP_S if ap is None else ap
        self.ae = C.AE if ae is None else ae
        self.fz = C.FZ if fz is None else fz
        self.n = C.N_MODES_DESIGN if n is None else int(n)
        self.sign = C.SIGN if sign is None else float(sign)
        self.tau = 60.0 / (N_TEETH * self.rpm)
        self.Omega = 2 * np.pi * self.rpm / 60.0
        w, z, H, D_obs, sign_loop = plant_vectors(plate, self.n)
        self.omega0, self.zeta0 = w, z
        self.H = H
        self.D_obs = D_obs
        self.sign_loop = sign_loop
        self.Cy = np.concatenate([D_obs, np.zeros(self.n)])[None, :]
        self.abar4 = self.sign * alpha4_average(self.rpm, self.ap, plate.hp,
                                                self.ae)
        self.a40 = 1.6 * self.abar4                      # Eq. (23)
        self.L_a = 1.3 * self.abar4
        self.removal = RemovalFamily(n=self.n) if removal is None else removal
        self._D = {}

    # ------------------------------------------------------------------
    def D(self, x_pos):
        """Mode-shape row at the contact point -- the only place x_P enters."""
        k = round(float(x_pos), 9)
        if k not in self._D:
            self._D[k] = self.plate.D_row(x_pos, self.plate.hp)[:self.n]
        return self._D[k]

    def a4_of_t(self, t):
        """The true non-smooth coefficient at time t (Eqs. 3-4)."""
        a3, a4 = alpha34(t, self.Omega, self.plate.hp - self.ap,
                         self.plate.hp, self.ae)
        return self.sign * a3, self.sign * a4

    def a4_series(self, m):
        a3, a4 = alpha4_series(self.rpm, self.ap, self.plate.hp, m, self.ae)
        return self.sign * a3, self.sign * a4

    # ------------------------------------------------------------------
    def modal(self, eta=0.0, xi=1.0, zeta_scale=1.0):
        if eta <= 0.0 and zeta_scale == 1.0:
            M = np.eye(self.n)
            K = np.diag(self.omega0 ** 2)
            Cm = np.diag(2 * self.zeta0 * self.omega0)
            return M, Cm, K
        return self.removal.matrices(eta, xi, zeta_scale,
                                     d_eta=self.ap / self.plate.hp)

    def matrices(self, x_pos=0.0, a4=None, eta=0.0, xi=1.0, zeta_scale=1.0):
        """(A, A_d, B, E, C_y) at one milling position and one coefficient."""
        n = self.n
        M, Cm, K = self.modal(eta, xi, zeta_scale)
        Minv = np.linalg.inv(M)
        D = self.D(x_pos)
        W = (self.a40 if a4 is None else float(a4)) * np.outer(D, D)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ (K + W)
        A[n:, n:] = -Minv @ Cm
        Ad = np.zeros((2 * n, 2 * n))
        Ad[n:, :n] = Minv @ W
        B = np.concatenate([np.zeros(n), Minv @ self.H])[:, None]
        E = np.concatenate([np.zeros(n), Minv @ D])[:, None]
        return A, Ad, B, E, self.Cy

    def structure_only(self, eta=0.0, zeta_scale=1.0):
        """(A, B, C_y) of the structure alone -- no cutting term at all.

        This is what a controller that ignores the milling position must be
        designed on; it is the same for every x_P, which is precisely why such a
        controller cannot exploit the spatial dependence.
        """
        n = self.n
        M, Cm, K = self.modal(eta, 1.0, zeta_scale)
        Minv = np.linalg.inv(M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ K
        A[n:, n:] = -Minv @ Cm
        B = np.concatenate([np.zeros(n), Minv @ self.H])[:, None]
        return A, B, self.Cy

    # ------------------------------------------------------------------
    def positions(self, n_pos=None):
        return np.array(C.POSITIONS if n_pos is None
                        else np.linspace(0.0, 1.0, n_pos)) * self.plate.lp

    def feed_speed(self):
        return self.fz * N_TEETH * self.rpm / 60.0

    def pass_duration(self):
        return self.plate.lp / self.feed_speed()
