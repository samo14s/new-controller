"""
sim_ctrl.py — time-domain controller wrappers.
==============================================
`Combined` realises  u = u_LTI(y) + K_Pp y(t-tau) + K_Pd ydot(t-tau),
with the amplifier saturation of the test rig (PI E-420, gain 100).
`SlidingMode` realises the true saturated law, of which the LTI form used by
the Floquet analysis is the linear region.
"""
import numpy as np

import config as C
from sim_controller import LTIController


class Combined:
    def __init__(self, ss, pd=None, dt=1e-5, tau=None, u_sat=None):
        self.lti = LTIController(ss, dt)
        self.pd = (0.0, 0.0) if pd is None else (float(pd[0]), float(pd[1]))
        self.n_tau = 0 if tau is None else max(int(round(tau / dt)), 1)
        self.buf_y = np.zeros(self.n_tau + 1)
        self.buf_v = np.zeros(self.n_tau + 1)
        self.i = 0
        self.u_sat = C.U_SAT if u_sat is None else u_sat
        self.u_rob_last = 0.0
        self.u_pd_last = 0.0

    def reset(self):
        self.lti.reset()
        self.buf_y[:] = 0.0
        self.buf_v[:] = 0.0
        self.i = 0

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        u_r = self.lti(y=y, yd=yd, t=t, k=k)
        u_d = 0.0
        if self.n_tau:
            j = (self.i - self.n_tau) % len(self.buf_y)
            u_d = self.pd[0] * self.buf_y[j] + self.pd[1] * self.buf_v[j]
            self.buf_y[self.i] = y
            self.buf_v[self.i] = yd
            self.i = (self.i + 1) % len(self.buf_y)
        self.u_rob_last = u_r
        self.u_pd_last = u_d
        u = u_r + u_d
        return float(np.clip(u, -self.u_sat, self.u_sat))


class SlidingMode(Combined):
    """u = -k sat(s / phi) with s = G x_hat, x_hat from the same observer."""

    def __init__(self, info, dt, tau=None, u_sat=None):
        A, B, Cy, L = info['A'], info['B'], info['Cy'], info['L']
        from scipy.signal import cont2discrete
        n = A.shape[0]
        Aobs = A - L @ Cy
        Ad, Bd, _, _, _ = cont2discrete(
            (Aobs, np.hstack([L, B]), np.eye(n), np.zeros((n, 2))), dt, 'zoh')
        self.Ad, self.Bd = Ad, Bd
        self.x = np.zeros(n)
        self.G, self.k, self.phi = info['G'], info['k'], info['phi']
        self.u_sat = C.U_SAT if u_sat is None else u_sat
        self.u_rob_last = 0.0
        self.u_pd_last = 0.0
        self.n_tau = 0

    def reset(self):
        self.x[:] = 0.0

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        s = float(self.G @ self.x)
        u = -self.k * float(np.clip(s / self.phi, -1.0, 1.0))
        u = float(np.clip(u, -self.u_sat, self.u_sat))
        self.x = self.Ad @ self.x + self.Bd @ np.array([y, u])
        self.u_rob_last = u
        return u
