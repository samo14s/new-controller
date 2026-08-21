"""
sim_ctrl2.py — time-domain realisation of a scheduled controller.
==================================================================
The members of a scheduled family share one state ordering -- observer state
then roll-off state, all in the same physical coordinates -- so the state can be
carried across a switch: only the matrices change, the estimate does not jump.
The tool advances 0.02 mm per tooth period, so the gain is effectively frozen
over the delay and the switching is slow compared with every closed-loop mode.
"""
import numpy as np
from scipy.signal import cont2discrete

import config as C


class ScheduledLTI:
    """y -> u, with the member selected by the current tool position."""

    def __init__(self, ctrl, dt, lp, tau=None, n_grid=21, eta=0.0,
                 u_sat=None, feed=None, moving=True, x0=0.0):
        self.ctrl = ctrl
        self.dt = dt
        self.lp = lp
        self.xs = np.linspace(0.0, lp, n_grid)
        self.eta = eta
        self.u_sat = C.U_SAT if u_sat is None else u_sat
        self.mats = []
        n = 0
        for x in self.xs:
            ss, pd = ctrl.at(float(x), eta)
            A, B, Cc, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
            B = B.reshape(-1, 1)
            n = A.shape[0]
            Ad, Bd, _, _, _ = cont2discrete((A, B, Cc.reshape(1, -1),
                                             D.reshape(1, 1)), dt, 'zoh')
            self.mats.append((Ad, Bd[:, 0], Cc.reshape(1, -1).ravel(),
                              float(D.reshape(1, 1)[0, 0]),
                              (0.0, 0.0) if pd is None else
                              (float(pd[0]), float(pd[1]))))
        self.x = np.zeros(n)
        self.n_tau = 0 if tau is None else max(int(round(tau / dt)), 1)
        self.buf_y = np.zeros(self.n_tau + 1)
        self.buf_v = np.zeros(self.n_tau + 1)
        self.i = 0
        self.feed = feed
        self.moving = bool(moving)
        self.x0 = float(x0)
        self.x_tool = float(x0)
        self.u_rob_last = 0.0
        self.u_pd_last = 0.0

    def reset(self):
        self.x_tool = self.x0
        self.x[:] = 0.0
        self.buf_y[:] = 0.0
        self.buf_v[:] = 0.0
        self.i = 0

    def set_position(self, x_tool):
        self.x_tool = float(x_tool)

    def __call__(self, y=0.0, yd=0.0, t=0.0, k=0):
        # the machine commands the feed, so the tool position is known exactly;
        # the controller reads it rather than estimating it
        if self.moving and self.feed:
            self.x_tool = min(self.feed * float(t), self.lp)
        j = int(np.clip(round(self.x_tool / self.lp * (len(self.xs) - 1)),
                        0, len(self.xs) - 1))
        Ad, Bd, Cc, D, pd = self.mats[j]
        u_r = D * y + (float(Cc @ self.x) if self.x.size else 0.0)
        self.x = Ad @ self.x + Bd * y
        u_d = 0.0
        if self.n_tau and (pd[0] or pd[1]):
            m = (self.i - self.n_tau) % len(self.buf_y)
            u_d = pd[0] * self.buf_y[m] + pd[1] * self.buf_v[m]
        if self.n_tau:
            self.buf_y[self.i] = y
            self.buf_v[self.i] = yd
            self.i = (self.i + 1) % len(self.buf_y)
        self.u_rob_last, self.u_pd_last = u_r, u_d
        return float(np.clip(u_r + u_d, -self.u_sat, self.u_sat))
