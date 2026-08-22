"""
ctrl2.py — STAGES 3 and 4: the four controllers the plan fixes.
================================================================
    FOPID      fractional-order baseline                     5 parameters
    LQG        optimal output feedback, Kalman observer       4 parameters
    MU_TDC     mu-synthesis + time-delay control (Du 2024)    4 weights + 2 gains
    PS_AC      PROPOSED: position-scheduled active control    4 parameters

Every controller is returned as a `Ctrl`, which answers one question: given the
milling position and the removal state, what is the LTI controller in the loop
right now?  A fixed controller ignores both arguments; the proposed one does not.

The proposed controller has EXACTLY the same parameter count and the same
structure as LQG.  The only difference is the model each is designed on:

    LQG    designs on the structure alone, so its gain cannot depend on x_P;
    PS_AC  designs on A(x_P) INCLUDING the nominal cutting term
           alpha_40 D(x_P)^T D(x_P), so its gain follows the position.

That is the whole research gap of Stage 4, reduced to one line of code and zero
extra tuning freedom.  The position is not estimated: the machine commands the
feed, so x_P(t) is known exactly.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

import config as C
from fopid import fopid_ss, rolloff_ss, series


# ---------------------------------------------------------------------------
class Ctrl:
    """A controller in the loop: fixed LTI, or scheduled on (x_P, eta)."""

    def __init__(self, name, n_params, ss=None, pd=None, builder=None,
                 scheduled=False, meta=None):
        self.name = name
        self.n_params = int(n_params)
        self._ss = ss
        self._pd = pd
        self._builder = builder
        self.scheduled = bool(scheduled)
        self.meta = meta or {}
        self._cache = {}

    def at(self, x_pos=None, eta=0.0):
        if not self.scheduled:
            return self._ss, self._pd
        key = (None if x_pos is None else round(float(x_pos), 6),
               round(float(eta), 6))
        if key not in self._cache:
            self._cache[key] = self._builder(x_pos, eta)
        return self._cache[key]

    @property
    def order(self):
        ss, _ = self.at(0.0)
        return 0 if ss is None else np.atleast_2d(ss[0]).shape[0]


def _rolloff(ss):
    return series(ss, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))


def _lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, np.atleast_2d(R))
    return np.linalg.solve(np.atleast_2d(R), B.T @ P)


def _kalman_gain(A, Cy, W, v):
    P = solve_continuous_are(A.T, Cy.T, W, np.atleast_2d(v))
    return P @ Cy.T / float(v)


def _observer_ctrl(A, B, Cy, K, L):
    return (A - B @ K - L @ Cy, L, -K, np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# 1. FOPID
# ---------------------------------------------------------------------------
def fopid(plant, Kp, Ki, Kd, lam, mu):
    ss = fopid_ss(Kp, Ki, Kd, lam, mu, C.OUST_WB, C.OUST_WH, C.OUST_N,
                  plant.sign_loop)
    return Ctrl('FOPID', 5, ss=_rolloff(ss))


# ---------------------------------------------------------------------------
# 2. LQG
# ---------------------------------------------------------------------------
def _mean_process_noise(plant):
    """W = mean_x E(x) E(x)^T : the cutting force enters through E(x_P), and a
    position-blind observer must average over the positions it will meet."""
    n = plant.n
    W = np.zeros((2 * n, 2 * n))
    xs = np.linspace(0.0, plant.plate.lp, 21)
    for x in xs:
        _, _, _, E, _ = plant.matrices(x_pos=x)
        W += E @ E.T
    return W / len(xs)


def lqg(plant, q_pos, q_vel, r, ratio, eta=0.0):
    A, B, Cy = plant.structure_only(eta)
    n = plant.n
    Q = np.diag(np.concatenate([q_pos * np.ones(n), q_vel * np.ones(n)]))
    K = _lqr_gain(A, B, Q, r)
    W = _mean_process_noise(plant) * ratio
    L = _kalman_gain(A, Cy, W + 1e-12 * np.eye(2 * n), 1.0)
    return Ctrl('LQG', 4, ss=_rolloff(_observer_ctrl(A, B, Cy, K, L)))


# ---------------------------------------------------------------------------
# 3. mu-synthesis + time-delay control (Du 2024, Sections 3.3-3.4)
# ---------------------------------------------------------------------------
def cancellation_gain(plant, x_pos=None):
    """K_Pp that cancels the OWN-mode regeneration in the least-squares sense.

    Section 3.4: "if the active control force satisfies -alpha_4 D^T y(t-tau),
    F_PI will have no time delay part".  With one patch the cancellation cannot
    be exact -- two rank-one matrices with different directions -- so the
    diagonal (own-mode) part is cancelled:

        K_Pp = -alpha_40 sum_i D_i^2 g_i / sum_i g_i^2 ,   g = H_Pe . D_obs
    """
    x_pos = 0.5 * plant.plate.lp if x_pos is None else x_pos
    D = plant.D(x_pos)
    g = plant.H * plant.D_obs
    return float(-plant.a40 * float((D ** 2) @ g) / float(g @ g))


def mu_tdc(plant, ss_mu, kpd, kdd, name='MU_TDC'):
    """The published benchmark: robust controller + Eq. (30) delayed PD.

    `ss_mu` is the mu-synthesis controller; which uncertainty set produced it is
    the only thing that separates MU_TDC from MU_PHYS_TDC.  The delayed PD is
    tuned by the same two-parameter PSO in both cases, so the comparison isolates
    the uncertainty description.
    """
    k0 = cancellation_gain(plant)
    pd = (kpd * k0, kdd * k0 / plant.omega0[0])
    return Ctrl(name, 6, ss=ss_mu, pd=pd,
                meta=dict(K_Pp0=k0))


# ---------------------------------------------------------------------------
# 4. PROPOSED - position-scheduled active control
# ---------------------------------------------------------------------------
def ps_ac(plant, q_pos, q_vel, r, ratio, schedule_eta=False, n_grid=21,
          sched_K=True, sched_L=True, name=None):
    """u(t) = -K(x_P) xhat(t),  xhat from an observer also built at x_P.

    Design model at position x:

        A(x) = [ 0                                  I      ]
               [ -M^-1(K + alpha_40 D(x)^T D(x))  -M^-1 C  ]

    i.e. the NOMINAL cutting stiffness at that position is part of the design
    model instead of being pushed into an uncertainty box.  The gain is computed
    on a grid of positions and interpolated; within one tooth period the tool
    advances 0.02 mm, so the gain is effectively frozen over the delay.
    """
    n = plant.n
    Q = np.diag(np.concatenate([q_pos * np.ones(n), q_vel * np.ones(n)]))
    xs = np.linspace(0.0, plant.plate.lp, n_grid)

    Wbar = _mean_process_noise(plant)

    def build(x_pos, eta=0.0):
        x = 0.5 * plant.plate.lp if x_pos is None else float(x_pos)
        x = float(np.clip(x, xs[0], xs[-1]))
        e = float(eta) if schedule_eta else 0.0
        A, _, B, E, Cy = plant.matrices(x_pos=x, eta=e)
        A0, B0, _ = plant.structure_only(e)
        # state feedback: on the cutting-loaded model only if sched_K
        Ak = A if sched_K else A0
        K = _lqr_gain(Ak, B, Q, r)
        # observer: the cutting force enters through E(x_P), so a scheduled
        # filter knows the disturbance DIRECTION at the current position while a
        # fixed one has to average over the whole edge
        W = (E @ E.T) if sched_L else Wbar
        L = _kalman_gain(A0, Cy, ratio * W + 1e-12 * np.eye(2 * n), 1.0)
        # the controller state matrix must use the model the observer runs on
        Aobs = Ak if sched_K else A0
        return _rolloff(_observer_ctrl(Aobs, B, Cy, K, L)), None

    if name is None:
        name = 'PS_AC' + ('+eta' if schedule_eta else '')
    return Ctrl(name, 5 if schedule_eta else 4, builder=build, scheduled=True,
                meta=dict(grid=xs, sched_K=sched_K, sched_L=sched_L))


# ---------------------------------------------------------------------------
def build(kind, plant, u, ss_mu=None):
    """Decoded parameter dict -> Ctrl."""
    if kind == 'fopid':
        return fopid(plant, 10 ** u['log_Kp'], 10 ** u['log_Ki'],
                     10 ** u['log_Kd'], u['lam'], u['mu'])
    if kind == 'lqg':
        return lqg(plant, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                   10 ** u['log_r'], 10 ** u['log_ratio'])
    if kind == 'mu_tdc':
        return mu_tdc(plant, ss_mu, u['kpd'], u['kdd'])
    if kind == 'mu_phys_tdc':
        return mu_tdc(plant, ss_mu, u['kpd'], u['kdd'], name='MU_PHYS_TDC')
    if kind in ('ps_ac', 'ps_ac_eta', 'ps_ac_obs', 'ps_ac_full'):
        # 'ps_ac' is the law the plan writes, u = -K(x_P) x: the GAIN is
        # scheduled and nothing else.  The two diagnostics that also schedule
        # the observer are kept because they show why that restriction matters:
        # a position-scheduled Kalman filter is tuned to the disturbance
        # direction at the current position, and against the NO-CUTTING plant --
        # which is what the modulus margin is measured on, and what the tool
        # actually sees at entry and exit -- that pushes Ms past the constraint.
        return ps_ac(plant, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                     10 ** u['log_r'], 10 ** u['log_ratio'],
                     schedule_eta=(kind == 'ps_ac_eta'),
                     sched_K=(kind != 'ps_ac_obs'),
                     sched_L=(kind in ('ps_ac_obs', 'ps_ac_full')),
                     name={'ps_ac_obs': 'PS_AC(observer only)',
                           'ps_ac_full': 'PS_AC(K and observer)'}.get(kind))
    raise ValueError(kind)
