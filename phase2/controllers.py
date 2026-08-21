"""
controllers.py — PHASES 6 and 8: the five controllers compared.
================================================================
Every factory returns a controller in the SAME form: an LTI state space
(A_c, B_c, C_c, D_c) mapping the measured displacement y to the actuator
voltage u, plus optionally a pair of delayed gains (K_Pp, K_Pd) acting on
y(t - tau) and ydot(t - tau) -- the structure of Eq. (30) of Du et al. (2024).

    1  PID          classical baseline, 4 parameters
    2  LQR          optimal state feedback + Luenberger observer, 5 parameters
    3  SMC          sliding-mode with boundary layer + observer, 5 parameters
    4  MU           mu-synthesis by D-K iteration -- the paper's own controller
    5  PROPOSED     PB-RAC, 7 parameters

Fairness: one roll-off filter, one plant, one sensor, one sign, one objective,
one optimiser, one evaluation.  Only the structure changes, and the parameter
count is reported next to every result.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

import config as C
from fopid import rolloff_ss, series
from plate_model import plant_vectors


# ---------------------------------------------------------------------------
def design_model(plate, n=None):
    """(A, B, Cy) of the structure alone -- no cutting term.

    This is the model every model-based controller is designed on; the cutting
    term and the delay are what they then have to survive.
    """
    n = C.N_MODES_DESIGN if n is None else n
    w, z, H, D_obs, _ = plant_vectors(plate, n)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(w ** 2)
    A[n:, n:] = -np.diag(2 * z * w)
    B = np.concatenate([np.zeros(n), H])[:, None]
    Cy = np.concatenate([D_obs, np.zeros(n)])[None, :]
    return A, B, Cy


def _lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, np.atleast_2d(R))
    return np.linalg.solve(np.atleast_2d(R), B.T @ P)


def observer_gain(A, Cy, qo, ro):
    n = A.shape[0]
    return _lqr(A.T, Cy.T, qo * np.eye(n), ro).T


def _obs_ctrl(A, B, Cy, K, L):
    """(A_c, B_c, C_c, D_c) of observer + state feedback."""
    Ac = A - B @ K - L @ Cy
    return Ac, L, -K, np.zeros((1, 1))


def with_rolloff(ss):
    return series(ss, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))


# ---------------------------------------------------------------------------
# 1. PID
# ---------------------------------------------------------------------------
def pid(plate, Kp, Ki, Kd, Nd, n=None):
    """u = sign_loop (Kp + Ki/s + Kd Nd s/(s + Nd)) y, negative feedback."""
    _, _, _, _, sl = plant_vectors(plate, C.N_MODES_DESIGN if n is None else n)
    A = np.array([[0.0, 0.0], [0.0, -Nd]])
    B = np.array([[1.0], [1.0]])
    Cc = sl * np.array([[Ki, -Kd * Nd * Nd]])
    Dc = sl * np.array([[Kp + Kd * Nd]])
    return with_rolloff((A, B, Cc, Dc))


# ---------------------------------------------------------------------------
# 2. LQR + observer
# ---------------------------------------------------------------------------
def lqr(plate, q_pos, q_vel, r, qo, ro, n=None):
    A, B, Cy = design_model(plate, n)
    m = A.shape[0] // 2
    Q = np.diag(np.concatenate([q_pos * np.ones(m), q_vel * np.ones(m)]))
    K = _lqr(A, B, Q, r)
    L = observer_gain(A, Cy, qo, ro)
    return with_rolloff(_obs_ctrl(A, B, Cy, K, L))


# ---------------------------------------------------------------------------
# 3. Sliding mode with a boundary layer
# ---------------------------------------------------------------------------
def smc_surface(plate, lam, n=None):
    """s = G x,  G = [lam H^T, H^T]: the sliding variable of a mechanical
    system, projected on the actuator direction so that G B != 0."""
    n = C.N_MODES_DESIGN if n is None else n
    _, _, H, _, _ = plant_vectors(plate, n)
    return np.concatenate([lam * H, H])[None, :]


def smc(plate, lam, k, phi, qo, ro, n=None):
    """Boundary-layer SMC.  Inside the layer u = -(k/phi) s, which is the LTI
    form used by the Floquet analysis; the saturated law is used in the time
    domain.  Reported gains: (G, k, phi)."""
    A, B, Cy = design_model(plate, n)
    G = smc_surface(plate, lam, n)
    L = observer_gain(A, Cy, qo, ro)
    Keq = (k / phi) * G
    ss = with_rolloff(_obs_ctrl(A, B, Cy, Keq, L))
    return ss, dict(G=G, k=k, phi=phi, A=A, B=B, Cy=Cy, L=L)


# ---------------------------------------------------------------------------
# 5. PROPOSED — PB-RAC
# ---------------------------------------------------------------------------
def cancellation_gain(plate, alpha40, x_pos=None, n=None):
    """The gain that cancels the OWN-mode regeneration of every mode in the
    least-squares sense -- the analytic reading of the sentence in Section 3.4
    ("if the active control force satisfies -alpha4 D^T y(t-tau) ...").

        K_Pp = -alpha40 sum_i D_i^2 g_i / sum_i g_i^2 ,  g_i = H_i D_obs,i
    """
    n = C.N_MODES_DESIGN if n is None else n
    _, _, H, D_obs, _ = plant_vectors(plate, n)
    x_pos = 0.5 * plate.lp if x_pos is None else x_pos
    D = plate.D_row(x_pos, plate.hp)[:n]
    g = H * D_obs
    return float(-alpha40 * float((D ** 2) @ g) / float(g @ g))


def proposed(plate, q_pos, q_vel, r, qo, ro, kpd, kdd, eta_hat=0.0,
             alpha40=None, n=None, removal=None):
    """PB-RAC: eta-scheduled observer + state feedback + delayed PD.

    * the observer and the feedback are built on the model at the CURRENT
      removal state eta_hat, which is what the physics-based set says actually
      moves (mass down, stiffness almost fixed);
    * the delayed PD term attacks the regenerative coupling directly, and its
      gains are expressed as multiples of the analytic cancellation gain so the
      optimiser searches a normalised, physically meaningful interval.

    Returns (ss, pd) with pd = (K_Pp, K_Pd) for `closed_loop`.
    """
    n = C.N_MODES_DESIGN if n is None else n
    A, B, Cy = design_model(plate, n)
    if eta_hat > 0.0 and removal is not None:
        M, Cm, K = removal.matrices(eta_hat)
        Minv = np.linalg.inv(M)
        A = A.copy()
        A[n:, :n] = -Minv @ K
        A[n:, n:] = -Minv @ Cm
    m = A.shape[0] // 2
    Q = np.diag(np.concatenate([q_pos * np.ones(m), q_vel * np.ones(m)]))
    K = _lqr(A, B, Q, r)
    L = observer_gain(A, Cy, qo, ro)
    ss = with_rolloff(_obs_ctrl(A, B, Cy, K, L))
    Kpp0 = 0.0 if alpha40 is None else cancellation_gain(plate, alpha40, n=n)
    w, _, _, _, _ = plant_vectors(plate, n)
    return ss, (kpd * Kpp0, kdd * Kpp0 / w[0])


# ---------------------------------------------------------------------------
# builders used by the optimiser
# ---------------------------------------------------------------------------
def build(kind, plate, u, alpha40=None, removal=None, eta_hat=0.0):
    """(ss, pd, params) from a decoded parameter dict `u`."""
    if kind == 'pid':
        ss = pid(plate, 10 ** u['log_Kp'], 10 ** u['log_Ki'],
                 10 ** u['log_Kd'], 10 ** u['log_Nd'])
        return ss, None, u
    if kind == 'lqr':
        ss = lqr(plate, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                 10 ** u['log_r'], 10 ** u['log_qo'], 10 ** u['log_ro'])
        return ss, None, u
    if kind == 'smc':
        ss, _ = smc(plate, 10 ** u['log_lam'], 10 ** u['log_k'],
                    10 ** u['log_phi'], 10 ** u['log_qo'], 10 ** u['log_ro'])
        return ss, None, u
    if kind == 'proposed':
        ss, pd = proposed(plate, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                          10 ** u['log_r'], 10 ** u['log_qo'],
                          10 ** u['log_ro'], u['kpd'], u['kdd'],
                          eta_hat=eta_hat, alpha40=alpha40, removal=removal)
        return ss, pd, u
    raise ValueError(kind)


N_PARAMS = dict(pid=4, lqr=5, smc=5, mu=None, proposed=7)
