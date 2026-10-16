"""
theory.py — tonal FxLMS reduced to what it actually is.
========================================================
Probe of research line S1 (docs/03): can the (mu, L) convergence map of
Sections 5.1-5.3 of Ouyang et al. (2026) be PREDICTED instead of swept?

1.  Exact two-weight reduction.  With a pure-tone reference
        x(k) = A cos(w k + phi0)
    an L-tap FIR only ever acts through two numbers, p = c.w and q = s.w,
    c_l = cos(w l), s_l = sin(w l):
        y(k) = A [ p cos(w k + phi0) + q sin(w k + phi0) ].
    Projecting the FxLMS update w <- w - 2 mu e r_vec on c and s gives, with
    r(k) = A |Gc| cos(w k + phi0 + th_c) (Gc at steady state),
        [p; q](k+1) = [p; q](k) - 2 mu A |Gc| e(k) Gamma [C_k; S_k],
        Gamma = [[c.c, c.s], [c.s, s.s]],
        C_k = cos(w k + phi0 + th_c),  S_k = sin(w k + phi0 + th_c).
    This is EXACT (not averaged) once the buffers are full: the L-tap filter
    is a 2-state system whatever L is.  L enters only through Gamma:
        c.c, s.s = L/2 +/- sin(L w) cos((L-1) w) / (2 sin w),
        c.s      = sin(L w) sin((L-1) w) / (2 sin w),
    whose eigenvalues are L/2 +/- |sin(L w)| / (2 |sin w|).

2.  Averaged LTI equivalent.  When Gamma ~ (L/2) I (L spans many half
    periods), the controller is the linear time-invariant filter
        y = -beta [cos(w - th_c) z^-1 - cos(th_c) z^-2] / (1 - 2 cos w z^-1 + z^-2) e,
        beta = mu L A^2 |Gc|
    (Glover 1977; Elliott et al. 1987; Morgan & Sanford 1992 for FxLMS).
    So the whole (mu, L) plane collapses onto ONE variable, beta ~ mu L,
    and stability / convergence time are those of an LTI loop made of the
    secondary path G_s and this resonator: no sweep needed.
"""
import numpy as np
from numba import njit


def gram(L, w):
    """Gamma(L, w) and its eigenvalues."""
    l = np.arange(L)
    c, s = np.cos(w * l), np.sin(w * l)
    G = np.array([[c @ c, c @ s], [c @ s, s @ s]])
    return G, np.linalg.eigvalsh(G)


@njit(cache=True)
def reduced_fxlms(lam_p, Bz_p, Cz_p, z0, a, w, phi0, gc_abs, th_c, Gam, mu,
                  noise, k_on, ns):
    """Exact two-weight tonal FxLMS, one actuator, one error sensor.
    Plant inputs [x, y]; returns e (ns,)."""
    nz = lam_p.shape[0]
    zp = z0.copy()
    p = 0.0
    q = 0.0
    e = np.zeros(ns)
    for k in range(ns):
        ph = w * k + phi0
        xk = a * np.cos(ph)
        yk = a * (p * np.cos(ph) + q * np.sin(ph)) if k >= k_on else 0.0
        acc = 0.0
        for m in range(nz):
            acc += (Cz_p[0, m] * zp[m]).real
        e[k] = acc + noise[k]
        if k >= k_on:
            Ck = np.cos(ph + th_c)
            Sk = np.sin(ph + th_c)
            g = 2.0 * mu * a * gc_abs * e[k]
            p -= g * (Gam[0, 0] * Ck + Gam[0, 1] * Sk)
            q -= g * (Gam[1, 0] * Ck + Gam[1, 1] * Sk)
        for m in range(nz):
            zp[m] = lam_p[m] * zp[m] + Bz_p[m, 0] * xk + Bz_p[m, 1] * yk
    return e


def averaged_controller(betas, w, th_c):
    """Real state space (Ah, Bh, Ch) of the averaged LTI equivalent for n_a
    actuators sharing one error sensor: y_j = -beta_j H(th_c_j) e."""
    betas = np.atleast_1d(betas)
    th_c = np.atleast_1d(th_c)
    na = betas.size
    Ah = np.zeros((2 * na, 2 * na))
    Bh = np.zeros((2 * na, 1))
    Ch = np.zeros((na, 2 * na))
    for j in range(na):
        # H(z) = (b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2), controllable form
        b1, b2 = np.cos(w - th_c[j]), -np.cos(th_c[j])
        a1, a2 = -2 * np.cos(w), 1.0
        i = 2 * j
        Ah[i, i], Ah[i, i + 1] = -a1, -a2
        Ah[i + 1, i] = 1.0
        Bh[i, 0] = 1.0
        Ch[j, i], Ch[j, i + 1] = -betas[j] * b1, -betas[j] * b2
    return Ah, Bh, Ch


def closed_loop(Ag, Bg, Cg, Ah, Bh, Ch):
    """e = d + Cg xg ; xg+ = Ag xg + Bg y ; y = Ch xh ; xh+ = Ah xh + Bh e.
    Returns (Acl, Bcl_d, Ccl) with d entering through e."""
    ng, nh = Ag.shape[0], Ah.shape[0]
    Acl = np.block([[Ag, Bg @ Ch], [Bh @ Cg, Ah]])
    Bd = np.vstack([np.zeros((ng, 1)), Bh])
    Ccl = np.hstack([Cg, np.zeros((Cg.shape[0], nh))])
    return Acl, Bd, Ccl


def lti_t5(Acl, Bd, Ccl, d_amp, w, fs, f_exc, t_cap=20.0, pre=1.0, fast=True,
           dec=10, frac=0.05):
    """Convergence time of the averaged loop for d(k) = Re(d_amp e^{i w k})
    switched on with the controller at rest and the secondary-path state at
    zero.  The closed-loop transfer d -> e has zeros at e^{+-iw} (the
    resonator), so after the switch e(k) is pure transient,
        e(k) = -Ccl Acl^k x_f(0),
    x_f the forced closed-loop state.  e(k) is evaluated modally at every
    sample and scored with the SAME metric as the time-domain runs
    (sim.convergence_time).  Returns (t5, spectral radius)."""
    from sim import convergence_time
    lam, V = np.linalg.eig(Acl)
    rho = np.max(np.abs(lam))
    # with more actuators than error sensors the resonators keep a marginal
    # (|lam| = 1) direction that is unobservable in e: tolerate it
    if rho > 1.0 + 1e-9:
        return t_cap, rho
    I = np.eye(Acl.shape[0])
    Mw = np.exp(1j * w) * I - Acl
    try:
        Xp = np.linalg.solve(Mw, Bd[:, 0] * d_amp / 2)
    except np.linalg.LinAlgError:
        # two actuators, one sensor: e^{iw} is an eigenvalue of the marginal
        # null-space direction, unobservable in e -- any particular solution
        Xp = np.linalg.lstsq(Mw, Bd[:, 0] * d_amp / 2, rcond=None)[0]
    xf0 = 2 * Xp.real
    coef = np.linalg.solve(V, -xf0)
    amp = (Ccl[0] @ V) * coef
    if fast:
        # analytic-signal envelope: positive-frequency modes only (x2), real
        # modes once; exact for a narrowband transient, evaluated every `dec`
        # samples.  Same 5 % threshold as the time-domain metric.
        pos = lam.imag > 1e-12
        real = np.abs(lam.imag) <= 1e-12
        lm = np.r_[lam[pos], lam[real]].astype(complex)
        am = np.r_[2 * amp[pos], amp[real]]
        ks = np.arange(0, int(round(t_cap * fs)) + 1, dec)
        env = np.abs(np.exp(np.log(lm)[None, :] * ks[:, None]) @ am)
        above = np.nonzero(env > frac * abs(d_amp))[0]
        if above.size == 0:
            return 0.0, rho
        if above[-1] == ks.size - 1:
            return t_cap, rho
        return ks[above[-1] + 1] / fs, rho
    npost = int(round((t_cap + 1.0) * fs))
    k_on = int(round(pre * fs))
    ks = np.arange(npost)
    loglam = np.log(lam.astype(complex))
    e_post = np.zeros(npost)
    for m in range(lam.size):                     # modal sum, O(n_modes * n)
        if abs(amp[m]) > 0:
            e_post += np.real(amp[m] * np.exp(loglam[m] * ks))
    kpre = np.arange(-k_on, 0)
    e_pre = np.real(d_amp * np.exp(1j * w * kpre))
    t5, _ = convergence_time(np.r_[e_pre, e_post], fs, f_exc, k_on, t_cap=t_cap)
    return t5, rho
