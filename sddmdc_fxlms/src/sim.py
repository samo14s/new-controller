"""
sim.py — compiled kernels shared by the virtual experiment and the
model-based simulation.
=====================================================================
The SAME closed-loop kernel runs both sides of the paper's Fig. 9:

  * "simulation"  : plant = identified sd-DMDc model, compensation = same model;
  * "experiment"  : plant = truth model of the virtual testbed (+ noise),
                    compensation = identified sd-DMDc model.

So any difference between the two comes from the plant, never from code.

Plants are given in complex-diagonal (modal) form
    z(k+1) = lam * z(k) + Bz u(k),   y(k) = Re(Cz z(k))
which is exact for any diagonalisable discrete LTI system and makes a
45 000-step closed loop cost milliseconds.

FxLMS convention (Fig. 2 / Eqs. 18-21 of the paper, with the sign fixed):
the sensor superposes the two paths PHYSICALLY,
    e(k) = d(k) + sum_j (G_s,ij * y_j)(k) + n(k),
and the update that descends E[e^2] is
    w_j(k+1) = w_j(k) - 2 mu_j sum_i e_i(k) r_ij(k),
r_ij = (G_c,ij * x)(k) the reference filtered by the compensation path.
Eq. (18) of the paper writes e = d - G_s y while Eq. (20) keeps the minus
sign in the update; the pair as printed ascends the gradient unless the
actuator sign is inverted (see docs/01).
"""
import numpy as np
from numba import njit


def to_modal(A, B, C):
    """Complex-diagonal form (lam, Bz, Cz) of a discrete (A, B, C)."""
    lam, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    return lam.astype(np.complex128), (Vi @ B).astype(np.complex128), \
        (C @ V).astype(np.complex128)


@njit(cache=True)
def modal_sim(lam, Bz, Cz, u, z0):
    """Open-loop response. u: (n_steps, n_in). Returns y: (n_steps, n_out)."""
    ns = u.shape[0]
    nz = lam.shape[0]
    ny = Cz.shape[0]
    nu = Bz.shape[1]
    y = np.zeros((ns, ny))
    z = z0.copy()
    for k in range(ns):
        for i in range(ny):
            acc = 0.0
            for m in range(nz):
                acc += (Cz[i, m] * z[m]).real
            y[k, i] = acc
        for m in range(nz):
            s = lam[m] * z[m]
            for j in range(nu):
                s += Bz[m, j] * u[k, j]
            z[m] = s
    return y


def steady_state_modal(lam, Bz, u_amp, omega_d):
    """Complex amplitude Z such that z(k) = Re-free phasor Z e^{i w k} is the
    steady state for u(k) = Re(u_amp e^{i w k}) (u_amp complex vector).
    Because the plant is real, z(k) = Z+ e^{iwk} + Z- e^{-iwk} with
    Z+ = (e^{iw} - lam)^-1 Bz u_amp/2 and Z- = (e^{-iw} - lam)^-1 Bz conj(u_amp)/2.
    Returns (Zp, Zm)."""
    ep, em = np.exp(1j * omega_d), np.exp(-1j * omega_d)
    bu = Bz @ u_amp
    bc = Bz @ np.conj(u_amp)
    return bu / (2 * (ep - lam)), bc / (2 * (em - lam))


@njit(cache=True)
def fxlms_loop(lam_p, Bz_p, Cz_p, z0,          # plant (inputs: [x, y_1..y_na])
               lam_c, Bz_c, Cz_c, zc0,          # compensation, inputs: actuators
               x, noise, mu, L, k_on, w0):
    """Multichannel FxLMS with ONE reference x (the excitation command).

    Plant inputs are [x, y_1, ..., y_na]; plant outputs are the n_e error
    sensors. The compensation model maps actuator j -> sensor i and is run on
    the reference x for each actuator j separately (r_ij = Gc_ij * x).
    mu: (na,) step sizes; L: (na,) filter lengths (<= Lmax = w0.shape[1]).
    Control starts at sample k_on. Returns e: (ns, n_e), y: (ns, na), w final.
    """
    ns = x.shape[0]
    na = mu.shape[0]
    ne = Cz_p.shape[0]
    nzp = lam_p.shape[0]
    nzc = lam_c.shape[0]
    Lmax = w0.shape[1]
    w = w0.copy()
    zp = z0.copy()
    # one compensation state per actuator (input j driven by x)
    zc = zc0.copy()                     # (na, nzc)
    xbuf = np.zeros(Lmax)               # x(k), x(k-1), ...
    rbuf = np.zeros((na, ne, Lmax))     # r_ij(k), r_ij(k-1), ...
    e = np.zeros((ns, ne))
    yout = np.zeros((ns, na))
    yk = np.zeros(na)
    for k in range(ns):
        # shift reference buffer
        for l in range(Lmax - 1, 0, -1):
            xbuf[l] = xbuf[l - 1]
        xbuf[0] = x[k]
        # actuator outputs y_j(k) = w_j^T x_buf
        for j in range(na):
            acc = 0.0
            if k >= k_on:
                for l in range(L[j]):
                    acc += w[j, l] * xbuf[l]
            yk[j] = acc
            yout[k, j] = acc
        # plant output (strictly proper: depends on z(k) only)
        for i in range(ne):
            acc = 0.0
            for m in range(nzp):
                acc += (Cz_p[i, m] * zp[m]).real
            e[k, i] = acc + noise[k, i]
        # filtered references r_ij(k) = Re(Cz_c[i] . zc[j]) ; shift buffers
        for j in range(na):
            for i in range(ne):
                for l in range(Lmax - 1, 0, -1):
                    rbuf[j, i, l] = rbuf[j, i, l - 1]
                acc = 0.0
                for m in range(nzc):
                    acc += (Cz_c[i, m] * zc[j, m]).real
                rbuf[j, i, 0] = acc
        # LMS update
        if k >= k_on:
            for j in range(na):
                for l in range(L[j]):
                    g = 0.0
                    for i in range(ne):
                        g += e[k, i] * rbuf[j, i, l]
                    w[j, l] -= 2.0 * mu[j] * g
        # advance compensation states (input j = x(k))
        for j in range(na):
            for m in range(nzc):
                zc[j, m] = lam_c[m] * zc[j, m] + Bz_c[m, j] * x[k]
        # advance plant: inputs [x, y]
        for m in range(nzp):
            s = lam_p[m] * zp[m] + Bz_p[m, 0] * x[k]
            for j in range(na):
                s += Bz_p[m, 1 + j] * yk[j]
            zp[m] = s
    return e, yout, w


def convergence_time(e, fs, f_exc, k_on, frac=0.05, t_cap=20.0, pre=1.0):
    """Paper's metric (Sec. 4.2): time for the sensing-voltage amplitude to
    drop to `frac` of its pre-control amplitude.  Amplitude = RMS over a
    window of an integer number of excitation periods (>= 50 ms).  The time
    is the first instant after which the envelope STAYS below the threshold;
    runs that never settle within t_cap, or diverge, return t_cap
    (the paper assigns 20 s to divergent / > 20 s cases)."""
    e = np.asarray(e, float)
    per = fs / f_exc
    nwin = int(round(per * max(1, np.ceil(0.05 * f_exc))))
    sq = np.convolve(e**2, np.ones(nwin) / nwin, mode='valid')
    env = np.sqrt(np.maximum(sq, 0.0))
    # env[i] = RMS of e[i : i + nwin]; reference: window ending at k_on
    i_pre0 = max(0, k_on - int(pre * fs))
    ref = np.sqrt(np.mean(e[i_pre0:k_on]**2))
    thr = frac * ref
    post = env[k_on:]
    if post.size == 0:
        return t_cap, ref
    above = np.nonzero(post > thr)[0]
    if above.size == 0:
        t = 0.0
    elif above[-1] >= post.size - 1:
        return t_cap, ref
    else:
        t = (above[-1] + 1 + nwin // 2) / fs     # window centre
    return min(t, t_cap), ref
