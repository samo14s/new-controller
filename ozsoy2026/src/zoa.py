"""Zero-order (Altintas-Budak) stability limit for the 1-DOF closed loop.

With the workpiece flexible along y only, the oriented transfer function
reduces to a scalar.  Averaging the dynamic chip force over the tooth pitch
gives  F = (N Kt a / 4 pi) * alpha_yy * (1 - e^{-sT}) X,  so the characteristic
equation is

    1 - (N Kt a / 4 pi) alpha_yy (1 - e^{-i w T}) Gcl(i w) = 0.

For a fixed spindle speed (fixed T) every real-positive root a(w) of this
equation is a boundary crossing; the smallest one is a_lim.  Solving it per
speed avoids interpolating lobes and handles any number of modes in Gcl,
including the actuator mode that DVF adds near 8 Hz.
"""
import numpy as np
from params import K_TC, K_RC, N_TEETH, immersion_angles, FLEX_DIR
from structure import Gcl

KR = K_RC / K_TC


def alpha_matrix(milling="down", ae_ratio=0.5):
    """Average directional factors (Altintas 2012, Eq. 3.49), with the 1/2."""
    st, ex = immersion_angles(milling, ae_ratio)

    def ev(f):
        return f(ex) - f(st)

    axx = 0.5 * ev(lambda p: np.cos(2 * p) - 2 * KR * p + KR * np.sin(2 * p))
    axy = 0.5 * ev(lambda p: -np.sin(2 * p) - 2 * p + KR * np.cos(2 * p))
    ayx = 0.5 * ev(lambda p: -np.sin(2 * p) + 2 * p + KR * np.cos(2 * p))
    ayy = 0.5 * ev(lambda p: -np.cos(2 * p) - 2 * KR * p - KR * np.sin(2 * p))
    return np.array([[axx, axy], [ayx, ayy]])


ALPHA = alpha_matrix()[1, 1] if FLEX_DIR == "y" else alpha_matrix()[0, 0]

# chatter-frequency grid: covers the actuator mode (8 Hz) and the structure.
# Below 60 Hz no root exists for g < ~2.4e4 V s/m (verification/zoa_sdm), so a
# coarse safeguard band is enough there; 60-600 Hz at 0.02 Hz.
F_GRID = np.concatenate([np.linspace(0.5, 60, 1200), np.linspace(60, 600, 27001)[1:]])
W_GRID = 2 * np.pi * F_GRID


def _gcl_grid(g, w=W_GRID):
    return Gcl(1j * w, g)


def a_lim(rpm, g, gcl=None, w=W_GRID, return_freq=False):
    """Limiting axial depth (m) at one spindle speed for DVF gain g."""
    if gcl is None:
        gcl = _gcl_grid(g, w)
    T = 60.0 / (N_TEETH * rpm)
    # a(w) = 4 pi / (N Kt alpha (1 - e^{-iwT}) Gcl)
    den = N_TEETH * K_TC * ALPHA * (1 - np.exp(-1j * w * T)) * gcl
    a = 4 * np.pi / den
    im = a.imag
    sgn = np.sign(im)
    idx = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
    # a(w) has poles at the tooth-passing harmonics (1 - e^{-iwT} = 0); a sign
    # change bracketing or touching one is a pole, not a root
    ph = w * T / (2 * np.pi)
    near = np.abs(ph - np.round(ph)) < 1e-6
    keep = (np.floor(ph[idx]) == np.floor(ph[idx + 1])) & ~near[idx] & ~near[idx + 1]
    idx = idx[keep]
    if idx.size == 0:
        return (np.inf, np.nan) if return_freq else np.inf
    # linear interpolation of the root of Im a(w)
    t = im[idx] / (im[idx] - im[idx + 1])
    are = a.real[idx] + t * (a.real[idx + 1] - a.real[idx])
    wc = w[idx] + t * (w[idx + 1] - w[idx])
    ok = are > 0
    if not np.any(ok):
        return (np.inf, np.nan) if return_freq else np.inf
    j = np.argmin(np.where(ok, are, np.inf))
    if return_freq:
        return are[j], wc[j] / (2 * np.pi)
    return are[j]


def sld(rpms, g, return_freq=False):
    """a_lim over a speed grid for a fixed gain (gain may be an array per speed)."""
    rpms = np.atleast_1d(rpms)
    gs = np.broadcast_to(np.asarray(g, float), rpms.shape)
    out = np.empty(rpms.shape)
    fc = np.empty(rpms.shape)
    cache = {}
    for i, (n, gi) in enumerate(zip(rpms, gs)):
        if gi not in cache:
            cache[gi] = _gcl_grid(gi)
        out[i], fc[i] = a_lim(n, gi, gcl=cache[gi], return_freq=True)
    return (out, fc) if return_freq else out
