"""
mu_tight.py — mu upper bound with FULL D-scales on repeated blocks.
====================================================================
`dk_synthesis.mu_upper_bound` gives one SCALAR scale per block, which is exact
for the 1x1 blocks of Eqs. (22)-(25) but conservative for a repeated real block
delta I_k: the commuting set of delta I_k is every invertible k x k matrix, not
just a multiple of the identity.

The physics-based set uses repeated blocks (that is the point -- a repeated
scalar is what says "these k channels are driven by ONE physical quantity"), so
reporting its mu with scalar scales would penalise it for its own structure.
This module minimises

        sigma_bar( D_y M D_u^-1 ),   D_block = exp(H),  H = H* for 'S' blocks,
                                     D_block = exp(p) I for 'F' blocks,
                                     D_block = I        for 'P',

over the Hermitian parameters H, warm-started from the scalar solution.  Any D
in the commuting set gives a valid upper bound, so the result is a bound however
the optimisation ends -- it is only ever tighter than the scalar one.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

import config as _C            # noqa: F401  (puts baseline/control_old on the path)
from dk_synthesis import mu_upper_bound, n_scales


def _unpack(blocks, x):
    """(D_y, D_u) block-diagonal from the parameter vector."""
    Dy, Du, i = [], [], 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            Dy.append(np.eye(no, dtype=complex))
            Du.append(np.eye(ni, dtype=complex))
            continue
        if kind == 'F' or no == 1:
            d = np.exp(x[i]); i += 1
            Dy.append(d * np.eye(no, dtype=complex))
            Du.append(d * np.eye(ni, dtype=complex))
            continue
        k = no
        H = np.zeros((k, k), complex)
        H[np.diag_indices(k)] = x[i:i + k]; i += k
        for a in range(k):
            for b in range(a + 1, k):
                H[a, b] = x[i] + 1j * x[i + 1]
                H[b, a] = x[i] - 1j * x[i + 1]
                i += 2
        D = expm(H)
        Dy.append(D)
        Du.append(D)
    return _blkdiag(Dy), _blkdiag(Du)


def _blkdiag(mats):
    n = sum(m.shape[0] for m in mats)
    m2 = sum(m.shape[1] for m in mats)
    out = np.zeros((n, m2), complex)
    r = c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]; c += m.shape[1]
    return out


def _has_repeated(blocks):
    """True when some block is a REPEATED scalar, delta I_k with k > 1 -- the
    only case in which a full-matrix D-scale is tighter than a scalar one."""
    return any(kind == 'S' and no > 1 for (kind, no, ni) in blocks)


def n_params(blocks):
    n = 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            continue
        n += 1 if (kind == 'F' or no == 1) else no * no
    return n


def _x0(blocks, d0):
    x, j = [], 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            continue
        v = np.log(max(d0[j], 1e-12)); j += 1
        if kind == 'F' or no == 1:
            x.append(v)
        else:
            x += [v] * no + [0.0] * (no * (no - 1))
    return np.array(x)


def mu_bound(M, blocks, d0=None, maxiter=200, x0=None):
    """(mu_scalar, mu_full, d_scalar, x_full) at one frequency."""
    mu_s, ds = mu_upper_bound(M, blocks, d0)
    if not _has_repeated(blocks):
        return mu_s, mu_s, ds, None
    starts = [_x0(blocks, ds)] + ([] if x0 is None else [x0])

    def cost(x):
        Dy, Du = _unpack(blocks, x)
        return float(np.linalg.norm(Dy @ M @ np.linalg.inv(Du), 2))

    best, bx = mu_s, starts[0]
    for s in starts:
        r = minimize(cost, s, method='L-BFGS-B',
                     options=dict(maxiter=maxiter, ftol=1e-12))
        if r.fun < best:
            best, bx = float(r.fun), r.x
    return mu_s, best, ds, bx


def mu_curve_tight(H, blocks, warm=True, maxiter=120, refine=8):
    """Both bounds over a frequency grid.  H: array of FRF matrices.

    The full-D optimisation costs about a second per frequency, and only the
    PEAK is reported, so it is applied to the `refine` frequencies where the
    cheap scalar bound is largest.  Everywhere else the scalar value is kept,
    and since mu_full <= mu_scalar pointwise, the max of what is kept is still
    a valid upper bound of the peak -- never an under-estimate.
    """
    H = np.asarray(H)
    n = len(H)
    ms = np.empty(n)
    d0 = None
    for k in range(n):
        ms[k], d0 = mu_upper_bound(H[k], blocks, d0)
    if not _has_repeated(blocks):
        return ms, ms.copy()
    mf = ms.copy()
    order = np.argsort(ms)[::-1][:max(1, int(refine))]
    xw = None
    for k in np.sort(order):
        _, mf[k], _, xw = mu_bound(H[k], blocks, None, maxiter, xw)
    return ms, mf
