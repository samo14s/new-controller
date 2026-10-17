"""Classic lobe sweep of the zero-order (Altintas-Budak) solution, written from
the derivation in README.md (Sec. 1), independent of src/zoa.py.

Characteristic equation (1-DOF):  1 - c a (1 - e^{-i w T}) Gcl(i w) = 0,
c = N Kt alpha_yy / (4 pi) < 0.
Realness of a gives  w T = eps + 2 k pi,  eps = mod(2 psi + pi, 2 pi),
psi = arg Gcl(i w), and then
    a = Re Gcl / (c (1 - cos eps) |Gcl|^2),
which is positive only where Re Gcl < 0.  Each lobe k gives a curve
(n_k(w), a(w)), n_k = 60 w / (N (eps + 2 k pi)).  The SLD at a speed is the
lower envelope of all lobes.
"""
import numpy as np
from scipy.optimize import brentq
import model as M

C_CUT = M.NT * M.KT * M.ALPHA_YY / (4 * np.pi)


def lobe_curves(g, zeta, ideal=False, f_lo=0.05, f_hi=1500.0, df=0.004):
    f = np.arange(f_lo, f_hi, df)
    w = 2 * np.pi * f
    G = M.Gcl(1j * w, g, zeta, ideal)
    psi = np.angle(G)
    eps = np.mod(2 * psi + np.pi, 2 * np.pi)
    a = G.real / (C_CUT * (1 - np.cos(eps)) * np.abs(G) ** 2)
    ok = G.real < 0
    return f, w, eps, a, ok


def envelope(rpms, g, zeta, ideal=False, kmax=60, curves=None, return_all=False,
             polish=True):
    """Lower envelope a_lim(n) [m] and chatter frequency [Hz] from the lobes.

    A segment between two consecutive frequency samples is used only if both
    have Re Gcl < 0 and eps does not wrap between them.  On each lobe the
    curve n_k(w) is cut into monotone pieces and read off by interpolation, so
    a lobe that folds back in speed contributes all its branches.
    """
    rpms = np.atleast_1d(np.asarray(rpms, float))
    f, w, eps, a, ok = curves if curves is not None else lobe_curves(g, zeta, ideal)
    seg_ok = ok[:-1] & ok[1:] & (np.abs(np.diff(eps)) < np.pi)
    best = np.full(rpms.shape, np.inf)
    fbest = np.full(rpms.shape, np.nan)
    kbest = np.full(rpms.shape, -1)
    roots = [[] for _ in rpms]
    nlo, nhi = rpms.min(), rpms.max()
    for k in range(kmax + 1):
        nk = 60 * w / (M.NT * (eps + 2 * k * np.pi))
        # quick reject: lobe entirely outside the speed range
        if not np.any(seg_ok & (np.minimum(nk[:-1], nk[1:]) <= nhi) & (np.maximum(nk[:-1], nk[1:]) >= nlo)):
            continue
        dn = np.diff(nk)
        # pieces: maximal runs of valid segments with one sign of dn
        mono_brk = np.r_[False, np.sign(dn[1:]) != np.sign(dn[:-1])]
        prev_ok = np.r_[False, seg_ok[:-1]]
        starts = np.flatnonzero(seg_ok & (mono_brk | ~prev_ok))
        bad = np.flatnonzero(~seg_ok)
        nxt_start = np.r_[starts[1:], seg_ok.size]
        nxt_bad = bad[np.minimum(np.searchsorted(bad, starts), bad.size - 1)] if bad.size else np.full(starts.shape, seg_ok.size)
        nxt_bad = np.where(nxt_bad > starts, nxt_bad, seg_ok.size)
        ends = np.minimum(nxt_start, nxt_bad)          # exclusive segment index
        for s0, s1 in zip(starts, ends):
            idx = np.arange(s0, s1 + 1)               # sample indices of the piece
            x = nk[idx]
            if x[0] > x[-1]:
                idx = idx[::-1]
                x = x[::-1]
            sel = (rpms >= x[0]) & (rpms <= x[-1])
            if not np.any(sel):
                continue
            ai = np.interp(rpms[sel], x, a[idx])
            fi = np.interp(rpms[sel], x, f[idx])
            pos = np.clip(np.searchsorted(x, rpms[sel]) - 1, 0, x.size - 2)
            for jj, aj, fj, pj in zip(np.flatnonzero(sel), ai, fi, pos):
                wa, wb = w[idx[pj]], w[idx[pj + 1]]
                roots[jj].append((aj, fj, k, min(wa, wb), max(wa, wb)))
                if aj < best[jj]:
                    best[jj], fbest[jj], kbest[jj] = aj, fj, k
    if polish:
        # Brent-refine every candidate within 5 % of the interpolated minimum:
        # n_k(w) - n has a sign change on the stored [wa, wb] bracket.
        for jj, n in enumerate(rpms):
            cands = [r for r in roots[jj] if r[0] <= 1.05 * best[jj]]
            newr = []
            for (aj, fj, k, wa, wb) in cands:
                fun = lambda ww: 60 * ww / (M.NT * (_eps(ww, g, zeta, ideal) + 2 * k * np.pi)) - n
                try:
                    wr = brentq(fun, wa, wb, xtol=1e-13, rtol=1e-14)
                except ValueError:
                    newr.append((aj, fj, k))
                    continue
                newr.append((_a(wr, g, zeta, ideal), wr / (2 * np.pi), k))
            if newr:
                aj, fj, k = min(newr)
                best[jj], fbest[jj], kbest[jj] = aj, fj, k
    if return_all:
        return best, fbest, kbest, roots
    return best, fbest, kbest


def _eps(w, g, zeta, ideal):
    G = M.Gcl(1j * w, g, zeta, ideal)
    return np.mod(2 * np.angle(G) + np.pi, 2 * np.pi)


def _a(w, g, zeta, ideal):
    G = M.Gcl(1j * w, g, zeta, ideal)
    return G.real / (C_CUT * (1 - np.cos(_eps(w, g, zeta, ideal))) * np.abs(G) ** 2)
