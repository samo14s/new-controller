"""Independent textbook SaDE (Qin, Huang & Suganthan, IEEE TEVC 13(2), 2009).

Written from the algorithm description for comparison with src/sade.py; it shares
no code with it.  Minimisation is the native form; `maximise=True` wraps it.
Differences from src/sade.py that are deliberate (they follow the 2009 paper):
  * strategies assigned by stochastic universal sampling (SUS),
  * F ~ N(0.5, 0.3) without clipping,
  * CR ~ N(CRm_k, 0.1) regenerated until it lies in [0, 1],
  * default learning period LP = 50 generations,
  * no stagnation stop (fixed generation budget), unless window is given.
Common to both: four strategies, success/failure memories over LP generations,
p_k = S_k / sum S_k with S_k = ns/(ns+nf) + 0.01, CRm_k = median(CRMemory_k),
greedy (<=) selection, out-of-range components re-sampled uniformly.
"""
import numpy as np


def _sus(rng, probs, n):
    """Stochastic universal sampling of n strategy labels."""
    c = np.cumsum(probs)
    c[-1] = 1.0
    pts = (rng.random() + np.arange(n)) / n
    lab = np.searchsorted(c, pts, side="right")
    rng.shuffle(lab)
    return np.minimum(lab, len(probs) - 1)


def ref_sade(fun, lo, hi, np_=20, max_gen=100, lp=50, seed=0, maximise=False,
             window=None, tol=0.0, init=None):
    rng = np.random.default_rng(seed)
    lo, hi = np.atleast_1d(lo).astype(float), np.atleast_1d(hi).astype(float)
    d = lo.size
    sgn = -1.0 if maximise else 1.0
    f = (lambda x: sgn * fun(x))
    pop = lo + rng.random((np_, d)) * (hi - lo)
    if init is not None:
        init = np.atleast_2d(init)
        pop[: len(init)] = init
    val = np.array([f(x) for x in pop])
    nfe = np_
    K = 4
    p = np.full(K, 0.25)
    crm = np.full(K, 0.5)
    mem_s, mem_f, mem_cr = [], [], []
    hist = [val.min()]
    g = 0
    for g in range(1, max_gen + 1):
        ib = int(np.argmin(val))
        lab = _sus(rng, p, np_)
        F = rng.normal(0.5, 0.3, np_)
        CR = rng.normal(crm[lab], 0.1)
        bad = (CR < 0) | (CR > 1)
        while bad.any():
            CR[bad] = rng.normal(crm[lab[bad]], 0.1)
            bad = (CR < 0) | (CR > 1)
        ns, nf = np.zeros(K), np.zeros(K)
        crs = [[] for _ in range(K)]
        newpop, newval = pop.copy(), val.copy()
        for i in range(np_):
            others = np.delete(np.arange(np_), i)
            r1, r2, r3, r4, r5 = rng.permutation(others)[:5]
            x = pop[i]
            k = lab[i]
            if k == 0:      # DE/rand/1/bin
                v = pop[r1] + F[i] * (pop[r2] - pop[r3])
            elif k == 1:    # DE/rand-to-best/2/bin
                v = x + F[i] * (pop[ib] - x) + F[i] * (pop[r1] - pop[r2]) \
                    + F[i] * (pop[r3] - pop[r4])
            elif k == 2:    # DE/rand/2/bin
                v = pop[r1] + F[i] * (pop[r2] - pop[r3]) + F[i] * (pop[r4] - pop[r5])
            else:           # DE/current-to-rand/1 (no crossover)
                Kc = rng.random()
                v = x + Kc * (pop[r1] - x) + F[i] * (pop[r2] - pop[r3])
            if k < 3:
                jr = rng.integers(d)
                m = rng.random(d) < CR[i]
                m[jr] = True
                u = np.where(m, v, x)
            else:
                u = v.copy()
            o = (u < lo) | (u > hi)
            if o.any():
                u[o] = lo[o] + rng.random(o.sum()) * (hi[o] - lo[o])
            fu = f(u)
            nfe += 1
            if fu <= val[i]:
                newpop[i], newval[i] = u, fu
                ns[k] += 1
                crs[k].append(CR[i])
            else:
                nf[k] += 1
        pop, val = newpop, newval
        mem_s.append(ns), mem_f.append(nf), mem_cr.append(crs)
        if len(mem_s) > lp:
            mem_s.pop(0), mem_f.pop(0), mem_cr.pop(0)
        if g >= lp:
            s, fl = np.sum(mem_s, 0), np.sum(mem_f, 0)
            S = np.where(s + fl > 0, s / np.maximum(s + fl, 1), 0.0) + 0.01
            p = S / S.sum()
            for k in range(K):
                allcr = [c for gen in mem_cr for c in gen[k]]
                if allcr:
                    crm[k] = float(np.median(allcr))
        hist.append(val.min())
        if window and g >= window and hist[-1 - window] - hist[-1] <= tol:
            break
    ib = int(np.argmin(val))
    return dict(x=pop[ib].copy(), f=float(sgn * val[ib]), n_eval=nfe, gens=g, probs=p)
