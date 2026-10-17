"""Self-adaptive differential evolution (SaDE; Qin, Huang & Suganthan 2009).

The paper cites [56] for SADE and describes: NP trial gains drawn over the
search interval, DE mutation/crossover from a strategy pool, greedy selection,
adaptation of (F, CR) and of the strategy probabilities from the success
record of a learning period, and termination on a stagnation window or a
generation cap.  This module implements exactly that loop for maximisation.
"""
import numpy as np

STRATEGIES = ("rand/1/bin", "rand-to-best/2/bin", "rand/2/bin", "current-to-rand/1")


def sade(fun, lo, hi, np_=20, max_gen=40, lp=5, window=8, tol=1e-6, seed=0,
         init=None):
    """Maximise fun over the box [lo, hi].

    Returns dict(x, f, history, n_eval, gens, probs).
    lp      learning period (generations) for strategy / CR adaptation
    window  stagnation window: stop when the best fitness improved by less
            than tol over the last `window` generations
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.atleast_1d(lo).astype(float), np.atleast_1d(hi).astype(float)
    d = lo.size
    K = len(STRATEGIES)
    pop = lo + rng.random((np_, d)) * (hi - lo)
    if init is not None:
        init = np.atleast_2d(init)
        pop[: len(init)] = init
    fit = np.array([fun(x) for x in pop])
    n_eval = np_
    probs = np.full(K, 1.0 / K)
    crm = np.full(K, 0.5)
    ns_hist, nf_hist, cr_hist = [], [], []
    best_hist = [fit.max()]

    for gen in range(1, max_gen + 1):
        ib = int(np.argmax(fit))
        strat = rng.choice(K, size=np_, p=probs)
        F = np.clip(rng.normal(0.5, 0.3, np_), 0.05, 2.0)
        CR = np.clip(rng.normal(crm[strat], 0.1), 0.0, 1.0)
        ns, nf = np.zeros(K), np.zeros(K)
        cr_ok = [[] for _ in range(K)]
        new_pop, new_fit = pop.copy(), fit.copy()
        for i in range(np_):
            idx = [j for j in range(np_) if j != i]
            r = rng.choice(idx, 5, replace=False)
            x, b = pop[i], pop[ib]
            k = strat[i]
            if k == 0:
                v = pop[r[0]] + F[i] * (pop[r[1]] - pop[r[2]])
            elif k == 1:
                v = x + F[i] * (b - x) + F[i] * (pop[r[0]] - pop[r[1]]) \
                    + F[i] * (pop[r[2]] - pop[r[3]])
            elif k == 2:
                v = pop[r[0]] + F[i] * (pop[r[1]] - pop[r[2]]) \
                    + F[i] * (pop[r[3]] - pop[r[4]])
            else:
                kk = rng.random()
                v = x + kk * (pop[r[0]] - x) + F[i] * (pop[r[1]] - pop[r[2]])
            if k != 3:                       # binomial crossover
                mask = rng.random(d) <= CR[i]
                mask[rng.integers(d)] = True
                u = np.where(mask, v, x)
            else:
                u = v
            out = (u < lo) | (u > hi)        # re-sample violated components
            u[out] = lo[out] + rng.random(out.sum()) * (hi[out] - lo[out])
            fu = fun(u)
            n_eval += 1
            if fu >= fit[i]:
                new_pop[i], new_fit[i] = u, fu
                ns[k] += 1
                cr_ok[k].append(CR[i])
            else:
                nf[k] += 1
        pop, fit = new_pop, new_fit
        ns_hist.append(ns)
        nf_hist.append(nf)
        cr_hist.append(cr_ok)
        if len(ns_hist) > lp:
            ns_hist.pop(0)
            nf_hist.pop(0)
            cr_hist.pop(0)
        if gen >= lp:
            S = np.sum(ns_hist, 0)
            Fl = np.sum(nf_hist, 0)
            rate = S / np.maximum(S + Fl, 1) + 0.01
            probs = rate / rate.sum()
            for k in range(K):
                vals = [c for g in cr_hist for c in g[k]]
                if vals:
                    crm[k] = float(np.median(vals))
        best_hist.append(fit.max())
        # stagnation stop; a population that has not yet found a feasible
        # member (best = 0) is not converged and keeps searching
        if (gen >= window and best_hist[-1] > 0
                and best_hist[-1] - best_hist[-1 - window] <= tol):
            break
    ib = int(np.argmax(fit))
    return dict(x=pop[ib].copy(), f=float(fit[ib]), history=np.array(best_hist),
                n_eval=n_eval, gens=gen, probs=probs)
