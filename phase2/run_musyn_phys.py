"""
run_musyn_phys.py — mu-synthesis on the PHYSICS-BASED uncertainty set.
=======================================================================
Same method as run_musyn.py (Eqs. 26-29 of Du et al. 2024), same weight grid,
same objective, same constraints, same plant, actuator and sensor.  Only the
uncertainty description changes, and it changes in three separable ways:

    mu_phys       one parameter for M/C/K, eta one-sided, W rank one
    mu_phys_sym   the same, but eta symmetric about the un-machined plate
                  -> isolates the ONE-SIDEDNESS
    mu_phys_ind   the same, but M, C and K get independent parameters
                  -> isolates the CORRELATION

Together with mu_paper (Eqs. 22-25 as published) and mu_exact (the same box,
corrected envelope and physics-sized bounds), this measures where the gain of a
correctly shaped uncertainty set comes from, instead of asserting it.

    python run_musyn_phys.py
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import musyn
from plate_model import build_plate
from objective import evaluate

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_musyn_phys.txt'), 'w')

VARIANTS = [('mu_phys', dict()),
            ('mu_phys_sym', dict(one_sided=False)),
            ('mu_phys_ind', dict(correlated=False))]


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def search(plate, tag, set_kw, grid):
    best = dict(J=-np.inf)
    best_feas = dict(J=-np.inf)
    n_try = 0
    for kf, ku, fcf in grid:
        n_try += 1
        t0 = time.time()
        try:
            import weights as W
            W.W_PF_DEF = dict(k=kf, fc_hz=fcf, M=250.0)
            W.W_PU_DEF = dict(k=ku, fc_hz=2500.0, M=60.0)
            r = musyn.design_phys(plate, n_iter=2, reduce_to=None,
                                  verbose=False, **set_kw)
            J, info = evaluate(plate, r['ss'], detail=True)
        except Exception as e:                                # noqa: BLE001
            log(f'    k_f={kf:.3g} k_u={ku:.3g} fc={fcf:.0f} -> failed '
                f'({type(e).__name__}: {e})')
            continue
        feas = bool(info['feasible'])
        log(f'    k_f={kf:.3g} k_u={ku:.3g} fc={fcf:.0f} -> mu={r["mu"]:.3f} '
            f'order={r["order"]:2d} J={J:+.5f} Ms={info["Ms"]:.3f} '
            f'V={info["V"]:.0f} maxRe={info["max_re"]:.1f} '
            f'[{time.time()-t0:.0f}s] {"OK" if feas else "constraint violated"}')
        cand = dict(J=J, ss=r['ss'], mu=r['mu'], gamma=r['gamma'],
                    order=r['order'], kf=kf, ku=ku, fc=fcf, info=info,
                    feasible=feas, mus=r['mus'], f=r['f'])
        if J > best['J']:
            best = dict(cand)
        if feas and J > best_feas['J']:
            best_feas = dict(cand)
    pick = best_feas if 'ss' in best_feas else best
    pick['n_try'] = n_try
    pick['tag'] = tag
    pick['any_feasible'] = 'ss' in best_feas
    return pick


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    grid = list(itertools.product([1e5, 3e5, 1e6, 3e6],
                                  [1.7e-3, 5e-3, 1.7e-2, 6e-2, 2e-1],
                                  [800.0, 1500.0, 3000.0]))
    log('=' * 74)
    log('MU-SYNTHESIS ON THE PHYSICS-BASED UNCERTAINTY SET')
    log('=' * 74)
    log(f'  weight grid: {len(grid)} trials per variant, identical to '
        f'run_musyn.py and scored by the same objective')

    out = {}
    for tag, kw in VARIANTS:
        log(f'\n  [{tag}]  {kw if kw else "one-sided, correlated, rank one"}')
        r = search(plate, tag, kw, grid)
        if 'ss' not in r:
            log(f'  {tag}: no stabilising design in {r["n_try"]} trials')
            continue
        log(f'\n  {tag}: J = {r["J"]:+.5f}, mu = {r["mu"]:.3f}, '
            f'order {r["order"]}, weights k_f={r["kf"]:.3g} k_u={r["ku"]:.3g} '
            f'fc={r["fc"]:.0f}, {r["n_try"]} trials, constraints '
            f'{"met" if r.get("any_feasible") else "NEVER met"}')
        out.update({f'{tag}_ss{i}': np.asarray(r['ss'][i], float)
                    for i in range(4)})
        out[f'{tag}_mu'] = r['mu']
        out[f'{tag}_J'] = r['J']
        out[f'{tag}_np'] = 4
        out[f'{tag}_mus'] = r['mus']
        out[f'{tag}_f'] = r['f']
        out[f'{tag}_w'] = np.array([r['kf'], r['ku'], r['fc']])
        np.savez(os.path.join(OUT, 'musyn_phys.npz'), **out)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/musyn_phys.npz')


if __name__ == '__main__':
    main()
