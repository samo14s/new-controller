"""
run_musyn.py — controller 4: mu-synthesis, the paper's own method.
==================================================================
Two variants, same machinery, different uncertainty description:

  mu_paper : Eqs. (22)-(25) exactly as published -- Eq. (25) without its cross
             terms, and the flat 10 % / 20 % box of Section 3.2;
  mu_exact : the same D-K iteration on the CORRECTED description -- Eq. (25)
             with the two cross terms restored (alpha_coupling='exact') and the
             physics-based bounds of docs/03 in place of the flat box.

The gap between the two is the cost of the algebraic slip, measured on the
paper's own synthesis method.  The synthesis weights, which the paper does not
publish, are searched on a small grid scored by the SAME objective as every
other controller, and the number of trials is reported.

    python run_musyn.py            (about 45 min)
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
LOG = open(os.path.join(OUT, 'log_musyn_phase2.txt'), 'w')

# physics-based bounds from docs/03 (worst case over eta in [0, eta_max])
PHYS = dict(mass_pert=0.35, stiff_pert=0.064, damp_pert=0.41)


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def search(plate, tag, alpha_coupling, perts, grid):
    best = dict(J=-np.inf)
    n_try = 0
    for kf, ku in grid:
        n_try += 1
        try:
            import weights as W
            W.W_PF_DEF = dict(k=kf, fc_hz=1500.0, M=250.0)
            W.W_PU_DEF = dict(k=ku, fc_hz=2500.0, M=60.0)
            r = musyn.design(plate, alpha_coupling=alpha_coupling, n_iter=2,
                             reduce_to=None, verbose=False, **perts)
            J, info = evaluate(plate, r['ss'], detail=True)
        except Exception as e:                                # noqa: BLE001
            log(f'    k_f={kf:.3g} k_u={ku:.3g} -> failed ({type(e).__name__})')
            continue
        log(f'    k_f={kf:.3g} k_u={ku:.3g} -> mu={r["mu"]:.3f} '
            f'order={r["order"]:2d} J={J:+.5f} Ms={info["Ms"]:.3f} '
            f'V={info["V"]:.0f} maxRe={info["max_re"]:.1f}')
        if J > best['J']:
            best = dict(J=J, ss=r['ss'], mu=r['mu'], gamma=r['gamma'],
                        order=r['order'], kf=kf, ku=ku, info=info)
    best['n_try'] = n_try
    best['tag'] = tag
    return best


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    grid = list(itertools.product([3e5, 1e6, 3e6], [5e-3, 1.7e-2, 6e-2]))
    log('=' * 74)
    log('CONTROLLER 4 - MU-SYNTHESIS (D-K iteration, Eqs. 26-29)')
    log('=' * 74)
    log(f'  weight grid: {len(grid)} trials per variant, scored by the same '
        f'objective as the other controllers')

    out = {}
    log('\n  [A] uncertainty set exactly as published')
    a = search(plate, 'mu_paper', 'paper',
               dict(mass_pert=0.10, stiff_pert=0.10, damp_pert=0.20), grid)
    log('\n  [B] same method, corrected envelope + physics-based bounds')
    b = search(plate, 'mu_exact', 'exact', PHYS, grid)

    for r in (a, b):
        if 'ss' not in r:
            log(f'  {r["tag"]}: no stabilising design found in '
                f'{r["n_try"]} trials')
            continue
        log(f'\n  {r["tag"]}: J = {r["J"]:+.5f}, mu = {r["mu"]:.3f}, '
            f'order {r["order"]}, weights k_f={r["kf"]:.3g} k_u={r["ku"]:.3g}, '
            f'{r["n_try"]} trials')
        out.update({f'{r["tag"]}_ss{i}': np.asarray(r['ss'][i], float)
                    for i in range(4)})
        out[f'{r["tag"]}_mu'] = r['mu']
        out[f'{r["tag"]}_J'] = r['J']
        out[f'{r["tag"]}_np'] = 4
    np.savez(os.path.join(OUT, 'musyn_phase2.npz'), **out)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/musyn_phase2.npz')


if __name__ == '__main__':
    main()
