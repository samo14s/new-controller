"""
run_stage56.py — STAGES 5 and 6.
=================================
Stage 5: a stability condition valid over the admissible milling-position
domain, for the delayed closed loop.
Stage 6: Delta_max and tau_max.

    python run_stage56.py
Writes results/stage56.pkl and results/log_stage56.txt
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import certify2 as CF2
import ctrl2 as K2
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_stage56.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def main(n_pos=9, n_eta=3):
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    tau0 = 60.0 / (3 * C.RPM_S)
    etas = tuple(np.linspace(0.0, C.ETA_MAX, n_eta))
    zetas = (C.ZETA_LO, C.ZETA_HI)
    base = dict(n_pos=n_pos, etas=etas, zetas=zetas, xis=(0.5, 1.0))

    log('=' * 78)
    log('STAGES 5-6 - STABILITY OVER THE POSITION DOMAIN, AND MARGINS')
    log('=' * 78)
    log(f'  position domain : x_P in [0, {plate.lp*1e3:.0f}] mm, {n_pos} points')
    log(f'  uncertainty     : alpha4 in [{C.ALPHA_LO}, {C.ALPHA_HI}] abar4, '
        f'eta in [0, {C.ETA_MAX}], zeta x [{C.ZETA_LO}, {C.ZETA_HI}], '
        f'xi in {{0.5, 1}}')
    log(f'  tau_0           = {tau0*1e3:.4f} ms at {C.RPM_S} rpm')
    log('')
    log('  peak      sup_omega rho((j omega I - A_cl)^-1 A_d,cl) over the family;')
    log('            below 1 the loop is stable for EVERY delay')
    log('  tau_max   first delay at which a characteristic root reaches the')
    log('            imaginary axis; inf when peak < 1')
    log('  a_p^inf   largest depth at which peak < 1 everywhere')
    log('  delta_max largest inflation of the uncertainty set keeping peak < 1')
    log('  LK        common-P Lyapunov-Krasovskii feasibility at a_p^inf:')
    log('            also covers arbitrarily fast motion of x_P, eta, alpha4')
    log('')
    log(f'{"controller":<12}{"peak":>8}{"tau_max/tau0":>14}{"a_p^inf mm":>12}'
        f'{"delta_max":>11}{"LK":>6}')

    plant0 = ControlledPlant(plate, ap=C.AP_S)
    ctrls = load_controllers(plate, plant0)
    res = {}
    for name, mk in ctrls.items():
        t = time.time()
        c0 = mk(plant0)
        tm, peak, ok = CF2.analyse(plant0, c0, **base)
        ratio = 'inf' if np.isinf(tm) else f'{tm/tau0:.2f}'

        def plant_of(ap):
            return ControlledPlant(plate, ap=ap)

        # the headline peak/tau_max use the FULL vertex family; the two
        # bisections use a reduced one, because they call the analysis 14-16
        # times each.  The reduced family is a subset, so the depth it certifies
        # is an upper bound on the full-family answer -- stated, not hidden.
        light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
        ap_inf = CF2.depth_bisect(plant_of, mk, n_iter=16, **light)
        dmax = CF2.margin_bisect(plant0, c0, n_iter=14,
                                 base_kw=dict(n_pos=5, xis=(1.0,)))
        pl_lk = ControlledPlant(plate, ap=max(ap_inf, 1e-6))
        lk = CF2.lk_common_P(pl_lk, mk(pl_lk), n_pos=3,
                             etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
        log(f'{name:<12}{peak:>8.3f}{ratio:>14}{ap_inf*1e3:>12.4f}'
            f'{dmax:>11.2f}{("yes" if lk["feasible"] else "no"):>6}'
            f'   [{time.time()-t:.0f}s]')
        res[name] = dict(peak=peak, tau_max=tm, tau_ratio=tm / tau0,
                         ap_inf=ap_inf, delta_max=dmax,
                         lk=bool(lk['feasible']))

    with open(os.path.join(OUT, 'stage56.pkl'), 'wb') as f:
        pickle.dump(res, f)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/stage56.pkl')
    return res


if __name__ == '__main__':
    main()
