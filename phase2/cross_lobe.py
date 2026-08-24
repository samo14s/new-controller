"""cross_lobe.py — the crossing pair (a_p^inf, delta_max) of the
lobe-first champion, by the exact stage-5/6 machinery (certify2), so its
Table row is complete to the same standard as the benchmark's and the
r5 member's.

Same vertex family and bisections as run_psacrf.certify():
  * analyse: peak + tau_max over the full family (9 positions,
    3 etas, 2 zetas, xi in {0.5, 1});
  * depth_bisect: a_p^inf, the largest depth certified by the
    delay-free crossing argument over the light family;
  * margin_bisect: delta_max, the uniform parameter-inflation margin;
  * common-P LK at a_p^inf.

Appends to results/log_lobe1.txt; stores under ps_ac_rfa_l2 in
stage56.pkl and writes results/cross_l.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import certify2 as CF2
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

KIND = sys.argv[1] if len(sys.argv) > 1 else 'ps_ac_rfa_l2'

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


if __name__ == '__main__':
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    tau0 = 60.0 / (3 * C.RPM_S)
    etas = tuple(np.linspace(0.0, C.ETA_MAX, 3))
    zetas = (C.ZETA_LO, C.ZETA_HI)
    full = dict(n_pos=9, etas=etas, zetas=zetas, xis=(0.5, 1.0))
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    mk = load_controllers(plate, plant0)[KIND]
    c0 = mk(plant0)

    log('')
    log('=' * 78)
    log(f'CROSSING PAIR ({KIND})  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('  same vertex family and bisections as stage 5-6')
    log('=' * 78)
    t = time.time()
    tm, peak, ok = CF2.analyse(plant0, c0, **full)
    ratio = 'inf' if np.isinf(tm) else f'{tm / tau0:.2f}'
    log(f'  peak = {peak:.3f}   tau_max/tau0 = {ratio}   '
        f'[{time.time() - t:.0f}s]')

    def plant_of(ap):
        return ControlledPlant(plate, ap=ap)

    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    t = time.time()
    ap_inf = CF2.depth_bisect(plant_of, mk, n_iter=16, **light)
    log(f'  a_p^inf = {ap_inf * 1e3:.4f} mm   (mu-TDC 0.4364, r5 0.2841)'
        f'   [{time.time() - t:.0f}s]')
    t = time.time()
    dmax = CF2.margin_bisect(plant0, c0, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    log(f'  delta_max = {dmax:.3f}   (mu-TDC 1.543, r5 0.918)   '
        f'[{time.time() - t:.0f}s]')
    pl_lk = ControlledPlant(plate, ap=max(ap_inf, 1e-6))
    lk = CF2.lk_common_P(pl_lk, mk(pl_lk), n_pos=3,
                         etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    lk_txt = ('yes' if lk['feasible']
              else ('no' if lk.get('attempted', True) else 'n/a'))
    log(f'  common-P LK at a_p^inf: {lk_txt}')

    p = os.path.join(C.RESULTS, 'stage56.pkl')
    store = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
    store[KIND] = dict(peak=peak, tau_max=tm, tau_ratio=tm / tau0,
                       ap_inf=ap_inf, delta_max=dmax,
                       lk=bool(lk['feasible']),
                       lk_attempted=bool(lk.get('attempted', True)))
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    np.savez_compressed(os.path.join(C.RESULTS,
                                 f'cross_{KIND}.npz'),
                        peak=peak, tau_max=tm, ap_inf=ap_inf,
                        delta_max=dmax, lk=bool(lk['feasible']))
    log(f'  -> stage56.pkl [{KIND}], results/cross_{KIND}.npz')
    log('cross_lobe done.')
