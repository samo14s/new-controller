"""mu21_l.py — the full 21-node mixed-gate sweep for the lobe-first
champion, completing its G1 record to the same standard as the witness
and the r5 member (mu_real_21.npz).

Same judge, same grid, same nodes; the only addition is a 4-worker pool
over the nodes (each worker single-threaded), which changes nothing in
any number — every node is an independent judge_mixed call.

Appends to results/log_lobe1.txt; writes results/mu_real_l21.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers
from mu_real import judge_mixed
from robust_design import freq_grid

KIND = 'ps_ac_rfa_l2'
N_NODES = 21

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate, ap=C.AP_S)
ctrl = load_controllers(plate, plant)[KIND](plant)
F = freq_grid()
XS = np.linspace(0.0, plate.lp, N_NODES)


def node(i):
    x = float(XS[i])
    t0 = time.time()
    r = judge_mixed(plate, ctrl.at(x)[0], x, F)
    return i, r['mixed'], r['cx'], r['f_mixed'], time.time() - t0


if __name__ == '__main__':
    log('')
    log('=' * 78)
    log(f'FULL 21-NODE MIXED GATE SWEEP: {KIND}  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 78)
    t0 = time.time()
    mx = np.zeros(N_NODES)
    cx = np.zeros(N_NODES)
    with Pool(4) as pool:
        for i, m, c, fm, el in pool.imap_unordered(node, range(N_NODES)):
            mx[i], cx[i] = m, c
            log(f'  node {i:2d}  x = {XS[i] * 1e3:5.1f} mm:  complex '
                f'{c:7.3f}  ->  mixed {m:7.3f}  at {fm:6.0f} Hz'
                + ('' if m < 1 else '  >= 1') + f'   [{el:.0f}s]')
    log(f'  sup over the {N_NODES} nodes = {mx.max():.3f} (worst node '
        f'{int(np.argmax(mx))}, x = {XS[np.argmax(mx)] * 1e3:.1f} mm)  '
        + ('< 1  G1(real) MET ON EVERY NODE' if mx.max() < 1
           else '>= 1: NOT MET')
        + f'   [total {time.time() - t0:.0f}s]')
    np.savez_compressed(os.path.join(C.RESULTS, 'mu_real_l21.npz'),
                        x=XS, mixed=mx, cx=cx)
    log('-> results/mu_real_l21.npz')
    log('mu21_l done.')
