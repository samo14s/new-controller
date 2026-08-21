"""
run_stage3.py — STAGES 3 and 4: design the four controllers.
=============================================================
    python run_stage3.py          (about 25 min)
Writes results/stage3_controllers.npz and results/log_stage3.txt
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
from plate_model import build_plate
from plant_ss import ControlledPlant
from design2 import Design2, optimise
from eval2 import evaluate

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_stage3.txt'), 'w')
KINDS = ('fopid', 'lqg', 'mu_tdc', 'ps_ac', 'ps_ac_eta',
         'ps_ac_obs', 'ps_ac_full')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def load_mu():
    p = os.path.join(OUT, 'musyn_phase2.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if 'mu_paper_ss0' not in d.files:
        return None
    return tuple(d[f'mu_paper_ss{i}'] for i in range(4))


def main(kinds=KINDS):
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    ss_mu = load_mu()

    log('=' * 74)
    log('STAGES 3-4 - CONTROLLER DESIGN UNDER ONE PROTOCOL')
    log('=' * 74)
    log(f'  plant      : Du 2024, patch {C.PATCH_SIDE}, sensor upper corner, '
        f'sign {C.SIGN:+.0f}')
    log(f'  operating  : {C.RPM_S} rpm, a_e = {C.AE*1e3} mm, '
        f'f_z = {C.FZ*1e3} mm/tooth, down milling')
    log(f'  probes     : a_p = {[round(a*1e3,2) for a in C.AP_PROBE]} mm, '
        f'positions {C.POSITIONS_DESIGN}')
    log(f'  evaluation : Floquet on the {C.N_MODES}-mode model')
    log(f'  constraints: Ms <= {C.MS_MAX}, effort <= {C.V_PER_N} V/N, '
        f'nominal poles <= -1 1/s')
    log(f'  optimiser  : PSO {C.OPT["n_particles"]}x{C.OPT["n_iter"]}, '
        f'seeds {C.OPT["seeds"]} -- identical for every structure')
    log(f'  mu-synthesis controller loaded: {ss_mu is not None}')

    p_store = os.path.join(OUT, 'stage3_controllers.pkl')
    store = (pickle.load(open(p_store, 'rb'))
             if os.path.exists(p_store) else {})
    for kind in kinds:
        if kind == 'mu_tdc' and ss_mu is None:
            log(f'\n--- {kind.upper()}: skipped, no mu controller on disk')
            continue
        log(f'\n--- {kind.upper()} ' + '-' * (68 - len(kind)))
        d = Design2(kind, plant, plate, ss_mu)
        t = time.time()
        r = optimise(d)
        J, info = evaluate(plate, r['ctrl'], detail=True)
        log(f'  J = {r["J"]:+.5f}   order {r["order"]}   '
            f'{r["n_params"]} parameters   '
            f'{"SCHEDULED" if r["ctrl"].scheduled else "fixed"}   '
            f'[{time.time()-t:.0f}s]')
        log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
            f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
        log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                         for k, v in r['params'].items()))
        store[kind] = dict(x=r['x'], J=r['J'], params=r['params'],
                           n_params=r['n_params'], order=r['order'],
                           Ms=info['Ms'], V=info['V'])

    with open(p_store, 'wb') as f:
        pickle.dump(store, f)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/stage3_controllers.pkl')


if __name__ == '__main__':
    main(sys.argv[1:] or KINDS)
