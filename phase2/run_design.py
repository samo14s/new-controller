"""
run_design.py — PHASES 6 and 8: design every controller under one protocol.
===========================================================================
    python run_design.py            (about 20 min)
Writes results/controllers_phase2.npz and results/log_design_phase2.txt
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from design import Design, optimise
from objective import evaluate
from uncertainty import RemovalFamily
from milling_dynamics import alpha4_average

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_design_phase2.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    abar4 = alpha4_average(C.RPM_S, C.AP_S, plate.hp, C.AE)
    alpha40 = C.SIGN * 1.6 * abar4
    removal = RemovalFamily(n=C.N_MODES_DESIGN)

    log('=' * 74)
    log('PHASE 6 + 8 - CONTROLLER DESIGN, ONE PROTOCOL FOR ALL')
    log('=' * 74)
    log(f'  plate      : patch {C.PATCH_SIDE}, sensor upper corner, sign {C.SIGN:+.0f}')
    log(f'  design pt  : {C.RPM_S} rpm, a_e = {C.AE*1e3} mm, '
        f'probe depths {[round(a*1e3,2) for a in C.AP_PROBE]} mm')
    log(f'  evaluation : Floquet on the {C.N_MODES}-mode model, '
        f'positions {C.POSITIONS_DESIGN}')
    log(f'  constraints: Ms <= {C.MS_MAX}, effort <= {C.V_PER_N} V/N, '
        f'nominal poles <= -1 1/s')
    log(f'  optimiser  : PSO {C.OPT["n_particles"]}x{C.OPT["n_iter"]}, '
        f'seeds {C.OPT["seeds"]}, identical for every structure')
    log(f'  alpha40    : {alpha40:.1f} N/m')

    out = {}
    for kind in ('pid', 'lqr', 'smc', 'proposed'):
        log(f'\n--- {kind.upper()} ' + '-' * (68 - len(kind)))
        d = Design(kind, plate, alpha40=alpha40, removal=removal)
        t = time.time()
        r = optimise(d)
        J, info = evaluate(plate, r['ss'], pd=r['pd'], detail=True)
        log(f'  J = {r["J"]:+.5f}   order {r["order"]}   '
            f'{r["n_params"]} parameters   [{time.time()-t:.0f}s]')
        log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
            f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
        log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                         for k, v in r['params'].items()))
        if r['pd'] is not None:
            log(f'  delayed PD : K_Pp = {r["pd"][0]:.4g} V/m, '
                f'K_Pd = {r["pd"][1]:.4g} V s/m')
        out[kind] = r

    np.savez(os.path.join(OUT, 'controllers_phase2.npz'),
             **{f'{k}_{f}': np.asarray(v[f], dtype=object)
                for k, v in out.items()
                for f in ('x', 'J', 'order', 'n_params')},
             **{f'{k}_ss{i}': np.asarray(v['ss'][i], float)
                for k, v in out.items() for i in range(4)},
             **{f'{k}_pd': np.asarray(v['pd'] if v['pd'] is not None
                                      else [0.0, 0.0], float)
                for k, v in out.items()},
             **{f'{k}_params': np.asarray(list(v['params'].values()), float)
                for k, v in out.items()},
             **{f'{k}_pnames': np.asarray(list(v['params'].keys()))
                for k, v in out.items()})
    log(f'\ntotal {time.time()-t0:.0f}s -> results/controllers_phase2.npz')


if __name__ == '__main__':
    main()
