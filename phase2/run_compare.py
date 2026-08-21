"""
run_compare.py — PHASES 12, 13, 14: the comparison.
====================================================
    python run_compare.py [scenarios]     e.g.  python run_compare.py 1 2
Writes results/compare_phase2.npz and results/log_compare_phase2.txt
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import scenarios as S
from plate_model import build_plate
from uncertainty import RemovalFamily
from run_certify import load
import controllers as K
from milling_dynamics import alpha4_average

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_compare_phase2.txt'), 'a')
ORDER = ('open', 'pid', 'lqr', 'smc', 'mu_paper', 'mu_exact', 'proposed')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def controllers(plate):
    c = load(plate)
    c['open'] = dict(ss=None, pd=None, n_params=0)
    return {k: c[k] for k in ORDER if k in c}


def scheduled_proposed(plate, eta_hat):
    """Rebuild PB-RAC at the removal state eta_hat, same optimised parameters.

    This is the adaptive half of the proposed controller: the observer and the
    feedback are re-formed on the model the physics-based set says the plate has
    actually become.  The other four structures have no such mechanism, which is
    exactly the structural difference under test -- so they are left untouched,
    and the unscheduled PB-RAC is reported alongside to show where the gain
    comes from.
    """
    d = np.load(os.path.join(OUT, 'controllers_phase2.npz'), allow_pickle=True)
    names = [str(x) for x in d['proposed_pnames']]
    vals = np.asarray(d['proposed_params'], float)
    u = dict(zip(names, vals))
    abar4 = alpha4_average(C.RPM_S, C.AP_S, plate.hp, C.AE)
    rem = RemovalFamily(n=C.N_MODES_DESIGN)
    ss, pd, _ = K.build('proposed', plate, u, alpha40=C.SIGN * 1.6 * abar4,
                        removal=rem, eta_hat=float(eta_hat))
    return dict(ss=ss, pd=pd, n_params=7)


def _lim(plate, c, rpm=None, mode_scale=None, zeta_scale=1.0, m=None):
    return S.floquet_limits(plate, c['ss'], c['pd'],
                            C.RPM_S if rpm is None else rpm,
                            mode_scale=mode_scale, zeta_scale=zeta_scale, m=m)


def scenario1(plate, ctrls, store):
    log('\n' + '=' * 74)
    log('SCENARIO 1 - NOMINAL   (Delta = 0, tau = tau_0, '
        f'{C.RPM_S} rpm, a_e = {C.AE*1e3} mm)')
    log('=' * 74)
    log(f'{"controller":<12}{"par":>4}{"a_p,lim min":>13}{"mean":>9}'
        f'{"A_max um":>10}{"A_rms um":>10}{"T_s s":>8}{"E_u V2s":>10}'
        f'{"u_max V":>9}')
    for k, c in ctrls.items():
        t = time.time()
        L = _lim(plate, c)
        tm = S.run_time(plate, c['ss'], c['pd'], C.RPM_S, C.AP_S)
        log(f'{k:<12}{c["n_params"]:>4}{L.min()*1e3:>13.4f}{L.mean()*1e3:>9.4f}'
            f'{tm["A_max"]:>10.3f}{tm["A_rms"]:>10.3f}{tm["T_s"]:>8.2f}'
            f'{tm["E_u"]:>10.3g}{tm["u_max"]:>9.2f}'
            + ('   DIVERGED' if tm['diverged'] else '')
            + f'   [{time.time()-t:.0f}s]')
        store[f'S1_{k}_limits'] = L
        for f, v in tm.items():
            if isinstance(v, (int, float, bool)):
                store[f'S1_{k}_{f}'] = float(v)


def scenario2(plate, ctrls, store):
    log('\n' + '=' * 74)
    log('SCENARIO 2 - MATERIAL REMOVAL   a_p,lim (mm), min over positions')
    log('=' * 74)
    rem = RemovalFamily(n=C.N_MODES_DESIGN)
    etas = np.linspace(0.0, C.ETA_MAX, 5)
    box = [(-0.10, 0.10), (-0.05, 0.05), (0.0, 0.0), (0.05, -0.05),
           (0.10, -0.10)]
    log('  (a) physics-based: eta = ' + ', '.join(f'{e:.4f}' for e in etas))
    log(f'{"controller":<12}' + ''.join(f'{f"eta={e:.3f}":>12}' for e in etas))
    items = list(ctrls.items())
    if 'proposed' in ctrls:
        items = items + [('proposed*', None)]
    for k, c in items:
        row = []
        for e in etas:
            ms = S.mode_scale_from_removal(rem, float(e), C.N_MODES)
            cc = scheduled_proposed(plate, e) if c is None else c
            row.append(_lim(plate, cc, mode_scale=ms).min() * 1e3)
        log(f'{k:<12}' + ''.join(f'{v:>12.4f}' for v in row)
            + ('   <- eta-scheduled' if c is None else ''))
        store[f'S2a_{k}'] = np.array(row)
    log("\n  (b) the paper's symmetric box: (dm, dk) = "
        + ', '.join(f'({a:+.2f},{b:+.2f})' for a, b in box))
    log(f'{"controller":<12}' + ''.join(f'{f"{a:+.0%}":>12}' for a, b in box))
    for k, c in ctrls.items():
        row = []
        for dm, dk in box:
            ms = S.paper_box_scale(dm, dk, C.N_MODES)
            row.append(_lim(plate, c, mode_scale=ms).min() * 1e3)
        log(f'{k:<12}' + ''.join(f'{v:>12.4f}' for v in row))
        store[f'S2b_{k}'] = np.array(row)


def scenario3(plate, ctrls, store):
    log('\n' + '=' * 74)
    log('SCENARIO 3 - STIFFNESS AND DAMPING   a_p,lim (mm), min over positions')
    log('=' * 74)
    cases = [('zeta x0.8', None, 0.8), ('zeta x1.2', None, 1.2),
             ('K -10%', np.full(C.N_MODES, np.sqrt(0.9)), 1.0),
             ('K +10%', np.full(C.N_MODES, np.sqrt(1.1)), 1.0),
             ('K -10% z0.8', np.full(C.N_MODES, np.sqrt(0.9)), 0.8)]
    log(f'{"controller":<12}' + ''.join(f'{n:>14}' for n, _, _ in cases))
    for k, c in ctrls.items():
        row = [_lim(plate, c, mode_scale=ms, zeta_scale=zs).min() * 1e3
               for _, ms, zs in cases]
        log(f'{k:<12}' + ''.join(f'{v:>14.4f}' for v in row))
        store[f'S3_{k}'] = np.array(row)


def scenario4(plate, ctrls, store):
    log('\n' + '=' * 74)
    log('SCENARIO 4 - DELAY   tau/tau_0 through the spindle speed')
    log('=' * 74)
    ratios = (1.0, 1.25, 1.5, 1.75, 2.0)
    rpms = [C.RPM_S / r for r in ratios]
    log(f'{"controller":<12}' + ''.join(
        f'{f"{r:.2f}t0":>12}' for r in ratios) + '     tau/tau_0 stable range')
    for k, c in ctrls.items():
        row = [_lim(plate, c, rpm=float(r)).min() * 1e3 for r in rpms]
        dm = S.delay_margin(plate, c['ss'], c['pd'], C.AP_S)
        log(f'{k:<12}' + ''.join(f'{v:>12.4f}' for v in row)
            + f'     [{dm["ratio_lo"]:.2f}, {dm["ratio_hi"]:.2f}]')
        store[f'S4_{k}'] = np.array(row)
        store[f'S4_{k}_ratio'] = np.array([dm['ratio_lo'], dm['ratio_hi']])


def main(which=(1, 2, 3, 4), only=None):
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    ctrls = controllers(plate)
    if only:
        ctrls = {k: v for k, v in ctrls.items() if k in only}
    store = {}
    log('\n' + '#' * 74)
    log(f'# PHASES 12-14   controllers: {list(ctrls)}')
    log('#' * 74)
    fns = {1: scenario1, 2: scenario2, 3: scenario3, 4: scenario4}
    for w in which:
        fns[w](plate, ctrls, store)
    p = os.path.join(OUT, 'compare_phase2.npz')
    old = dict(np.load(p)) if os.path.exists(p) else {}
    old.update(store)
    np.savez(p, **old)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/compare_phase2.npz')


if __name__ == '__main__':
    argv = sys.argv[1:]
    only = None
    if '--only' in argv:
        i = argv.index('--only')
        only = argv[i + 1].split(',')
        argv = argv[:i]
    args = [int(a) for a in argv] or [1, 2, 3, 4]
    main(args, only)
