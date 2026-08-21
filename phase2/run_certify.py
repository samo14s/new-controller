"""
run_certify.py — PHASES 9, 10, 11.
==================================
For each designed controller:

  Phase 9   delay-independent Lyapunov-Krasovskii certificate over the
            physics-based uncertainty set -> the depth a_p^LK below which no
            spindle speed and no parameter value in the set can destabilise
            the closed loop;
  Phase 10  delta_max, the largest inflation of that set still certified;
  Phase 11  the delay margin, from an exact Floquet sweep over spindle speed
            at the design depth (the LK certificate already covers every delay,
            so the sweep measures how much the EXACT limit tolerates).

    python run_certify.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import certify as CF
from plate_model import build_plate
from uncertainty import UncertaintySet
from scenarios import delay_margin

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_certify_phase2.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def load(plate):
    d = np.load(os.path.join(OUT, 'controllers_phase2.npz'), allow_pickle=True)
    out = {}
    for k in ('pid', 'lqr', 'smc', 'proposed'):
        ss = tuple(d[f'{k}_ss{i}'] for i in range(4))
        pd = d[f'{k}_pd']
        out[k] = dict(ss=ss, pd=None if np.allclose(pd, 0) else tuple(pd),
                      n_params=int(d[f'{k}_n_params']))
    p = os.path.join(OUT, 'musyn_phase2.npz')
    if os.path.exists(p):
        m = np.load(p, allow_pickle=True)
        for tag in ('mu_paper', 'mu_exact'):
            if f'{tag}_ss0' in m.files:
                npar = int(m[f'{tag}_np']) if f'{tag}_np' in m.files else 4
                out[tag] = dict(ss=tuple(m[f'{tag}_ss{i}'] for i in range(4)),
                                pd=None, n_params=npar)
    return out


def main(kinds=None, n_eta=3, n_x=9):
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    ctrls = load(plate)
    kinds = list(ctrls.keys()) if kinds is None else kinds

    def uset_of(ap):
        return UncertaintySet(plate, rpm=C.RPM_S, ap=ap, n=C.N_MODES_DESIGN,
                              eta_max=C.ETA_MAX)

    log('=' * 74)
    log('PHASES 9-11 - DELAY-EXPLICIT CERTIFICATE AND MARGINS')
    log('=' * 74)
    U0 = uset_of(C.AP_S)
    V0 = U0.vertices(1.0, n_eta, n_x)
    log(f'  uncertainty set : eta in [0, {C.ETA_MAX}], xi in [0, 1], '
        f'x in [0, l_P], alpha4 in [{C.ALPHA_LO}, {C.ALPHA_HI}] abar4, '
        f'zeta x [{C.ZETA_LO}, {C.ZETA_HI}]')
    log(f'  polytope        : {len(V0)} vertices')
    log('')
    log('  a_p^DI     largest depth at which EVERY vertex is stable for EVERY')
    log('             delay (exact frequency test) - the lobe minimum as a theorem')
    log('  delta_max  largest inflation of the set still delay-independently')
    log('             stable at the nominal depth')
    log('  tau range  widest connected tau/tau_0 interval around the design')
    log('             point keeping the EXACT (Floquet) limit stable at a_p')
    log('  LK         quadratic Lyapunov-Krasovskii certificate at a_p^LK: it')
    log('             also covers arbitrarily fast parameter variation, at the')
    log('             price of quadratic conservatism')
    log('')
    log(f'{"controller":<12}{"a_p^DI mm":>11}{"delta_max":>11}'
        f'{"tau/tau_0 range":>20}{"peak rho":>10}')

    res = {}
    for k in kinds:
        c = ctrls[k]
        t = time.time()
        _, peak = CF.di_stable_set(U0, c['ss'], c['pd'], n_eta=n_eta, n_x=n_x)
        ap_di = CF.di_depth(uset_of, c['ss'], c['pd'], n_eta=n_eta, n_x=n_x)
        dmax = CF.di_margin(U0, c['ss'], c['pd'], n_eta=n_eta, n_x=n_x)
        dm = delay_margin(plate, c['ss'], c['pd'], C.AP_S)
        rng = f'[{dm["ratio_lo"]:.2f}, {dm["ratio_hi"]:.2f}]'
        log(f'{k:<12}{ap_di*1e3:>11.4f}{dmax:>11.2f}{rng:>20}'
            f'{peak:>10.3f}   [{time.time()-t:.0f}s]')
        res[k] = dict(ap_lk=ap_di, delta_max=dmax, peak=peak,
                      ratio_lo=dm['ratio_lo'], ratio_hi=dm['ratio_hi'],
                      feasible=bool(ap_di >= C.AP_S))
    np.savez(os.path.join(OUT, 'certify_phase2.npz'),
             **{f'{k}_{f}': v[f] for k, v in res.items() for f in v})
    log('\n-> results/certify_phase2.npz')
    return res


if __name__ == '__main__':
    main()
