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
            if f'{tag}_ss0' in m:
                out[tag] = dict(ss=tuple(m[f'{tag}_ss{i}'] for i in range(4)),
                                pd=None, n_params=int(m.get(f'{tag}_np', 4)))
    return out


def main(kinds=None, n_eta=3, n_x=5):
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
    eps0 = U0.gap_bound(V0, scaled=True, n_eta=5, n_xi=3, n_x=17, n_a=3, n_z=2)
    log(f'  uncertainty set : eta in [0, {C.ETA_MAX}], xi in [0, 1], '
        f'x in [0, l_P], alpha4 in [{C.ALPHA_LO}, {C.ALPHA_HI}] abar4, '
        f'zeta x [{C.ZETA_LO}, {C.ZETA_HI}]')
    log(f'  polytope        : {len(V0)} vertices')
    log(f'  covering radius : eps = {eps0:.4g} (scaled), '
        f'{100*eps0/np.linalg.norm(U0.scale(*U0.nominal())[0], 2):.2f} % of ||A||')
    log('')

    res = {}
    for k in kinds:
        c = ctrls[k]
        t = time.time()
        r, eps = CF.certify(U0, c['ss'], c['pd'], n_eta=n_eta, n_x=n_x)
        ap_lk = CF.certified_depth(uset_of, c['ss'], c['pd'], n_eta=n_eta,
                                   n_x=n_x, eps=eps)
        dmax = CF.uncertainty_margin(U0, c['ss'], c['pd'], n_eta=n_eta,
                                     n_x=n_x)
        dm = delay_margin(plate, c['ss'], c['pd'], C.AP_S)
        log(f'  {k:<10} certified at a_p = {C.AP_S*1e3:.2f} mm : '
            f'{r["feasible"]}   a_p^LK = {ap_lk*1e3:.4f} mm   '
            f'delta_max = {dmax:.2f}   '
            f'tau/tau_0 in [{dm["ratio_lo"]:.2f}, {dm["ratio_hi"]:.2f}]'
            f'   [{time.time()-t:.0f}s]')
        res[k] = dict(feasible=r['feasible'], ap_lk=ap_lk, delta_max=dmax,
                      ratio_lo=dm['ratio_lo'], ratio_hi=dm['ratio_hi'],
                      rpm_lo=float(60 / (3 * dm['tau_hi']))
                      if np.isfinite(dm['tau_hi']) else np.nan)
    np.savez(os.path.join(OUT, 'certify_phase2.npz'),
             **{f'{k}_{f}': v[f] for k, v in res.items() for f in v})
    log('\n-> results/certify_phase2.npz')
    return res


if __name__ == '__main__':
    main()
