"""
isolate_mu.py — how much of the certified margin belongs to the delayed PD?
============================================================================
Stage 5-6 certifies mu_tdc and mu_phys_tdc, and both are mu PLUS the delayed
controller of Eq. (30).  So the comparison between them says nothing about
which uncertainty description produced the better mu controller: the margins
could belong entirely to the PD, which is tuned separately by the same PSO in
both cases.

This module certifies the STORED winning designs -- the ones every table in the
study reports -- with and without the delayed PD, under the identical protocol.
The difference is what the PD is worth; the without-PD row is what the
uncertainty description is worth on its own.

    python isolate_mu.py
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify2 as CF2
import config as C
import ctrl2 as K2
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_mu_ss

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_isolate_mu.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def certify(plate, plant, ss, pd):
    zetas = (C.ZETA_LO, C.ZETA_HI)
    base = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
                zetas=zetas, xis=(0.5, 1.0))
    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))

    def mk(p, ss=ss, pd=pd):
        return K2.Ctrl('MU', 4, ss=ss, pd=pd)

    c0 = mk(plant)
    _, peak, _ = CF2.analyse(plant, c0, **base)
    ap = CF2.depth_bisect(lambda a: ControlledPlant(plate, ap=a), mk,
                          n_iter=16, **light)
    dmax = CF2.margin_bisect(plant, c0, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    return peak, ap, dmax


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as fh:
        st = pickle.load(fh)

    cases = [('mu_paper', 'mu_tdc'), ('mu_phys', 'mu_phys_tdc')]
    log('=' * 84)
    log('HOW MUCH OF THE CERTIFIED MARGIN BELONGS TO THE DELAYED PD?')
    log('=' * 84)
    log('  the STORED winning designs, certified under the identical protocol,')
    log('  with the Eq. (30) delayed PD in the loop and with it removed.')
    log('')
    log(f'{"design":>14}{"delayed PD":>13}{"peak":>9}{"a_p^inf mm":>13}'
        f'{"delta_max":>11}')
    res = {}
    for mu_tag, tdc_tag in cases:
        ss = load_mu_ss(mu_tag)
        if ss is None or tdc_tag not in st:
            log(f'{mu_tag:>14}   not on disk')
            continue
        u = dict(st[tdc_tag]['params'])
        c = K2.build(tdc_tag, plant, u, ss)
        pd = c.at(0.0)[1]
        for lab, p in (('yes', pd), ('no', None)):
            t = time.time()
            peak, ap, dm = certify(plate, plant, ss, p)
            log(f'{mu_tag:>14}{lab:>13}{peak:9.3f}{ap*1e3:13.4f}{dm:11.3f}'
                f'   [{time.time()-t:.0f}s]')
            res[f'{mu_tag}_{lab}'] = (peak, ap, dm)

    log('')
    for mu_tag, _ in cases:
        a = res.get(f'{mu_tag}_yes'); b = res.get(f'{mu_tag}_no')
        if a and b:
            log(f'  {mu_tag}: the delayed PD is worth '
                f'{(a[1]-b[1])*1e3:+.4f} mm of certified depth '
                f'({100*(a[1]/max(b[1],1e-12)-1):+.0f} %) and '
                f'{a[2]-b[2]:+.3f} of delta_max')
    if 'mu_paper_no' in res and 'mu_phys_no' in res:
        p, q = res['mu_paper_no'], res['mu_phys_no']
        log('')
        log('  WITHOUT the delayed PD -- what the uncertainty description is')
        log('  worth on its own:')
        log(f'      paper set   a_p^inf {p[1]*1e3:.4f} mm, delta_max {p[2]:.3f}')
        log(f'      physics set a_p^inf {q[1]*1e3:.4f} mm, delta_max {q[2]:.3f}')
    np.savez(os.path.join(OUT, 'isolate_mu.npz'),
             **{k: np.array(v) for k, v in res.items()})
    log(f'\ntotal {time.time()-t0:.0f}s -> results/isolate_mu.npz')


if __name__ == '__main__':
    main()
