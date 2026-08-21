"""
run_certify_sched.py — the certificate a GAIN-SCHEDULED controller earns.
=========================================================================
A scheduled controller must not be certified against the whole removal range at
once: it is not one controller but a family, and at any instant the member in
use is the one matched to the current removal state.  So the range is split into
K sub-intervals, the member designed at the midpoint of each is certified
against THAT sub-interval, and the guaranteed depth is the smallest over the K.

The scheduling variable is not a disturbance to be estimated: in milling, eta is
known exactly from the pass schedule -- the machine knows how many passes it has
made and at what depth.  That is what makes scheduling on eta legitimate here,
where scheduling on an unmeasurable parameter would not be.

    python run_certify_sched.py
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
import controllers as K
from plate_model import build_plate
from uncertainty import UncertaintySet, RemovalFamily
from milling_dynamics import alpha4_average

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_certify_sched.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def build_member(plate, eta_hat, rem, alpha40):
    d = np.load(os.path.join(OUT, 'controllers_phase2.npz'), allow_pickle=True)
    u = dict(zip([str(x) for x in d['proposed_pnames']],
                 np.asarray(d['proposed_params'], float)))
    ss, pd, _ = K.build('proposed', plate, u, alpha40=alpha40, removal=rem,
                        eta_hat=float(eta_hat))
    return ss, pd


def main(n_seg=3, n_eta=3, n_x=9):
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    rem = RemovalFamily(n=C.N_MODES_DESIGN)
    abar4 = alpha4_average(C.RPM_S, C.AP_S, plate.hp, C.AE)
    a40 = C.SIGN * 1.6 * abar4
    edges = np.linspace(0.0, C.ETA_MAX, n_seg + 1)

    log('=' * 74)
    log('SCHEDULED CERTIFICATE - PB-RAC over eta sub-intervals')
    log('=' * 74)
    log(f'  eta split into {n_seg} sub-intervals: '
        + ', '.join(f'[{edges[i]:.4f}, {edges[i+1]:.4f}]' for i in range(n_seg)))
    log(f'{"sub-interval":<24}{"eta_hat":>9}{"a_p^DI mm":>11}{"delta_max":>11}')

    aps, dms = [], []
    for i in range(n_seg):
        lo, hi = float(edges[i]), float(edges[i + 1])
        ehat = 0.5 * (lo + hi)
        ss, pd = build_member(plate, ehat, rem, a40)
        t = time.time()

        def uset_of(ap):
            return UncertaintySet(plate, rpm=C.RPM_S, ap=ap,
                                  n=C.N_MODES_DESIGN, eta_max=C.ETA_MAX)

        ap = CF.di_depth(uset_of, ss, pd, n_eta=n_eta, n_x=n_x,
                         eta_range=(lo, hi))
        dm = CF.di_margin(uset_of(C.AP_S), ss, pd, n_eta=n_eta, n_x=n_x,
                          eta_range=(lo, hi))
        aps.append(ap)
        dms.append(dm)
        log(f'[{lo:.4f}, {hi:.4f}]{"":<10}{ehat:>9.4f}{ap*1e3:>11.4f}'
            f'{dm:>11.2f}   [{time.time()-t:.0f}s]')

    log('')
    log(f'  guaranteed over the whole removal range (worst sub-interval):')
    log(f'    a_p^DI    = {min(aps)*1e3:.4f} mm')
    log(f'    delta_max = {min(dms):.2f}')
    np.savez(os.path.join(OUT, 'certify_sched.npz'),
             edges=edges, ap=np.array(aps), delta=np.array(dms),
             ap_min=min(aps), delta_min=min(dms))
    log('\n-> results/certify_sched.npz')


if __name__ == '__main__':
    main()
