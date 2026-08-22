"""
matched_certify.py — is the physics set worse, or did the search pick a corner?
===============================================================================
The certified metrics came out WORSE for the physics-set controller:

    a_p^inf   0.3778 mm  against  0.4364 for mu-TDC on the paper's set
    delta_max 1.35       against  1.54

but the two designs did not come from the same weights.  The objective is a
Floquet stability margin, and on the physics set 42 % of the weight grid is
feasible against 12 %, so the search reached a corner with a control penalty
thirty-five times lighter (k_u = 1.7e-3 against 6e-2).  A lighter penalty buys
nominal cutting depth and costs robustness -- which is exactly the pattern in
the numbers (mu_RS 1.851 against 1.479).

So the certified comparison as it stands measures the SEARCH, not the SET.  This
module removes that confound: it rebuilds the physics-set controller at the
weights that won on the PAPER's set, tunes its delayed PD with the same PSO and
the same seeds, and certifies it under the same protocol.  Then the only thing
that differs between the two rows is the uncertainty description.

    python matched_certify.py
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
import musyn
import weights as W
from design2 import Design2, optimise
from eval2 import evaluate            # takes a Ctrl object
from objective import evaluate as evaluate_ss   # takes a raw state space
from plate_model import build_plate
from plant_ss import ControlledPlant

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_matched_certify.txt'), 'w')

# the weight triple that won on the paper's set (run_musyn.py, variant A)
PAPER_W = dict(kf=3e5, ku=6e-2, fc=3000.0)


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    log('=' * 78)
    log('THE PHYSICS SET AT THE PAPER SET\'S OWN WINNING WEIGHTS')
    log('=' * 78)
    log(f'  weights k_f={PAPER_W["kf"]:.3g} k_u={PAPER_W["ku"]:.3g} '
        f'fc={PAPER_W["fc"]:.0f} -- the triple that won variant A')

    W.W_PF_DEF = dict(k=PAPER_W['kf'], fc_hz=PAPER_W['fc'], M=250.0)
    W.W_PU_DEF = dict(k=PAPER_W['ku'], fc_hz=2500.0, M=60.0)
    r = musyn.design_phys(plate, n_iter=2, reduce_to=None, verbose=False)
    J0, info0 = evaluate_ss(plate, r['ss'], detail=True)
    log(f'\n  mu controller alone: mu = {r["mu"]:.3f}, order {r["order"]}, '
        f'J = {J0:+.5f}, Ms = {info0["Ms"]:.3f}, V = {info0["V"]:.0f}, '
        f'feasible = {info0["feasible"]}')

    d = Design2('mu_phys_tdc', plant, plate, r['ss'])
    res = optimise(d)
    J, inf2 = evaluate(plate, res['ctrl'], detail=True)
    log(f'  + delayed PD (same PSO, same seeds): J = {J:+.5f}, '
        f'Ms = {inf2["Ms"]:.3f}, V = {inf2["V"]:.0f}, '
        f'kpd = {res["params"]["kpd"]:.4f}, kdd = {res["params"]["kdd"]:.4f}')

    zetas = (C.ZETA_LO, C.ZETA_HI)
    base = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
                zetas=zetas, xis=(0.5, 1.0))
    tau0 = 60.0 / (3 * C.RPM_S)

    def mk(p):
        import ctrl2 as K2
        return K2.build('mu_phys_tdc', p, dict(res['params']), r['ss'])

    c0 = mk(plant)
    tm, peak, _ = CF2.analyse(plant, c0, **base)
    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    ap_inf = CF2.depth_bisect(lambda ap: ControlledPlant(plate, ap=ap), mk,
                              n_iter=16, **light)
    dmax = CF2.margin_bisect(plant, c0, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    ratio = 'inf' if np.isinf(tm) else f'{tm/tau0:.2f}'
    log(f'\n  certified: peak = {peak:.3f}, tau_max/tau0 = {ratio}, '
        f'a_p^inf = {ap_inf*1e3:.4f} mm, delta_max = {dmax:.2f}')
    log('\n  for comparison, at each design\'s OWN winning weights:')
    log('    mu_tdc      (paper set)   a_p^inf 0.4364 mm  delta_max 1.54  '
        'peak 0.603')
    log('    mu_phys_tdc (physics set) a_p^inf 0.3778 mm  delta_max 1.35  '
        'peak 0.762')
    np.savez(os.path.join(OUT, 'matched_certify.npz'),
             peak=peak, ap_inf=ap_inf, delta_max=dmax, J=J, mu=r['mu'],
             Ms=inf2['Ms'], V=inf2['V'],
             kpd=res['params']['kpd'], kdd=res['params']['kdd'],
             ss=np.array([np.asarray(x, float) for x in r['ss']],
                         dtype=object))
    log(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
