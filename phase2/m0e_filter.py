"""m0e_filter.py — stage M0e: the synthesis-stability-filtered weight search.

M0d's diagnosis: the D-K chain is an ill-conditioned map at some weight
triples -- ppm-level environmental drift is amplified into a complete flip of
controller quality, so the historical winners are lucky draws whose recipes do
not reproduce.  A scheduled FAMILY cannot be built out of lucky draws: every
node's member must be re-obtainable.  docs/09 section 5 therefore declares this
stage: search the weight grid, and at every candidate triple REPEAT the
synthesis under a tiny input jitter, keeping only designs whose judged quality
is stable under it.

Protocol, declared before the run:

  * node: mid-span (the best node of M0/M0b/M0c), the winner then checked at
    the two edge nodes;
  * grid: kf in {1e4, 3e4, 1e5, 3e5} x (ku, fc) in {(1.7e-3, 800),
    (1.7e-2, 1500), (6e-2, 1500), (1.7e-2, 3000), (6e-2, 3000)} x
    n_iter in {1, 3}.  n_iter=1 is the pure hinfsyn arm -- no D-scale fit at
    all, hence none of the ill-conditioned map -- included because the M0
    smoke run's single iteration (RS 1.400) beat every full-iteration result;
  * jitter: the synthesis frequency grid AND the weight corner frequencies
    multiplied by (1 +/- 1e-3) -- one thousand times the ppm drift that
    flipped the historical winners, so surviving it is a meaningful
    reproducibility statement.  The JUDGE never moves: mu_RS position-free at
    the node, actuator block included, on the canonical grid (G1 of docs/09);
  * certified quality of a triple = max of the judged RS over {nominal, +, -}.
    Since the max includes the nominal, only a triple whose nominal RS is
    already under 1 can certify under 1 -- the filter can only revoke wins,
    never manufacture them;
  * verdict: min certified < 1 at mid-span opens the gate with a REPRODUCIBLE
    triple (stage M1 justified); otherwise G1 on this chain is judged "not
    attainable reproducibly", the negative result docs/09 pre-registered,
    with the stored membership witness (optimum <= 1.156) as the paradox.

Environment note: this container reproduces the M0-era numbers exactly (the
stored mu_paper artifact judges to 1.1564/1.1570/1.2052 against the recorded
1.156-1.205, and the M0 row (3e5, 1.7e-2, 1500) replays to the byte:
mu_RP 4.218, RS 1.799), so every comparison against M0/M0b/M0c is direct.

    python phase2/m0e_filter.py

Appends to results/log_musyn_sched.txt, writes results/m0e_filter.npz.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from ctrl2 import series
from fopid import rolloff_ss
from plate_model import build_plate
from musyn_sched import plant_node, judge_point, judge_node, log

OUT = C.RESULTS
KF = (1e4, 3e4, 1e5, 3e5)
KU_FC = ((1.7e-3, 800.0), (1.7e-2, 1500.0), (6e-2, 1500.0),
         (1.7e-2, 3000.0), (6e-2, 3000.0))
N_ITER = (1, 3)
JIT = 1e-3                       # +/-0.1 % on the synthesis inputs
SURVIVOR_RS = 1.25               # phase-2 entry: also documents the spread
MAX_SURVIVORS = 8


def design_jit(plate, x, kf, ku, fc, n_iter, eps=0.0):
    """One synthesis with the INPUTS perturbed by (1 + eps): the frequency
    grid the D-fits see, and the corner frequencies of both synthesis
    weights.  eps=0 is the nominal chain, byte-identical to design_node."""
    from dk_synthesis import (prepare_plant, unscale_controller, dk_iteration,
                              reduce_controller)
    from robust_design import freq_grid

    m = 1.0 + eps
    wpf = dict(k=kf, fc_hz=fc * m, M=250.0)
    wpu = dict(k=ku, fc_hz=2500.0 * m, M=60.0)
    U, Pp = plant_node(plate, x, wpf=wpf, wpu=wpu)
    Ps, info = prepare_plant(Pp, alpha_shift=12.0)
    f = freq_grid() * m
    res = dk_iteration(Ps, Pp['blocks'], 2 * np.pi * f / info['w0'],
                       n_iter=n_iter, orders=(0, 1, 1, 1), verbose=False)
    K = reduce_controller(unscale_controller(res['K'], info), 12)
    ss = series((np.array(K.A), np.array(K.B), np.array(K.C), np.array(K.D)),
                rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
    return ss, float(res['mu']), ss[0].shape[0]


def main():
    from robust_design import freq_grid
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    f = freq_grid()                       # the judge's grid: never jittered
    x_mid = 0.5 * plate.lp

    log('')
    log('M0e - SYNTHESIS-STABILITY-FILTERED WEIGHT SEARCH (docs/09 sec. 5)')
    log('=' * 78)
    log(f'  node: mid-span; grid {len(KF)} kf x {len(KU_FC)} (ku,fc) x '
        f'{len(N_ITER)} n_iter = {len(KF)*len(KU_FC)*len(N_ITER)} triples;')
    log(f'  jitter +/-{JIT:.0e} on synthesis f-grid and weight corners; '
        'judge fixed (G1)')
    log('')

    # ---- phase 1: the nominal chain over the whole grid --------------------
    rows = []
    for n_iter in N_ITER:
        for kf in KF:
            for ku, fc in KU_FC:
                t = time.time()
                try:
                    ss, mu_rp, order = design_jit(plate, x_mid, kf, ku, fc,
                                                  n_iter)
                    rs = judge_point(plate, x_mid, ss, f)
                except Exception as e:
                    log(f'  n={n_iter} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f}'
                        f' -> FAILED ({type(e).__name__})')
                    rows.append((n_iter, kf, ku, fc, np.nan, np.nan))
                    continue
                rows.append((n_iter, kf, ku, fc, mu_rp, rs))
                log(f'  n={n_iter} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> '
                    f'mu_RP={mu_rp:7.3f}  RS={rs:7.3f}  [{time.time()-t:.0f}s]'
                    + ('  <-- nominal < 1' if rs < 1 else ''))

    arr = np.array(rows, float)
    ok = np.isfinite(arr[:, 5])
    order_idx = np.argsort(arr[:, 5])
    surv = [i for i in order_idx if ok[i] and arr[i, 5] < SURVIVOR_RS]
    surv = surv[:MAX_SURVIVORS]

    # ---- phase 2: the jitter filter on the survivors -----------------------
    log('')
    log(f'  phase 2: jitter filter on the {len(surv)} best rows '
        f'(nominal RS < {SURVIVOR_RS})')
    cert = {}
    for i in surv:
        n_iter, kf, ku, fc = int(arr[i, 0]), arr[i, 1], arr[i, 2], arr[i, 3]
        vals = [arr[i, 5]]
        for eps in (+JIT, -JIT):
            t = time.time()
            try:
                ss, _, _ = design_jit(plate, x_mid, kf, ku, fc, n_iter, eps)
                vals.append(judge_point(plate, x_mid, ss, f))
            except Exception as e:
                vals.append(np.inf)
                log(f'    jitter {eps:+.0e} FAILED ({type(e).__name__})')
        certified = float(np.max(vals))
        spread = float(np.max(vals) - np.min(vals))
        cert[i] = (certified, spread, vals)
        log(f'  n={n_iter} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> '
            f'RS nom {vals[0]:6.3f}  +jit {vals[1]:6.3f}  -jit {vals[2]:6.3f}'
            f'  certified {certified:6.3f}  spread {spread:6.3f}'
            + ('  <-- G1 REPRODUCIBLY' if certified < 1 else ''))

    # ---- verdict -----------------------------------------------------------
    log('')
    if cert:
        best_i = min(cert, key=lambda i: cert[i][0])
        best_cert = cert[best_i][0]
    else:
        best_i, best_cert = None, np.inf
    if best_cert < 1.0:
        n_iter, kf, ku, fc = (int(arr[best_i, 0]), arr[best_i, 1],
                              arr[best_i, 2], arr[best_i, 3])
        log(f'  GATE OPEN, reproducibly: certified RS = {best_cert:.3f} at '
            f'n_iter={n_iter} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f}')
        log('  checking the winning recipe at the edge nodes + the cell:')
        for frx in (0.0, 0.5, 1.0):
            t = time.time()
            ss, mu_rp, order = design_jit(plate, frx * plate.lp, kf, ku, fc,
                                          n_iter)
            rs_pt, rs_cell = judge_node(plate, frx * plate.lp, ss, f)
            log(f'    x={frx:4.0%} -> RS(point)={rs_pt:7.3f}  '
                f'RS(cell)={rs_cell:7.3f}  order {order}  '
                f'[{time.time()-t:.0f}s]')
    else:
        nom_best = float(np.nanmin(arr[:, 5])) if ok.any() else np.inf
        log(f'  GATE NOT OPEN REPRODUCIBLY: best nominal RS = {nom_best:.3f},'
            f'  best certified RS = {best_cert:.3f} -- both above 1.')
        log('  Verdict pre-registered in docs/09: G1 on this D-K chain is')
        log('  "not attainable reproducibly" over the searched grid; the')
        log('  stored membership witness (per-node optimum <= 1.156) stands')
        log('  as an existence proof without a recipe.')

    np.savez(os.path.join(OUT, 'm0e_filter.npz'),
             rows=arr,
             columns=['n_iter', 'kf', 'ku', 'fc', 'mu_rp', 'rs_nominal'],
             survivors=np.array(surv, int),
             certified=np.array([[i, cert[i][0], cert[i][1]] for i in cert],
                                float) if cert else np.zeros((0, 3)),
             jit=JIT)
    log(f'\nM0e total {time.time()-t0:.0f}s -> results/m0e_filter.npz')


if __name__ == '__main__':
    main()
