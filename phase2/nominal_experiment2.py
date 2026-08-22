"""
nominal_experiment2.py — the nominal sweep, with the weights re-searched.
==========================================================================
The first version of this experiment held the synthesis weights fixed at the
triple that won on the paper's set.  That confounded it: moving the nominal
changes the design problem, and at fixed weights most off-centre nominals came
out violating Ms <= 2, so their margins are not comparable to anything.  Half
the rows were inadmissible designs.

Here each nominal gets its own small weight search under the SAME rule the rest
of the study uses -- best objective among the trials that satisfy Ms <= 2,
effort <= 450 V/N and the pole bound -- and only then is certified.  So every
row is an admissible design, and the only thing that differs between rows is
where the LFT nominal sits on the D(x) locus.

The prediction under test: the certified margins climb as the nominal leaves the
node of mode 2, where the nominal cutting stiffness on that mode is nearly zero.

    python nominal_experiment2.py
"""
import itertools
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify2 as CF2
import config as C
import ctrl2 as K2
import musyn
import weights as W
from objective import evaluate as evaluate_ss
from plate_model import build_plate
from plant_ss import ControlledPlant

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_nominal_experiment2.txt'), 'w')
FRACS = [None, 0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
GRID = list(itertools.product([1e5, 3e5, 1e6], [1.7e-3, 1.7e-2, 6e-2],
                              [800.0, 3000.0]))


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def best_design(plate, x_nom):
    """Best FEASIBLE design over the small grid -- the study's own rule."""
    best = None
    for kf, ku, fc in GRID:
        W.W_PF_DEF = dict(k=kf, fc_hz=fc, M=250.0)
        W.W_PU_DEF = dict(k=ku, fc_hz=2500.0, M=60.0)
        try:
            r = musyn.design_phys(plate, n_iter=2, reduce_to=None, x_nom=x_nom)
            J, info = evaluate_ss(plate, r['ss'], detail=True)
        except Exception:                                     # noqa: BLE001
            continue
        if not info['feasible']:
            continue
        if best is None or J > best['J']:
            best = dict(J=J, info=info, ss=r['ss'], mu=r['mu'], U=r['U'],
                        kf=kf, ku=ku, fc=fc)
    return best


def certify(plate, plant, ss):
    zetas = (C.ZETA_LO, C.ZETA_HI)
    base = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
                zetas=zetas, xis=(0.5, 1.0))
    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))

    def mk(p, ss=ss):
        return K2.Ctrl('MU', 4, ss=ss)

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
    log('=' * 92)
    log('THE NOMINAL SWEEP, WITH THE WEIGHTS RE-SEARCHED AT EVERY NOMINAL')
    log('=' * 92)
    log(f'  {len(GRID)} weight trials per nominal; the design kept is the best')
    log('  FEASIBLE one, the same rule used everywhere else in the study.')
    log('  mu controller alone -- no delayed PD -- so only the uncertainty')
    log('  description differs between rows.')
    log('')
    log(f'{"x_nom/l_P":>10}{"W_c(2,2)":>13}{"mu":>9}{"J":>11}{"Ms":>7}{"V":>6}'
        f'{"k_f":>8}{"k_u":>8}{"peak":>8}{"a_p^inf mm":>12}{"delta_max":>11}')
    rows = []
    for f in FRACS:
        xn = None if f is None else f * plate.lp
        t = time.time()
        b = best_design(plate, xn)
        tag = 'centre' if f is None else f'{f:.2f}'
        if b is None:
            log(f'{tag:>10}   no feasible design in {len(GRID)} trials '
                f'[{time.time()-t:.0f}s]')
            continue
        Wc = b['U'].a0 * np.outer(b['U'].d_c, b['U'].d_c)
        peak, ap, dmax = certify(plate, plant, b['ss'])
        log(f'{tag:>10}{Wc[1, 1]:13.4g}{b["mu"]:9.3f}{b["J"]:+11.5f}'
            f'{b["info"]["Ms"]:7.3f}{b["info"]["V"]:6.0f}{b["kf"]:8.3g}'
            f'{b["ku"]:8.4g}{peak:8.3f}{ap*1e3:12.4f}{dmax:11.3f}'
            f'   [{time.time()-t:.0f}s]')
        rows.append(dict(frac=(np.nan if f is None else f), W22=Wc[1, 1],
                         mu=b['mu'], J=b['J'], Ms=b['info']['Ms'],
                         V=b['info']['V'], peak=peak, ap_inf=ap,
                         delta_max=dmax, kf=b['kf'], ku=b['ku'], fc=b['fc']))

    if rows:
        a = np.array([r['ap_inf'] for r in rows]) * 1e3
        dm = np.array([r['delta_max'] for r in rows])
        w = np.abs([r['W22'] for r in rows])
        i = int(np.argmax(a))
        c = [r for r in rows if np.isnan(r['frac'])]
        tag = 'centre' if np.isnan(rows[i]['frac']) else f'{rows[i]["frac"]:.2f}'
        log('')
        log(f'  best certified depth at x_nom/l_P = {tag}: a_p^inf {a[i]:.4f} mm, '
            f'delta_max {dm[i]:.3f}')
        if c:
            log(f'  the natural centre (on the node):     a_p^inf '
                f'{c[0]["ap_inf"]*1e3:.4f} mm, delta_max {c[0]["delta_max"]:.3f}')
        log('  reference, paper set at its own winner: a_p^inf 0.4364 mm, '
            'delta_max 1.543   (from stage56)')
        if len(rows) > 2:
            ra = float(np.corrcoef(np.log10(np.maximum(w, 1e-9)), a)[0, 1])
            rd = float(np.corrcoef(np.log10(np.maximum(w, 1e-9)), dm)[0, 1])
            log('')
            log('  THE TEST — correlation of the certified margins with the')
            log('  mechanism variable log|W_c(2,2)|, the nominal cutting')
            log('  stiffness on mode 2:')
            log(f'      a_p^inf   r = {ra:+.3f}')
            log(f'      delta_max r = {rd:+.3f}')
            log('  the mechanism predicts a STRONG POSITIVE correlation for both.')
    np.savez(os.path.join(OUT, 'nominal_experiment2.npz'),
             **{k: np.array([r[k] for r in rows])
                for k in ('frac', 'W22', 'mu', 'J', 'Ms', 'V', 'peak',
                          'ap_inf', 'delta_max', 'kf', 'ku', 'fc')})
    log(f'\ntotal {time.time()-t0:.0f}s -> results/nominal_experiment2.npz')


if __name__ == '__main__':
    main()
