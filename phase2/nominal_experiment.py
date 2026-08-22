"""
nominal_experiment.py — does the LFT NOMINAL explain the negative result?
==========================================================================
The physics-based set is better as a description on every measure, and worse as
a design model: at the paper set's own winning weights it loses 27 % on both
certified margins.  The proposed mechanism was that reshaping moves the LFT
nominal onto the physical manifold, whose middle happens to be the NODE of mode
two -- so the nominal cutting stiffness on that mode is nearly zero, and the
synthesis is never asked to damp it, while the certificate demands it at every
other position.

That is a falsifiable claim, and this is the experiment.  The nominal is moved
along the D(x) locus while the SET is kept covering every position (the
half-widths grow to compensate), so only the design point changes.  If the
mechanism is right, the certified margins should climb as the nominal leaves the
node -- and the whole gap to the paper's design should close.

    W_c(2,2), the nominal cutting stiffness on mode 2, spans
        -96      at the node        (the natural centre)
        -1.8e6   at the edge        a factor of eighteen thousand

Weights are held at the triple that won on the PAPER's set, so this experiment
and matched_certify.py are directly comparable.

    python nominal_experiment.py
"""
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
LOG = open(os.path.join(OUT, 'log_nominal_experiment.txt'), 'w')
PAPER_W = dict(kf=3e5, ku=6e-2, fc=3000.0)
FRACS = [None, 0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    W.W_PF_DEF = dict(k=PAPER_W['kf'], fc_hz=PAPER_W['fc'], M=250.0)
    W.W_PU_DEF = dict(k=PAPER_W['ku'], fc_hz=2500.0, M=60.0)

    log('=' * 86)
    log('DOES THE LFT NOMINAL EXPLAIN THE NEGATIVE RESULT?')
    log('=' * 86)
    log(f'  weights fixed at the PAPER set\'s winner: k_f={PAPER_W["kf"]:.3g} '
        f'k_u={PAPER_W["ku"]:.3g} fc={PAPER_W["fc"]:.0f}')
    log('  the set still covers every tool position at every nominal; only the')
    log('  design point moves.  mu controller alone, no delayed PD, so nothing')
    log('  but the uncertainty description differs between rows.')
    log('')
    log(f'{"x_nom/l_P":>10}{"W_c(2,2)":>13}{"mu":>9}{"J":>11}{"Ms":>7}'
        f'{"V":>6}{"peak":>8}{"a_p^inf mm":>12}{"delta_max":>11}')

    zetas = (C.ZETA_LO, C.ZETA_HI)
    base = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
                zetas=zetas, xis=(0.5, 1.0))
    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    rows = []
    for f in FRACS:
        xn = None if f is None else f * plate.lp
        t = time.time()
        try:
            r = musyn.design_phys(plate, n_iter=2, reduce_to=None, x_nom=xn)
        except Exception as e:                                # noqa: BLE001
            log(f'{("centre" if f is None else f"{f:.2f}"):>10}   design failed '
                f'({type(e).__name__}: {e})')
            continue
        U = r['U']
        Wc = U.a0 * np.outer(U.d_c, U.d_c)
        J, info = evaluate_ss(plate, r['ss'], detail=True)

        def mk(p, ss=r['ss']):
            return K2.Ctrl('MU_PHYS', 4, ss=ss)

        c0 = mk(plant)
        _, peak, _ = CF2.analyse(plant, c0, **base)
        ap = CF2.depth_bisect(lambda a: ControlledPlant(plate, ap=a), mk,
                              n_iter=16, **light)
        dmax = CF2.margin_bisect(plant, c0, n_iter=14,
                                 base_kw=dict(n_pos=5, xis=(1.0,)))
        log(f'{("centre" if f is None else f"{f:.2f}"):>10}'
            f'{Wc[1, 1]:13.4g}{r["mu"]:9.3f}{J:+11.5f}{info["Ms"]:7.3f}'
            f'{info["V"]:6.0f}{peak:8.3f}{ap*1e3:12.4f}{dmax:11.3f}'
            f'   [{time.time()-t:.0f}s]')
        rows.append(dict(frac=(np.nan if f is None else f), W22=Wc[1, 1],
                         mu=r['mu'], J=J, Ms=info['Ms'], V=info['V'],
                         peak=peak, ap_inf=ap, delta_max=dmax))

    # --------------------------------------------------- the reference design
    log('')
    log('  for comparison, the PAPER set at the same weights, mu controller alone:')
    try:
        rp = musyn.design(plate, alpha_coupling='paper', n_iter=2,
                          reduce_to=None)

        def mkp(p, ss=rp['ss']):
            return K2.Ctrl('MU_PAPER', 4, ss=ss)

        cp = mkp(plant)
        _, pk, _ = CF2.analyse(plant, cp, **base)
        app = CF2.depth_bisect(lambda a: ControlledPlant(plate, ap=a), mkp,
                               n_iter=16, **light)
        dmp = CF2.margin_bisect(plant, cp, n_iter=14,
                                base_kw=dict(n_pos=5, xis=(1.0,)))
        log(f'{"paper":>10}{"":>13}{rp["mu"]:9.3f}'
            f'{evaluate_ss(plate, rp["ss"]):+11.5f}{"":>13}'
            f'{pk:8.3f}{app*1e3:12.4f}{dmp:11.3f}')
        ref = dict(peak=pk, ap_inf=app, delta_max=dmp, mu=rp['mu'])
    except Exception as e:                                    # noqa: BLE001
        log(f'   (paper reference failed: {type(e).__name__}: {e})')
        ref = {}

    if rows:
        a = np.array([r['ap_inf'] for r in rows]) * 1e3
        dm = np.array([r['delta_max'] for r in rows])
        i = int(np.argmax(a))
        tag = ('centre' if np.isnan(rows[i]['frac'])
               else f'{rows[i]["frac"]:.2f}')
        log('')
        log(f'  best certified depth at x_nom/l_P = {tag}:  '
            f'a_p^inf {a[i]:.4f} mm, delta_max {dm[i]:.3f}')
        c = [r for r in rows if np.isnan(r['frac'])]
        if c:
            log(f'  against the natural centre:      a_p^inf '
                f'{c[0]["ap_inf"]*1e3:.4f} mm, delta_max {c[0]["delta_max"]:.3f}'
                f'   ({100*(a[i]/(c[0]["ap_inf"]*1e3)-1):+.1f} % and '
                f'{100*(dm[i]/c[0]["delta_max"]-1):+.1f} %)')
        if ref:
            log(f'  against the paper set:           a_p^inf '
                f'{ref["ap_inf"]*1e3:.4f} mm, delta_max {ref["delta_max"]:.3f}')

    np.savez(os.path.join(OUT, 'nominal_experiment.npz'),
             frac=np.array([r['frac'] for r in rows]),
             W22=np.array([r['W22'] for r in rows]),
             mu=np.array([r['mu'] for r in rows]),
             J=np.array([r['J'] for r in rows]),
             Ms=np.array([r['Ms'] for r in rows]),
             V=np.array([r['V'] for r in rows]),
             peak=np.array([r['peak'] for r in rows]),
             ap_inf=np.array([r['ap_inf'] for r in rows]),
             delta_max=np.array([r['delta_max'] for r in rows]),
             **{f'ref_{k}': v for k, v in ref.items()})
    log(f'\ntotal {time.time()-t0:.0f}s -> results/nominal_experiment.npz')


if __name__ == '__main__':
    main()
