"""musyn_sched.py — stage M0 of the scheduled D-K project: the feasibility probe.

The next paper's controller is a FAMILY K_mu(x_j): one D-K synthesis per
scheduling node, each on the physics set with the position channels REMOVED
(the position is a scheduling signal, not an uncertainty).  Everything else of
the inherited chain is kept byte-for-byte: additive actuator weights, synthesis
weights, spectral shift, scaling, frequency grid, D-scale orders, reduction,
rolloff.

This module supplies the position-free plant and runs the gate question of
docs/09, criterion G1: is mu_RS < 1 -- ACTUATOR BLOCK INCLUDED -- attainable at
a node at all?  Nothing in the study has reached it (best fixed design: 1.205
under the scheduled reading).  A small weight grid at three nodes answers it;
the full 21-node family is stage M1 and only worth running if this gate opens.

    python phase2/musyn_sched.py            (the M0 probe)
    python phase2/musyn_sched.py m0c        (the two M0c arms)

Writes results/musyn_sched_probe.{txt,npz} and appends to log_musyn_sched.txt.
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
from uncertain_phys import PhysUncertainSystem

OUT = C.RESULTS
NODES_M0 = (0.0, 0.5, 1.0)                 # fractions of l_P
WEIGHTS_M0 = ((3e5, 1.7e-3, 800.0),        # the mu_phys winner
              (1e6, 1.7e-2, 1500.0),       # the paper-era default
              (3e5, 1.7e-2, 1500.0))       # between the two
POS = ('d_D', 'd_r')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(OUT, 'log_musyn_sched.txt'), 'a') as fh:
        fh.write(line + '\n')


# ---------------------------------------------------------------------------
def slice_position(P, U):
    """The generalized plant with the d_D/d_r channels physically removed.

    Dropping an uncertainty channel is freezing its delta at zero, which is
    exactly what 'the position is known' means for the synthesis.  The block
    list, the initial D-scales and the channel names shrink with it.
    """
    drop = sorted(np.concatenate([U.idx['d_D'], U.idx['d_r']]).tolist())
    n_in = P['B'].shape[1]
    n_out = P['C'].shape[0]
    in_drop = {1 + c for c in drop}            # inputs:  [u_Pa | u_Pr | F n u]
    out_drop = {2 + c for c in drop}           # outputs: [af au | y_Pr | z y]
    ki = [i for i in range(n_in) if i not in in_drop]
    ko = [o for o in range(n_out) if o not in out_drop]

    blocks = [P['blocks'][0]]
    dsc = [float(P['dscale_init'][0])]
    names = [P['names'][0]]
    for j, (nm, m) in enumerate(U.blocks_used):
        if nm in POS:
            continue
        blocks.append(('S', m, m))
        dsc.append(float(P['dscale_init'][j + 1]))
        names.append(nm)
    blocks.append(P['blocks'][-1])
    names.append(P['names'][-1])

    return dict(A=P['A'], B=P['B'][:, ki], C=P['C'][ko, :],
                D=P['D'][np.ix_(ko, ki)], nmeas=1, ncon=1, blocks=blocks,
                n_unc_in=P['n_unc_in'] - len(drop),
                n_unc_out=P['n_unc_out'] - len(drop),
                n_w=P['n_w'], n_z=P['n_z'], dscale_init=np.array(dsc),
                names=names,
                channels=[c for k, c in enumerate(P['channels'])
                          if k not in set(drop)])


def plant_node(plate, x, wpf=None, wpu=None):
    """(U, P_sliced) at one scheduling node, position channels removed."""
    U = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN,
                            n_modes=C.N_MODES_DESIGN, x_nom=float(x),
                            wpf=wpf, wpu=wpu)
    return U, slice_position(U.generalized_plant(), U)


# ---------------------------------------------------------------------------
def design_node(plate, x, kf, ku, fc, n_iter=3, reduce_to=12, verbose=False):
    """One D-K synthesis at one node, inherited chain unchanged."""
    from dk_synthesis import (prepare_plant, unscale_controller, dk_iteration,
                              reduce_controller)
    from robust_design import freq_grid

    wpf = dict(k=kf, fc_hz=fc, M=250.0)
    wpu = dict(k=ku, fc_hz=2500.0, M=60.0)
    U, Pp = plant_node(plate, x, wpf=wpf, wpu=wpu)
    Ps, info = prepare_plant(Pp, alpha_shift=12.0)
    f = freq_grid()
    res = dk_iteration(Ps, Pp['blocks'], 2 * np.pi * f / info['w0'],
                       n_iter=n_iter, orders=(0, 1, 1, 1), verbose=verbose)
    K_full = unscale_controller(res['K'], info)
    K = reduce_controller(K_full, reduce_to) if reduce_to else K_full
    ss = series((np.array(K.A), np.array(K.B), np.array(K.C), np.array(K.D)),
                rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
    return dict(ss=ss, mu_rp=float(res['mu']), U=U, Pp=Pp, f=f,
                order=ss[0].shape[0])


def judge_node(plate, x, ss, f):
    """mu_RS at the node: point (position frozen) and cell (interpolation
    error inside the set), the actuator block included in both."""
    from cross_mu import mu_peaks
    from mu_scheduled import plant_at

    cell_h = plate.lp / 40.0
    _, Pp = plant_node(plate, x)                       # default weights: the
    rs_pt = mu_peaks(Pp, ss, f=f)[0]                   # RS cut has no W_f/W_u
    U_cell = plant_at(plate, x, cell_h)
    rs_cell = mu_peaks(U_cell.generalized_plant(), ss, f=f)[0]
    return rs_pt, rs_cell


# ---------------------------------------------------------------------------
# M0c -- two arms, isolating what keeps the gate shut (docs/09 section 5)
# ---------------------------------------------------------------------------
def judge_point(plate, x, ss, f):
    from cross_mu import mu_peaks
    _, Pp = plant_node(plate, x)
    return mu_peaks(Pp, ss, f=f)[0]


def m0c(n_iter_b=5, orders_b=(0, 2, 2, 2)):
    """(b) heavier D-K on the physics node; (c) paper-set synthesis over a
    weight grid, judged by the position-free per-node test at three nodes.

    (c) isolates the synthesis SET from the per-nodeness: mu_paper already
    proves a paper-set design scores 1.156-1.205 under this judge, but its
    weights were chosen to maximise the Floquet objective J, not this judge.
    If re-weighting pushes a paper-set design under 1, the gate opens without
    per-node synthesis at all -- and the physics-set penalty is confirmed as
    the blocker, not the D-K chain.
    """
    from dk_synthesis import prepare_plant  # noqa: F401  (chain sanity)
    from robust_design import freq_grid
    import weights as W
    import musyn

    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    f = freq_grid()
    nodes = tuple(fr * plate.lp for fr in NODES_M0)

    log('')
    log('M0c(b) - HEAVIER D-K ON THE PHYSICS NODE (mid-span): '
        f'orders {orders_b}, n_iter {n_iter_b}')
    best_b = (np.inf, None)
    for kf, ku, fc in ((3e5, 1.7e-2, 1500.0), (3e5, 1.7e-3, 800.0)):
        t = time.time()
        try:
            from dk_synthesis import (unscale_controller, dk_iteration,
                                      reduce_controller)
            wpf = dict(k=kf, fc_hz=fc, M=250.0)
            wpu = dict(k=ku, fc_hz=2500.0, M=60.0)
            U, Pp = plant_node(plate, 0.5 * plate.lp, wpf=wpf, wpu=wpu)
            Ps, info = prepare_plant(Pp, alpha_shift=12.0)
            res = dk_iteration(Ps, Pp['blocks'],
                               2 * np.pi * f / info['w0'],
                               n_iter=n_iter_b, orders=orders_b)
            K = reduce_controller(unscale_controller(res['K'], info), 12)
            ss = series((np.array(K.A), np.array(K.B), np.array(K.C),
                         np.array(K.D)),
                        rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
            rs = judge_point(plate, 0.5 * plate.lp, ss, f)
        except Exception as e:
            log(f'  kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> FAILED ({e})')
            continue
        log(f'  kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> '
            f'mu_RP={res["mu"]:7.3f}  RS(point)={rs:7.3f}  '
            f'[{time.time()-t:.0f}s]' + (' <-- G1' if rs < 1 else ''))
        best_b = min(best_b, (rs, (kf, ku, fc)))

    log('')
    log('M0c(c) - PAPER-SET SYNTHESIS, JUDGED POSITION-FREE AT THREE NODES')
    log('         (mu_paper witness under this judge: sup 1.205)')
    grid = [(kf, ku, fc)
            for kf in (1e5, 3e5, 1e6)
            for ku, fc in ((1.7e-3, 800.0), (1.7e-2, 1500.0),
                           (6e-2, 1500.0))]
    keep_f, keep_u = dict(W.W_PF_DEF), dict(W.W_PU_DEF)
    best_c = (np.inf, None)
    try:
        for kf, ku, fc in grid:
            t = time.time()
            try:
                W.W_PF_DEF = dict(k=kf, fc_hz=fc, M=250.0)
                W.W_PU_DEF = dict(k=ku, fc_hz=2500.0, M=60.0)
                r = musyn.design(plate, alpha_coupling='paper', n_iter=3,
                                 reduce_to=12)
                sup = max(judge_point(plate, x, r['ss'], f) for x in nodes)
            except Exception as e:
                log(f'  kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> FAILED ({e})')
                continue
            log(f'  kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> '
                f'sup RS(3 nodes)={sup:7.3f}  order {r["order"]}  '
                f'[{time.time()-t:.0f}s]' + (' <-- G1' if sup < 1 else ''))
            best_c = min(best_c, (sup, (kf, ku, fc)))
    finally:
        W.W_PF_DEF, W.W_PU_DEF = keep_f, keep_u

    log('')
    for tag, best in (('(b) physics node, heavier D-K', best_b),
                      ('(c) paper set, re-weighted', best_c)):
        if best[1] is None:
            log(f'  {tag}: every run failed')
        else:
            log(f'  {tag}: best RS = {best[0]:.3f} at '
                f'kf={best[1][0]:.0e} ku={best[1][1]:.0e} fc={best[1][2]:.0f}'
                + ('  -> GATE OPEN' if best[0] < 1 else ''))
    log(f'  M0c total {time.time()-t0:.0f}s')


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    log('=' * 78)
    log('M0 - FEASIBILITY GATE OF THE SCHEDULED D-K PROJECT (docs/09, G1)')
    log('=' * 78)
    log('  question: is mu_RS < 1, ACTUATOR INCLUDED, attainable at a node?')
    log(f'  nodes {tuple(int(100*v) for v in NODES_M0)} % of l_P, '
        f'{len(WEIGHTS_M0)} weight triples, D-K chain unchanged')
    log('')

    rows = []
    best = dict(rs=np.inf)
    for fr in NODES_M0:
        x = fr * plate.lp
        for kf, ku, fc in WEIGHTS_M0:
            t = time.time()
            try:
                d = design_node(plate, x, kf, ku, fc)
                rs_pt, rs_cell = judge_node(plate, x, d['ss'], d['f'])
            except Exception as e:                     # a failed synthesis is
                log(f'  x={fr:4.0%} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f}'
                    f' -> FAILED ({type(e).__name__}: {e})')
                rows.append((fr, kf, ku, fc, np.nan, np.nan, np.nan, 0))
                continue                               # a data point too
            rows.append((fr, kf, ku, fc, d['mu_rp'], rs_pt, rs_cell,
                         d['order']))
            hit = ' <-- G1' if rs_pt < 1.0 else ''
            log(f'  x={fr:4.0%} kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> '
                f'mu_RP={d["mu_rp"]:7.3f}  RS(point)={rs_pt:7.3f}  '
                f'RS(cell)={rs_cell:7.3f}  order {d["order"]}  '
                f'[{time.time()-t:.0f}s]{hit}')
            if rs_pt < best['rs']:
                best = dict(rs=rs_pt, cell=rs_cell, x=fr, kf=kf, ku=ku, fc=fc)

    log('')
    if np.isfinite(best['rs']) and best['rs'] < 1.0:
        log(f'  GATE OPEN: mu_RS(point) = {best["rs"]:.3f} '
            f'(cell {best["cell"]:.3f}) at x = {best["x"]:.0%}, '
            f'weights kf={best["kf"]:.0e} ku={best["ku"]:.0e} '
            f'fc={best["fc"]:.0f} -> stage M1 is justified')
    elif np.isfinite(best['rs']):
        log(f'  GATE NOT open on this grid: best mu_RS(point) = '
            f'{best["rs"]:.3f}.  Reported as exactly that; a wider weight')
        log('  search is the next probe, not a conclusion.')
    else:
        log('  every synthesis failed -- the slicing or the chain needs work')

    arr = np.array(rows, float)
    np.savez(os.path.join(OUT, 'musyn_sched_probe.npz'),
             rows=arr, columns=['node_frac', 'kf', 'ku', 'fc', 'mu_rp',
                                'rs_point', 'rs_cell', 'order'])
    with open(os.path.join(OUT, 'musyn_sched_probe.txt'), 'w') as fh:
        fh.write('M0 probe: see log_musyn_sched.txt for the annotated run\n')
        for r in rows:
            fh.write('  ' + ' '.join(f'{v:g}' for v in r) + '\n')
    log(f'\ntotal {time.time()-t0:.0f}s -> results/musyn_sched_probe.*')


if __name__ == '__main__':
    m0c() if 'm0c' in sys.argv[1:] else main()
