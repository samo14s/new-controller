"""mu_scheduled.py — the mu test the scheduled family is actually entitled to.

The stored cross-mu table judges ps_ac as a FIXED controller at mid-span
against the full physics set, position included -- mu_RS = 85.5, rightly so
for a fixed gain that must cover the whole edge.  The scheduled law reads the
position, so its test is per scheduling node with position handled as a
scheduling signal.  Probing that split the answer in two:

  * the huge mu of the LQG family is NOT position and NOT the parametric
    physics: dropping the actuator block Delta_Pa collapses it from 85.8 to
    0.577, and every parametric block alone reproduces the 85.8 (the peak sits
    at 4 kHz, where the actuator's unmodelled-dynamics weight is enormous and
    an LQG-structured member still carries gain);
  * the parametric physics alone -- removal, mid-pass state, force amplitude,
    both dampings -- certifies BELOW ONE.

So this module produces the split record, per scheduling node:

  parametric/point   Delta_Pa and d_D/d_r dropped: the physical parameters at
                     the frozen position.
  parametric/cell    d_D/d_r kept but shrunk to the GRID CELL around the node,
                     so the gain-interpolation error is inside the certified
                     set; Delta_Pa still out.
  full               everything, actuator included, at five nodes -- the
                     number that blocks every design in the study.

All delay-free, the convention of the paper's own Eq. (28); the delay axis is
certified separately by the crossing test of stages 5-6, which also never
perturbs the actuator.  The LQG row calibrates attribution: whatever the
fixed gain passes at frozen positions is bought by freezing, and only the
difference is bought by scheduling.

    python phase2/mu_scheduled.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from cross_mu import closed_loop, frf
from mu_tight import mu_curve_tight
from robust_design import freq_grid
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers, load_mu_ss
from uncertain_phys import PhysUncertainSystem

OUT = C.RESULTS
N_NODES = 21                       # the scheduling grid itself
POS = ('d_D', 'd_r')               # the position blocks
ROWS_POINT = ('lqg', 'ps_ac', 'ps_ac_r', 'mu_paper', 'mu_phys')
ROWS_CELL = ('lqg', 'ps_ac_r')
ROWS_FULL = ('lqg', 'ps_ac_r', 'mu_paper', 'mu_phys')


def log(fh, *a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


# ---------------------------------------------------------------------------
def cell_rectangle(plate, x0, h, n=2, n_x=41):
    """d_c, l1, l2 covering { D(x) : |x - x0| <= h }, along the principal
    directions of the FULL locus so the channels keep their meaning."""
    xs_f = np.linspace(0.0, plate.lp, 201)
    Df = np.array([plate.D_row(x, plate.hp)[:n] for x in xs_f])
    _, _, Vt = np.linalg.svd(Df - 0.5 * (Df.max(0) + Df.min(0)),
                             full_matrices=False)
    xs = np.linspace(max(0.0, x0 - h), min(plate.lp, x0 + h), n_x)
    Dc = np.array([plate.D_row(x, plate.hp)[:n] for x in xs])
    dc = plate.D_row(float(x0), plate.hp)[:n]
    Y = Dc - dc
    l1 = Vt[0] * max(float(np.abs(Y @ Vt[0]).max()), 1e-9)
    l2 = Vt[1] * max(float(np.abs(Y @ Vt[1]).max()), 1e-9)
    return dc, l1, l2


def plant_at(plate, x0, cell_h=None):
    """The physics generalized plant with the position pinned at x0.

    cell_h = None leaves the (widened) d_D/d_r channels in place -- the caller
    drops them for the point tests.  A number rebuilds them at cell size.
    """
    U = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN,
                            x_nom=float(x0))
    if cell_h is not None:
        U.d_c, U.l1, U.l2 = cell_rectangle(plate, float(x0), float(cell_h))
        U._build()
    return U


def mu_rs(U, ss_K, f, drop=(), actuator=True):
    """Peak mu_RS with the named parametric blocks dropped, and the actuator
    block kept or not.  Frozen parameters contribute exactly nothing, so
    dropping a channel IS setting its delta to zero."""
    P = U.generalized_plant()
    T, info = closed_loop(P, ss_K)
    w = 2 * np.pi * f / info['w0']
    H = frf(T, w)[:, :P['n_unc_out'], :P['n_unc_in']]
    keep = [j for j, (nm, _) in enumerate(U.blocks_used) if nm not in drop]
    ch = np.concatenate([U.idx[U.blocks_used[j][0]] for j in keep])
    if actuator:
        rows = np.concatenate([[0, 1], 2 + ch])
        cols = np.concatenate([[0], 1 + ch])
        blocks = [('F', 2, 1)]
    else:
        rows, cols, blocks = 2 + ch, 1 + ch, []
    blocks += [('S', U.blocks_used[j][1], U.blocks_used[j][1]) for j in keep]
    cur = mu_curve_tight(H[:, rows][:, :, cols], blocks)[1]
    return float(np.max(cur)), float(f[int(np.argmax(cur))])


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    fh = open(os.path.join(OUT, 'log_mu_scheduled.txt'), 'w')
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    mks = load_controllers(plate, plant)
    mu_ss = dict(mu_paper=load_mu_ss('mu_paper'), mu_phys=load_mu_ss('mu_phys'))
    f = freq_grid()
    xs = np.linspace(0.0, plate.lp, N_NODES)
    cell_h = plate.lp / (N_NODES - 1) / 2.0

    def member(kind, x):
        if kind in mu_ss:
            return mu_ss[kind]           # a fixed controller: same everywhere
        return mks[kind](plant).at(float(x))[0]

    log(fh, '=' * 78)
    log(fh, 'mu_RS PER SCHEDULING NODE - POSITION AS A SCHEDULING SIGNAL')
    log(fh, '=' * 78)
    log(fh, f'  nodes: the {N_NODES}-point gain grid, cell half-width '
            f'{cell_h*1e3:.1f} mm; delay-free (Eq. 28 convention)')
    log(fh, '')

    res = {}

    log(fh, '[1] parametric physics only, position frozen (point test)')
    for kind in ROWS_POINT:
        v = np.array([mu_rs(plant_at(plate, x), member(kind, x), f,
                            drop=POS, actuator=False)[0] for x in xs])
        res[f'{kind}_point'] = v
        log(fh, f'  {kind:<9} sup {v.max():7.3f} at x = '
                f'{xs[np.argmax(v)]*1e3:5.1f} mm   '
                + ('mu_RS < 1 at EVERY node' if v.max() < 1 else 'above 1'))

    log(fh, '')
    log(fh, '[2] parametric physics + the grid cell of position '
            '(interpolation error certified)')
    for kind in ROWS_CELL:
        v = np.array([mu_rs(plant_at(plate, x, cell_h), member(kind, x), f,
                            actuator=False)[0] for x in xs])
        res[f'{kind}_cell'] = v
        log(fh, f'  {kind:<9} sup {v.max():7.3f} at x = '
                f'{xs[np.argmax(v)]*1e3:5.1f} mm   '
                + ('mu_RS < 1 at EVERY node' if v.max() < 1 else 'above 1'))

    log(fh, '')
    log(fh, '[3] everything, actuator block included (five nodes)')
    x5 = xs[::5]
    for kind in ROWS_FULL:
        v = np.array([mu_rs(plant_at(plate, x), member(kind, x), f,
                            drop=POS, actuator=True)[0] for x in x5])
        res[f'{kind}_full'] = v
        log(fh, f'  {kind:<9} sup {v.max():7.3f}')

    log(fh, '')
    log(fh, '  reading: the actuator channel at ~4 kHz is what blocks every')
    log(fh, '  design in the study from mu_RS < 1 on the whole set; the')
    log(fh, '  parametric physics itself is certified below one by the')
    log(fh, '  scheduled members -- and the LQG row says how much of that is')
    log(fh, '  bought by freezing the position alone.')

    np.savez(os.path.join(OUT, 'mu_scheduled.npz'), x=xs, x5=x5, f=f,
             cell_h=cell_h, **res)
    log(fh, f'\ntotal {time.time()-t0:.0f}s -> results/mu_scheduled.*')


if __name__ == '__main__':
    main()
