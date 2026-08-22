"""Why the proposed law schedules the gain and not the observer (Section 5.2).

Scheduling the Kalman filter on the tool position looks like the better design:
it gives the highest nominal limit and the smallest spread across the pass.  It
then collapses to zero under the reference's own +10 % mass / -10 % stiffness
box -- a 9.5 % drop in the natural frequencies.

The failure is not in cutting stability.  It is in the nominal closed loop, with
NO CUTTING AT ALL: at mid-edge the second mode passes through a node, so the
assumed disturbance direction E(l_P/2) carries no information about it, the
scheduled filter gives that mode almost no estimation gain, and a 9.5 %
frequency error is enough to push the estimation loop unstable.

This script recomputes that table and writes it to results/observer_collapse.txt
so Section 5.2 quotes a measured number.

    python phase2/observer_collapse.py
"""
import os

import numpy as np

import config as C
from objective import nominal_poles
from plant_ss import ControlledPlant
from scenarios import _perturbed, paper_box_scale
from stage_common import load_controllers

SHOW = [('lqg', 'LQG (fixed)'),
        ('ps_ac', 'K(x_P) only  -- the proposed law'),
        ('ps_ac_obs', 'L(x_P) only'),
        ('ps_ac_full', 'K and L together')]


def main():
    plate = C.build_plate(n_modes=C.N_MODES, patch=C.PATCH_SIDE)
    plant = ControlledPlant(plate, ap=C.AP_S)
    made = load_controllers(plate, plant)
    scale = paper_box_scale(0.10, -0.10, C.N_MODES)
    drop = 100.0 * (1.0 - float(scale[0]))
    pl2 = _perturbed(plate, scale, 1.0)
    x_mid = 0.5 * plate.lp

    out = ['=' * 78,
           'SECTION 5.2 - WHY THE OBSERVER IS NOT SCHEDULED',
           '=' * 78,
           f"  perturbation : the reference's +10 % mass / -10 % stiffness box",
           f'                 -> the natural frequencies drop by {drop:.1f} %',
           f'  evaluated at : x_P = {x_mid*1e3:.1f} mm (mid-edge, the node of',
           '                 the second mode), NO CUTTING',
           '',
           f'  {"controller":<36}{"max Re(poles) 1/s":>20}',
           '  ' + '-' * 56]

    poles = {}
    for key, label in SHOW:
        ctrl = made[key](plant)
        ss, pd = ctrl.at(x_mid, 0.0)
        ev = nominal_poles(pl2, ss, pd)
        mre = float(np.max(ev.real))
        poles[key] = mre
        flag = '   <-- unstable' if mre > 0 else ''
        out.append(f'  {label:<36}{mre:>20.2f}{flag}')

    out += ['',
            '  The two variants that schedule the observer are the ones that',
            '  lose the nominal loop.  Scheduling the gain alone stays where',
            '  the fixed design is, to within a fraction of a percent.',
            '',
            '  mode shape at mid-edge (design model):']
    xs, D = plate.D_top_edge(401)
    i = int(np.argmin(np.abs(xs - x_mid)))
    row = D[i, :C.N_MODES_DESIGN]
    out.append(f'    D(l_P/2) = {np.array2string(row, precision=4)}'
               f'   -> |D_2|/|D_1| = {abs(row[1])/abs(row[0]):.2e}')
    out.append('    so E(l_P/2) ~ [0; D_1; 0]: the assumed disturbance carries')
    out.append('    no second-mode content for the scheduled filter to use.')

    txt = '\n'.join(out) + '\n'
    path = os.path.join(C.RESULTS, 'observer_collapse.txt')
    with open(path, 'w') as fh:
        fh.write(txt)
    print(txt)
    print('written to', path)


if __name__ == '__main__':
    main()
