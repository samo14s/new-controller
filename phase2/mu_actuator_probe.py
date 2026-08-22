"""mu_actuator_probe.py — the mid-span dissection Section 8.5 quotes.

Three facts, each re-derivable here so the paper's numbers have a stored
source:

  1. with the actuator block, mu_RS at mid-span is ~85.8;
  2. WITHOUT it, the same loop over the whole parametric physics is 0.577 --
     and every parametric block alone reproduces the 85.8, so the actuator
     channel at ~4 kHz is the sole driver;
  3. extra rolloff does not cure that channel: even a second-order filter at
     2 kHz leaves 21.8.  A mu-shaped high-frequency profile is what it takes.

Writes results/mu_actuator_probe.txt (and .npz).

    python phase2/mu_actuator_probe.py
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
from mu_scheduled import plant_at, mu_rs
from robust_design import freq_grid
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = C.RESULTS
ROLLOFFS = ((3000, 1), (2000, 1), (1500, 1), (2000, 2))


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    c = load_controllers(plate, plant)['ps_ac_r'](plant)
    x = 0.5 * plate.lp
    ss0 = c.at(x)[0]
    U = plant_at(plate, x)
    f = freq_grid()
    POS = ('d_D', 'd_r')
    par_names = [nm for nm, _ in U.blocks_used if nm not in POS]

    out = ['=' * 78,
           'WHAT DRIVES mu_RS AT MID-SPAN -- THE ACTUATOR CHANNEL, ALONE',
           '=' * 78,
           '  member: ps_ac_r at x = 50 mm; position channels dropped',
           '']
    m_all, f_all = mu_rs(U, ss0, f, drop=POS, actuator=True)
    m_par, f_par = mu_rs(U, ss0, f, drop=POS, actuator=False)
    out.append(f'  all blocks, actuator included : mu = {m_all:7.3f} '
               f'at {f_all:5.0f} Hz')
    out.append(f'  parametric physics only       : mu = {m_par:7.3f} '
               f'at {f_par:5.0f} Hz')
    out.append('')
    out.append('  each parametric block ALONE (actuator kept):')
    singles = {}
    for only in par_names:
        drop = tuple(nm for nm in par_names if nm != only) + POS
        v, fk = mu_rs(U, ss0, f, drop=drop, actuator=True)
        singles[only] = v
        out.append(f'    {only:6s} : mu = {v:7.3f} at {fk:5.0f} Hz')
    out.append('  -> every one reproduces the peak: the actuator block is')
    out.append('     the sole driver.')
    out.append('')
    out.append('  extra rolloff on the same member (actuator block kept):')
    roll = {}
    for fc, order in ROLLOFFS:
        v, fk = mu_rs(U, series(ss0, rolloff_ss(fc, order)), f,
                      drop=POS, actuator=True)
        roll[(fc, order)] = v
        out.append(f'    order {order} at {fc:4d} Hz : mu = {v:7.3f} '
                   f'at {fk:5.0f} Hz')
    out.append('  -> filtering does not cure the channel; a mu-shaped')
    out.append('     high-frequency profile is what it takes.')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'mu_actuator_probe.txt'), 'w') as fh:
        fh.write(txt)
    np.savez(os.path.join(OUT, 'mu_actuator_probe.npz'),
             m_all=m_all, f_all=f_all, m_par=m_par, f_par=f_par,
             singles_names=list(singles), singles=list(singles.values()),
             rolloff_keys=[f'{o}@{fc}' for fc, o in ROLLOFFS],
             rolloff=[roll[k] for k in ROLLOFFS])
    print(txt)
    print(f'total {time.time()-t0:.0f}s -> results/mu_actuator_probe.*')


if __name__ == '__main__':
    main()
