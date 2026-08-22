"""margin_landscape.py — can ANY delayed pair on the PS-AC base reach mu-TDC?

The frozen-base PS-TDC search pinned kpd at its -1.5 bound and still certified
only a_p^inf = 0.3934 mm against mu-TDC's 0.4364.  Two readings are possible:
the bound stops the search short, or no delayed pair on this base reaches the
target at all.  This script settles it structurally, not by more searching:

  1. sweep the whole (kpd, kdd) plane, far beyond the search bounds;
  2. screen each point with the SAME protocol constraints (nominal poles,
     Ms <= 2, effort);
  3. for every feasible point, one delay-independence check of the light
     vertex family at the TARGET depth 0.4364 mm -- pass/fail is the question;
  4. refine the passers (if any) with the same light bisection stage 5-6 uses,
     and score their J, so a passer that ruins the performance side is visible.

Writes results/margin_landscape.npz and a plain-text verdict.

    python phase2/margin_landscape.py
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
from eval2 import evaluate
from plate_model import build_plate
from plant_ss import ControlledPlant

OUT = C.RESULTS
TARGET_AP = 0.4364e-3            # mu-TDC's certified depth
KPD = np.linspace(-6.0, 2.0, 33)
KDD = np.linspace(-1.5, 3.0, 19)

LIGHT = dict(n_pos=5, etas=(0.0, C.ETA_MAX),
             zetas=(C.ZETA_LO, C.ZETA_HI), xis=(1.0,))


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f:
        base = dict(pickle.load(f)['ps_ac']['params'])

    plant_t = ControlledPlant(plate, ap=TARGET_AP + 2e-6)

    J = np.full((KPD.size, KDD.size), np.nan)
    feas = np.zeros((KPD.size, KDD.size), bool)
    hits = np.zeros((KPD.size, KDD.size), bool)
    n_f = n_h = 0
    for i, kp in enumerate(KPD):
        for j, kd in enumerate(KDD):
            c = K2.ps_tdc(plant, base, float(kp), float(kd))
            Jij = evaluate(plate, c)
            J[i, j] = Jij
            if Jij <= -99.0:                       # constraint screen failed
                continue
            feas[i, j] = True
            n_f += 1
            ct = K2.ps_tdc(plant_t, base, float(kp), float(kd))
            if CF2.di_stable(plant_t, ct, **LIGHT):
                hits[i, j] = True
                n_h += 1
        print(f'  kpd = {kp:+.2f}: {feas[i].sum()} feasible, '
              f'{hits[i].sum()} above the target   [{time.time()-t0:.0f}s]',
              flush=True)

    out = ['=' * 78,
           'CAN ANY DELAYED PAIR ON THE PS-AC BASE REACH MU-TDC\'S DEPTH?',
           '=' * 78,
           f'  grid    : kpd in [{KPD[0]:.1f}, {KPD[-1]:.1f}] x '
           f'kdd in [{KDD[0]:.1f}, {KDD[-1]:.1f}], '
           f'{KPD.size}x{KDD.size} = {KPD.size*KDD.size} points',
           f'  screen  : the protocol constraints (poles, Ms, effort)',
           f'  test    : delay-independence of the light family at '
           f'{TARGET_AP*1e3:.4f} mm',
           '',
           f'  feasible points          : {n_f}',
           f'  of them above the target : {n_h}']

    best = []
    if n_h:
        for i, j in zip(*np.nonzero(hits)):
            def plant_of(ap):
                return ControlledPlant(plate, ap=ap)

            def mk(p, kp=float(KPD[i]), kd=float(KDD[j])):
                return K2.ps_tdc(p, base, kp, kd)

            ap = CF2.depth_bisect(plant_of, mk, n_iter=12, **LIGHT)
            best.append((ap, float(KPD[i]), float(KDD[j]), float(J[i, j])))
            print(f'  refine kpd={KPD[i]:+.2f} kdd={KDD[j]:+.2f}: '
                  f'a_p^inf(light) = {ap*1e3:.4f} mm, J = {J[i, j]:+.4f}',
                  flush=True)
        best.sort(reverse=True)
        out.append('')
        out.append('  the passers, refined with the stage 5-6 light bisection:')
        out.append(f'    {"a_p^inf mm":>11} {"kpd":>7} {"kdd":>7} {"J":>9} '
                   '  inside the +-1.5 search box?')
        for ap, kp, kd, Jv in best:
            inside = abs(kp) <= 1.5 and abs(kd) <= 1.5
            out.append(f'    {ap*1e3:>11.4f} {kp:>7.2f} {kd:>7.2f} {Jv:>9.4f}'
                       f'   {"yes" if inside else "NO"}')
    else:
        out.append('')
        out.append('  NO feasible pair reaches the target depth: the bound is')
        out.append('  not what stops the frozen-base search -- the constraint')
        out.append('  set is.  On this base the delayed term cannot buy')
        out.append('  mu-TDC\'s certified depth at any gain.')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'margin_landscape.txt'), 'w') as fh:
        fh.write(txt)
    np.savez(os.path.join(OUT, 'margin_landscape.npz'),
             kpd=KPD, kdd=KDD, J=J, feas=feas, hits=hits,
             best=np.array(best) if best else np.zeros((0, 4)),
             target_ap=TARGET_AP)
    print(txt)
    print(f'total {time.time()-t0:.0f}s -> results/margin_landscape.*')


if __name__ == '__main__':
    main()
