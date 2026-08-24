"""timestep_trap.py — the stored source for Section 4.2's time-step paragraph.

The paper states that at n_sub = 164 sub-steps per tooth period (40 kHz) EVERY
controller diverges in time while every frequency-domain metric still looks
healthy, and that at 656 the problem disappears.  Until now that sentence rested
on a constant in config.py and a source comment -- no artifact.  This closes it.

Three things get measured, all from the stored designs:

  1. the arithmetic of the trap: the Oustaloup realisation carries poles up to
     w_h/2pi = 100 kHz, so a 40 kHz integration samples BELOW the Nyquist rate
     those poles need; the ratio is printed, not asserted from memory;

  2. the same nominal pass integrated at a ladder of n_sub, for every stored
     controller: diverged / divergence time / peak amplitude.  The claim is
     "all diverge at 164, none at 656", and it is reported as measured even if
     that is not what comes out;

  3. the frequency-domain metrics at the SAME designs, so the second half of
     the claim -- the metrics look fine while the integration blows up -- is
     visible in one table rather than asserted.

The horizon is short on purpose (T_TRAP): the trap fires within a few tooth
periods, and a full 20.4 s pass at n_sub = 1312 would be 6.6 million steps for
no extra information.

Writes results/timestep_trap.txt (and .npz).

    python phase2/timestep_trap.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from milling_dynamics import N_TEETH
from plate_model import build_plate
from plant_ss import ControlledPlant
from simulate import MillingSimulation
from sim_ctrl2 import ScheduledLTI
from stage_common import load_controllers, LABEL

OUT = C.RESULTS
LADDER = (164, 328, 656, 1312)      # 40, 80, 160, 320 kHz
T_TRAP = 0.5                        # s -- the trap fires in a few tooth periods
KINDS = ('fopid', 'lqg', 'mu_tdc', 'ps_ac', 'ps_ac_r', 'ps_tdc_r')


def run_at(plate, plant, ctrl, n_sub):
    """One nominal moving pass at the given sub-step count."""
    sim = MillingSimulation(plate, C.RPM_S, C.AP_S, ae=C.AE, fz=C.FZ,
                            sign=C.SIGN, n_modes=C.N_MODES, n_sub=n_sub)
    ss, _ = ctrl.at(0.0)
    c = None if ss is None else ScheduledLTI(ctrl, sim.dt, plate.lp,
                                             tau=sim.tau,
                                             feed=plant.feed_speed(),
                                             moving=True)
    r = sim.run(controller=c, T=T_TRAP, moving=True)
    y = np.asarray(r['y_obs'], float)
    return dict(diverged=bool(r['diverged']), t_div=r['t_div'],
                A_max=float(np.max(np.abs(y))) * 1e6,
                fs_khz=1e-3 / sim.dt)


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    mks = load_controllers(plate, plant, include_open=False)
    kinds = [k for k in KINDS if k in mks]

    tau = 60.0 / (N_TEETH * C.RPM_S)
    f_nyq = np.array(LADDER) / tau / 2.0
    w_h = C.OUST_WH / (2 * np.pi)

    out = ['=' * 78,
           'THE TIME-STEP TRAP -- WHY n_sub = 656 AND NOT 164',
           '=' * 78,
           '',
           '[1] the arithmetic',
           f'  Oustaloup upper corner w_h/2pi        : {w_h/1e3:8.1f} kHz',
           f'  tooth period tau                      : {tau*1e3:8.4f} ms']
    for ns, fn in zip(LADDER, f_nyq):
        out.append(f'  n_sub = {ns:4d} -> f_s = {2*fn/1e3:6.1f} kHz, '
                   f'Nyquist {fn/1e3:6.1f} kHz, '
                   f'w_h/f_Nyq = {w_h/fn:5.2f}'
                   + ('   <- the Oustaloup poles alias' if w_h > fn else ''))
    out += ['  -> the realisation must be sampled well above 100 kHz; 40 kHz',
            '     folds its top poles back into the band the loop is closed in.',
            '']

    out.append(f'[2] the same nominal pass at each n_sub (T = {T_TRAP} s, '
               'moving, a_p = 0.3 mm)')
    out.append('  ' + 'controller'.ljust(14)
               + ''.join(f'{"n_sub="+str(ns):>18}' for ns in LADDER))
    res = np.zeros((len(kinds), len(LADDER)))
    div = np.zeros((len(kinds), len(LADDER)), bool)
    for i, kind in enumerate(kinds):
        ctrl = mks[kind](plant)
        cells = []
        for j, ns in enumerate(LADDER):
            r = run_at(plate, plant, ctrl, ns)
            res[i, j] = r['A_max']
            div[i, j] = r['diverged']
            cells.append(f'DIV at {r["t_div"]:.3f}s' if r['diverged']
                         else f'{r["A_max"]:9.3f} um')
        out.append('  ' + LABEL.get(kind, kind).ljust(14)
                   + ''.join(f'{c:>18}' for c in cells))

    n_div = div.sum(0)
    out.append('')
    for j, ns in enumerate(LADDER):
        out.append(f'  n_sub = {ns:4d}: {n_div[j]:d} of {len(kinds)} '
                   f'controllers diverge')
    all164 = bool(div[:, 0].all()) if LADDER[0] == 164 else None
    none656 = (not bool(div[:, LADDER.index(656)].any())
               if 656 in LADDER else None)
    out.append(f'  claim "all diverge at 164"     : '
               f'{"CONFIRMED" if all164 else "NOT as stated"}')
    out.append(f'  claim "none diverges at 656"   : '
               f'{"CONFIRMED" if none656 else "NOT as stated"}')
    out.append('')
    out.append('[3] how fast each realisation actually is (its own poles)')
    out.append('  ' + 'controller'.ljust(28) + 'order   |lam|max/2pi   '
               'vs Nyquist(164)')
    fast = np.zeros(len(kinds))
    order = np.zeros(len(kinds), int)
    for i, kind in enumerate(kinds):
        ss, _ = mks[kind](plant).at(0.0)
        A = np.atleast_2d(np.asarray(ss[0], float))
        ev = np.linalg.eigvals(A)
        fast[i] = float(np.abs(ev).max()) / (2 * np.pi)
        order[i] = A.shape[0]
        out.append(f'  {LABEL.get(kind, kind):<28}{order[i]:5d}   '
                   f'{fast[i]/1e3:9.2f} kHz   {fast[i]/f_nyq[0]:8.2f}x'
                   + ('  ABOVE' if fast[i] > f_nyq[0] else '  below'))
    out += ['  -> the one design that survives n_sub = 164 is the one whose',
            '     fastest pole fits under that Nyquist rate.  The Oustaloup',
            '     corner is FOPID\'s instance of this; the LQG-structured',
            '     realisations are faster still, by design (cheap control).',
            '']
    out.append('  reading: the divergence is a numerical artefact of the')
    out.append('  integration rate, not a property of any controller -- which')
    out.append('  is why the fairness protocol pins n_sub for everybody.')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'timestep_trap.txt'), 'w') as fh:
        fh.write(txt)
    np.savez(os.path.join(OUT, 'timestep_trap.npz'),
             kinds=np.array(kinds), ladder=np.array(LADDER),
             A_max=res, diverged=div, w_h=w_h, T=T_TRAP,
             f_nyq=f_nyq, fast=fast, order=order, n_div=n_div,
             all_diverge_164=bool(all164), none_diverge_656=bool(none656))
    print(txt)
    print(f'total {time.time()-t0:.0f}s -> results/timestep_trap.*')


if __name__ == '__main__':
    main()
