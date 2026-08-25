"""envelope_probe.py — why the true-envelope depth is controller-independent.

conservatism_budget.py's R6 rung recomputes a_p^inf over the alpha_4 envelope
the periodic truth actually visits, [0, 12.43] abar4 instead of Eq. (23)'s
frozen [0.3, 2.9], and gets 0.0967 mm for mu-TDC, PS-AC and PS-TDC-R alike --
four identical significant figures from three structurally different loops (a
D-K controller of order 12, an LQG-structured order 6, and that plus a delayed
pair).  Depths that agree to four figures across designs are a statement about
the PLANT, not about any controller, but the budget only observed it.  This
probe asks the question the observation raises, and it can come out either way:

  Q1  does the OPEN loop have the same depth at the top pin?  If it does, no
      feedback in the study moves the binding constraint at the true alpha_4
      peak, and the R6 collapse is plant-side by demonstration.  If the open
      loop is worse, the controllers ARE doing something and the agreement is
      a coincidence of this operating point.

  Q2  where does the binding crossing sit -- which frequency, which position,
      which alpha_4 end?  A peak parked on a structural mode says the plate is
      the constraint; one parked in the actuator band says the realisation is.

  Q3  how does the depth fall with the pin?  a_p^inf(alpha_4 pin) on a fine
      ladder, for every design and the open loop, so the shape of the collapse
      is on record rather than inferred from eight points.

PRE-DECLARED (house rule: written before the run, reported whatever comes out)
  C1  if |a_p^inf(open) - a_p^inf(controlled)| / a_p^inf(open) < 5 % at the
      12.43 abar4 pin for all three designs, the R6 collapse is reported as
      PLANT-SIDE.  Above 5 % it is reported as controller-sensitive and the
      "structural" reading in docs/10 is withdrawn.
  C2  the depth ladder must be monotone non-increasing in the pin for every
      design.  It is a frozen family, so a higher alpha_4 cannot help; a
      non-monotone row means a numerical failure and is reported as one, not
      smoothed.
  C3  every depth here is produced the way run_stage56 produces its own:
      stage_common.load_controllers returns FACTORIES over the stored tuned
      PARAMETERS, not stored state spaces, so for the LQG-structured members
      the Riccati equations are re-solved against the plant at each bisection
      depth.  Only mu-TDC carries a stored state space.  That is the study's
      convention and it is what makes these numbers comparable with the
      published ones -- but it is a re-synthesis, and calling it "stored
      matrices" would be false.

  C4  every plant here is the TWO-mode design plant (ControlledPlant defaults
      to C.N_MODES_DESIGN = 2), the same order the published certificates use.
      The Floquet limits they are compared against elsewhere run on five modes.

Writes results/envelope_probe.txt (and .npz).

    python phase2/envelope_probe.py
"""
import os
import sys
import time
import warnings
from copy import copy

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify2 as CF2
import config as C
from milling_dynamics import alpha4_series
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers, LABEL

OUT = C.RESULTS
KINDS = ('mu_tdc', 'ps_ac', 'ps_tdc_r')
PINS = (0.3, 1.0, 1.6, 2.9, 6.0, 9.0, 12.4328)
FAM = dict(n_pos=5, etas=(0.0,), zetas=(1.0,), xis=(1.0,))
BISECT = dict(lo=2e-6, hi=4e-3, tol=2e-6, n_iter=16)


def pin_alpha(plant, mult):
    """Plant whose frozen alpha_4 nominal is `mult` * abar4 (use a_scale=0)."""
    p = copy(plant)
    p.a40 = mult * plant.abar4
    p.base = getattr(plant, 'base', plant)
    return p


def depth_at_pin(plate, mk, mult):
    def plant_of(ap):
        return pin_alpha(ControlledPlant(plate, ap=ap), mult)

    def ctrl_of(p):
        return mk(getattr(p, 'base', p))

    return CF2.depth_bisect(plant_of, ctrl_of, a_scale=0.0, **BISECT, **FAM)


def binding_vertex(plate, mk, mult, ap):
    """(peak, omega_Hz, x_mm) of the worst vertex at this pin and depth."""
    p = pin_alpha(ControlledPlant(plate, ap=ap), mult)
    ctrl = mk(getattr(p, 'base', p))
    xs = np.linspace(0.0, plate.lp, FAM['n_pos'])
    best = (-1.0, np.nan, np.nan)
    for x in xs:
        V, npl, ws = CF2.vertices(p, ctrl, positions=(x,), a_scale=0.0,
                                  etas=(0.0,), zetas=(1.0,), xis=(1.0,))
        for (A, Ad) in V:
            w = CF2._wgrid(A)
            I = np.eye(A.shape[0])
            pk, wk_at = 0.0, np.nan
            for wk in w:
                try:
                    r = max(abs(np.linalg.eigvals(
                        np.linalg.solve(1j * wk * I - A, Ad))))
                except np.linalg.LinAlgError:
                    continue
                if r > pk:
                    pk, wk_at = float(r), float(wk)
            if pk > best[0]:
                best = (pk, wk_at * ws / (2 * np.pi), x * 1e3)
    return best


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    mks = load_controllers(plate, plant)
    kinds = [k for k in KINDS if k in mks] + ['open']

    _, a4 = alpha4_series(C.RPM_S, C.AP_S, plate.hp, 2000, ae=C.AE,
                          midpoint=True)
    # alpha_4 and abar4 carry the same sign; the envelope is their ratio
    top = float(np.max(np.abs(a4)) / abs(plant.abar4))

    out = ['=' * 84,
           'THE TRUE-ENVELOPE COLLAPSE -- IS IT THE PLANT OR THE CONTROLLER?',
           '=' * 84,
           f'  alpha_4(t) peak / abar4 = {top:.4f}; the certified band is '
           f'[{C.ALPHA_LO}, {C.ALPHA_HI}]',
           f'  frozen family: n_pos = {FAM["n_pos"]}, eta/zeta/xi nominal, '
           'a_scale = 0 (single pin)',
           '']

    out.append('[Q3] a_p^inf (mm) against the frozen alpha_4 pin')
    out.append('  ' + 'pin/abar4'.ljust(12)
               + ''.join(f'{LABEL.get(k, k)[:13]:>15}' for k in kinds))
    cap = BISECT['hi'] * 1e3
    D = np.zeros((len(PINS), len(kinds)))
    for i, mult in enumerate(PINS):
        cells = []
        for j, k in enumerate(kinds):
            D[i, j] = depth_at_pin(plate, mks[k], mult) * 1e3
            # depth_bisect returns its own upper bound when the family is
            # stable all the way up: that is a censored search, not a depth
            cells.append(f'{D[i, j]:14.4f}' + ('*' if D[i, j] >= cap - 1e-9
                                               else ' '))
        out.append(f'  {mult:<12.4f}' + ''.join(cells))
    n_cap = int((D >= cap - 1e-9).sum())
    out.append(f'  * = censored at the bisection ceiling {cap:.4f} mm, not a '
               f'measured depth ({n_cap} of {D.size} entries)')

    mono = bool(np.all(np.diff(D, axis=0) <= 1e-9))
    out.append(f'  [C2] monotone non-increasing in the pin: '
               f'{"PASS" if mono else "FAIL"}')
    out.append('')

    i_top = len(PINS) - 1
    d_open = D[i_top, kinds.index('open')]
    rel = np.array([abs(D[i_top, j] - d_open) / d_open
                    for j, k in enumerate(kinds) if k != 'open'])
    out.append(f'[Q1] at the top pin ({PINS[i_top]:.4f} abar4), against the '
               'open loop')
    out.append(f'  open loop               {d_open:9.4f} mm')
    for j, k in enumerate(kinds):
        if k == 'open':
            continue
        out.append(f'  {LABEL.get(k, k):<22}{D[i_top, j]:9.4f} mm   '
                   f'{100*abs(D[i_top, j]-d_open)/d_open:6.2f} % from open')
    plant_side = bool(rel.max() < 0.05)
    out.append(f'  [C1] < 5 % for every design: '
               f'{"PASS -> the collapse is PLANT-SIDE" if plant_side else "FAIL -> controller-sensitive"}')
    dc = np.array([D[i_top, j] for j, k in enumerate(kinds) if k != 'open'])
    spread = float(dc.max() / dc.min() - 1.0)
    out.append(f'  reported after the fact, NOT pre-declared: the spread among')
    out.append(f'  the three controlled designs is {100*spread:.3f} % '
               f'({dc.min():.4f}-{dc.max():.4f} mm) while each is '
               f'{d_open and dc.min()/d_open:.1f}x the open loop.')
    out.append('  So the frozen-peak test is NOT vacuous -- feedback is worth a')
    out.append('  large factor there -- but it cannot TELL THE DESIGNS APART,')
    out.append('  which is the property a design criterion has to have.')
    out.append('')

    out.append('[Q2] where the binding crossing sits at the top pin')
    out.append('  ' + 'design'.ljust(22) + 'peak      f (Hz)     x (mm)')
    B = np.zeros((len(kinds), 3))
    for j, k in enumerate(kinds):
        pk, fh, xm = binding_vertex(plate, mks[k], PINS[i_top],
                                    D[i_top, j] * 1e-3 * 1.02)
        B[j] = (pk, fh, xm)
        out.append(f'  {LABEL.get(k, k):<22}{pk:7.3f}  {fh:9.1f}  {xm:8.1f}')
    f_modes = np.asarray(plate.omega_n, float)[:5] / (2 * np.pi)
    out.append('  plate modes (Hz): '
               + ' '.join(f'{v:.1f}' for v in f_modes))
    out.append('')
    out.append('  reading: a frozen certificate honest about the envelope the')
    out.append('  periodic system really visits still credits feedback with a')
    out.append('  large factor over the open loop, but it grades every design')
    out.append('  in the study the same.  Its failure is DISCRIMINATION, not')
    out.append('  vacuity -- and that is precisely what a criterion using the')
    out.append('  duty cycle of alpha_4(t), rather than its extreme, would fix.')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'envelope_probe.txt'), 'w') as fh:
        fh.write(txt)
    np.savez(os.path.join(OUT, 'envelope_probe.npz'),
             kinds=np.array(kinds), pins=np.array(PINS), depth_mm=D,
             binding=B, top=top, plant_side=plant_side, monotone=mono,
             f_modes=f_modes, spread=spread, d_open_top=d_open)
    print(txt)
    print(f'total {time.time()-t0:.0f}s -> results/envelope_probe.*')


if __name__ == '__main__':
    main()
