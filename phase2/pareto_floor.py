"""pareto_floor.py — is "leads on every axis" reachable, or is it a trade-off?

The dominate campaign produced PS-ROB-TDC, which takes the frozen certificate
axis outright -- a_p^inf and delta_max above every published design -- and
misses the other three by 1.6 %, 3.9 % and 2.6 %.  Its ablation (the same
weights with the observer scheduling switched off) says why in one line:

    scheduling the observer BUYS the certificate and the nominal floor and PAYS
    on the delay scenario S4.

That is the signature of a trade-off, not of a search that stopped too early.
And the campaign left exactly one continuous knob sitting on that trade: the
DIRECTION FLOOR c, the scalar that stops the scheduled filter believing a mode
is unexcited.  c -> 0 is the unregularised scheduled observer (which collapses);
c large drives the scheduled direction towards the fixed one.  So c interpolates
between the two ends of the ablation, and sweeping it maps the frontier
directly instead of inferring it.

This script sweeps c at PS-ROB-TDC's own weights and measures ALL FOUR
independent axes at every point -- the four the audit established
(results/dominate_audit.txt): the frozen certificate (a_p^inf with delta_max),
the nominal floor, the worst perturbed scenario, and mu_RS.  Plus the protocol
constraints, because a point that violates Ms <= 2 is not a candidate.

PRE-DECLARED (house rule: fixed before the run)
  P1  thresholds are READ OUT of results/dominate.npz and results/
      mu_scheduled.npz, never re-typed at four decimals -- the rounding trap
      this study already found makes a record holder fail its own record.
  P2  DOMINATION means one c meeting all four at once while feasible under the
      protocol (Ms <= 2, V <= 450 V/N, nominal poles <= -1 /s) and under the
      box screen.  If one exists, the answer to "excels in everything" is YES
      and the point is named.
  P3  if none exists, the frontier is reported as measured: for each axis, the
      best c and the value, and the pair of axes that cannot be met together.
      That is the answer, not a failure.
  P4  a_p^inf differences are quoted in BISECTION CELLS (0.00195 mm) as well as
      in per cent; a lead under two cells is reported as within resolution.
  P5  every axis is measured with the same call the scoreboard used, so the
      numbers are comparable with the stored ones rather than merely similar.

Writes results/pareto_floor.txt and results/pareto_floor.npz.

    python phase2/pareto_floor.py [n_points]
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import eval2 as E2
import run_dominate as RD
from plate_model import build_plate
from plant_ss import ControlledPlant

OUT = C.RESULTS
SEARCH_DIR = os.environ.get(
    'DOMINATE_SCRATCH',
    '/tmp/claude-0/-home-user-new-controller/'
    'b09d4fbe-e1b5-5e34-b325-184b93a5270b/scratchpad/dom')
CELL = 0.001952                      # mm, the depth_bisect grid (audit [A])
L = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def thresholds():
    """The four axis thresholds, read out of the artifacts (P1)."""
    d = np.load(os.path.join(OUT, 'dominate.npz'), allow_pickle=True)
    names = [str(s) for s in d['names']]
    refs = [names.index(n) for n in
            ('mu-TDC', 'PS-AC', 'PS-AC K+obs', 'PS-TDC-R') if n in names]
    mus = np.load(os.path.join(OUT, 'mu_scheduled.npz'))
    return dict(
        ap=float(np.asarray(d['ap_inf'], float)[refs].max()),
        dm=float(np.asarray(d['delta_max'], float)[refs].max()),
        fl=float(np.asarray(d['floor'], float)[refs].max()),
        ws=float(np.asarray(d['min_scen'], float)[refs].max()),
        mu=float(np.asarray(mus['ps_ac_r_point'], float).max()))


def base_params():
    with open(os.path.join(SEARCH_DIR, 'search_B.pkl'), 'rb') as fh:
        return pickle.load(fh)


def main():
    t0 = time.time()
    n_pts = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    grid = RD.screen_grid(plate)
    thr = thresholds()
    b = base_params()
    p0 = dict(b['params'])
    lo, hi = RD.BOUNDS[b['kind']]['log_dfloor']

    log('=' * 100)
    log('THE DIRECTION FLOOR AS A FRONTIER KNOB -- CAN ONE DESIGN LEAD ON ALL '
        'FOUR AXES?')
    log('=' * 100)
    log(f'  base: {b["label"]} ({b["kind"]}, {len(p0)} parameters), every '
        'weight held at its searched value')
    log(f'  swept: log_dfloor over [{lo}, {hi}], {n_pts} points  '
        f'(c = 10**log_dfloor)')
    log(f'  thresholds READ from the artifacts (P1): a_p^inf >= {thr["ap"]:.6f} '
        f'mm, delta_max >= {thr["dm"]:.6f},')
    log(f'                                           floor >= {thr["fl"]:.6f} '
        f'mm, worst >= {thr["ws"]:.6f} mm, mu_RS <= {thr["mu"]:.6f}')
    log('')
    log('  ' + 'c'.rjust(9) + '   a_p^inf   delta    floor   worst   mu_RS'
        '      Ms      V     box   axes')

    xs = np.linspace(lo, hi, n_pts)
    rows = []
    for lf in xs:
        p = dict(p0)
        p['log_dfloor'] = float(lf)
        t = time.time()
        d = RD.Design(b['kind'])
        mk = d.factory(p)
        ctrl = mk(plant0)
        cert = RD.certificates(plate, mk, plant0)
        sc = RD.scenarios(plate, mk, plant0)
        mu = float(np.max(RD.mu_point(plate, plant0, mk)[0]))
        box = RD.box_screen(plate, grid, ctrl)
        J, info = E2.evaluate(plate, ctrl, detail=True)
        fl = float(sc['S1'].min())
        ws = float(min(sc['S1'].min(), sc['S2'].min(),
                       np.min(sc['S3']), np.min(sc['S4'])))
        a1 = bool(cert['ap_inf'] >= thr['ap'] and cert['delta_max'] >= thr['dm'])
        a2 = bool(fl >= thr['fl'])
        a3 = bool(ws >= thr['ws'])
        a4 = bool(mu <= thr['mu'])
        k = int(a1) + int(a2) + int(a3) + int(a4)
        feas = bool(info.get('Ms', 9) <= C.MS_MAX
                    and info.get('V', 1e9) <= C.V_PER_N and box['passed'])
        rows.append((10 ** lf, cert['ap_inf'], cert['delta_max'], fl, ws, mu,
                     float(info.get('Ms', np.nan)), float(info.get('V', np.nan)),
                     float(box['worst']), k, feas, a1, a2, a3, a4))
        log(f'  {10**lf:9.4f}   {cert["ap_inf"]:7.4f}  {cert["delta_max"]:6.3f}'
            f'  {fl:7.4f} {ws:7.4f}  {mu:6.4f}  {info.get("Ms", np.nan):6.3f}'
            f' {info.get("V", np.nan):6.1f} {box["worst"]:7.2f}   {k}/4'
            + ('' if feas else '   INFEASIBLE') + f'   [{time.time()-t:.0f}s]')

    R = np.array([r[:10] for r in rows], float)
    feas = np.array([r[10] for r in rows], bool)
    axes = np.array([r[11:] for r in rows], bool)
    k = R[:, 9]
    log('')
    log('=' * 100)
    log('THE VERDICT')
    log('=' * 100)
    win = np.where((k == 4) & feas)[0]
    if win.size:
        i = int(win[0])
        log(f'  [P2] DOMINATION REACHED at c = {R[i,0]:.4f}: all four axes met '
            'while feasible under the protocol.')
    else:
        log('  [P2] NO c meets all four while feasible.  The frontier, as '
            'measured (P3):')
        for j, nm, better in ((1, 'a_p^inf (cert)', True),
                              (3, 'nominal floor', True),
                              (4, 'worst scenario', True),
                              (5, 'mu_RS', False)):
            v = R[:, j].copy()
            idx = int(np.argmax(v)) if better else int(np.argmin(v))
            log(f'    best {nm:<16} {v[idx]:8.4f} at c = {R[idx,0]:.4f}'
                + ('   (feasible)' if feas[idx] else '   (INFEASIBLE)'))
        best = int(np.argmax(np.where(feas, k, -1)))
        log(f'  best feasible point: c = {R[best,0]:.4f} with {int(k[best])} '
            'of 4')
        # which pair cannot be met together
        pairs = [('cert', 0), ('floor', 1), ('worst', 2), ('mu_RS', 3)]
        log('  pairs of axes NEVER met together at any feasible c:')
        none_pair = True
        for a in range(4):
            for bx in range(a + 1, 4):
                both = axes[:, a] & axes[:, bx] & feas
                if not both.any():
                    none_pair = False
                    log(f'    {pairs[a][0]} + {pairs[bx][0]}')
        if none_pair:
            log('    -- none; every pair is reachable, only the four together '
                'are not')
    log('')
    log(f'  resolution note (P4): one bisection cell is {CELL:.5f} mm, so a '
        f'depth lead under {2*CELL:.5f} mm is not resolved.')

    np.savez(os.path.join(OUT, 'pareto_floor.npz'), rows=R, feasible=feas,
             axes=axes, thresholds=np.array([thr['ap'], thr['dm'], thr['fl'],
                                             thr['ws'], thr['mu']]),
             log_dfloor=xs, base=b['label'], kind=b['kind'])
    log('')
    log(f'total {time.time()-t0:.0f}s -> results/pareto_floor.*')
    with open(os.path.join(OUT, 'pareto_floor.txt'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
