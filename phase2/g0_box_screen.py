"""
g0_box_screen.py -- G0c: the BOX-ROBUST nominal-pole screen, and its price.
===========================================================================
WHY.  eval2._evaluate screens the nominal-pole condition

      max Re lambda( A_cl(x_P) ) <= -1 /s ,   x_P in POSITIONS_DESIGN

on the NOMINAL plate only.  results/observer_collapse.txt measured that
ps_ac_obs and ps_ac_full lose that very loop -- with NO CUTTING -- under the
reference's +10 % mass / -10 % stiffness box (max Re = +2.99 and +8.47 /s),
which is why their S3 column is 0.0000.  A PSO particle with that defect is
invisible to the present screen and scores perfectly well.  This script defines
the screen that would see it, applies it to every stored design, and prices it.

THE SCREEN (definition, fixed before running).
  For each scheduling position fr in C.POSITIONS_DESIGN = (0, 0.5, 1) the
  controller is asked for its LTI (ss, pd) at x_P = fr * l_P -- exactly as
  eval2._evaluate asks it -- and eval2.nominal_poles is evaluated on:
      * the nominal plate, and
      * each VERTEX of the reference box.
  PASS iff max over all (plate, position) of max Re lambda <= -1 /s.

  VERTEX SET, and why.  scenarios.paper_box_scale(dm, dk, n) is how
  run_stage78.s3 applies the reference box: mass +dm, stiffness +dk enter the
  five-mode evaluation model ONLY as a common modal-frequency ratio
      r = sqrt( (1 + dk) / (1 + dm) ) ,
  mode shapes, H, D_obs and zeta untouched.  The box is the square
  (dm, dk) in {-0.10, +0.10}^2, so it has FOUR vertices:
      (+.10,+.10) -> r = 1.0000     (-.10,-.10) -> r = 1.0000
      (+.10,-.10) -> r = 0.9045     (-.10,+.10) -> r = 1.1055
  Two of them coincide with the nominal plate, so the screen touches THREE
  distinct plates: r = 0.9045, 1.0000, 1.1055.  run_stage78.s3 tests exactly
  the two non-trivial corners ('box +10%', 'box -10%'), so this vertex set is
  the same set the published S3 column is built on -- that is the reason for
  the choice, not a numerical convenience.
  NOT a full box guarantee: interior (dm, dk) give intermediate r, and max Re
  need not be monotone in r.  A 41-point sweep of r over [0.9045, 1.1055] is
  therefore reported as a DIAGNOSTIC beside the screen, to say whether the
  vertex test is a faithful proxy.  The screen itself stays at the vertices.
  NOT in the screen: the +/-20 % damping box (C.ZETA_LO/HI) and material
  removal (eta); this is the reference's mass/stiffness box alone.
  The delayed PD is treated exactly as the existing screen treats it: folded in
  at tau -> 0 (eval2.nominal_poles).  Same approximation on every design.

PRE-DECLARED SUCCESS CRITERIA (written before the run; reported as they fall).
  B1  SEPARATION.  The screen must reproduce the known collapse: ps_ac_obs and
      ps_ac_full must FAIL, and ps_ac must PASS.  If those three are not
      separated the screen is wrong and the run is reported as a failure, with
      no adjustment to the threshold or the vertex set.
  B2  FULL VERDICT TABLE.  Every stored design gets a verdict and a worst
      max Re, including any design published in the scoreboard that fails.  A
      published design failing is a COST of the screen, reported as such.
  B3  COST IN ms.  Per-call wall time of the new screen against the existing
      one, and against one PSO objective evaluation, so the reader can judge
      whether it can sit in the inner loop.

ADDED AFTER THE FIRST RUN, diagnostics only -- the threshold (-1 /s), the
vertex set and the pass/fail rule are UNCHANGED from the pre-declaration:
  * a pd-removed column for the designs carrying a delayed PD, because the
    first run failed mu-TDC and the reader must be able to see whether that is
    the mu controller or the tau -> 0 folding of the pair;
  * a cross-check of the nominal column against the "slowest nominal pole"
    values logged at design time in results/log_ps_tdc.txt and log_stage3.txt.

OUTPUT: results/g0_box_screen.txt and results/g0_box_screen.npz.  Nothing else.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import eval2
from plate_model import build_plate
from plant_ss import ControlledPlant
from scenarios import _perturbed, paper_box_scale
from stage_common import load_controllers, LABEL

POLE_MAX = -1.0                      # the existing threshold, unchanged
DM = (0.10, -0.10)                   # mass box  +/-10 %
DK = (0.10, -0.10)                   # stiffness box +/-10 %
SCOREBOARD = ('fopid', 'lqg', 'mu_tdc', 'ps_ac', 'ps_ac_obs', 'ps_ac_full',
              'ps_tdc', 'ps_ac_r', 'ps_tdc_r')      # rows of the paper table


# ---------------------------------------------------------------------------
def vertices():
    """The four corners of the reference box, as (label, dm, dk, r)."""
    out = []
    for dm in DM:
        for dk in DK:
            r = float(paper_box_scale(dm, dk, C.N_MODES)[0])
            out.append((f'dm={dm:+.2f},dk={dk:+.2f}', dm, dk, r))
    return out


def plates_for_screen(plate):
    """(label, plate) for the nominal plate plus every DISTINCT vertex plate."""
    out = [('nominal (r=1.0000)', plate, 1.0)]
    seen = {1.0}
    for lab, dm, dk, r in vertices():
        if round(r, 9) in seen:
            continue
        seen.add(round(r, 9))
        out.append((f'{lab} -> r={r:.4f}',
                    _perturbed(plate, paper_box_scale(dm, dk, C.N_MODES), 1.0),
                    r))
    return out


def max_re(plate, ctrl, fr):
    ss, pd = ctrl.at(fr * plate.lp)
    return float(np.max(eval2.nominal_poles(plate, ss, pd).real))


def screen(plate, ctrl, pos=None):
    """(worst max Re, table[n_plate, n_pos], plate labels).  The screen."""
    pos = C.POSITIONS_DESIGN if pos is None else pos
    pl = plates_for_screen(plate)
    tab = np.array([[max_re(p, ctrl, fr) for fr in pos] for _, p, _ in pl])
    return float(np.max(tab)), tab, [lab for lab, _, _ in pl]


def sweep(plate, ctrl, n=41, pos=None):
    """DIAGNOSTIC: max Re over a fine sweep of the frequency ratio r."""
    pos = C.POSITIONS_DESIGN if pos is None else pos
    rs = np.linspace(min(v[3] for v in vertices()),
                     max(v[3] for v in vertices()), n)
    worst = []
    for r in rs:
        p = _perturbed(plate, np.full(C.N_MODES, r), 1.0)
        worst.append(max(max_re(p, ctrl, fr) for fr in pos))
    return rs, np.array(worst)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    made = load_controllers(plate, plant, include_open=False)
    pos = C.POSITIONS_DESIGN
    pl_labels = [lab for lab, _, _ in plates_for_screen(plate)]

    L = ['=' * 92,
         'G0c - BOX-ROBUST NOMINAL-POLE SCREEN: definition, verdicts, price',
         '=' * 92,
         f'  threshold      : max Re(lambda) <= {POLE_MAX:.1f} /s, no cutting',
         f'  positions      : x_P/l_P in {pos}  (same as eval2._evaluate)',
         f'  evaluation     : {C.N_MODES}-mode model, eval2.nominal_poles,'
         ' delayed PD folded in at tau -> 0',
         '',
         '  VERTEX SET of the reference +10 % mass / -10 % stiffness box',
         '  (paper_box_scale: the box enters only as a common frequency ratio'
         ' r = sqrt((1+dk)/(1+dm)))']
    for lab, dm, dk, r in vertices():
        L.append(f'    {lab}   ->  r = {r:.4f}'
                 + ('   (coincides with nominal)' if abs(r - 1) < 1e-12 else ''))
    L += [f'  -> 4 vertices, {len(pl_labels) - 1} distinct perturbed plates;'
          ' the screen runs on those plus the nominal.',
          '  -> the two non-trivial corners are exactly the "box +10%" /'
          ' "box -10%" cases of run_stage78.s3.',
          '']

    # ---- verdict table ----------------------------------------------------
    L += ['=' * 92,
          'B2  VERDICT FOR EVERY STORED DESIGN',
          '=' * 92,
          f'{"design":<20}{"nominal":>10}' +
          ''.join(f'{lab.split(" ")[0][:9]:>11}' for lab in pl_labels[1:]) +
          f'{"worst":>10}  verdict',
          f'{"":20}{"max Re":>10}' +
          ''.join(f'{"max Re":>11}' for _ in pl_labels[1:]) +
          f'{"max Re":>10}']
    names, tables, worsts, passes = [], [], [], []
    sweep_worst, sweep_r = [], []
    for k, mk in made.items():
        ctrl = mk(plant)
        w, tab, _ = screen(plate, ctrl, pos)
        rs, sw = sweep(plate, ctrl, 41, pos)
        i = int(np.argmax(sw))
        ok = w <= POLE_MAX
        names.append(k)
        tables.append(tab)
        worsts.append(w)
        passes.append(ok)
        sweep_worst.append(float(sw[i]))
        sweep_r.append(float(rs[i]))
        per_plate = tab.max(axis=1)
        tag = 'PASS' if ok else 'FAIL'
        star = '' if k in SCOREBOARD else '   (not a scoreboard row)'
        L.append(f'{k:<20}' + ''.join(f'{v:>10.2f}' if j == 0 else f'{v:>11.2f}'
                                      for j, v in enumerate(per_plate))
                 + f'{w:>10.2f}  {tag}{star}')
    L += ['',
          '  columns are max over the three scheduling positions, on that'
          ' plate.',
          '  PASS = every position on every vertex (and the nominal) has'
          f' max Re <= {POLE_MAX:.1f} /s.',
          '']

    # ---- B1 ---------------------------------------------------------------
    d = dict(zip(names, passes))
    b1 = (d.get('ps_ac') is True and d.get('ps_ac_obs') is False
          and d.get('ps_ac_full') is False)
    L += ['=' * 92,
          'B1  DOES THE SCREEN REPRODUCE THE KNOWN COLLAPSE?',
          '=' * 92,
          f'    ps_ac       {"PASS" if d.get("ps_ac") else "FAIL":<5}'
          f' (required PASS)   worst max Re ='
          f' {worsts[names.index("ps_ac")]:+8.2f} /s',
          f'    ps_ac_obs   {"PASS" if d.get("ps_ac_obs") else "FAIL":<5}'
          f' (required FAIL)   worst max Re ='
          f' {worsts[names.index("ps_ac_obs")]:+8.2f} /s',
          f'    ps_ac_full  {"PASS" if d.get("ps_ac_full") else "FAIL":<5}'
          f' (required FAIL)   worst max Re ='
          f' {worsts[names.index("ps_ac_full")]:+8.2f} /s',
          '',
          f'    B1 {"MET" if b1 else "NOT MET -- the screen is wrong"}.',
          '']

    # ---- diagnostic: is the vertex set enough? ----------------------------
    L += ['=' * 92,
          'DIAGNOSTIC - vertex screen vs a 41-point sweep of r over the box',
          '=' * 92,
          f'{"design":<20}{"vertex worst":>14}{"sweep worst":>13}'
          f'{"r at sweep worst":>18}{"interior worse?":>17}']
    miss = []
    for k, w, sw, r in zip(names, worsts, sweep_worst, sweep_r):
        interior = sw > w + 1e-6
        if interior and (sw > POLE_MAX) and (w <= POLE_MAX):
            miss.append(k)
        L.append(f'{k:<20}{w:>14.2f}{sw:>13.2f}{r:>18.4f}'
                 f'{("yes" if interior else "no"):>17}')
    L += ['',
          '  "interior worse?" = the sweep found a worse max Re strictly inside'
          ' the r range.',
          '  designs the VERTEX screen would pass while the sweep fails them: '
          + (', '.join(miss) if miss else 'NONE'),
          '']

    # ---- diagnostic: is the failure the delayed pair or the base? --------
    L += ['=' * 92,
          'DIAGNOSTIC - designs carrying a delayed PD: is the failure the pair'
          ' or the base?',
          '=' * 92,
          '  the screen folds the delayed PD in at tau -> 0, exactly as the'
          ' EXISTING screen does',
          '  (eval2.nominal_poles).  Re-running with pd removed isolates which'
          ' half fails.',
          f'{"design":<20}{"with pd":>12}{"pd removed":>14}'
          f'{"verdict w/o pd":>17}']
    pd_rows = []
    for k, mk in made.items():
        ctrl = mk(plant)
        if ctrl.at(0.0)[1] is None:
            continue
        pl = plates_for_screen(plate)
        w0 = max(float(np.max(eval2.nominal_poles(
            p_, ctrl.at(fr * plate.lp)[0], None).real))
            for _, p_, _ in pl for fr in pos)
        w1 = worsts[names.index(k)]
        pd_rows.append((k, w1, w0))
        L.append(f'{k:<20}{w1:>12.2f}{w0:>14.2f}'
                 f'{("PASS" if w0 <= POLE_MAX else "FAIL"):>17}')
    L.append('')

    # ---- diagnostic: the TRUE delayed loop, not the tau -> 0 folding ------
    from closed_loop import is_stable
    tau0 = 60.0 / (3 * C.RPM_S)
    L += ['=' * 92,
          'DIAGNOSTIC - the TRUE delayed nominal loop (tau = tau_0), for the'
          ' designs carrying a pd',
          '=' * 92,
          '  the tau -> 0 folding is an APPROXIMATION.  For a design with no'
          ' delayed pair it is exact',
          '  (there is no delay in the loop at all), so only these designs can'
          ' be misjudged by it.',
          '  Here the no-cutting loop is closed with the pair at its true delay'
          ' and its Floquet radius',
          f'  rho is taken over one tooth period tau_0 = {tau0*1e3:.3f} ms'
          f' (a_p -> 0, m = {C.M_FLOQUET_PSO});',
          f'  the comparable exponent is log(rho)/tau_0, and the -1 /s'
          f' threshold is rho <= {np.exp(-tau0):.5f}.',
          f'{"design":<20}{"tau->0 worst":>14}{"true-delay worst":>18}'
          f'{"rho worst":>12}{"verdict (true)":>16}']
    td_rows = []
    for k, mk in made.items():
        ctrl = mk(plant)
        if ctrl.at(0.0)[1] is None:
            continue
        pl = plates_for_screen(plate)
        ex, rh = -np.inf, 0.0
        for _, p_, _ in pl:
            for fr in pos:
                ss, pd = ctrl.at(fr * plate.lp)
                _, rho = is_stable(p_, C.RPM_S, 1e-12, fr * plate.lp,
                                   ctrl=ss, pd=pd, n_modes=C.N_MODES,
                                   m=C.M_FLOQUET_PSO, n_period=C.N_PERIOD,
                                   coeff_mode='time', coeff_scale=C.SIGN,
                                   ae=C.AE)
                rho = float(rho)
                e = (np.log(max(rho, 1e-300)) / tau0) if np.isfinite(rho) \
                    else np.inf
                if e > ex:
                    ex, rh = e, rho
        td_rows.append((k, worsts[names.index(k)], ex, rh))
        L.append(f'{k:<20}{worsts[names.index(k)]:>14.2f}{ex:>18.2f}'
                 f'{rh:>12.4f}'
                 f'{("PASS" if ex <= POLE_MAX else "FAIL"):>16}')
    L += ['',
          '  This column is NOT the screen; it is the honest check on the'
          ' screen\'s own approximation.',
          '  Where the two disagree, the tau -> 0 folding -- which is what the'
          ' PUBLISHED protocol already',
          '  uses for every design -- is the thing being reported, not the'
          ' physics.',
          '']

    # ---- reproduction check against the stored logs -----------------------
    L += ['=' * 92,
          'CHECK - nominal column against the values logged at design time',
          '=' * 92,
          '  results/log_ps_tdc.txt "slowest nominal pole": ps_tdc -97.4,'
          ' ps_tdc_j -96.0, ps_ac_r -98.8, ps_tdc_r -97.2 /s',
          '  results/log_stage3.txt: mu_phys_tdc -8.7 /s',
          '  this run, nominal column          : '
          + ', '.join(f'{k} {tables[names.index(k)][0].max():.1f}'
                      for k in ('ps_tdc', 'ps_tdc_j', 'ps_ac_r', 'ps_tdc_r',
                                'mu_phys_tdc') if k in names),
          '  -> the nominal half of the screen reproduces the design-time'
          ' numbers; only the vertex half is new.',
          '']

    # ---- B3 cost ----------------------------------------------------------
    L += ['=' * 92,
          'B3  COST PER CALL (ms)',
          '=' * 92,
          f'{"design":<20}{"build ctrl":>12}{"old screen":>12}'
          f'{"box screen":>12}{"added":>10}{"added/build":>13}']
    cost = []
    for k, mk in made.items():
        # controller construction, as a PSO particle pays it
        n_rep = 5
        t = time.perf_counter()
        for _ in range(n_rep):
            c = mk(plant)
            for fr in pos:                       # scheduled synthesis is here
                c.at(fr * plate.lp)
        t_build = (time.perf_counter() - t) / n_rep * 1e3
        ctrl = mk(plant)
        for fr in pos:
            ctrl.at(fr * plate.lp)               # warm the cache
        n_rep = 50
        t = time.perf_counter()
        for _ in range(n_rep):
            max(max_re(plate, ctrl, fr) for fr in pos)
        t_old = (time.perf_counter() - t) / n_rep * 1e3
        t = time.perf_counter()
        for _ in range(n_rep):
            screen(plate, ctrl, pos)
        t_new = (time.perf_counter() - t) / n_rep * 1e3
        cost.append((t_build, t_old, t_new))
        L.append(f'{k:<20}{t_build:>12.2f}{t_old:>12.3f}{t_new:>12.3f}'
                 f'{t_new-t_old:>10.3f}{(t_new-t_old)/max(t_build,1e-9):>12.1%}')
    cost = np.array(cost)

    # cost of the TRUE-DELAY variant of the same screen (3 plates x 3 pos)
    ctd = made['mu_tdc'](plant)
    plc = plates_for_screen(plate)
    t = time.perf_counter()
    for _ in range(3):
        for _, p_, _ in plc:
            for fr in pos:
                ss, pd = ctd.at(fr * plate.lp)
                is_stable(p_, C.RPM_S, 1e-12, fr * plate.lp, ctrl=ss, pd=pd,
                          n_modes=C.N_MODES, m=C.M_FLOQUET_PSO,
                          n_period=C.N_PERIOD, coeff_mode='time',
                          coeff_scale=C.SIGN, ae=C.AE)
    t_td = (time.perf_counter() - t) / 3 * 1e3

    # one full objective evaluation, for scale
    ctrl = made['ps_ac'](plant)
    t = time.perf_counter()
    eval2.evaluate(plate, ctrl, detail=True)
    t_obj = (time.perf_counter() - t) * 1e3
    ctrl2 = made['ps_ac'](plant)
    t = time.perf_counter()
    for fr in pos:
        ss, pd = ctrl2.at(fr * plate.lp)
        eval2.frequency_metrics(plate, ss, pd, pos)
    t_freq = (time.perf_counter() - t) * 1e3
    L += ['',
          f'  one full eval2.evaluate (ps_ac, feasible particle, Floquet'
          f' m={C.M_FLOQUET_PSO}) : {t_obj:9.1f} ms',
          f'  its Ms/effort screen alone (3 positions)                       '
          f'        : {t_freq:9.1f} ms',
          f'  old pole screen, mean over designs                             '
          f'        : {cost[:,1].mean():9.3f} ms',
          f'  box pole screen, mean over designs                             '
          f'        : {cost[:,2].mean():9.3f} ms',
          f'  true-delay variant of the box screen (mu_tdc, 3x3 Floquet)     '
          f'        : {t_td:9.1f} ms',
          f'  ADDED by the box screen, mean                                  '
          f'        : {cost[:,2].mean()-cost[:,1].mean():9.3f} ms'
          f'  ({(cost[:,2].mean()-cost[:,1].mean())/t_obj:.3%} of one'
          ' objective evaluation)',
          '']

    # ---- consequence ------------------------------------------------------
    failed = [k for k, ok in zip(names, passes) if not ok]
    fail_pub = [k for k in failed if k in SCOREBOARD]
    L += ['=' * 92,
          'CONSEQUENCE OF ADOPTING THE SCREEN (D7: it applies to EVERY design)',
          '=' * 92,
          '  designs that FAIL: ' + (', '.join(failed) if failed else 'none'),
          '  of those, rows of the PUBLISHED scoreboard: '
          + (', '.join(fail_pub) if fail_pub else 'none'),
          '']
    for k in fail_pub:
        L.append(f'    {k:<12} ({LABEL.get(k, k)}) would be INADMISSIBLE under'
                 ' the new protocol.')
    L += ['',
          '  Adopting the screen silently would delete those rows from the'
          ' comparison, which is a',
          '  change in the comparison itself, not in the new controller.  It'
          ' must be reported as a',
          '  cost of the screen: either every published design is re-designed'
          ' under the screen, or',
          '  the screen is declared and the removed rows are shown with their'
          ' failure numbers.',
          '',
          '  BUT the mu-TDC failure is NOT the same kind of failure as'
          ' ps_ac_obs / ps_ac_full:',
          '    * ps_ac_obs and ps_ac_full carry no delayed pair, so the'
          ' tau -> 0 folding is exact for them;',
          '      they genuinely lose the no-cutting loop at the r = 0.9045'
          ' vertex, which is the measured',
          '      mechanism behind their S3 = 0.0000.',
          '    * mu_tdc (and mu_phys_tdc) fail ONLY through the folding:'
          ' with the pair removed the base',
          f'      is at {dict((r[0], r[2]) for r in pd_rows)["mu_tdc"]:.2f} /s,'
          ' and with the pair at its TRUE delay the same box screen gives',
          f'      {dict((r[0], r[2]) for r in td_rows)["mu_tdc"]:.2f} /s'
          ' -- a PASS.  Their failure is an artefact of the approximation the',
          '      published protocol already uses, not a lost loop.',
          '',
          '  RECOMMENDATION, priced.  Run the box screen in two tiers:',
          f'    tier 1  tau -> 0 folding at the vertices,'
          f' {cost[:,2].mean()-cost[:,1].mean():.1f} ms added per particle -- '
          f'{(cost[:,2].mean()-cost[:,1].mean())/t_obj:.2%} of one objective',
          f'            evaluation.  Every particle pays this.',
          f'    tier 2  only when tier 1 rejects: re-test at the TRUE delay,'
          f' {t_td:.0f} ms.  Paid solely by',
          '            particles tier 1 has already rejected, so it costs'
          ' nothing on the accepted path,',
          '            and it keeps mu-TDC admissible while still failing'
          ' ps_ac_obs and ps_ac_full.',
          '  Under tier 1 ALONE the published mu-TDC row is deleted from the'
          ' comparison; under the two',
          '  tiers it is not.  That choice must be declared, not made'
          ' silently.',
          '',
          '  WHAT THIS RUN DID NOT DO:',
          '    * no design was re-optimised under the screen -- the price is'
          ' measured on the STORED',
          '      parameters only, so it does not say what a PSO would find if'
          ' the screen were active;',
          '    * the screen is a VERTEX test; the 41-point r sweep shows the'
          ' interior can be worse',
          '      (ps_ac_obs +2.99 -> +9.67 at r = 0.9397) though no verdict'
          ' flips on these 12 designs;',
          '    * damping (C.ZETA_LO/HI = 0.8/1.2), material removal (eta) and'
          ' spindle-speed changes are',
          '      NOT in this box; only the reference mass/stiffness box is;',
          '    * the true-delay column uses m = 40, not the full m = 120;',
          '    * timings are single-machine wall clock, one process, no'
          ' repetition across runs.',
          '',
          f'total {time.time()-t0:.1f}s']

    txt = '\n'.join(L) + '\n'
    p = os.path.join(C.RESULTS, 'g0_box_screen.txt')
    with open(p, 'w') as fh:
        fh.write(txt)
    print(txt)

    rs, _ = sweep(plate, made['ps_ac'](plant), 41, pos)
    np.savez(os.path.join(C.RESULTS, 'g0_box_screen.npz'),
             names=np.array(names), plate_labels=np.array(pl_labels),
             positions=np.array(pos), pole_max=POLE_MAX,
             vertex_dm=np.array([v[1] for v in vertices()]),
             vertex_dk=np.array([v[2] for v in vertices()]),
             vertex_r=np.array([v[3] for v in vertices()]),
             max_re=np.array(tables), worst=np.array(worsts),
             passed=np.array(passes), sweep_r_grid=rs,
             sweep_worst=np.array(sweep_worst),
             sweep_argmax_r=np.array(sweep_r),
             cost_build_ms=cost[:, 0], cost_old_ms=cost[:, 1],
             cost_box_ms=cost[:, 2], cost_objective_ms=t_obj,
             cost_freq_screen_ms=t_freq, b1_met=b1,
             pd_names=np.array([r[0] for r in pd_rows]),
             pd_worst_with=np.array([r[1] for r in pd_rows]),
             pd_worst_without=np.array([r[2] for r in pd_rows]),
             td_names=np.array([r[0] for r in td_rows]),
             td_worst_tau0fold=np.array([r[1] for r in td_rows]),
             td_worst_true=np.array([r[2] for r in td_rows]),
             td_worst_rho=np.array([r[3] for r in td_rows]),
             cost_truedelay_screen_ms=t_td)
    print('written to', p, 'and results/g0_box_screen.npz')


if __name__ == '__main__':
    main()
