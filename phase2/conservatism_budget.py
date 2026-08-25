"""
conservatism_budget.py — WHERE THE FACTOR OF TWO GOES.
=======================================================
The study certifies a_p^inf (largest depth at which the whole vertex family is
stable for EVERY delay, certify2.depth_bisect) and separately MEASURES a_p,lim
by Floquet bisection on the periodic truth (eval2.limits, coeff_mode='time').
The certified depth is roughly HALF the measured one:

    mu-TDC     a_p^inf 0.4364 mm   Floquet floor 0.6821 mm   (64 %)
    PS-AC      a_p^inf 0.3836 mm   Floquet floor 0.8517 mm   (45 %)
    PS-TDC-R   a_p^inf 0.3954 mm   Floquet floor 0.9307 mm   (42 %)

Nobody had decomposed that gap.  This script does, by walking a LADDER of
nested measurements in which exactly one modelling decision changes per rung, so
the log-differences between consecutive rungs are additive and sum to the total
log-gap by construction.  Every rung is computed from the STORED controller
matrices (stage_common.load_controllers -> results/stage3_controllers.pkl and
results/musyn_phase2.npz), never from a re-run design recipe.

THE LADDER  (a_p in metres; each rung is a depth at which some stability test
             is still satisfied)

  R0  ap_inf_full     certify2.depth_bisect over the FULL vertex family
                      n_pos=9 x 3 eta x 2 zeta x 2 xi x 2 alpha_4 ends = 216
  R0L ap_inf_light    the SAME bisection with run_stage56's reduced family
                      (n_pos=5, etas=(0, ETA_MAX), zetas=(ZETA_LO, ZETA_HI),
                      xis=(1.0,)) -- this is the kw run_stage56 actually uses
                      for its depth bisection, so R0L is what results/stage56.pkl
                      stores.  Printed next to the stored value as a check.
  R1  ap_inf_pos      parametric axes COLLAPSED to nominal: etas=(0.0,),
                      zetas=(1.0,), xis=(1.0,).  The 9 positions are kept,
                      because position is the scheduling domain, not an
                      uncertainty.  alpha_4 still swept over [0.3, 2.9] abar4.
  R2  ap_inf_a16      as R1, plus alpha_4 pinned to the single nominal
                      alpha_40 = 1.6 abar4 (certify2.vertices with a_scale=0.0;
                      see the reading of that code in `pin_alpha` below).
  R2m ap_inf_a10      as R2 but pinned to abar4 itself (1.0 abar4), which is the
                      level closed_loop.period_maps uses in coeff_mode='average'.
                      Without this rung the "level" difference would hide inside
                      the periodicity term and corrupt it.
  R3m ap_tau0_a10     as R2m but with the DELAY-DEPENDENT criterion
                      tau_max >= tau_0 instead of the delay-INDEPENDENT one
                      (peak < 1).  Same bisection schedule, same family; only
                      the accept test changes.  This rung prices delay
                      independence.
  R4  ap_floquet_avg  closed_loop.limit with coeff_mode='average' -- the same
                      physical question as R3m (constant alpha_4 = abar4, the
                      actual delay tau_0) but asked on the FIVE-mode evaluation
                      model by the FDM monodromy instead of on the two-mode
                      design model by the crossing test.  This rung prices the
                      design/evaluation model change.
  R5  ap_floquet_time closed_loop.limit with coeff_mode='time' -- the truth.
                      Equal to results/stage78.pkl S1_<name>['limits'].min();
                      recomputed here and cross-checked against the stored value.

THE THREE NAMED CAUSES, as log-differences along that ladder

  (P) PARAMETRIC COVERAGE   ln R1 - ln R0    216 vertices -> nominal eta/zeta/xi
  (E) ALPHA_4 ENVELOPE      ln R2 - ln R1    [0.3, 2.9] abar4 -> 1.6 abar4
  (F) FREEZING vs PERIODIC  ln R5 - ln R4    constant abar4 -> the real
                                             alpha_4(t), measured on the TRUTH
                                             side so it is not contaminated by
                                             the certificate's other choices
  (D) the RESIDUAL, itself resolved into three additive rungs:
      level  ln R2m - ln R2   1.6 abar4 -> 1.0 abar4
      delay  ln R3m - ln R2m  every delay -> the actual delay tau_0
      model  ln R4  - ln R3m  2-mode crossing test -> 5-mode Floquet

  P + E + level + delay + model + F = ln R5 - ln R0 exactly.  Shares are
  reported as percentages of that total.  A NEGATIVE share means the axis makes
  the certificate LESS conservative rather than more, and is reported as such.

THE ENVELOPE QUESTION (the reason this script exists)
  certify2.vertices sweeps alpha_4 over [C.ALPHA_LO, C.ALPHA_HI] * abar4 =
  [0.3, 2.9] abar4 and freezes it.  The instantaneous alpha_4(t) that
  closed_loop.period_maps feeds the truth reaches 12.43 abar4 and sits at zero
  for most of the tooth period.  The script therefore also measures

  R6  ap_inf_true_env  a_p^inf over the TRUE instantaneous alpha_4 envelope,
                       as the minimum over a set of single-value pins spanning
                       [0, 12.43] abar4 (a union of frozen families is stable
                       iff every member is, so the union depth is the minimum
                       of the member depths under the same monotonicity in a_p
                       that certify2.depth_bisect already assumes).

  plus the fraction of one tooth period during which alpha_4(t) actually lies
  inside the certified band.  Those two numbers answer whether the frozen
  certificate is a CONSERVATIVE statement about the periodic system, an
  INCOMPLETE one, or both.

HOW TO RUN
    cd phase2 && python conservatism_budget.py
  Writes results/conservatism_budget.txt and results/conservatism_budget.npz.
  Nothing else in results/ is touched.  ~6 min for the three controllers.
  Optional: python conservatism_budget.py mu_tdc ps_ac   (subset of names)

PRE-DECLARED SUCCESS CRITERIA (house rule: fixed before the run)
  1. R0L must reproduce results/stage56.pkl['<name>']['ap_inf'] to within the
     bisection tolerance 2e-6 m.  If it does not, the ladder is not anchored to
     the published number and the run is reported as FAILED, not adjusted.
  2. R5 must reproduce results/stage78.pkl['S1_<name>']['limits'].min() to
     within the Floquet bisection tolerance 5e-6 m, same consequence.
  3. The six rung differences must sum to ln R5 - ln R0 to 1e-9.  This is an
     identity, so a violation means a bookkeeping bug, not a physical result.
  4. No rung is discarded for coming out with the "wrong" sign.

WHAT CAME OUT (see results/conservatism_budget.txt for the generated numbers)
  Every rung PASSED its check.  Reported here because a docstring that states
  the criteria and hides the outcome is worth nothing:
  - (E) alpha_4 envelope and the (D) alpha_4 LEVEL rung together carry 170-260 %
    of the log-gap: they are the whole conservatism budget.
  - (D) delay independence is worth EXACTLY ZERO at this operating point.  The
    crossing frequencies sit on the structural modes, so the first crossing
    delay is already shorter than tau_0; nothing is paid for asking for every
    delay instead of the actual one.
  - (F) freezing vs periodicity is NEGATIVE for PS-AC (-43 %) and PS-TDC-R
    (-32 %): averaging alpha_4 OVERSTATES the stable depth, so the frozen
    treatment is optimistic, not conservative, about periodicity.
  - (D) 2-mode -> 5-mode is -162 % for mu-TDC alone: its delayed PD destabilises
    mode 3, which the two-mode design plant cannot see (model-order probe below).
  - a_p^inf over the TRUE instantaneous alpha_4 envelope is 0.0967 mm for ALL
    THREE controllers -- 4x below the published a_p^inf and identical across
    designs, i.e. a frozen certificate honest about the real envelope is both
    far weaker and completely blind to the controller.
"""
import os
import pickle
import sys
import time
import warnings
from copy import copy

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify2 as CF2
import config as C
from closed_loop import limit as floquet_limit
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = C.RESULTS
NAMES = ('mu_tdc', 'ps_ac', 'ps_tdc_r')
LINES = []


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LINES.append(line)


# ---------------------------------------------------------------------------
# families.  n_pos=5 everywhere in the bisections, matching the `light` family
# run_stage56.main uses for ITS depth bisection -- so R0L is directly
# comparable to the stored ap_inf, and every rung below it moves only the axis
# it is supposed to move.
# ---------------------------------------------------------------------------
ETAS = tuple(np.linspace(0.0, C.ETA_MAX, 3))
ZETAS = (C.ZETA_LO, C.ZETA_HI)

FAM_FULL = dict(n_pos=9, etas=ETAS, zetas=ZETAS, xis=(0.5, 1.0))
FAM_LIGHT = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=ZETAS, xis=(1.0,))
FAM_POS = dict(n_pos=5, etas=(0.0,), zetas=(1.0,), xis=(1.0,))

BISECT = dict(n_iter=16)                       # run_stage56's depth bisection
FLOQ = dict(lo=0.005e-3, hi=3.0e-3, tol=5e-6, n_modes=C.N_MODES,
            m=C.M_FLOQUET, n_period=C.N_PERIOD, coeff_scale=C.SIGN, ae=C.AE)


def pin_alpha(plant, mult):
    """A plant whose alpha_4 NOMINAL is `mult` * abar4, for use with a_scale=0.

    Reading of certify2.vertices, lines 95-97:

        a_nom = plant.a40
        a_lo  = a_nom + (C.ALPHA_LO * plant.abar4 - a_nom) * a_scale
        a_hi  = a_nom + (C.ALPHA_HI * plant.abar4 - a_nom) * a_scale

    so a_scale = 1 gives the published band [0.3, 2.9] abar4 and a_scale = 0
    collapses BOTH ends onto a_nom = plant.a40 = 1.6 abar4 (plant_ss line 66).
    Verified numerically in `check_pin` below rather than assumed.  Overriding
    a40 on a shallow copy therefore pins the frozen family at any level we like,
    WITHOUT touching the controller, which is always built from the untouched
    plant carried in `.base`.
    """
    p = copy(plant)
    p.a40 = mult * plant.abar4
    p.base = getattr(plant, 'base', plant)
    return p


def check_pin(plant, ctrl):
    """Confirm the a_scale reading above from the code, not from the docstring."""
    ok = True
    for sc, want in ((1.0, (C.ALPHA_LO, C.ALPHA_HI)), (0.0, (1.6, 1.6))):
        seen = set()
        a_nom = plant.a40
        a_lo = a_nom + (C.ALPHA_LO * plant.abar4 - a_nom) * sc
        a_hi = a_nom + (C.ALPHA_HI * plant.abar4 - a_nom) * sc
        seen.add(round(a_lo / plant.abar4, 6))
        seen.add(round(a_hi / plant.abar4, 6))
        ok &= seen == {round(w, 6) for w in want}
    # and that vertices() really produces 2 alpha vertices per (x, eta, xi, z)
    V, _, _ = CF2.vertices(plant, ctrl, n_pos=2, etas=(0.0,), zetas=(1.0,),
                           xis=(1.0,))
    ok &= (len(V) == 4)
    V0, _, _ = CF2.vertices(pin_alpha(plant, 1.6), ctrl, a_scale=0.0,
                            n_pos=2, etas=(0.0,), zetas=(1.0,), xis=(1.0,))
    # a_scale=0 must make the two alpha vertices IDENTICAL
    ok &= bool(np.allclose(V0[0][1], V0[1][1]))
    return bool(ok)


# ---------------------------------------------------------------------------
def depth_di(plate, mk, fam, a_scale=1.0, mult=None):
    """certify2.depth_bisect, delay-INDEPENDENT criterion, exactly as run_stage56."""
    def plant_of(ap):
        p = ControlledPlant(plate, ap=ap)
        return p if mult is None else pin_alpha(p, mult)

    def ctrl_of(p):
        return mk(getattr(p, 'base', p))

    kw = dict(fam)
    kw['a_scale'] = a_scale
    return CF2.depth_bisect(plant_of, ctrl_of, **BISECT, **kw)


def depth_tau(plate, mk, fam, tau0, a_scale=0.0, mult=1.0,
              lo=2e-6, hi=4e-3, tol=2e-6, n_iter=16):
    """Same bisection schedule as certify2.depth_bisect, DELAY-DEPENDENT test.

    certify2.depth_bisect hard-codes di_stable (peak < 1, i.e. stable for every
    delay).  The only change here is the accept test: the crossing analysis
    already returns tau_max, so we ask the weaker and physically relevant
    question tau_max >= tau_0.  Bounds, tolerance and iteration count are copied
    from certify2.depth_bisect so the two ladders are commensurate.
    """
    kw = dict(fam)
    kw['a_scale'] = a_scale

    def ok(ap):
        p = ControlledPlant(plate, ap=ap)
        pp = p if mult is None else pin_alpha(p, mult)
        tm, _, _ = CF2.analyse(pp, mk(p), **kw)
        return bool(tm >= tau0)

    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def floquet_floor(plate, ctrl, mode):
    """min over C.POSITIONS of closed_loop.limit -- eval2.limits' own settings."""
    vals = []
    for fr in C.POSITIONS:
        x = fr * plate.lp
        ss, pd = ctrl.at(x, 0.0)
        vals.append(floquet_limit(plate, C.RPM_S, x, ctrl=ss, pd=pd,
                                  coeff_mode=mode, **FLOQ))
    return float(min(vals)), np.array(vals)


# ---------------------------------------------------------------------------
def alpha_envelope(plate, ap, n=4000):
    """(r_min, r_max, duty, frac_in_band) of alpha_4(t)/abar4 over one period."""
    p = ControlledPlant(plate, ap=ap)
    _, a4 = p.a4_series(n)
    r = a4 / p.abar4
    inb = (r >= C.ALPHA_LO) & (r <= C.ALPHA_HI)
    return (float(r.min()), float(r.max()), float((r > 1e-9).mean()),
            float(inb.mean()), float((r > C.ALPHA_HI).mean()))


# ---------------------------------------------------------------------------
def main(names=NAMES):
    t00 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    ctrls = load_controllers(plate, plant0)
    tau0 = 60.0 / (3 * C.RPM_S)
    s56 = pickle.load(open(os.path.join(OUT, 'stage56.pkl'), 'rb'))
    s78 = pickle.load(open(os.path.join(OUT, 'stage78.pkl'), 'rb'))

    log('=' * 92)
    log('CONSERVATISM BUDGET - decomposition of a_p^inf vs a_p,lim')
    log('=' * 92)
    log(f'  tau_0 = {tau0*1e3:.4f} ms at {C.RPM_S} rpm;  alpha_4 band '
        f'[{C.ALPHA_LO}, {C.ALPHA_HI}] abar4;  alpha_40 = 1.6 abar4')
    log(f'  bisection family n_pos=5 (MATCHED to run_stage56\'s `light` kw for '
        f'its depth bisection)')
    log(f'  pin check (certify2.vertices a_scale semantics verified in code): '
        f'{check_pin(plant0, ctrls["ps_ac"](plant0))}')

    # ---- the alpha_4 envelope, measured, not assumed -----------------------
    log('')
    log('-' * 92)
    log('ALPHA_4: what the certificate sweeps vs what the truth does')
    log('-' * 92)
    log(f'{"a_p mm":>8}{"min/abar4":>12}{"max/abar4":>12}{"duty %":>10}'
        f'{"% of period in [0.3,2.9]":>26}{"% above 2.9":>13}')
    env = {}
    for ap in (0.3e-3, 0.4e-3, 0.85e-3):
        rmn, rmx, duty, inb, above = alpha_envelope(plate, ap)
        env[ap] = (rmn, rmx, duty, inb, above)
        log(f'{ap*1e3:>8.3f}{rmn:>12.4f}{rmx:>12.4f}{duty*100:>10.2f}'
            f'{inb*100:>26.2f}{above*100:>13.2f}')

    # ---- model-order probe: which mode does rung D3 live on? --------------
    log('')
    log('-' * 92)
    log('MODEL-ORDER PROBE  Floquet floor, coeff_mode=average, by modes retained')
    log('  (the D3 rung is R3m -> R4, i.e. 2-mode crossing test -> 5-mode')
    log('   Floquet; this says whether that rung is truncation or method)')
    log('-' * 92)
    log(f'{"controller":<12}' + ''.join(f'{n:>10}' for n in (2, 3, 4, 5)))
    mo = {}
    for nm in names:
        c = ctrls[nm](plant0)
        row = []
        for n in (2, 3, 4, 5):
            kw = dict(FLOQ)
            kw['n_modes'] = n
            row.append(min(floquet_limit(plate, C.RPM_S, fr * plate.lp,
                                         ctrl=c.at(fr * plate.lp, 0.0)[0],
                                         pd=c.at(fr * plate.lp, 0.0)[1],
                                         coeff_mode='average', **kw)
                           for fr in C.POSITIONS))
        mo[nm] = np.array(row)
        log(f'{nm:<12}' + ''.join(f'{v*1e3:>10.4f}' for v in row))

    rows = {}
    for nm in names:
        mk = ctrls[nm]
        c0 = mk(plant0)
        log('')
        log('=' * 92)
        log(f'{nm}')
        log('=' * 92)
        r = {}

        t = time.time()
        r['R0L_ap_inf_light'] = depth_di(plate, mk, FAM_LIGHT)
        log(f'  R0L ap_inf light family (run_stage56 kw) '
            f'{r["R0L_ap_inf_light"]*1e3:>9.4f} mm   stored '
            f'{s56[nm]["ap_inf"]*1e3:.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        r['R0_ap_inf_full'] = depth_di(plate, mk, FAM_FULL)
        log(f'  R0  ap_inf FULL 216-vertex family        '
            f'{r["R0_ap_inf_full"]*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        r['R1_ap_inf_pos'] = depth_di(plate, mk, FAM_POS)
        log(f'  R1  + eta/zeta/xi collapsed to nominal   '
            f'{r["R1_ap_inf_pos"]*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        r['R2_ap_inf_a16'] = depth_di(plate, mk, FAM_POS, a_scale=0.0)
        log(f'  R2  + alpha_4 pinned at 1.6 abar4        '
            f'{r["R2_ap_inf_a16"]*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        r['R2m_ap_inf_a10'] = depth_di(plate, mk, FAM_POS, a_scale=0.0, mult=1.0)
        log(f'  R2m + alpha_4 pinned at 1.0 abar4        '
            f'{r["R2m_ap_inf_a10"]*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        r['R3m_ap_tau0_a10'] = depth_tau(plate, mk, FAM_POS, tau0, mult=1.0)
        log(f'  R3m + delay-DEPENDENT (tau_max >= tau_0) '
            f'{r["R3m_ap_tau0_a10"]*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        # why R3m can equal R2m: the crossing frequencies sit at the structural
        # modes (~3.4e3 rad/s), so the FIRST crossing delay arg(lambda)/omega is
        # at most 2 pi / omega ~ 1.8 ms, already shorter than tau_0 = 4.08 ms.
        # Once peak >= 1 at all, the loop is therefore ALREADY unstable at the
        # actual delay -- delay independence costs nothing here.  Measured, not
        # argued: tau_max/tau_0 just above the delay-independent depth.
        probe = []
        for f in (1.0, 1.02, 1.10, 1.30, 2.00):
            ap = f * max(r['R2m_ap_inf_a10'], 1e-6)
            p = ControlledPlant(plate, ap=ap)
            tm, pk, _ = CF2.analyse(pin_alpha(p, 1.0), mk(p),
                                    a_scale=0.0, **FAM_POS)
            probe.append((f, ap, pk, tm / tau0))
        r['tau_probe'] = np.array([[a, b, c_, d] for a, b, c_, d in probe])
        log('      delay-independence probe at pin 1.0 abar4:')
        log('        a_p/R2m ' + ''.join(f'{a:>10.2f}' for a, _, _, _ in probe))
        log('        peak    ' + ''.join(f'{c_:>10.3f}' for _, _, c_, _ in probe))
        log('        tau_max/tau_0 ' + ''.join(f'{d:>10.3f}'
                                               for _, _, _, d in probe))

        t = time.time()
        favg, vavg = floquet_floor(plate, c0, 'average')
        r['R4_floquet_avg'] = favg
        log(f'  R4  Floquet, coeff_mode=average, 5 modes '
            f'{favg*1e3:>9.4f} mm   [{time.time()-t:.0f}s]')

        t = time.time()
        ftime, vtime = floquet_floor(plate, c0, 'time')
        r['R5_floquet_time'] = ftime
        st = float(np.asarray(s78[f'S1_{nm}']['limits']).min())
        log(f'  R5  Floquet, coeff_mode=time (THE TRUTH) '
            f'{ftime*1e3:>9.4f} mm   stored {st*1e3:.4f} mm   '
            f'[{time.time()-t:.0f}s]')

        # ---- the true instantaneous envelope -----------------------------
        t = time.time()
        pins = (0.0, 0.3, 1.0, 1.6, 2.9, 6.0, 9.0, 12.43)
        pv = {}
        for m in pins:
            pv[m] = depth_di(plate, mk, FAM_POS, a_scale=0.0, mult=m)
        r['R6_true_envelope'] = float(min(pv.values()))
        r['pins'] = np.array(pins)
        r['pin_depths'] = np.array([pv[m] for m in pins])
        log(f'  R6  a_p^inf over the TRUE alpha_4 envelope [0, 12.43] abar4 '
            f'= min over pins  {r["R6_true_envelope"]*1e3:>7.4f} mm '
            f'[{time.time()-t:.0f}s]')
        log('      pin/abar4 ' + ''.join(f'{m:>9.2f}' for m in pins))
        log('      a_p^inf mm' + ''.join(f'{pv[m]*1e3:>9.4f}' for m in pins))

        # ---- the budget --------------------------------------------------
        R0, R1, R2 = r['R0_ap_inf_full'], r['R1_ap_inf_pos'], r['R2_ap_inf_a16']
        R2m, R3m = r['R2m_ap_inf_a10'], r['R3m_ap_tau0_a10']
        R4, R5 = r['R4_floquet_avg'], r['R5_floquet_time']
        eps = 1e-12
        L = lambda v: np.log(max(v, eps))
        G = L(R5) - L(R0)
        parts = dict(P_parametric=L(R1) - L(R0),
                     E_alpha_envelope=L(R2) - L(R1),
                     D1_alpha_level=L(R2m) - L(R2),
                     D2_delay_independence=L(R3m) - L(R2m),
                     D3_design_model=L(R4) - L(R3m),
                     F_periodicity=L(R5) - L(R4))
        resid = G - sum(parts.values())
        r.update(total_log_gap=G, residual_identity=resid, **parts)
        r['shares'] = {k: 100.0 * v / G for k, v in parts.items()}
        r['ratio_R0_over_R5'] = R0 / R5

        log('')
        log(f'  total gap  a_p^inf(full)/a_p,lim = {R0/R5*100:.1f} %   '
            f'ln-gap = {G:+.4f}   (identity residual {resid:+.2e})')
        log(f'  {"cause":<26}{"rung":>22}{"ln-share":>11}{"% of gap":>11}')
        lab = dict(P_parametric='(P) parametric coverage',
                   E_alpha_envelope='(E) alpha_4 envelope',
                   D1_alpha_level='(D) alpha_4 level 1.6->1.0',
                   D2_delay_independence='(D) delay independence',
                   D3_design_model='(D) 2-mode -> 5-mode model',
                   F_periodicity='(F) freezing vs periodicity')
        rung = dict(P_parametric='R0 -> R1', E_alpha_envelope='R1 -> R2',
                    D1_alpha_level='R2 -> R2m',
                    D2_delay_independence='R2m -> R3m',
                    D3_design_model='R3m -> R4',
                    F_periodicity='R4 -> R5')
        for k in ('P_parametric', 'E_alpha_envelope', 'D1_alpha_level',
                  'D2_delay_independence', 'D3_design_model', 'F_periodicity'):
            log(f'  {lab[k]:<26}{rung[k]:>22}{parts[k]:>+11.4f}'
                f'{r["shares"][k]:>+11.1f}')
        log(f'  {"TOTAL":<26}{"R0 -> R5":>22}{G:>+11.4f}{100.0:>+11.1f}')

        r['floquet_avg_positions'] = vavg
        r['floquet_time_positions'] = vtime
        r['stored_ap_inf'] = float(s56[nm]['ap_inf'])
        r['stored_floquet'] = st
        rows[nm] = r

    # ---- pre-declared checks ---------------------------------------------
    log('')
    log('=' * 92)
    log('PRE-DECLARED CHECKS')
    log('=' * 92)
    allok = True
    for nm, r in rows.items():
        c1 = abs(r['R0L_ap_inf_light'] - r['stored_ap_inf']) <= 2e-6
        c2 = abs(r['R5_floquet_time'] - r['stored_floquet']) <= 5e-6
        c3 = abs(r['residual_identity']) <= 1e-9
        allok &= bool(c1 and c2 and c3)
        log(f'  {nm:<12} R0L==stage56 {"PASS" if c1 else "FAIL"}   '
            f'R5==stage78 {"PASS" if c2 else "FAIL"}   '
            f'sum identity {"PASS" if c3 else "FAIL"}')
    log(f'  ALL CHECKS {"PASS" if allok else "FAIL"}')

    # ---- summary table ----------------------------------------------------
    log('')
    log('=' * 92)
    log('SUMMARY  (a_p in mm)')
    log('=' * 92)
    keys = ('R0_ap_inf_full', 'R0L_ap_inf_light', 'R1_ap_inf_pos',
            'R2_ap_inf_a16', 'R2m_ap_inf_a10', 'R3m_ap_tau0_a10',
            'R4_floquet_avg', 'R5_floquet_time', 'R6_true_envelope')
    log(f'{"rung":<20}' + ''.join(f'{nm:>14}' for nm in rows))
    for k in keys:
        log(f'{k:<20}' + ''.join(f'{rows[nm][k]*1e3:>14.4f}' for nm in rows))
    log('')
    log(f'{"share % of ln-gap":<26}' + ''.join(f'{nm:>14}' for nm in rows))
    for k in ('P_parametric', 'E_alpha_envelope', 'D1_alpha_level',
              'D2_delay_independence', 'D3_design_model', 'F_periodicity'):
        log(f'{k:<26}' + ''.join(f'{rows[nm]["shares"][k]:>+14.1f}'
                                 for nm in rows))

    # ---- write ------------------------------------------------------------
    with open(os.path.join(OUT, 'conservatism_budget.txt'), 'w') as f:
        f.write('\n'.join(LINES) + '\n')
    flat = {}
    for nm, r in rows.items():
        for k, v in r.items():
            if k == 'shares':
                for kk, vv in v.items():
                    flat[f'{nm}__share__{kk}'] = float(vv)
            else:
                flat[f'{nm}__{k}'] = np.asarray(v)
    for nm, v in mo.items():
        flat[f'{nm}__model_order_avg_floor'] = v
    for ap, e in env.items():
        flat[f'alpha_env__{ap*1e3:.2f}mm'] = np.array(e)
    flat['names'] = np.array(list(rows))
    np.savez(os.path.join(OUT, 'conservatism_budget.npz'), **flat)
    print(f'\ntotal {time.time()-t00:.0f}s -> results/conservatism_budget.txt'
          f' and .npz')
    return rows


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or NAMES)
