"""
g0_lambda_probe.py -- G0a: is there a NON-COLLAPSING observer-scheduled region?
==============================================================================
The campaign's kill switch.  ps_ac_obs / ps_ac_full schedule the Kalman filter
as well as the gain; they already beat mu-TDC on delta_max (1.66 / 1.68 vs 1.54)
but they LOSE the nominal loop under the reference's +10 % mass / -10 %
stiffness box, so their S3 limit is 0.0000 mm.  This script asks whether a
regularised version of the observer scheduling exists that keeps the margin and
survives the box.

WHAT IS BUILT HERE (by hand, ctrl2.py is NOT touched)
-----------------------------------------------------
The construction of ctrl2.ps_ac is reproduced verbatim (same Q, same _lqr_gain,
same _kalman_gain on the STRUCTURE-ONLY A0, same _observer_ctrl, same
_rolloff), with the observer gain replaced by one of three regularisations:

  R1  GAIN BLEND        L_lam(x) = (1-lam) L_0 + lam L(x)
                        L_0 from the position-averaged Wbar (the ps_ac
                        observer), L(x) from W(x) = E(x) E(x)^T (the
                        ps_ac_obs observer).  lam = 0 / 1 are the endpoints.
  R2  COVARIANCE BLEND  L from W_lam = (1-lam) Wbar + lam E(x) E(x)^T, i.e. the
                        DESIGN MODEL the filter is built on is blended, not the
                        gain.  Same two endpoints, different interior.
  R3  DIRECTION FLOOR   full scheduling (lam = 1) but the second component of
                        the disturbance direction E(x) -- the one the collapse
                        mechanism names, |D_2|/|D_1| = 1.10e-02 at mid-edge --
                        is floored at c |D_1|.  c = 0 is R1 at lam = 1.
  (At eta = 0 the modal mass is the identity, so the lower block of E is exactly
   D(x); the floor is applied to that block.)

Because the STORED ps_ac and ps_ac_obs differ in TWO things -- the four weights
(ps_ac: log_q_pos 18.8027 ...; ps_ac_obs: 19.8153 ...) and whether the state
feedback is scheduled (ps_ac: sched_K = True, ps_ac_obs: sched_K = False) -- no
single lam-sweep can have both stored designs as its endpoints.  Two sweeps are
therefore run, and every stored endpoint is anchored in the sweep that contains
it:
  SWEEP A  stored ps_ac weights,     sched_K = True   -> lam = 0 IS ps_ac
  SWEEP B  stored ps_ac_obs weights, sched_K = False  -> lam = 1 IS ps_ac_obs
  ANCHOR   stored ps_ac_obs weights, sched_K = True, lam = 1 IS ps_ac_full
R2 and R3 are swept on the SWEEP A weights (sched_K = True).

PRE-DECLARED SUCCESS CRITERIA  (written before the run; not edited after)
-------------------------------------------------------------------------
K1  ENDPOINT / ANCHOR CHECK.  The hand-built family must reproduce the stored
    designs: at (A, lam=0) the stored ps_ac delta_max = 1.4023 and S3(box +10%)
    = 0.7903 mm; at (B, lam=1) the stored ps_ac_obs delta_max = 1.6602 and S3 =
    0.0000 mm; at the anchor the stored ps_ac_full delta_max = 1.6758 and S3 =
    0.0000 mm.  Tolerance: delta_max to 0.02, S3 to 0.01 mm.  Additionally the
    state-space matrices are compared elementwise against ctrl2.build at five
    positions (reported, not part of the pass/fail).  If K1 fails the run is
    reported as UNANCHORED and no conclusion is drawn from the sweeps.

K2  THE GATE.  Some regularised point gives  S3 > 0.05 mm  AND  delta_max >=
    1.54 (mu-TDC's).  If no point does, report the best achievable
    (delta_max, S3) pair and declare the BLENDING ROUTE CLOSED.  That is the
    answer, not a failure.  The gate is evaluated on all three regularisations.

K3  THE WHOLE SWEEP IS REPORTED, including points worse than both endpoints,
    and including points that violate Ms <= 2 or V <= 450 V/N (reported, and
    flagged as protocol-inadmissible, never silently dropped).

WHAT IS MEASURED PER POINT
--------------------------
  max Re of the nominal closed-loop poles (no cutting) at the NOMINAL plate,
    max over the five evaluation positions;
  the same under the reference +10 % mass / -10 % stiffness box (the collapse
    indicator), max over positions and at mid-edge;
  delta_max        certify2.margin_bisect, n_iter = 14, base_kw n_pos = 5,
                   xis = (1.0,)   -- run_stage56's call, copied;
  a_p^inf          certify2.depth_bisect, n_iter = 16, on the LIGHT family
                   n_pos = 5, etas = (0, ETA_MAX), zetas = (0.8, 1.2),
                   xis = (1.0,)   -- run_stage56's call, copied;
  S3               eval2.limits under paper_box_scale(+0.10, -0.10), min over
                   C.POSITIONS, m = 120, five-mode model -- run_stage78's s3
                   'box +10%' case, copied;
  nominal floor    eval2.limits on the nominal plate, min over C.POSITIONS;
  J, Ms, V         eval2.evaluate(detail=True), the protocol screen.

NOT DONE (declared): no PSO -- every point uses STORED weights, so this probe
maps an existing family, it does not tune one.  a_p^inf and delta_max use the
reduced vertex families run_stage56 uses (upper bounds on the full-family
answer, exactly as that stage states).  The removal axis eta is not scheduled.
Only the +10 % box of S3 is evaluated, not the other five S3 cases.

ADDENDUM (run AFTER the main sweep, appended to the same two artifacts)
-----------------------------------------------------------------------
`python phase2/g0_lambda_probe.py refine` re-measures a short list of extra
lam values and APPENDS them to results/g0_lambda.{txt,npz}.  It exists because
the main sweep put the collapse cliff somewhere in lam in (0.90, 1.00) and the
campaign needs to know how far into the scheduled corner one may go.  It adds
measurements only; no criterion above is changed by it, and the appended points
are marked ADDENDUM in both artifacts.

    python phase2/g0_lambda_probe.py            # the pre-declared sweep
    python phase2/g0_lambda_probe.py refine     # the cliff addendum
Writes results/g0_lambda.txt and results/g0_lambda.npz
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
import certify2 as CF2
import ctrl2 as K2
import eval2 as E2
from plate_model import build_plate
from plant_ss import ControlledPlant
from scenarios import _perturbed, paper_box_scale

OUT = C.RESULTS
TXT = os.path.join(OUT, 'g0_lambda.txt')
NPZ = os.path.join(OUT, 'g0_lambda.npz')
LINES = []


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LINES.append(line)


# ---------------------------------------------------------------------------
# the hand-built regularised family: ctrl2.ps_ac with a regularised observer
# ---------------------------------------------------------------------------
def blended(plant, q_pos, q_vel, r, ratio, lam=1.0, sched_K=True, mode='L',
            dfloor=0.0, n_grid=21, a4_mult=1.0, name='BLEND'):
    n = plant.n
    Q = np.diag(np.concatenate([q_pos * np.ones(n), q_vel * np.ones(n)]))
    xs = np.linspace(0.0, plant.plate.lp, n_grid)
    Wbar = K2._mean_process_noise(plant)
    eps = 1e-12 * np.eye(2 * n)

    def build(x_pos, eta=0.0):
        x = 0.5 * plant.plate.lp if x_pos is None else float(x_pos)
        x = float(np.clip(x, xs[0], xs[-1]))
        A, _, B, E, Cy = plant.matrices(x_pos=x, eta=0.0,
                                        a4=a4_mult * plant.a40)
        A0, _, _ = plant.structure_only(0.0)
        Ak = A if sched_K else A0
        K = K2._lqr_gain(Ak, B, Q, r)
        if mode == 'L':                      # R1: blend the observer GAIN
            L0 = K2._kalman_gain(A0, Cy, ratio * Wbar + eps, 1.0)
            Lx = K2._kalman_gain(A0, Cy, ratio * (E @ E.T) + eps, 1.0)
            L = (1.0 - lam) * L0 + lam * Lx
        elif mode == 'W':                    # R2: blend the DESIGN COVARIANCE
            W = (1.0 - lam) * Wbar + lam * (E @ E.T)
            L = K2._kalman_gain(A0, Cy, ratio * W + eps, 1.0)
        elif mode == 'D':                    # R3: floor the direction
            Ef = E.copy()
            d = Ef[n:, 0]
            f = dfloor * abs(float(d[0]))
            for i in range(1, n):
                if abs(float(d[i])) < f:
                    d[i] = f * (1.0 if float(d[i]) >= 0.0 else -1.0)
            W = (1.0 - lam) * Wbar + lam * (Ef @ Ef.T)
            L = K2._kalman_gain(A0, Cy, ratio * W + eps, 1.0)
        else:
            raise ValueError(mode)
        Aobs = Ak if sched_K else A0
        return K2._rolloff(K2._observer_ctrl(Aobs, B, Cy, K, L)), None

    return K2.Ctrl(name, 4, builder=build, scheduled=True,
                   meta=dict(grid=xs, lam=lam, mode=mode, dfloor=dfloor,
                             sched_K=sched_K))


def factory(u, **kw):
    """plant -> Ctrl, re-synthesised at each call (depth_bisect needs this)."""
    def mk(p):
        return blended(p, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                       10 ** u['log_r'], 10 ** u['log_ratio'], **kw)
    return mk


# ---------------------------------------------------------------------------
LIGHT = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=(C.ZETA_LO, C.ZETA_HI),
             xis=(1.0,))


def measure(plate, plant0, plate_box, mk, tag):
    t0 = time.time()
    c = mk(plant0)
    # nominal-loop poles, no cutting
    mre_nom, mre_box = -np.inf, -np.inf
    for fr in C.POSITIONS:
        ss, pd = c.at(fr * plate.lp, 0.0)
        mre_nom = max(mre_nom, float(np.max(E2.nominal_poles(plate, ss, pd).real)))
        mre_box = max(mre_box, float(np.max(E2.nominal_poles(plate_box, ss, pd).real)))
    ss, pd = c.at(0.5 * plate.lp, 0.0)
    mre_box_mid = float(np.max(E2.nominal_poles(plate_box, ss, pd).real))
    # protocol screen
    J, info = E2.evaluate(plate, c, detail=True)
    # certificates
    dmax = CF2.margin_bisect(plant0, c, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    ap_inf = CF2.depth_bisect(lambda ap: ControlledPlant(plate, ap=ap), mk,
                              n_iter=16, **LIGHT)
    # Floquet limits
    Lnom = E2.limits(plate, c)
    Lbox = E2.limits(plate_box, c)
    d = dict(tag=tag, mre_nom=mre_nom, mre_box=mre_box, mre_box_mid=mre_box_mid,
             J=float(J), Ms=float(info['Ms']), V=float(info['V']),
             delta_max=float(dmax), ap_inf=float(ap_inf * 1e3),
             floor=float(Lnom.min() * 1e3), S3=float(Lbox.min() * 1e3),
             Lnom=Lnom * 1e3, Lbox=Lbox * 1e3, secs=time.time() - t0)
    return d


HDR = (f'{"point":<26}{"maxRe nom":>11}{"maxRe box":>11}{"Ms":>7}{"V":>8}'
       f'{"J":>9}{"a_p^inf":>9}{"delta":>8}{"floor":>8}{"S3":>8}')


def row(d):
    flag = ''
    if d['Ms'] > C.MS_MAX + 1e-9 or d['V'] > C.V_PER_N + 1e-9:
        flag += '  [INADMISSIBLE: protocol screen]'
    if d['S3'] <= 0.05:
        flag += '  [COLLAPSE]'
    return (f'{d["tag"]:<26}{d["mre_nom"]:>11.2f}{d["mre_box"]:>11.2f}'
            f'{d["Ms"]:>7.3f}{d["V"]:>8.1f}{d["J"]:>9.4f}{d["ap_inf"]:>9.4f}'
            f'{d["delta_max"]:>8.3f}{d["floor"]:>8.4f}{d["S3"]:>8.4f}{flag}')


# ---------------------------------------------------------------------------
def main(n_lam=11):
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    box = paper_box_scale(0.10, -0.10, C.N_MODES)
    plate_box = _perturbed(plate, box, 1.0)
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f:
        store = pickle.load(f)
    u_ac = dict(store['ps_ac']['params'])
    u_obs = dict(store['ps_ac_obs']['params'])

    log('=' * 118)
    log('G0a - IS THERE A NON-COLLAPSING OBSERVER-SCHEDULED REGION?')
    log('=' * 118)
    log(f'  plate          : build_plate(patch={C.PATCH_SIDE}, freqs=F_MEASURED), '
        f'{C.N_MODES}-mode evaluation, m = {C.M_FLOQUET}')
    log(f'  box            : +10 % mass / -10 % stiffness -> omega x {box[0]:.4f} '
        f'({100*(1-box[0]):.1f} % drop)')
    log(f'  stored weights : ps_ac     ' + ' '.join(f'{k}={v:.4f}' for k, v in u_ac.items()))
    log(f'                   ps_ac_obs ' + ' '.join(f'{k}={v:.4f}' for k, v in u_obs.items()))
    log('')

    # ---------------- K1a: state-space identity against ctrl2.build ---------
    log('-' * 118)
    log('K1(a)  STATE-SPACE IDENTITY of the hand-built family vs ctrl2.build')
    log('-' * 118)
    ident = {}
    checks = [('ps_ac', u_ac, dict(lam=0.0, sched_K=True, mode='L')),
              ('ps_ac_obs', u_obs, dict(lam=1.0, sched_K=False, mode='L')),
              ('ps_ac_full', u_obs, dict(lam=1.0, sched_K=True, mode='L'))]
    for kind, u, kw in checks:
        ref = K2.build(kind, plant0, u, None)
        mine = factory(u, **kw)(plant0)
        worst = 0.0
        for fr in C.POSITIONS:
            a = ref.at(fr * plate.lp, 0.0)[0]
            b = mine.at(fr * plate.lp, 0.0)[0]
            for Ma, Mb in zip(a, b):
                Ma, Mb = np.atleast_2d(Ma), np.atleast_2d(Mb)
                sc = max(float(np.max(np.abs(Ma))), 1e-30)
                worst = max(worst, float(np.max(np.abs(Ma - Mb))) / sc)
        ident[kind] = worst
        log(f'  {kind:<12} max relative elementwise difference over 5 positions '
            f'= {worst:.3e}   ' + ('OK' if worst < 1e-10 else 'DIFFERS'))
    log('')

    # ---------------- the sweeps -------------------------------------------
    lams = np.round(np.linspace(0.0, 1.0, n_lam), 4)
    runs = []          # (group, label, u, kw, x-value)
    for lam in lams:
        runs.append(('A', f'A lam={lam:.2f}', u_ac,
                     dict(lam=float(lam), sched_K=True, mode='L'), float(lam)))
    for lam in lams:
        runs.append(('B', f'B lam={lam:.2f}', u_obs,
                     dict(lam=float(lam), sched_K=False, mode='L'), float(lam)))
    runs.append(('ANCHOR', 'ANCHOR ps_ac_full', u_obs,
                 dict(lam=1.0, sched_K=True, mode='L'), 1.0))
    for lam in lams:
        runs.append(('R2', f'R2 W-blend lam={lam:.2f}', u_ac,
                     dict(lam=float(lam), sched_K=True, mode='W'), float(lam)))
    for c_ in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        runs.append(('R3', f'R3 floor c={c_:.2f}', u_ac,
                     dict(lam=1.0, sched_K=True, mode='D', dfloor=float(c_)),
                     float(c_)))

    res = []
    cur = None
    for grp, tag, u, kw, xv in runs:
        if grp != cur:
            cur = grp
            title = dict(A='SWEEP A  R1 gain blend, ps_ac weights, sched_K = True'
                           '   (lam = 0 IS the stored ps_ac)',
                         B='SWEEP B  R1 gain blend, ps_ac_obs weights, sched_K = False'
                           '   (lam = 1 IS the stored ps_ac_obs)',
                         ANCHOR='ANCHOR   ps_ac_obs weights, sched_K = True, lam = 1'
                                '   (IS the stored ps_ac_full)',
                         R2='SWEEP R2  covariance blend, ps_ac weights, sched_K = True',
                         R3='SWEEP R3  direction floor at full scheduling, '
                            'ps_ac weights, sched_K = True')[grp]
            log('-' * 118)
            log(title)
            log('-' * 118)
            log(HDR)
        d = measure(plate, plant0, plate_box, factory(u, **kw), tag)
        d['group'], d['x'] = grp, xv
        res.append(d)
        log(row(d))
    log('')

    # ---------------- K1 verdict -------------------------------------------
    stored = dict(ps_ac=(1.4023, 0.7903), ps_ac_obs=(1.6602, 0.0000),
                  ps_ac_full=(1.6758, 0.0000))
    got = {'ps_ac': next(d for d in res if d['tag'] == 'A lam=0.00'),
           'ps_ac_obs': next(d for d in res if d['tag'] == 'B lam=1.00'),
           'ps_ac_full': next(d for d in res if d['tag'] == 'ANCHOR ps_ac_full')}
    log('=' * 118)
    log('K1  ENDPOINT / ANCHOR CHECK   (delta_max to 0.02, S3 to 0.01 mm)')
    log('=' * 118)
    log(f'  {"design":<14}{"delta stored":>14}{"delta here":>12}{"S3 stored":>12}'
        f'{"S3 here":>10}{"verdict":>10}')
    k1 = True
    for k, (ds, ss3) in stored.items():
        d = got[k]
        ok = abs(d['delta_max'] - ds) <= 0.02 and abs(d['S3'] - ss3) <= 0.01
        k1 &= ok
        log(f'  {k:<14}{ds:>14.4f}{d["delta_max"]:>12.4f}{ss3:>12.4f}'
            f'{d["S3"]:>10.4f}{"PASS" if ok else "FAIL":>10}')
    log(f'  K1: {"PASS - the sweep is ANCHORED" if k1 else "FAIL - the run is UNANCHORED"}')
    log('')

    # ---------------- K2 verdict -------------------------------------------
    log('=' * 118)
    log('K2  THE GATE:  S3 > 0.05 mm  AND  delta_max >= 1.54')
    log('=' * 118)
    pas = [d for d in res if d['S3'] > 0.05 and d['delta_max'] >= 1.54]
    adm = [d for d in pas if d['Ms'] <= C.MS_MAX + 1e-9 and d['V'] <= C.V_PER_N + 1e-9]
    if pas:
        for d in pas:
            log('  GATE MET: ' + row(d))
        log(f'  {len(adm)} of {len(pas)} gate-passing points also satisfy the '
            f'protocol screen Ms <= {C.MS_MAX}, V <= {C.V_PER_N}.')
    else:
        alive = [d for d in res if d['S3'] > 0.05]
        best_d = max(alive, key=lambda d: d['delta_max']) if alive else None
        best_s = max(res, key=lambda d: d['S3'])
        log('  NO point in any of the three regularisations meets the gate.')
        if best_d is not None:
            log('  best delta_max among NON-COLLAPSING points:')
            log('    ' + row(best_d))
        log('  largest S3 in the whole probe:')
        log('    ' + row(best_s))
        hi = max(res, key=lambda d: d['delta_max'])
        log('  largest delta_max in the whole probe (collapse status shown):')
        log('    ' + row(hi))
        log('')
        log('  => on these two stored weight sets, the BLENDING ROUTE IS CLOSED:')
        log('     delta_max >= 1.54 and a surviving box limit are not achieved')
        log('     together by any of R1 (gain blend), R2 (covariance blend) or')
        log('     R3 (direction floor).')
    log('')
    log(f'total {time.time()-t0:.0f}s')

    # ---------------- write -------------------------------------------------
    with open(TXT, 'w') as fh:
        fh.write('\n'.join(LINES) + '\n')
    keys = ('mre_nom', 'mre_box', 'mre_box_mid', 'J', 'Ms', 'V', 'delta_max',
            'ap_inf', 'floor', 'S3', 'x', 'secs')
    np.savez(NPZ,
             tags=np.array([d['tag'] for d in res]),
             groups=np.array([d['group'] for d in res]),
             Lnom=np.array([d['Lnom'] for d in res]),
             Lbox=np.array([d['Lbox'] for d in res]),
             positions=np.array(C.POSITIONS),
             identity=np.array([ident[k] for k in ('ps_ac', 'ps_ac_obs', 'ps_ac_full')]),
             k1_pass=np.array([k1]),
             **{k: np.array([d[k] for d in res], float) for k in keys})
    print('written', TXT, 'and', NPZ)


# ---------------------------------------------------------------------------
# ADDENDUM: the points the main sweep showed were the decisive ones.
#   - the cliff of each sweep lies in lam in (0.90, 1.00): measure inside it;
#   - R3 (direction floor) converted the collapse into the LARGEST box limit in
#     the probe, but only at the ps_ac weights, where Ms = 2.337 breaks the
#     protocol.  Applying the same floor at the ps_ac_obs weights, where the
#     stored designs sit at Ms = 2.000/1.999, asks the one question the campaign
#     actually needs answered: is there a FULLY observer-scheduled point that is
#     non-collapsing AND protocol-admissible?
# ---------------------------------------------------------------------------
def _refine_points(u_ac, u_obs):
    pts = []
    for lam in (0.95, 0.99):
        pts.append((f'ADD A lam={lam:.2f}', u_ac,
                    dict(lam=lam, sched_K=True, mode='L'), lam))
        pts.append((f'ADD B lam={lam:.2f}', u_obs,
                    dict(lam=lam, sched_K=False, mode='L'), lam))
    for c_ in (0.02, 0.05):
        pts.append((f'ADD B floor c={c_:.2f}', u_obs,
                    dict(lam=1.0, sched_K=False, mode='D', dfloor=c_), c_))
        pts.append((f'ADD full floor c={c_:.2f}', u_obs,
                    dict(lam=1.0, sched_K=True, mode='D', dfloor=c_), c_))
    return pts


def refine():
    """Cliff and floored-observer addendum, appended to the same artifacts."""
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    plate_box = _perturbed(plate, paper_box_scale(0.10, -0.10, C.N_MODES), 1.0)
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f:
        store = pickle.load(f)
    u_ac = dict(store['ps_ac']['params'])
    u_obs = dict(store['ps_ac_obs']['params'])
    old = np.load(NPZ, allow_pickle=True)

    log('')
    log('=' * 118)
    log('ADDENDUM - THE CLIFF, AND THE DIRECTION FLOOR AT THE ps_ac_obs WEIGHTS')
    log('  Added after the main sweep; NO pre-declared criterion is changed.')
    log('  "ADD A/B lam"      : inside the collapse cliff, lam in (0.90, 1.00).')
    log('  "ADD B floor"      : R3 floor at the ps_ac_obs weights, sched_K =')
    log('                       False, lam = 1 -- i.e. the stored ps_ac_obs with')
    log('                       the mid-edge direction degeneracy floored.')
    log('  "ADD full floor"   : the same floor on the stored ps_ac_full')
    log('                       (sched_K = True, lam = 1).')
    log('=' * 118)
    log(HDR)
    res = []
    for tag, u, kw, xv in _refine_points(u_ac, u_obs):
        d = measure(plate, plant0, plate_box, factory(u, **kw), tag)
        d['group'], d['x'] = 'ADDENDUM', float(xv)
        res.append(d)
        log(row(d))

    ok = [d for d in res
          if d['S3'] > 0.05 and d['delta_max'] >= 1.54
          and d['Ms'] <= C.MS_MAX + 1e-9 and d['V'] <= C.V_PER_N + 1e-9]
    log('')
    if ok:
        best = max(ok, key=lambda d: d['ap_inf'])
        log('  ADDENDUM points that are non-collapsing, meet the K2 gate AND')
        log(f'  satisfy the protocol screen: {len(ok)} of {len(res)}.  Largest')
        log(f'  a_p^inf among them: {best["ap_inf"]:.4f} mm ({best["tag"]}), '
            f'against D1 = 0.4364.')
    else:
        log('  no ADDENDUM point is simultaneously non-collapsing, gate-passing')
        log('  and protocol-admissible.')
    log(f'addendum {time.time()-t0:.0f}s')

    with open(TXT, 'a') as fh:
        fh.write('\n'.join(LINES) + '\n')
    keys = ('mre_nom', 'mre_box', 'mre_box_mid', 'J', 'Ms', 'V', 'delta_max',
            'ap_inf', 'floor', 'S3', 'x', 'secs')
    out = {k: old[k] for k in old.files}
    out['tags'] = np.concatenate([old['tags'], [d['tag'] for d in res]])
    out['groups'] = np.concatenate([old['groups'], [d['group'] for d in res]])
    out['Lnom'] = np.concatenate([old['Lnom'], [d['Lnom'] for d in res]])
    out['Lbox'] = np.concatenate([old['Lbox'], [d['Lbox'] for d in res]])
    for k in keys:
        out[k] = np.concatenate([old[k], [d[k] for d in res]])
    np.savez(NPZ, **out)
    print('appended to', TXT, 'and', NPZ)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'refine':
        refine()
    else:
        main()
