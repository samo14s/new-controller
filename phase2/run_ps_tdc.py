"""run_ps_tdc.py — design and judge the PS-TDC hybrids in one pass.

The failure map after stages 5-8 is complementary, and nobody wins it all:

                     a_p^inf   delta_max   S1 floor   worst-of-4   box    LK
    mu-TDC            0.4364     1.54       0.6821      0.442      ok     no
    PS-AC             0.3836     1.40       0.8517      0.653      ok     no
    PS-AC K+obs       0.4246     1.68       0.8927      0.000    FAILS   yes

PS-TDC = the scheduled law of PS-AC plus the Eq. (30) delayed PD.  The
mechanism is measured, not guessed: isolating the delayed PD on the stored mu
design (results/isolate_mu.npz) showed the PD alone carries +5.7 % of a_p^inf.

Two variants, both under the stage-3 protocol (same J, PSO, seeds,
constraints):

  ps_tdc    frozen base -- the stored PS-AC gains, PSO on (kpd, kdd) only.
            Mirrors mu-TDC's procedure exactly (frozen mu design + tuned
            pair).  A first probe showed the trap: the stored base sits at
            Ms = 1.990 of the 2.0 bound, so the pair has almost no room.
  ps_tdc_j  joint -- all six parameters searched together, so the base can
            back off the modulus margin to buy the delayed term.  Six tuned
            parameters, the same freedom the tables grant mu-TDC.

Success criteria, PRE-DECLARED before the run:

    C1  a_p^inf    >  0.4364 mm    (beat mu-TDC's certified depth)
    C2  delta_max  >= 1.54         (beat mu-TDC's certified inflation)
    C3  S1 floor   >= 0.8459 mm    (keep at least LQG's nominal floor)
    C4  worst-of-4 >= 0.6528 mm    (keep PS-AC's worst case)
    C5  no scenario collapses to 0 (survive the +10 % box)

Anything short of all five is reported as exactly what it is.

    python phase2/run_ps_tdc.py                     (both variants, full)
    python phase2/run_ps_tdc.py ps_tdc_j            (one variant)
    python phase2/run_ps_tdc.py verdict             (re-print from disk)
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
from design2 import Design2, optimise
from eval2 import evaluate
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import LABEL, load_controllers

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_ps_tdc.txt'), 'a')

KINDS = ('ps_tdc', 'ps_tdc_j', 'ps_ac_r', 'ps_tdc_r')
TARGET = dict(ap_inf=0.4364e-3, dmax=1.54, floor=0.8459e-3, worst=0.6528)


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def upd(fname, patch):
    p = os.path.join(OUT, fname)
    store = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
    store.update(patch)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    return store


# ---------------------------------------------------------------------------
def design(kind, plate, plant):
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f:
        st3 = pickle.load(f)
    base = (dict(st3['ps_ac']['params']) if kind == 'ps_tdc'
            else dict(st3['ps_ac_r']['params']) if kind == 'ps_tdc_r'
            else None)
    what = dict(ps_tdc='PSO on (kpd, kdd), base = stored PS-AC gains',
                ps_tdc_j='joint PSO on all six parameters',
                ps_ac_r='PSO on the four gains + the design coefficient',
                ps_tdc_r='PSO on (kpd, kdd), base = stored PS-AC-R gains')
    log(f'\n--- DESIGN {kind}: ' + what.get(kind, kind) + ' ' + '-' * 8)
    d = Design2(kind, plant, plate, base)
    t = time.time()
    r = optimise(d)
    J, info = evaluate(plate, r['ctrl'], detail=True)
    log(f'  J = {r["J"]:+.5f}   order {r["order"]}   '
        f'{r["n_params"]} parameters   [{time.time()-t:.0f}s]')
    log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
        f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in r['params'].items()))
    upd('stage3_controllers.pkl',
        {kind: dict(x=r['x'], J=r['J'], params=r['params'],
                    n_params=r['n_params'], order=r['order'],
                    Ms=info['Ms'], V=info['V'])})
    log('  -> stage3_controllers.pkl')


# ---------------------------------------------------------------------------
def certify(kind, plate):
    """The exact stage-56 block, for this controller alone."""
    tau0 = 60.0 / (3 * C.RPM_S)
    etas = tuple(np.linspace(0.0, C.ETA_MAX, 3))
    zetas = (C.ZETA_LO, C.ZETA_HI)
    full = dict(n_pos=9, etas=etas, zetas=zetas, xis=(0.5, 1.0))

    plant0 = ControlledPlant(plate, ap=C.AP_S)
    mk = load_controllers(plate, plant0)[kind]
    c0 = mk(plant0)

    log(f'\n--- CERTIFY {kind}: same vertex family and bisections as '
        'stage 5-6 ' + '-' * 6)
    t = time.time()
    tm, peak, ok = CF2.analyse(plant0, c0, **full)
    ratio = 'inf' if np.isinf(tm) else f'{tm/tau0:.2f}'
    log(f'  peak = {peak:.3f}   tau_max/tau0 = {ratio}   '
        f'[{time.time()-t:.0f}s]')

    def plant_of(ap):
        return ControlledPlant(plate, ap=ap)

    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    t = time.time()
    ap_inf = CF2.depth_bisect(plant_of, mk, n_iter=16, **light)
    log(f'  a_p^inf = {ap_inf*1e3:.4f} mm   '
        f'(mu-TDC: {TARGET["ap_inf"]*1e3:.4f})   [{time.time()-t:.0f}s]')
    t = time.time()
    dmax = CF2.margin_bisect(plant0, c0, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    log(f'  delta_max = {dmax:.2f}   (mu-TDC: {TARGET["dmax"]:.2f})   '
        f'[{time.time()-t:.0f}s]')
    pl_lk = ControlledPlant(plate, ap=max(ap_inf, 1e-6))
    lk = CF2.lk_common_P(pl_lk, mk(pl_lk), n_pos=3,
                         etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    lk_txt = ('yes' if lk['feasible']
              else ('no' if lk.get('attempted', True) else 'n/a'))
    log(f'  common-P LK at a_p^inf: {lk_txt}')

    upd('stage56.pkl',
        {kind: dict(peak=peak, tau_max=tm, tau_ratio=tm / tau0,
                    ap_inf=ap_inf, delta_max=dmax,
                    lk=bool(lk['feasible']),
                    lk_attempted=bool(lk.get('attempted', True)))})
    log('  -> stage56.pkl')


# ---------------------------------------------------------------------------
def scenarios(kind, plate):
    """Stages 7-8 for this controller alone, appended to stage78.pkl.

    run_stage78 opens its log in append mode, so importing and reusing its
    scenario functions is safe and guarantees the numbers come from the same
    code path as everyone else's.
    """
    import run_stage78 as S78
    plant = ControlledPlant(plate)
    mk = load_controllers(plate, plant)[kind]
    ctrls = {kind: mk}
    p = os.path.join(OUT, 'stage78.pkl')
    store = pickle.load(open(p, 'rb'))
    for fn in (S78.s1, S78.s2, S78.s3, S78.s4):
        fn(plate, plant, ctrls, store)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    log('  -> stage78.pkl')


# ---------------------------------------------------------------------------
def verdict(kind):
    s56 = pickle.load(open(os.path.join(OUT, 'stage56.pkl'), 'rb'))
    s78 = pickle.load(open(os.path.join(OUT, 'stage78.pkl'), 'rb'))
    if kind not in s56 or f'S1_{kind}' not in s78:
        log(f'\n  {kind}: not fully on disk yet, no verdict')
        return
    r = s56[kind]
    S1 = np.asarray(s78[f'S1_{kind}']['limits'], float)
    S2 = np.asarray(s78[f'S2_{kind}'], float)
    S3 = np.asarray(s78[f'S3_{kind}'], float)
    S4 = np.asarray(s78[f'S4_{kind}'], float)
    worst4 = min(S1.min() * 1e3, S2.min() * 1e3, S3.min(), S4.min())

    checks = [
        ('C1  a_p^inf  > 0.4364 mm  (mu-TDC)',
         f'{r["ap_inf"]*1e3:.4f}', r['ap_inf'] > TARGET['ap_inf']),
        ('C2  delta_max >= 1.54     (mu-TDC)',
         f'{r["delta_max"]:.2f}', r['delta_max'] >= TARGET['dmax']),
        ('C3  S1 floor >= 0.8459 mm (LQG)',
         f'{S1.min()*1e3:.4f}', S1.min() >= TARGET['floor']),
        ('C4  worst-of-4 >= 0.6528  (PS-AC)',
         f'{worst4:.4f}', worst4 >= TARGET['worst'] - 1e-9),
        ('C5  no scenario collapses to 0',
         f'{min(S3.min(), S4.min()):.4f}', min(S3.min(), S4.min()) > 0.05),
    ]
    log(f'\n--- VERDICT {LABEL.get(kind, kind)} against the pre-declared '
        'criteria ' + '-' * 12)
    for name, val, ok in checks:
        log(f'  {"PASS" if ok else "FAIL"}  {name:<38s} {val}')
    n = sum(ok for _, _, ok in checks)
    log(f'  {n}/5 criteria met'
        + ('' if n == 5 else ' -- reported as exactly that'))
    log(f'  (LK common-P: {"yes" if r["lk"] else "no"})')


# ---------------------------------------------------------------------------
def main(args):
    t0 = time.time()
    kinds = tuple(a for a in args if a in KINDS) or KINDS
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    log('=' * 74)
    log('PS-TDC - THE HYBRIDS: SCHEDULED GAIN + EQ. (30) DELAYED PD')
    log('=' * 74)
    if 'verdict' not in args:
        for kind in kinds:
            design(kind, plate, plant)
            certify(kind, plate)
            scenarios(kind, plate)
    for kind in kinds:
        verdict(kind)
    log(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])
