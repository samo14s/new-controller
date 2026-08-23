"""run_wtdc.py — mu-TDC-PS: the witness base with the position-scheduled pair.

The fairness balance of docs/09 named the open gap exactly: the fixed witness
holds G1(real) + the G3 floor + G4 from the first study's own tables, the
scheduled members hold parametric depth and a gateless scheduling floor, and
nobody holds both.  This member is the cheapest possible attack on that gap:

    mu-TDC-PS :  the STORED mu_paper controller (untouched), plus the
                 Eq. (30) delayed pair scheduled on the milling position --
                 pd(x) = (kpd k0(x), kdd k0(x)/omega_1), k0(x) the
                 cancellation gain AT the position.

Same six parameters as mu-TDC (four identified weights + two tuned gains;
the schedule reads the known position, adding none).  Its G1(real) standing
is INHERITED EXACTLY -- the mu judge is delay-free (Eq. 28 convention) and
tests the very matrices the witness scores 0.248-0.419 with -- so the whole
question lives on the delay axis: the shared J, the crossing certificate and
the stage-7/8 protocol, against the pre-declared bars:

    C1  a_p^inf    >  0.4364 mm   (beat mu-TDC's certified depth)
    C2  delta_max  >= 1.54        (beat mu-TDC's certified inflation)
    C3  S1 floor   >= 0.6821 mm   (keep mu-TDC's own nominal floor)
    C5  no scenario collapses     (> 0.05 mm)

Anything short is reported as exactly that.

    python phase2/run_wtdc.py                 (design + certify + scenarios)
    python phase2/run_wtdc.py design certify scenarios verdict

Appends to results/log_wtdc.txt; stores under 'mu_tdc_ps' in the stage pkls.
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
from plate_model import build_plate
from plant_ss import ControlledPlant
from design2 import Design2, optimise
from eval2 import evaluate
from stage_common import load_controllers, load_mu_ss

OUT = C.RESULTS
KIND = 'mu_tdc_ps'
TARGET = dict(ap_inf=0.4364e-3, dmax=1.54, floor=0.6821e-3)
LOG = open(os.path.join(OUT, 'log_wtdc.txt'), 'a')


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
def design(plate, plant):
    log('')
    log('--- DESIGN mu_tdc_ps: the stage-3 PSO on (kpd, kdd), witness base '
        'frozen ---')
    d = Design2(KIND, plant, plate, load_mu_ss('mu_paper'))
    t0 = time.time()
    r = optimise(d)
    J, info = evaluate(plate, r['ctrl'], detail=True)
    log(f'  J = {r["J"]:+.5f}   order {r["order"]}   {r["n_params"]} '
        f'parameters   SCHEDULED pair   [{time.time()-t0:.0f}s]')
    log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
        f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in r['params'].items()))
    upd('stage3_controllers.pkl',
        {KIND: dict(x=r['x'], J=r['J'], params=r['params'],
                    n_params=r['n_params'], order=r['order'],
                    Ms=float(info['Ms']), V=float(info['V']))})
    log('  -> stage3_controllers.pkl')


# ---------------------------------------------------------------------------
def certify(plate):
    """The exact stage-56 block: crossing + margins + common-P LK."""
    import certify2 as CF2
    tau0 = 60.0 / (3 * C.RPM_S)
    etas = tuple(np.linspace(0.0, C.ETA_MAX, 3))
    zetas = (C.ZETA_LO, C.ZETA_HI)
    full = dict(n_pos=9, etas=etas, zetas=zetas, xis=(0.5, 1.0))
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    mk = load_controllers(plate, plant0)[KIND]
    c0 = mk(plant0)

    log('')
    log(f'--- CERTIFY {KIND}: same vertex family and bisections as '
        'stage 5-6 ---')
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
    log(f'  a_p^inf = {ap_inf*1e3:.4f} mm   (mu-TDC: '
        f'{TARGET["ap_inf"]*1e3:.4f})   [{time.time()-t:.0f}s]')
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
        {KIND: dict(peak=peak, tau_max=tm, tau_ratio=tm / tau0,
                    ap_inf=ap_inf, delta_max=dmax, lk=bool(lk['feasible']),
                    lk_attempted=bool(lk.get('attempted', True)))})
    log('  -> stage56.pkl')


# ---------------------------------------------------------------------------
def scenarios(plate):
    import run_stage78 as S78
    plant = ControlledPlant(plate)
    mk = load_controllers(plate, plant)[KIND]
    log('')
    log(f'--- SCENARIOS {KIND}: stages 7-8, same code path as everyone ---')
    p = os.path.join(OUT, 'stage78.pkl')
    store = pickle.load(open(p, 'rb'))
    for fn in (S78.s1, S78.s2, S78.s3, S78.s4):
        fn(plate, plant, {KIND: mk}, store)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    log('  -> stage78.pkl')


# ---------------------------------------------------------------------------
def verdict():
    s56 = pickle.load(open(os.path.join(OUT, 'stage56.pkl'), 'rb'))
    s78 = pickle.load(open(os.path.join(OUT, 'stage78.pkl'), 'rb'))
    r = s56.get(KIND, {})
    if f'S1_{KIND}' not in s78 or not r:
        log(f'\n  {KIND}: not fully on disk yet, no verdict')
        return
    S1 = np.asarray(s78[f'S1_{KIND}']['limits'], float)
    S3 = np.asarray(s78[f'S3_{KIND}'], float)
    S4 = np.asarray(s78[f'S4_{KIND}'], float)
    checks = [
        ('C1  a_p^inf  > 0.4364 mm  (mu-TDC)',
         f'{r["ap_inf"]*1e3:.4f}', r['ap_inf'] > TARGET['ap_inf']),
        ('C2  delta_max >= 1.54     (mu-TDC)',
         f'{r["delta_max"]:.2f}', r['delta_max'] >= TARGET['dmax']),
        ('C3  S1 floor >= 0.6821 mm (mu-TDC)',
         f'{S1.min()*1e3:.4f}', S1.min() >= TARGET['floor']),
        ('C5  no scenario collapses to 0',
         f'{min(S3.min(), S4.min()):.4f}', min(S3.min(), S4.min()) > 0.05),
        ('G1(real), inherited from the witness verbatim',
         '0.248-0.419', True),
    ]
    log('')
    log('--- VERDICT mu-TDC-PS against the pre-declared bars ' + '-' * 18)
    for name, val, ok in checks:
        log(f'  {"PASS" if ok else "FAIL"}  {name:<44s} {val}')
    n = sum(ok for _, _, ok in checks)
    log(f'  {n}/5 criteria met'
        + ('' if n == 5 else ' -- reported as exactly that'))
    log(f'  (common-P LK: {"yes" if r.get("lk") else "no"})')


# ---------------------------------------------------------------------------
STAGES = dict(design=design, certify=lambda pl, pt: certify(pl),
              scenarios=lambda pl, pt: scenarios(pl),
              verdict=lambda pl, pt: verdict())


def main(args):
    t0 = time.time()
    which = [a for a in args if a in STAGES] or list(STAGES)
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    log('=' * 74)
    log('mu-TDC-PS - THE WITNESS BASE WITH THE POSITION-SCHEDULED PAIR')
    log('=' * 74)
    for name in which:
        STAGES[name](plate, plant)
    log(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])
