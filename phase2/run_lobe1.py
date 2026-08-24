"""run_lobe1.py — the lobe-first redesign round (suffix -L) of the two
head-to-head scheduled members.

Motivation, measured (results/lobes.npz): the shared objective J probes the
Floquet margin at the DESIGN SPEED only (4900 rpm), so the stored members
were never asked about the rest of the operating band, and the benchmark
mu-TDC's lobe floor (0.474 mm over 3600-7200 rpm, min over tool-path
positions) beats both members (RFA r5 0.340 mm, RI 0.232 mm) at 35-36 of
37 speeds — while the members leave most of the allowed authority unused
(Ms 1.68/1.82 of 2.0, effort 109-151 of 450 V/N).

This round keeps EVERYTHING of each member's own protocol — structure,
parameter count, search box, its declared synthesis pressure (r5's corner
envelope + actuator-band proxy + realizability pole screen; RI's
realizable box + pole screen), the shared constraint screens (nominal
poles, Ms <= 2, effort <= 450 V/N), the optimiser and its settings and
seeds — and changes exactly two things, both declared:

  1. the objective becomes the lobe-floor functional over the operating
     band: J_L = -mean_{a_p in (0.5, 0.7, 0.9) mm}
                  max_{rpm in GRID, x in POSITIONS_DESIGN} log rho,
     GRID = 9 speeds covering 3600-7200 rpm (the stored J is this same
     functional confined to rpm = 4900 and probes (0.3, 0.6, 1.0) mm);
  2. each swarm is warm-started at its member's stored J-protocol winner
     (particle 0 of the Latin hypercube is replaced; everything else,
     including the RNG stream, is unchanged), so the round can only move
     away from the incumbent if the lobe floor pays for it.

Search-loop economy (heuristic only — winners are re-validated at full
resolution): speeds are visited worst-pocket-first and a probe stops early
once its running max log rho exceeds BAIL = 2.0 (such a particle is far
below the competitive band whatever the exact value).

PRE-DECLARED success criteria, judged on the FULL-resolution lobes
(Floquet m = 120, 3600:100:7200, min over positions {0, L/2, L}):

    L1  floor(member-L)  >  floor(member)      (the round pays at all)
    L2  floor(member-L)  >  0.474 mm           (beats the benchmark floor)
    L3  Ms <= 2.0, effort <= 450 V/N, poles <= 2*pi*20 kHz   (constraints)

The mixed-mu gate and the whole-pass certificate are judged AFTER this
round by the standing machinery (a lobe-first member that loses the gate
or the certificate is reported as exactly that).

    python phase2/run_lobe1.py

Appends to results/log_lobe1.txt; stores winners as ps_ac_rfa_l /
ps_ac_ri_l in stage3_controllers.pkl; writes results/lobes_l.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from act_filter import band_proxy
from design2 import Design2
from eval2 import (floquet_log_rho, frequency_metrics, nominal_poles,
                   evaluate, limits)
from pso import latin_hypercube
from run_psacrf import vertex_screens

RPM_GRID = (3600.0, 4000.0, 4400.0, 4900.0, 5300.0, 5600.0, 5900.0,
            6400.0, 7000.0)
RPM_ORDER = (5600.0, 5900.0, 5300.0, 4900.0, 6400.0, 4400.0, 4000.0,
             3600.0, 7000.0)
PROBES_L = (0.5e-3, 0.7e-3, 0.9e-3)
BAIL = 2.0
POLE_MAX = 2 * np.pi * 20e3
RE_VERT, MS_VERT = -0.5, 2.0
PROXY_HI, PROXY_LO = 0.85, 1.6
N_WORKERS = 4

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


# ---------------------------------------------------------------------------
def eval_lobe(plate, ctrl, detail=False):
    """The shared screens of eval2._evaluate, then the lobe-floor objective
    J_L over RPM_GRID x POSITIONS_DESIGN x PROBES_L."""
    info = dict(feasible=False, reason='', Ms=np.nan, V=np.nan, J=-np.inf)
    try:
        mre = -np.inf
        for fr in C.POSITIONS_DESIGN:
            ss, pd = ctrl.at(fr * plate.lp)
            mre = max(mre, float(np.max(nominal_poles(plate, ss, pd).real)))
        info['max_re'] = mre
        if not np.isfinite(mre) or mre > -1.0:
            info['reason'] = 'nominal loop unstable'
            info['J'] = -1e3 - max(mre, 0.0)
            return (info['J'], info) if detail else info['J']
        Ms, V = -np.inf, -np.inf
        for fr in C.POSITIONS_DESIGN:
            ss, pd = ctrl.at(fr * plate.lp)
            a, b = frequency_metrics(plate, ss, pd, C.POSITIONS_DESIGN)
            Ms, V = max(Ms, a), max(V, b)
        info['Ms'], info['V'] = Ms, V
        pen = 0.0
        if Ms > C.MS_MAX:
            pen += 10.0 * (Ms / C.MS_MAX - 1.0)
        if V > C.V_PER_N:
            pen += 10.0 * (V / C.V_PER_N - 1.0)
        if pen > 0.0:
            info['reason'] = f'constraint violated (Ms={Ms:.2f}, V={V:.0f})'
            info['J'] = -100.0 - pen
            return (info['J'], info) if detail else info['J']
        margins, argw = [], []
        for ap in PROBES_L:
            worst, wat = -np.inf, None
            for rpm in RPM_ORDER:
                for fr in C.POSITIONS_DESIGN:
                    r = floquet_log_rho(plate, ctrl, rpm, ap, fr * plate.lp)
                    if r > worst:
                        worst, wat = r, (rpm, fr)
                    if worst > BAIL:
                        break
                if worst > BAIL:
                    break
            margins.append(worst)
            argw.append(wat)
            if worst > BAIL:
                break
        while len(margins) < len(PROBES_L):     # bailed: pessimistic copy
            margins.append(margins[-1])
            argw.append(argw[-1])
        J = -float(np.mean(margins))
        info.update(feasible=True, J=J, margins=margins, argworst=argw)
        return (J, info) if detail else J
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info.update(reason='numerical failure', J=-1e4)
        return (info['J'], info) if detail else info['J']


# ---------------------------------------------------------------------------
# module state inherited by forked workers
plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate)
CUR = {}


def fit_rfa(u):
    """r5's declared pressure ladder, objective swapped for J_L."""
    d = CUR['design']
    try:
        c = d.build(u)
        ss, _ = c.at(0.05)
        ev = np.linalg.eigvals(np.atleast_2d(np.asarray(ss[0], float)))
        if np.abs(ev).max() > POLE_MAX:
            return -400.0 - min(np.abs(ev).max() / POLE_MAX, 50.0)
        re_v, ms_v = vertex_screens(plate, plant, c)
    except Exception:
        return -1e4
    pen = 0.0
    if re_v > RE_VERT:
        pen += 10.0 * min((re_v - RE_VERT) / 10.0, 10.0)
    if ms_v > MS_VERT:
        pen += 10.0 * min(ms_v / MS_VERT - 1.0, 10.0)
    if pen > 0.0:
        return -300.0 - pen
    try:
        hi = lo = 0.0
        for fr in C.POSITIONS_DESIGN:
            h, l = band_proxy(plate, c.at(fr * plate.lp)[0])
            hi, lo = max(hi, h), max(lo, l)
    except Exception:
        return -1e4
    pen = 0.0
    if hi > PROXY_HI:
        pen += 10.0 * min(hi / PROXY_HI - 1.0, 8.0)
    if lo > PROXY_LO:
        pen += 10.0 * min(lo / PROXY_LO - 1.0, 8.0)
    if pen > 0.0:
        return -200.0 - pen
    J, info = eval_lobe(plate, c, detail=True)
    if not info['feasible']:
        return max(-140.0, -100.0 + J / 50.0)
    return J


def fit_ri(u):
    """RI's declared pressure (realizable box + pole screen) + J_L."""
    d = CUR['design']
    try:
        c = d.build(u)
        ss, _ = c.at(0.05)
        ev = np.linalg.eigvals(np.atleast_2d(np.asarray(ss[0], float)))
        if np.abs(ev).max() > POLE_MAX:
            return -300.0 - min(np.abs(ev).max() / POLE_MAX, 50.0)
    except Exception:
        return -1e4
    J, info = eval_lobe(plate, c, detail=True)
    return J


def _fit_wrap(u):
    return CUR['fit'](u)


# ---------------------------------------------------------------------------
def pso_warm_par(pool, n_dim, seed, warm=None):
    """pso.pso verbatim (same RNG stream, same updates), with two declared
    changes: particle 0 of the initial hypercube is replaced by `warm`, and
    the per-particle fitness evaluations run on a worker pool (order-exact,
    so results are identical to the serial loop)."""
    cfg = dict(C.PSO)
    n_particles, n_iter = cfg['n_particles'], cfg['n_iter']
    w, c1, c2, v_max = cfg['w'], cfg['c1'], cfg['c2'], cfg['v_max']
    rng = np.random.default_rng(seed)
    x = latin_hypercube(n_particles, n_dim, rng)
    if warm is not None:
        x[0] = np.clip(np.asarray(warm, float), 0.0, 1.0)
    v = (rng.random((n_particles, n_dim)) - 0.5) * v_max
    f = np.array(pool.map(_fit_wrap, list(x), chunksize=1))
    p_best, p_val = x.copy(), f.copy()
    g = int(np.argmax(p_val))
    g_best, g_val = p_best[g].copy(), float(p_val[g])
    hist = [g_val]
    for it in range(n_iter):
        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        below, above = x < 0.0, x > 1.0
        x[below] = -x[below]
        v[below] *= -0.5
        x[above] = 2.0 - x[above]
        v[above] *= -0.5
        x = np.clip(x, 0.0, 1.0)
        f = np.array(pool.map(_fit_wrap, list(x), chunksize=1))
        imp = f > p_val
        p_best[imp], p_val[imp] = x[imp], f[imp]
        g = int(np.argmax(p_val))
        if p_val[g] > g_val:
            g_best, g_val = p_best[g].copy(), float(p_val[g])
        hist.append(g_val)
        log(f'    [it {it + 1:2d}] best J_L = {g_val:+.4f}')
    return g_best, g_val, hist


# ---------------------------------------------------------------------------
def design_round(kind_base, kind_l, fit):
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    warm = np.asarray(store[kind_base]['x'], float)
    d = Design2(kind_base, plant, plate, None)
    CUR['design'], CUR['fit'] = d, fit
    log('')
    log(f'--- LOBE-FIRST ROUND ({kind_l}): structure/box/screens/optimiser/'
        'seeds of the incumbent,')
    log(f'    objective J_L over {len(RPM_GRID)} speeds x '
        f'{len(C.POSITIONS_DESIGN)} positions x probes '
        f'{tuple(p * 1e3 for p in PROBES_L)} mm, warm-started ---')
    t0 = time.time()
    best = (None, -np.inf)
    with Pool(N_WORKERS) as pool:
        Jw = pool.map(_fit_wrap, [warm], chunksize=1)[0]
        log(f'  incumbent under J_L: {Jw:+.4f}')
        for sd in C.OPT['seeds']:
            t = time.time()
            x, J, hist = pso_warm_par(pool, d.n, seed=sd, warm=warm)
            log(f'  seed {sd}: J_L = {J:+.4f}   [{time.time() - t:.0f}s]')
            if J > best[1]:
                best = (x, J)
    x, JL = best
    c = d.build(x)
    JL2, iL = eval_lobe(plate, c, detail=True)
    Jold, iold = evaluate(plate, c, detail=True)
    ev = np.linalg.eigvals(np.atleast_2d(np.asarray(c.at(0.05)[0][0], float)))
    params = d.decode(x)
    log(f'  winner: J_L = {JL2:+.5f} (search {JL:+.5f}, incumbent {Jw:+.5f})'
        f'   J(4900, old probes) = {Jold:+.4f}')
    log(f'  margins per probe: '
        + '  '.join(f'{p * 1e3:.1f}mm:{m:+.3f}@{a}'
                    for p, m, a in zip(PROBES_L, iL['margins'],
                                       iL['argworst'])))
    log(f'  Ms = {iL["Ms"]:.3f}  effort = {iL["V"]:.1f} V/N  '
        f'max|pole| = {np.abs(ev).max():.3e} rad/s  order {c.order}  '
        f'{c.n_params} parameters  [{time.time() - t0:.0f}s]')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in params.items()))
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    store[kind_l] = dict(x=x, J=float(Jold), J_lobe=float(JL2),
                         params=params, n_params=c.n_params, order=c.order,
                         Ms=float(iL['Ms']), V=float(iL['V']))
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'wb') as f:
        pickle.dump(store, f)
    log(f'  -> stage3_controllers.pkl [{kind_l}]')
    return c


# ---------------------------------------------------------------------------
def full_lobes(winners):
    """Full-resolution verdict: Floquet m = 120, 3600:100:7200, min over
    the tool-path positions, against the stored curves of lobes.npz."""
    log('')
    log('FULL-RESOLUTION LOBES OF THE -L WINNERS (m = 120, tol 5 um, '
        'min over {0, L/2, L})')
    ref = np.load(os.path.join(C.RESULTS, 'lobes.npz'))
    rpms = ref['rpm']
    out = {'rpm': rpms}
    for k in ('open', 'mu_tdc', 'ps_ac_rfa', 'ps_ac_ri'):
        out[k] = ref[k]
    for kind_l, c in winners.items():
        t0 = time.time()
        cur = []
        for r in rpms:
            L = limits(plate, c, rpm=float(r), positions=(0.0, 0.5, 1.0))
            cur.append(L.min())
        out[kind_l] = np.array(cur)
        log(f'  {kind_l:12s}: floor {min(cur) * 1e3:.3f}  max '
            f'{max(cur) * 1e3:.3f} mm  [{time.time() - t0:.0f}s]')
    np.savez_compressed(os.path.join(C.RESULTS, 'lobes_l.npz'), **out)
    log('-> results/lobes_l.npz')
    log('')
    log('PRE-DECLARED VERDICT (L1 floor>incumbent, L2 floor>0.474 mm '
        '= mu-TDC, L3 constraints):')
    for kind_l, base in (('ps_ac_rfa_l', 'ps_ac_rfa'),
                         ('ps_ac_ri_l', 'ps_ac_ri')):
        if kind_l not in out:
            continue
        fl, fb = out[kind_l].min() * 1e3, out[base].min() * 1e3
        fm = out['mu_tdc'].min() * 1e3
        log(f'  {kind_l:12s}: floor {fl:.3f} mm | incumbent {fb:.3f} '
            f'(L1 {"PASS" if fl > fb else "FAIL"}) | mu-TDC {fm:.3f} '
            f'(L2 {"PASS" if fl > fm else "FAIL"})')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    log('=' * 78)
    log('LOBE-FIRST REDESIGN ROUND  (run_lobe1.py)  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 78)
    winners = {}
    winners['ps_ac_rfa_l'] = design_round('ps_ac_rfa', 'ps_ac_rfa_l',
                                          fit_rfa)
    winners['ps_ac_ri_l'] = design_round('ps_ac_ri', 'ps_ac_ri_l', fit_ri)
    full_lobes(winners)
    log('run_lobe1 done.')
