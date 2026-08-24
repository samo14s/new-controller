"""run_lobe3.py — lobe-first round 3: the pocket polish at full Floquet
resolution.

Round 2's corner-parity arm (ps_ac_rfa_l2) lifted the full-resolution
floor to 0.439 mm with the SHARED nominal constraint now binding exactly
(Ms = 1.999 of 2.0 — the bar the benchmark itself sits at, 2.00).  Only
six of the 37 speeds still sit below the benchmark's floor of 0.474 mm:
the 5600-5900 rpm pocket (0.439-0.442) and the narrow 3700/4300 dips
(0.451/0.469).  Everything global has been traded; what remains is local
reshaping and the m = 60 -> 120 surrogate mismatch.

This round therefore polishes ps_ac_rfa_l2 AT THE JUDGE'S OWN RESOLUTION
(Floquet m = 120) on exactly the binding speeds, with probes bracketing
the decision depth:

    speeds  (3700, 4300, 5600, 5700, 5800, 5900) x positions {0, L/2, L}
    probes  (0.44, 0.48, 0.52) mm

Pressures, box, optimiser, seeds: exactly round 2's arm A (corner bar at
the measured benchmark parity 2.119; shared nominal screens unchanged).
Warm-started at the round-2 winner.  The full-resolution 37-speed sweep
remains the judge: a polish that sinks a speed outside the polished set
is caught there, and the per-class champion logic (results/
lobe_champions.pkl) keeps whichever member measures best.

Appends to results/log_lobe1.txt; updates lobes_l2.npz curves with
ps_ac_rfa_l3 and refreshes lobe_champions.pkl.
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
from stage_common import load_controllers

RPM_POL = (5700.0, 5900.0, 5800.0, 5600.0, 3700.0, 4300.0)
PROBES_P = (0.44e-3, 0.48e-3, 0.52e-3)
M_POL = 120
BAIL = 2.0
POLE_MAX = 2 * np.pi * 20e3
RE_VERT = -0.5
PROXY_HI, PROXY_LO = 0.85, 1.6
N_WORKERS = 4
KIND = 'ps_ac_rfa_l3'

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate)
CUR = {}


def eval_pol(ctrl, detail=False):
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
        for ap in PROBES_P:
            worst, wat = -np.inf, None
            for rpm in RPM_POL:
                for fr in C.POSITIONS_DESIGN:
                    r = floquet_log_rho(plate, ctrl, rpm, ap, fr * plate.lp,
                                        m=M_POL)
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
        while len(margins) < len(PROBES_P):
            margins.append(margins[-1])
            argw.append(argw[-1])
        J = -float(np.mean(margins))
        info.update(feasible=True, J=J, margins=margins, argworst=argw)
        return (J, info) if detail else J
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info.update(reason='numerical failure', J=-1e4)
        return (info['J'], info) if detail else info['J']


def fit_pol(u):
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
    if ms_v > CUR['ms_vert']:
        pen += 10.0 * min(ms_v / CUR['ms_vert'] - 1.0, 10.0)
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
    J, info = eval_pol(c, detail=True)
    if not info['feasible']:
        return max(-140.0, -100.0 + J / 50.0)
    return J


def _fit_wrap(u):
    return CUR['fit'](u)


def pso_warm_par(pool, n_dim, seed, warm=None):
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
        log(f'    [it {it + 1:2d}] best J_P = {g_val:+.4f}')
    return g_best, g_val


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    log('=' * 78)
    log('LOBE-FIRST ROUND 3 (pocket polish, m = 120)  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 78)
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    mks = load_controllers(plate, plant)
    _, ms_bm = vertex_screens(plate, plant, mks['mu_tdc'](plant))
    log(f'corner bar (benchmark parity, re-measured): {ms_bm:.3f}')

    d = Design2(KIND, plant, plate, None)
    CUR['design'], CUR['fit'], CUR['ms_vert'] = d, fit_pol, ms_bm
    warm = np.asarray(store['ps_ac_rfa_l2']['x'], float)
    log(f'polish speeds {tuple(int(r) for r in RPM_POL)} rpm, probes '
        f'{tuple(round(p * 1e3, 2) for p in PROBES_P)} mm, m = {M_POL}')
    t0 = time.time()
    best = (None, -np.inf)
    with Pool(N_WORKERS) as pool:
        Jw = pool.map(_fit_wrap, [warm], chunksize=1)[0]
        log(f'  round-2 winner under the polish objective: {Jw:+.4f}')
        for sd in C.OPT['seeds']:
            t = time.time()
            x, J = pso_warm_par(pool, d.n, seed=sd, warm=warm)
            log(f'  seed {sd}: J_P = {J:+.4f}   [{time.time() - t:.0f}s]')
            if J > best[1]:
                best = (x, J)
    x, JP = best
    c = d.build(x)
    JP2, iP = eval_pol(c, detail=True)
    Jold, _ = evaluate(plate, c, detail=True)
    ev = np.linalg.eigvals(np.atleast_2d(np.asarray(c.at(0.05)[0][0], float)))
    params = d.decode(x)
    log(f'  winner: J_P = {JP2:+.5f} (warm {Jw:+.5f})   J(4900, old '
        f'probes) = {Jold:+.4f}')
    log(f'  margins per probe: '
        + '  '.join(f'{p * 1e3:.2f}mm:{m:+.3f}@{a}'
                    for p, m, a in zip(PROBES_P, iP['margins'],
                                       iP['argworst'])))
    log(f'  Ms = {iP["Ms"]:.3f}  effort = {iP["V"]:.1f} V/N  max|pole| = '
        f'{np.abs(ev).max():.3e} rad/s  [{time.time() - t0:.0f}s]')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in params.items()))
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    store[KIND] = dict(x=x, J=float(Jold), J_lobe=float(JP2), params=params,
                       n_params=c.n_params, order=c.order,
                       Ms=float(iP['Ms']), V=float(iP['V']))
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'wb') as f:
        pickle.dump(store, f)
    log(f'  -> stage3_controllers.pkl [{KIND}]')

    # ---- full-resolution verdict + champion refresh ---------------------
    log('')
    log('FULL-RESOLUTION LOBES (m = 120): ps_ac_rfa_l3')
    p = os.path.join(C.RESULTS, 'lobes_l2.npz')
    ref = np.load(p)
    rpms = ref['rpm']
    out = {k: ref[k] for k in ref.files}
    t0 = time.time()
    cur = []
    for r in rpms:
        L = limits(plate, c, rpm=float(r), positions=(0.0, 0.5, 1.0))
        cur.append(L.min())
    out[KIND] = np.array(cur)
    log(f'  {KIND}: floor {min(cur) * 1e3:.3f}  max {max(cur) * 1e3:.3f} '
        f'mm  [{time.time() - t0:.0f}s]')
    np.savez_compressed(p, **out)
    log('-> results/lobes_l2.npz (updated)')

    log('')
    log('PER-CLASS CHAMPIONS refreshed:')
    champs = {}
    for cls, cand in (('filtered', ('ps_ac_rfa', 'ps_ac_rfa_l',
                                    'ps_ac_rfa_l2', 'ps_ac_rfl_l',
                                    'ps_ac_rfa_l3')),
                      ('realizable', ('ps_ac_ri', 'ps_ac_ri_l',
                                      'ps_ac_ri_l2'))):
        floors = {k: float(out[k].min()) for k in cand if k in out}
        champ = max(floors, key=floors.get)
        champs[cls] = champ
        fm = float(out['mu_tdc'].min()) * 1e3
        log(f'  {cls:10s}: ' + '  '.join(f'{k} {v * 1e3:.3f}'
                                         for k, v in floors.items()))
        log(f'    champion {champ}  floor {floors[champ] * 1e3:.3f} mm  '
            f'(L2 vs benchmark {fm:.3f}: '
            f'{"PASS" if floors[champ] * 1e3 > fm else "FAIL"})')
    with open(os.path.join(C.RESULTS, 'lobe_champions.pkl'), 'wb') as f:
        pickle.dump(champs, f)
    log('-> results/lobe_champions.pkl')
    log('run_lobe3 done.')
