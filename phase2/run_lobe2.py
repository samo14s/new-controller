"""run_lobe2.py — lobe-first round 2: corner parity, the lead structure,
and the widened realizable box.

What round 1 (run_lobe1) measured: the -L winners lifted the full-
resolution floor to 0.407 mm (RFA-L, from 0.340) and 0.238 mm (RI-L, from
0.232) against the benchmark's 0.474 mm — with BOTH shared constraints
still slack (Ms 1.81-1.86 of 2.0, effort 112-160 of 450 V/N).  The
binding pressures were measured directly:

  * RFA-L stands at the corner-envelope screen (corner Ms 1.935 of the
    self-imposed 2.0) — while the benchmark controller itself measures
    corner Ms = 2.119 on the same eight physics corners.  The shared
    protocol bounds the NOMINAL modulus margin (<= 2.0, met by everyone);
    the corner bar was this member's own extra synthesis pressure, held
    STRICTER than what the benchmark achieves.  Round 2 sets the corner
    bar to the benchmark's own measured corner value (parity, not
    relaxation: the shared nominal constraint is unchanged).
  * RI-L stands at its search-box walls (log_ratio 8.999/9, a4_mult
    1.935/2, poles 1.246e5 of the 1.257e5 cap), with corner Ms only 1.09.
    Round 2 widens the GAIN/RATIO box (log_q_pos to 18, log_ratio to 10,
    a4_mult to 2.4); the realizability pole cap — what the certificate
    machinery actually needs — is unchanged and screened as before.

Round-2 arms, all warm-started at the round-1 winners:

  A  ps_ac_rfa_l2 : rfa box, r5 pressures with the corner bar at parity;
  B  ps_ac_rfl_l  : the lead structure (10 params) under the same
                    pressures — phase shaping against the 5.6-5.9 krpm
                    pocket that sets the remaining floor;
  C  ps_ac_ri_l2  : the widened realizable box, pole screen unchanged.

Surrogate fidelity is raised for this round (declared): Floquet m = 60
(round 1: 40) and the speed grid gains 5700 rpm — the full-resolution
floor of round 1 sat at 5700/5900 rpm, between round-1 grid points.

Champions are picked PER CLASS by the full-resolution floor (m = 120,
3600:100:7200, min over positions) among round-1 and round-2 winners, and
recorded in results/lobe_champions.pkl; every arm keeps its own key in
stage3_controllers.pkl (no structure ever overwrites another).

The pre-declared bars are unchanged from run_lobe1: L1 floor > incumbent
(0.340 / 0.232 mm), L2 floor > 0.474 mm (the benchmark), L3 the shared
constraints.  Appends to results/log_lobe1.txt; writes
results/lobes_l2.npz.
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

RPM_GRID = (3600.0, 4000.0, 4400.0, 4900.0, 5300.0, 5600.0, 5700.0,
            5900.0, 6400.0, 7000.0)
RPM_ORDER = (5700.0, 5900.0, 5600.0, 5300.0, 4900.0, 6400.0, 4400.0,
             4000.0, 3600.0, 7000.0)
PROBES_L = (0.5e-3, 0.7e-3, 0.9e-3)
M_SUR = 60
BAIL = 2.0
POLE_MAX = 2 * np.pi * 20e3
RE_VERT = -0.5
PROXY_HI, PROXY_LO = 0.85, 1.6
N_WORKERS = 4

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


# ---------------------------------------------------------------------------
plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate)
CUR = {}


def eval_lobe2(ctrl, detail=False):
    """eval_lobe of run_lobe1 at the round-2 grid and m = 60."""
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
                    r = floquet_log_rho(plate, ctrl, rpm, ap, fr * plate.lp,
                                        m=M_SUR)
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
        while len(margins) < len(PROBES_L):
            margins.append(margins[-1])
            argw.append(argw[-1])
        J = -float(np.mean(margins))
        info.update(feasible=True, J=J, margins=margins, argworst=argw)
        return (J, info) if detail else J
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        info.update(reason='numerical failure', J=-1e4)
        return (info['J'], info) if detail else info['J']


def fit_filtered(u):
    """r5's pressure ladder, corner bar at benchmark parity, J_L2."""
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
    J, info = eval_lobe2(c, detail=True)
    if not info['feasible']:
        return max(-140.0, -100.0 + J / 50.0)
    return J


def fit_ri(u):
    d = CUR['design']
    try:
        c = d.build(u)
        ss, _ = c.at(0.05)
        ev = np.linalg.eigvals(np.atleast_2d(np.asarray(ss[0], float)))
        if np.abs(ev).max() > POLE_MAX:
            return -300.0 - min(np.abs(ev).max() / POLE_MAX, 50.0)
    except Exception:
        return -1e4
    J, info = eval_lobe2(c, detail=True)
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
        log(f'    [it {it + 1:2d}] best J_L = {g_val:+.4f}')
    return g_best, g_val


def encode_params(design, prm, extra=None):
    """Physical parameter dict -> [0,1]^n in `design`'s box (warm re-map)."""
    u = np.empty(design.n)
    for i, nm in enumerate(design.names):
        lo, hi = design.lo[i], design.hi[i]
        if nm in prm:
            v = float(prm[nm])
        elif extra and nm in extra:
            v = float(extra[nm])
        else:
            v = 0.5 * (lo + hi)
        u[i] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return u


def design_round(kind, fit, warm, ms_vert=None):
    d = Design2(kind, plant, plate, None)
    CUR['design'], CUR['fit'] = d, fit
    CUR['ms_vert'] = ms_vert
    log('')
    log(f'--- ROUND 2 ARM {kind}: m = {M_SUR}, grid {len(RPM_GRID)} speeds'
        + (f', corner bar {ms_vert:.3f} (benchmark parity)'
           if ms_vert else '') + ' ---')
    t0 = time.time()
    best = (None, -np.inf)
    with Pool(N_WORKERS) as pool:
        Jw = pool.map(_fit_wrap, [warm], chunksize=1)[0]
        log(f'  warm start under this arm: {Jw:+.4f}')
        for sd in C.OPT['seeds']:
            t = time.time()
            x, J = pso_warm_par(pool, d.n, seed=sd, warm=warm)
            log(f'  seed {sd}: J_L = {J:+.4f}   [{time.time() - t:.0f}s]')
            if J > best[1]:
                best = (x, J)
    x, JL = best
    c = d.build(x)
    JL2, iL = eval_lobe2(c, detail=True)
    Jold, _ = evaluate(plate, c, detail=True)
    ev = np.linalg.eigvals(np.atleast_2d(np.asarray(c.at(0.05)[0][0], float)))
    params = d.decode(x)
    log(f'  winner: J_L = {JL2:+.5f}   J(4900, old probes) = {Jold:+.4f}')
    log(f'  margins per probe: '
        + '  '.join(f'{p * 1e3:.1f}mm:{m:+.3f}@{a}'
                    for p, m, a in zip(PROBES_L, iL['margins'],
                                       iL['argworst'])))
    log(f'  Ms = {iL["Ms"]:.3f}  effort = {iL["V"]:.1f} V/N  max|pole| = '
        f'{np.abs(ev).max():.3e} rad/s  order {c.order}  {c.n_params} '
        f'parameters  [{time.time() - t0:.0f}s]')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in params.items()))
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    store[kind] = dict(x=x, J=float(Jold), J_lobe=float(JL2), params=params,
                       n_params=c.n_params, order=c.order,
                       Ms=float(iL['Ms']), V=float(iL['V']))
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'wb') as f:
        pickle.dump(store, f)
    log(f'  -> stage3_controllers.pkl [{kind}]')
    return c


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    log('=' * 78)
    log('LOBE-FIRST ROUND 2  (run_lobe2.py)  ' + time.strftime('%Y-%m-%d '
                                                               '%H:%M'))
    log('=' * 78)
    store = pickle.load(open(os.path.join(C.RESULTS,
                                          'stage3_controllers.pkl'), 'rb'))
    mks = load_controllers(plate, plant)

    # the parity bar, measured on the benchmark itself, on the record
    _, ms_bm = vertex_screens(plate, plant, mks['mu_tdc'](plant))
    log(f'benchmark corner envelope, measured: corner Ms = {ms_bm:.3f} '
        f'(the round-1 bar was 2.000; nominal shared bar stays '
        f'{C.MS_MAX:.1f})')

    winners = {}
    # arm A: rfa box under parity
    warmA = np.asarray(store['ps_ac_rfa_l']['x'], float)
    winners['ps_ac_rfa_l2'] = design_round('ps_ac_rfa_l2', fit_filtered,
                                           warmA, ms_vert=ms_bm)
    # arm B: the lead structure, warm-mapped from the round-1 winner
    dB = Design2('ps_ac_rfl_l', plant, plate, None)
    warmB = encode_params(dB, store['ps_ac_rfa_l']['params'],
                          extra=dict(log_f_lead=2.925, k_lead=1.5))
    winners['ps_ac_rfl_l'] = design_round('ps_ac_rfl_l', fit_filtered,
                                          warmB, ms_vert=ms_bm)
    # arm C: RI in the widened box (pole cap unchanged)
    dC = Design2('ps_ac_ri_l2', plant, plate, None)
    warmC = encode_params(dC, store['ps_ac_ri_l']['params'])
    winners['ps_ac_ri_l2'] = design_round('ps_ac_ri_l2', fit_ri, warmC)

    # ---- full-resolution verdict + per-class champions ------------------
    log('')
    log('FULL-RESOLUTION LOBES OF THE ROUND-2 ARMS (m = 120, tol 5 um)')
    ref = np.load(os.path.join(C.RESULTS, 'lobes_l.npz'))
    rpms = ref['rpm']
    out = {k: ref[k] for k in ref.files}
    for kind, c in winners.items():
        t0 = time.time()
        cur = []
        for r in rpms:
            L = limits(plate, c, rpm=float(r), positions=(0.0, 0.5, 1.0))
            cur.append(L.min())
        out[kind] = np.array(cur)
        log(f'  {kind:12s}: floor {min(cur) * 1e3:.3f}  max '
            f'{max(cur) * 1e3:.3f} mm  [{time.time() - t0:.0f}s]')
    np.savez_compressed(os.path.join(C.RESULTS, 'lobes_l2.npz'), **out)
    log('-> results/lobes_l2.npz')

    log('')
    log('PER-CLASS CHAMPIONS by full-resolution floor:')
    champs = {}
    for cls, cand in (('filtered', ('ps_ac_rfa', 'ps_ac_rfa_l',
                                    'ps_ac_rfa_l2', 'ps_ac_rfl_l')),
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
    log('run_lobe2 done.')
