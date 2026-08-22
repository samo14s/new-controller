"""
cross_mu.py — every controller judged on EVERY uncertainty description.
========================================================================
mu is a property of a PAIR (controller, uncertainty set).  Reporting mu_phys on
the physics set against mu_paper on the paper's set compares two different
questions, so it proves nothing.  This module builds the table both ways:

    rows    controllers  (mu_paper, mu_exact, mu_phys, ... , FOPID, LQG, PS-AC)
    columns uncertainty descriptions (paper, exact, physics)

Two peaks are reported for each cell:

    mu_RS   robust STABILITY only -- the performance block is removed, so the
            number answers "does the loop survive every plant in the set?"
            (mu_RS < 1 is a guarantee, and it is comparable across rows)
    mu_RP   robust PERFORMANCE -- the full block structure of Eq. (29), which is
            what the D-K iteration minimises.

The diagonal of the table is each design judged on the set it was designed for;
the off-diagonal is the honest cross-check.

    python cross_mu.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control as ct

import config as C
from dk_synthesis import prepare_plant, mu_curve, frf
from mu_tight import mu_curve_tight
from plate_model import build_plate
from robust_design import freq_grid
from uncertain_phys import PhysUncertainSystem

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_cross_mu.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


# ---------------------------------------------------------------------------
def closed_loop(P, ss_K):
    """T = F_l(P_scaled, K_scaled) and the frequency grid, in scaled units."""
    Ps, info = prepare_plant(P)
    w0, u0, y0 = info['w0'], info['u0'], info['y0']
    A, B, Cm, D = [np.atleast_2d(np.asarray(m, float)) for m in ss_K]
    if A.size == 0:
        Ks = ct.ss([], [], [], D * y0 / u0)
    else:
        Ks = ct.ss(A / w0, B / w0 * y0, Cm / u0, D * y0 / u0)
    T = ct.ss(ct.ss(Ps).lft(Ks, P['ncon'], P['nmeas']))
    return T, info


def mu_peaks(P, ss_K, f=None, tight=True):
    """(mu_RS, mu_RP) peaks over the frequency grid.

    With `tight`, repeated real blocks get FULL D-scales (mu_tight); without,
    one scalar per block as in the frozen D-K code.  The paper's set has only
    1x1 blocks, so the two coincide there and the comparison is like for like.
    """
    f = freq_grid() if f is None else np.asarray(f, float)
    T, info = closed_loop(P, ss_K)
    w = 2 * np.pi * f / info['w0']
    H = frf(T, w)
    blocks = P['blocks']
    n_out, n_in = P['n_unc_out'], P['n_unc_in']
    if tight:
        rs = mu_curve_tight(H[:, :n_out, :n_in], blocks[:-1])[1]
        rp = mu_curve_tight(H, blocks)[1]
    else:
        rs = mu_curve(H[:, :n_out, :n_in], blocks[:-1], w)[0]
        rp = mu_curve(H, blocks, w)[0]
    return float(np.max(rs)), float(np.max(rp)), f, rs, rp


# ---------------------------------------------------------------------------
def build_plants(plate):
    """The three generalized plants, all with the SAME synthesis weights."""
    import uncertain_plant as up
    out = {}
    U = up.MillingUncertainSystem(plate, C.RPM_S, C.AP_S, C.AE, n_modes=2,
                                  alpha_coupling='paper', sign=C.SIGN)
    out['paper'] = U.generalized_plant()
    orig = up.nominal_and_perturbations
    up.nominal_and_perturbations = (
        lambda pl, r, a, ae=C.AE, n_modes=2, alpha_coupling='paper',
        mass_pert=0.35, stiff_pert=0.064, damp_pert=0.41, n_pos=201:
        orig(pl, r, a, ae, n_modes, alpha_coupling, 0.35, 0.064, 0.41, n_pos))
    try:
        U = up.MillingUncertainSystem(plate, C.RPM_S, C.AP_S, C.AE, n_modes=2,
                                      alpha_coupling='exact', sign=C.SIGN)
        out['exact'] = U.generalized_plant()
    finally:
        up.nominal_and_perturbations = orig
    Up = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN)
    out['physics'] = Up.generalized_plant()
    return out


def load_controllers():
    out = {}
    p1 = os.path.join(OUT, 'musyn_phase2.npz')
    p2 = os.path.join(OUT, 'musyn_phys.npz')
    for path, tags in ((p1, ('mu_paper', 'mu_exact')),
                       (p2, ('mu_phys', 'mu_phys_sym', 'mu_phys_ind'))):
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        for t in tags:
            if f'{t}_ss0' in d.files:
                out[t] = tuple(d[f'{t}_ss{i}'] for i in range(4))
    # the non-robust baselines, for scale
    try:
        import pickle
        import ctrl2 as K2
        from plant_ss import ControlledPlant
        with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f:
            store = pickle.load(f)
        plant = ControlledPlant(build_plate(patch=C.PATCH_SIDE,
                                            freqs=C.F_MEASURED),
                                C.RPM_S, C.AP_S, n=C.N_MODES_DESIGN)
        for kind in ('fopid', 'lqg', 'ps_ac'):
            if kind not in store:
                continue
            c = K2.build(kind, plant, dict(store[kind]['params']), None)
            ss, _ = c.at(0.5 * plant.plate.lp)
            if ss is not None:
                out[kind + ' (mid-span)'] = ss
    except Exception as e:                                    # noqa: BLE001
        log(f'  (baseline controllers not loaded: {type(e).__name__}: {e})')
    return out


def main():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plants = build_plants(plate)
    ctrls = load_controllers()
    log('=' * 88)
    log('mu OF EVERY CONTROLLER ON EVERY UNCERTAINTY DESCRIPTION')
    log('=' * 88)
    for k, P in plants.items():
        log(f'  set "{k}": {len(P["blocks"])-2} parametric blocks, '
            f'{P["n_unc_in"]-1} channels')
    log('')
    hdr = f'  {"controller":22s}' + ''.join(f'{k:>22s}' for k in plants)
    log(hdr)
    log('  ' + ' ' * 20 + ''.join(f'{"mu_RS / mu_RP":>22s}' for _ in plants))
    log('  ' + '-' * (20 + 22 * len(plants)))
    store = {}
    for name, ss in ctrls.items():
        cells = []
        for k, P in plants.items():
            try:
                rs, rp, f, cs, cp = mu_peaks(P, ss)
                cells.append(f'{rs:9.3f} /{rp:9.3f}')
                store[f'{name}|{k}'] = np.array([rs, rp])
            except Exception as e:                            # noqa: BLE001
                cells.append(f'{"failed":>19s}')
                log(f'    ({name} on {k}: {type(e).__name__}: {e})')
        log(f'  {name:22s}' + ''.join(f'{c:>22s}' for c in cells))
    log('')
    log('  mu_RS < 1 : robust stability GUARANTEED over the whole set')
    log('  mu_RP     : the quantity Eq. (29) minimises (stability + performance)')
    np.savez(os.path.join(OUT, 'cross_mu.npz'),
             **{k.replace('|', '__').replace(' ', '_').replace('(', '')
                 .replace(')', '').replace('-', '_'): v
                for k, v in store.items()})


if __name__ == '__main__':
    main()
