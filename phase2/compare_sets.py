"""
compare_sets.py — geometry of the three uncertainty descriptions.
==================================================================
Two numbers decide whether an uncertainty set is the right one, and they pull in
opposite directions:

  COVERAGE GAP    the largest distance from a point of the PHYSICAL family to the
                  set.  Non-zero means the design is not robust to something the
                  process actually does -- the guarantee is void.
  SPURIOUS RADIUS the largest distance from a point of the SET to the physical
                  family.  Large means the design is paying for perturbations
                  the process cannot produce -- performance is given away.

Both are measured in the same norm, ||dA||_F / ||A_0||_F, on the state matrix of
the 2-mode design model, so they are directly comparable.  The Frobenius norm is
used because it is smooth: the minimisation behind each number then converges,
and the numbers are geometry rather than optimiser artefacts.  The distance to the
physical family is taken to its CONVEX HULL (non-negative least squares on a
dense sample), which under-estimates the spurious radius: any gap it reports is
a lower bound of the true one.

    python compare_sets.py
"""
import itertools
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import minimize, nnls

import config as C
from plate_model import build_plate
from uncertain_phys import PhysUncertainSystem

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_compare_sets.txt'), 'w')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


# ---------------------------------------------------------------------------
def lft_A(sp, dv):
    """A(delta) of an LFT given as (A, Bunc, Cunc, Dunc_unc)."""
    Dl = np.diag(dv)
    n = len(dv)
    return sp['A'] + sp['Bunc'] @ np.linalg.solve(np.eye(n) - Dl @ sp['Dunc_unc'],
                                                  Dl @ sp['Cunc'])


def paper_sp(plate, mass=0.10, stiff=0.10, damp=0.20, coupling='paper'):
    import uncertain_plant as up
    U = up.MillingUncertainSystem(plate, C.RPM_S, C.AP_S, C.AE, n_modes=2,
                                  alpha_coupling=coupling, sign=C.SIGN)
    # the perturbation sizes are baked in at construction; rebuild with others
    if (mass, stiff, damp) != (0.10, 0.10, 0.20):
        orig = up.nominal_and_perturbations
        up.nominal_and_perturbations = (
            lambda pl, r, a, ae=C.AE, n_modes=2, alpha_coupling='paper',
            mass_pert=mass, stiff_pert=stiff, damp_pert=damp, n_pos=201:
            orig(pl, r, a, ae, n_modes, alpha_coupling, mass, stiff, damp,
                 n_pos))
        try:
            U = up.MillingUncertainSystem(plate, C.RPM_S, C.AP_S, C.AE,
                                          n_modes=2, alpha_coupling=coupling,
                                          sign=C.SIGN)
        finally:
            up.nominal_and_perturbations = orig
    return U.ss_plant, 10


# ---------------------------------------------------------------------------
def physical_sample(U, n_eta=9, n_xi=3, n_x=13, n_a=5, n_z=3):
    """A(theta) over a dense grid of the PHYSICAL set."""
    out, par = [], []
    for eta in np.linspace(0.0, U.eta_max, n_eta):
        for xi in np.linspace(0.0, 1.0, n_xi):
            for x in np.linspace(0.0, U.plate.lp, n_x):
                for am in np.linspace(C.ALPHA_LO, C.ALPHA_HI, n_a):
                    for z in np.linspace(C.ZETA_LO, C.ZETA_HI, n_z):
                        out.append(U.sample(eta, xi, x, am, z)[0])
                        par.append((eta, xi, x, am, z))
    return np.array(out), par


def coverage_gap(sp, npar, targets, scale, n_start=3, seed=0):
    """max over the physical points of  min_{|d|<=1} ||A_true - A_LFT(d)||_F.

    The Frobenius norm, not the max-norm: it is smooth, so `least_squares` finds
    the true minimiser instead of stalling the way a quasi-Newton method does on
    a max.  A gap reported here is geometry, not an optimiser artefact -- which
    matters, because the whole claim rests on one of these numbers being zero
    and another not.
    """
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    worst, arg = 0.0, None
    for k, At in enumerate(targets):
        best = np.inf
        for s in range(n_start):
            x0 = (np.zeros(npar) if s == 0 else
                  np.clip(rng.uniform(-1, 1, npar), -0.99, 0.99))
            try:
                r = least_squares(lambda v: (At - lft_A(sp, v)).ravel() / scale,
                                  x0, bounds=(-1.0, 1.0), xtol=1e-14,
                                  ftol=1e-14, gtol=1e-14, max_nfev=600)
                best = min(best, float(np.sqrt(2.0 * r.cost)))
            except Exception:                                 # noqa: BLE001
                pass
            if best < 1e-10:
                break
        if best > worst:
            worst, arg = best, k
    return worst, arg


def spurious_radius(sp, npar, phys, scale, n_corner=256, seed=0):
    """max over the SET of the distance to conv(physical family)."""
    rng = np.random.default_rng(seed)
    P = np.array([A.ravel() for A in phys]).T
    rho = 1e3 * np.abs(P).max()
    Paug = np.vstack([P, rho * np.ones((1, P.shape[1]))])
    corners = []
    if 2 ** npar <= n_corner:
        corners = [np.array(c, float)
                   for c in itertools.product([-1.0, 1.0], repeat=npar)]
    else:
        corners = [rng.choice([-1.0, 1.0], npar) for _ in range(n_corner)]
    worst = 0.0
    for dv in corners:
        b = np.concatenate([lft_A(sp, dv).ravel(), [rho]])
        _, res = nnls(Paug, b)
        worst = max(worst, res / scale)
    return worst


# ---------------------------------------------------------------------------
def main():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    Uph = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN)
    phys, par = physical_sample(Uph)
    scale = np.linalg.norm(phys[0])
    log('=' * 74)
    log('GEOMETRY OF THE UNCERTAINTY SETS')
    log('=' * 74)
    log(f'  physical family sampled at {len(phys)} points; '
        f'norm reference ||A_0||_F = {scale:.4e}')

    PHYS = dict(mass=0.35, stiff=0.064, damp=0.41)
    sets = {}
    sp, n = paper_sp(plate)
    sets['paper  (Eqs. 22-25, 10 %/10 %/20 %)'] = (sp, n)
    sp, n = paper_sp(plate, PHYS['mass'], PHYS['stiff'], PHYS['damp'], 'exact')
    sets['exact  (corrected envelope, physics-sized box)'] = (sp, n)
    for tag, kw in (('phys   (one-sided, correlated, rank one)', {}),
                    ('phys_sym  (symmetric in eta)', dict(one_sided=False)),
                    ('phys_ind  (M, C, K independent)', dict(correlated=False))):
        U = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN,
                                **kw)
        sets[tag] = (U.ss_plant, U.n_par)

    # a coarse subset for the (expensive) coverage optimisation
    sub = phys[::37]
    log(f'\n  coverage gap measured on {len(sub)} physical points, '
        f'spurious radius on the set corners\n')
    log(f'  {"set":48s} {"params":>7s} {"coverage":>10s} {"spurious":>10s}')
    log('  ' + '-' * 78)
    res = {}
    for tag, (sp, n) in sets.items():
        cov, _ = coverage_gap(sp, n, sub, scale)
        spu = spurious_radius(sp, n, phys, scale)
        res[tag] = (n, cov, spu)
        log(f'  {tag:48s} {n:7d} {100*cov:9.3f} % {100*spu:9.3f} %')
    log('')
    log('  coverage  = max distance PHYSICS -> set   (0 % = the set contains '
        'every operating point)')
    log('  spurious  = max distance set -> PHYSICS   (lower = less of the set '
        'is unreachable)')
    np.savez(os.path.join(OUT, 'set_geometry.npz'),
             **{k.split()[0] + '_' + s: v
                for k, (n, c, sp_) in res.items()
                for s, v in (('n', n), ('cov', c), ('spu', sp_))})


if __name__ == '__main__':
    main()
