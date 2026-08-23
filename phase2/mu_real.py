"""mu_real.py — the mixed real/complex mu upper bound: the price of the
complex reading, measured.

Every mu number in this repository -- Eq. (29) of the paper, dk_synthesis,
mu_tight, the G1 judge -- treats the parametric deltas as COMPLEX, "as in
Eq. (29)".  The physics is real: removed volume, mid-pass state, force
coefficient, damping ratios are real scalars.  For a lightly damped plate the
gap between the two readings is not a nicety: a COMPLEX stiffness
perturbation is a disguised negative damping, so the complex-mu of the
almost-open loop sits near |dK|/(2 zeta k) -- measured at ~40 on this plant --
while no REAL parameter combination does anything of the kind.  Any judgement
of "robust stability over the physics set" charged at complex prices
overpays; how much, this module measures.

The bound is the standard mixed-mu upper bound (Fan-Tits-Doyle; Young):

    mu(M)^2  <=  inf { s :  exists D in D+, G in G :
                   M^H D M + j (G M - M^H G) - s D  <=  0 } ,

with D Hermitian positive definite commuting with the structure (full blocks
on repeated scalars -- the mu_tight freedom), and G Hermitian, supported ONLY
on the REAL blocks (zero on Delta_Pa and any complex block).  At G = 0 it is
exactly the full-D complex bound, so it can only be tighter.  The
one-input/two-output actuator block is squared up by inserting a zero column
(the padded input drives nothing, so the bound is unchanged -- the standard
embedding).

The LMI is solved per frequency by bisection on s (linear in (D, G) at fixed
s), and the verdict never rests on the solver: the returned (D, G) are
re-checked numerically -- any feasible pair certifies mu <= sqrt(s) whatever
the solver's status.

    python phase2/mu_real.py        judge the stored artifacts both ways

Writes results/log_mu_real.txt and results/mu_real.npz.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cvxpy as cp
except Exception:                                            # pragma: no cover
    cp = None

import config as C

OUT = C.RESULTS


# ---------------------------------------------------------------------------
def _square_up(M, blocks):
    """Insert zero columns so every non-square ('F', no, ni) block becomes
    no x no full complex; returns (M_sq, sizes, kinds)."""
    sizes, kinds = [], []
    cols = []
    c = 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            raise ValueError('RS cut only -- drop the performance block')
        sizes.append(no)
        kinds.append('r' if kind == 'S' else 'c')
        block = M[:, c:c + ni]
        if ni < no:
            block = np.hstack([block, np.zeros((M.shape[0], no - ni),
                                               complex)])
        cols.append(block)
        c += ni
    return np.hstack(cols), sizes, kinds


def _feasible(M, sizes, kinds, s, eps_pd=1e-7):
    """The fixed-s feasibility SDP.  Returns (ok, D, G, margin)."""
    n = M.shape[1]
    Ds, Gs = [], []
    for m, k in zip(sizes, kinds):
        Ds.append(cp.Variable((m, m), hermitian=True))
        Gs.append(cp.Variable((m, m), hermitian=True) if k == 'r' else None)

    def blkdiag(mats, default_zero=False):
        rows = []
        off = 0
        full = []
        for m, X in zip(sizes, mats):
            full.append(X if X is not None else np.zeros((m, m)))
        Z = []
        for i, Xi in enumerate(full):
            row = []
            for j, Xj in enumerate(full):
                if i == j:
                    row.append(Xi)
                else:
                    row.append(np.zeros((sizes[i], sizes[j])))
            Z.append(row)
        return cp.bmat(Z)

    D = blkdiag(Ds)
    G = blkdiag(Gs)
    lhs = cp.conj(M).T @ D @ M + 1j * (G @ M - cp.conj(M).T @ G) - s * D
    cons = [D >> eps_pd * np.eye(n), cp.trace(D) == 1.0,
            lhs << 0]
    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver='SCS', eps=1e-6, max_iters=8000)
    except Exception:
        return False, None, None, None
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return False, None, None, None
    Dv = np.zeros((n, n), complex)
    Gv = np.zeros((n, n), complex)
    off = 0
    for m, k, Dx, Gx in zip(sizes, kinds, Ds, Gs):
        Dv[off:off + m, off:off + m] = 0.5 * (Dx.value + Dx.value.conj().T)
        if Gx is not None:
            Gv[off:off + m, off:off + m] = 0.5 * (Gx.value
                                                  + Gx.value.conj().T)
        off += m
    # solver-independent check
    lam = np.linalg.eigvalsh(Dv)
    if lam.min() <= 0:
        return False, None, None, None
    T = M.conj().T @ Dv @ M + 1j * (Gv @ M - M.conj().T @ Gv) - s * Dv
    marg = float(np.linalg.eigvalsh(0.5 * (T + T.conj().T)).max())
    return bool(marg <= 1e-9), Dv, Gv, marg


def mu_mixed(M, blocks, mu_cx=None, n_iter=9, floor=1e-4):
    """(mu_mixed_ub, mu_complex_ub) at one frequency.

    mu_cx seeds the bisection ceiling (the mixed bound can only be tighter);
    if absent the scalar-D complex bound is computed first.
    """
    from dk_synthesis import mu_upper_bound
    if mu_cx is None:
        mu_cx = mu_upper_bound(M, blocks)[0]
    Msq, sizes, kinds = _square_up(np.asarray(M, complex), blocks)
    hi = max(mu_cx ** 2, floor ** 2)
    if _feasible(Msq, sizes, kinds, hi)[0] is False:
        return float(mu_cx), float(mu_cx)      # cannot even match: keep cx
    lo = 0.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if mid <= floor ** 2:
            break
        ok = _feasible(Msq, sizes, kinds, mid)[0]
        if ok:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(hi)), float(mu_cx)


def mu_curve_mixed(H, blocks, verbose=False):
    """Mixed and complex upper-bound curves over a frequency grid.

    The complex curve uses the scalar-D bound as elsewhere.  The mixed bound
    is computed frequency by frequency in DESCENDING order of the complex
    bound, keeping the running best mixed value b; any remaining frequency
    with cx <= b is skipped, because its true mixed value is <= cx <= b and
    cannot move the peak.  The reported mixed curve therefore carries cx at
    the skipped points (a valid over-estimate everywhere), and its maximum
    over the COMPUTED points -- what the callers report -- is a certified
    upper bound of the true mixed peak, at a fraction of the solves."""
    from dk_synthesis import mu_upper_bound
    H = np.asarray(H)
    n = len(H)
    cx = np.empty(n)
    d0 = None
    for k in range(n):
        cx[k], d0 = mu_upper_bound(H[k], blocks, d0)
    mixed = cx.copy()
    best = 0.0
    for k in np.argsort(cx)[::-1]:
        if cx[k] <= best:
            break
        mixed[k] = mu_mixed(H[k], blocks, mu_cx=cx[k])[0]
        best = max(best, mixed[k])
        if verbose:
            print(f'    f[{k}] cx {cx[k]:.3f} -> mixed {mixed[k]:.3f}',
                  flush=True)
    return mixed, cx


# ---------------------------------------------------------------------------
def judge_mixed(plate, ss_K, x, f, drop=('d_D', 'd_r'), actuator=True):
    """Peak mixed-mu RS at a scheduling node -- the G1 quantity re-read with
    the parametrics real.  Machinery mirrors mu_scheduled.mu_rs."""
    from mu_scheduled import plant_at
    from cross_mu import closed_loop, frf
    U = plant_at(plate, x)
    P = U.generalized_plant()
    T, info = closed_loop(P, ss_K)
    w = 2 * np.pi * np.asarray(f, float) / info['w0']
    H = frf(T, w)[:, :P['n_unc_out'], :P['n_unc_in']]
    keep = [j for j, (nm, _) in enumerate(U.blocks_used) if nm not in drop]
    ch = np.concatenate([U.idx[U.blocks_used[j][0]] for j in keep])
    if actuator:
        rows = np.concatenate([[0, 1], 2 + ch]).astype(int)
        cols = np.concatenate([[0], 1 + ch]).astype(int)
        blocks = [('F', 2, 1)]
    else:
        rows, cols, blocks = (2 + ch).astype(int), (1 + ch).astype(int), []
    blocks += [('S', U.blocks_used[j][1], U.blocks_used[j][1]) for j in keep]
    mixed, cx = mu_curve_mixed(H[:, rows][:, :, cols], blocks)
    km, kc = int(np.argmax(mixed)), int(np.argmax(cx))
    return dict(mixed=float(mixed[km]), f_mixed=float(f[km]),
                cx=float(cx[kc]), f_cx=float(f[kc]),
                curve_mixed=mixed, curve_cx=cx)


def sweep21():
    """M1-real: the stored witness judged at ALL 21 scheduling nodes under
    the mixed reading -- one fixed artifact, the whole gate family."""
    from plate_model import build_plate
    from stage_common import load_mu_ss
    from robust_design import freq_grid

    fh = open(os.path.join(OUT, 'log_mu_real.txt'), 'a')

    def log(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()

    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    ss = load_mu_ss('mu_paper')
    f = freq_grid()
    xs = np.linspace(0.0, plate.lp, 21)
    log('')
    log('M1-real: the stored mu_paper witness at the 21 nodes, mixed reading')
    mx, cxs = [], []
    for x in xs:
        t = time.time()
        r = judge_mixed(plate, ss, float(x), f)
        mx.append(r['mixed'])
        cxs.append(r['cx'])
        log(f'  x = {x*1e3:5.1f} mm:  complex {r["cx"]:7.3f}  ->  mixed '
            f'{r["mixed"]:7.3f}' + ('' if r['mixed'] < 1 else '  >= 1')
            + f'   [{time.time()-t:.0f}s]')
    mx, cxs = np.array(mx), np.array(cxs)
    log(f'  sup over 21 nodes: complex {cxs.max():.3f}  ->  mixed '
        f'{mx.max():.3f}'
        + ('   <-- G1(real) MET AT EVERY NODE' if mx.max() < 1 else ''))
    np.savez(os.path.join(OUT, 'mu_real_21.npz'), x=xs, mixed=mx, cx=cxs)
    log(f'total {time.time()-t0:.0f}s -> results/mu_real_21.npz')


def main():
    from plate_model import build_plate
    from plant_ss import ControlledPlant
    from stage_common import load_controllers, load_mu_ss
    from robust_design import freq_grid

    fh = open(os.path.join(OUT, 'log_mu_real.txt'), 'a')

    def log(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()

    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    f = freq_grid()
    mks = load_controllers(plate, plant)

    log('=' * 78)
    log('MIXED REAL/COMPLEX mu - THE G1 QUANTITY RE-READ WITH THE '
        'PARAMETRICS REAL')
    log('=' * 78)
    log('  same judge machinery, same grid; only the delta set changes:')
    log('  real scalars get the G-scale the complex reading denies them.')
    log('')

    res = {}
    rows = [('mu_paper', load_mu_ss('mu_paper'), None)]
    if 'ps_ac_r' in mks:
        rows.append(('ps_ac_r', None, mks['ps_ac_r'](plant)))
    if 'ps_ac_rfa' in mks:
        rows.append(('ps_ac_rfa', None, mks['ps_ac_rfa'](plant)))
    for name, ss_fixed, ctrl in rows:
        log(f'[{name}]')
        for frx in (0.0, 0.5, 1.0):
            x = frx * plate.lp
            ss = ss_fixed if ss_fixed is not None else ctrl.at(x)[0]
            t = time.time()
            r = judge_mixed(plate, ss, x, f)
            res[f'{name}_{int(100*frx)}_mixed'] = r['curve_mixed']
            res[f'{name}_{int(100*frx)}_cx'] = r['curve_cx']
            log(f'  x={frx:4.0%}: complex {r["cx"]:7.3f} at '
                f'{r["f_cx"]:6.0f} Hz  ->  MIXED {r["mixed"]:7.3f} at '
                f'{r["f_mixed"]:6.0f} Hz'
                + ('   <-- G1(real) MET' if r['mixed'] < 1 else '')
                + f'   [{time.time()-t:.0f}s]')
        log('')

    res['f'] = f
    np.savez(os.path.join(OUT, 'mu_real.npz'), **res)
    log(f'total {time.time()-t0:.0f}s -> results/mu_real.npz')


if __name__ == '__main__':
    sweep21() if 'sweep' in sys.argv[1:] else main()
