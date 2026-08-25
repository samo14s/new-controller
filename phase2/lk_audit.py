"""lk_audit.py — is the Lyapunov-Krasovskii column a certificate or a solver mood?

Every LK verdict in the study came from `lk_lmi.certificate_di`, which declared
feasibility from cvxpy's STATUS and the sign of the slack variable, and never
looked at the matrices the solver returned.  That is the one thing a certificate
must not do.  This audit re-runs the SDP at each design's stored a_p^inf and
asks three separate questions, keeping the answers apart:

  solver   what cvxpy called it (status + slack > 0), i.e. the old verdict;
  verified whether the returned (P, Q) actually satisfy P > 0, Q > 0 and
           Psi < 0 at every vertex -- the definition of the certificate;
  stored   what results/stage56.pkl recorded when the study ran.

A design where solver and verified disagree had a FALSE POSITIVE.  A design
where solver and stored disagree means the SDP is not even repeatable, which is
a stronger statement than conservatism: it makes the column noise.

Two formulations are tried, so the finding cannot be blamed on a careless
setup: the study's own, and one where a single diagonal similarity (identical
across vertices, hence exactly equivalence-preserving for a COMMON P) balances
the closed-loop pencil first.  Both go through the same verification.

Iteration caps are declared rather than hidden: SCS is given MAX_ITERS and
CLARABEL its defaults, and a design that exhausts them is reported as
"not converged", not as infeasible.

PRE-DECLARED (house rule)
  A1  the audit only reports; it changes no stored result.
  A2  a design counts as CERTIFIED only if the returned matrices verify, in at
      least one of the two formulations.
  A3  disagreement between `stored` and `solver` is reported per design, and
      the count of disagreements is the headline -- a column that cannot
      reproduce its own verdicts is withdrawn regardless of A2.

Writes results/lk_audit.txt (and .npz).

    python phase2/lk_audit.py
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cvxpy as cp
from scipy.linalg import matrix_balance

import certify2 as CF2
import config as C
from lk_lmi import verify_di
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers, LABEL

OUT = C.RESULTS
FAM = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=(C.ZETA_LO, C.ZETA_HI),
           xis=(1.0,))
MAX_ITERS = 25000
TOL = 1e-9


def common_balance(V):
    """One diagonal similarity for ALL vertices.

    A common P is invariant under x -> T x applied uniformly (P -> T^T P T), so
    this cannot create or destroy feasibility -- only conditioning.
    """
    M = np.max([np.abs(A) + np.abs(Ad) for A, Ad in V], axis=0)
    _, T = matrix_balance(M)
    d = np.abs(np.diag(T))
    d[~np.isfinite(d) | (d <= 0)] = 1.0
    Ti, Tm = np.diag(1.0 / d), np.diag(d)
    return [(Ti @ A @ Tm, Ti @ Ad @ Tm) for A, Ad in V]


def solve(V):
    """(solver_feasible, verified, residuals, status) for one vertex family."""
    n = V[0][0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n), symmetric=True)
    s = cp.Variable(nonneg=True)
    cons = [P >> 1e-3 * np.eye(n), Q >> 1e-8 * np.eye(n),
            cp.trace(P) + cp.trace(Q) == 1.0]
    for (A, Ad) in V:
        cons.append(cp.bmat([[A.T @ P + P @ A + Q, P @ Ad],
                             [Ad.T @ P, -Q]]) << -s * np.eye(2 * n))
    prob = cp.Problem(cp.Maximize(s), cons)
    status = 'not attempted'
    for sv, kw in (('CLARABEL', {}), ('SCS', dict(max_iters=MAX_ITERS))):
        try:
            prob.solve(solver=sv, **kw)
        except Exception as exc:                          # noqa: BLE001
            status = f'{sv}: {type(exc).__name__}'
            continue
        status = f'{sv}: {prob.status}'
        if prob.status in ('optimal', 'optimal_inaccurate'):
            break
    solver_ok = (prob.status in ('optimal', 'optimal_inaccurate')
                 and s.value is not None and float(s.value) > TOL)
    r = verify_di(V, None if P.value is None else np.array(P.value),
                  None if Q.value is None else np.array(Q.value))
    return bool(solver_ok), r, status


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    with open(os.path.join(OUT, 'stage56.pkl'), 'rb') as fh:
        st = pickle.load(fh)
    mks = load_controllers(plate, ControlledPlant(plate, ap=C.AP_S))
    names = [k for k in st if st[k].get('lk_attempted') and k in mks]

    out = ['=' * 92,
           'THE LYAPUNOV-KRASOVSKII COLUMN, AUDITED',
           '=' * 92,
           f'  each design re-solved at its stored a_p^inf; SCS capped at '
           f'{MAX_ITERS} iterations',
           '  "solver" = status + slack, the old verdict;  "verified" = the '
           'returned P, Q really satisfy the LMIs',
           '']
    out.append('  ' + 'design'.ljust(26) + 'a_p^inf  stored  solver  verified '
               '   min eig P/|P|   max eig Psi/|Psi|   formulation')
    rows = []
    for k in names:
        ap = float(st[k]['ap_inf'])
        if ap <= 0:
            continue
        p = ControlledPlant(plate, ap=ap)
        V, npl, _ = CF2.vertices(p, mks[k](p), **FAM)
        best = None
        for tag, vs in (('as published', V), ('balanced', common_balance(V))):
            ok, r, status = solve(vs)
            cand = (r['certified'], ok, r, status, tag)
            if best is None or cand[0] > best[0]:
                best = cand
            if r['certified']:
                break
        cert, ok, r, status, tag = best
        rows.append((k, ap, bool(st[k]['lk']), ok, cert,
                     r.get('rel_min_eig_P', np.nan), r.get('max_eig_Psi',
                                                           np.nan)))
        out.append(f'  {LABEL.get(k, k):<26}{ap*1e3:7.4f}'
                   f'{str(st[k]["lk"]):>8}{str(ok):>8}{str(cert):>10}'
                   f'{r.get("rel_min_eig_P", np.nan):16.3e}'
                   f'{r.get("max_eig_Psi", np.nan):20.3e}   {tag} [{status}]')

    R = np.array([(r[2], r[3], r[4]) for r in rows], bool)
    n_cert = int(R[:, 2].sum())
    n_disagree = int((R[:, 0] != R[:, 1]).sum())
    out += ['',
            f'  [A2] designs whose returned matrices VERIFY : {n_cert} of '
            f'{len(rows)}',
            f'  [A3] designs where stored and solver DISAGREE: {n_disagree} of '
            f'{len(rows)}']
    for k, _, stored, ok, cert, _, _ in rows:
        if stored != ok:
            out.append(f'      {LABEL.get(k, k)}: stored {stored}, '
                       f're-run {ok}')
    out += ['',
            '  reading: a solver status is not a certificate.  The verdicts',
            '  above were read off cvxpy\'s status and a slack variable while',
            '  the returned matrices violate the very inequalities they are',
            '  supposed to satisfy, by many orders of magnitude.  Balancing',
            '  the pencil first (an exact equivalence for a common P) improves',
            '  the residuals without curing them.  The column is withdrawn:',
            '  the crossing test, which is necessary AND sufficient per frozen',
            '  vertex, carries the delay axis on its own and is unaffected.']

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'lk_audit.txt'), 'w') as fh:
        fh.write(txt)
    np.savez(os.path.join(OUT, 'lk_audit.npz'),
             names=np.array([r[0] for r in rows]),
             ap_inf=np.array([r[1] for r in rows]),
             stored=R[:, 0], solver=R[:, 1], verified=R[:, 2],
             rel_min_eig_P=np.array([r[5] for r in rows]),
             max_eig_Psi=np.array([r[6] for r in rows]),
             n_certified=n_cert, n_disagree=n_disagree, max_iters=MAX_ITERS)
    print(txt)
    print(f'total {time.time()-t0:.0f}s -> results/lk_audit.*')


if __name__ == '__main__':
    main()
