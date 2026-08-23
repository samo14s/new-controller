"""sched_cert.py — the whole-pass certificate a scheduled family was missing.

The repository holds two delay certificates and each misses what milling
actually does.  The exact crossing test (certify2) freezes the schedule: every
vertex is judged with its own member, but nothing certifies the instant the
loop TRAVERSES from one member to the next.  The common-P LK certificate
(lk_lmi.certificate_di) certifies arbitrarily fast variation -- stronger than
needed, and it must squeeze one functional through every member of the family
at once, which is exactly where it dies as the members differ more.  What
milling does is neither frozen nor arbitrarily fast: the position moves at the
FEED SPEED and the removal advances with the PASS SCHEDULE -- both known, both
slow.  This module turns that knowledge into the theorem.

Why the delay-independent functional.  The tooth-period delay spans ~17
periods of mode 2 (tau_scaled = 20.5 at ws = 2 pi 800), so any Jensen-bounded
delay-DEPENDENT functional is hopeless here -- measured: the 3-block form is
infeasible even where the DI form of the study certifies.  The study's own
convention stands: the delay AXIS is certified exactly by the crossing test,
and the LK functional's job is VARIATION.  So the cell functional is the
study's own DI form, given an exponential rate:

    V_i = x'P_i x + int_{t-tau}^t e^{2a(s-t)} x'Q_i x ds ,      P, Q > 0,

    Vdot_i + 2a V_i <= [x; x_tau]' Phi [x; x_tau],

    Phi = [ A'P + PA + 2aP + Q      P A_d          ]
          [ A_d'P                   -e^{-2a tau} Q ]   < 0.

At a = 0 this is byte-for-byte lk_lmi.certificate_di.  Feasibility at (a, tau)
implies it for every delay tau' in [0, tau] (the true (2,2) block is then more
negative), so one solve covers the whole spindle range whose tooth period does
not exceed tau.  Phi is affine in (A, A_d): vertices + the S-procedure
residual cover the cell continuum exactly as in lk_lmi.

Cells and jumps.  The scheduling domain is split into cells; cell i carries
the member designed at its node, frozen; the cell's vertex family spans the
position continuum inside the cell (eps_i covers the gap to the sampled
vertices), the removal interval, the mid-pass state, the damping band and the
force-coefficient range.  At a switch i -> i+1 the state and history are
continuous, so V_{i+1} <= mu V_i for every history iff P_{i+1} <= mu P_i and
Q_{i+1} <= mu Q_i.  The chain is solved SEQUENTIALLY, each cell minimising its
own jump factor subject to its LMIs (min t with P <= t P_prev, Q <= t Q_prev
-- linear in the variables, one SDP), because a freely re-normalised family
would show jumps that are artifacts of the solver's scale choice.

Theorem (whole-pass stability under bounded scheduling rate).  Let every cell
of the tube around the pass trajectory be certified at common rate a > 0, and
let the traversal be monotone with dwell time >= h_k / v at boundary k.  Then

    V(t) <= exp(-2a (t-t0) + sum_{switches k before t} ln mu_k) V(t0),

and if  ln mu_k <= 2 a h_k / v  for every k the envelope is non-increasing at
every switch: the loop is exponentially stable over the ENTIRE operating
domain -- every delay in [0, tau], every parameter path in the tube, arbitrary
in-cell variation -- with the explicit admissible rate

    v_cert = min_k  2 a h_k / ln mu_k          (infinite if mu_k <= 1).

The certificate consumes the known slowness of the schedule instead of paying
common-P's price for a speed the machine never uses; v_cert against the actual
feed speed says exactly how much rate margin the guarantee has.

    python phase2/sched_cert.py          (validate on the stored PS-AC-R /
                                          PS-TDC-R members, reference depth)

Writes results/log_sched_cert.txt and results/sched_cert.npz.
"""
import os
import sys
import time
import warnings

import numpy as np
from scipy.linalg import eigh

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cvxpy as cp
except Exception:                                            # pragma: no cover
    cp = None

import config as C
from certify2 import augment, _scale

OUT = C.RESULTS


# ---------------------------------------------------------------------------
# the exponential delay-independent LK cell certificate
# ---------------------------------------------------------------------------
def certificate_alpha(vertices, tau, alpha=0.0, eps=0.0, n_plant=None,
                      solver=None, verbose=False, prev=None, slack_min=None,
                      tol=1e-9):
    """Exponential-rate DI-form LK certificate over one cell's vertex family.

    Identical to lk_lmi.certificate_di at alpha = 0 (same functional, same
    residual S-procedure, same normalisation); alpha > 0 adds the decay and
    the e^{-2 alpha tau} discount the derivation requires.

    `prev` chains the cells: given the previous cell's (P, Q), the objective
    switches from maximising the slack to MINIMISING the jump factor t
    subject to P <= t P_prev, Q <= t Q_prev and slack >= slack_min; the trace
    normalisation is dropped there because the predecessor sets the scale.

    Returns dict(feasible, slack, status, P, Q[, t]).
    """
    if cp is None:
        raise RuntimeError('cvxpy is required for the certificate')
    n = vertices[0][0].shape[0]
    npl = n if n_plant is None else int(n_plant)
    P = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(nonneg=True)
    c = float(np.exp(-2.0 * alpha * tau))

    Sp = np.zeros((npl, n))
    Sp[:, :npl] = np.eye(npl)
    E0 = eps * np.vstack([np.eye(npl), np.zeros((n - npl, npl))])
    SS = Sp.T @ Sp
    Zn = np.zeros((n, n))
    FtF = np.block([[SS, Zn], [Zn, SS]])

    cons = [P >> 1e-3 * np.eye(n), Q >> 1e-8 * np.eye(n)]
    if prev is None:
        cons.append(cp.trace(P) + cp.trace(Q) + lam == 1.0)
    for (A, Ad) in vertices:
        Phi = cp.bmat([[A.T @ P + P @ A + 2.0 * alpha * P + Q, P @ Ad],
                       [Ad.T @ P, -c * Q]])
        if eps > 0.0:
            Phi = Phi + lam * FtF
            Ecal = cp.vstack([P @ E0, np.zeros((n, npl))])
            Phi = cp.bmat([[Phi, Ecal], [Ecal.T, -lam * np.eye(npl)]])
        cons.append(Phi << -s * np.eye(Phi.shape[0]))

    if prev is None:
        obj = cp.Maximize(s)
    else:
        t = cp.Variable(nonneg=True)
        cons += [P << t * prev['P'], Q << t * prev['Q'],
                 s >= (tol if slack_min is None else float(slack_min))]
        obj = cp.Minimize(t)
    prob = cp.Problem(obj, cons)
    # CLARABEL 0.11/cvxpy 1.9 dies with a zero step on this constraint class
    # (measured: NumericalError at iteration 1 even with entries at 26), so
    # SCS carries the solve -- and the VERDICT never rests on the solver: the
    # returned matrices are re-checked numerically below, so an inaccurate
    # solve can only produce a certificate that verifies, or none.
    for sv in ([solver] if solver else ['SCS']):
        try:
            prob.solve(solver=sv, verbose=verbose, eps=1e-6, max_iters=25000)
        except Exception:
            continue
        if prob.status in ('optimal', 'optimal_inaccurate'):
            break
    ok = (prob.status in ('optimal', 'optimal_inaccurate')
          and s.value is not None and float(s.value) > tol
          and P.value is not None and Q.value is not None)
    worst = None
    if ok:
        v = verify_certificate(vertices, tau, alpha, eps, npl,
                               np.array(P.value), np.array(Q.value),
                               float(lam.value))
        worst = v['phi_max']
        ok = bool(v['phi_max'] < -tol and v['p_min'] > 1e-8
                  and v['q_min'] > 1e-12)
    out = dict(feasible=bool(ok),
               slack=None if s.value is None else float(s.value),
               status=prob.status, worst_eig=worst,
               P=None if P.value is None else np.array(P.value),
               Q=None if Q.value is None else np.array(Q.value),
               lam=None if lam.value is None else float(lam.value))
    if prev is not None:
        # the recorded jump factor is computed from the returned matrices,
        # not read off the solver: exact for what is stored
        out['t'] = jump_factor(prev, out) if ok else None
    return out


def verify_certificate(vertices, tau, alpha, eps, n_plant, P, Q, lam):
    """Largest eigenvalue of every certificate LMI, rebuilt numerically from
    the stored (P, Q, lam) -- the solver-independent check.  Strictly
    negative = the functional certifies; anything else is no certificate,
    whatever the solver claimed."""
    n = P.shape[0]
    npl = int(n_plant)
    c = float(np.exp(-2.0 * alpha * tau))
    Sp = np.zeros((npl, n))
    Sp[:, :npl] = np.eye(npl)
    E0 = eps * np.vstack([np.eye(npl), np.zeros((n - npl, npl))])
    SS = Sp.T @ Sp
    Zn = np.zeros((n, n))
    FtF = np.block([[SS, Zn], [Zn, SS]])
    P = 0.5 * (P + P.T)
    Q = 0.5 * (Q + Q.T)
    phi_max = -np.inf
    for (A, Ad) in vertices:
        Phi = np.block([[A.T @ P + P @ A + 2.0 * alpha * P + Q, P @ Ad],
                        [Ad.T @ P, -c * Q]])
        if eps > 0.0:
            Phi = Phi + lam * FtF
            Ecal = np.vstack([P @ E0, np.zeros((n, npl))])
            Phi = np.block([[Phi, Ecal],
                            [Ecal.T, -lam * np.eye(npl)]])
        phi_max = max(phi_max, float(np.linalg.eigvalsh(
            0.5 * (Phi + Phi.T)).max()))
    return dict(phi_max=phi_max,
                p_min=float(np.linalg.eigvalsh(P).min()),
                q_min=float(np.linalg.eigvalsh(Q).min()))


def jump_factor(cert_i, cert_j):
    """mu_ij = min mu with V_j <= mu V_i for every history: the largest
    generalized eigenvalue over the pairs (P_j, P_i) and (Q_j, Q_i)."""
    mu = 0.0
    for k in ('P', 'Q'):
        Xi, Xj = cert_i[k], cert_j[k]
        Xi = 0.5 * (Xi + Xi.T) + 1e-12 * np.eye(len(Xi))
        Xj = 0.5 * (Xj + Xj.T)
        mu = max(mu, float(np.max(eigh(Xj, Xi, eigvals_only=True))))
    return mu


# ---------------------------------------------------------------------------
# cells of the scheduling tube, in ONE common coordinate system
# ---------------------------------------------------------------------------
class FamilyCells:
    """Vertex families of every cell, all in the same scaled coordinates.

    certify2.vertices re-derives its balancing per call, which is fine for one
    family judged alone and fatal here: the jump factors compare functionals
    across cells, so every cell must live in the SAME coordinates.  One
    diagonal transformation is computed from the mid-domain member and applied
    everywhere.
    """

    def __init__(self, plant, ctrl, n_cells=20, etas=None, a_scale=1.0,
                 zetas=(C.ZETA_LO, C.ZETA_HI), xis=(0.0, 1.0), n_x_cell=2,
                 n_dense=7, ws=2 * np.pi * 800.0, eta_pass=None):
        self.plant, self.ctrl, self.ws = plant, ctrl, ws
        self.n = plant.n
        lp = plant.plate.lp
        self.edges = np.linspace(0.0, lp, n_cells + 1)
        self.nodes = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.h = np.diff(self.edges)
        self.etas = (0.0, C.ETA_MAX) if etas is None else tuple(etas)
        self.a_scale = a_scale
        self.zetas, self.xis = zetas, xis
        self.n_x_cell, self.n_dense = n_x_cell, n_dense
        # eta drift of ONE pass inside one cell: the tube, not the box
        self.eta_pass = eta_pass
        self._cell_cache = {}

        # ---- one coordinate system for the whole family -------------------
        # The observer members carry Kalman gains up to ~1e11, so the raw
        # closed loop is numerically hopeless for an SDP (CLARABEL dies at
        # dres ~ 1e6 even after per-block balancing).  A diagonal balance of
        # the FULL representative closed loop (permute=False: pure scaling,
        # strictly positive) brings every entry to O(1e2) and is one fixed
        # similarity, so functionals of different cells stay comparable.
        mid = float(self.nodes[len(self.nodes) // 2])
        ss0, pd0 = ctrl.at(mid, 0.0)
        A0, Ad0, B0, _, Cy0 = plant.matrices(x_pos=mid)
        Acl0, Adcl0, npl = augment(A0, Ad0, B0, Cy0, ss0, pd0, plant.n)
        self.npl = npl
        Acl0, _ = _scale(Acl0, Adcl0, plant.n, npl, ws)
        from scipy.linalg import matrix_balance
        try:
            _, T = matrix_balance(Acl0, permute=False)
            dd = np.abs(np.diag(T))
        except Exception:
            dd = np.ones(Acl0.shape[0])
        if not np.all(np.isfinite(dd)) or np.min(dd) <= 0:
            dd = np.ones(Acl0.shape[0])
        self.Tc, self.Tci = np.diag(dd), np.diag(1.0 / dd)

    # --------------------------------------------------------------
    def _closed(self, ss, pd, x, eta, xi, z, a):
        A, Ad, B, _, Cy = self.plant.matrices(x_pos=float(x), a4=a,
                                              eta=float(eta), xi=float(xi),
                                              zeta_scale=float(z))
        Acl, Adcl, npl = augment(A, Ad, B, Cy, ss, pd, self.plant.n)
        Acl, Adcl = _scale(Acl, Adcl, self.plant.n, npl, self.ws)
        return self.Tci @ Acl @ self.Tc, self.Tci @ Adcl @ self.Tc

    def _a_range(self):
        a_nom = self.plant.a40
        a_lo = a_nom + (C.ALPHA_LO * self.plant.abar4 - a_nom) * self.a_scale
        a_hi = a_nom + (C.ALPHA_HI * self.plant.abar4 - a_nom) * self.a_scale
        return a_lo, a_hi

    def cell(self, i):
        """(vertices, eps_i) of cell i: the member frozen at the node, the
        plant swept over the cell's x-samples and the theta corners; eps_i
        covers the x-continuum between the samples (measured 2-norm gap to
        the linear interpolant, worst over the theta corners).  Memoized:
        the alpha ladder re-reads every cell at each rung."""
        if i in self._cell_cache:
            return self._cell_cache[i]
        ss, pd = self.ctrl.at(float(self.nodes[i]), 0.0)
        xs = np.linspace(self.edges[i], self.edges[i + 1], self.n_x_cell)
        a_lo, a_hi = self._a_range()
        etas = self.etas
        if self.eta_pass is not None:            # the pass tube in eta
            e0, de = self.eta_pass
            etas = (e0, min(e0 + de, C.ETA_MAX))
        V = []
        for x in xs:
            for eta in etas:
                for xi in self.xis:
                    for z in self.zetas:
                        for a in (a_lo, a_hi):
                            V.append(self._closed(ss, pd, x, eta, xi, z, a))
        # residual: x-continuum against the piecewise-linear interpolant
        eps = 0.0
        dense = np.linspace(self.edges[i], self.edges[i + 1], self.n_dense)
        for eta in etas:
            for xi in self.xis:
                for z in self.zetas:
                    for a in (a_lo, a_hi):
                        pts = [self._closed(ss, pd, x, eta, xi, z, a)
                               for x in dense]
                        ends = {x: self._closed(ss, pd, x, eta, xi, z, a)
                                for x in xs}
                        for k, x in enumerate(dense):
                            j = min(max(np.searchsorted(xs, x) - 1, 0),
                                    len(xs) - 2)
                            t = ((x - xs[j]) / (xs[j + 1] - xs[j])
                                 if xs[j + 1] > xs[j] else 0.0)
                            Al = ((1 - t) * ends[xs[j]][0]
                                  + t * ends[xs[j + 1]][0])
                            Adl = ((1 - t) * ends[xs[j]][1]
                                   + t * ends[xs[j + 1]][1])
                            gap = np.hstack([pts[k][0][:self.npl]
                                             - Al[:self.npl],
                                             pts[k][1][:self.npl]
                                             - Adl[:self.npl]])
                            eps = max(eps, float(np.linalg.norm(gap, 2)))
        self._cell_cache[i] = (V, eps)
        return V, eps

    def tau_scaled(self):
        return self.plant.tau * self.ws


# ---------------------------------------------------------------------------
def certify_family(cells, alpha, verbose=True, log=print):
    """Certify every cell at common rate alpha, chaining each cell's SDP to
    minimise the jump factor from its predecessor (traversal is one-way: the
    feed direction).  Returns certs + jump factors + the admissible traversal
    speed.  alpha and v_cert are in the SCALED time of the cells (t~ = ws t);
    the caller converts back with ws."""
    tau = cells.tau_scaled()
    certs, mus = [], []
    prev = None
    for i in range(len(cells.nodes)):
        V, eps = cells.cell(i)
        if prev is None:
            r = certificate_alpha(V, tau, alpha=alpha, eps=eps,
                                  n_plant=cells.npl)
        else:
            r = certificate_alpha(V, tau, alpha=alpha, eps=eps,
                                  n_plant=cells.npl, prev=prev,
                                  slack_min=0.1 * certs[0]['slack'])
            if not r['feasible']:      # fall back: feasibility over tightness
                r = certificate_alpha(V, tau, alpha=alpha, eps=eps,
                                      n_plant=cells.npl)
                if r['feasible']:
                    r['t'] = jump_factor(prev, r)
        certs.append(r)
        if verbose:
            log(f'    cell {i:2d} [{cells.edges[i]*1e3:5.1f},'
                f'{cells.edges[i+1]*1e3:5.1f}] mm  eps={eps:8.2e}  '
                f'{"ok" if r["feasible"] else "INFEASIBLE"}'
                + (f'  slack {r["slack"]:.2e}' if r['slack'] else '')
                + (f'  jump {r["t"]:.3f}' if r.get('t') else ''))
        if not r['feasible']:
            return dict(feasible=False, certs=certs, failed_cell=i)
        if prev is not None:
            mus.append(float(r['t']))
        prev = r
    # ln mu <= 2 alpha h / v  =>  v <= 2 alpha h / ln mu, per boundary
    v_bounds = []
    for i, mu in enumerate(mus):
        h = cells.h[i] * 0.5 + cells.h[i + 1] * 0.5   # node-to-node dwell
        v_bounds.append(np.inf if mu <= 1.0
                        else 2.0 * alpha * h / np.log(mu))
    return dict(feasible=True, certs=certs, mus=np.array(mus),
                v_cert=float(np.min(v_bounds)) if v_bounds else np.inf,
                v_bounds=np.array(v_bounds))


def alpha_ladder(cells, ladder=(5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2),
                 log=print):
    """Climb a geometric ladder of decay rates; keep the last feasible rung.
    Returns (alpha, result) -- alpha = 0 with the base result if even the
    smallest rung fails."""
    best = (0.0, None)
    for a in ladder:
        r = certify_family(cells, a, verbose=False, log=log)
        if not r['feasible']:
            break
        best = (a, r)
    if best[1] is None:
        return 0.0, certify_family(cells, 0.0, verbose=False, log=log)
    return best


# ---------------------------------------------------------------------------
def main():
    from plate_model import build_plate
    from plant_ss import ControlledPlant
    from stage_common import load_controllers

    fh = open(os.path.join(OUT, 'log_sched_cert.txt'), 'a')

    def log(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()

    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    mks = load_controllers(plate, plant)
    v_feed = plant.feed_speed()

    log('=' * 78)
    log('WHOLE-PASS CERTIFICATE - EXPONENTIAL DI-LK + DWELL-BOUNDED '
        'TRAVERSAL')
    log('=' * 78)
    log(f'  reference depth a_p = {C.AP_S*1e3:.1f} mm, delay tau = '
        f'{plant.tau*1e3:.3f} ms (covers every tau\' <= tau), feed speed '
        f'v = {v_feed*1e3:.2f} mm/s')
    log('  cells: 20 over the edge; theta per cell: removal x mid-pass x '
        'damping x force coeff.')
    log('')

    out = {}
    for kind in ('ps_ac_r', 'ps_tdc_r'):
        if kind not in mks:
            continue
        ctrl = mks[kind](plant)
        log(f'[{kind}]')
        cells = FamilyCells(plant, ctrl)
        t = time.time()
        r0 = certify_family(cells, 0.0, verbose=True, log=log)
        if not r0['feasible']:
            log(f'  alpha = 0 infeasible at cell {r0["failed_cell"]} -- '
                'no dwell statement possible on this tube')
            continue
        a_star, r = alpha_ladder(cells, log=log)
        ws = cells.ws
        log(f'  alpha* = {a_star:.4f} (scaled) = {a_star*ws:.2f} 1/s')
        if a_star > 0.0 and len(r['mus']):
            log(f'  jump factors mu: min {r["mus"].min():.3f}  '
                f'max {r["mus"].max():.3f}')
            vc = r['v_cert'] * ws          # scaled rate * m -> m/s
            log(f'  certified traversal speed v_cert = '
                + ('unbounded' if np.isinf(vc) else f'{vc*1e3:.2f} mm/s')
                + f'   (actual feed {v_feed*1e3:.2f} mm/s -> margin '
                + ('inf' if np.isinf(vc) else f'{vc/v_feed:.1f}x') + ')')
            out[f'{kind}_alpha'] = a_star * ws
            out[f'{kind}_mus'] = r['mus']
            out[f'{kind}_v_cert'] = vc
        log(f'  [{time.time()-t:.0f}s]')
        log('')

    out['v_feed'] = v_feed
    np.savez(os.path.join(OUT, 'sched_cert.npz'), **out)
    log(f'total {time.time()-t0:.0f}s -> results/sched_cert.npz')


if __name__ == '__main__':
    main()
