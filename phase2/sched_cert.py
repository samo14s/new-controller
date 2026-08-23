"""sched_cert.py — the whole-pass certificate a scheduled family was missing.

The repository holds two delay certificates and each misses what milling
actually does.  The exact crossing test (certify2) freezes the schedule: every
vertex is judged with its own member, but nothing certifies the instant the
loop TRAVERSES from one member to the next.  The common-P LK certificate
(lk_lmi) certifies arbitrarily fast variation -- stronger than needed, and it
must squeeze one functional through every member of the family at once, which
is exactly where it dies as the members differ more.  What milling does is
neither frozen nor arbitrarily fast: the position moves at the FEED SPEED and
the removal advances with the PASS SCHEDULE -- both known, both slow.  This
module turns that knowledge into the theorem.

Construction
------------
The scheduling domain is split into cells; cell i carries the member designed
at its node, frozen.  Over each cell the closed loop is

    xdot(t) = A_i(theta(t)) x(t) + A_di(theta(t)) x(t - tau),
    (A_i, A_di)  in  conv{vertices of cell i}  (+)  an eps_i-ball,

with theta collecting everything that moves inside the cell: the position
continuum between the sampled vertices (measured, it goes into eps_i), the
removal interval the pass sweeps, the force coefficient in [a_lo, a_hi], the
damping band.  Per cell the EXPONENTIAL Lyapunov-Krasovskii functional

    V_i = x'P_i x + int_{t-tau}^t e^{2a(s-t)} x'Q_i x ds
        + tau int_{-tau}^0 int_{t+s}^t e^{2a(r-t)} xdot'R_i xdot dr ds

certifies  Vdot_i <= -2a V_i  through the LMI (Jensen with the e^{-2a tau}
discount; affine in (A, A_d), so vertices + the S-procedure residual cover the
continuum):

    [ A'P+PA+2aP+Q-cR    PA_d+cR      tau A'R  ]
    [ A_d'P+cR           -c(Q+R)      tau A_d'R]  < 0,   c = e^{-2 a tau}.
    [ tau R A            tau R A_d    -R       ]

At a = 0 this is byte-for-byte the lk_lmi condition.  At a switch i -> j the
state and its history are continuous, so V_j <= mu_ij V_i holds for every
history iff  P_j <= mu P_i,  Q_j <= mu Q_i,  R_j <= mu R_i  (largest
generalized eigenvalue over the three pairs).

Theorem (whole-pass stability under bounded scheduling rate).  Let every cell
of the tube around the pass trajectory be certified at common rate a > 0, and
let the traversal be monotone with dwell time  t_i >= h_i / v  in cell i.
Then along the whole pass

    V(t) <= exp(-2a (t-t0) + sum_{switches k before t} ln mu_k) V(t0),

and if  ln mu_k <= 2 a h_k / v  for every boundary k the envelope is
non-increasing at every switch: the loop is exponentially stable over the
ENTIRE operating domain, with the explicit admissible rate

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
# the exponential LK cell certificate
# ---------------------------------------------------------------------------
def _psi_alpha(A, Ad, P, Q, R, tau, alpha):
    c = float(np.exp(-2.0 * alpha * tau))
    return cp.bmat([
        [A.T @ P + P @ A + 2.0 * alpha * P + Q - c * R,
         P @ Ad + c * R, tau * A.T @ R],
        [Ad.T @ P + c * R, -c * (Q + R), tau * Ad.T @ R],
        [tau * R @ A, tau * R @ Ad, -R],
    ])


def certificate_alpha(vertices, tau, alpha=0.0, eps=0.0, n_plant=None,
                      solver=None, verbose=False, kappa=1e7, prev=None,
                      slack_min=None):
    """Exponential-rate LK certificate over one cell's vertex family.

    Identical to lk_lmi.certificate at alpha = 0 (same functional, same
    residual S-procedure, same normalisation); alpha > 0 adds the decay and
    the e^{-2 alpha tau} discount the derivation requires.

    `prev` chains the cells: given the PREVIOUS cell's (P, Q, R), the
    objective switches from maximising the slack to MINIMISING the jump
    factor t subject to  P <= t P_prev, Q <= t Q_prev, R <= t R_prev  (linear
    in (P, Q, R, t) jointly, so still one SDP) and slack >= slack_min.  A
    freely re-normalised family would otherwise show jumps that are artifacts
    of the solver's scale choice, not of the functionals.

    Returns dict(feasible, slack, status, P, Q, R[, t]).
    """
    if cp is None:
        raise RuntimeError('cvxpy is required for the certificate')
    n = vertices[0][0].shape[0]
    npl = n if n_plant is None else int(n_plant)
    P = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n), symmetric=True)
    R = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(nonneg=True)

    Sp = np.zeros((npl, n))
    Sp[:, :npl] = np.eye(npl)
    E0 = eps * np.vstack([np.eye(npl), np.zeros((n - npl, npl))])
    SS = Sp.T @ Sp
    Zn = np.zeros((n, n))
    FtF = np.block([[SS, Zn, Zn], [Zn, SS, Zn], [Zn, Zn, Zn]])

    cons = [P >> np.eye(n), Q >> 1e-6 * np.eye(n), R >> 1e-6 * np.eye(n),
            cp.trace(P) + cp.trace(Q) + cp.trace(R) + lam <= kappa]
    for (A, Ad) in vertices:
        Psi = _psi_alpha(A, Ad, P, Q, R, tau, alpha)
        if eps > 0.0:
            Psi = Psi + lam * FtF
            Ecal = cp.vstack([P @ E0, np.zeros((n, npl)), tau * R @ E0])
            Th = cp.bmat([[Psi, Ecal], [Ecal.T, -lam * np.eye(npl)]])
        else:
            Th = Psi
        cons.append(Th << -s * np.eye(Th.shape[0]))

    if prev is None:
        obj = cp.Maximize(s)
    else:
        t = cp.Variable(nonneg=True)
        cons += [P << t * prev['P'], Q << t * prev['Q'], R << t * prev['R'],
                 s >= (1e-10 if slack_min is None else float(slack_min))]
        obj = cp.Minimize(t)
    prob = cp.Problem(obj, cons)
    for sv in ([solver] if solver else ['CLARABEL', 'SCS']):
        try:
            prob.solve(solver=sv, verbose=verbose)
        except Exception:
            continue
        if prob.status in ('optimal', 'optimal_inaccurate'):
            break
    ok = (prob.status in ('optimal', 'optimal_inaccurate')
          and s.value is not None and float(s.value) > 1e-10)
    out = dict(feasible=bool(ok),
               slack=None if s.value is None else float(s.value),
               status=prob.status,
               P=None if P.value is None else np.array(P.value),
               Q=None if Q.value is None else np.array(Q.value),
               R=None if R.value is None else np.array(R.value))
    if prev is not None:
        out['t'] = None if not ok else float(t.value)
    return out


def jump_factor(cert_i, cert_j):
    """mu_ij = min mu with V_j <= mu V_i for every history: the largest
    generalized eigenvalue over the pairs (P_j, P_i), (Q_j, Q_i), (R_j, R_i)."""
    mu = 0.0
    for k in ('P', 'Q', 'R'):
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

        ss0, pd0 = ctrl.at(float(self.nodes[len(self.nodes) // 2]), 0.0)
        A0, Ad0, B0, _, Cy0 = plant.matrices(x_pos=float(self.nodes[0]))
        Acl0, _, npl = augment(A0, Ad0, B0, Cy0, ss0, pd0, plant.n)
        self.npl = npl
        from scipy.linalg import matrix_balance
        try:
            _, T = matrix_balance(Acl0[npl:, npl:])
            dd = np.concatenate([np.ones(npl), np.abs(np.diag(T))])
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
        if not hasattr(self, '_cell_cache'):
            self._cell_cache = {}
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
                        for k, x in enumerate(dense):
                            j = min(np.searchsorted(xs, x) - 1,
                                    len(xs) - 2)
                            j = max(j, 0)
                            t = ((x - xs[j]) / (xs[j + 1] - xs[j])
                                 if xs[j + 1] > xs[j] else 0.0)
                            Al = ((1 - t) * self._closed(ss, pd, xs[j], eta,
                                                         xi, z, a)[0]
                                  + t * self._closed(ss, pd, xs[j + 1], eta,
                                                     xi, z, a)[0])
                            Adl = ((1 - t) * self._closed(ss, pd, xs[j], eta,
                                                          xi, z, a)[1]
                                   + t * self._closed(ss, pd, xs[j + 1], eta,
                                                      xi, z, a)[1])
                            gap = np.hstack([pts[k][0][:self.npl] -
                                             Al[:self.npl],
                                             pts[k][1][:self.npl] -
                                             Adl[:self.npl]])
                            eps = max(eps, float(
                                np.linalg.norm(gap, 2)))
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
                                  slack_min=0.2 * certs[0]['slack'])
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


def alpha_ladder(cells, ladder=(5e-4, 1e-3, 2e-3, 5e-3, 1e-2), log=print):
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

    fh = open(os.path.join(OUT, 'log_sched_cert.txt'), 'w')

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
    log('WHOLE-PASS CERTIFICATE - EXPONENTIAL LK + DWELL-BOUNDED TRAVERSAL')
    log('=' * 78)
    log(f'  reference depth a_p = {C.AP_S*1e3:.1f} mm, delay tau = '
        f'{plant.tau*1e3:.3f} ms, feed speed v = {v_feed*1e3:.2f} mm/s')
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
